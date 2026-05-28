import os
import string
import py_vncorenlp
import networkx as nx
from pke.unsupervised import TopicRank
from pke.data_structures import Candidate

_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VNCORENLP_DIR  = os.path.join(_BASE_DIR, "pretrained-models", "vncorenlp")
_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "vietnamese-stopwords-dash.txt")

with open(_STOPWORDS_PATH, encoding='utf-8') as f:
    STOPWORDS = {line.strip().lower() for line in f if line.strip()}

print("Đang khởi tạo VnCoreNLP...")
annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos"], save_dir=_VNCORENLP_DIR)

text = """
Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy
bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương
Mỵ Châu – Trọng Thủy. Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn
khám phá được những giá trị khảo cổ to lớn của Cổ Loa.
Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích
bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước.
"""

# B1: VnCoreNLP annotate
annotated = annotator.annotate_text(text)

POS_MAP = {
    'N':  'NOUN',  'Nc': 'NOUN', 'Nu': 'NOUN', 'Ny': 'NOUN',
    'Np': 'PROPN',
    'A':  'ADJ',
}

# Tạo sentences và pos cho PKE
sentences_tokens = []  # list[list[str]] - token gốc từ VnCoreNLP (có '_')
sentences_pos = []     # list[list[str]] - POS tag tương ứng

for sent_id in sorted(annotated.keys()):
    tokens = annotated[sent_id]
    tokens_list = []
    pos_list = []
    
    for tok in tokens:
        word = tok['wordForm'].lower()
        pos_tag = POS_MAP.get(tok['posTag'], 'X')
        tokens_list.append(word)
        pos_list.append(pos_tag)
    
    if tokens_list:
        sentences_tokens.append(tokens_list)
        sentences_pos.append(pos_list)


# B2: Khởi tạo TopicRank + load document
extractor = TopicRank()

# Tạo input cho PKE từ dữ liệu đã xử lý bằng VnCoreNLP
# Format: [[('word1', 'POS1'), ('word2', 'POS2'), ...], ...]
processed_sentences = []
for tokens, pos_tags in zip(sentences_tokens, sentences_pos):
    sent = []
    for token, pos in zip(tokens, pos_tags):
        # Tách token có '_' thành các từ đơn
        sub_tokens = token.split('_')
        for sub_token in sub_tokens:
            sent.append((sub_token, pos))
    processed_sentences.append(sent)

print(processed_sentences)


extractor.load_document(
    input=processed_sentences,
    language='en',
    normalization=None
)

# B3: Tạo candidates (chuỗi dài nhất NOUN/PROPN/ADJ)
candidates_dict = {}
offset = 0  # vị trí tuyệt đối của sub-token trong toàn văn bản

for sent_id, (tokens, pos_tags) in enumerate(zip(sentences_tokens, sentences_pos)):
    i = 0
    while i < len(tokens):
        # Bỏ qua nếu không phải POS hợp lệ
        if pos_tags[i] not in {'NOUN', 'PROPN', 'ADJ'}:
            offset += len(tokens[i].split('_'))
            i += 1
            continue
        
        # Tìm chuỗi dài nhất các POS hợp lệ liên tiếp
        start = i
        while i < len(tokens) and pos_tags[i] in {'NOUN', 'PROPN', 'ADJ'}:
            i += 1
        
        # Lấy chuỗi từ start đến i-1
        if i - start == 0:
            continue
            
        words = tokens[start:i]  # token gốc (có '_')
        surface = ' '.join(words)
        
        # Tách các sub-token để kiểm tra stopwords/punctuation
        all_sub_tokens = []
        for w in words:
            all_sub_tokens.extend(w.split('_'))
        
        # ── Các điều kiện lọc candidate ─────────────────────────────────────
        
        # 1. Kiểm tra stopwords CHỈ ở đầu/cuối/toàn bộ
        first_token = all_sub_tokens[0] if all_sub_tokens else ''
        last_token = all_sub_tokens[-1] if all_sub_tokens else ''
        
        # Toàn bộ là stopword
        all_stopwords = all(t in STOPWORDS for t in all_sub_tokens)
        # Bắt đầu bằng stopword
        starts_with_stopword = first_token in STOPWORDS
        # Kết thúc bằng stopword
        ends_with_stopword = last_token in STOPWORDS
        
        has_boundary_stopword = all_stopwords or starts_with_stopword or ends_with_stopword
        
        # 2. Chứa punctuation
        has_punct = any(sub_t in string.punctuation for sub_t in all_sub_tokens)
        
        # 3. Quá ngắn: chỉ giữ nếu là PROPN hoặc có ≥ 2 token gốc
        is_single_noun = len(words) == 1 and pos_tags[start] != 'PROPN'
        
        if has_boundary_stopword or has_punct or is_single_noun:
            offset += len(all_sub_tokens)
            continue
        
        # ── Candidate hợp lệ ────────────────────────────────────────────────
        if surface not in candidates_dict:
            candidates_dict[surface] = {
                'lexical_form': all_sub_tokens,
                'offsets': [offset],
                'sentence_ids': [sent_id],
                'surface_forms': [all_sub_tokens],
                'num_words': len(words)
            }
        else:
            candidates_dict[surface]['offsets'].append(offset)
            candidates_dict[surface]['sentence_ids'].append(sent_id)
            candidates_dict[surface]['surface_forms'].append(all_sub_tokens)
        
        offset += len(all_sub_tokens)

# B4: Inject candidates vào extractor
for surface, info in candidates_dict.items():
    if surface not in extractor.candidates:
        c = Candidate()
        c.lexical_form = info['lexical_form']
        c.surface_forms = info['surface_forms']
        c.offsets = info['offsets']
        c.sentence_ids = info['sentence_ids']
        extractor.candidates[surface] = c

# B5: TopicRank weighting (clustering + PageRank)
print(f"Số lượng candidates: {len(extractor.candidates)}")

# Gom nhóm candidates thành topics (HAC + Jaccard distance trên từ vựng)
extractor.topic_clustering(threshold=0.74, method='average')

# Xây đồ thị topic (cạnh dựa trên khoảng cách vị trí)
extractor.build_topic_graph()

# PageRank trên đồ thị topic
w = nx.pagerank(extractor.graph, alpha=0.85, weight='weight')

# Gán điểm cho candidate TỐT NHẤT trong mỗi topic
# Tiêu chí: ưu tiên candidate dài nhất, nếu bằng thì chọn cái xuất hiện sớm
for i, topic in enumerate(extractor.topics):
    best_candidate = max(
        topic,
        key=lambda t: (
            len(extractor.candidates[t].lexical_form),  # ưu tiên nhiều từ
            -extractor.candidates[t].offsets[0]          # ưu tiên xuất hiện sớm
        )
    )
    extractor.weights[best_candidate] = w[i]

# B6: Lấy kết quả
keyphrases = extractor.get_n_best(n=10) 

# ── In kết quả ───────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  TOP 10 KEYPHRASE (TOPICRANK + STOPWORDS Ở BIÊN)")
print(f"{'─'*50}")
for i, (kp, score) in enumerate(keyphrases, 1):
    kp_clean = kp.replace('_', ' ')
    print(f"  {i:2d}. {kp_clean:<35}  {score:.4f}")
print(f"{'─'*50}")
print("  (Score càng cao → từ khóa càng quan trọng)\n")

