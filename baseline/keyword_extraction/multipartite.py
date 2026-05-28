import os
import string
import math
from itertools import combinations

import py_vncorenlp
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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

# B2: Xây dựng cấu trúc sentences với việc merge các token có dấu gạch ngang
def merge_hyphenated_tokens(tokens_list, pos_list):
    """
    Merge các token được nối bởi dấu gạch ngang thành một token.
    Ví dụ: ['Biên_Hòa', '-', 'Vũng_Tàu'] -> ['Biên_Hòa_-_Vũng_Tàu']
    """
    merged_tokens = []
    merged_pos = []
    
    i = 0
    while i < len(tokens_list):
        # Nếu gặp dấu gạch ngang và xung quanh là PROPN hoặc NOUN
        if (tokens_list[i] == '-' and 
            i > 0 and i < len(tokens_list) - 1 and
            pos_list[i-1] in ['PROPN', 'NOUN'] and 
            pos_list[i+1] in ['PROPN', 'NOUN']):
            # Merge: token trước + dash + token sau
            prev_token = merged_tokens.pop()
            prev_pos = merged_pos.pop()
            merged_token = f"{prev_token}_-_{tokens_list[i+1]}"
            merged_tokens.append(merged_token)
            merged_pos.append(prev_pos)  # Giữ POS của token đầu
            i += 2  # Skip dash và token sau
        else:
            merged_tokens.append(tokens_list[i])
            merged_pos.append(pos_list[i])
            i += 1
    
    return merged_tokens, merged_pos

# Tạo sentences
sentences_tokens = []
sentences_pos = []

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
        # Merge hyphenated tokens
        tokens_list, pos_list = merge_hyphenated_tokens(tokens_list, pos_list)
        sentences_tokens.append(tokens_list)
        sentences_pos.append(pos_list)


# B3: MultipartiteRank class - tôn trọng thuật toán gốc
class MultipartiteRank(TopicRank):
    """Multipartite graph keyphrase extraction model.
    
    Kế thừa TopicRank và mở rộng theo đúng thuật toán gốc của Boudin (2018).
    """
    
    def __init__(self):
        super(MultipartiteRank, self).__init__()
        self.topic_identifiers = {}
        self.graph = nx.DiGraph()
    
    def topic_clustering(self, threshold=0.74, method='average'):
        """Clustering candidates into topics (giống thuật toán gốc)."""
        
        if len(self.candidates) == 1:
            candidate = list(self.candidates.keys())[0]
            self.topics.append([candidate])
            self.topic_identifiers[candidate] = 0
            return
        
        candidates, X = self.vectorize_candidates()
        Y = pdist(X, 'jaccard')
        Y = np.nan_to_num(Y)
        Z = linkage(Y, method=method)
        clusters = fcluster(Z, t=threshold, criterion='distance')
        
        for cluster_id in range(1, max(clusters) + 1):
            self.topics.append([candidates[j] for j in range(len(clusters))
                              if clusters[j] == cluster_id])
        
        for i, cluster_id in enumerate(clusters):
            self.topic_identifiers[candidates[i]] = cluster_id - 1
    
    def build_topic_graph(self):
        """Build the Multipartite graph (giống thuật toán gốc)."""
        
        self.graph.add_nodes_from(self.candidates.keys())
        
        for node_i, node_j in combinations(self.candidates.keys(), 2):
            
            if self.topic_identifiers[node_i] == self.topic_identifiers[node_j]:
                continue
            
            weights = []
            for p_i in self.candidates[node_i].offsets:
                for p_j in self.candidates[node_j].offsets:
                    len_i = len(self.candidates[node_i].lexical_form)
                    len_j = len(self.candidates[node_j].lexical_form)
                    gap = self.compute_gap(p_i, p_j, len_i, len_j)
                    weights.append(1.0 / gap)
            
            if weights:
                self.graph.add_edge(node_i, node_j, weight=sum(weights))
                self.graph.add_edge(node_j, node_i, weight=sum(weights))
    
    def weight_adjustment(self, alpha=1.1):
        """Adjust edge weights (giống thuật toán gốc)."""
        
        weighted_edges = {}
        
        for variants in self.topics:
            
            if len(variants) == 1:
                continue
            
            offsets = [self.candidates[v].offsets[0] for v in variants]
            first = variants[offsets.index(min(offsets))]
            
            for start, end in self.graph.edges(first):
                boosters = []
                for v in variants:
                    if v != first and self.graph.has_edge(v, end):
                        boosters.append(self.graph[v][end]['weight'])
                
                if boosters:
                    weighted_edges[(start, end)] = np.sum(boosters)
        
        for nodes, boosters in weighted_edges.items():
            node_i, node_j = nodes
            position_i = 1.0 / (1 + self.candidates[node_i].offsets[0])
            position_i = math.exp(position_i)
            self.graph[node_j][node_i]['weight'] += (
                boosters * alpha * position_i)
    
    def candidate_weighting(self, threshold=0.74, method='average', alpha=1.1):
        """Candidate weight calculation using random walk (giống thuật toán gốc)."""
        
        if not self.candidates:
            return
        
        self.topic_clustering(threshold=threshold, method=method)
        self.build_topic_graph()
        
        if alpha > 0.0:
            self.weight_adjustment(alpha)
        
        self.weights = nx.pagerank(self.graph)


extractor = MultipartiteRank()

# Tạo input cho PKE
processed_sentences = []
for tokens, pos_tags in zip(sentences_tokens, sentences_pos):
    sent = []
    for token, pos in zip(tokens, pos_tags):
        sub_tokens = token.split('_')
        for sub_token in sub_tokens:
            sent.append((sub_token, pos))
    processed_sentences.append(sent)

extractor.load_document(
    input=processed_sentences,
    language='en',
    normalization=None
)

# B4: Tạo candidates - giữ nguyên logic lọc của bạn
candidates_dict = {}
offset = 0

for sent_id, (tokens, pos_tags) in enumerate(zip(sentences_tokens, sentences_pos)):
    i = 0
    while i < len(tokens):
        if pos_tags[i] not in {'NOUN', 'PROPN', 'ADJ'}:
            offset += len(tokens[i].split('_'))
            i += 1
            continue
        
        start = i
        while i < len(tokens) and pos_tags[i] in {'NOUN', 'PROPN', 'ADJ'}:
            i += 1
        
        if i - start == 0:
            continue
            
        words = tokens[start:i]
        surface = ' '.join(words)
        
        all_sub_tokens = []
        for w in words:
            all_sub_tokens.extend(w.split('_'))
        
        # Kiểm tra stopwords CHỈ ở đầu/cuối/toàn bộ (giữ nguyên logic của bạn)
        first_token = all_sub_tokens[0] if all_sub_tokens else ''
        last_token = all_sub_tokens[-1] if all_sub_tokens else ''
        
        all_stopwords = all(t in STOPWORDS for t in all_sub_tokens)
        starts_with_stopword = first_token in STOPWORDS
        ends_with_stopword = last_token in STOPWORDS
        
        has_boundary_stopword = all_stopwords or starts_with_stopword or ends_with_stopword
        
        # Chứa punctuation
        has_punct = any(sub_t in string.punctuation for sub_t in all_sub_tokens)
        
        # Quá ngắn: chỉ giữ nếu là PROPN hoặc có ≥ 2 token gốc
        is_single_noun = len(words) == 1 and pos_tags[start] != 'PROPN'
        
        if has_boundary_stopword or has_punct or is_single_noun:
            offset += len(all_sub_tokens)
            continue
        
        # Candidate hợp lệ
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

# Inject candidates vào extractor
for surface, info in candidates_dict.items():
    if surface not in extractor.candidates:
        c = Candidate()
        c.lexical_form = info['lexical_form']
        c.surface_forms = info['surface_forms']
        c.offsets = info['offsets']
        c.sentence_ids = info['sentence_ids']
        extractor.candidates[surface] = c

# B5: MultipartiteRank weighting
print(f"\nSố lượng candidates: {len(extractor.candidates)}")
print("Danh sách candidates:")
for c in sorted(extractor.candidates.keys()):
    print(f"  - {c}")

# Chạy với tham số mặc định của thuật toán gốc
extractor.candidate_weighting(threshold=0.74, method='average', alpha=1.1)

# B6: Kết quả
keyphrases = extractor.get_n_best(n=10)

print(f"\n{'─'*55}")
print(f"  TOP 10 KEYPHRASE (MULTIPARTITERANK)")
print(f"{'─'*55}")
for i, (kp, score) in enumerate(keyphrases, 1):
    kp_clean = kp.replace('_', ' ')
    print(f"  {i:2d}. {kp_clean:<35}  {score:.4f}")
print(f"{'─'*55}")

print(f"\n{'─'*55}")
print(f"  THÔNG TIN TOPICS")
print(f"{'─'*55}")
print(f"  Số lượng topics: {len(extractor.topics)}")
for i, topic in enumerate(extractor.topics):
    print(f"\n  Topic {i+1}:")
    for candidate in topic:
        score = extractor.weights.get(candidate, 0)
        clean = candidate.replace('_', ' ')
        print(f"    - {clean:<35} score: {score:.4f}")
print(f"{'─'*55}\n")