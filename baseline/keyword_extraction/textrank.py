
import networkx as nx
from underthesea import sent_tokenize, pos_tag

# ========== Văn bản mẫu ==========
text = """
Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy
bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương
Mỵ Châu – Trọng Thủy. Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn
khám phá được những giá trị khảo cổ to lớn của Cổ Loa.
Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích
bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước.
"""

# ========== Tham số ==========
window = 2                       # cửa sổ đồng xuất hiện
valid_pos = {'N', 'Np', 'Ny', 'Nu', 'A'}   # tương đương NOUN, PROPN, ADJ
top_n = 10

# ========== 1. Tiền xử lý & tách câu ==========
sentences = sent_tokenize(text)

# Tạo danh sách phẳng toàn bộ các từ kèm POS và trạng thái "có được đưa vào đồ thị không"
all_words = []   # mỗi phần tử là (word_clean, pos, is_valid)
for sent in sentences:
    tagged = pos_tag(sent)
    for word, pos in tagged:
        w_clean = word.strip().replace(" ", "_")
        is_valid = pos in valid_pos
        all_words.append((w_clean, pos, is_valid))

# ========== 2. Xây đồ thị không trọng số ==========
graph = nx.Graph()

# Thêm đỉnh: chỉ những từ có is_valid = True
for w, pos, valid in all_words:
    if valid:
        graph.add_node(w)

# Thêm cạnh dựa trên cửa sổ, duyệt tất cả các từ (kể cả stopword, dấu câu)
for i, (word1, pos1, valid1) in enumerate(all_words):
    if not valid1:
        continue
    for j in range(i + 1, min(i + window, len(all_words))):
        word2, pos2, valid2 = all_words[j]
        if valid2 and word1 != word2:
            # Đồ thị không trọng số, có thể thêm nhiều lần nhưng vẫn là một cạnh duy nhất
            graph.add_edge(word1, word2)

# ========== 3. PageRank không trọng số ==========
word_ranks = nx.pagerank(graph, alpha=0.85, tol=0.0001, weight=None)

# ========== 4. Tạo ứng viên từ khóa ==========
# Lấy các chuỗi từ có POS hợp lệ dài nhất trong từng câu (longest_pos_sequence)
candidates = []
for sent in sentences:
    tagged = pos_tag(sent)
    seq = []
    for word, pos in tagged:
        w_clean = word.strip().replace(" ", "_")
        if pos in valid_pos:
            seq.append(w_clean)
        else:
            if seq:
                candidates.append(tuple(seq))
                seq = []
    if seq:
        candidates.append(tuple(seq))

# Loại bỏ trùng lặp
unique_candidates = []
for c in candidates:
    if c not in unique_candidates:
        unique_candidates.append(c)

# ========== 5. Tính điểm & phá vỡ hoà ==========
# Tính offset: vị trí xuất hiện đầu tiên của mỗi từ trong all_words
first_offset = {}
for idx, (w, _, _) in enumerate(all_words):
    if w not in first_offset:
        first_offset[w] = idx

candidate_scores = {}
for cand in unique_candidates:
    base_score = sum(word_ranks.get(word, 0.0) for word in cand)
    # Phá vỡ hoà bằng vị trí đầu tiên của từ đầu cụm (như pke dùng offsets[0] * 1e-8)
    first_word = cand[0]
    tie_breaker = first_offset.get(first_word, 0) * 1e-8
    candidate_scores[" ".join(cand)] = base_score + tie_breaker

# ========== 6. Sắp xếp & in kết quả ==========
sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

print("--- KẾT QUẢ TỪ KHÓA (TextRank mô phỏng pke) ---")
for i, (phrase, score) in enumerate(sorted_candidates[:top_n], 1):
    print(f"{i}. {phrase} ({score:.4f})")


# import string
# import networkx as nx
# from underthesea import word_tokenize, pos_tag

# # 1. Văn bản đầu vào
# text = """
# Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy
# bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương
# Mỵ Châu – Trọng Thủy. Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn
# khám phá được những giá trị khảo cổ to lớn của Cổ Loa.
# Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích
# bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước.
# """

# # 2. Tiền xử lý: Tách từ tiếng Việt
# text_segmented = word_tokenize(text, format="text").lower()

# # 3. Đọc danh sách stopwords ĐÃ CHUẨN ĐỊNH DẠNG
# with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
#     stopwords = set([line.strip().lower() for line in f if line.strip()])


# top_n = 10
# window_size = 4

# # 3. Tách từ, lọc POS tag (Danh/Động/Tính từ) và loại bỏ stopwords
# raw_tokens = pos_tag(text)
# allowed_pos = {'N', 'Np', 'Nu', 'Ny', 'A', 'V'}

# cleaned_tokens = []
# for word, pos in raw_tokens:
#     w_clean = word.lower().strip().replace(" ", "_")
#     if pos in allowed_pos and len(w_clean) > 1 and w_clean not in stopwords:
#         cleaned_tokens.append(w_clean)

# # đồ thị vô hướng với quan hệ đồng xuất hiện
# graph = nx.Graph()
# graph.add_nodes_from(cleaned_tokens)

# for i, word in enumerate(cleaned_tokens):
#     for j in range(i + 1, min(i + window_size, len(cleaned_tokens))):
#         next_word = cleaned_tokens[j]
#         if word != next_word:
#             if graph.has_edge(word, next_word):
#                 weight_past = graph[word][next_word].get('weight', 0)
#                 weight = weight_past + 1.0
#             else:
#                 weight = 1.0
#             graph.add_edge(word, next_word, weight=weight)

# # 5. Chạy PageRank để tính điểm cho các từ đơn lẻ
# word_ranks = nx.pagerank(graph, weight='weight')

# # 6. Quét lại văn bản gốc để ghép các từ đơn có điểm cao đứng cạnh nhau thành cụm từ
# # Lấy dấu câu làm điểm dừng cụm từ
# punctuations = set(string.punctuation)  # bao gồm: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# orig_words_pos = raw_tokens  # (word, pos)
# orig_words_processed = []
# for w, pos in orig_words_pos:
#     w_lower = w.lower().replace(" ", "_")
#     orig_words_processed.append(w_lower)

# top_words = set(sorted(word_ranks, key=word_ranks.get, reverse=True)[:top_n * 3])

# candidate_phrases = []
# current_phrase = []

# for i, word in enumerate(orig_words_processed):
#     # Lấy từ gốc (chưa thay dấu cách) để kiểm tra dấu câu
#     original_word_raw, _ = orig_words_pos[i]
#     is_punctuation = any(p in original_word_raw for p in punctuations)
    
#     if is_punctuation:
#         # Gặp dấu câu → kết thúc cụm từ hiện tại
#         if current_phrase:
#             candidate_phrases.append(tuple(current_phrase))
#             current_phrase = []
#     else:
#         if word in top_words:
#             current_phrase.append(word)
#         else:
#             if current_phrase:
#                 candidate_phrases.append(tuple(current_phrase))
#                 current_phrase = []

# # Kết thúc văn bản
# if current_phrase:
#     candidate_phrases.append(tuple(current_phrase))

# # Lọc bỏ các cụm trùng và cụm rỗng
# candidate_phrases = [p for p in candidate_phrases if p]

# print("Các cụm từ tiềm năng:", candidate_phrases)

# # 7. Tính tổng điểm cho các cụm từ vừa ghép và loại bỏ cụm trùng lặp trùng khít
# phrase_scores = {" ".join(p) : sum(word_ranks.get(w, 0) for w in p) for p in candidate_phrases}
# sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)

# unique_phrases = []
# for phrase, score in sorted_phrases:
#     if not any(phrase in seen for seen, _ in unique_phrases):
#         unique_phrases.append((phrase, score))

# # 8. In kết quả cuối cùng
# print("\n--- KẾT QUẢ TỪ KHÓA TRÍCH XUẤT ---")
# for i, (kw, score) in enumerate(unique_phrases[:top_n], 1):
#     print(f"{i}. {kw.title()} ({score:.4f})")
