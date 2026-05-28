import os
import yake
import py_vncorenlp

# ── Đường dẫn (chỉnh sửa nếu cần) ────────────────────────────────────────────
_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VNCORENLP_DIR  = os.path.join(_BASE_DIR, "pretrained-models", "vncorenlp")
_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "vietnamese-stopwords-dash.txt")

with open(_STOPWORDS_PATH, encoding='utf-8') as f:
    STOPWORDS = {line.strip().lower() for line in f if line.strip()}

# ── Khởi tạo annotator (một lần) ─────────────────────────────────────────────
annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=_VNCORENLP_DIR)

text = """
Rạng sáng 9/5, một vụ tai nạn xảy ra trên tuyến cao tốc Biên Hòa - Vũng Tàu thuộc thành phần 1 đoạn qua phường Phước Tân, TP Đồng Nai (đang thi công) khiến một nam thanh niên tử vong tại chỗ.
Theo thông tin ban đầu, khoảng 2h30 cùng ngày, nam thanh niên điều khiển xe máy mang BKS 60AA-628.XX lưu thông trên tuyến cao tốc đang thi công.
Khi di chuyển đến khu vực cầu vượt tại nút giao với đường Võ Nguyên Giáp (phường Phước Tân), xe máy bất ngờ tông mạnh vào dải phân cách bằng bê tông.
Cú tông mạnh khiến phương tiện mắc kẹt giữa các khối bê tông; nạn nhân văng qua khe hở của dải phân cách, rơi xuống phía dưới cầu vượt, tử vong tại chỗ.
Nhận tin báo, lực lượng chức năng TP Đồng Nai đã có mặt để phong tỏa, khám nghiệm hiện trường và điều tra nguyên nhân vụ tai nạn.
Được biết, cao tốc Biên Hòa - Vũng Tàu đoạn thành phần 1 vẫn đang thi công, nhiều hạng mục chưa hoàn thiện. Dự kiến đoạn tuyến sẽ được đưa vào khai thác trước ngày 18/5.

"""

# B1: Word segmentation — VnCoreNLP ghép từ đa âm bằng '_'
sentences = annotator.word_segment(text)
segmented = ' '.join(sentences).replace('_', '-')
 
# B2: YAKE trích xuất keyphrase (có kèm stopwords và dedup)
extractor = yake.KeywordExtractor(
    lan='en',
    n=3,                 # max_ngram
    dedupLim=0.4,
    dedupFunc='seqm',
    windowsSize=2,
    top=100,             # top_n * 10 (ở đây top_n=10)
    stopwords=STOPWORDS
)
raw_keywords = extractor.extract_keywords(segmented)

# B3: Lọc + dedup (word-level overlap) + chuẩn hoá
results = []
seen = set()
seen_tokens = []         # lưu các tập token đã chọn

for kw, score in raw_keywords:
    kw_clean = kw.replace('-', ' ').strip()
    kw_lower = kw_clean.lower()
    tokens = kw_lower.split()

    # Loại trùng chính xác
    if kw_lower in seen:
        continue
    # Loại toàn bộ là stopword
    if all(t in STOPWORDS for t in tokens):
        continue

    # Word-level overlap ≥ 50%
    token_set = set(tokens)
    overlap_flag = False
    for prev_tokens in seen_tokens:
        overlap = len(token_set & prev_tokens)
        min_len = min(len(token_set), len(prev_tokens))
        if min_len > 0 and overlap / min_len >= 0.5:
            overlap_flag = True
            break
    if overlap_flag:
        continue

    # Lưu lại
    seen.add(kw_lower)
    seen_tokens.append(token_set)
    results.append((kw_clean, score))

    if len(results) == 10:   # top_n
        break

# ── In kết quả ───────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  TOP 10 KEYPHRASE (YAKE + VnCoreNLP)")
print(f"{'─'*50}")
for i, (kp, score) in enumerate(results, 1):
    print(f"  {i:2d}. {kp:<35}  {score:.4f}")
print(f"{'─'*50}")
print("  (Score càng thấp → từ khóa càng quan trọng)\n")