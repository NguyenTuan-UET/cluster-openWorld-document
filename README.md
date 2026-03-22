# Combined Pipeline – Tóm tắt & Trích xuất từ khóa tiếng Việt

> **Self-contained** – toàn bộ code, model và venv nằm trong thư mục này,
> không phụ thuộc vào `text-summarize/` hay `keyword-extraction-viet/`.

## Luồng xử lý

```
Văn bản đầu vào
       │
       ▼
[textrank/]  ──  TextRank (VnCoreNLP wseg)
       │         → Danh sách câu quan trọng nhất
       ▼
Join thành đoạn văn tóm tắt
       │
       ▼
[keybert/]  ──  KeyBERT-Vi (PhoBERT + NER-ELECTRA, load từ .pt local)
       │        → Danh sách (keyword, score)
       ▼
     OUTPUT
  ┌───────────────────────────────────────────┐
  │  summary_sentences : List[str]            │
  │  summary_text      : str                  │
  │  keywords          : List[(str, float)]   │
  └───────────────────────────────────────────┘
```

**Ưu điểm:** dùng chung **1 instance VnCoreNLP** (`wseg + pos`) cho cả 2 bước → tiết kiệm RAM.

---

## Cấu trúc thư mục

```
combined_pipeline/
├── venv/                              ← virtual env riêng (Python 3.12)
├── pretrained-models/                 ← Tất cả models 
│   ├── phobert.pt                     ← PhoBERT full model
│   ├── ner-vietnamese-electra-base.pt ← NER-ELECTRA full model
│   └── vncorenlp/                    ← VnCoreNLP JAR + models
├── textrank/                          ← copy từ text-summarize/
│   ├── textrank_facade.py
│   ├── tools/
│   └── stopwords/
├── keybert/                           ← copy từ keyword-extraction-viet/
│   ├── pipeline.py
│   ├── model/
│   └── vietnamese-stopwords-dash.txt
├── combined_pipeline.py               ← Class CombinedPipeline + CombinedResult
├── app.py                             ← Gradio web interface
├── requirements.txt
└── README.md
```

---

## Sử dụng – Python API

```python
from combined_pipeline import CombinedPipeline

# Khởi tạo với tham số mặc định
pipeline = CombinedPipeline(
    top_n=10,
    ngram_n=(1, 3),
    min_freq=1,
    diversify_result=False,
)

# Load model (hoặc sẽ tự động khi run() lần đầu)
pipeline.load()

# Chạy pipeline
result = pipeline.run(
    text="Việt Nam là quốc gia đang phát triển nhanh ở Đông Nam Á...",
    title="Việt Nam",   # optional
)

# Kết quả
print(result.summary_sentences)   # List[str]             – các câu tóm tắt
print(result.summary_text)        # str                   – đoạn văn join
print(result.keywords)            # List[(str, float)]    – keyword + score

# In đẹp toàn bộ
print(result)
```

### Cấu trúc `CombinedResult`

| Thuộc tính          | Kiểu                     | Mô tả                                       |
|---------------------|--------------------------|---------------------------------------------|
| `original_text`     | `str`                    | Văn bản gốc truyền vào                      |
| `summary_sentences` | `List[str]`              | Danh sách câu quan trọng (đúng thứ tự gốc)  |
| `summary_text`      | `str`                    | Các câu tóm tắt join bằng dấu cách          |
| `keywords`          | `List[Tuple[str,float]]` | Từ khóa + điểm số, sắp xếp giảm dần         |

---
### Tham số trên giao diện

| Tham số          | Mặc định | Mô tả                                          |
|------------------|----------|------------------------------------------------|
| Top N keywords   | 10       | Số lượng từ khóa muốn lấy                      |
| Ngram low range  | 1        | Độ dài ngram tối thiểu (1 = từ đơn)            |
| Ngram high range | 3        | Độ dài ngram tối đa (3 = cụm 3 từ)            |
| Min frequency    | 1        | Tần suất tối thiểu (tăng lên cho văn bản dài)  |
| Diversify result | False    | Đa dạng hóa bằng K-means clustering            |

---

## Sử dụng – Gradio Web App
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Demo nhanh
```bash
python combined_pipeline.py
```

## Sử dụng – Gradio Web App
```bash
python app.py
```