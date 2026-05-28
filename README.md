# Combined Pipeline – Tóm tắt & Trích xuất từ khóa tiếng Việt

## Luồng xử lý

```
Văn bản đầu vào
       │
        ▼
backend/textrank/  ──  TextRank (Symmetric emb)
       │         → Danh sách câu quan trọng nhất
       ▼
backend/keybert/  ──  KeyBERT (Asymmetric emb + POS + NER + MMR)
       │        → Danh sách (keyword, score)
       ▼
     OUTPUT
  ┌─────────────────────┐
  │  summary_sentences  │
  │  summary_text       │
  │  keywords           │
  └─────────────────────┘
       │
       ▼
[LLM/] ── Multi-Label Clustering

```

## Danh sách các Models trong hệ thống

| Tên File / Thư mục | Nguồn gốc (Hugging Face) | Vai trò | Kích thước |
| :--- | :--- | :--- | :--- |
| **`vncorenlp/`** | [py_vncorenlp](https://github.com/vncorenlp/VnCoreNLP) | Tách từ (Word Segmentation) & gán nhãn từ loại (POS Tagging) | ~40 MB |
| **`ner-vietnamese-electra-base.pt`** | [NlpHUST/ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base) | Nhận diện thực thể tên riêng (Tên người, Địa điểm, Tổ chức) | ~532 MB |
| **`ner-tokenizer/`** | [NlpHUST/ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base) | Bộ tách từ (Tokenizer) đi kèm cho mô hình NER | ~1.4 MB |
| **`symmetric_emb/`** | [dangvantuan/vietnamese-embedding](https://huggingface.co/dangvantuan/vietnamese-embedding) | Embedding đồng dạng (Symmetric) dùng cho TextRank tóm tắt | ~540 MB |
| **`asymmetric_emb/`** | [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding) | Embedding bất đồng dạng (Asymmetric) dùng cho trích xuất từ khóa | ~2.2 GB |


---

```bash
# 1. Kích hoạt venv (nếu chưa kích hoạt)
source venv/bin/activate

# 2. Chạy script tải tự động
python download_models.py
```

---


## Sử dụng

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

print(result)
```
---

## Demo nhanh CLI
```bash
python -m backend.cluster.combine_pipeline
```


## Sử dụng – Backend
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Sử dụng – Frontend
```bash
cd frontend
npm install
npm run dev
```