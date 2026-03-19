"""
Combined Pipeline - Tóm tắt + Trích xuất từ khóa + Phân nhóm chủ đề
=====================================================================
Cấu trúc thư mục (self-contained):

  combined_pipeline/
  ├── venv/
  ├── vncorenlp/
  ├── pretrained-models/
  │   ├── phobert.pt
  │   └── ner-vietnamese-electra-base.pt
  ├── textrank/
  ├── keybert/
  ├── gemini_service.py          ← Gemini API cho phân nhóm
  ├── combined_pipeline.py       ← file này
  ├── app.py
  └── .env                       ← GEMINI_API_KEY

Luồng xử lý:
  1. TextRankFacade.summarize(text)   → List[str]  (câu quan trọng)
  2. KeywordExtractorPipeline(...)    → List[(keyword, score)]
  3. GeminiService.classify_document() → TopicLabel[] (phân nhóm chủ đề)

Bước 3 (Gemini) là TÙY CHỌN — pipeline vẫn hoạt động nếu không có API key.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─── Đường dẫn nội bộ (tất cả đều nằm trong combined_pipeline/) ──────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
TEXTRANK_DIR        = os.path.join(BASE_DIR, "textrank")
KEYBERT_DIR         = os.path.join(BASE_DIR, "keybert")
PRETRAINED_DIR      = os.path.join(BASE_DIR, "pretrained-models")
VNCORENLP_DIR       = os.path.join(PRETRAINED_DIR, "vncorenlp")
PHOBERT_PT          = os.path.join(PRETRAINED_DIR, "phobert.pt")
NER_PT              = os.path.join(PRETRAINED_DIR, "ner-vietnamese-electra-base.pt")

# ─── Thêm sys.path để import được textrank/ và keybert/ ──────────────────────
for p in [TEXTRANK_DIR, KEYBERT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─── Import từ textrank/ ──────────────────────────────────────────────────────
from textrank_facade import TextRankFacade          # noqa: E402
from stopwords.vietnamese import Vietnamese         # noqa: E402

# ─── Import từ keybert/ ───────────────────────────────────────────────────────
from pipeline import KeywordExtractorPipeline       # noqa: E402

# ─── Import Gemini service (phân nhóm chủ đề) ────────────────────────────────
sys.path.insert(0, BASE_DIR)
from gemini_service import GeminiService, TopicLabel, ClassifyResult  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass kết quả trả về
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CombinedResult:
    """Kết quả tổng hợp của toàn bộ pipeline cho 1 tài liệu."""

    original_text:     str
    summary_sentences: List[str]              = field(default_factory=list)
    summary_text:      str                    = ""
    keywords:          List[Tuple[str,float]] = field(default_factory=list)
    # ── Clustering (Bước 3 — Gemini, optional) ────────────────────────────────
    label_ids:         List[str]              = field(default_factory=list)
    title:             Optional[str]          = None

    def __str__(self) -> str:
        sep = "=" * 60
        kw_str = "\n".join(
            f"  {i+1:2d}. {kw:<30s}  (score: {sc:.4f})"
            for i, (kw, sc) in enumerate(self.keywords)
        )
        label_str = ", ".join(self.label_ids) if self.label_ids else "(chưa phân nhóm)"
        return (
            f"\n{sep}\n"
            f"VĂN BẢN GỐC:\n{self.original_text.strip()}\n\n"
            f"{sep}\n"
            f"TÓM TẮT ({len(self.summary_sentences)} câu):\n{self.summary_text}\n\n"
            f"{sep}\n"
            f"TỪ KHÓA ({len(self.keywords)} keywords):\n{kw_str}\n\n"
            f"{sep}\n"
            f"NHÃN CHỦ ĐỀ: {label_str}\n"
            f"{sep}\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Class chính
# ──────────────────────────────────────────────────────────────────────────────

class CombinedPipeline:
    """
    Pipeline tổng hợp: tóm tắt → trích xuất từ khóa → phân nhóm chủ đề.

    Bước 1–2: local (TextRank + KeyBERT-Vi), luôn hoạt động.
    Bước 3: Gemini API (phân nhóm), TÙY CHỌN — bật bằng enable_clustering=True.

    Parameters
    ----------
    top_n            : số từ khóa trả về        (default: 10)
    ngram_n          : khoảng ngram (low, high)  (default: (1, 3))
    min_freq         : tần suất tối thiểu        (default: 1)
    diversify_result : đa dạng hóa bằng K-means  (default: False)
    enable_clustering: bật phân nhóm Gemini       (default: False)
    gemini_api_key   : API key (hoặc dùng .env)

    Example
    -------
    >>> p = CombinedPipeline(enable_clustering=True)
    >>> results, labels = p.run_batch(texts, titles)
    """

    def __init__(
        self,
        top_n: int = 10,
        ngram_n: Tuple[int, int] = (1, 3),
        min_freq: int = 1,
        diversify_result: bool = False,
        enable_clustering: bool = False,
        gemini_api_key: Optional[str] = None,
    ):
        self.top_n             = top_n
        self.ngram_n           = ngram_n
        self.min_freq          = min_freq
        self.diversify_result  = diversify_result
        self.enable_clustering = enable_clustering
        self._gemini_api_key   = gemini_api_key
        self._is_loaded        = False

        # ── Clustering state (persist across run_batch) ───────────────────────
        self._labels: List[TopicLabel] = []
        self._gemini: Optional[GeminiService] = None

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def load(self) -> "CombinedPipeline":
        """
        Load tất cả model vào bộ nhớ.
        Gọi tường minh hoặc sẽ tự động chạy ở lần run() đầu tiên.
        """
        if self._is_loaded:
            return self

        import torch

        # ── [1/4] VnCoreNLP (dùng chung 1 instance) ──────────────────────────
        print("⏳ [1/4] Đang tải VnCoreNLP (wseg + pos)…")
        import py_vncorenlp
        self._vncorenlp = py_vncorenlp.VnCoreNLP(
            annotators=["wseg", "pos"],
            save_dir=VNCORENLP_DIR,
        )
        print("✅ VnCoreNLP sẵn sàng!\n")

        # ── [2/4] PhoBERT từ file .pt local ──────────────────────────────────
        print(f"⏳ [2/4] Đang tải PhoBERT từ {PHOBERT_PT} …")
        phobert = torch.load(PHOBERT_PT, map_location="cpu", weights_only=False)
        phobert.eval()
        print("✅ PhoBERT sẵn sàng!\n")

        # ── [3/4] NER model từ file .pt local ────────────────────────────────
        print(f"⏳ [3/4] Đang tải NER model từ {NER_PT} …")
        ner_model = torch.load(NER_PT, map_location="cpu", weights_only=False)
        ner_model.eval()
        print("✅ NER model sẵn sàng!\n")

        # ── [4/4] Khởi tạo TextRank + KeyBERT pipeline ───────────────────────
        print("⏳ [4/4] Đang khởi tạo TextRank + KeyBERT pipeline…")
        stopwords = Vietnamese()
        self._summarizer = TextRankFacade(self._vncorenlp, stopwords)

        # Truyền vncorenlp_instance → tái sử dụng, không load lại
        self._kw_pipeline = KeywordExtractorPipeline(
            model=phobert,
            ner_model=ner_model,
            vncorenlp_instance=self._vncorenlp,
        )
        print("✅ Tất cả pipeline sẵn sàng!\n")

        # ── [5/5] Gemini Service (optional) ───────────────────────────────────
        if self.enable_clustering:
            print("⏳ [5/5] Đang khởi tạo Gemini Service (phân nhóm chủ đề)…")
            try:
                self._gemini = GeminiService(api_key=self._gemini_api_key)
                self._gemini._ensure_client()
                print("✅ Gemini Service sẵn sàng!\n")
            except Exception as e:
                print(f"⚠️ Gemini Service không khả dụng: {e}")
                print("   → Bước phân nhóm sẽ bị bỏ qua.\n")
                self._gemini = None
        else:
            print("ℹ️  Phân nhóm (Gemini) tắt. Bật bằng enable_clustering=True.\n")

        self._is_loaded = True
        return self

    @property
    def labels(self) -> List[TopicLabel]:
        """Danh sách nhãn chủ đề hiện tại."""
        return list(self._labels)

    def reset_labels(self) -> None:
        """Xóa toàn bộ nhãn chủ đề (reset clustering)."""
        self._labels.clear()

    # ── Core method ───────────────────────────────────────────────────────────
    def run(
        self,
        text: str,
        title: Optional[str] = None,
        max_sentences: Optional[int] = None,
    ) -> CombinedResult:
        """
        Chạy toàn bộ pipeline: tóm tắt → trích xuất từ khóa → phân nhóm.

        Parameters
        ----------
        text          : văn bản tiếng Việt đầu vào
        title         : tiêu đề (optional)
        max_sentences : số câu tóm tắt tối đa; None = tự động

        Returns
        -------
        CombinedResult  (summary_sentences, summary_text, keywords, label_ids)
        """
        if not self._is_loaded:
            self.load()

        # ── Bước 1: Tóm tắt ──────────────────────────────────────────────────
        print("🔄 Bước 1: Tóm tắt văn bản…")
        summary_sentences: List[str] = self._summarizer.summarize(
            text, max_sentences=max_sentences
        )
        summary_text: str = " ".join(summary_sentences)
        print(f"   → {len(summary_sentences)} câu được chọn.\n")

        # ── Bước 2: Trích xuất từ khóa từ đoạn tóm tắt ───────────────────────
        print("🔄 Bước 2: Trích xuất từ khóa từ đoạn văn tóm tắt…")
        inp = {"text": summary_text, "title": title}
        raw_keywords = self._kw_pipeline(
            inputs=inp,
            min_freq=self.min_freq,
            ngram_n=self.ngram_n,
            top_n=self.top_n,
            diversify_result=self.diversify_result,
        )
        keywords = list(raw_keywords)
        print(f"   → {len(keywords)} từ khóa được trích xuất.\n")

        # ── Bước 3: Phân nhóm chủ đề (Gemini, optional) ──────────────────────
        label_ids: List[str] = []
        if self._gemini is not None:
            print("🔄 Bước 3: Phân nhóm chủ đề (Gemini)…")
            try:
                kw_names = [kw for kw, _ in keywords]
                classify_result = self._gemini.classify_document(
                    summary_text=summary_text,
                    keywords=kw_names,
                    title=title,
                    existing_labels=self._labels,
                )
                # Cập nhật label registry
                for new_lb in classify_result.new_labels:
                    self._labels.append(new_lb)
                # Cập nhật document count
                for lb_id in classify_result.assigned_label_ids:
                    for lb in self._labels:
                        if lb.id == lb_id:
                            lb.document_count += 1
                label_ids = classify_result.assigned_label_ids
                print(f"   → Gán {len(label_ids)} nhãn, "
                      f"{len(classify_result.new_labels)} nhãn mới được tạo.\n")
            except Exception as e:
                print(f"   ⚠️ Phân nhóm thất bại: {e}\n")
        else:
            print("ℹ️  Bỏ qua bước phân nhóm (Gemini không khả dụng).\n")

        return CombinedResult(
            original_text=text,
            summary_sentences=summary_sentences,
            summary_text=summary_text,
            keywords=keywords,
            label_ids=label_ids,
            title=title,
        )

    # ── Batch processing cho multi-doc clustering ─────────────────────────────
    def run_batch(
        self,
        texts: List[str],
        titles: Optional[List[Optional[str]]] = None,
        max_sentences: Optional[int] = None,
    ) -> Tuple[List[CombinedResult], List[TopicLabel]]:
        """
        Xử lý nhiều tài liệu → tóm tắt + keyword + phân nhóm.

        Quy trình mới:
          1. Tóm tắt + trích xuất từ khóa cho TẤT CẢ tài liệu trước.
          2. Gửi danh sách từ khóa của TẤT CẢ tài liệu cho Gemini để phân nhóm
             một lần duy nhất (thay vì phân nhóm từng tài liệu riêng lẻ).

        Parameters
        ----------
        texts  : danh sách văn bản
        titles : danh sách tiêu đề (hoặc None)

        Returns
        -------
        (results, labels) — results[i] tương ứng texts[i]
        """
        if titles is None:
            titles = [None] * len(texts)

        results: List[CombinedResult] = []

        # ── Bước 1+2: Tóm tắt + Trích xuất từ khóa cho TẤT CẢ tài liệu ────
        print("\n" + "=" * 60)
        print("  BƯỚC 1+2: Tóm tắt + Trích xuất từ khóa cho tất cả tài liệu")
        print("=" * 60)

        for i, (text, title) in enumerate(zip(texts, titles)):
            print(f"\n{'─' * 60}")
            print(f"📄 Tài liệu {i+1}/{len(texts)}: {(title or '(không tiêu đề)')[:50]}")
            print(f"{'─' * 60}")

            if not self._is_loaded:
                self.load()

            # ── Bước 1: Tóm tắt ──────────────────────────────────────────────
            print("🔄 Bước 1: Tóm tắt văn bản…")
            summary_sentences: List[str] = self._summarizer.summarize(
                text, max_sentences=max_sentences
            )
            summary_text: str = " ".join(summary_sentences)
            print(f"   → {len(summary_sentences)} câu được chọn.\n")

            # ── Bước 2: Trích xuất từ khóa ───────────────────────────────────
            print("🔄 Bước 2: Trích xuất từ khóa…")
            inp = {"text": summary_text, "title": title}
            raw_keywords = self._kw_pipeline(
                inputs=inp,
                min_freq=self.min_freq,
                ngram_n=self.ngram_n,
                top_n=self.top_n,
                diversify_result=self.diversify_result,
            )
            keywords = list(raw_keywords)
            print(f"   → {len(keywords)} từ khóa được trích xuất.\n")

            results.append(CombinedResult(
                original_text=text,
                summary_sentences=summary_sentences,
                summary_text=summary_text,
                keywords=keywords,
                label_ids=[],      # sẽ gán ở bước 3
                title=title,
            ))

        # ── Bước 3: Phân nhóm chủ đề bằng Gemini (1 lần cho TẤT CẢ) ────────
        if self._gemini is not None:
            print("\n" + "=" * 60)
            print("  BƯỚC 3: Phân nhóm chủ đề bằng Gemini (dựa trên từ khóa)")
            print("=" * 60)

            # Chuẩn bị danh sách từ khóa của tất cả tài liệu
            documents_keywords = []
            for i, r in enumerate(results):
                kw_names = [kw for kw, _ in r.keywords]
                documents_keywords.append({
                    "doc_index": i,
                    "title": r.title,
                    "keywords": kw_names,
                })
                print(f"   📄 Tài liệu {i+1}: {len(kw_names)} từ khóa → {', '.join(kw_names[:5])}{'…' if len(kw_names) > 5 else ''}")

            print(f"\n🔄 Đang gửi từ khóa của {len(results)} tài liệu cho Gemini để phân nhóm…")
            try:
                cluster_result = self._gemini.cluster_documents_by_keywords(
                    documents_keywords=documents_keywords,
                )

                # Cập nhật labels
                self._labels = cluster_result["labels"]

                # Gán label_ids cho từng document
                assignments = cluster_result["assignments"]
                for doc_idx, label_ids in assignments.items():
                    if 0 <= doc_idx < len(results):
                        results[doc_idx].label_ids = label_ids

                print(f"\n   ✅ Tạo {len(self._labels)} nhóm chủ đề:")
                for lb in self._labels:
                    doc_indices = [
                        str(i + 1) for i in range(len(results))
                        if lb.id in results[i].label_ids
                    ]
                    print(f"      🏷️ {lb.name} → Tài liệu [{', '.join(doc_indices)}]")

            except Exception as e:
                print(f"   ⚠️ Phân nhóm thất bại: {e}\n")
        else:
            print("\nℹ️  Bỏ qua bước phân nhóm (Gemini không khả dụng).\n")

        return results, list(self._labels)


# ──────────────────────────────────────────────────────────────────────────────
# Demo nhanh khi chạy trực tiếp
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_text = """
    Trí tuệ nhân tạo (AI) đang thay đổi mọi khía cạnh của cuộc sống hiện đại.
    Từ y tế, giáo dục đến giao thông vận tải, AI mang lại những cải tiến vượt bậc.
    Các thuật toán học máy giúp chẩn đoán bệnh chính xác hơn bác sĩ trong nhiều trường hợp.
    Xe tự lái ứng dụng deep learning để nhận diện đường đi và tránh va chạm.
    Chatbot được trang bị xử lý ngôn ngữ tự nhiên hỗ trợ khách hàng 24/7.
    Tuy nhiên, AI cũng đặt ra nhiều thách thức về đạo đức và quyền riêng tư.
    Các chuyên gia khuyến nghị cần có khung pháp lý rõ ràng để quản lý AI.
    Việt Nam đang đẩy mạnh ứng dụng AI vào các lĩnh vực trọng điểm quốc gia.
    """

    pipeline = CombinedPipeline(top_n=10, ngram_n=(1, 3), min_freq=1, diversify_result=False)
    result = pipeline.run(text=sample_text, title="Trí tuệ nhân tạo và tương lai")
    print(result)
