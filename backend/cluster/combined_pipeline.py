"""
Combined Pipeline - Tóm tắt + Trích xuất từ khóa
=================================================

Luồng xử lý:
  1. TextRankEmbedding.summarize(text)  → List[str]  (câu quan trọng)
  2. KeywordExtractorPipeline(...)      → List[(keyword, score)]
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─── Đường dẫn nội bộ ────────────────────────────────────────────────────────
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR       = os.path.join(PROJECT_ROOT, "backend")
TEXTRANK_DIR      = os.path.join(BACKEND_DIR, "cluster", "textrank")
KEYBERT_DIR       = os.path.join(BACKEND_DIR, "cluster", "keybert")
PRETRAINED_DIR    = os.path.join(PROJECT_ROOT, "pretrained-models")
VNCORENLP_DIR     = os.path.join(PRETRAINED_DIR, "vncorenlp")
NER_PT            = os.path.join(PRETRAINED_DIR, "ner-vietnamese-electra-base.pt")
SYMMETRIC_EMB_DIR = os.path.join(PRETRAINED_DIR, "symmetric_emb")
ASYMM_EMB_DIR     = os.path.join(PRETRAINED_DIR, "asymmetric_emb")

# ─── Thêm sys.path ────────────────────────────────────────────────────────────
CLUSTER_DIR = os.path.join(BACKEND_DIR, "cluster")
for p in [TEXTRANK_DIR, CLUSTER_DIR, KEYBERT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from textrank.textrank_embedding import TextRankEmbedding  # noqa: E402
from keybert.pipeline import KeywordExtractorPipeline      # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass kết quả
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CombinedResult:
    """Kết quả pipeline cho 1 tài liệu: tóm tắt + từ khóa."""

    original_text:     str
    summary_sentences: List[str]              = field(default_factory=list)
    summary_text:      str                    = ""
    keywords:          List[Tuple[str,float]] = field(default_factory=list)
    title:             Optional[str]          = None

    def __str__(self) -> str:
        sep = "=" * 60
        kw_str = "\n".join(
            f"  {i+1:2d}. {kw:<30s}  (score: {sc:.4f})"
            for i, (kw, sc) in enumerate(self.keywords)
        )
        return (
            f"\n{sep}\n"
            f"VĂN BẢN GỐC:\n{self.original_text.strip()}\n\n"
            f"{sep}\n"
            f"TÓM TẮT ({len(self.summary_sentences)} câu):\n{self.summary_text}\n\n"
            f"{sep}\n"
            f"TỪ KHÓA ({len(self.keywords)} keywords):\n{kw_str}\n"
            f"{sep}\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Class chính
# ──────────────────────────────────────────────────────────────────────────────

class CombinedPipeline:
    """
    Pipeline: tóm tắt (TextRank + Symmetric Embedding) → trích xuất từ khóa (KeyBERT-Vi).

    Parameters
    ----------
    top_n      : số từ khóa trả về       (default: 10)
    ngram_n    : khoảng ngram (low, high) (default: (1, 3))
    min_freq   : tần suất tối thiểu      (default: 1)
    use_mmr    : dùng MMR                (default: False)
    use_kmeans : dùng K-Means            (default: False)
    diversity  : hệ số MMR [0.0–1.0]    (default: 0.5)
    """

    def __init__(
        self,
        top_n: int = 10,
        ngram_n: Tuple[int, int] = (1, 3),
        min_freq: int = 1,
        use_mmr: bool = False,
        use_kmeans: bool = False,
        diversity: float = 0.5,
    ):
        self.top_n       = top_n
        self.ngram_n     = ngram_n
        self.min_freq    = min_freq
        self.use_mmr     = use_mmr
        self.use_kmeans  = use_kmeans
        self.diversity   = diversity
        self._is_loaded  = False

    def load(self) -> "CombinedPipeline":
        """Load tất cả model vào bộ nhớ."""
        if self._is_loaded:
            return self

        import torch
        from types import SimpleNamespace

        def _patch_transformers_compat(model):
            """Patch model cũ (pickle) để tương thích transformers mới."""
            for module in model.modules():
                if (hasattr(module, "self") and hasattr(module, "output")
                        and not hasattr(module, "is_cross_attention")):
                    module.is_cross_attention = False
                if hasattr(module, "attention_head_size") and hasattr(module, "query"):
                    if not hasattr(module, "config"):
                        module.config = SimpleNamespace(_attn_implementation="eager")
                    if not hasattr(module, "scaling"):
                        module.scaling = module.attention_head_size ** -0.5
                    if not hasattr(module, "is_causal"):
                        module.is_causal = False
                    if not hasattr(module, "layer_idx"):
                        module.layer_idx = None

        print("⏳ [1/3] Đang tải VnCoreNLP…")
        import py_vncorenlp
        self._vncorenlp = py_vncorenlp.VnCoreNLP(
            annotators=["wseg", "pos"], save_dir=VNCORENLP_DIR,
        )
        print("✅ VnCoreNLP sẵn sàng!\n")

        print(f"⏳ [2/3] Đang tải NER model…")
        ner_model = torch.load(NER_PT, map_location="cpu", weights_only=False)
        ner_model.eval()
        _patch_transformers_compat(ner_model)
        print("✅ NER model sẵn sàng!\n")

        print(f"⏳ [3/5] Đang tải Symmetric Embedding…")
        from sentence_transformers import SentenceTransformer
        sym_model = SentenceTransformer(SYMMETRIC_EMB_DIR)
        self._summarizer = TextRankEmbedding(sym_model)
        print("✅ Symmetric Embedding sẵn sàng!\n")

        print(f"⏳ [4/5] Đang tải Asymmetric Embedding…")
        # Asymmetric model path truyền từ ngoài vào
        self._kw_pipeline = KeywordExtractorPipeline(
            ner_model=ner_model,
            vncorenlp_instance=self._vncorenlp,
            asymm_model_path=ASYMM_EMB_DIR,
        )
        print("✅ Tất cả pipeline sẵn sàng!\n")

        self._is_loaded = True
        return self

    def run(
        self,
        text: str,
        title: Optional[str] = None,
        max_sentences: Optional[int] = None,
    ) -> CombinedResult:
        """Tóm tắt + trích xuất từ khóa cho 1 tài liệu."""
        if not self._is_loaded:
            self.load()

        summary_sentences = self._summarizer.summarize(text, max_sentences=max_sentences)
        summary_text = " ".join(summary_sentences)
        print(f"  📝 Tóm tắt : {len(summary_sentences)} câu")

        keywords = list(self._kw_pipeline(
            inputs={"text": summary_text, "title": title},
            min_freq=self.min_freq,
            ngram_n=self.ngram_n,
            top_n=self.top_n,
            use_mmr=self.use_mmr,
            use_kmeans=self.use_kmeans,
            diversity=self.diversity,
        ))
        print(f"  🔑 Từ khóa : {len(keywords)}")

        return CombinedResult(
            original_text=text,
            summary_sentences=summary_sentences,
            summary_text=summary_text,
            keywords=keywords,
            title=title,
        )

    def run_batch(
        self,
        texts: List[str],
        titles: Optional[List[Optional[str]]] = None,
        max_sentences: Optional[int] = None,
    ) -> List[CombinedResult]:
        """Tóm tắt + trích xuất từ khóa cho nhiều tài liệu."""
        if titles is None:
            titles = [None] * len(texts)
        if not self._is_loaded:
            self.load()

        results = []
        for i, (text, title) in enumerate(zip(texts, titles)):
            print(f"\n{'─' * 50}")
            print(f"  [{i+1}/{len(texts)}] {title or '(không tiêu đề)'}")
            print(f"{'─' * 50}")
            result = self.run(text=text, title=title, max_sentences=max_sentences)
            print(f"  📝 Tóm tắt : {len(result.summary_sentences)} câu")
            print(f"  🔑 Từ khóa : {len(result.keywords)}")
            results.append(result)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Demo
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
    pipeline = CombinedPipeline(top_n=10, ngram_n=(1, 3), min_freq=1, use_mmr=False)
    result = pipeline.run(text=sample_text, title="Trí tuệ nhân tạo")
    print(result)
