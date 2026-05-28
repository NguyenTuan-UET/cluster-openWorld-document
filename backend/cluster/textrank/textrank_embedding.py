"""
TextRank dựa trên Sentence Embedding (symmetric_emb + pyvi).

Thay thế word-overlap graph của TextRankFacade bằng sentence-level similarity graph:
  - Đỉnh  : mỗi câu trong văn bản
  - Cạnh  : cosine similarity giữa sentence embeddings
  - Trọng số encoding: pyvi (tách từ) → SentenceTransformer (dangvantuan/vietnamese-embedding)
  - PageRank: damping=0.85, 100 vòng lặp
"""

import re
import math
import numpy as np
from typing import List, Optional
from pyvi.ViTokenizer import tokenize as pyvi_tokenize
from sentence_transformers import SentenceTransformer


class TextRankEmbedding:

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def summarize(self, raw_text: str, max_sentences: Optional[int] = None) -> List[str]:
        sents = self._split_sentences(raw_text)
        n = len(sents)
        k = self._auto_k(n, max_sentences)
        if n <= k:
            return sents

        emb = self._model.encode(
            [pyvi_tokenize(s) for s in sents],
            normalize_embeddings=True,
        )

        adj = (emb @ emb.T).astype(float)       #tích vô hướng
        np.fill_diagonal(adj, 0)            #đường chéo chính = 0 khi tính tích vô hướng

        scores = self._pagerank(adj)

        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        return [sents[i] for i in sorted(ranked)]

    # tách văn bản tại chỗ dấu (.!? + khoảng trắng)
    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    # tự động số câu muốn lấy: 40% (trên 5 câu) - min 3 câu (dưới 5 câu) - lấy hết (dưới 3 câu)
    def _auto_k(self, n: int, max_sentences: Optional[int]) -> int:
        if max_sentences is not None:
            return max(1, min(int(max_sentences), n))
        if n <= 5:
            return min(n, 3)
        return max(5, math.ceil(n * 0.4))

    def _pagerank(self, sim: np.ndarray, damping: float = 0.85, iters: int = 100) -> np.ndarray:
        n = len(sim)
        row_sums = sim.sum(axis=1, keepdims=True) # ma trận cột với mỗi hàng = tổng similarity của câu - mẫu PageRank
        row_sums[row_sums == 0] = 1               # tránh chia cho 0
        scores = np.ones(n) / n                   # mảng mỗi ptu 1/n - tương đương nhau
        for _ in range(iters):
            scores = (1 - damping) / n + damping * (sim / row_sums).T @ scores
        return scores


if __name__ == "__main__":
    import os

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _SYMMETRIC_EMB_DIR = os.path.join(_PROJECT_ROOT, "pretrained-models", "symmetric_emb")

    print("⏳ Loading symmetric embedding model…")
    _model = SentenceTransformer(_SYMMETRIC_EMB_DIR)
    print("✅ Model sẵn sàng!\n")

    summarizer = TextRankEmbedding(_model)

    sample_text = """
    Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương Mỵ Châu – Trọng Thủy.
    Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn khám phá được những giá trị khảo cổ to lớn của Cổ Loa.
    Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước.
    Cổ Loa có hàng loạt di chỉ khảo cổ học đã được phát hiện, phản ánh quá trình phát triển liên tục của dân tộc ta từ sơ khai qua các thời kỳ đồ đồng, đồ đá và đồ sắt mà đỉnh cao là văn hóa Đông Sơn.
    Văn hóa Đông Sơn vẫn được coi là nền văn minh sông Hồng thời kỳ tiền sử của dân tộc Việt Nam.
    Cổ Loa từng là kinh đô của nhà nước Âu Lạc thời kỳ An Dương Vương (thế kỷ III TCN) và của nước Đại Việt thời Ngô Quyền (thế kỷ X).
    Thành Cổ Loa là một di tích minh chứng còn lại cho đến ngày nay.
    """

    print(f"Tổng số câu: {len(summarizer._split_sentences(sample_text))}\n")

    # - Tóm tắt tự động (auto_k) -
    print("======Tóm tắt (auto_k)======")
    summary = summarizer.summarize(sample_text)
    for i, sent in enumerate(summary, 1):
        print(f"  [{i}] {sent}")

    # - Tóm tắt với max_sentences cố định -
    print()
    print("======Tóm tắt (max_sentences=3)======")
    summary_3 = summarizer.summarize(sample_text, max_sentences=3)
    for i, sent in enumerate(summary_3, 1):
        print(f"  [{i}] {sent}")
