"""
TextRank dựa trên Sentence Embedding (symmetric_emb + pyvi).

Thay thế word-overlap graph của TextRankFacade bằng sentence-level similarity graph:
  - Đỉnh  : mỗi câu trong văn bản
  - Cạnh  : cosine similarity giữa sentence embeddings
  - Trọng số encoding: pyvi (tách từ) → SentenceTransformer (dangvantuan/vietnamese-embedding)
  - PageRank: damping=0.85, 100 vòng lặp

Interface giống TextRankFacade.summarize() để dùng thay thế trực tiếp.
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
        """
        Tóm tắt văn bản bằng TextRank + Symmetric Embedding.

        Luồng:
          1. Tách câu bằng regex (giống Parser._get_sentences)
          2. Tách từ pyvi → encode bằng symmetric_emb (normalize L2)
          3. Xây ma trận kề: adj[i][j] = cosine_sim(câu_i, câu_j), bỏ đường chéo
          4. PageRank (damping=0.85, 100 vòng) → điểm quan trọng từng câu
          5. Chọn top-k câu, trả về theo thứ tự gốc

        Args:
            raw_text     : văn bản đầu vào
            max_sentences: số câu muốn lấy; None → tự động theo tỉ lệ
        """
        sents = self._split_sentences(raw_text)
        n = len(sents)
        k = self._auto_k(n, max_sentences)
        if n <= k:
            return sents

        emb = self._model.encode(
            [pyvi_tokenize(s) for s in sents],
            normalize_embeddings=True,
        )

        adj = (emb @ emb.T).astype(float)
        np.fill_diagonal(adj, 0)

        scores = self._pagerank(adj)

        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        return [sents[i] for i in sorted(ranked)]

    # ──────────────────────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    def _auto_k(self, n: int, max_sentences: Optional[int]) -> int:
        """Giống logic trong TextRankFacade.summarize()."""
        if max_sentences is not None:
            return max(1, min(int(max_sentences), n))
        if n <= 5:
            return min(n, 3)
        return max(5, math.ceil(n * 0.4))

    def _pagerank(self, sim: np.ndarray, damping: float = 0.85, iters: int = 100) -> np.ndarray:
        n = len(sim)
        row_sums = sim.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        scores = np.ones(n) / n
        for _ in range(iters):
            scores = (1 - damping) / n + damping * (sim / row_sums).T @ scores
        return scores
