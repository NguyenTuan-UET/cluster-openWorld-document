"""
Gradio App - Combined Pipeline
Tóm tắt (TextRank) + Từ khóa (KeyBERT-Vi) + Phân nhóm chủ đề (Kilo AI / MiniMax)
========================================================================
Chạy:
    source venv/bin/activate
    python base_line/app.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Ensure project root is in sys.path so 'backend' module is importable
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import gradio as gr
from backend.cluster.combined_pipeline import CombinedPipeline, CombinedResult
from backend.cluster.llm_service import LLMService, AnalyzedDocument
import uuid
import time
from typing import List, Dict
import re
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
from pyvi.ViTokenizer import tokenize as pyvi_tokenize
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
# In-memory state (giống backend/main.py)
# ──────────────────────────────────────────────────────────────────────────────
_all_documents: List[dict] = []
_clusters: List[dict] = []

# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo pipeline (load model 1 lần khi server khởi động)
# ──────────────────────────────────────────────────────────────────────────────

if gr.NO_RELOAD:
    print("=" * 60)
    print("  NLP Pipeline — Đang khởi động server…")
    print("=" * 60)

    try:
        _pipeline = CombinedPipeline(use_mmr=True, use_kmeans=False, diversity=0.5)
        _pipeline.load()
        print("✅ CombinedPipeline loaded!")
    except Exception as e:
        print(f"⚠️ CombinedPipeline error: {e}")
        _pipeline = None

    try:
        _sym_model = SentenceTransformer(str(PROJECT_ROOT / "pretrained-models/symmetric_emb"))
        print("✅ symmetric_emb sẵn sàng!")
    except Exception as e:
        print(f"⚠️ symmetric_emb error: {e}")
        _sym_model = None

    try:
        _llm_service = LLMService()
        _llm_service._ensure_client()
        print("✅ LLM Service loaded!")
    except Exception as e:
        print(f"⚠️ LLM Service error: {e}")
        _llm_service = None

    print("=" * 60)
    print("  ✅ Server sẵn sàng!")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Xử lý đơn tài liệu
# ──────────────────────────────────────────────────────────────────────────────

def process_single(title, text, max_sentences, top_n, ngram_low, ngram_high, min_freq, diversify):
    _pipeline.top_n            = int(top_n)
    _pipeline.ngram_n          = (int(ngram_low), int(ngram_high))
    _pipeline.min_freq         = int(min_freq)
    _pipeline.diversify_result = diversify

    result = _pipeline.run(text=text, title=title.strip() or None, max_sentences=int(max_sentences) or None)

    n_input  = len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])
    n_summ   = len(result.summary_sentences)
    stats_md = (
        f"📄 **{n_input}** câu → 📝 **{n_summ}** câu tóm tắt ({round(n_summ / n_input * 100)}%) "
        f"| 🔑 **{len(result.keywords)}** từ khóa"
    )

    summary_output  = "\n\n".join(f"[{i}]  {s}" for i, s in enumerate(result.summary_sentences, 1))
    keywords_output = "\n".join(
        f"{i:>2}. {kw:<30}  {'█' * int(sc * 20)}{'░' * (20 - int(sc * 20))}  {sc:.4f}"
        for i, (kw, sc) in enumerate(result.keywords, 1)
    )
    return summary_output, keywords_output, stats_md


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Phân nhóm nhiều tài liệu — giống hệt backend/main.py
# ──────────────────────────────────────────────────────────────────────────────

def reset_batch():
    """Xóa toàn bộ state clusters + documents."""
    global _all_documents, _clusters
    _all_documents = []
    _clusters = []
    return "", "_State đã được reset._", "🗑️ Đã xóa toàn bộ clusters và documents."


def process_batch(docs_text, max_sentences, top_n, ngram_low, ngram_high, min_freq, diversify):
    global _all_documents, _clusters

    if _pipeline is None:
        return "", "⚠️ Pipeline chưa load.", ""
    if _llm_service is None:
        return "", "⚠️ LLM Service chưa load. Kiểm tra KILO_API_KEY.", ""

    blocks = [b.strip() for b in docs_text.split("===") if b.strip()]
    titles = [b.partition("\n")[0].strip() for b in blocks]
    texts  = [b.partition("\n")[2].strip() for b in blocks]
    n = len(texts)

    _pipeline.top_n   = int(top_n)
    _pipeline.ngram_n = (int(ngram_low), int(ngram_high))
    _pipeline.min_freq = int(min_freq)
    _pipeline.use_mmr  = diversify

    print(f"\n{'=' * 60}")
    print(f"  Batch processing — {n} tài liệu")
    print(f"{'=' * 60}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: TextRank + KeyBERT cho từng doc (không dùng LLM)
    # ═══════════════════════════════════════════════════════════════
    analyzed_docs: List[AnalyzedDocument] = []
    detail_lines: List[str] = []

    for i, (text, title) in enumerate(zip(texts, titles)):
        file_name = title or f"doc-{i}"
        print(f"\n{'─' * 50}")
        print(f"  [{i+1}/{n}] {file_name}")
        print(f"{'─' * 50}")

        pipeline_result = _pipeline.run(
            text=text,
            title=title or None,
            max_sentences=int(max_sentences) or None,
        )

        doc = AnalyzedDocument(
            id=f"doc-{uuid.uuid4().hex[:8]}",
            file_name=f"[{i+1}]",
            keyphrases=[kw for kw, _ in pipeline_result.keywords],
            summary=pipeline_result.summary_text,
        )
        analyzed_docs.append(doc)

        kw_lines = "\n".join(f"      {j+1:2d}. {kw}" for j, kw in enumerate(doc.keyphrases))
        print(kw_lines)

        detail_lines.append(
            f"{'─' * 50}\n"
            f"📄 [{i+1}/{n}] {file_name}\n"
            f"   📝 Tóm tắt: {pipeline_result.summary_text[:150]}…\n"
            f"   🔑 Từ khóa: {', '.join(doc.keyphrases[:8])}"
        )

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: LLM — Multi-label clustering
    # ═══════════════════════════════════════════════════════════════
    print(f"\n🤖 Phase 2/2 — LLM phân nhóm multi-label…")

    result = _llm_service.run_batch_clustering(analyzed_docs, _clusters)
    _all_documents.extend(result["documents"])
    # _clusters đã được cập nhật in-place bên trong run_batch_clustering

    print(f"\n{'=' * 60}")
    print(f"  ✅ Done! {len(_clusters)} clusters, {len(_all_documents)} docs")
    print(f"{'=' * 60}\n")

    # ── Format clusters thành Markdown ──
    cluster_parts = []
    for c in _clusters:
        doc_refs = " · ".join(d["fileName"] for d in c["documents"])
        cluster_parts.append(f"### 🏷️ {c['label']}\n{doc_refs}")
    clusters_md = "\n\n".join(cluster_parts) if cluster_parts else "_Không có nhóm nào được tạo._"

    return (
        "\n\n".join(detail_lines),
        clusters_md,
        f"📊 **{n}** tài liệu | **{len(_clusters)}** nhóm chủ đề | **{len(_all_documents)}** docs tổng cộng",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: So sánh phương pháp tóm tắt (baseline methods)
# ──────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def _pagerank(sim: np.ndarray, damping: float = 0.85, iters: int = 100) -> np.ndarray:
    n = len(sim)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    scores = np.ones(n) / n
    for _ in range(iters):
        scores = (1 - damping) / n + damping * (sim / row_sums).T @ scores
    return scores

def _top_k(sents: List[str], scores: np.ndarray, k: int) -> List[str]:
    ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[:k]
    return [sents[i] for i in sorted(ranked)]

def lead_k(text: str, k: int) -> List[str]:
    return _split_sentences(text)[:k]

def textrank_word_overlap(text: str, k: int) -> List[str]:
    sents = _split_sentences(text)
    if len(sents) <= k:
        return sents
    words = [set(s.lower().split()) for s in sents]
    n = len(sents)
    sim = np.array([
        [len(words[i] & words[j]) / (np.log(len(words[i]) + 1) + np.log(len(words[j]) + 1))
         if i != j and words[i] and words[j] else 0
         for j in range(n)] for i in range(n)
    ])
    return _top_k(sents, _pagerank(sim, damping=0.85), k)

def textrank_tfidf(text: str, k: int) -> List[str]:
    sents = _split_sentences(text)
    if len(sents) <= k:
        return sents
    sim = _cos_sim(TfidfVectorizer().fit_transform(sents)).astype(float)
    np.fill_diagonal(sim, 0)
    return _top_k(sents, _pagerank(sim), k)

def _sym_encode(sentences: List[str]) -> np.ndarray:
    return _sym_model.encode(sentences, normalize_embeddings=True)

def _auto_k(n: int) -> int:
    """Công thức tự động chọn số câu — giống TextRankFacade.summarize()."""
    if n <= 5:
        return min(n, 3)
    return max(5, math.ceil(n * 0.4))

def textrank_symmetric(text: str) -> List[str]:
    """
    TextRank với Symmetric Embedding làm trọng số cạnh đồ thị.

    Luồng:
      1. Tách câu (regex)
      2. Tách từ pyvi → encode bằng symmetric_emb (normalize L2)
      3. Xây đồ thị có hướng: đỉnh = câu, cạnh (i→j) = cosine sim(i, j)
         (ma trận kề = emb @ emb.T, bỏ đường chéo)
      4. PageRank (damping=0.85) trên đồ thị → điểm quan trọng từng câu
      5. Chọn top-k câu (k tự động theo công thức TextRankFacade),
         trả về theo thứ tự gốc trong văn bản
    """
    sents = _split_sentences(text)
    n     = len(sents)
    k     = _auto_k(n)
    if n <= k:
        return sents

    # Bước 2: tách từ pyvi → encode bằng AutoModel + mean pooling
    emb = _sym_encode([pyvi_tokenize(s) for s in sents])

    # Bước 3: ma trận kề — cạnh = cosine similarity (đã normalize → dot product)
    adj = (emb @ emb.T).astype(float)
    np.fill_diagonal(adj, 0)   # không có self-loop

    # Bước 4: PageRank
    scores = _pagerank(adj)

    # Bước 5: chọn top-k, giữ thứ tự gốc
    return _top_k(sents, scores, k)

def process_compare(title, text, k):
    k          = int(k) or 3
    fmt        = lambda sents: "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sents, 1))
    sym_result = textrank_symmetric(text)
    return (
        fmt(sym_result),
        fmt(lead_k(text, k)),
        fmt(textrank_word_overlap(text, k)),
        fmt(textrank_tfidf(text, k)),
        f"📊 Symmetric tự chọn: **{len(sym_result)}** câu &nbsp;|&nbsp; Baseline k = **{k}**",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dữ liệu mẫu
# ──────────────────────────────────────────────────────────────────────────────

SINGLE_EXAMPLES = [
    [
        "Thành Cổ Loa - Lịch sử và hiện tại",
        "Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương Mỵ Châu – Trọng Thủy. Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn khám phá được những giá trị khảo cổ to lớn của Cổ Loa.\nKhu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước. Cổ Loa có hàng loạt di chỉ khảo cổ học đã được phát hiện, phản ánh quá trình phát triển liên tục của dân tộc ta từ sơ khai qua các thời kỳ đồ đồng, đồ đá và đồ sắt mà đỉnh cao là văn hóa Đông Sơn, vẫn được coi là nền văn minh sông Hồng thời kỳ tiền sử của dân tộc Việt Nam.\nCổ Loa từng là kinh đô của nhà nước Âu Lạc thời kỳ An Dương Vương (thế kỷ III TCN) và của nước Đại Việt thời Ngô Quyền (thế kỷ X) mà thành Cổ Loa là một di tích minh chứng còn lại cho đến ngày nay.",
        3, 8, 1, 3, 1, False,
    ],
    [
        "Trí tuệ nhân tạo",
        "Trí tuệ nhân tạo (AI) đang thay đổi mọi khía cạnh của cuộc sống hiện đại. Từ y tế, giáo dục đến giao thông vận tải, AI mang lại những cải tiến vượt bậc. Các thuật toán học máy giúp chẩn đoán bệnh chính xác hơn bác sĩ trong nhiều trường hợp. Xe tự lái ứng dụng deep learning để nhận diện đường đi và tránh va chạm. Chatbot được trang bị xử lý ngôn ngữ tự nhiên hỗ trợ khách hàng 24/7. Tuy nhiên, AI cũng đặt ra nhiều thách thức về đạo đức và quyền riêng tư. Các chuyên gia khuyến nghị cần có khung pháp lý rõ ràng để quản lý AI. Việt Nam đang đẩy mạnh ứng dụng AI vào các lĩnh vực trọng điểm quốc gia.",
        0, 10, 1, 3, 1, True,
    ],
]

BATCH_EXAMPLE = """=== Trí tuệ nhân tạo trong y tế
Trí tuệ nhân tạo đang cách mạng hóa ngành y tế. Các thuật toán học máy có thể phân tích hình ảnh y khoa và phát hiện ung thư sớm hơn bác sĩ. AI hỗ trợ phát triển thuốc mới nhanh hơn bằng cách mô phỏng hàng triệu phân tử. Robot phẫu thuật được điều khiển bởi AI giúp ca mổ chính xác hơn.

=== Biến đổi khí hậu tại Việt Nam
Việt Nam là một trong những quốc gia chịu ảnh hưởng nặng nề nhất của biến đổi khí hậu. Mực nước biển dâng đe dọa vùng đồng bằng sông Cửu Long. Các hiện tượng thời tiết cực đoan như bão, lũ lụt ngày càng nghiêm trọng. Chính phủ đã cam kết đạt phát thải ròng bằng 0 vào năm 2050.

=== Ứng dụng blockchain trong tài chính
Công nghệ blockchain đang thay đổi ngành tài chính toàn cầu. Tiền mã hóa cho phép giao dịch xuyên biên giới nhanh chóng và chi phí thấp. Smart contract tự động hóa các hợp đồng tài chính. Các ngân hàng lớn đang nghiên cứu ứng dụng blockchain vào hệ thống thanh toán.

=== Machine Learning trong giáo dục
Học máy đang cá nhân hóa trải nghiệm học tập cho học sinh. Hệ thống gợi ý bài tập dựa trên trình độ của từng học sinh. Chatbot AI hỗ trợ giải đáp thắc mắc 24/7. Phân tích dữ liệu học tập giúp giáo viên phát hiện sớm học sinh gặp khó khăn."""


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
#run-btn { font-size: 1.1em; padding: 12px 0; }
#batch-btn { font-size: 1.1em; padding: 12px 0; }
#stats-row { background: #f0f4ff; border-radius: 8px; padding: 10px 16px; }
#batch-stats { background: #f0fff4; border-radius: 8px; padding: 10px 16px; }
#kw-box textarea { font-family: monospace; font-size: 0.88em; }
#summary-box textarea { font-size: 0.95em; line-height: 1.7; }
#docs-box textarea { font-family: monospace; font-size: 0.88em; line-height: 1.5; }
#cluster-box { min-height: 200px; }
"""

with gr.Blocks(title="Vietnamese NLP Pipeline") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🇻🇳 Vietnamese NLP Pipeline
        ### Tóm tắt văn bản · Trích xuất từ khóa · Phân nhóm chủ đề
        **Luồng:** Văn bản → ✂️ TextRank → 🔑 KeyBERT-Vi (PhoBERT + NER) → 🏷️ Kilo AI (clustering)
        """
    )

    with gr.Tabs():
        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: Đơn tài liệu
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("📄 Đơn tài liệu"):

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    t1_title = gr.Textbox(
                        label="📌 Tiêu đề (tùy chọn)",
                        placeholder="Ví dụ: Lịch sử Cổ Loa",
                        lines=1,
                    )
                    t1_text = gr.Textbox(
                        label="📄 Văn bản đầu vào",
                        placeholder="Dán văn bản tiếng Việt vào đây…",
                        lines=14,
                    )
                    t1_btn = gr.Button(
                        "▶  Chạy pipeline", variant="primary", elem_id="run-btn",
                    )

                with gr.Column(scale=1, min_width=220):
                    with gr.Accordion("✂️ Tóm tắt", open=True):
                        t1_max_sent = gr.Slider(
                            label="Số câu tóm tắt tối đa",
                            info="0 = tự động",
                            value=0, minimum=0, maximum=15, step=1,
                        )
                    with gr.Accordion("🔑 Keyword", open=True):
                        t1_top_n = gr.Slider(
                            label="Top N", value=10, minimum=1, maximum=30, step=1,
                        )
                        with gr.Row():
                            t1_ng_lo = gr.Number(label="Ngram min", value=1, minimum=1, maximum=5)
                            t1_ng_hi = gr.Number(label="Ngram max", value=3, minimum=1, maximum=5)
                        t1_min_freq = gr.Slider(
                            label="Min frequency", value=1, minimum=1, maximum=5, step=1,
                        )
                        t1_diversify = gr.Checkbox(
                            label="🎲 Diversify (K-means)", value=False,
                        )

            t1_stats = gr.Markdown("", elem_id="stats-row", visible=False)

            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=3):
                    t1_summary = gr.Textbox(
                        label="📝 Tóm tắt", lines=10, interactive=False,
                        elem_id="summary-box",
                    )
                with gr.Column(scale=2):
                    t1_keywords = gr.Textbox(
                        label="🔑 Từ khóa & điểm số", lines=10, interactive=False,
                        elem_id="kw-box",
                    )

            gr.Examples(
                examples=SINGLE_EXAMPLES,
                inputs=[
                    t1_title, t1_text, t1_max_sent, t1_top_n,
                    t1_ng_lo, t1_ng_hi, t1_min_freq, t1_diversify,
                ],
                label="💡 Văn bản mẫu",
            )

            def _run_single(*args):
                s, k, stats = process_single(*args)
                return s, k, gr.update(value=stats, visible=True)

            t1_btn.click(
                fn=_run_single,
                inputs=[
                    t1_title, t1_text, t1_max_sent, t1_top_n,
                    t1_ng_lo, t1_ng_hi, t1_min_freq, t1_diversify,
                ],
                outputs=[t1_summary, t1_keywords, t1_stats],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: Phân nhóm nhiều tài liệu (Batch Clustering)
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("📚 Phân nhóm chủ đề (Multi-doc)"):

            gr.Markdown(
                """
                ### 🏷️ Phân nhóm tài liệu theo chủ đề
                Nhập nhiều tài liệu, pipeline sẽ **tóm tắt + trích xuất từ khóa** cho từng tài liệu,
                sau đó dùng **Kilo AI (MiniMax)** phân nhóm chúng theo chủ đề.

                **Cách nhập:** Dùng `===` để ngăn cách tài liệu. Dòng sau `===` là tiêu đề.
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    t2_docs = gr.Textbox(
                        label="📚 Nhập nhiều tài liệu (ngăn cách bằng ===)",
                        placeholder="=== Tiêu đề 1\nNội dung…\n\n=== Tiêu đề 2\nNội dung…",
                        lines=18,
                        value=BATCH_EXAMPLE,
                    )
                    with gr.Row():
                        t2_btn = gr.Button(
                            "▶  Phân nhóm", variant="primary", elem_id="batch-btn",
                        )
                        t2_reset_btn = gr.Button(
                            "🗑️  Reset State", variant="secondary",
                        )

                with gr.Column(scale=1, min_width=220):
                    with gr.Accordion("⚙️ Tham số", open=True):
                        t2_max_sent = gr.Slider(
                            label="Số câu tóm tắt tối đa", info="0 = tự động",
                            value=3, minimum=0, maximum=15, step=1,
                        )
                        t2_top_n = gr.Slider(
                            label="Top N keywords", value=8, minimum=1, maximum=30, step=1,
                        )
                        with gr.Row():
                            t2_ng_lo = gr.Number(label="Ngram min", value=1, minimum=1, maximum=5)
                            t2_ng_hi = gr.Number(label="Ngram max", value=3, minimum=1, maximum=5)
                        t2_min_freq = gr.Slider(
                            label="Min frequency", value=1, minimum=1, maximum=5, step=1,
                        )
                        t2_diversify = gr.Checkbox(label="🎲 Diversify", value=False)

            t2_stats = gr.Markdown("", elem_id="batch-stats", visible=False)

            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1):
                    t2_clusters = gr.Markdown(
                        label="🏷️ Nhãn chủ đề (Clusters)",
                        value="",
                        elem_id="cluster-box",
                    )
                with gr.Column(scale=2):
                    t2_details = gr.Textbox(
                        label="📄 Chi tiết từng tài liệu",
                        lines=20,
                        interactive=False,
                        elem_id="docs-box",
                    )

            def _run_batch(*args):
                docs, clusters, stats = process_batch(*args)
                return docs, clusters, gr.update(value=stats, visible=True)

            def _reset_batch():
                docs, clusters, stats = reset_batch()
                return docs, clusters, gr.update(value=stats, visible=True)

            t2_btn.click(
                fn=_run_batch,
                inputs=[
                    t2_docs, t2_max_sent, t2_top_n,
                    t2_ng_lo, t2_ng_hi, t2_min_freq, t2_diversify,
                ],
                outputs=[t2_details, t2_clusters, t2_stats],
            )

            t2_reset_btn.click(
                fn=_reset_batch,
                inputs=[],
                outputs=[t2_details, t2_clusters, t2_stats],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: So sánh phương pháp tóm tắt
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("📊 So sánh tóm tắt"):

            gr.Markdown(
                """
                ### 📊 So sánh các phương pháp tóm tắt
                Chạy cùng văn bản qua 4 phương pháp và so sánh kết quả.
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    t3_title = gr.Textbox(label="📌 Tiêu đề (tùy chọn)", lines=1)
                    t3_text  = gr.Textbox(
                        label="📄 Văn bản đầu vào",
                        placeholder="Dán văn bản tiếng Việt vào đây…",
                        lines=14,
                    )
                    t3_btn = gr.Button("▶  So sánh", variant="primary")
                with gr.Column(scale=1, min_width=220):
                    t3_k = gr.Slider(
                        label="Số câu tóm tắt k (Lead-K / Word Overlap / TF-IDF)",
                        info="Symmetric tự động chọn k theo độ dài văn bản",
                        value=3, minimum=1, maximum=15, step=1,
                    )

            t3_stats = gr.Markdown("", elem_id="stats-row", visible=False)
            gr.Markdown("---")

            with gr.Row():
                t3_current = gr.Textbox(label="🔬 TextRank + Symmetric Embedding (pyvi)", lines=10, interactive=False)
                t3_lead    = gr.Textbox(label="📌 Lead-K", lines=10, interactive=False)
            with gr.Row():
                t3_wo    = gr.Textbox(label="📐 TextRank gốc (Word Overlap, d=0.85)", lines=10, interactive=False)
                t3_tfidf = gr.Textbox(label="📈 TextRank + TF-IDF (Cosine Similarity)", lines=10, interactive=False)

            def _run_compare(*args):
                cur, lead, wo, tfidf, stats = process_compare(*args)
                return cur, lead, wo, tfidf, gr.update(value=stats, visible=True)

            t3_btn.click(
                fn=_run_compare,
                inputs=[t3_title, t3_text, t3_k],
                outputs=[t3_current, t3_lead, t3_wo, t3_tfidf, t3_stats],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        <small>
        **Gợi ý:** Top N = 8–12 · Ngram (1, 3) · Min freq = 1 (văn bản ngắn), 2–3 (văn bản dài) · Diversify = đa dạng chủ đề<br>
        **Phân nhóm:** Cần có `KILO_API_KEY` trong file `.env` để bật tính năng clustering.
        </small>
        """
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CSS,
    )
