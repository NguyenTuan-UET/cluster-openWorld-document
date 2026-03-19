"""
Gradio App - Combined Pipeline
Tóm tắt (TextRank) + Từ khóa (KeyBERT-Vi) + Phân nhóm chủ đề (Gemini)
========================================================================
Chạy:
    cd combined_pipeline
    source venv/bin/activate
    python app.py
"""

import gradio as gr
from combined_pipeline import CombinedPipeline, CombinedResult
from gemini_service import TopicLabel
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo pipeline (load model 1 lần khi server khởi động)
# ──────────────────────────────────────────────────────────────────────────────

if gr.NO_RELOAD:
    print("=" * 60)
    print("  Combined Pipeline - Đang khởi động server…")
    print("=" * 60)
    _pipeline = CombinedPipeline(enable_clustering=True)
    _pipeline.load()
    print("=" * 60)
    print("  ✅ Server sẵn sàng!")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Xử lý đơn tài liệu
# ──────────────────────────────────────────────────────────────────────────────

def process_single(
    title: str,
    text: str,
    max_sentences: int,
    top_n: int,
    ngram_low: int,
    ngram_high: int,
    min_freq: int,
    diversify: bool,
) -> tuple:
    if not text.strip():
        return "⚠️ Vui lòng nhập văn bản.", "", ""

    _pipeline.top_n            = int(top_n)
    _pipeline.ngram_n          = (int(ngram_low), int(ngram_high))
    _pipeline.min_freq         = int(min_freq)
    _pipeline.diversify_result = diversify

    title_val    = title.strip() if title.strip() else None
    max_sent_val = int(max_sentences) if max_sentences and int(max_sentences) > 0 else None

    result = _pipeline.run(text=text, title=title_val, max_sentences=max_sent_val)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_input = len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])
    n_summ  = len(result.summary_sentences)
    n_kw    = len(result.keywords)
    ratio   = round(n_summ / n_input * 100) if n_input > 0 else 0
    label_names = []
    for lid in result.label_ids:
        for lb in _pipeline.labels:
            if lb.id == lid:
                label_names.append(lb.name)
    label_info = ", ".join(label_names) if label_names else "—"
    stats_md = (
        f"📄 **{n_input}** câu đầu vào &nbsp;→&nbsp; "
        f"📝 **{n_summ}** câu tóm tắt ({ratio}%) &nbsp;|&nbsp; "
        f"🔑 **{n_kw}** từ khóa &nbsp;|&nbsp; "
        f"��️ Nhãn: **{label_info}**"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_lines = [f"[{i}]  {sent}" for i, sent in enumerate(result.summary_sentences, 1)]
    summary_output = "\n\n".join(summary_lines) or "(Không có câu nào được chọn)"

    # ── Keywords ──────────────────────────────────────────────────────────────
    kw_lines = []
    for i, (kw, score) in enumerate(result.keywords, 1):
        bar_len = int(score * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        kw_lines.append(f"{i:>2}. {kw:<30}  {bar}  {score:.4f}")
    keywords_output = "\n".join(kw_lines) or "(Không trích xuất được từ khóa)"

    return summary_output, keywords_output, stats_md


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Phân nhóm nhiều tài liệu (Batch Clustering)
# ──────────────────────────────────────────────────────────────────────────────

def process_batch(
    docs_text: str,
    max_sentences: int,
    top_n: int,
    ngram_low: int,
    ngram_high: int,
    min_freq: int,
    diversify: bool,
) -> tuple:
    """
    Mỗi tài liệu cách nhau bởi 1 dòng bắt đầu bằng '==='.
    Dòng đầu tiên sau === là tiêu đề, phần còn lại là nội dung.
    """
    if not docs_text.strip():
        return "⚠️ Vui lòng nhập tài liệu.", "", ""

    # ── Parse documents ───────────────────────────────────────────────────────
    texts: List[str] = []
    titles: List[Optional[str]] = []

    blocks = docs_text.split("===")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        title = lines[0].strip() if lines else None
        body  = lines[1].strip() if len(lines) > 1 else lines[0].strip()
        if not body:
            continue
        titles.append(title)
        texts.append(body)

    if not texts:
        return "⚠️ Không tìm thấy tài liệu. Dùng '===' để ngăn cách.", "", ""

    # ── Update pipeline params ────────────────────────────────────────────────
    _pipeline.top_n            = int(top_n)
    _pipeline.ngram_n          = (int(ngram_low), int(ngram_high))
    _pipeline.min_freq         = int(min_freq)
    _pipeline.diversify_result = diversify
    _pipeline.reset_labels()

    max_sent_val = int(max_sentences) if max_sentences and int(max_sentences) > 0 else None

    results, labels = _pipeline.run_batch(
        texts=texts, titles=titles, max_sentences=max_sent_val
    )

    # ── Format: Kết quả từng tài liệu ────────────────────────────────────────
    detail_lines = []
    for i, r in enumerate(results, 1):
        doc_title = r.title or "(không tiêu đề)"
        label_names = []
        for lid in r.label_ids:
            for lb in labels:
                if lb.id == lid:
                    label_names.append(lb.name)
        label_str = ", ".join(label_names) if label_names else "—"

        kw_str = ", ".join(kw for kw, _ in r.keywords[:8])
        detail_lines.append(
            f"{'─' * 50}\n"
            f"📄 Tài liệu {i}: {doc_title}\n"
            f"   📝 Tóm tắt: {r.summary_text[:150]}…\n"
            f"   🔑 Từ khóa (dùng để phân nhóm): {kw_str}\n"
            f"   🏷️ Nhãn: {label_str}"
        )
    docs_output = "\n\n".join(detail_lines)

    # ── Format: Cluster overview ──────────────────────────────────────────────
    if labels:
        cluster_lines = []
        for lb in sorted(labels, key=lambda x: x.document_count, reverse=True):
            doc_indices = [
                str(i + 1)
                for i, r in enumerate(results)
                if lb.id in r.label_ids
            ]
            docs_in = ", ".join(doc_indices) if doc_indices else "—"

            # Hiển thị từ khóa đại diện cho cluster (gom từ khóa các tài liệu thuộc nhóm)
            cluster_keywords = set()
            for i, r in enumerate(results):
                if lb.id in r.label_ids:
                    cluster_keywords.update(kw for kw, _ in r.keywords[:5])
            kw_display = ", ".join(list(cluster_keywords)[:10]) if cluster_keywords else "—"

            cluster_lines.append(
                f"🏷️ **{lb.name}** (ID: {lb.id})\n"
                f"   Mô tả: {lb.description}\n"
                f"   📄 Tài liệu: [{docs_in}] ({lb.document_count} docs)\n"
                f"   🔑 Từ khóa đại diện: {kw_display}"
            )
        cluster_output = "\n\n".join(cluster_lines)
    else:
        cluster_output = "(Không có nhãn — Gemini không khả dụng hoặc chưa bật clustering)"

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats_md = (
        f"📊 **{len(texts)}** tài liệu &nbsp;→&nbsp; "
        f"🏷️ **{len(labels)}** nhãn chủ đề"
    )

    return docs_output, cluster_output, stats_md


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
        **Luồng:** Văn bản → ✂️ TextRank → 🔑 KeyBERT-Vi (PhoBERT + NER) → 🏷️ Gemini (clustering)
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
                sau đó dùng **Gemini AI** phân nhóm chúng theo chủ đề.

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
                    t2_btn = gr.Button(
                        "▶  Phân nhóm", variant="primary", elem_id="batch-btn",
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

            t2_btn.click(
                fn=_run_batch,
                inputs=[
                    t2_docs, t2_max_sent, t2_top_n,
                    t2_ng_lo, t2_ng_hi, t2_min_freq, t2_diversify,
                ],
                outputs=[t2_details, t2_clusters, t2_stats],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        <small>
        **Gợi ý:** Top N = 8–12 · Ngram (1, 3) · Min freq = 1 (văn bản ngắn), 2–3 (văn bản dài) · Diversify = đa dạng chủ đề<br>
        **Phân nhóm:** Cần có `GEMINI_API_KEY` trong file `.env` để bật tính năng clustering.
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
