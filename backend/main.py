"""
NLP Pipeline API — FastAPI backend
===================================
Chạy:
    cd /home/nqtuan/workSpace/Devtamin/Thesis/Github/cluster-openWorld-document
    source venv/bin/activate
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
import uuid

# ── In-memory state (module-level) ───────────────────────────────────────────
_all_documents: List[dict] = []
_clusters: List[dict] = []
_pipeline_loaded = False
_gemini_loaded = False
_pipeline = None   # type: ignore
_gemini = None     # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_loaded, _gemini_loaded, _pipeline, _gemini
    print("=" * 60)
    print("  NLP Pipeline API — Đang khởi động…")
    print("=" * 60)

    try:
        from backend.cluster.combined_pipeline import CombinedPipeline
        _pipeline = CombinedPipeline(
            use_mmr=True,
            use_kmeans=False,
            diversity=0.5,
        )
        _pipeline.load()
        _pipeline_loaded = True
        print("✅ CombinedPipeline loaded!")
    except Exception as e:
        print(f"⚠️ CombinedPipeline error: {e}")

    try:
        from backend.cluster.llm_service import LLMService
        _gemini = LLMService()
        _gemini._ensure_client()
        _gemini_loaded = True
        print("✅ LLM Service loaded!")
    except Exception as e:
        print(f"⚠️ LLM Service error: {e}")

    print("=" * 60)
    print("  ✅ Server sẵn sàng!")
    print("=" * 60)
    yield
    print("Server shutting down...")


app = FastAPI(title="NLP Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    texts: List[str]
    file_names: Optional[List[str]] = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "NLP Pipeline API",
        "pipeline": _pipeline_loaded,
        "gemini": _gemini_loaded,
    }


@app.get("/state")
def get_state():
    return {"clusters": _clusters, "documents": _all_documents}


@app.post("/reset")
def reset_state():
    global _all_documents, _clusters
    _all_documents = []
    _clusters = []
    return {"status": "ok", "message": "Reset thành công"}


@app.post("/process-and-cluster")
def process_and_cluster(req: ExtractRequest):
    """
    Batch clustering với multi-label:

      Phase 1 (local): TextRank + KeyBERT cho TẤT CẢ tài liệu → keyphrases + summary
      Phase 2 (LLM):
        a. Gán mỗi doc vào NHIỀU clusters hiện có (multi-label)
        b. Docs không khớp cluster nào → tạo clusters mới
        c. Cập nhật label space và trả về kết quả
    """
    global _all_documents, _clusters, _pipeline, _gemini

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is required")
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline chưa load.")
    if _gemini is None:
        raise HTTPException(status_code=500, detail="LLM Service chưa load. Kiểm tra KILO_API_KEY.")

    from backend.cluster.llm_service import AnalyzedDocument, DocumentCluster

    n = len(req.texts)
    print(f"\n{'=' * 60}")
    print(f"  Batch processing — {n} tài liệu")
    print(f"{'=' * 60}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: TextRank + KeyBERT cho TẤT CẢ (local, không LLM)
    # ═══════════════════════════════════════════════════════════════════════
    analyzed_docs: List[AnalyzedDocument] = []

    for i, text in enumerate(req.texts):
        file_name = req.file_names[i] if req.file_names and i < len(req.file_names) else f"doc-{i}"
        title_for_pipeline = os.path.splitext(file_name)[0] or None

        print(f"\n{'─' * 50}")
        print(f"  [{i+1}/{n}] {file_name}")
        print(f"{'─' * 50}")

        pipeline_result = _pipeline.run(text=text, title=title_for_pipeline)

        doc = AnalyzedDocument(
            id=f"doc-{uuid.uuid4().hex[:8]}",
            file_name=file_name,
            keyphrases=[kw for kw, _ in pipeline_result.keywords],
            summary=pipeline_result.summary_text,
        )
        analyzed_docs.append(doc)

        kw_lines = "\n".join(f"      {j+1:2d}. {kw}" for j, kw in enumerate(doc.keyphrases))
        print(kw_lines)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: LLM — Multi-label clustering
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n🤖 Phase 2/2 — LLM phân nhóm multi-label…")

    # doc_id → list of labels (có thể nhiều label)
    doc_to_labels: Dict[str, List[str]] = {d.id: [] for d in analyzed_docs}

    # Build existing clusters from state
    existing: List[DocumentCluster] = [
        DocumentCluster(
            label=c["label"],
            documents=[
                AnalyzedDocument(
                    id=d["id"], file_name=d["fileName"],
                    keyphrases=d["keyphrases"], summary=d["summary"],
                )
                for d in c.get("documents", [])
            ],
        )
        for c in _clusters
    ]

    unassigned_docs = list(analyzed_docs)

    # ── 2a. Gán vào clusters hiện có (multi-label) ───────────────────────
    renames: Dict[str, str] = {}
    if existing:
        try:
            assign_result = _gemini.assign_to_existing_clusters_multilabel(analyzed_docs, existing)
            assignments   = assign_result.get("assignments", {})   # {doc_id: [labels]}
            renames       = assign_result.get("renames", {})
            unassigned_docs = assign_result["unassigned"]

            for doc_id, labels in assignments.items():
                if doc_id in doc_to_labels:
                    doc_to_labels[doc_id].extend(labels)

            assigned_count = sum(1 for v in doc_to_labels.values() if v)
            print(f"  → Gán vào existing clusters: {assigned_count}/{n} docs (multi-label)")
        except Exception as e:
            print(f"  ⚠️ Assign error: {e}")

    # ── 2b. Tạo clusters mới cho docs không khớp cluster nào ────────────
    if unassigned_docs:
        try:
            new_clusters = _gemini.cluster_unassigned_documents(unassigned_docs)
            existing_label_set = {c["label"] for c in _clusters}

            for nc in new_clusters:
                new_label = nc.label
                # Tránh trùng tên
                if new_label in existing_label_set:
                    new_label = f"{new_label} (Mới)"
                    nc.label = new_label
                _clusters.append({"label": new_label, "documents": []})
                existing_label_set.add(new_label)

                for doc in nc.documents:
                    if doc.id in doc_to_labels:
                        doc_to_labels[doc.id].append(new_label)

            print(f"  → Tạo {len(new_clusters)} clusters mới cho {len(unassigned_docs)} docs")
        except Exception as e:
            print(f"  ⚠️ Cluster new error: {e}")

    # ── 2c. Apply renames ────────────────────────────────────────────────
    for old_label, new_label in renames.items():
        for c in _clusters:
            if c["label"] == old_label:
                print(f"  🔄 Rename: '{old_label}' → '{new_label}'")
                c["label"] = new_label
                # Cập nhật doc_to_labels
                for doc_id in doc_to_labels:
                    doc_to_labels[doc_id] = [
                        new_label if lb == old_label else lb
                        for lb in doc_to_labels[doc_id]
                    ]
                break

    # ── 2d. Thêm docs vào cluster (multi-label) ─────────────────────────
    doc_map = {d.id: d for d in analyzed_docs}
    for doc_id, labels in doc_to_labels.items():
        doc = doc_map.get(doc_id)
        if not doc:
            continue

        doc_dict = {
            "id":         doc.id,
            "fileName":   doc.file_name,
            "keyphrases": doc.keyphrases,
            "summary":    doc.summary,
        }

        unique_labels = list(dict.fromkeys(labels))  # deduplicate, preserve order
        if unique_labels:
            for label in unique_labels:
                target = next((c for c in _clusters if c["label"] == label), None)
                if target:
                    target["documents"].append(doc_dict)
            print(f"  🏷️  {doc.file_name} → {unique_labels}")
        else:
            # Fallback: thêm vào "Linh tinh"
            misc = next((c for c in _clusters if c["label"] == "Linh tinh"), None)
            if misc is None:
                _clusters.append({"label": "Linh tinh", "documents": []})
                misc = _clusters[-1]
            misc["documents"].append(doc_dict)
            print(f"  ↳  {doc.file_name} → Linh tinh (fallback)")

        _all_documents.append(doc_dict)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Done! {len(_clusters)} clusters, {len(_all_documents)} docs")
    print(f"{'=' * 60}\n")
    return {"final_clusters": _clusters, "all_documents": _all_documents}
