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
from typing import List, Optional
import uuid

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_loaded, _gemini_loaded
    print("=" * 60)
    print("  NLP Pipeline API — Đang khởi động…")
    print("=" * 60)

    try:
        from backend.cluster.combined_pipeline import CombinedPipeline
        global _pipeline
        _pipeline = CombinedPipeline(enable_clustering=True)
        _pipeline.load()
        _pipeline_loaded = True
        print("✅ CombinedPipeline loaded!")
    except Exception as e:
        print(f"⚠️ CombinedPipeline error: {e}")

    try:
        from backend.cluster.gemini_service import GeminiService
        global _gemini
        _gemini = GeminiService(model="gemini-2.5-flash")
        _gemini._ensure_client()
        _gemini_loaded = True
        print("✅ Gemini Service loaded!")
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")

    print("=" * 60)
    print("  ✅ Server sẵn sàng!")
    print("=" * 60)
    yield
    print("Server shutting down...")


app = FastAPI(title="NLP Pipeline API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state ──────────────────────────────────────────────────────────
_all_documents: List[dict] = []
_clusters: List[dict] = []
_pipeline_loaded = False
_gemini_loaded = False
_pipeline = None  # type: ignore


# ── Pydantic schemas ────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    texts: List[str]
    file_names: Optional[List[str]] = None


@app.get("/")
def root():
    return {"status": "ok", "message": "NLP Pipeline API", "pipeline": _pipeline_loaded, "gemini": _gemini_loaded}


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
    global _all_documents, _clusters, _pipeline

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is required")

    from backend.cluster.gemini_service import GeminiService, AnalyzedDocument, DocumentCluster

    # STEP 1: Extract bằng CombinedPipeline (reuse từ startup)
    print(f"Processing {len(req.texts)} documents...")
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline chưa load. Kiểm tra backend startup.")
    pipeline = _pipeline

    analyzed_docs: List[AnalyzedDocument] = []
    for i, text in enumerate(req.texts):
        file_name = req.file_names[i] if req.file_names and i < len(req.file_names) else f"doc-{i}"
        result = pipeline.run(text=text, title=file_name)

        doc = AnalyzedDocument(
            id=f"doc-{uuid.uuid4().hex[:8]}",
            file_name=file_name,
            keyphrases=[kw for kw, _ in result.keywords],
            summary=result.summary_text,
        )
        analyzed_docs.append(doc)
        print(f"  ✅ {file_name}: {len(result.keywords)} keywords")

    # STEP 2: Assign to existing clusters (có thể rename label)
    assignments: dict = {}
    renames: dict = {}
    unassigned_docs: List[AnalyzedDocument] = list(analyzed_docs)

    if _clusters and analyzed_docs:
        try:
            gemini = GeminiService(model="gemini-2.5-flash")
            gemini._ensure_client()
            existing = [DocumentCluster(label=c["label"], documents=[]) for c in _clusters]
            for c in _clusters:
                for d in c.get("documents", []):
                    existing[next(i for i, x in enumerate(existing) if x.label == c["label"])].documents.append(
                        AnalyzedDocument(id=d["id"], file_name=d["fileName"], keyphrases=d["keyphrases"], summary=d["summary"])
                    )
            result = gemini.assign_to_existing_clusters(analyzed_docs, existing)
            assignments = result["assignments"]
            renames = result.get("renames", {})
            unassigned_docs = result["unassigned"]
        except Exception as e:
            print(f"⚠️ Assign error: {e}")

    # STEP 3: Cluster unassigned
    new_clusters: List[DocumentCluster] = []
    if unassigned_docs:
        try:
            gemini = GeminiService(model="gemini-2.5-flash")
            gemini._ensure_client()
            new_clusters = gemini.cluster_unassigned_documents(unassigned_docs)
        except Exception as e:
            print(f"⚠️ Cluster error: {e}")

    # ── Merge: existing + assigned + new ──
    final_map: dict = {}

    # 1. Copy existing clusters
    for c in _clusters:
        final_map[c["label"]] = {"label": c["label"], "documents": list(c.get("documents", []))}

    # 2. Apply renames (old_label → new_label) nếu Gemini đề xuất
    for old_label, new_label in renames.items():
        if old_label in final_map:
            docs_in_old = list(final_map[old_label]["documents"])
            final_map[new_label] = {"label": new_label, "documents": docs_in_old}
            del final_map[old_label]
            print(f"  🔄 Rename: '{old_label}' → '{new_label}'")

    # 3. Gán doc mới vào clusters (dùng label đã rename)
    doc_map = {d.id: d for d in analyzed_docs}
    for doc_id, cluster_label in assignments.items():
        doc = doc_map.get(doc_id)
        if doc and cluster_label in final_map:
            final_map[cluster_label]["documents"].append({
                "id": doc.id, "fileName": doc.file_name,
                "keyphrases": doc.keyphrases, "summary": doc.summary
            })

    # 4. Thêm clusters mới cho unassigned docs
    for nc in new_clusters:
        label = nc.label
        if label in final_map:
            label = f"{label} (Mới)"
        final_map[label] = {"label": label, "documents": [
            {"id": d.id, "fileName": d.file_name, "keyphrases": d.keyphrases, "summary": d.summary}
            for d in nc.documents
        ]}

    _clusters = list(final_map.values())
    _all_documents = _all_documents + [
        {"id": d.id, "fileName": d.file_name, "keyphrases": d.keyphrases, "summary": d.summary}
        for d in analyzed_docs
    ]

    print(f"✅ Done! {len(_clusters)} clusters, {len(_all_documents)} docs")
    return {"final_clusters": _clusters, "all_documents": _all_documents}
