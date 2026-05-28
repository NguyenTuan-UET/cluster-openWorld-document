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
_pipeline = None
_gemini = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_loaded, _gemini_loaded, _pipeline, _gemini
    print("=" * 60)
    print("  NLP Pipeline API — Đang khởi động…")
    print("=" * 60)

    try:
        from backend.cluster.combined_pipeline import CombinedPipeline
        _pipeline = CombinedPipeline(
            use_mmr=False,
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


# data backend
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

    # 1. TextRank + KeyBERT cho TẤT CẢ (local, không LLM)
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

    # 2. LLM — Multi-label clustering
    print(f"\n🤖 Phase 2/2 — LLM phân nhóm multi-label…")

    result = _gemini.run_batch_clustering(analyzed_docs, _clusters)
    _all_documents.extend(result["documents"])
    # _clusters đã được cập nhật in-place bên trong run_batch_clustering

    print(f"\n{'=' * 60}")
    print(f"  ✅ Done! {len(_clusters)} clusters, {len(_all_documents)} docs")
    print(f"{'=' * 60}\n")
    return {"final_clusters": _clusters, "all_documents": _all_documents}
