"""
LLM Service — Phân nhóm tài liệu theo chủ đề
==============================================

Luồng 2 bước (sau TextRank + KeyBERT):
  1. assign_to_existing_clusters_multilabel() — Gán document vào clusters hiện có (multi-label)
  2. cluster_unassigned_documents()           — Gom nhóm documents chưa gán thành clusters MỚI

Provider: Kilo AI (MiniMax) — OpenAI-compatible API
Endpoint: https://api.kilo.ai/api/gateway
Model: kilo-auto/free
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ─── Load API key từ .env ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalyzedDocument:
    """
    Document sau khi trích xuất keyphrases + summary (TextRank + KeyBERT).
    """
    id: str
    file_name: str = ""
    file_size: int = 0
    type: str = "text/plain"
    keyphrases: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class DocumentCluster:
    """
    Cluster = label + danh sách documents.
    """
    label: str
    documents: List[AnalyzedDocument] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "documents": [
                {
                    "id": d.id,
                    "fileName": d.file_name,
                    "keyphrases": d.keyphrases,
                    "summary": d.summary,
                }
                for d in self.documents
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# LLM Client (Kilo AI / MiniMax — OpenAI-compatible API)
# ──────────────────────────────────────────────────────────────────────────────

class LLMService:
    """
    Wrapper gọi Kilo AI (MiniMax) qua OpenAI SDK.

    2 bước chính:
      1. assign_to_existing_clusters_multilabel() → assignments + unassigned docs
      2. cluster_unassigned_documents()           → DocumentCluster[]
    """

    KILO_BASE_URL = "https://api.kilo.ai/api/gateway"
    DEFAULT_MODEL = "kilo-auto/free"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("KILO_API_KEY", "")
        self._model_name = model or os.environ.get("KILO_MODEL", self.DEFAULT_MODEL)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise ValueError(
                "Chưa có KILO_API_KEY. "
                "Hãy tạo file .env với KILO_API_KEY=... "
                "hoặc truyền api_key khi khởi tạo LLMService."
            )
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self.KILO_BASE_URL,
        )

    def _call(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Gọi chat completions API và trả về nội dung text (JSON).
        """
        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Always respond with valid JSON only. Do not include any explanation or markdown, just raw JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content or "{}"

        # Clean markdown code blocks if the model ignores the instruction
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n", "", content)
            content = re.sub(r"\n```\s*$", "", content)

        return content

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Gán document vào clusters hiện có (MULTI-LABEL)
    # ─────────────────────────────────────────────────────────────────────────
    def assign_to_existing_clusters_multilabel(
        self,
        new_documents: List[AnalyzedDocument],
        existing_clusters: List[DocumentCluster],
    ) -> Dict[str, Any]:
        if not existing_clusters or not new_documents:
            return {"assignments": {}, "renames": {}, "unassigned": list(new_documents)}

        # ── Build context: existing clusters (label + representative keyphrases) ──
        existing_ctx = "\n".join(
            f'  - "{c.label}": [{", ".join(kw for d in c.documents[:5] for kw in d.keyphrases[:4])}]'
            for c in existing_clusters
        )

        # ── Build context: new documents ──
        new_docs_ctx = "\n".join(
            f'  - ID: "{d.id}", Keyphrases: [{", ".join(d.keyphrases[:8])}]'
            for d in new_documents
        )

        prompt = f"""You are an advanced AI system specializing in "Unsupervised Vietnamese Multi-label Text Classification".

TASK: Assign each new document to ONE OR MORE existing labels based on its keyphrases.

EXISTING LABEL SPACE (Clusters):
{existing_ctx}

NEW UNLABELLED DOCUMENTS:
{new_docs_ctx}

MULTI-LABEL ASSIGNMENT RULES:
1. MULTI-LABEL IS MANDATORY: If a document covers N different topics, you MUST assign N corresponding labels.
   Example: Keyphrases ["trí_tuệ_nhân_tạo", "chẩn_đoán bệnh", "bệnh_viện"] → assign to both "Y tế" AND "Công nghệ".
2. STRICT ASSIGNMENT THRESHOLD: Assign a label ONLY if the provided keyphrases strongly and explicitly justify it. DO NOT over-assign or hallucinate topics not present in the keyphrases. Keep assignments minimal and accurate.
3. NO OMISSION: Each document must be assigned at least 1 label if suitable. If NO existing labels fit, return an empty array [].
4. USE EXISTING LABELS ONLY: Only assign labels from the provided EXISTING LABEL SPACE. Do not invent new labels here.
5. LABEL RENAME SUGGESTION: If an existing label is too narrow or doesn't perfectly encapsulate the newly added documents, suggest a better, broader label name in the "renames" field.
   Example: The label "Ung thư" could be renamed to "Y tế" if new documents broaden the scope.

Respond ONLY with valid JSON, NO explanations:
{{
  "assignments": [
    {{
      "documentId": "<id>",
      "clusterLabels": ["<label1>", "<label2>"],
      "renames": {{"<old_label>": "<new_label>"}}
    }}
  ]
}}"""

        try:
            raw = self._call(prompt)
            result = json.loads(raw)
        except Exception:
            result = {"assignments": []}

        existing_labels = {c.label for c in existing_clusters}
        assignments: Dict[str, List[str]] = {}
        renames: Dict[str, str] = {}

        for item in result.get("assignments", []):
            doc_id         = item.get("documentId", "")
            cluster_labels = item.get("clusterLabels", [])
            rename_map     = item.get("renames", {}) or {}

            valid_labels = [lb for lb in cluster_labels if lb in existing_labels]
            if valid_labels:
                assignments[doc_id] = valid_labels

            for old_lb, new_lb in rename_map.items():
                if old_lb in existing_labels and new_lb and new_lb.strip():
                    renames[old_lb] = new_lb.strip()

        unassigned = [d for d in new_documents if not assignments.get(d.id)]

        return {
            "assignments": assignments,
            "renames":     renames,
            "unassigned":  unassigned,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Tạo clusters MỚI cho documents chưa được gán
    # ─────────────────────────────────────────────────────────────────────────
    def cluster_unassigned_documents(
        self,
        documents: List[AnalyzedDocument],
    ) -> List[DocumentCluster]:
        if not documents:
            return []

        docs_for_clustering = [
            {
                "id": d.id,
                "fileName": d.file_name,
                "keyphrases": d.keyphrases[:5],
            }
            for d in documents
        ]

        if len(documents) == 1:
            doc = docs_for_clustering[0]
            prompt = f"""You are an advanced AI system for "Unsupervised Vietnamese Multi-label Text Classification".

TASK: Discover ALL suitable topic labels for the document below based on its keyphrases.

Keywords: {', '.join(doc['keyphrases'])}

RULES:
1. MINIMAL MULTI-LABEL: List the MINIMUM number of topics necessary to cover the core content. Base your decision STRICTLY on the provided keyphrases. Do not generate tangential or sprawling labels.
   Example: Keyphrases ["trí_tuệ_nhân_tạo", "chẩn_đoán", "bệnh_viện"] → ["Y tế", "Công nghệ"]
2. Labels MUST be in Vietnamese, concise (2-4 words), and represent a broad parent category (e.g., "Y tế", "Giáo dục", "Kinh tế").
3. SINGLE CONCEPT PER LABEL: Each label MUST represent ONE single concept. DO NOT use "A và B" or "A & B" formats.
4. ABSTRACTION: DO NOT use raw keywords directly as labels. Abstract them into higher-level taxonomic categories.

Respond ONLY with valid JSON:
{{ "labels": ["<label_1>", "<label_2>"] }}"""

            try:
                raw = self._call(prompt, temperature=0.3)
                result = json.loads(raw)
                labels = result.get("labels", []) or [result.get("label", "Tài liệu đơn lẻ")]
            except (json.JSONDecodeError, Exception):
                labels = [documents[0].keyphrases[0]] if documents[0].keyphrases else ["Tài liệu đơn lẻ"]

            return [DocumentCluster(label=lb, documents=list(documents)) for lb in labels]

        prompt = f"""You are an advanced AI system specializing in "Unsupervised Vietnamese Multi-label Text Classification".

TASK: Group the unlabelled documents below into topic clusters. A single document can belong to MULTIPLE clusters.

MULTI-LABEL CLUSTERING RULES:
1. ORTHOGONAL TAXONOMY: Create distinct, broad Vietnamese labels (2-4 words) that do not overlap.
   Good examples: "Y tế", "Giáo dục", "Công nghệ", "Môi trường", "Kinh tế", "Xã hội"
2. MULTI-LABEL MANDATORY: If a document relates to N topics, it MUST appear in N clusters.
   Example: Keyphrases about "ô nhiễm không khí ảnh hưởng sức khỏe" → assign to both "Môi trường" AND "Y tế" clusters.
3. SINGLE CONCEPT PER LABEL: DO NOT use "A và B" or "A & B". If a document covers A and B, put it in two separate clusters.
4. ABSTRACTION: Do not use raw keywords as labels. Synthesize them into broader parent categories.
5. STRICTLY KEYWORD-BASED & MINIMAL CLUSTERS: Generate the MINIMUM number of clusters needed to cover the core topics. Cluster labels MUST be strictly grounded in the provided keyphrases. Avoid sprawling, tangential, or overly granular clusters.

INPUT DOCUMENTS TO CLUSTER:
{json.dumps(docs_for_clustering, ensure_ascii=False, indent=2)}

Respond ONLY with valid JSON, NO explanations:
{{
  "clusters": [
    {{ "label": "<vietnamese_label>", "documents": ["id1", "id2"] }}
  ]
}}"""

        try:
            raw = self._call(prompt, temperature=0.1)
            result = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            result = {"clusters": []}

        # ── Hydrate: convert doc IDs → AnalyzedDocument objects ──
        doc_map = {d.id: d for d in documents}
        clusters: List[DocumentCluster] = []

        for cluster_data in result.get("clusters", []):
            cluster_docs = [
                doc_map[doc_id]
                for doc_id in cluster_data.get("documents", [])
                if doc_id in doc_map
            ]
            if cluster_docs:
                clusters.append(DocumentCluster(
                    label=cluster_data.get("label", "Không xác định"),
                    documents=cluster_docs,
                ))

        # ── Missed documents → "Linh tinh" cluster ──
        clustered_ids = {d.id for c in clusters for d in c.documents}
        missed = [d for d in documents if d.id not in clustered_ids]
        if missed:
            clusters.append(DocumentCluster(label="Linh tinh", documents=missed))

        return clusters

    # ─────────────────────────────────────────────────────────────────────────
    # ORCHESTRATION: Phase 2 — Multi-label clustering (dùng chung cho cả
    # FastAPI và Gradio, tránh duplicate logic)
    # ─────────────────────────────────────────────────────────────────────────
    def run_batch_clustering(
        self,
        analyzed_docs: List[AnalyzedDocument],
        clusters_state: List[dict],
    ) -> Dict[str, Any]:
        """
        Phase 2: Multi-label clustering với LLM.

        Args:
            analyzed_docs  : Danh sách AnalyzedDocument đã có keyphrases (từ TextRank + KeyBERT).
            clusters_state : State hiện tại — list of dicts {"label": str, "documents": [...]}.
                             Được cập nhật IN-PLACE và trả về lại.

        Returns:
            {
                "clusters":  List[dict]  — clusters_state đã cập nhật,
                "documents": List[dict]  — các doc vừa được thêm vào (để append vào all_documents),
            }
        """
        n = len(analyzed_docs)
        doc_to_labels: Dict[str, List[str]] = {d.id: [] for d in analyzed_docs}

        # Build DocumentCluster objects từ state dicts
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
            for c in clusters_state
        ]

        unassigned_docs = list(analyzed_docs)
        renames: Dict[str, str] = {}

        # ── 2a. Gán vào clusters hiện có (multi-label) ──
        if existing:
            try:
                assign_result   = self.assign_to_existing_clusters_multilabel(analyzed_docs, existing)
                assignments     = assign_result.get("assignments", {})
                renames         = assign_result.get("renames", {})
                unassigned_docs = assign_result["unassigned"]

                for doc_id, labels in assignments.items():
                    if doc_id in doc_to_labels:
                        doc_to_labels[doc_id].extend(labels)

                assigned_count = sum(1 for v in doc_to_labels.values() if v)
                print(f"  → Gán vào existing clusters: {assigned_count}/{n} docs (multi-label)")
            except Exception as e:
                print(f"  ⚠️ Assign error: {e}")

        # ── 2b. Tạo clusters mới cho docs chưa gán ──
        if unassigned_docs:
            try:
                new_clusters = self.cluster_unassigned_documents(unassigned_docs)
                existing_label_set = {c["label"] for c in clusters_state}

                for nc in new_clusters:
                    new_label = nc.label
                    if new_label in existing_label_set:
                        new_label = f"{new_label} (Mới)"
                        nc.label = new_label
                    clusters_state.append({"label": new_label, "documents": []})
                    existing_label_set.add(new_label)

                    for doc in nc.documents:
                        if doc.id in doc_to_labels:
                            doc_to_labels[doc.id].append(new_label)

                print(f"  → Tạo {len(new_clusters)} clusters mới cho {len(unassigned_docs)} docs")
            except Exception as e:
                print(f"  ⚠️ Cluster new error: {e}")

        # ── 2c. Apply renames ──
        for old_label, new_label in renames.items():
            for c in clusters_state:
                if c["label"] == old_label:
                    print(f"  🔄 Rename: '{old_label}' → '{new_label}'")
                    c["label"] = new_label
                    for doc_id in doc_to_labels:
                        doc_to_labels[doc_id] = [
                            new_label if lb == old_label else lb
                            for lb in doc_to_labels[doc_id]
                        ]
                    break

        # ── 2d. Thêm docs vào clusters (multi-label) ──
        doc_map = {d.id: d for d in analyzed_docs}
        new_docs: List[dict] = []

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

            unique_labels = list(dict.fromkeys(labels))
            if unique_labels:
                for label in unique_labels:
                    target = next((c for c in clusters_state if c["label"] == label), None)
                    if target:
                        target["documents"].append(doc_dict)
                print(f"  🏷️  {doc.file_name} → {unique_labels}")
            else:
                # Fallback: "Linh tinh"
                misc = next((c for c in clusters_state if c["label"] == "Linh tinh"), None)
                if misc is None:
                    clusters_state.append({"label": "Linh tinh", "documents": []})
                    misc = clusters_state[-1]
                misc["documents"].append(doc_dict)
                print(f"  ↳  {doc.file_name} → Linh tinh (fallback)")

            new_docs.append(doc_dict)

        return {
            "clusters":  clusters_state,
            "documents": new_docs,
        }
