"""
Gemini Service — Phân nhóm tài liệu theo chủ đề
=================================================
Dựa trên docucluster-ai/services/geminiService.ts

Luồng 3 bước (y hệt docucluster-ai):
  1. extractInfoFromText()      — Trích xuất keyphrases + summary từ văn bản
  2. assignToExistingClusters()  — Gán document vào clusters hiện có (nếu phù hợp)
  3. clusterUnassignedDocuments()— Gom nhóm documents chưa gán thành clusters MỚI

Chạy:
    python app.py
"""

import os
import json
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ─── Load API key từ .env ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Data classes  (y hệt types.ts trong docucluster-ai)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalyzedDocument:
    """
    Document sau khi trích xuất keyphrases + summary.
    Tương ứng với AnalyzedDocument trong docucluster-ai.
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
    Tương ứng với DocumentCluster trong docucluster-ai.
    Đây là đơn vị cơ bản — không có Label object riêng.
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
# Gemini Client
# ──────────────────────────────────────────────────────────────────────────────

class GeminiService:
    """
    Wrapper gọi Gemini API — theo y hệt luồng docucluster-ai.

    3 bước:
      1. extractInfoFromText()      → AnalyzedDocument
      2. assignToExistingClusters() → assignments + unassigned docs
      3. clusterUnassignedDocuments() → DocumentCluster[]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self._api_key:
            raise ValueError(
                "Chưa có GEMINI_API_KEY. "
                "Hãy tạo file .env với GEMINI_API_KEY=... "
                "hoặc truyền api_key khi khởi tạo GeminiService."
            )
        from google import genai
        self._client = genai.Client(api_key=self._api_key)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Trích xuất keyphrases + summary từ văn bản
    #  (y hệt extractInfoFromDocument trong docucluster-ai)
    # ─────────────────────────────────────────────────────────────────────────
    def extract_info_from_text(
        self,
        text: str,
        doc_id: str,
        file_name: str = "",
        file_size: int = 0,
    ) -> AnalyzedDocument:
        """
        Trích xuất 5-10 keyphrases và 1 câu summary từ văn bản.

        Tương ứng với extractInfoFromDocument() trong docucluster-ai,
        nhưng nhận text thuần thay vì File object.
        """
        self._ensure_client()
        from google.genai import types

        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "keyphrases": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Array of 5-10 keyphrases/keywords extracted from the document.",
                ),
                "summary": types.Schema(
                    type=types.Type.STRING,
                    description="One-sentence summary of the document.",
                ),
            },
            required=["keyphrases", "summary"],
        )

        prompt = (
            "Extract 5-10 keyphrases and a one-sentence summary from this document.\n\n"
            f"Document Content: {text}"
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1,
            ),
        )

        try:
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            data = {}

        keyphrases = data.get("keyphrases", [])
        summary = data.get("summary", "")

        # Giới hạn 5-10 keyphrases như docucluster-ai
        if len(keyphrases) > 10:
            keyphrases = keyphrases[:10]

        return AnalyzedDocument(
            id=doc_id,
            file_name=file_name,
            file_size=file_size,
            type="text/plain",
            keyphrases=keyphrases,
            summary=summary,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Gán document vào clusters hiện có
    #  (y hệt assignToExistingClusters trong docucluster-ai)
    # ─────────────────────────────────────────────────────────────────────────
    def assign_to_existing_clusters(
        self,
        new_documents: List[AnalyzedDocument],
        existing_clusters: List[DocumentCluster],
    ) -> Dict[str, Any]:
        """
        Gán mỗi document mới vào cluster hiện có PHÙ HỢP NHẤT.

        Returns:
            {
                "assignments": {doc_id: cluster_label, ...},
                "unassigned": [AnalyzedDocument, ...]
            }

        Quy tắc (y hệt docucluster-ai):
          • Mỗi doc được gán vào cluster phù hợp nhất
          • Nếu KHÔNG phù hợp cluster nào → gán null (BE STRICT)
          • Chỉ chấp nhận label THỰC SỰ có trong existing_clusters
        """
        if not existing_clusters or not new_documents:
            return {
                "assignments": {},
                "unassigned": new_documents,
            }

        self._ensure_client()
        from google.genai import types

        # ── Build prompt (y hệt prompt trong docucluster-ai) ──
        existing_ctx = "\n".join(
            f'- "{c.label}": Represents topics like [{", ".join(
                kw for d in c.documents[:3] for kw in d.keyphrases
            )}]'
            for c in existing_clusters
        )

        # Chỉ dùng TOP 5 keyphrases (KeyBERT đã extract rồi)
        new_docs_ctx = "\n".join(
            f'- ID: "{d.id}", Keyphrases: [{", ".join(d.keyphrases[:5])}]'
            for d in new_documents
        )

        prompt = f"""You are a document clustering expert. Compare each new document against existing cluster labels and decide the best action.

EXISTING CLUSTERS:
{existing_ctx}

NEW DOCUMENTS:
{new_docs_ctx}

INSTRUCTIONS — For each new document:
1. Compare its keywords against the EXISTING cluster labels.
2. If the document is SIMILAR to an existing cluster:
   - Assign it to that cluster.
   - If the cluster label could be IMPROVED to better reflect both old docs AND the new doc, suggest a renamed label.
3. If the document is DIFFERENT from all existing clusters:
   - Assign it to "NEW_CLUSTER" — a new cluster will be created for it.

OUTPUT FORMAT:
{{
  "assignments": [
    {{
      "documentId": "<id>",
      "clusterLabel": "<existing_label_or_NEW_CLUSTER>",
      "newLabel": "<renamed_label_if_needed_or_null>"
    }}
  ]
}}
"""

        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "assignments": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "documentId": types.Schema(type=types.Type.STRING),
                            "clusterLabel": types.Schema(type=types.Type.STRING),
                            "newLabel": types.Schema(
                                type=types.Type.STRING,
                                nullable=True,
                            ),
                        },
                        required=["documentId", "clusterLabel"],
                    ),
                ),
            },
            required=["assignments"],
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        try:
            result = json.loads(response.text or '{"assignments": []}')
        except json.JSONDecodeError:
            result = {"assignments": []}

        # ── Validation: chỉ chấp nhận label THỰC SỰ có trong existing_clusters ──
        existing_labels = {c.label for c in existing_clusters}
        assignments: Dict[str, str] = {}
        renames: Dict[str, str] = {}  # old_label -> new_label
        unassigned_ids: set = {d.id for d in new_documents}

        for item in result.get("assignments", []):
            doc_id = item.get("documentId", "")
            cluster_label = item.get("clusterLabel", "")
            new_label = item.get("newLabel") or ""

            if cluster_label == "NEW_CLUSTER":
                # Doc khác biệt → sẽ tạo cluster mới
                continue

            if cluster_label in existing_labels:
                assignments[doc_id] = cluster_label
                unassigned_ids.discard(doc_id)

                # Nếu Gemini đề xuất đổi tên label
                if new_label and new_label.strip():
                    renames[cluster_label] = new_label.strip()

        unassigned = [d for d in new_documents if d.id in unassigned_ids]

        return {
            "assignments": assignments,
            "unassigned": unassigned,
            "renames": renames,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Gom nhóm documents chưa gán thành clusters MỚI
    #  (y hệt clusterUnassignedDocuments trong docucluster-ai)
    # ─────────────────────────────────────────────────────────────────────────
    def cluster_unassigned_documents(
        self,
        documents: List[AnalyzedDocument],
    ) -> List[DocumentCluster]:
        """
        Gom nhóm documents CHƯA được gán vào clusters hiện có,
        tạo clusters MỚI với labels được LLM sinh ra.

        Quy tắc (y hệt docucluster-ai):
          • Labels phải bằng TIẾNG VIỆT
          • Dùng ÍT clusters nhất có thể (topic distinct)
          • KHÔNG giải thích lý do
          • Doc không gán được → cluster "Linh tinh"
        """
        if not documents:
            return []

        self._ensure_client()
        from google.genai import types

        # Chỉ dùng TOP 5 keyphrases — KHÔNG gửi raw text, KHÔNG gửi summary
        docs_for_clustering = [
            {
                "id": d.id,
                "fileName": d.file_name,
                # CHỈ gửi keyphrases[:5] — KeyBERT đã extract rồi
                "keyphrases": d.keyphrases[:5],
            }
            for d in documents
        ]

        # Nếu chỉ 1 doc → Gemini đặt tên label cho nó
        if len(documents) == 1:
            doc = docs_for_clustering[0]
            prompt = f"""You are an expert in document classification. Based on the following keywords, assign a coarse-grained (high-level) Vietnamese label (2-4 words) to this document.

Keywords: {', '.join(doc['keyphrases'])}

Rules:
- Label MUST be in Vietnamese, 2-4 words.
- Use a GENERAL, COARSE category (e.g., "Giáo dục", "Công nghệ", "Y tế", "Kinh tế", "Thể thao").
- Do NOT use the keywords directly as label.
- Be concise and generic.

Output Format:
{{ "label": "<vietnamese_label>" }}
"""
            response_schema = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "label": types.Schema(type=types.Type.STRING),
                },
                required=["label"],
            )

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.3,
                ),
            )

            try:
                result = json.loads(response.text or '{"label": ""}')
                label = result.get("label", "Tài liệu đơn lẻ")
            except json.JSONDecodeError:
                label = documents[0].keyphrases[0] if documents[0].keyphrases else "Tài liệu đơn lẻ"

            return [DocumentCluster(label=label, documents=documents)]

        # Nhiều docs → nhờ Gemini gom nhóm và đặt label
        prompt = f"""You are an expert in document clustering. Group the following documents into meaningful clusters based ONLY on their keywords. Assign a coarse-grained (high-level), descriptive Vietnamese label to each cluster.

Rules:
- Labels MUST be in Vietnamese (2-4 words), e.g., "Nghiên cứu AI", "Báo cáo tài chính", "Giáo dục", "Y tế".
- Use a GENERAL, COARSE category — be broad and generic, not specific.
- Use as few clusters as possible while keeping topics distinct.
- Do NOT use the keywords directly as labels. Generate NEW labels.
- Do NOT explain your reasoning.

Input:
{json.dumps(docs_for_clustering, ensure_ascii=False, indent=2)}

Output Format:
{{
  "clusters": [
    {{ "label": "<vietnamese_label>", "documents": ["id1", "id2"] }}
  ]
}}
"""

        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "clusters": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "label": types.Schema(type=types.Type.STRING),
                            "documents": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(type=types.Type.STRING),
                            ),
                        },
                        required=["label", "documents"],
                    ),
                ),
            },
            required=["clusters"],
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1,
            ),
        )

        try:
            result = json.loads(response.text or '{"clusters": []}')
        except json.JSONDecodeError:
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
    # MAIN WORKFLOW: processAndClusterNewDocuments
    #  (y hệt processAndClusterNewDocuments trong docucluster-ai)
    # ─────────────────────────────────────────────────────────────────────────
    def process_and_cluster_new_documents(
        self,
        texts: List[str],
        existing_clusters: List[DocumentCluster],
        existing_documents: List[AnalyzedDocument],
        file_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Luồng chính — 3 bước y hệt docucluster-ai.

        Args:
            texts: Danh sách văn bản cần xử lý
            existing_clusters: Clusters hiện có (label + documents)
            existing_documents: Documents đã analyzed trước đó
            file_names: Tên file tương ứng (optional)

        Returns:
            {
                "final_clusters": List[DocumentCluster],
                "all_documents": List[AnalyzedDocument],
            }
        """
        if file_names is None:
            file_names = [f"doc-{i}" for i in range(len(texts))]

        # ── STEP 1: Extract keyphrases + summary từ TỪNG văn bản (song song) ──
        newly_analyzed: List[AnalyzedDocument] = []
        for i, text in enumerate(texts):
            doc_id = f"doc-{int(time.time())}-{i}-{random.randint(1000, 9999)}"
            time.sleep(0.01)  # ensure unique timestamp
            analyzed = self.extract_info_from_text(
                text=text,
                doc_id=doc_id,
                file_name=file_names[i] if i < len(file_names) else "",
            )
            newly_analyzed.append(analyzed)

        # ── STEP 2: Gán vào clusters hiện có ──
        assignments: Dict[str, str] = {}
        unassigned_docs: List[AnalyzedDocument] = newly_analyzed

        if existing_clusters and newly_analyzed:
            result = self.assign_to_existing_clusters(newly_analyzed, existing_clusters)
            assignments = result["assignments"]
            unassigned_docs = result["unassigned"]

        # ── STEP 3: Cluster documents chưa gán thành clusters MỚI ──
        new_clusters = self.cluster_unassigned_documents(unassigned_docs)

        # ── STEP 4: Merge kết quả ──
        final_clusters_map: Dict[str, DocumentCluster] = {}

        # Copy existing clusters
        for cluster in existing_clusters:
            final_clusters_map[cluster.label] = DocumentCluster(
                label=cluster.label,
                documents=list(cluster.documents),
            )

        # Gán newly analyzed docs vào existing clusters
        new_docs_map = {d.id: d for d in newly_analyzed}
        for doc_id, cluster_label in assignments.items():
            doc = new_docs_map.get(doc_id)
            if doc and cluster_label in final_clusters_map:
                final_clusters_map[cluster_label].documents.append(doc)

        # Thêm clusters mới (tránh trùng label)
        for new_cluster in new_clusters:
            label = new_cluster.label
            if label in final_clusters_map:
                label = f"{label} (Mới)"
            final_clusters_map[label] = new_cluster
            # Update label của cluster object
            new_cluster.label = label

        final_clusters = list(final_clusters_map.values())
        all_documents = list(existing_documents) + list(newly_analyzed)

        return {
            "final_clusters": final_clusters,
            "all_documents": all_documents,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Backward-compatible methods (dùng bởi CombinedPipeline)
    # ─────────────────────────────────────────────────────────────────────────

    def classify_document(
        self,
        summary_text: str,
        keywords: List[str],
        title: Optional[str],
        existing_labels: List["TopicLabel"],
    ) -> "ClassifyResult":
        """
        Phân loại 1 document vào label space.
        Dùng process_and_cluster_new_documents với 1 doc duy nhất.
        """
        # Wrap thành AnalyzedDocument
        doc_id = f"doc-{int(time.time())}"
        doc = self.extract_info_from_text(
            text=summary_text,
            doc_id=doc_id,
            file_name=title or "",
        )

        # Convert existing_labels → DocumentCluster
        clusters: List[DocumentCluster] = []
        for lb in existing_labels:
            clusters.append(DocumentCluster(label=lb.name, documents=[]))

        # Run process_and_cluster
        result = self.process_and_cluster_new_documents(
            texts=[summary_text],
            existing_clusters=clusters,
            existing_documents=[],
            file_names=[title or doc_id],
        )

        final_clusters = result["final_clusters"]
        assigned_ids = []
        new_labels: List[TopicLabel] = []

        for cluster in final_clusters:
            for d in cluster.documents:
                if d.id == doc_id:
                    assigned_ids.append(cluster.label)

        return ClassifyResult(
            assigned_label_ids=assigned_ids,
            new_labels=new_labels,
            used_keywords=keywords,
        )

    def cluster_documents_by_keywords(
        self,
        documents_keywords: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Phân nhóm nhiều documents dựa trên keywords.
        Dùng process_and_cluster_new_documents.
        """
        texts = []
        titles = []
        for dk in documents_keywords:
            texts.append(", ".join(dk.get("keywords", [])))
            titles.append(dk.get("title") or dk.get("doc_index", ""))

        result = self.process_and_cluster_new_documents(
            texts=texts,
            existing_clusters=[],
            existing_documents=[],
            file_names=titles,
        )

        # Convert DocumentCluster[] → labels + assignments
        labels_out: List[TopicLabel] = []
        assignments: Dict[int, List[str]] = {}

        for cluster in result["final_clusters"]:
            label = TopicLabel.from_cluster(cluster)
            labels_out.append(label)
            for doc in cluster.documents:
                try:
                    idx = int(doc.file_name) if doc.file_name.isdigit() else 0
                    if idx not in assignments:
                        assignments[idx] = []
                    assignments[idx].append(label.id)
                except (ValueError, TypeError):
                    pass

        return {
            "labels": labels_out,
            "assignments": assignments,
        }



# ──────────────────────────────────────────────────────────────────────────────
# Backward compatibility — giữ nguyên API cũ cho CombinedPipeline
# TopicLabel = DocumentCluster (label là string, có documents)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicLabel:
    """
    Alias cho backward-compatibility với CombinedPipeline.
    Mỗi TopicLabel tương ứng với 1 DocumentCluster.
    """
    id: str
    name: str  # = cluster.label
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    document_count: int = 0

    @staticmethod
    def from_cluster(cluster: DocumentCluster, doc_keywords: List[str] = None) -> "TopicLabel":
        """Convert DocumentCluster → TopicLabel."""
        return TopicLabel(
            id=f"label-{cluster.label}",
            name=cluster.label,
            description="",
            keywords=doc_keywords or [],
            document_count=len(cluster.documents),
        )


@dataclass
class ClassifyResult:
    """Alias cho backward-compatibility."""
    assigned_label_ids: List[str] = field(default_factory=list)
    new_labels: List[TopicLabel] = field(default_factory=list)
    used_keywords: List[str] = field(default_factory=list)
