"""
LLM Service — Phân nhóm tài liệu theo chủ đề
==============================================
Dựa trên docucluster-ai/services/geminiService.ts

Luồng 3 bước (y hệt docucluster-ai):
  1. extractInfoFromText()      — Trích xuất keyphrases + summary từ văn bản
  2. assignToExistingClusters()  — Gán document vào clusters hiện có (nếu phù hợp)
  3. clusterUnassignedDocuments()— Gom nhóm documents chưa gán thành clusters MỚI

Provider: Kilo AI (MiniMax) — OpenAI-compatible API
Endpoint: https://api.kilo.ai/api/gateway
Model: kilo-auto/free

Chạy:
    python app.py
"""

import os
import json
import time
import random
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
# LLM Client (Kilo AI / MiniMax — OpenAI-compatible API)
# ──────────────────────────────────────────────────────────────────────────────

class LLMService:
    """
    Wrapper gọi Kilo AI (MiniMax) qua OpenAI SDK — theo y hệt luồng docucluster-ai.

    3 bước:
      1. extractInfoFromText()      → AnalyzedDocument
      2. assignToExistingClusters() → assignments + unassigned docs
      3. clusterUnassignedDocuments() → DocumentCluster[]
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
        Gọi chat completions API và trả về nội dung text.
        Yêu cầu model trả về JSON trong system prompt.
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
            # Remove the first ```json (or just ```) and the last ```
            content = re.sub(r"^```(?:json)?\s*\n", "", content)
            content = re.sub(r"\n```\s*$", "", content)
            
        return content

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: Tạo AnalyzedDocument từ dữ liệu đã extract sẵn (TextRank + KeyBERT)
    #  KHÔNG gọi LLM — keyphrases và summary do Stage 1+2 cung cấp.
    # ─────────────────────────────────────────────────────────────────────────
    def make_analyzed_document(
        self,
        doc_id: str,
        keyphrases: List[str],
        summary: str = "",
        file_name: str = "",
        file_size: int = 0,
    ) -> AnalyzedDocument:
        """
        Tạo AnalyzedDocument từ keyphrases + summary đã có (TextRank + KeyBERT).
        Không gọi LLM.
        """
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

    # Backward-compat alias (không còn dùng LLM — chỉ wrap make_analyzed_document)
    def extract_info_from_text(
        self,
        text: str,
        doc_id: str,
        file_name: str = "",
        file_size: int = 0,
        keyphrases: List[str] = None,
        summary: str = "",
    ) -> AnalyzedDocument:
        """
        Backward-compatible: tạo AnalyzedDocument từ keyphrases + summary đã có.
        Nếu keyphrases không được truyền vào, dùng các từ đầu của text làm fallback.
        KHÔNG gọi LLM.
        """
        if keyphrases is None:
            # Fallback đơn giản: lấy các từ dài hơn 3 ký tự từ văn bản
            words = list(dict.fromkeys(
                w for w in text.split() if len(w) > 3
            ))[:10]
            keyphrases = words
        if not summary and text:
            # Lấy câu đầu tiên làm summary
            summary = text.split(".")[0].strip()[:200]
        return self.make_analyzed_document(
            doc_id=doc_id,
            keyphrases=keyphrases,
            summary=summary,
            file_name=file_name,
            file_size=file_size,
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

Respond ONLY with valid JSON in this exact format:
{{
  "assignments": [
    {{
      "documentId": "<id>",
      "clusterLabel": "<existing_label_or_NEW_CLUSTER>",
      "newLabel": "<renamed_label_if_needed_or_null>"
    }}
  ]
}}"""

        try:
            raw = self._call(prompt)
            result = json.loads(raw)
        except (json.JSONDecodeError, Exception):
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

                # Nếu LLM đề xuất đổi tên label
                if new_label and new_label.strip():
                    renames[cluster_label] = new_label.strip()

        unassigned = [d for d in new_documents if d.id in unassigned_ids]

        return {
            "assignments": assignments,
            "unassigned": unassigned,
            "renames": renames,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-LABEL: gán mỗi document vào NHIỀU clusters hiện có
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

        # ── Build context: new documents (chỉ keyphrases, không summary) ──
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

            # Tạo 1 cluster cho mỗi label, đều chứa doc này (multi-label)
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
    # MAIN WORKFLOW: processAndClusterNewDocuments
    #  (y hệt processAndClusterNewDocuments trong docucluster-ai)
    # ─────────────────────────────────────────────────────────────────────────
    def process_and_cluster_new_documents(
        self,
        texts: List[str],
        existing_clusters: List[DocumentCluster],
        existing_documents: List[AnalyzedDocument],
        file_names: List[str] = None,
        analyzed_docs: List[AnalyzedDocument] = None,
    ) -> Dict[str, Any]:
        """
        Luồng chính — phân nhóm theo chủ đề.

        Args:
            texts            : Danh sách văn bản gốc (dùng làm fallback nếu không có analyzed_docs)
            existing_clusters: Clusters hiện có (label + documents)
            existing_documents: Documents đã analyzed trước đó
            file_names       : Tên file tương ứng (optional)
            analyzed_docs    : ⭐ Nếu truyền vào, BỎ QUA bước extract — dùng keyphrases/summary
                               đã được tính bởi TextRank + KeyBERT (Stage 1+2). KHÔNG gọi LLM.

        Returns:
            {
                "final_clusters": List[DocumentCluster],
                "all_documents": List[AnalyzedDocument],
            }
        """
        if file_names is None:
            file_names = [f"doc-{i}" for i in range(len(texts))]

        # ── STEP 1: Dùng analyzed_docs đã có (TextRank + KeyBERT) — KHÔNG gọi LLM ──
        if analyzed_docs is not None:
            newly_analyzed = list(analyzed_docs)
            print(f"  ℹ️  Dùng {len(newly_analyzed)} docs đã extract sẵn (TextRank + KeyBERT) — bỏ qua LLM extract.")
        else:
            # Fallback: nếu không có analyzed_docs, tạo AnalyzedDocument từ text (không dùng LLM)
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


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible alias — các file cũ import GeminiService vẫn hoạt động
# ──────────────────────────────────────────────────────────────────────────────
GeminiService = LLMService
