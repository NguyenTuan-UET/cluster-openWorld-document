"""
Gemini Service — Phân nhóm tài liệu theo chủ đề
=================================================
Dùng Google Gemini API để:
  1. classify_document(): Gán nhãn chủ đề cho 1 tài liệu (multi-label)
  2. refine_labels(): Gộp các nhãn trùng lặp

Lấy ý tưởng từ docucluster-ai, viết lại bằng Python SDK.
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
    pass  # dotenv optional — có thể set env var trực tiếp


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicLabel:
    """Nhãn chủ đề."""
    id:             str
    name:           str
    description:    str
    document_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "document_count": self.document_count,
        }


@dataclass
class ClassifyResult:
    """Kết quả phân nhóm 1 tài liệu."""
    assigned_label_ids: List[str]  = field(default_factory=list)
    new_labels:         List[TopicLabel] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Gemini Client
# ──────────────────────────────────────────────────────────────────────────────

class GeminiService:
    """
    Wrapper gọi Gemini API cho tác vụ phân nhóm tài liệu.

    Parameters
    ----------
    api_key : Gemini API key (hoặc set env GEMINI_API_KEY)
    model   : tên model Gemini (default: gemini-2.0-flash)
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

    # ──────────────────────────────────────────────────────────────────────────
    # classify_document  (giống processNewDocument trong docucluster-ai)
    # ──────────────────────────────────────────────────────────────────────────
    def classify_document(
        self,
        summary_text: str,
        keywords: List[str],
        title: Optional[str],
        existing_labels: List[TopicLabel],
    ) -> ClassifyResult:
        """
        Phân tích tài liệu và phân loại vào các nhóm chủ đề.

        Giống hệt logic docucluster-ai/services/geminiService.ts:
        - Dùng response_schema (structured output) thay vì prompt-based JSON.
        - Ưu tiên gán vào nhãn hiện có để tạo cluster.
        - Chỉ gợi ý nhãn mới nếu nội dung thực sự khác biệt.
        """
        self._ensure_client()
        from google.genai import types

        # ── Response schema (giống responseSchema trong TS) ───────────────────
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "assignedExistingLabelIds": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Danh sách ID của các nhãn hiện có mà tài liệu này thuộc về.",
                ),
                "suggestedNewLabels": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "name": types.Schema(type=types.Type.STRING),
                            "description": types.Schema(type=types.Type.STRING),
                        },
                        required=["name", "description"],
                    ),
                    description="Các chủ đề mới cần tạo nhãn nếu tài liệu này mang nội dung hoàn toàn khác biệt.",
                ),
            },
            required=["assignedExistingLabelIds", "suggestedNewLabels"],
        )

        # ── Label context (giống labelContext trong TS) ───────────────────────
        if existing_labels:
            label_ctx = (
                "DANH SÁCH NHÃN HIỆN TẠI (Ưu tiên gán vào đây để gom nhóm):\n"
                + "\n".join(
                    f"- ID: {lb.id}, Tên: {lb.name}" for lb in existing_labels
                )
            )
        else:
            label_ctx = "Chưa có nhãn nào. Hãy tạo nhãn đầu tiên dựa trên nội dung."

        # ── Content parts (giống contentParts trong TS) ───────────────────────
        kw_str = ", ".join(keywords) if keywords else "(không có)"
        title_str = title or "(không có tiêu đề)"

        doc_content = (
            f"Nội dung tài liệu:\n"
            f"Tiêu đề: {title_str}\n"
            f"Tóm tắt: {summary_text}\n"
            f"Từ khóa: {kw_str}"
        )

        task_prompt = (
            f"Nhiệm vụ: Phân tích tài liệu và phân loại vào các nhóm chủ đề.\n"
            f"QUY TẮC QUAN TRỌNG:\n"
            f"1. Một tài liệu CÓ THỂ thuộc nhiều nhãn cùng lúc nếu nội dung liên quan.\n"
            f"2. ƯU TIÊN gán tài liệu vào các NHÃN HIỆN CÓ để tạo thành nhóm tài liệu (cluster).\n"
            f"3. Chỉ gợi ý NHÃN MỚI nếu nội dung thực sự khác biệt và có tiềm năng tạo thành một nhóm mới sau này.\n"
            f"\n{label_ctx}"
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[doc_content, task_prompt],
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

        # ── Tạo new labels với unique IDs (giống createdLabels trong TS) ──────
        new_labels = []
        for item in data.get("suggestedNewLabels", []):
            label_id = f"label-{int(time.time())}-{random.randint(1000, 9999)}"
            new_labels.append(TopicLabel(
                id=label_id,
                name=item.get("name", "Unknown"),
                description=item.get("description", ""),
                document_count=0,
            ))
            time.sleep(0.01)  # ensure unique timestamp

        # Gộp existing + new label IDs (giống doc.labelIds trong TS)
        assigned_ids = list(data.get("assignedExistingLabelIds", []))
        assigned_ids.extend(lb.id for lb in new_labels)

        return ClassifyResult(
            assigned_label_ids=assigned_ids,
            new_labels=new_labels,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # cluster_documents_by_keywords — Phân nhóm dựa trên danh sách từ khóa
    # ──────────────────────────────────────────────────────────────────────────
    def cluster_documents_by_keywords(
        self,
        documents_keywords: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Nhận danh sách từ khóa của TẤT CẢ tài liệu, dùng Gemini để:
          1. Tạo các nhãn chủ đề (cluster) phù hợp
          2. Gán mỗi tài liệu vào 1 hoặc nhiều nhãn

        Parameters
        ----------
        documents_keywords : list of dict, mỗi dict có dạng:
            {
                "doc_index": int,          # chỉ số tài liệu (0-based)
                "title": str or None,      # tiêu đề tài liệu
                "keywords": List[str],     # danh sách từ khóa đã trích xuất
            }

        Returns
        -------
        dict:
            {
                "labels": List[TopicLabel],
                "assignments": Dict[int, List[str]],  # doc_index → [label_id, ...]
            }
        """
        self._ensure_client()
        from google.genai import types

        # ── Response schema ───────────────────────────────────────────────────
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "clusters": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "label_name": types.Schema(
                                type=types.Type.STRING,
                                description="Tên ngắn gọn cho nhóm chủ đề.",
                            ),
                            "label_description": types.Schema(
                                type=types.Type.STRING,
                                description="Mô tả ngắn về nhóm chủ đề.",
                            ),
                            "document_indices": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(type=types.Type.INTEGER),
                                description="Danh sách chỉ số tài liệu thuộc nhóm này (0-based).",
                            ),
                        },
                        required=["label_name", "label_description", "document_indices"],
                    ),
                    description="Danh sách các nhóm chủ đề được tạo từ việc phân tích từ khóa.",
                ),
            },
            required=["clusters"],
        )

        # ── Tạo nội dung mô tả từng tài liệu bằng từ khóa ──────────────────
        doc_descriptions = []
        for doc in documents_keywords:
            idx = doc["doc_index"]
            title = doc.get("title") or "(không có tiêu đề)"
            kw_str = ", ".join(doc["keywords"]) if doc["keywords"] else "(không có từ khóa)"
            doc_descriptions.append(
                f"  Tài liệu {idx}: Tiêu đề: {title} | Từ khóa: {kw_str}"
            )
        docs_content = "\n".join(doc_descriptions)

        prompt = (
            f"Nhiệm vụ: Phân nhóm (clustering) các tài liệu theo chủ đề dựa trên DANH SÁCH TỪ KHÓA đã được trích xuất của mỗi tài liệu.\n\n"
            f"DANH SÁCH TÀI LIỆU VÀ TỪ KHÓA:\n{docs_content}\n\n"
            f"QUY TẮC:\n"
            f"1. Phân tích từ khóa của các tài liệu để tìm ra các chủ đề chung.\n"
            f"2. Tài liệu có từ khóa tương tự hoặc liên quan nên được gom vào cùng một nhóm.\n"
            f"3. Một tài liệu CÓ THỂ thuộc nhiều nhóm nếu từ khóa của nó liên quan đến nhiều chủ đề.\n"
            f"4. Mỗi nhóm cần có tên (label_name) ngắn gọn, mô tả (label_description) rõ ràng.\n"
            f"5. Số lượng nhóm nên hợp lý — không quá ít (gom tất cả thành 1) cũng không quá nhiều (mỗi tài liệu 1 nhóm riêng).\n"
            f"6. Tên nhãn nên phản ánh chủ đề chung của các từ khóa trong nhóm.\n"
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

        # ── Parse kết quả thành TopicLabel + assignments ──────────────────────
        labels: List[TopicLabel] = []
        assignments: Dict[int, List[str]] = {}

        for cluster in data.get("clusters", []):
            label_id = f"label-{int(time.time())}-{random.randint(1000, 9999)}"
            label = TopicLabel(
                id=label_id,
                name=cluster.get("label_name", "Unknown"),
                description=cluster.get("label_description", ""),
                document_count=len(cluster.get("document_indices", [])),
            )
            labels.append(label)

            for doc_idx in cluster.get("document_indices", []):
                if doc_idx not in assignments:
                    assignments[doc_idx] = []
                assignments[doc_idx].append(label_id)

            time.sleep(0.01)  # ensure unique timestamp

        return {
            "labels": labels,
            "assignments": assignments,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # refine_labels  (giống refineLabelSpace trong docucluster-ai)
    # ──────────────────────────────────────────────────────────────────────────
    def refine_labels(
        self,
        labels: List[TopicLabel],
    ) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra và gộp nhãn trùng lặp / quá giống nhau.

        Returns
        -------
        None nếu không cần gộp, hoặc dict:
          {"merged_label_ids": [...], "new_label": TopicLabel}
        """
        if len(labels) < 2:
            return None

        self._ensure_client()
        from google.genai import types

        # ── Response schema (giống responseSchema trong TS) ───────────────────
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "shouldMerge": types.Schema(type=types.Type.BOOLEAN),
                "mergeLabelIds": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
                "newName": types.Schema(type=types.Type.STRING),
                "newDescription": types.Schema(type=types.Type.STRING),
            },
            required=["shouldMerge"],
        )

        labels_json = json.dumps(
            [lb.to_dict() for lb in labels],
            ensure_ascii=False,
        )

        prompt = (
            f"Xem xét các nhãn chủ đề sau. Có nhãn nào trùng lặp hoặc quá giống nhau không?\n"
            f"Mục tiêu là gom nhóm tài liệu hiệu quả nhất. Nếu có, hãy đề xuất gộp chúng lại.\n"
            f"Nhãn: {labels_json}"
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
            data = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            return None

        if not data.get("shouldMerge") or len(data.get("mergeLabelIds", [])) < 2:
            return None

        merged_label = TopicLabel(
            id=f"label-merged-{int(time.time())}",
            name=data.get("newName", "Merged"),
            description=data.get("newDescription", ""),
            document_count=0,
        )

        return {
            "merged_label_ids": data["mergeLabelIds"],
            "new_label": merged_label,
        }
