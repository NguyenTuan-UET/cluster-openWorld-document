"""
KeyBERT Keyword Extraction – Baseline (standalone)
===================================================
Gom toàn bộ logic từ  backend/cluster/keybert/  vào một file duy nhất
để chạy độc lập trong thư mục  baseline/keyword_extraction/.

Cần:
  - py_vncorenlp, underthesea, pyvi
  - sentence-transformers (asymmetric_emb)
  - transformers  (NER – NlpHUST/ner-vietnamese-electra-base)
  - scikit-learn, numpy, torch
"""

import os
import re
import numpy as np
import torch
from string import punctuation as _std_punctuation
from operator import itemgetter
from typing import List, Tuple

import py_vncorenlp
from underthesea import sent_tokenize
from pyvi.ViTokenizer import tokenize as pyvi_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình đường dẫn
# ══════════════════════════════════════════════════════════════════════════════
_BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PRETRAINED_DIR = os.path.join(_BASE_DIR, "pretrained-models")
_VNCORENLP_DIR  = os.path.join(_PRETRAINED_DIR, "vncorenlp")
_ASYMM_MODEL    = os.path.join(_PRETRAINED_DIR, "asymmetric_emb")
_NER_PT         = os.path.join(_PRETRAINED_DIR, "ner-vietnamese-electra-base.pt")
_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "vietnamese-stopwords-dash.txt")

with open(_STOPWORDS_PATH, encoding="utf-8") as f:
    STOPWORDS = [w.strip() for w in f.readlines()]

# Dấu câu (giữ lại '_')
PUNCTUATION = [c for c in _std_punctuation if c != "_"]
PUNCTUATION += ["\u201c", "\u2013", "\u002c", "\u2026", "\u201d", "\u2013"]


# ══════════════════════════════════════════════════════════════════════════════
# Xử lý văn bản (process_text.py)
# ══════════════════════════════════════════════════════════════════════════════
def process_text_pipeline(text: str) -> str:
    out = text.strip()
    while "\n\n" in out:
        out = out.replace("\n\n", "\n")
    out = _process_sticking_sentences(out)
    while "  " in out:
        out = out.replace("  ", " ")
    return out


def _process_sticking_sentences(full_text: str) -> str:
    for i in range(len(full_text) - 1):
        c1, c2 = full_text[i], full_text[i + 1]
        if c1 in _std_punctuation and c2.isalpha() and c2.isupper():
            full_text = full_text[: i + 1] + " " + full_text[i + 1 :]
        if c1.isalpha() and c1.islower() and c2.isalpha() and c2.isupper():
            full_text = full_text[: i + 1] + ". " + full_text[i + 1 :]
    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# Named Entity Recognition (named_entities.py)
# ══════════════════════════════════════════════════════════════════════════════
def _substring(w, ls):
    for w2 in ls:
        if w != w2 and w in w2:
            return True
    return False
    


def _get_ner_phrases(sent_ner_result):
    ner_list = []
    current_ner = [sent_ner_result[0]["word"]]
    current_idx = sent_ner_result[0]["index"]
    for i in range(1, len(sent_ner_result)):
        if sent_ner_result[i]["index"] == current_idx + 1:
            current_ner.append(sent_ner_result[i]["word"])
        else:
            ner_list.append((" ".join(current_ner), sent_ner_result[i - 1]["entity"]))
            current_ner = [sent_ner_result[i]["word"]]
        current_idx = sent_ner_result[i]["index"]
    ner_list.append((" ".join(current_ner), sent_ner_result[-1]["entity"]))
    return ner_list


def get_named_entities(nlp, doc):
    ner_lists = []
    for sent in sent_tokenize(doc):
        sent_ner_result = nlp(sent)
        if len(sent_ner_result) > 0:
            ner_lists += _get_ner_phrases(sent_ner_result)

    ner_list_non_dup = []
    for entity, ner_type in ner_lists:
        if entity not in ner_list_non_dup and ner_type.startswith("I"):
            ner_list_non_dup.append(entity)

    ner_list_final = [
        w.replace(" ##", "") for w in ner_list_non_dup if not _substring(w, ner_list_non_dup)
    ]
    return ner_list_final


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions (keyword_extraction_utils.py)
# ══════════════════════════════════════════════════════════════════════════════
def sub_sentence(sentence):
    sent = []
    start_index = 0
    while start_index < len(sentence):
        idx_list = []
        for p in PUNCTUATION:
            idx = sentence.find(p, start_index)
            if idx != -1:
                idx_list.append(idx)
        if len(idx_list) == 0:
            sent.append(sentence[start_index:].strip())
            break
        end_index = min(idx_list)
        subsent = sentence[start_index:end_index].strip()
        if len(subsent) > 0:
            sent.append(subsent)
        start_index = end_index + 1
    return sent


def check_for_stopwords(ngram, stopwords_ls):
    for ngram_elem in ngram.split():
        for w in stopwords_ls:
            if ngram_elem == w:
                return True
    return False

    # words = ngram.split()
    
    # # Nếu n-gram rỗng → loại
    # if len(words) == 0:
    #     return True
    
    # # Kiểm tra toàn bộ n-gram có phải stopword không
    # if ngram in stopwords_ls:
    #     return True
    
    # # Kiểm tra từ đầu tiên có phải stopword không
    # if words[0] in stopwords_ls:
    #     return True
    
    # # Kiểm tra từ cuối cùng có phải stopword không
    # if words[-1] in stopwords_ls:
    #     return True
    
    # return False
    
 

def compute_ngram_list(segmentised_doc, ngram_n, stopwords_ls, subsentences=True):
    if subsentences:
        output_sub_sentences = []
        for sentence in segmentised_doc:
            output_sub_sentences += sub_sentence(sentence)
    else:
        output_sub_sentences = segmentised_doc

    ngram_list = []
    for sentence in output_sub_sentences:
        sent = sentence.split()
        for i in range(len(sent) - ngram_n + 1):
            ngram = " ".join(sent[i : i + ngram_n])
            if ngram not in ngram_list and not check_for_stopwords(ngram, stopwords_ls):
                ngram_list.append(ngram)

    final_ngram_list = []
    for ngram in ngram_list:
        if not any(char.isnumeric() for char in ngram):
            final_ngram_list.append(ngram)
    return final_ngram_list


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ── Asymmetric embedding helpers ─────────────────────────────────────────────
def get_doc_embeddings_asymm(segmentised_doc, sentence_model, stopwords):
    """Doc embedding bằng SentenceTransformer (asymmetric_emb).
    Câu đầu tiên (title nếu có) được nhân đôi trọng số.
    """
    embeddings, weights = [], []
    for i, sentence in enumerate(segmentised_doc):
        sent_tokenized = pyvi_tokenize(sentence)
        emb = sentence_model.encode([sent_tokenized], normalize_embeddings=True)[0]
        embeddings.append(emb)
        weights.append(2.0 if i == 0 else 1.0)
    return np.average(np.array(embeddings), axis=0, weights=np.array(weights))


def compute_ngram_embeddings_asymm(sentence_model, ngram_list):
    ngram_embeddings = {}
    for ngram in ngram_list:
        ngram_copy = ngram.lower() if ngram.isupper() else ngram
        ngram_tokenized = pyvi_tokenize(ngram_copy)
        emb = sentence_model.encode([ngram_tokenized], normalize_embeddings=True)[0]
        ngram_embeddings[ngram] = emb
    return ngram_embeddings


def compute_ngram_similarity_asymm(ngram_list, ngram_embeddings, doc_embedding):
    ngram_similarity_dict = {}
    for ngram in ngram_list:
        a = ngram_embeddings[ngram]
        similarity_score = cosine_similarity(a, doc_embedding)
        ngram_similarity_dict[ngram] = float(similarity_score)
    return ngram_similarity_dict


# ── Segmentation & filtering ─────────────────────────────────────────────────
def get_segmentised_doc(nlp, rdrsegmenter, title, doc):
    segmentised_doc = rdrsegmenter.word_segment(doc)
    if title is not None:
        segmentised_doc = rdrsegmenter.word_segment(title) + rdrsegmenter.word_segment(doc)
    ne_ls = set(get_named_entities(nlp, doc))
    segmentised_doc_ne = []
    for sent in segmentised_doc:
        for ne in ne_ls:
            sent = sent.replace(ne, "_".join(ne.split()))
        segmentised_doc_ne.append(sent)
    return ne_ls, segmentised_doc_ne


def compute_filtered_text(annotator, title, text):
    annotated = annotator.annotate_text(text)
    if title is not None:
        annotated = annotator.annotate_text(title + ". " + text)
    keep_tags = ["N", "Np", "V", "Nc"]
    filtered_sentences = []
    for key in annotated.keys():
        sent = " ".join([d["wordForm"] for d in annotated[key] if d["posTag"] in keep_tags])
        filtered_sentences.append(sent)
    return filtered_sentences


def get_candidate_ngrams(segmentised_doc, filtered_segmentised_doc, ngram_n, stopwords_ls):
    actual_ngram_list = compute_ngram_list(segmentised_doc, ngram_n, stopwords_ls, subsentences=True)
    filtered_ngram_list = compute_ngram_list(
        filtered_segmentised_doc, ngram_n, stopwords_ls, subsentences=False
    )
    return [ngram for ngram in filtered_ngram_list if ngram in actual_ngram_list]


def remove_overlapping_ngrams(ngram_list):
    to_remove = set()
    for ngram1 in ngram_list:
        for ngram2 in ngram_list:
            if len(ngram1.split()) > len(ngram2.split()) and (
                ngram1.startswith(ngram2) or ngram1.endswith(ngram2)
            ):
                to_remove.add(ngram2)
    for kw in to_remove:
        ngram_list.remove(kw)
    return ngram_list


def limit_minimum_frequency(doc_segmentised, ngram_list, min_freq=1):
    ngram_dict_freq = {}
    for ngram in ngram_list:
        ngram_n = len(ngram.split())
        count = 0
        for sentence in doc_segmentised:
            sent = sentence.split()
            for i in range(len(sent) - ngram_n + 1):
                if " ".join(sent[i : i + ngram_n]) == ngram:
                    count += 1
        if count >= min_freq:
            ngram_dict_freq[ngram] = count
    return ngram_dict_freq


def remove_duplicates(ngram_result):
    to_remove = set()
    for ngram in ngram_result:
        for ngram2 in ngram_result:
            if ngram not in to_remove and ngram != ngram2 and ngram.lower() == ngram2.lower():
                new_score = np.mean([ngram_result[ngram], ngram_result[ngram2]])
                ngram_result[ngram] = new_score
                to_remove.add(ngram2)
    for ngram in to_remove:
        ngram_result.pop(ngram)
    return ngram_result




# ── Diversification: MMR ─────────────────────────────────────────────────────
def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int = 5,
    diversity: float = 0.8,
) -> List[Tuple[str, float]]:
    """Maximal Marginal Relevance (MMR) – cân bằng relevance vs. diversity."""
    if len(words) == 0:
        return []
    if doc_embedding.ndim == 1:
        doc_embedding = doc_embedding.reshape(1, -1)

    word_doc_similarity = sklearn_cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = sklearn_cosine_similarity(word_embeddings)

    keywords_idx = [int(np.argmax(word_doc_similarity))]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        if not candidates_idx:
            break
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        mmr_scores = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(
            -1, 1
        )
        mmr_idx = candidates_idx[int(np.argmax(mmr_scores))]
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    keywords = [
        (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
        for idx in keywords_idx
    ]
    return sorted(keywords, key=itemgetter(1), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline chính
# ══════════════════════════════════════════════════════════════════════════════
class KeywordExtractorPipeline:
    def __init__(self, ner_model, vncorenlp_instance=None, asymm_model_path=None):
        # ── VnCoreNLP ──
        if vncorenlp_instance is not None:
            self.annotator = vncorenlp_instance
        else:
            print("⏳ Đang khởi tạo VnCoreNLP...")
            self.annotator = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos"],
                save_dir=_VNCORENLP_DIR,
            )
            print("✅ VnCoreNLP sẵn sàng!")

        # ── Asymmetric SentenceTransformer ──
        model_path = asymm_model_path if asymm_model_path is not None else _ASYMM_MODEL
        print("⏳ Đang tải Asymmetric Embedding model...")
        self.sentence_model = SentenceTransformer(model_path)
        print("✅ Asymmetric Embedding sẵn sàng!")

        # ── NER pipeline ──
        print("⏳ Đang khởi tạo NER pipeline...")
        ner_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        self.ner_pipeline = hf_pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        print("✅ NER pipeline sẵn sàng!")

        # ── Stopwords ──
        self.stopwords = STOPWORDS

    def __call__(self, inputs, ngram_n=(1, 3), min_freq=1, top_n=5,
                 use_mmr=False, diversity=0.5):
        model_inputs = self.preprocess(inputs)
        model_outputs = self._forward(model_inputs, ngram_n=ngram_n, min_freq=min_freq)
        return self.postprocess(
            model_outputs, top_n=top_n, use_mmr=use_mmr, diversity=diversity
        )

    def preprocess(self, inputs):
        title = None
        if inputs.get("title"):
            title = process_text_pipeline(inputs["title"])
        text = process_text_pipeline(inputs["text"])
        return {"text": text, "title": title}

    def _forward(self, model_inputs, ngram_n=(1, 3), min_freq=1):
        text = model_inputs["text"]
        title = model_inputs["title"]

        # Phân đoạn từ + nhận diện thực thể tên
        ne_ls, doc_segmentised = get_segmentised_doc(self.ner_pipeline, self.annotator, title, text)
        filtered_doc_segmentised = compute_filtered_text(self.annotator, title, text)

        # Doc embedding dùng asymmetric model
        doc_embedding = get_doc_embeddings_asymm(filtered_doc_segmentised, self.sentence_model, self.stopwords)

        # Danh sách n-gram ứng viên
        ngram_list = self.generate_ngram_list(
            doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq
        )

        # Ngram embeddings dùng asymmetric model
        ngram_embeddings = compute_ngram_embeddings_asymm(self.sentence_model, ngram_list)

        return {"ngram_list": ngram_list, "ngram_embeddings": ngram_embeddings, "doc_embedding": doc_embedding}

    def postprocess(self, model_outputs, top_n=5, use_mmr=False, diversity=0.5):
        ngram_list = model_outputs["ngram_list"]
        ngram_embeddings = model_outputs["ngram_embeddings"]
        doc_embedding = model_outputs["doc_embedding"]

        ngram_result = self.extract_keywords(doc_embedding, ngram_list, ngram_embeddings)

        if use_mmr:
            words = list(ngram_result.keys())
            word_embeddings = np.array([ngram_embeddings[w] for w in words])
            doc_emb_np = doc_embedding.reshape(1, -1)
            return mmr(
                doc_embedding=doc_emb_np,
                word_embeddings=word_embeddings,
                words=words,
                top_n=top_n,
                diversity=diversity,
            )

        # Default: sắp xếp theo cosine similarity
        return sorted(
            [(ngram, ngram_result[ngram]) for ngram in ngram_result],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

    def generate_ngram_list(self, doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq):
        ngram_low, ngram_high = ngram_n
        ngram_list = set()
        for n in range(ngram_low, ngram_high + 1):
            ngram_list.update(
                get_candidate_ngrams(doc_segmentised, filtered_doc_segmentised, n, self.stopwords)
            )
        # Thêm named entities
        ne_ls_segmented = [self.annotator.word_segment(ne)[0] for ne in ne_ls]
        ngram_list.update(ne_ls_segmented)
        # Loại bỏ các ngram chồng lấp
        ngram_list = remove_overlapping_ngrams(ngram_list)
        if min_freq > 1:
            ngram_list = limit_minimum_frequency(doc_segmentised, ngram_list, min_freq=min_freq)
            return ngram_list.keys()
        return ngram_list

    def extract_keywords(self, doc_embedding, ngram_list, ngram_embeddings):
        ngram_result = compute_ngram_similarity_asymm(ngram_list, ngram_embeddings, doc_embedding)
        ngram_result = remove_duplicates(ngram_result)
        return ngram_result


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    text = """
Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương Mỵ Châu – Trọng Thủy.
Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước.
Cổ Loa từng là kinh đô của nhà nước Âu Lạc thời kỳ An Dương Vương (thế kỷ III TCN) và của nước Đại Việt thời Ngô Quyền (thế kỷ X) mà thành Cổ Loa là một di tích minh chứng còn lại cho đến ngày nay.
"""
 
    print("=" * 60)
    print("  KEYBERT KEYWORD EXTRACTION – BASELINE")
    print("=" * 60)

    def _patch_transformers_compat(model):
        """Patch model cũ (pickle) để tương thích transformers mới."""
        from types import SimpleNamespace
        for module in model.modules():
            if (hasattr(module, "self") and hasattr(module, "output")
                    and not hasattr(module, "is_cross_attention")):
                module.is_cross_attention = False
            if hasattr(module, "attention_head_size") and hasattr(module, "query"):
                if not hasattr(module, "config"):
                    module.config = SimpleNamespace(_attn_implementation="eager")
                if not hasattr(module, "scaling"):
                    module.scaling = module.attention_head_size ** -0.5
                if not hasattr(module, "is_causal"):
                    module.is_causal = False
                if not hasattr(module, "layer_idx"):
                    module.layer_idx = None

    # Load NER model từ file .pt
    print("\n⏳ Đang tải NER model từ file .pt...")
    ner_model = torch.load(_NER_PT, map_location="cpu", weights_only=False)
    ner_model.eval()
    _patch_transformers_compat(ner_model)
    print("✅ NER model sẵn sàng!")

    # Khởi tạo pipeline
    kw_pipeline = KeywordExtractorPipeline(ner_model=ner_model)

    inp = {"text": text, "title": None}

    # --- Default (sorted by similarity) ---
    kws_default = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=10, use_mmr=False)

    print(f"\n{'─'*50}")
    print(f"  TOP 10 KEYPHRASE (KEYBERT – DEFAULT)")
    print(f"{'─'*50}")
    for i, (kp, score) in enumerate(kws_default, 1):
        kp_clean = kp.replace("_", " ")
        print(f"  {i:2d}. {kp_clean:<35}  {score:.4f}")
    print(f"{'─'*50}")

    # --- MMR diversity=0.8 ---
    kws_mmr = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=10, use_mmr=True, diversity=0.5)

    print(f"\n{'─'*50}")
    print(f"  TOP 10 KEYPHRASE (KEYBERT – MMR diversity=0.5)")
    print(f"{'─'*50}")
    for i, (kp, score) in enumerate(kws_mmr, 1):
        kp_clean = kp.replace("_", " ")
        print(f"  {i:2d}. {kp_clean:<35}  {score:.4f}")
    print(f"{'─'*50}")
    print("  (Score càng cao → từ khóa càng quan trọng)\n")
