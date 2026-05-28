import py_vncorenlp
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import os
import numpy as np

from .model.keyword_extraction_utils import (
    get_doc_embeddings_asymm,
    compute_ngram_embeddings_asymm,
    compute_ngram_similarity_asymm,
    get_segmentised_doc,
    compute_filtered_text,
    get_candidate_ngrams,
    remove_overlapping_ngrams,
    limit_minimum_frequency,
    remove_duplicates,
    mmr,
    # diversify_result_kmeans,
)
from .model.process_text import process_text_pipeline

_KEYBERT_DIR     = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT    = os.path.abspath(os.path.join(_KEYBERT_DIR, "..", "..", ".."))
_PRETRAINED_DIR  = os.path.join(_PROJECT_ROOT, "pretrained-models")

VNCORENLP_DIR    = os.path.join(_PRETRAINED_DIR, "vncorenlp")
ASYMM_MODEL_PATH = os.path.join(_PRETRAINED_DIR, "asymmetric_emb")
NER_TOKENIZER_DIR = os.path.join(_PRETRAINED_DIR, "ner-tokenizer")
NER_PT            = os.path.join(_PRETRAINED_DIR, "ner-vietnamese-electra-base.pt")
STOPWORDS_PATH   = os.path.join(_KEYBERT_DIR, "vietnamese-stopwords-dash.txt")


class KeywordExtractorPipeline:
    # load model từ local vào RAM
    def __init__(self, ner_model, vncorenlp_instance=None, asymm_model_path=None):
        # ── VnCoreNLP ──
        if vncorenlp_instance is not None:
            self.annotator = vncorenlp_instance
        else:
            self.annotator = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos"],
                save_dir=VNCORENLP_DIR,
            )

        # ── Asymmetric SentenceTransformer ──
        model_path = asymm_model_path if asymm_model_path is not None else ASYMM_MODEL_PATH
        self.sentence_model = SentenceTransformer(model_path)
        print("✅ Asymmetric Embedding sẵn sàng!\n")

        # ── NER pipeline ──
        print("⏳ [5/5] Đang khởi tạo NER pipeline…")
        ner_tokenizer = AutoTokenizer.from_pretrained(NER_TOKENIZER_DIR)
        self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        print("✅ NER pipeline sẵn sàng!\n")

        # ── Stopwords ──
        with open(STOPWORDS_PATH) as f:
            self.stopwords = [w.strip() for w in f.readlines()]

    # script chạy 3 việc: clean - process - score
    def __call__(self, inputs, ngram_n=(1, 3), min_freq=1, top_n=5,
                 use_mmr=True, diversity=0.8):
        model_inputs = self.preprocess(inputs)
        model_outputs = self._forward(model_inputs, ngram_n=ngram_n, min_freq=min_freq)
        return self.postprocess(model_outputs, top_n=top_n,
                                use_mmr=use_mmr, diversity=diversity)
    # xử lý clean text
    def preprocess(self, inputs):
        title = None
        if inputs.get('title'):
            title = process_text_pipeline(inputs['title'])
        text = process_text_pipeline(inputs['text'])
        return {"text": text, "title": title}

    # doc_emb, ngram_list, ngram_emb
    def _forward(self, model_inputs, ngram_n=(1, 3), min_freq=1):
        text = model_inputs['text']
        title = model_inputs['title']

        # word_segment + ner
        ne_ls, doc_segmentised = get_segmentised_doc(self.ner_pipeline, self.annotator, title, text)
        
        # lọc POS từng câu đã segment
        filtered_doc_segmentised = compute_filtered_text(self.annotator, title, text)

        # Doc embedding dùng asymmetric model
        doc_embedding = get_doc_embeddings_asymm(filtered_doc_segmentised, self.sentence_model, self.stopwords)

        # Danh sách n-gram ứng viên
        ngram_list = self.generate_ngram_list(doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq)

        # Ngram embeddings dùng asymmetric model
        ngram_embeddings = compute_ngram_embeddings_asymm(self.sentence_model, ngram_list)

        return {"ngram_list": ngram_list, "ngram_embeddings": ngram_embeddings, "doc_embedding": doc_embedding}

    # top n keyword bằng MMR
    def postprocess(self, model_outputs, top_n=5, use_mmr=False, diversity=0.8):
        ngram_list = model_outputs['ngram_list']
        ngram_embeddings = model_outputs['ngram_embeddings']
        doc_embedding = model_outputs['doc_embedding']

        ngram_result = self.extract_keywords(doc_embedding, ngram_list, ngram_embeddings)

        # MMR: sắp xếp theo mmr
        if use_mmr:
            words = list(ngram_result.keys())
            word_embeddings = np.array([ngram_embeddings[w] for w in words])  # [n, dim]
            doc_emb_np = doc_embedding.reshape(1, -1)                          # chuyển ma trận cột -> ma trận hàng
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
            key=lambda x: x[1], # index 1: cosine similarity
            reverse=True,
        )[:top_n]

    # tạo candidate 1,2,..n gram 
    def generate_ngram_list(self, doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq):
        ngram_low, ngram_high = ngram_n

        ngram_list = set()
        for n in range(ngram_low, ngram_high + 1):
            ngram_list.update(get_candidate_ngrams(doc_segmentised, filtered_doc_segmentised, n, self.stopwords))

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


if __name__ == "__main__":
    import torch
    from types import SimpleNamespace

    def _patch_transformers_compat(model):
        """Patch model cũ (pickle) để tương thích transformers mới."""
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

    print(f"Loading NER model from local: {NER_PT}")
    ner_model = torch.load(NER_PT, map_location="cpu", weights_only=False)
    ner_model.eval()
    _patch_transformers_compat(ner_model)

    kw_pipeline = KeywordExtractorPipeline(ner_model=ner_model)
 
   
    inp = {
        "title": "Cổ Loa",
        "text": """
       Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương Mỵ Châu – Trọng Thủy.
       Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước.
       Cổ Loa từng là kinh đô của nhà nước Âu Lạc thời kỳ An Dương Vương (thế kỷ III TCN) và của nước Đại Việt thời Ngô Quyền (thế kỷ X) mà thành Cổ Loa là một di tích minh chứng còn lại cho đến ngày nay.
       """
    }

    # --- Không dùng MMR (mặc định) ---
    kws_default = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=5, use_mmr=False)
    print("\nKeywords (default – sorted by similarity):")
    for kw, score in kws_default:
        print(f"  - {kw}: {score:.4f}")

    # --- Dùng MMR với diversity=0.8 ---
    kws_mmr = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=10, use_mmr=True, diversity=0.5)
    print("\nKeywords (MMR diversity=0.5):")
    for kw, score in kws_mmr:
        print(f"  - {kw}: {score:.4f}")
    print()