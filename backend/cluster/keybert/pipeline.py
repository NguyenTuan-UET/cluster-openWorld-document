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
    diversify_result_kmeans,
)
from .model.process_text import process_text_pipeline

_keybert_dir = os.path.dirname(os.path.realpath(__file__))
_pretrained_dir = os.path.join(_keybert_dir, "..", "..", "..", "pretrained-models")
dir_path = _keybert_dir

ASYMM_MODEL_PATH = os.path.join(_pretrained_dir, "asymmetric_emb")


class KeywordExtractorPipeline:
    def __init__(self, ner_model, vncorenlp_instance=None, asymm_model_path=None):
        # ── VnCoreNLP ──
        if vncorenlp_instance is not None:
            self.annotator = vncorenlp_instance
        else:
            self.annotator = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos"],
                save_dir=os.path.join(_pretrained_dir, "vncorenlp"),
            )

        # ── Asymmetric SentenceTransformer ──
        model_path = asymm_model_path if asymm_model_path is not None else ASYMM_MODEL_PATH
        self.sentence_model = SentenceTransformer(model_path)
        print("✅ Asymmetric Embedding sẵn sàng!\n")

        # ── NER pipeline ──
        print("⏳ [5/5] Đang khởi tạo NER pipeline…")
        ner_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        print("✅ NER pipeline sẵn sàng!\n")

        # ── Stopwords ──
        stopwords_file_path = os.path.join(dir_path, "vietnamese-stopwords-dash.txt")
        with open(stopwords_file_path) as f:
            self.stopwords = [w.strip() for w in f.readlines()]

    def __call__(self, inputs, ngram_n=(1, 3), min_freq=1, top_n=5,
                 use_mmr=False, use_kmeans=False, diversity=0.8):
        """Gọi pipeline giống giao diện cũ."""
        model_inputs = self.preprocess(inputs)
        model_outputs = self._forward(model_inputs, ngram_n=ngram_n, min_freq=min_freq)
        return self.postprocess(model_outputs, top_n=top_n,
                                use_mmr=use_mmr, use_kmeans=use_kmeans, diversity=diversity)

    def preprocess(self, inputs):
        title = None
        if inputs.get('title'):
            title = process_text_pipeline(inputs['title'])
        text = process_text_pipeline(inputs['text'])
        return {"text": text, "title": title}

    def _forward(self, model_inputs, ngram_n=(1, 3), min_freq=1):
        text = model_inputs['text']
        title = model_inputs['title']

        # Phân đoạn từ + nhận diện thực thể tên
        ne_ls, doc_segmentised = get_segmentised_doc(self.ner_pipeline, self.annotator, title, text)
        filtered_doc_segmentised = compute_filtered_text(self.annotator, title, text)

        # Doc embedding dùng asymmetric model
        doc_embedding = get_doc_embeddings_asymm(filtered_doc_segmentised, self.sentence_model, self.stopwords)

        # Danh sách n-gram ứng viên
        ngram_list = self.generate_ngram_list(doc_segmentised, filtered_doc_segmentised, ne_ls, ngram_n, min_freq)

        # Ngram embeddings dùng asymmetric model
        ngram_embeddings = compute_ngram_embeddings_asymm(self.sentence_model, ngram_list)

        return {"ngram_list": ngram_list, "ngram_embeddings": ngram_embeddings, "doc_embedding": doc_embedding}

    def postprocess(self, model_outputs, top_n=5, use_mmr=False, use_kmeans=False, diversity=0.8):
        ngram_list = model_outputs['ngram_list']
        ngram_embeddings = model_outputs['ngram_embeddings']
        doc_embedding = model_outputs['doc_embedding']

        ngram_result = self.extract_keywords(doc_embedding, ngram_list, ngram_embeddings)

        if use_mmr:
            words = list(ngram_result.keys())
            word_embeddings = np.array([ngram_embeddings[w] for w in words])  # [n, dim]
            doc_emb_np = doc_embedding.reshape(1, -1)                          # [1, dim]
            return mmr(
                doc_embedding=doc_emb_np,
                word_embeddings=word_embeddings,
                words=words,
                top_n=top_n,
                diversity=diversity,
            )

        if use_kmeans:
            return diversify_result_kmeans(ngram_result, ngram_embeddings, top_n=top_n)

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
    from transformers import AutoModelForTokenClassification

    print("Loading NER model from HuggingFace...")
    ner_model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    ner_model.eval()

    kw_pipeline = KeywordExtractorPipeline(ner_model=ner_model)

    text_file_path = os.path.join(dir_path, "input.txt")
    with open(text_file_path, 'r') as f:
        text = ' '.join([ln.strip() for ln in f.readlines()])

    inp = {"text": text, "title": None}

    # --- Không dùng MMR (mặc định) ---
    kws_default = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=5, use_mmr=False)
    print("\nKeywords (default – sorted by similarity):")
    for kw, score in kws_default:
        print(f"  - {kw}: {score:.4f}")

    # --- Dùng MMR với diversity=0.8 ---
    kws_mmr = kw_pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=5, use_mmr=True, diversity=0.8)
    print("\nKeywords (MMR diversity=0.8):")
    for kw, score in kws_mmr:
        print(f"  - {kw}: {score:.4f}")
    print()