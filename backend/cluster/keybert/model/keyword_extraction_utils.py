from string import punctuation
import numpy as np
import torch
import re
from operator import itemgetter
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from .named_entities import get_named_entities
from pyvi.ViTokenizer import tokenize as pyvi_tokenize

punctuation = [c for c in punctuation if c != "_"]
punctuation += [""", "–", ",", "…", """, "–"]

# bẻ câu thành từng đoạn
def sub_sentence(sentence):
    sent = []

    start_index = 0
    while start_index < len(sentence):
        idx_list = []
        for p in punctuation:
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


def compute_ngram_list(segmentised_doc, ngram_n, stopwords_ls, subsentences=True):
    if subsentences:
        output_sub_sentences = []
        for sentence in segmentised_doc:
            output_sub_sentences += sub_sentence(sentence)
    else:
        output_sub_sentences = segmentised_doc

    ngram_list = []
    for sentence in output_sub_sentences:
        sent = sentence.split() #cắt dựa vào khoảng trắng

        # windown sliding
        for i in range(len(sent) - ngram_n + 1):
            ngram = ' '.join(sent[i:i + ngram_n])
            if ngram not in ngram_list and not check_for_stopwords(ngram, stopwords_ls):
                ngram_list.append(ngram)

    #lọc ngram có số
    final_ngram_list = []
    for ngram in ngram_list:
        contains_number = False
        for char in ngram:
            if char.isnumeric():
                contains_number = True
                break
        if not contains_number:
            final_ngram_list.append(ngram)

    return final_ngram_list


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# segment văn bản
def get_segmentised_doc(nlp, rdrsegmenter, title, doc):
    segmentised_doc = rdrsegmenter.word_segment(doc)    #mỗi item là 1 câu được word segment

    if title is not None:
        segmentised_doc = rdrsegmenter.word_segment(title) + rdrsegmenter.word_segment(doc)

    ne_ls = set(get_named_entities(nlp, doc))   #entities list

    segmentised_doc_ne = [] #clear document với word_segment và ner chuẩn
    for sent in segmentised_doc:
        for ne in ne_ls:
            words = ne.split()
            pattern = r'[ _]'.join(map(re.escape, words))
            replacement = '_'.join(words)

            sent = re.sub(pattern, replacement, sent)   #tìm trong sent các pattern rồi thay bằng replacement

        segmentised_doc_ne.append(sent)
    return ne_ls, segmentised_doc_ne

# emb văn bản
def get_doc_embeddings_asymm(segmentised_doc, sentence_model, stopwords):
    embeddings = []
    weights = []

    for i, sentence in enumerate(segmentised_doc):
        sent_tokenized = pyvi_tokenize(sentence)  # encode toàn câu
        emb = sentence_model.encode([sent_tokenized], normalize_embeddings=True)[0]  # shape [dim]
        embeddings.append(emb)
        weights.append(2.0 if i == 0 else 1.0)  # title có trọng số x2

    weights = np.array(weights)
    embeddings = np.array(embeddings)  # [n, dim]
    # tính trung bình có trọng số theo từng cột
    doc_emb = np.average(embeddings, axis=0, weights=weights)
    return doc_emb  # numpy array [dim]

# emb ngram
def compute_ngram_embeddings_asymm(sentence_model, ngram_list):
    ngram_embeddings = {}

    for ngram in ngram_list:
        ngram_copy = ngram.lower() if ngram.isupper() else ngram
        ngram_tokenized = pyvi_tokenize(ngram_copy)
        emb = sentence_model.encode([ngram_tokenized], normalize_embeddings=True)[0]
        ngram_embeddings[ngram] = emb

    return ngram_embeddings

# cosine similarity ngram emb - doc emb
def compute_ngram_similarity_asymm(ngram_list, ngram_embeddings, doc_embedding):

    ngram_similarity_dict = {}

    for ngram in ngram_list:
        a = ngram_embeddings[ngram]
        b = doc_embedding
        similarity_score = cosine_similarity(a, b)
        ngram_similarity_dict[ngram] = float(similarity_score)

    return ngram_similarity_dict

def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int = 5,
    diversity: float = 0.8,
) -> List[Tuple[str, float]]:

    if len(words) == 0:
        return []

    # kiểm tra mảng 2 chiều (cho sklearn)
    if doc_embedding.ndim == 1:
        doc_embedding = doc_embedding.reshape(1, -1)

    # tính simi giữa từ - từ, từ - văn bản
    word_doc_similarity = sklearn_cosine_similarity(word_embeddings, doc_embedding)  # [n, 1]
    word_similarity = sklearn_cosine_similarity(word_embeddings)                     # [n, n]

    # top1: simi với văn bản nhất
    keywords_idx = [int(np.argmax(word_doc_similarity))]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        if not candidates_idx:
            break

        # lấy simi còn lại - văn bản tìm top 2,3,4....
        candidate_similarities = word_doc_similarity[candidates_idx, :]  # [c, 1]

        # simi lớn nhất giữa hiện tại và danh sách top đã chọn trên matrix từ-từ, tính theo hàng
        target_similarities = np.max(
            word_similarity[candidates_idx][:, keywords_idx], axis=1
        )

        # MMR score: đa dạng x liên quan
        mmr_scores = (
            (1 - diversity) * candidate_similarities
            - diversity * target_similarities.reshape(-1, 1)
        )

        # index của candidate hiện tại
        mmr_idx = candidates_idx[int(np.argmax(mmr_scores))]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # lấy từ khóa từ index
    keywords = [
        (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
        for idx in keywords_idx
    ]
    keywords = sorted(keywords, key=itemgetter(1), reverse=True)    # xếp điểm số giảm dần
    return keywords

# xóa các n-gram giống nhau
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

# lọc từ dựa trên Từ loại
def compute_filtered_text(annotator, title, text):
    annotated = annotator.annotate_text(text)
    if title is not None:
        annotated = annotator.annotate_text(title + '. ' + text)
    filtered_sentences = []
    keep_tags = ['N', 'Np', 'V', 'Nc']
    for key in annotated.keys():
        sent = ' '.join([dict_['wordForm'] for dict_ in annotated[key] if dict_['posTag'] in keep_tags])
        filtered_sentences.append(sent)
    return filtered_sentences

# đảm bảo các ngram thỏa mãn: lọc post và thực sự đi cùng với nhau thành cụm
def get_candidate_ngrams(segmentised_doc, filtered_segmentised_doc, ngram_n, stopwords_ls):
    # danh sách câu word_segment chưa được lọc
    actual_ngram_list = compute_ngram_list(segmentised_doc, ngram_n, stopwords_ls, subsentences=True)

    # danh sách câu word_segment và lọc POS
    filtered_ngram_list = compute_ngram_list(filtered_segmentised_doc, ngram_n, stopwords_ls,
                                             subsentences=False)

    # lấy phần giao
    candidate_ngram = [ngram for ngram in filtered_ngram_list if ngram in actual_ngram_list]
    return candidate_ngram

# đếm tần suất bằng sliding windown
def limit_minimum_frequency(doc_segmentised, ngram_list, min_freq=1):
    ngram_dict_freq = {}
    for ngram in ngram_list:
        ngram_n = len(ngram.split())
        count = 0
        for sentence in doc_segmentised:
            sent = sentence.split()
            for i in range(len(sent) - ngram_n + 1):
                pair = ' '.join(sent[i:i + ngram_n])
                if pair == ngram:
                    count += 1
        if count >= min_freq:
            ngram_dict_freq[ngram] = count

    return ngram_dict_freq


def remove_overlapping_ngrams(ngram_list):
    to_remove = set()
    for ngram1 in ngram_list:
        for ngram2 in ngram_list:
            if len(ngram1.split()) > len(ngram2.split()) and (ngram1.startswith(ngram2) or ngram1.endswith(ngram2)):
                to_remove.add(ngram2)

    for kw in to_remove:
        ngram_list.remove(kw)
    return ngram_list


# def get_doc_embeddings(segmentised_doc, tokenizer, phobert, stopwords):
#     doc_embedding = torch.zeros(size=(len(segmentised_doc), 768))

#     for i, sentence in enumerate(segmentised_doc):
#         sent_removed_stopwords = ' '.join([word for word in sentence.split() if word not in stopwords])

#         sentence_embedding = tokenizer.encode(sent_removed_stopwords, truncation=True, max_length=256)
#         input_ids = torch.tensor([sentence_embedding])
#         with torch.no_grad():
#             features = phobert(input_ids)

#         if i == 0:
#             doc_embedding[i, :] = 2 * features.pooler_output.flatten()
#         else:
#             doc_embedding[i, :] = features.pooler_output.flatten()

#     return torch.mean(doc_embedding, axis=0)

# ----------------------------------------------------------------------------------------

# def diversify_result_kmeans(ngram_result, ngram_embeddings, top_n=5):
#     best_ngrams = sorted(ngram_result, key=ngram_result.get, reverse=True)[:top_n * 4]
#     best_ngram_embeddings = np.array([ngram_embeddings[ngram] for ngram in best_ngrams]).squeeze()
#     vote = {}

#     for niter in range(100):
#         kmeans = KMeans(n_clusters=top_n, init='k-means++', random_state=niter * 2, n_init="auto").fit(
#             best_ngram_embeddings)
#         kmeans_result = kmeans.labels_

#         res = {}
#         for i in range(len(kmeans_result)):
#             if kmeans_result[i] not in res:
#                 res[kmeans_result[i]] = []
#             res[kmeans_result[i]].append((best_ngrams[i], ngram_result[best_ngrams[i]]))

#         final_result = [res[k][0] for k in res]
#         for keyword in final_result:
#             if keyword not in vote:
#                 vote[keyword] = 0
#             vote[keyword] += 1

#     diversify_result_ls = sorted(vote, key=vote.get, reverse=True)

#     return diversify_result_ls[:top_n]


# ----------------------------------------------------------------------------------------

# def compute_ngram_embeddings(tokenizer, phobert, ngram_list):
#     ngram_embeddings = {}

#     for ngram in ngram_list:
#         ngram_copy = ngram
#         if ngram.isupper():
#             ngram_copy = ngram.lower()
#         word_embedding = tokenizer.encode(ngram_copy, truncation=True, max_length=256)
#         input_ids = torch.tensor([word_embedding])
#         with torch.no_grad():
#             word_features = phobert(input_ids)

#         ngram_embeddings[ngram] = word_features.pooler_output
#     return ngram_embeddings

# ----------------------------------------------------------------------------------------

# def compute_ngram_similarity(ngram_list, ngram_embeddings, doc_embedding):
#     ngram_similarity_dict = {}

#     for ngram in ngram_list:
#         a = ngram_embeddings[ngram].flatten().detach().cpu().numpy()
#         b = doc_embedding.flatten().detach().cpu().numpy()
#         similarity_score = cosine_similarity(a, b).flatten()[0]
#         ngram_similarity_dict[ngram] = similarity_score

#     return ngram_similarity_dict