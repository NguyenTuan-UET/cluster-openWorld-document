from underthesea import sent_tokenize

# check chuỗi con
def substring(w, ls):
    for w2 in ls:
        if w != w2 and w in w2:
            return True
    return False

# ghép các token liên tiếp thành 1 cụm từ 
def get_ner_phrases(sent_ner_result):
    ner_list = []
    current_ner = [sent_ner_result[0]["word"]]
    current_idx = sent_ner_result[0]["index"]
    for i in range(1, len(sent_ner_result)):
        if sent_ner_result[i]["index"] == current_idx + 1:
            current_ner.append(sent_ner_result[i]["word"])
        else:
            ner_list.append((' '.join(current_ner), sent_ner_result[i - 1]['entity']))
            current_ner = [sent_ner_result[i]["word"]]

        current_idx = sent_ner_result[i]["index"]

    ner_list.append((' '.join(current_ner), sent_ner_result[len(sent_ner_result) - 1]['entity']))
    return ner_list

def get_named_entities(nlp, doc):
    ner_lists = []
    for sent in sent_tokenize(doc): # tách câu
        sent_ner_result = nlp(sent) # nhận dạng thực thể
        if len(sent_ner_result) > 0:
            ner_lists += get_ner_phrases(sent_ner_result)

    # print(ner_lists)

    ner_list_non_dup = []
    for (entity, ner_type) in ner_lists:
        if entity not in ner_list_non_dup and ner_type.startswith('I'):
            ner_list_non_dup.append(entity)

    ner_list_final = [w for w in ner_list_non_dup if not substring(w, ner_list_non_dup)]
    return ner_list_final