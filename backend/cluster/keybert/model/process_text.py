from string import punctuation

# xử lý đầu vào: dấu câu, khoảng trắng
def process_text_pipeline(text):
    full_text_processed = text.strip()

    while '\n\n' in full_text_processed:
        full_text_processed = full_text_processed.replace('\n\n', '\n')

    full_text_processed = process_sticking_sentences(full_text_processed)

    return full_text_processed


def process_sticking_sentences(full_text):
    for i in range(len(full_text) - 1):
        c1 = full_text[i]
        c2 = full_text[i + 1]

        # chữ in hoa ngay sau dấu câu -> thêm dấu cách
        if c1 in punctuation and c2.isalpha() and c2.isupper():
            before = full_text[:i + 1]
            after = full_text[i + 1:]

            full_text = before + " " + after

        # chữ thường ngay sau chữ Hoa -> thêm dấu chấm cách
        if c1.isalpha() and c1.islower() and c2.isalpha() and c2.isupper():
            before = full_text[:i + 1]
            after = full_text[i + 1:]

            full_text = before + ". " + after
    return full_text