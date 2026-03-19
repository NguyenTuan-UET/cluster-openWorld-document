from string import punctuation


def process_text_pipeline(text):
    full_text_processed = replace_all(text.strip())

    while '\n\n' in full_text_processed:
        full_text_processed = full_text_processed.replace('\n\n', '\n')

    full_text_processed = process_sticking_sentences(full_text_processed)

    while '  ' in full_text_processed:
        full_text_processed = full_text_processed.replace('  ', ' ')
    return full_text_processed


def replace_all(text):
    dict_map = {
        "òa": "oà",
        "Òa": "Oà",
        "ÒA": "OÀ",
        "óa": "oá",
        "Óa": "Oá",
        "ÓA": "OÁ",
        "ỏa": "oả",
        "Ỏa": "Oả",
        "ỎA": "OẢ",
        "õa": "oã",
        "Õa": "Oã",
        "ÕA": "OÃ",
        "ọa": "oạ",
        "Ọa": "Oạ",
        "ỌA": "OẠ",
        "òe": "oè",
        "Òe": "Oè",
        "ÒE": "OÈ",
        "óe": "oé",
        "Óe": "Oé",
        "ÓE": "OÉ",
        "ỏe": "oẻ",
        "Ỏe": "Oẻ",
        "ỎE": "OẺ",
        "õe": "oẽ",
        "Õe": "Oẽ",
        "ÕE": "OẼ",
        "ọe": "oẹ",
        "Ọe": "Oẹ",
        "ỌE": "OẸ",
        "ùy": "uỳ",
        "Ùy": "Uỳ",
        "ÙY": "UỲ",
        "úy": "uý",
        "Úy": "Uý",
        "ÚY": "UÝ",
        "ủy": "uỷ",
        "Ủy": "Uỷ",
        "ỦY": "UỶ",
        "ũy": "uỹ",
        "Ũy": "Uỹ",
        "ŨY": "UỸ",
        "ụy": "uỵ",
        "Ụy": "Uỵ",
        "ỤY": "UỴ",
        "\xa0": " ",
        "…": "...",
        "''": '"',
        "&#34;": '"',
        "&#39;": "'",
        "H'Mông": "Hmông",
        "H'mông": "Hmông",
        "H’mông": "Hmông",
        "H’Mông": "Hmông",
        "H’MÔNG": "Hmông",
        "M'Nông": "Mnông",
        "M'nông": "Mnông",
        "M'NÔNG": "Mnông",
        "M’Nông": "Mnông",
        "M’NÔNG": "Mnông",
        '\u200b\u200b': ""
    }
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


def process_sticking_sentences(full_text):
    for i in range(len(full_text) - 1):
        c1 = full_text[i]
        c2 = full_text[i + 1]

        # 'end of sentence.Start'
        if c1 in punctuation and c2.isalpha() and c2.isupper():
            before = full_text[:i + 1]
            after = full_text[i + 1:]

            full_text = before + " " + after

        # 'end of sentenceStart'
        if c1.isalpha() and c1.islower() and c2.isalpha() and c2.isupper():
            before = full_text[:i + 1]
            after = full_text[i + 1:]

            full_text = before + ". " + after
    return full_text
