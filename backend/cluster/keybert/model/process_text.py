from string import punctuation


def process_text_pipeline(text):
    full_text_processed = text.strip()

    while '\n\n' in full_text_processed:
        full_text_processed = full_text_processed.replace('\n\n', '\n')

    full_text_processed = process_sticking_sentences(full_text_processed)

    while '  ' in full_text_processed:
        full_text_processed = full_text_processed.replace('  ', ' ')
    return full_text_processed


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
