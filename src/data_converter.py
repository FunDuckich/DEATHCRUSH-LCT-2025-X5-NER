from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
except Exception as e:
    print(f"Ошибка при загрузке токенизатора: {e}")
    tokenizer = None


def _tokenize_and_filter(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    if tokenizer is None:
        raise RuntimeError("Токенизатор не был инициализирован.")
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    filtered_tokens, filtered_offsets = [], []
    for token, offset in zip(tokens, offsets):
        if offset != (0, 0):
            filtered_tokens.append(token)
            filtered_offsets.append(offset)
    return filtered_tokens, filtered_offsets


def sanitize_annotations(text: str, annotations: list) -> tuple[str, list]:
    new_annotations = []
    for start, end, label in annotations:
        entity_text = text[start:end]
        stripped_entity_text_right = entity_text.rstrip()
        new_end = start + len(stripped_entity_text_right)
        stripped_entity_text_left = stripped_entity_text_right.lstrip()
        new_start = new_end - len(stripped_entity_text_left)
        if new_start < new_end:
            new_annotations.append((new_start, new_end, label))
    return text, new_annotations


def indices_to_bio(text: str, annotations: list) -> tuple[list[str], list[str]]:
    filtered_tokens, filtered_offsets = _tokenize_and_filter(text)
    bio_tags = ["O"] * len(filtered_tokens)
    for ann_start, ann_end, ann_label in annotations:
        if ann_label == "O":
            continue
        clean_label = ann_label.split("-")[-1]
        is_first_token_in_entity = True
        for i, (tok_start, tok_end) in enumerate(filtered_offsets):
            if max(tok_start, ann_start) < min(tok_end, ann_end):
                prefix = "B-" if is_first_token_in_entity else "I-"
                bio_tags[i] = f"{prefix}{clean_label}"
                is_first_token_in_entity = False
    return filtered_tokens, bio_tags


def bio_to_indices(text: str, bio_tags: list) -> list:
    filtered_tokens, filtered_offsets = _tokenize_and_filter(text)
    if len(bio_tags) != len(filtered_tokens):
        raise ValueError("Количество BIO-тегов не совпадает с количеством отфильтрованных токенов")

    annotations = []
    for i, tag in enumerate(bio_tags):
        if tag != "O":
            start_char, end_char = filtered_offsets[i]
            annotations.append((start_char, end_char, tag))

    return annotations
