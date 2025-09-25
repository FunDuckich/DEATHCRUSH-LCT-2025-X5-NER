from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
except Exception as e:
    print(f"Ошибка при загрузке токенизатора: {e}")
    tokenizer = None


def _tokenize_and_filter(text: str):
    """
    Возвращает (tokens, offsets) с add_special_tokens=False и без нулевых offset'ов.
    offsets — список кортежей (start_char, end_char) (end exclusive).
    """
    if tokenizer is None:
        raise RuntimeError("Токенизатор не инициализирован")

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoding.tokens()
    offsets = encoding["offset_mapping"]

    filtered_tokens = []
    filtered_offsets = []
    for tok, off in zip(tokens, offsets):
        # Иногда off == (0,0) для некоторых токенов (или None) — пропускаем
        if off is None:
            continue
        if isinstance(off, (list, tuple)) and len(off) == 2:
            if not (off[0] == 0 and off[1] == 0):
                filtered_tokens.append(tok)
                filtered_offsets.append((int(off[0]), int(off[1])))
    return filtered_tokens, filtered_offsets


def sanitize_annotations(text: str, annotations: list):
    """
    Обрезает пробелы по краям у аннотаций, возвращает (text, new_annotations).
    Ожидает аннотации в символах (start,end,label).
    """
    new_annotations = []
    for start, end, label in annotations:
        # защититься от неверных индексов
        start = max(0, int(start))
        end = max(0, int(end))
        start = min(len(text), start)
        end = min(len(text), end)
        if start >= end:
            continue
        entity_text = text[start:end]
        stripped_right = entity_text.rstrip()
        new_end = start + len(stripped_right)
        stripped_both = stripped_right.lstrip()
        new_start = new_end - len(stripped_both)
        if new_start < new_end:
            new_annotations.append((new_start, new_end, label))
    return text, new_annotations


def indices_to_bio(text: str, annotations: list):
    filtered_tokens, filtered_offsets = _tokenize_and_filter(text)
    bio_tags = ["O"] * len(filtered_tokens)

    anns_sorted = sorted(annotations, key=lambda x: x[0])
    for ann_start, ann_end, ann_label in anns_sorted:
        if ann_label == "O":
            continue
        clean_label = ann_label.split("-")[-1]
        first = True
        for i, (tok_start, tok_end) in enumerate(filtered_offsets):
            if max(tok_start, ann_start) < min(tok_end, ann_end):
                prefix = "B-" if first else "I-"
                if bio_tags[i] == "O":
                    bio_tags[i] = f"{prefix}{clean_label}"
                else:
                    pass
                first = False
            if tok_start >= ann_end:
                break
    return filtered_tokens, bio_tags


def bio_to_indices(text: str, bio_tags: list, include_O: bool = True):
    if tokenizer is None:
        raise RuntimeError("Токенизатор не инициализирован")

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoding.tokens()
    offsets_all = encoding["offset_mapping"]

    offsets = []
    for off in offsets_all:
        if off is None:
            continue
        if isinstance(off, (list, tuple)) and not (off[0] == 0 and off[1] == 0):
            offsets.append((int(off[0]), int(off[1])))

    if len(offsets) != len(bio_tags):
        if len(offsets) > len(bio_tags):
            offsets = offsets[:len(bio_tags)]
        else:
            raise ValueError(
                f"Mismatch: filtered offsets ({len(offsets)}) < bio_tags ({len(bio_tags)}). "
                "Вам нужно убедиться, что bio_tags соответствует токенизации tokenizer(text, add_special_tokens=False)."
            )

    annotations = []
    i = 0
    n = len(bio_tags)
    while i < n:
        tag = bio_tags[i]
        start_char, end_char = offsets[i]
        if tag == "O" or not isinstance(tag, str):
            if include_O:
                annotations.append((start_char, end_char, "O"))
            i += 1
            continue

        if tag.startswith("B-"):
            label = tag[2:]
            start_token_idx = i
            end_token_idx = i
            j = i + 1
            while j < n and bio_tags[j] == f"I-{label}":
                end_token_idx = j
                j += 1
            start_char = offsets[start_token_idx][0]
            end_char = offsets[end_token_idx][1]
            annotations.append((start_char, end_char, f"B-{label}"))
            i = j
            continue

        if tag.startswith("I-"):
            label = tag[2:]
            start_token_idx = i
            end_token_idx = i
            j = i + 1
            while j < n and bio_tags[j] == f"I-{label}":
                end_token_idx = j
                j += 1
            start_char = offsets[start_token_idx][0]
            end_char = offsets[end_token_idx][1]
            annotations.append((start_char, end_char, f"B-{label}"))
            i = j
            continue

        i += 1

    return sorted(annotations, key=lambda x: x[0])
