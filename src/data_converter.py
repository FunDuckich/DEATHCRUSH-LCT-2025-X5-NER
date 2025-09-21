from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
except Exception as e:
    print(f"Ошибка при загрузке токенизатора: {e}")
    tokenizer = None


def _tokenize_and_filter(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    if tokenizer is None:
        raise RuntimeError("Токенизатор не был инициализирован.")
    # Явно указываем добавление специальных токенов и смещения
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    filtered_tokens: list[str] = []
    filtered_offsets: list[tuple[int, int]] = []

    # Фильтрация спец. токенов и "пустых" смещений
    for token, offset in zip(tokens, offsets):
        if offset is None:
            continue
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            if not (offset[0] == 0 and offset[1] == 0):
                filtered_tokens.append(token)
                filtered_offsets.append((int(offset[0]), int(offset[1])))
        # Иначе пропускаем
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
    """
    Преобразует BIO-теги в список аннотаций (start, end, label) на основе символьных индексов.
    Сохраняет исходный B/I префикс первого токена сущности.
    """
    if tokenizer is None:
        raise RuntimeError("Токенизатор не был инициализирован.")

    # ... (весь код токенизации и выравнивания остается тем же) ...
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    all_offsets = encoding["offset_mapping"]

    valid_mask = [off != (0, 0) for off in all_offsets]

    filtered_offsets = [tuple(map(int, off)) for off, m in zip(all_offsets, valid_mask) if m]

    if len(bio_tags) == len(filtered_offsets):
        tags_aligned = list(bio_tags)
    elif len(bio_tags) == len(all_tokens):
        tags_aligned = [tag for tag, m in zip(bio_tags, valid_mask) if m]
    else:
        # ... (код обработки ошибок остается) ...
        raise ValueError(
            "Количество BIO-тегов не совпадает..."
        )

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---

    annotations: list[tuple[int, int, str]] = []

    current_entity_tags = []
    current_entity_indices = []

    for i, tag in enumerate(tags_aligned):
        if tag.startswith("B-"):
            # Если уже есть открытая сущность, закрываем ее
            if current_entity_tags:
                start_char = filtered_offsets[current_entity_indices[0]][0]
                end_char = filtered_offsets[current_entity_indices[-1]][1]
                # Используем первый тег как основной
                annotations.append((start_char, end_char, current_entity_tags[0]))

            # Начинаем новую сущность
            current_entity_tags = [tag]
            current_entity_indices = [i]

        elif tag.startswith("I-"):
            # Если есть открытая сущность и теги совпадают, продолжаем
            if current_entity_tags and tag[2:] == current_entity_tags[0][2:]:
                current_entity_tags.append(tag)
                current_entity_indices.append(i)
            else:
                # Некорректный I-тег или нет открытой сущности, закрываем старую и начинаем новую
                if current_entity_tags:
                    start_char = filtered_offsets[current_entity_indices[0]][0]
                    end_char = filtered_offsets[current_entity_indices[-1]][1]
                    annotations.append((start_char, end_char, current_entity_tags[0]))

                # Начинаем новую сущность с этого I-тега (восстановление после ошибки)
                current_entity_tags = [f"B-{tag[2:]}"]  # Превращаем I- в B-
                current_entity_indices = [i]

        else:  # Тег 'O' или другой
            # Закрываем любую открытую сущность
            if current_entity_tags:
                start_char = filtered_offsets[current_entity_indices[0]][0]
                end_char = filtered_offsets[current_entity_indices[-1]][1]
                annotations.append((start_char, end_char, current_entity_tags[0]))

            current_entity_tags = []
            current_entity_indices = []

    # Не забываем закрыть последнюю сущность, если она была
    if current_entity_tags:
        start_char = filtered_offsets[current_entity_indices[0]][0]
        end_char = filtered_offsets[current_entity_indices[-1]][1]
        annotations.append((start_char, end_char, current_entity_tags[0]))

    return annotations
