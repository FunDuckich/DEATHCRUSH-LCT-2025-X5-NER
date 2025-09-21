import re

VOLUME_UNITS_LIST = [
    "л", "литр", "литра", "литров", "кг", "килограмм", "килограмма",
    "г", "гр", "грамм", "мл", "миллилитров", "шт", "штук", "уп", "упак",
    "l", "ml", "g", "kg"
]
PERCENT_UNITS_LIST = [
    "%", "проц", "процент", "процентов", "жирн", "жирность"
]

volume_units_pattern = r"(?:" + "|".join(VOLUME_UNITS_LIST) + r")"
percent_units_pattern = r"(?:" + "|".join(PERCENT_UNITS_LIST) + r")"

num_simple = r"\d+[\.,]?\d*"
num_range = r"\d+[\.,]?\d*\s*-\s*\d+[\.,]?\d*"
num_pattern_any = f"(?:{num_range}|{num_simple})"

PATTERN_PERCENT_FULL = re.compile(f"({num_pattern_any}\\s*{percent_units_pattern})", re.IGNORECASE)
PATTERN_VOLUME_FULL = re.compile(f"({num_pattern_any}\\s*{volume_units_pattern})", re.IGNORECASE)
PATTERN_NUMBER_ONLY = re.compile(f"({num_pattern_any})")

PERCENT_CONTEXT_WORDS = {"сливки", "творог", "сметана", "сливочное", "молоко", "кефир", "масло", "жир", "жирности"}
VOLUME_CONTEXT_WORDS = {"вода", "сок", "бутылка", "пачка", "упаковка", "объем", "объём"}


def extract_volume_percent(text: str) -> list:
    found_entities = []
    covered_indices = [False] * len(text)

    for match in re.finditer(PATTERN_PERCENT_FULL, text):
        start, end = match.span(1)
        if not any(covered_indices[i] for i in range(start, end)):
            found_entities.append((start, end, "B-PERCENT"))
            for i in range(start, end): covered_indices[i] = True

    for match in re.finditer(PATTERN_VOLUME_FULL, text):
        full_start, full_end = match.span(1)
        if not any(covered_indices[i] for i in range(full_start, full_end)):
            full_text = match.group(1)
            number_match = re.search(PATTERN_NUMBER_ONLY, full_text)
            if number_match:
                found_entities.append((full_start, full_end, "B-VOLUME"))
                for i in range(full_start, full_end): covered_indices[i] = True

    for match in re.finditer(r"\b(" + num_simple + r")\b", text):
        start, end = match.span(1)
        if not any(covered_indices[i] for i in range(start, end)):
            preceding_text = text[:start].strip()
            if not preceding_text: continue

            last_word = preceding_text.split()[-1].lower()

            label = None
            if last_word in PERCENT_CONTEXT_WORDS:
                label = "B-PERCENT"
            elif last_word in VOLUME_CONTEXT_WORDS:
                label = "B-VOLUME"

            if label:
                found_entities.append((start, end, label))

    return sorted(found_entities)
