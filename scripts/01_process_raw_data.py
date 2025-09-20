import os
import sys
import pandas as pd
import ast
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_converter import sanitize_annotations, indices_to_bio

RAW_DATA_PATH = "data/raw/train.csv"
PROCESSED_TRAIN_PATH = "data/processed/train_bio.csv"
PROCESSED_VAL_PATH = "data/processed/validation_bio.csv"


def fix_zero_tags(annotations: list) -> list:
    return [(start, end, 'O' if label == '0' else label) for start, end, label in annotations]


def main():
    print("Шаг 1: Загрузка и парсинг сырых данных...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, sep=";")
        df["annotation"] = df["annotation"].apply(ast.literal_eval)
        print(f"Загружено {len(df)} записей.")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {RAW_DATA_PATH}")
        return

    print("Шаг 2: Исправление ошибочных тегов ('0' -> 'O')...")
    original_zeros = sum(1 for annotations in df['annotation'] for _, _, label in annotations if label == '0')
    df['annotation'] = df['annotation'].apply(fix_zero_tags)
    print(f"Исправлено {original_zeros} ошибочных тегов.")

    print("Шаг 3: Очистка аннотаций...")
    processed_annotations = df.apply(
        lambda row: sanitize_annotations(row["sample"], row["annotation"])[1],
        axis=1
    )
    df["annotation"] = processed_annotations

    print("Шаг 4: Преобразование данных в BIO-формат...")
    processed_bio = df.apply(
        lambda row: indices_to_bio(row["sample"], row["annotation"]),
        axis=1
    )
    df_bio = pd.DataFrame(processed_bio.tolist(), columns=["tokens", "tags"])
    df_bio["sample"] = df["sample"]
    df_bio = df_bio[["sample", "tokens", "tags"]]
    print("Преобразование успешно завершено.")

    print("Шаг 5: Разделение на обучающую и валидационную выборки (85/15)...")
    train_df, val_df = train_test_split(df_bio, test_size=0.15, random_state=42)
    print(f"Размер обучающей выборки: {len(train_df)}")
    print(f"Размер валидационной выборки: {len(val_df)}")

    print("Шаг 6: Сохранение обработанных данных...")
    train_df.to_csv(PROCESSED_TRAIN_PATH, sep=";", index=False)
    val_df.to_csv(PROCESSED_VAL_PATH, sep=";", index=False)
    print(f"Обучающие данные сохранены в: {PROCESSED_TRAIN_PATH}")
    print(f"Валидационные данные сохранены в: {PROCESSED_VAL_PATH}")
    print("\nЗадача успешно выполнена!")


if __name__ == "__main__":
    main()
