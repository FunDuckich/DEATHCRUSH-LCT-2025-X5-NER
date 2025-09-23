import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

AUGMENTED_DATA_PATH = "data/processed/train_targeted_augmented_v1.csv"

FINAL_TRAIN_PATH = "data/processed/final_train.csv"
FINAL_VAL_PATH = "data/processed/final_val.csv"


def get_stratify_key(tags: list) -> str:
    b_tags = sorted([tag for tag in tags if tag.startswith('B-')])

    return "_".join(b_tags) if b_tags else "O_I_only"


def main():
    print("--- Шаг 1: Загрузка полного аугментированного датасета ---")
    try:
        df = pd.read_csv(AUGMENTED_DATA_PATH, sep=";")
        df["tokens"] = df["tokens"].apply(eval)
        df["tags"] = df["tags"].apply(eval)
        print(f"Загружено {len(df)} аугментированных записей.")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {AUGMENTED_DATA_PATH}")
        print("Убедитесь, что ноутбук для генерации аугментации был успешно выполнен.")
        return

    print("\n--- Шаг 2: Создание ключей для стратифицированного разбиения ---")
    df['stratify_key'] = df['tags'].progress_apply(get_stratify_key)
    print("Ключи для стратификации успешно созданы.")

    stratify_counts = df['stratify_key'].value_counts()
    single_member_classes = stratify_counts[stratify_counts == 1]
    if not single_member_classes.empty:
        print(f"\nПредупреждение: Найдено {len(single_member_classes)} страт с одним элементом. "
              "Они будут отнесены к обучающей выборке.")
        df_stratifiable = df[~df['stratify_key'].isin(single_member_classes.index)]
        df_singletons = df[df['stratify_key'].isin(single_member_classes.index)]
    else:
        df_stratifiable = df
        df_singletons = pd.DataFrame()

    print("\n--- Шаг 3: Выполнение стратифицированного разбиения (85/15) ---")
    train_df_strat, val_df = train_test_split(
        df_stratifiable,
        test_size=0.15,
        random_state=42,
        stratify=df_stratifiable['stratify_key']
    )

    train_df = pd.concat([train_df_strat, df_singletons], ignore_index=True)

    train_df = train_df.drop(columns=['stratify_key'])
    val_df = val_df.drop(columns=['stratify_key'])

    print("Разбиение успешно завершено.")
    print(f"  Размер итоговой обучающей выборки: {len(train_df)}")
    print(f"  Размер итоговой валидационной выборки: {len(val_df)}")

    def has_rare_class(tags):
        return any(t.endswith('VOLUME') or t.endswith('PERCENT') for t in tags)

    val_rare_count = val_df['tags'].apply(has_rare_class).sum()
    print(f"  Количество примеров с редкими классами в валидации: {val_rare_count} (ожидается > 0)")

    print("\n--- Шаг 4: Сохранение финальных датасетов ---")
    train_df.to_csv(FINAL_TRAIN_PATH, sep=";", index=False)
    val_df.to_csv(FINAL_VAL_PATH, sep=";", index=False)
    print(f"Финальные обучающие данные сохранены в: {FINAL_TRAIN_PATH}")
    print(f"Финальные валидационные данные сохранены в: {FINAL_VAL_PATH}")

    print("\nЗадача успешно выполнена!")


if __name__ == "__main__":
    from tqdm.auto import tqdm

    tqdm.pandas()
    main()
