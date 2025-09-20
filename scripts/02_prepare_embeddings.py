import os
import requests
import pickle
import numpy as np
from tqdm import tqdm

EXTERNAL_DATA_PATH = "../data/external"
EMBEDDINGS_PATH = os.path.join(EXTERNAL_DATA_PATH, "embeddings")
OUTPUT_PICKLE_PATH = os.path.join(EMBEDDINGS_PATH, "ru_en_aligned.pkl")

RU_VECTORS_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.ru.align.vec"
EN_VECTORS_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec"

RU_VECTORS_FILE = os.path.join(EMBEDDINGS_PATH, "wiki.ru.align.vec")
EN_VECTORS_FILE = os.path.join(EMBEDDINGS_PATH, "wiki.en.align.vec")


def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"Файл {local_filename} уже существует. Скачивание пропущено.")
        return
    print(f"Скачивание файла из {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    print(f"Файл успешно скачан и сохранен как {local_filename}")


def load_and_combine_vectors(ru_path, en_path):
    embeddings_dict = {}

    for lang, file_path in [("Русский", ru_path), ("Английский", en_path)]:
        print(f"Обработка векторов... Язык: {lang}")
        with open(file_path, "r", encoding="utf-8") as f:
            next(f)
            for line in tqdm(f):
                parts = line.rstrip().split(" ")
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings_dict[word] = vector
    return embeddings_dict


def main():
    print("--- Запуск скрипта подготовки эмбеддингов ---")
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

    download_file(RU_VECTORS_URL, RU_VECTORS_FILE)
    download_file(EN_VECTORS_URL, EN_VECTORS_FILE)

    print("\nШаг 2: Загрузка и объединение векторов...")
    combined_embeddings = load_and_combine_vectors(RU_VECTORS_FILE, EN_VECTORS_FILE)
    print(f"Объединенный словарь создан. Общее количество векторов: {len(combined_embeddings)}")

    print(f"\nШаг 3: Сохранение словаря в файл {OUTPUT_PICKLE_PATH}...")
    with open(OUTPUT_PICKLE_PATH, "wb") as f:
        pickle.dump(combined_embeddings, f)
    print(f"Словарь успешно сохранен.")

    print(f"\nШаг 4: Очистка исходных текстовых файлов...")
    try:
        os.remove(RU_VECTORS_FILE)
        print(f" - Файл {RU_VECTORS_FILE} удален.")
        os.remove(EN_VECTORS_FILE)
        print(f" - Файл {EN_VECTORS_FILE} удален.")
    except OSError as e:
        print(f"Ошибка при удалении файлов: {e}")

    print("\n--- Скрипт успешно завершил работу! ---")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
