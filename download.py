import os
import requests
import subprocess
from tqdm import tqdm

# Путь к 7-Zip (пример для диска E)
SEVEN_ZIP_PATH = r"E:\7-Zip\7z.exe"

# Словарь с категориями и ссылками
datasets = {
    "feat": [
        "https://archive.nyu.edu/rest/bitstreams/89087/retrieve"
    ]
}

def download_file(url: str, path: str):
    """
    Скачивает файл по указанному URL и сохраняет под именем path,
    выводя прогресс по объёму (байты).
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(path, "wb") as f, tqdm(
            desc=f"Загрузка {os.path.basename(path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(1024):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        raise RuntimeError(f"Ошибка при скачивании {url}, статус: {response.status_code}")

def extract_files_in_folder(folder: str):
    """
    Ищет все .7z файлы в папке folder и распаковывает их
    с помощью 7-Zip, выводя прогресс по количеству файлов.
    """
    files = [f for f in os.listdir(folder) if f.endswith(".7z")]
    if not files:
        print(f"Нет .7z файлов для распаковки в папке {folder}.")
        return

    print(f"📦 Распаковка {len(files)} архив(ов) из папки {folder} ...")
    for file in tqdm(files, desc="Распаковка", unit="архив"):
        archive_path = os.path.join(folder, file)
        try:
            subprocess.run([SEVEN_ZIP_PATH, "x", archive_path, f"-o{folder}", "-y"],
                           check=True, capture_output=True)
            os.remove(archive_path)
        except Exception as e:
            print(f"[X] Ошибка при распаковке {file}: {e}")

def main():
    """
    1) Формируем общий список (folder, url, filename) для всех файлов.
    2) Скачиваем поочерёдно все файлы, выводя общий прогресс по кол-ву.
    3) Распаковываем архивы в каждой папке.
    """
    all_tasks = []
    for folder, urls in datasets.items():
        os.makedirs(folder, exist_ok=True)
        for i, url in enumerate(urls[:1000], start=1):
            filename = f"abc_{str(i).zfill(4)}_{folder}_v00.7z"
            path = os.path.join(folder, filename)
            all_tasks.append((folder, url, path))

    total_files = len(all_tasks)
    if total_files == 0:
        print("Нет файлов для скачивания.")
        return

    print(f"Нужно скачать {total_files} файл(ов). Начинаем загрузку...\n")

    # Единый прогресс по количеству файлов
    with tqdm(total=total_files, desc="Всего файлов загружено", unit="файл") as pbar:
        for folder, url, path in all_tasks:
            # Проверяем, не скачан ли уже
            if os.path.exists(path):
                print(f"[✓] Уже скачано: {os.path.basename(path)}")
            else:
                try:
                    download_file(url, path)
                    print(f"[✓] Скачано: {os.path.basename(path)}")
                except Exception as e:
                    print(f"[X] Ошибка при скачивании {url}: {e}")

            pbar.update(1)  # Увеличиваем счётчик на 1 файл

    # После скачивания распаковываем в каждой папке
    for folder in datasets.keys():
        extract_files_in_folder(folder)

if __name__ == "__main__":
    main()
