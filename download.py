import os
import requests
import subprocess
from tqdm import tqdm

# Путь к 7-Zip (изменен для диска E)
SEVEN_ZIP_PATH = r"E:\7-Zip\7z.exe"

# Словарь с категориями и ссылками
datasets = {
    "meta": [
        "https://archive.nyu.edu/rest/bitstreams/88595/retrieve"
    ],
    "feat": [
        "https://archive.nyu.edu/rest/bitstreams/89087/retrieve"
    ],
    "stat": [
        "https://archive.nyu.edu/rest/bitstreams/89086/retrieve"
    ]
}

# Функция для скачивания файла
def download_file(url, folder, filename, total_files, current_index):
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        print(f"[✓] Уже скачано: {filename} ({current_index}/{total_files})")
        return

    print(f"⬇ Скачивание {filename} ({current_index}/{total_files}) ...")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(path, "wb") as file, tqdm(
            desc=f"Загрузка {filename}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(1024):
                file.write(chunk)
                bar.update(len(chunk))

        print(f"[✓] Скачано: {filename} ({current_index}/{total_files})")
    else:
        print(f"[X] Ошибка при скачивании: {filename}")

# Функция для распаковки `.7z` файлов в папке
def extract_files(folder):
    print(f"📦 Распаковка файлов в папке {folder} ...")
    folder_path = os.path.abspath(folder)

    for file in tqdm(os.listdir(folder), desc="Распаковка архива", unit="файл"):
        if file.endswith(".7z"):
            archive_path = os.path.join(folder_path, file)

            if not os.path.exists(archive_path):
                print(f"[X] Файл отсутствует: {file}")
                continue

            print(f"📂 Распаковка {file} ...")
            try:
                subprocess.run([SEVEN_ZIP_PATH, "x", archive_path, f"-o{folder_path}", "-y"], check=True)
                os.remove(archive_path)
                print(f"[✓] Успешно распаковано и удалено: {file}")
            except Exception as e:
                print(f"[X] Ошибка при распаковке {file}: {e}")

# Основной процесс скачивания и распаковки
for folder, urls in datasets.items():
    os.makedirs(folder, exist_ok=True)
    total_files = len(urls)

    for i, url in enumerate(urls, start=1):
        filename = f"abc_{str(i).zfill(4)}_{folder}_v00.7z"
        download_file(url, folder, filename, total_files, i)

    extract_files(folder)
