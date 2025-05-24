import os
import time
from tqdm import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# ========== НАСТРОЙКИ ==========
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "my_abc_db")
# Заливка в отдельную коллекцию для stat
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "stat_documents")

# Обрабатываем корневую папку "stat" и её подпапки с моделями
FOLDERS_TO_PROCESS = [("stat", "stat")]

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Подключение к MongoDB
def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

# Чтение полного файла YAML как текста
def parse_whole_file(file_path: str, doc_type: str) -> dict:
    model_id = os.path.basename(os.path.dirname(file_path))
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Ошибка чтения {file_path}: {e}")
        return None

    return {
        "model_id": model_id,
        "doc_type": doc_type,
        "content": content,
        "file_path": file_path
    }

# Собираем все документы из указанной папки и подпапок

def gather_all_docs(root_folder: str, doc_type: str):
    docs = []
    for root, _, files in os.walk(root_folder):
        for fn in files:
            if fn.endswith(".yml"):
                path = os.path.join(root, fn)
                doc = parse_whole_file(path, doc_type)
                if doc:
                    docs.append(doc)
    return docs

# ========== MAIN ==========

def main():
    start_time = time.time()
    collection = get_mongo_collection()
    # Очищаем коллекцию перед загрузкой
    collection.delete_many({})
    logging.info("Коллекция MongoDB (stat_documents) очищена.")

    # Сбор документов из корневой папки "stat" и всех её подпапок
    all_docs = []
    for folder, doc_type in FOLDERS_TO_PROCESS:
        logging.info(f"Сбор файлов из папки: {folder}")
        folder_docs = gather_all_docs(folder, doc_type)
        all_docs.extend(folder_docs)

    total = len(all_docs)
    logging.info(f"Найдено файлов для загрузки: {total}")
    if total == 0:
        logging.info("Нет данных для загрузки. Завершаем.")
        return

    # Вставка всех документов целиком
    try:
        collection.insert_many(all_docs)
        logging.info(f"Успешно загружено {total} документов в stat_documents.")
    except Exception as e:
        logging.error(f"Ошибка при вставке в MongoDB: {e}")

    duration = time.time() - start_time
    logging.info(f"Загрузка завершена за {duration:.2f} сек.")

if __name__ == "__main__":
    main()
