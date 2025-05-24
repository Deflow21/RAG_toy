import os
import yaml
import time
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# ========== НАСТРОЙКИ ==========
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "my_abc_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "abc_documents")

# Папка "feat" и тип документов:
FOLDERS_TO_PROCESS = [("feat_unzip", "feat")]

# Количество воркеров для параллельной загрузки и парсинга файлов:
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

# Размер батча для encode (batch_size=128 внутри метода):
# Но имеется в виду общий батч "документов" на один вызов encode; внутри самой модели мы можем 
# указывать batch_size=128, чтобы GPU не упирался в память, но всё обрабатывалось единым вызовом
EMBED_CHUNK_SIZE = 2000

# Размер батча для массового вставления в MongoDB:
INSERT_CHUNK_SIZE = 2000

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Глобальное подключение к MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db[MONGO_COLLECTION]

# ВАЖНО: подаём device='cuda', чтобы использовать GPU; если модель поддерживает FP16 – включите его
model_embed = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cuda',
    # use_fp16=True  # Раскомментируйте, если поддерживается и требуется
)

def get_mongo_collection():
    return mongo_collection

# ========== ГЕНЕРАЦИЯ SUMMARY ==========
def generate_entity_summary(entity) -> str:
    if not isinstance(entity, dict):
        return str(entity)
    
    parts = [f"{entity.get('type', 'UnknownType')} summary:"]

    for key in ['sharp', 'vert_indices', 'vert_parameters', 'face_indices']:
        if key in entity:
            parts.append(f"{key}={entity[key]};")

    geom_type = entity.get("type", "UnknownType")

    if geom_type == "Line":
        parts.append(f"direction={entity.get('direction', None)};")
        parts.append(f"location={entity.get('location', None)};")

    elif geom_type == "Circle":
        for key in ["location", "z_axis", "radius", "x_axis", "y_axis"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Ellipse":
        for key in ["focus1", "focus2", "x_axis", "y_axis", "z_axis", "x_radius", "y_radius"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "BSpline":
        closed = entity.get("closed", None)
        rational = entity.get("rational", None)
        degree = entity.get("degree", None)
        poles_count = len(entity.get("poles", []))
        knots_count = len(entity.get("knots", []))
        parts.append(f"closed={closed}; rational={rational}; degree={degree}; "
                     f"poles={poles_count}; knots={knots_count};")

    elif geom_type == "Plane":
        for key in ["location", "x_axis", "y_axis", "z_axis", "coefficients"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Cylinder":
        for key in ["location", "x_axis", "y_axis", "z_axis", "coefficients", "radius"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Cone":
        for key in ["location", "x_axis", "y_axis", "z_axis", "coefficients", "radius", "angle", "apex"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Sphere":
        for key in ["location", "x_axis", "y_axis", "z_axis", "coefficients", "radius"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Torus":
        for key in ["location", "x_axis", "y_axis", "z_axis", "max_radius", "min_radius"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Revolution":
        for key in ["location", "z_axis", "curve"]:
            parts.append(f"{key}={entity.get(key, None)};")

    elif geom_type == "Extrusion":
        for key in ["direction", "curve"]:
            parts.append(f"{key}={entity.get(key, None)};")

    else:
        for key, value in entity.items():
            if key == "type":
                continue
            parts.append(f"{key}={value};")

    return " ".join(parts)

def parse_yaml_file(file_path: str, doc_type: str):
    """
    Считываем YAML, генерируем summaries, возвращаем список "сырой" структуры без эмбеддинга:
    [
        {
            "model_id": ...,
            "doc_type": ...,
            "content": ...,
            "file_path": ...,
            "chunk_index": ...,
        },
        ...
    ]
    """
    model_id = os.path.basename(os.path.dirname(file_path))

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Ошибка при загрузке {file_path}: {e}")
        return []

    if not data or not isinstance(data, dict):
        return []

    entity_types = ['curves', 'surfaces', 'BSpline', 'line']
    all_entities = []
    for etype in entity_types:
        if etype in data:
            arr = data[etype]
            if not isinstance(arr, list):
                arr = [arr]
            all_entities.extend(arr)

    if not all_entities:
        return []

    # Генерация summaries
    all_summaries = [generate_entity_summary(entity) for entity in all_entities]

    docs_to_insert = []
    for i, summary_text in enumerate(all_summaries):
        docs_to_insert.append({
            "model_id": model_id,
            "doc_type": doc_type,
            "content": summary_text,
            "file_path": file_path,
            "chunk_index": i
        })

    return docs_to_insert

def gather_all_docs(root_folder: str, doc_type: str, max_workers: int = 8):
    """
    Рекурсивно ищем все .yml файлы и в пуле потоков парсим их, возвращаем список (без эмбеддингов).
    """
    yml_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".yml"):
                yml_files.append(os.path.join(root, file))

    results = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(parse_yaml_file, fp, doc_type): fp for fp in yml_files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Чтение YAML в {doc_type}", unit="file"):
            try:
                file_docs = future.result()
                if file_docs:
                    results.extend(file_docs)
            except Exception as e:
                logging.error(f"Ошибка при обработке {futures[future]}: {e}")

    return results

def main():
    collection = get_mongo_collection()
    collection.delete_many({})
    logging.info("Коллекция MongoDB очищена.")

    all_docs = []

    # 1. Сбор всех документов (БЕЗ эмбеддингов) из всех папок:
    for folder, doc_type in FOLDERS_TO_PROCESS:
        folder_docs = gather_all_docs(folder, doc_type, max_workers=MAX_WORKERS)
        all_docs.extend(folder_docs)

    logging.info(f"Собрано всего документов (строк для эмбеддинга): {len(all_docs)}")

    if not all_docs:
        logging.info("Нет данных для обработки. Завершаем.")
        return

    # 2. Генерация эмбеддингов батчами
    all_contents = [d["content"] for d in all_docs]
    logging.info("Начинаем encode эмбеддингов ...")

    # Чтобы не перегружать GPU, обрабатываем контент кусками:
    total_docs = len(all_docs)
    num_chunks = math.ceil(total_docs / EMBED_CHUNK_SIZE)

    embeddings = []
    start_idx = 0

    # Добавляем прогресс‐бар на цикл по кускам
    with tqdm(total=num_chunks, desc="Генерация эмбеддингов", unit="batch") as pbar:
        for _ in range(num_chunks):
            end_idx = min(start_idx + EMBED_CHUNK_SIZE, total_docs)
            batch_contents = all_contents[start_idx:end_idx]

            # Собственно вызов encode
            batch_embeddings = model_embed.encode(batch_contents, batch_size=128)
            embeddings.extend(batch_embeddings)

            start_idx = end_idx

            # Обновляем прогресс
            pbar.update(1)
            pbar.set_postfix({"Обработано эмбеддингов": len(embeddings)})
    assert len(embeddings) == total_docs, "Размер embeddings не совпадает с количеством документов!"

    logging.info("Эмбеддинги рассчитаны. Сохраняем в MongoDB...")

    # 3. Вставляем документы в MongoDB
    # Объединим все данные (docs + эмбеддинги) и пошлём chunkами в insert_many
    docs_to_insert = []
    for i, emb in enumerate(embeddings):
        doc = all_docs[i]
        doc["embedding"] = emb.tolist()
        docs_to_insert.append(doc)

    # Разбиваем на части, чтобы не вставлять 10-50 тысяч документов единым махом
    total_insert = len(docs_to_insert)
    insert_chunks = math.ceil(total_insert / INSERT_CHUNK_SIZE)
    start_idx = 0

    with tqdm(total=insert_chunks, desc="Запись в MongoDB", unit="batch") as pbar:
        for _ in range(insert_chunks):
            end_idx = min(start_idx + INSERT_CHUNK_SIZE, total_insert)
            chunk_docs = docs_to_insert[start_idx:end_idx]

            try:
                collection.insert_many(chunk_docs)
            except Exception as e:
                logging.error(f"Ошибка при вставке в MongoDB: {e}")

            start_idx = end_idx
            pbar.update(1)
            pbar.set_postfix({"Всего вставлено": start_idx})

    end_time = time.time()
    logging.info(f"Готово! Все записи загружены. Общее время: {end_time - start_time:.2f} сек.")


if __name__ == "__main__":
    main()
