import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# ========== НАСТРОЙКИ ==========
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "my_abc_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "abc_documents")
FOLDERS_TO_PROCESS = [("feat", "feat")]
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

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
    
    # Общие поля
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
        parts.append(f"closed={closed}; rational={rational}; degree={degree}; poles={poles_count}; knots={knots_count};")
    
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

def process_yaml_file(file_path: str, doc_type: str):
    model_id = os.path.basename(os.path.dirname(file_path))
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Ошибка при загрузке {file_path}: {e}")
        return

    if not data or not isinstance(data, dict):
        return

    entity_types = ['curves', 'surfaces', 'BSpline', 'line']
    all_entities = []
    for etype in entity_types:
        if etype in data:
            arr = data[etype]
            if not isinstance(arr, list):
                arr = [arr]
            all_entities.extend(arr)

    if not all_entities:
        return

    # Генерация summaries
    all_summaries = [generate_entity_summary(entity) for entity in all_entities]

    # Один вызов encode на весь список с оптимальным batch_size
    embeddings = model_embed.encode(all_summaries, batch_size=128)

    docs_to_insert = []
    for i, summary_text in enumerate(all_summaries):
        docs_to_insert.append({
            "model_id": model_id,
            "doc_type": doc_type,
            "content": summary_text,
            "embedding": embeddings[i].tolist(),
            "file_path": file_path,
            "chunk_index": i
        })

    if docs_to_insert:
        try:
            collection = get_mongo_collection()
            collection.insert_many(docs_to_insert)
        except Exception as e:
            logging.error(f"Ошибка при вставке в MongoDB для файла {file_path}: {e}")

def process_folder(root_folder: str, doc_type: str, max_workers: int = 8):
    yml_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".yml"):
                yml_files.append(os.path.join(root, file))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_yaml_file, fp, doc_type): fp for fp in yml_files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Обработка {doc_type}", unit="file"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Ошибка при обработке {futures[future]}: {e}")

def main():
    collection = get_mongo_collection()
    collection.delete_many({})
    logging.info("Коллекция очищена.")

    for folder, doc_type in FOLDERS_TO_PROCESS:
        process_folder(folder, doc_type, max_workers=MAX_WORKERS)

    logging.info("Готово! Все записи загружены.")

if __name__ == "__main__":
    main()
