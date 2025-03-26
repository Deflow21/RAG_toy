import os
import psycopg2
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройки подключения к PostgreSQL
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "113245"
DB_HOST = "localhost"
DB_PORT = 5432

# Инициализация модели эмбеддингов (глобально, чтобы не загружать несколько раз)
model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# Перед загрузкой очищаем таблицу (открываем одно соединение)
conn = get_db_connection()
cur = conn.cursor()
cur.execute("TRUNCATE TABLE abc_documents;")
conn.commit()
cur.close()
conn.close()
print("Таблица очищена.")

def generate_entity_summary(entity_type: str, entity) -> str:
    """
    Генерирует текстовое описание (summary) для отдельного объекта.
    Если entity не является словарём, возвращает его строковое представление.
    """
    if not isinstance(entity, dict):
        return str(entity)
    
    summary = f"{entity_type} summary:\n"
    
    if entity_type.lower() == "curves":
        summary += f"Type: {entity.get('type', 'N/A')}; "
        summary += f"Location: {entity.get('location', 'N/A')}; "
        if 'radius' in entity:
            summary += f"Radius: {entity.get('radius')}; "
        summary += f"Sharp: {entity.get('sharp', 'N/A')}; "
        summary += f"Vertices: {entity.get('vert_indices', 'N/A')}; "
        summary += f"Vertex Params: {entity.get('vert_parameters', 'N/A')}; "
        if 'x_axis' in entity:
            summary += f"x_axis: {entity.get('x_axis')}; "
        if 'y_axis' in entity:
            summary += f"y_axis: {entity.get('y_axis')}; "
        if 'z_axis' in entity:
            summary += f"z_axis: {entity.get('z_axis')}; "
    elif entity_type.lower() == "surfaces":
        summary += f"Coefficients: {entity.get('coefficients', 'N/A')}; "
        summary += f"Face Indices: {entity.get('face_indices', 'N/A')}; "
    elif entity_type.lower() == "bspline":
        summary += f"Closed: {entity.get('closed', 'N/A')}; "
        summary += f"Continuity: {entity.get('continuity', 'N/A')}; "
        summary += f"Degree: {entity.get('degree', 'N/A')}; "
        summary += f"Knots count: {len(entity.get('knots', []))}; "
        summary += f"Poles count: {len(entity.get('poles', []))}; "
        summary += f"Rational: {entity.get('rational', 'N/A')}; "
        summary += f"Sharp: {entity.get('sharp', 'N/A')}; "
        summary += f"Vertices: {entity.get('vert_indices', 'N/A')}; "
    elif entity_type.lower() == "line":
        summary += f"Direction: {entity.get('direction', 'N/A')}; "
        summary += f"Location: {entity.get('location', 'N/A')}; "
        summary += f"Sharp: {entity.get('sharp', 'N/A')}; "
        summary += f"Vertices: {entity.get('vert_indices', 'N/A')}; "
        summary += f"Vertex Params: {entity.get('vert_parameters', 'N/A')}; "
    else:
        for key, value in entity.items():
            summary += f"{key}: {value}; "
    
    return summary.strip()

def process_yaml_file(file_path: str, doc_type: str):
    """
    Читает YAML-файл, парсит его и для каждого логического объекта (например, curves, surfaces, BSpline, line)
    генерирует краткое описание (summary), делает эмбеддинг и записывает в таблицу.
    Каждая обработка файла происходит в отдельном соединении с БД.
    """
    model_id = os.path.basename(os.path.dirname(file_path))
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
            return

    # Открываем новое соединение для текущего файла
    conn = get_db_connection()
    cur = conn.cursor()
    
    for entity_type in ['curves', 'surfaces', 'BSpline', 'line']:
        if entity_type in data:
            entities = data[entity_type] if isinstance(data[entity_type], list) else [data[entity_type]]
            for idx, entity in enumerate(entities):
                summary = generate_entity_summary(entity_type, entity)
                embedding = model_embed.encode(summary).tolist()
                
                cur.execute("""
                    INSERT INTO abc_documents (model_id, doc_type, content, embedding, file_path, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    model_id,
                    doc_type,  
                    summary,
                    embedding,
                    file_path,
                    idx
                ))
    
    conn.commit()
    cur.close()
    conn.close()

def process_folder(root_folder: str, doc_type: str, max_workers: int = 8):
    """
    Рекурсивно обходит папку root_folder, ищет все .yml-файлы,
    и параллельно обрабатывает их с помощью ThreadPoolExecutor.
    """
    yml_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".yml"):
                yml_files.append(os.path.join(root, file))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_yaml_file, file_path, doc_type): file_path for file_path in yml_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Обработка {doc_type}", unit="file"):
            try:
                future.result()
            except Exception as e:
                print(f"Ошибка при обработке {futures[future]}: {e}")

# Запускаем обработку для нужных папок
process_folder("meta", "meta", max_workers=8)
process_folder("feat", "feat", max_workers=8)

print("Готово! Записи обновлены в БД.")
