import os
from dotenv import load_dotenv
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# === ЗАГРУЖАЕМ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "my_abc_db")
# Используем коллекцию для stat
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "stat_documents")

# Глобальный клиент MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_collection = mongo_db[MONGO_COLLECTION]

def get_mongo_collection():
    return mongo_collection

# === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ЭМБЕДДИНГОВ ===
model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

# === 1) ИНИЦИАЛИЗАЦИЯ МОДЕЛИ QWEN2.5-VL ===
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    device_map="sequential",
    torch_dtype=torch.bfloat16
)


# === 1) ФУНКЦИЯ ДЛЯ ПОИСКА В БД С DOC_TYPE='stat' ===
def retrieve_stat_docs(query_text, model_id=None, top_k=3):
    """
    Семантический поиск по doc_type='stat'.
    Возвращает список кортежей (model_id, content, similarity).
    Если указан model_id, ограничивает поиск.
    """
    collection = get_mongo_collection()

    # Эмбеддинг запроса
    query_embedding = model_embed.encode(query_text, batch_size=64)
    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        query_norm = 1e-10

    # Фильтр поиска по stat
    search_filter = {"doc_type": "stat"}
    if model_id:
        search_filter["model_id"] = model_id

    # Загружаем документы из MongoDB
    docs = list(collection.find(search_filter, {"model_id": 1, "content": 1, "embedding": 1}).limit(1000000))
    if not docs:
        return []

    # Собираем эмбеддинги и считаем нормы
    embeddings = np.array([np.array(doc.get("embedding", [])) for doc in docs])
    if embeddings.size == 0:
        return []
    doc_norms = np.linalg.norm(embeddings, axis=1)
    doc_norms[doc_norms == 0] = 1e-10

    # Косинусная схожесть
    sims = np.dot(embeddings, query_vec) / (doc_norms * query_norm)
    scored_docs = [(doc["model_id"], doc["content"], sim) for doc, sim in zip(docs, sims)]
    scored_docs.sort(key=lambda x: x[2], reverse=True)
    return scored_docs[:top_k]

# === 2) ОСНОВНАЯ ФУНКЦИЯ RAG С DOC_TYPE='stat' ===
def generate_json_from_image_with_rag_stat(image_path):
    """
    RAG с использованием документов stat для генерации JSON-описания.
    """
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")

    # Пример запроса и получение релевантных stat-документов
    user_query = "3D model, curves, surfaces, manufacturing process"
    relevant_docs = retrieve_stat_docs(user_query, model_id=None, top_k=15)

    # Ограничение длины контекста
    MAX_CONTEXT_LENGTH = 1200
    if relevant_docs:
        contexts = []
        for model_id, content, sim in relevant_docs:
            truncated = content[:MAX_CONTEXT_LENGTH]
            contexts.append(f"[model_id={model_id}]\n{truncated}")
        stat_context = "\n\n".join(contexts)
    else:
        stat_context = "No matching stat document found."

    # Подготавливаем оригинальный промпт (тот же, что и для feat)
    original_english_prompt = """Create a detailed description of the technological process of manufacturing a product based on an image providing three dimensions of a 3D model in JPG format. The process should be described in JSON format and include all stages of machining, starting from analysing the drawing and ending with obtaining the finished product. The response should be JSON only, with no additional text. JSON should contain the following information:
1. File name: the name of the input file with the drawing.
2. Operation name: the type of the main operation (e.g. turning, milling, grinding, etc.).
3. Part information: initial parameters of the part (length, width, height/thickness, weight).
4. Steps: the sequence of operations performed to machine the part. Each step must include:
4.1. Step number.
4.2. Description of the step (e.g., 'roughing', 'finishing', 'drilling holes', etc.).
4.3. The equipment used (e.g. lathe, milling machine, measuring tool, etc.).
4.4. Changes in the part parameters (length, width, height/thickness, weight) after the step has been executed. If the parameter does not change, specify a value of -1.

JSON example:

{
    "File name": "Input file name",
    "Name of operation": "Lathe",
    "Part information": {
        "Length": 10,
        "Width": 10,
        "Height/thickness": 10,
        "Weight": 10
    },
    "Steps": [
        {
            "Step number": 1,
            "Action": "Roughing the workpiece",
            "Equipment": ["Turning lathe"],
            "Significance": {
                "Length": 9,
                "Width": 9,
                "Height/thickness": 9,
                "Weight": 8
            }
        },
        {
            "Step number": 2,
            "Action": "Finishing of the surface",
            "Equipment": ["Turning lathe"],
            "Significance": {
                "Length": 8.5,
                "Width": 8.5,
                "Height/thickness": 8.5,
                "Weight": 7.5
            }
        }
    ]
}

Requirements:
(a) The description of each step shall be as accurate as possible and shall correspond to actual machining processes.
(b) The equipment used must be specified with respect to the specifics of the operation.
(c) Changes in part parameters must be calculated to realistic process tolerances.
(d) The response should be JSON only, with no additional text.
"""

    prompt = (
        f"[STAT CONTEXT from doc_type='stat']:\n{stat_context}\n\n"
        f"{original_english_prompt}\n\n"
        f"File name (input): {os.path.basename(image_path)}\n\n"
        "Important:\n"
        "- Provide only JSON, ending with the '}' character (no extra text).\n"
        "Your answer must end with '}'. No more text after the closing brace."
    )

    # Формируем сообщения для Qwen2.5-VL
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text_prompt, list):
        text_prompt = text_prompt[0]

    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    json_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return json_response

# === ПРИМЕР ЗАПУСКА ===
if __name__ == "__main__":
    result = generate_json_from_image_with_rag_stat("fng.jpg")
    print(result)
