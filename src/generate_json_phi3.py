import os

from transformers.cache_utils import DynamicCache
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image


# "добавляем" get_max_length через существующий get_max_cache_shape
DynamicCache.get_max_length = lambda self: self.get_max_cache_shape()

# === ЗАГРУЖАЕМ .env и MongoDB ===
load_dotenv()
MONGO_URI        = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME    = os.getenv("MONGO_DB_NAME", "my_abc_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "stat_documents")

mongo_client     = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION]

# === Модель эмбеддингов для RAG-контекста ===
model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

def get_mongo_collection():
    return mongo_collection

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

# === ИНИЦИАЛИЗАЦИЯ Phi-3-Vision-128k-Instruct ===
# === ИНИЦИАЛИЗАЦИЯ Phi-3-Vision-128k-Instruct ===
MODEL_NAME = "microsoft/phi-3-vision-128k-instruct"
processor  = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model      = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager"       # или 'flash_attention_2', если flash-attn установлена
)
# === ОРИГИНАЛЬНЫЙ ПРОМПТ ===
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

def generate_json_from_image_with_phi3(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")

    # 1) RAG-контекст
    docs = retrieve_stat_docs(
        "3D model, curves, surfaces, manufacturing process", top_k=10
    )
    context = "\n\n".join(f"[model_id={mid}]\n{cont[:1200]}"
                         for mid,cont,_ in docs) or "No matching stat document found."

    # 2) Собираем тело промпта
    prompt_body = (
        f"[STAT CONTEXT from doc_type='stat']:\n{context}\n\n"
        f"{original_english_prompt}\n\n"
        f"File name: {os.path.basename(image_path)}\n\n"
        "Important:\n"
        "- Provide only JSON, ending with '}'. No extra text."
    )

    # 3) Оборачиваем в чат-формат с тегом <|image_1|>
    chat_input = (
        "<|user|>\n"
        "<|image_1|>\n"
        f"{prompt_body}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    # 4) Инференс
    inputs = processor(text=chat_input, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    # 5) Декодируем ответ
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    print(generate_json_from_image_with_phi3("чертежи/example.jpg"))
