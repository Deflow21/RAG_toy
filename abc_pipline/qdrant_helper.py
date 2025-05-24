"""
qdrant_helper.py
Мини-обёртка над Qdrant для хранения/поиска эмбеддингов.
"""

from qdrant_client import QdrantClient, models
import os

# ────────── параметры подключения ──────────
HOST = os.getenv("QDRANT_HOST", "localhost")
PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = "stat_docs"   # любое имя
VECTOR_SIZE = 384           # all-MiniLM-L6-v2

client = QdrantClient(host=HOST, port=PORT)


DROP_BEFORE_LOAD = bool(int(os.getenv("QDRANT_DROP", "0")))  # 1 → очистить

def init_collection() -> None:
    if DROP_BEFORE_LOAD and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def upsert_docs(docs):
    """
    docs = list[dict] со структурой
      {"id": str|int, "embedding": list[float],
       "model_id": str, "content": str}
    """
    points = [
        models.PointStruct(
            id=doc["id"],
            vector=doc["embedding"],
            payload={
                "model_id": doc["model_id"],
                "content":  doc["content"],
            },
        )
        for doc in docs
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def search(query_vec, top_k: int = 3):
    """
    Возвращает список кортежей
        (model_id, content, score, doc_type)
    doc_type берётся из payload; если его нет (старые точки) → 'stat'.
    """
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k,
    )

    return [
        (
            h.payload.get("model_id", ""),
            h.payload.get("content", ""),
            h.score,
            h.payload.get("doc_type", "stat"),   # ← добавили 4-й элемент
        )
        for h in hits
    ]
