#!/usr/bin/env python3
"""
embed_and_upsert.py
▶ Читает cad_stat + cad_operation из MongoDB (БД cad_rag)
▶ Считает эмбеддинги (MiniLM-L6-v2)
▶ Загружает точки в Qdrant        (payload: model_id, content, doc_type)
▶ Опционально сохраняет embedding в исходных документах MongoDB
"""

from __future__ import annotations
import uuid

from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from qdrant_helper import init_collection, upsert_docs   # ваша обёртка

# ──────────────── конфиг ────────────────
MONGO_URI = "mongodb://127.0.0.1:27017"
DB_NAME   = "cad_rag"

COLLS = {               # {doc_type: collection_name}
    "stat": "cad_stat_enriched",
    "op"  : "cad_operation_enriched",
}

BATCH_SIZE = 128        # размер пакета при инференсе
WRITE_BACK = True       # True → сохранить embedding и qdrant_id в Mongo
# ────────────────────────────────────────

client   = MongoClient(MONGO_URI)[DB_NAME]
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
init_collection()       # создаём коллекцию Qdrant, если её ещё нет


def gen_batches(cursor, size):
    """Yield documents by batches of <size>."""
    batch = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


total_inserted = 0

for doc_type, coll_name in COLLS.items():
    coll   = client[coll_name]
    total  = coll.estimated_document_count()
    cursor = coll.find({}, {"_id": 1, "model_id": 1, "content": 1})

    if not total:
        print(f"⚠️  {coll_name}: коллекция пуста — пропускаю")
        continue

    print(f"→ {coll_name}: {total} docs")

    for chunk in gen_batches(cursor, BATCH_SIZE):
        # 1. Текст для эмбеддинга
        texts = [d.get("content", "") for d in chunk]
        # 2. Считаем вектора
        vecs  = embedder.encode(texts, batch_size=len(chunk))

        # 3. Формируем точки для Qdrant
        points = []
        mongo_updates = []

        for d, v in zip(chunk, vecs):
            q_id = str(uuid.uuid4())          # валидный UUID
            points.append({
                "id":        q_id,
                "embedding": v.tolist(),
                "model_id":  d.get("model_id", ""),
                "content":   d.get("content", ""),
                "doc_type":  doc_type,
            })

            if WRITE_BACK:
                mongo_updates.append(
                    UpdateOne({"_id": d["_id"]},
                              {"$set": {"embedding": v.tolist(),
                                        "qdrant_id": q_id}})
                )

        # 4. Записываем в Qdrant
        upsert_docs(points)
        total_inserted += len(points)

        # 5. (опц.) Записываем обратно в Mongo
        if WRITE_BACK and mongo_updates:
            coll.bulk_write(mongo_updates, ordered=False)

        tqdm.write(f"✔️  {len(points)} docs → Qdrant  ({doc_type})")

print(f"\nTotal inserted into Qdrant: {total_inserted}")
