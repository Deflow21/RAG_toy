#!/usr/bin/env python3
"""
rag_build.py — «всё-в-одном»:
1. Скачивает первые 10 архивов abc_*_feat_v00.7z
2. Распаковывает YAML-файлы по пачкам во временную папку
3. Заливает Part / Feature в MongoDB
4. Делает текстовые «карточки», строит эмбеддинги (Sentence-Transformers)
5. Кладёт векторы + payload в Qdrant
6. После каждой пачки освобождает место на диске (удаляет только распакованную папку)

~100 000 файлов (~10 архивов по 10 000 моделей) обрабатывается около часа на обычном ПК.
"""
import sys
import json
import hashlib
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

import py7zr
import yaml
from tqdm import tqdm
from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http.models import VectorParams, Distance

# -------------------------------------------------------------------------
LINKS = [
    "https://archive.nyu.edu/rest/bitstreams/89087/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89090/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89093/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89096/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89099/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89102/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89105/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89108/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89111/retrieve",
    "https://archive.nyu.edu/rest/bitstreams/89114/retrieve",
]
WORKDIR      = Path("work")
ARCHIVES_DIR = WORKDIR / "7z"   # для .7z-файлов
TMP_DIR      = WORKDIR / "tmp"  # для распаковки одной пачки
MONGO_URL    = "mongodb://localhost:27017"
DBNAME       = "rag_abc"
COL_PARTS    = "parts"
COL_FEATURES = "features"
QDRANT_URL   = ":memory:"       # in-proc Qdrant
COLLECT_NAME = "abc_feat"
BATCH        = 1024
EMB_MODEL    = "intfloat/e5-large-v2"
# -------------------------------------------------------------------------

def mongo() -> Any:
    """Подключаемся к MongoDB и создаём индексы."""
    cli = MongoClient(MONGO_URL)
    db  = cli[DBNAME]
    db[COL_PARTS].create_index([("part_id", ASCENDING)], unique=True)
    db[COL_FEATURES].create_index(
        [("part_id", ASCENDING), ("feature_id", ASCENDING)], unique=True
    )
    return db

def download(link: str, out_dir: Path) -> Path:
    """Скачиваем архив (или пропускаем, если уже есть)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx  = LINKS.index(link)
    name = f"abc_{idx:04d}_feat_v00.7z"
    dst  = out_dir / name
    if dst.exists():
        print(f"→ Архив уже есть, пропускаем скачивание: {name}")
    else:
        print(f"→ Скачиваем: {name}")
        subprocess.run([
            "aria2c", "-x16", "-s16", "-k1M",
            "-d", str(out_dir), "-o", name, link
        ], check=True)
    return dst

def extract_7z_with_py7zr(archive: Path, out_dir: Path):
    """
    Распаковывает .7z во временную папку:
    1) py7zr сразу извлекает все файлы (папки + YAML) в out_dir
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(str(archive), mode='r') as sz:
        sz.extractall(path=str(out_dir))
    # НЕ удаляем сам .7z, чтобы оставить его в ARCHIVES_DIR

def etl_yaml_batch(db: Any, dir_path: Path) -> List[int]:
    """
    Заливаем все .yml из dir_path (включая вложенные папки) в MongoDB.
    Возвращаем список загруженных part_id.
    """
    part_ids: List[int] = []
    for fp in sorted(dir_path.rglob("*.yml")):
        data = yaml.safe_load(fp.open("r", encoding="utf-8"))
        # part_id — это имя папки, в которой лежит файл
        pid = int(fp.parent.name)
        part_ids.append(pid)

        db[COL_PARTS].update_one(
            {"part_id": pid},
            {"$setOnInsert": {
                "bbox"      : data.get("bbox"),
                "raw_file"  : str(fp),
                "n_features": len(data.get("features", []))
            }}, upsert=True
        )
        for feat in data.get("features", []):
            doc = {
                "part_id"     : pid,
                "feature_id"  : feat["id"],
                "feature_type": feat["type"],
                "geom"        : feat["geom"],
                "md5"         : hashlib.md5(
                                   json.dumps(feat, sort_keys=True).encode()
                               ).hexdigest()
            }
            db[COL_FEATURES].update_one(
                {"part_id": pid, "feature_id": feat["id"]},
                {"$set": doc}, upsert=True
            )
    return part_ids

def make_card(ft: Dict[str, Any]) -> str:
    """Строим текстовую карточку для эмбеддинга."""
    g  = ft["geom"]
    ax = g.get("axis", "?")
    r  = g.get("radius", "?")
    return f"Part {ft['part_id']} | {ft['feature_type']} #{ft['feature_id']} | Ø{r} мм ось {ax}"

def embed_batch(
    db: Any,
    qdr: qdrant_client.QdrantClient,
    model: SentenceTransformer,
    part_ids: List[int]
):
    """Считаем эмбеддинги и заливаем их в Qdrant."""
    texts, metas = [], []
    for pid in part_ids:
        for ft in db[COL_FEATURES].find({"part_id": pid}, {"_id":0}):
            texts.append(make_card(ft))
            metas.append(ft)
            if len(texts) >= BATCH:
                vecs = model.encode(texts, batch_size=64, normalize_embeddings=True)
                pts  = [
                    {"id": m["md5"], "vector": v, "payload": {**m, "text": t}}
                    for m,v,t in zip(metas, vecs, texts)
                ]
                qdr.upsert(collection_name=COLLECT_NAME, points=pts)
                texts.clear(); metas.clear()

    if texts:
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True)
        pts  = [
            {"id": m["md5"], "vector": v, "payload": {**m, "text": t}}
            for m,v,t in zip(metas, vecs, texts)
        ]
        qdr.upsert(collection_name=COLLECT_NAME, points=pts)

def stage_1(db: Any, qdr: qdrant_client.QdrantClient, model: SentenceTransformer):
    """Главный цикл: скачиваем → распаковываем → Mongo → Qdrant → чистим."""
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    for idx, link in enumerate(LINKS, 1):
        archive   = download(link, ARCHIVES_DIR)
        batch_dir = TMP_DIR / f"batch_{idx:02d}"
        extract_7z_with_py7zr(archive, batch_dir)

        pids = etl_yaml_batch(db, batch_dir)
        embed_batch(db, qdr, model, pids)

        # удаляем только распакованную папку
        shutil.rmtree(batch_dir)
        tqdm.write(f"✔ Пачка {idx}/{len(LINKS)} обработана.")
        if idx == 1:
            tqdm.write("⏸ Остановлено после первой пачки для проверки БД.")
            break

    tqdm.write("==> Все пачки загружены и векторизованы.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true",
                   help="пропустить загрузку и распаковку (если всё уже готово)")
    args = p.parse_args()

    db    = mongo()
    model = SentenceTransformer(EMB_MODEL)
    dim   = model.get_sentence_embedding_dimension()
    qdr   = qdrant_client.QdrantClient(QDRANT_URL)
    qdr.recreate_collection(
        COLLECT_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    if not args.skip_download:
        stage_1(db, qdr, model)
    else:
        print("⏭ Пропускаем download+extract+load+embed — считаем, что всё готово.")

    print(
        f"\nMongo: parts={db[COL_PARTS].count_documents({})}, "
        f"features={db[COL_FEATURES].count_documents({})}"
    )
    print("🎉 Готово! Теперь RAG/vLLM может ходить в Qdrant за 100 000 карточек.")

if __name__ == "__main__":
    main()
