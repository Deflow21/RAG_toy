# db.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DBNAME    = "abc_parts"

COL_PARTS      = "parts"
COL_FEATURES   = "features"
COL_OPERATIONS = "operations"
COL_EMBEDS     = "embeddings"
COL_ARCH_STATE = "archive_state"


def ensure_index(coll, keys, **opts):
    """
    Создаёт индекс, только если точного аналога ещё нет.
    keys : [("field", 1), ...]
    opts : те же kwargs, что и у create_index (name=…, unique=…)
    """
    wanted_key = dict(keys)
    for idx in coll.list_indexes():
        if idx["key"] == wanted_key:
            # индекс уже существует — выходим
            return
    coll.create_index(keys, **opts)


def mongo():
    db = MongoClient(MONGO_URI, tz_aware=True)[DBNAME]

    # features: (part_id, feature_index)
    ensure_index(
        db[COL_FEATURES],
        [("part_id", ASCENDING), ("feature_index", ASCENDING)],
        name="idx_part_feature"
    )

    # operations: (part_id, op_index)
    ensure_index(
        db[COL_OPERATIONS],
        [("part_id", ASCENDING), ("op_index", ASCENDING)],
        name="idx_part_op"
    )

    # archive_state — индекс _id создаётся Mongo автоматически
    return db
