# loader.py
import pymongo, bson
from pymongo.errors import BulkWriteError
from db import mongo, COL_PARTS, COL_FEATURES, COL_OPERATIONS

db = mongo()

def upsert_part(rec: dict):
    pid = rec["id"]

    db[COL_PARTS].update_one(
        {"_id": pid},
        {"$setOnInsert": {
            "part_hash": rec["part_hash"],
            "stats":     rec["stats"],
            "generated_at": bson.datetime.datetime.utcnow(),
        }},
        upsert=True)

    feat_ops = [
        pymongo.UpdateOne(
            {"_id": f"{pid}:{i}"},
            {"$setOnInsert": {
                "part_id": pid,
                "feature_index": i,
                "kind":  ft["kind"],
                "params":ft["params"],
            }},
            upsert=True)
        for i, ft in enumerate(rec["features"])
    ]
    if feat_ops:
        try: db[COL_FEATURES].bulk_write(feat_ops, ordered=False)
        except BulkWriteError: pass

    op_ops = [
        pymongo.UpdateOne(
            {"_id": f"{pid}:{i}"},
            {"$setOnInsert": {
                "part_id": pid,
                "op_index": i,
                "type":  op["type"],
                "params":op["params"],
            }},
            upsert=True)
        for i, op in enumerate(rec["ops"])
    ]
    if op_ops:
        try: db[COL_OPERATIONS].bulk_write(op_ops, ordered=False)
        except BulkWriteError: pass
