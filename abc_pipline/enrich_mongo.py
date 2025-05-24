#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create *enriched* Mongo collections with a human-readable `content` field.
cad_stat  → cad_stat_enriched
cad_operation → cad_operation_enriched
"""
from pymongo import MongoClient
from math import sqrt

# ─── CONFIG ─────────────────────────────────────────
MONGO_URI   = "mongodb://localhost:27017"
DB_NAME     = "cad_rag"
SRC_STAT    = "cad_stat"
SRC_OPER    = "cad_operation"
DST_STAT    = "cad_stat_enriched"
DST_OPER    = "cad_operation_enriched"
BATCH_SIZE  = 500
# ────────────────────────────────────────────────────

cli   = MongoClient(MONGO_URI)
db    = cli[DB_NAME]
ssrc  = db[SRC_STAT]
osrc  = db[SRC_OPER]
sdst  = db[DST_STAT]
odst  = db[DST_OPER]

# ——— helper: bbox → short description ———
def describe_bbox(bmax, bmin):
    L, W, H = [abs(bmax[i]-bmin[i]) for i in range(3)]
    dims = sorted((L, W, H), reverse=True)   # dims[0]≥dims[1]≥dims[2]
    L, W, H = dims
    if H < min(L, W)*0.2:
        shape = "thin plate"
    elif L/W > 3 and W/H < 1.5:
        shape = "rod/shaft"
    elif abs(L-W) < 0.2*L and abs(W-H) < 0.2*W:
        shape = "block"
    else:
        shape = "prismatic"
    return shape, (round(L), round(W), round(H))

# ------------- ENRICH cad_stat -------------
bulk = []
for doc in ssrc.find({}, no_cursor_timeout=True):
    bmax = doc.get("bbox_max")
    bmin = doc.get("bbox_min")
    if not (bmax and bmin):
        continue
    shape, (L,W,H) = describe_bbox(bmax, bmin)
    vol   = doc.get("volume")
    sharp = doc.get("sharp_edges", 0)
    doc["content"] = (f"{shape}; approx {L}×{W}×{H} mm; "
                      f"volume≈{vol:.2e}; sharp_edges={sharp}")
    bulk.append(doc)
    if len(bulk) >= BATCH_SIZE:
        sdst.insert_many(bulk)
        bulk.clear()
if bulk: sdst.insert_many(bulk)
print("cad_stat_enriched populated:", sdst.estimated_document_count())

# ------------- ENRICH cad_operation ----------
bulk = []
for doc in osrc.find({}, no_cursor_timeout=True):
    ops  = {o["op_type"] for o in doc.get("operations", [])}
    # quick tags
    if "revolve" in ops: ops.add("rotational")
    if any("pattern" in o for o in ops): ops.add("pattern")
    doc["content"] = " ".join(sorted(ops))
    bulk.append(doc)
    if len(bulk) >= BATCH_SIZE:
        odst.insert_many(bulk)
        bulk.clear()
if bulk: odst.insert_many(bulk)
print("cad_operation_enriched populated:", odst.estimated_document_count())
