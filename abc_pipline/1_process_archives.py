#!/usr/bin/env python3
"""
Optimised loader for ABC dataset archives into MongoDB.

Изменения v3
------------
* **Групповая обработка** (stat → ofs → feat) сохранена.
* Добавлен флаг `--verbose` (`-v`) для уровня логов DEBUG.
* Логи выводятся через `tqdm.write`, поэтому их видно поверх прогресс‑баров.
* В конце каждого архива печатается краткая статистика по вставленным документам.
"""
import os
import re
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import pymongo
import py7zr
from pymongo import InsertOne
from tqdm.auto import tqdm

from config import RAW_DIR, UNZIP_DIR

# ────────────────────────── Logging + tqdm integration ───────
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass

def setup_logging(level: str):
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

setup_logging(os.getenv("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ────────────────────────── YAML Loader ──────────────────────
try:
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader
    logger.warning("libyaml not found – YAML parsing slower.")

# ────────────────────────── MongoDB ──────────────────────────
client = pymongo.MongoClient(
    "mongodb://localhost:27017/",
    w=1,
    maxPoolSize=100,
    socketTimeoutMS=30000,
)

db              = client["cad_rag"]
cad_stat_col    = db["cad_stat"]
cad_patch_col   = db["cad_patch"]
cad_curve_col   = db["cad_curve"]
cad_op_col      = db["cad_operation"]
arch_state_col  = db["arch_state"]

# ────────────────────────── Helpers ─────────────────────────

def split_id(folder_name: str):
    m = re.match(r"(\d{8})_(\w+)", folder_name)
    return (m.group(1), m.group(2)) if m else (folder_name, None)


def bulk_insert_surfs_curves(model_id: str, data: dict):
    surf_ops, curve_ops = [], []

    for surf in data.get("surfaces", []):
        surf_ops.append(InsertOne({
            "model_id": model_id,
            "type": surf.get("type"),
            "props": {k: v for k, v in surf.items() if k not in ("type", "vert_indices", "face_indices")},
            "vert_indices": surf.get("vert_indices"),
            "face_indices": surf.get("face_indices"),
        }))

    for curve in data.get("curves", []):
        curve_ops.append(InsertOne({
            "model_id": model_id,
            "type": curve.get("type"),
            "props": {k: v for k, v in curve.items() if k not in ("type", "vert_indices")},
            "vert_indices": curve.get("vert_indices"),
        }))

    if surf_ops:
        cad_patch_col.bulk_write(surf_ops, ordered=False)
    if curve_ops:
        cad_curve_col.bulk_write(curve_ops, ordered=False)

    return len(surf_ops), len(curve_ops)

# ────────────────────────── Per‑archive routine ─────────────

def process_archive(archive_path: Path):
    arch_id = archive_path.stem
    if arch_state_col.find_one({"_id": arch_id}):
        logger.debug("%s already processed – skip", arch_id)
        return False

    extract_path = UNZIP_DIR / arch_id
    extract_path.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, "r") as zf:
        zf.extractall(path=extract_path)

    inserted_stats = inserted_ops = inserted_surfs = inserted_curves = 0

    for model_folder in sorted(extract_path.iterdir()):
        if not model_folder.is_dir():
            continue
        try:
            yml_file = next(model_folder.glob("*.yml"))
        except StopIteration:
            continue

        model_id, _ = split_id(model_folder.name)
        m = re.search(r"(stat|featurescript|features)[_-]?(\d+)", yml_file.name, re.IGNORECASE)
        file_type, file_id = (m.group(1).lower(), m.group(2)) if m else ("unknown", "unknown")
        mongo_id = f"{model_id}_{file_type}_{file_id}"

        data = yaml.load(yml_file.read_text(), Loader=YamlLoader)
        low = yml_file.name.lower()

        if "_stat" in low:
            doc = {
                "_id": mongo_id,
                "model_id": model_id,
                "edges": data.get("#edges"),
                "faces": data.get("#faces"),
                "verts": data.get("#verts"),
                "surfs": data.get("#surfs"),
                "sharp_edges": data.get("#sharp"),
                "tolerance": data.get("tolerance"),
                "volume": data.get("volume"),
            }
            bbox = data.get("bbox", [])
            if len(bbox) >= 6:
                doc["bbox_min"], doc["bbox_max"] = bbox[:3], bbox[3:6]
            cad_stat_col.update_one({"_id": mongo_id}, {"$set": doc}, upsert=True)
            inserted_stats += 1

        elif "_featurescript" in low:
            if not cad_stat_col.count_documents({"model_id": model_id}, limit=1):
                continue
            operations = []
            for idx, feat in enumerate(data.get("features", [])):
                msg = feat.get("message") or {}
                ftype = msg.get("featureType")
                if not ftype:
                    continue
                params = {p["message"].get("parameterId"): (p["message"].get("value") or p["message"].get("expression"))
                          for p in msg.get("parameters", []) if "message" in p}
                operations.append({"op_index": idx, "op_type": ftype, "params": params})
            cad_op_col.update_one({"model_id": model_id}, {"$set": {"operations": operations}}, upsert=True)
            inserted_ops += 1

        elif "_features" in low:
            if not cad_op_col.count_documents({"model_id": model_id}, limit=1):
                continue
            s_cnt, c_cnt = bulk_insert_surfs_curves(model_id, data)
            inserted_surfs += s_cnt
            inserted_curves += c_cnt

    shutil.rmtree(extract_path)
    arch_state_col.insert_one({"_id": arch_id})

    logger.info("Done %-20s | stats:%4d ops:%3d surfs:%5d curves:%5d",
                arch_id, inserted_stats, inserted_ops, inserted_surfs, inserted_curves)
    return True

# ────────────────────────── Utils ───────────────────────────

def group_archives(archives):
    groups = defaultdict(list)
    for a in archives:
        key = "stat" if "_stat_" in a.stem else "ofs" if "_ofs_" in a.stem else "feat"
        groups[key].append(a)
    for k in groups:
        groups[k].sort(key=lambda p: int(re.search(r"_(\d+)_", p.stem).group(1)))
    return [groups[k] for k in ("stat", "ofs", "feat") if groups[k]]


def ensure_indexes():
    cad_stat_col.create_index("model_id")
    cad_op_col.create_index("model_id")
    cad_patch_col.create_index("model_id")

# ────────────────────────── Main ────────────────────────────

def main(auto_continue: bool, workers: int | None = None, sequential: bool = False):
    UNZIP_DIR.mkdir(parents=True, exist_ok=True)
    all_archives = sorted(RAW_DIR.glob("*.7z"))

    for group in group_archives(all_archives):
        group_name = "stat" if "_stat_" in group[0].stem else "ofs" if "_ofs_" in group[0].stem else "feat"
        logger.info("Starting %s archives (%d)…", group_name, len(group))

        if sequential:
            for arch in tqdm(group, desc=f"Processing {group_name} archives"):
                process_archive(arch)
        else:
            workers = workers or os.cpu_count() * 2
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(process_archive, a): a for a in group}
                for fut in tqdm(as_completed(futs), total=len(group), desc=f"Processing {group_name} archives"):
                    ok = fut.result()
                    if ok and not auto_continue:
                        input("Press Enter to continue…")

    ensure_indexes()
    logger.info("All archives processed. Indexes built.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimised loader for ABC dataset archives → MongoDB (grouped by type).")
    parser.add_argument("--auto", action="store_true", help="Run without pauses between archives")
    parser.add_argument("--workers", type=int, default=None, help="Thread pool size (default: 2 × CPU)")
    parser.add_argument("--sequential", action="store_true", help="Disable parallel processing — exactly one archive at a time")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging")
    args = parser.parse_args()

    if args.verbose:
        setup_logging("DEBUG")

    main(auto_continue=args.auto, workers=args.workers, sequential=args.sequential)
