#!/usr/bin/env python3
"""
Mini-RAG 2-коллекции: stat + operation → JSON-техпроцесс
========================================================
• Читает контекст из коллекций **cad_stat** и **cad_operation**
• Принимает JPG/PNG **или PDF** (рендерит 1-ю страницу)
• Ищет в MongoDB/Qdrant одним индексом над обеими коллекциями
• Модель: Qwen-2-5-VL-7B-Instruct + sentence-transformers

Запуск
------
1. Отредактируйте блок USER CONFIG
2. `python rag_minimal.py`
"""
from __future__ import annotations

import json, os, re, sys, gc
from pathlib import Path
from typing import Any, Dict, List

from pathlib import Path

import numpy as np, torch
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pdf2image import convert_from_path
import regex as re
from qwen_vl_utils import process_vision_info 

# ─────────────── USER CONFIG ───────────────
IMAGE_PATH = r"C:\Users\culic\RAG_toy-3\abc_pipline\drawings\00009735.pdf"
USER_QUERY = (
    "3D model manufacturing process machining operations surfaces curves "
    "dimensions tolerances materials roughing finishing drilling milling turning"
)

USE_PROMPT_FILE = True
PROMPT_FILE     = r"C:\Users\culic\RAG_toy-3\abc_pipline\promt_2.txt"

TEMPERATURES  = [0.25]
PAGE_IDX      = 0          # страница PDF (0-based)
MODEL_ID      = None       # фильтрация по model_id (или None)
EVAL_MATRIX   = False      # печатать матрицу сходства
# ───────────────────────────────────────────

# ─────────────── infrastructure ───────────────
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

# Qdrant helper (локальный контейнер 6333)
from qdrant_helper import init_collection, search as qdrant_search
init_collection()                    # создаём коллекцию, если её нет

print("🔄  loading models …")
# вынес embedder на CPU, чтобы освободить память GPU; меняйте на "cuda", если хватает VRAM
embedder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
llm       = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)

# ────────────── system prompt  ──────────────
SYS_PROMPT = (
    "You are a senior manufacturing process engineer. "
    "Reply with ONE valid JSON object only. "
    "No markdown, no commentary, nothing after the final '}'."
)

# ──────── JSON-экстрактор на всякий случай ────────
JSON_RE = re.compile(r"\{(?:[^{}]|(?R))*\}")

def first_json(text: str) -> Dict[str, Any]:
    """Возвращает первый валидный JSON-объект в тексте."""
    for m in JSON_RE.finditer(text):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model output")

# ─────────────── helpers ────────────────
def load_image(path: str, max_side: int = 1024) -> Image.Image:
    """Открываем картинку / рендерим PDF и даунскейлим до max_side."""
    suf = Path(path).suffix.lower()
    if suf == ".pdf":
        pages = convert_from_path(path, dpi=120, fmt="png")  # dpi↓
        img = pages[PAGE_IDX % len(pages)].convert("RGB")
    else:
        img = Image.open(path).convert("RGB")

    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def retrieve_docs(query: str, top_k: int = 15) -> List[Dict[str, Any]]:
    """Векторный поиск в Qdrant + фильтр по MODEL_ID (если задан)."""
    q_vec = embedder.encode(query, convert_to_numpy=True).tolist()
    hits  = qdrant_search(q_vec, top_k=top_k * 3)            # лёгкий over-fetch

    if MODEL_ID:
        hits = [h for h in hits if h[0] == MODEL_ID]

    docs = []
    for model_id, content, score, doc_type in hits[:top_k]:
        docs.append({
            "model_id": model_id,
            "content":  content,
            "score":    score,
            "doc_type": doc_type,
        })
    return docs

PROMPT_TEMPLATE = ""

def get_prompt_tpl() -> str:
    if USE_PROMPT_FILE:
        return Path(PROMPT_FILE).read_text(encoding="utf-8")
    return PROMPT_TEMPLATE  # переменная должна быть где-то объявлена


def make_user_prompt(ctx: str, filename: str) -> str:
    """
    Подставляем контекст и имя файла,
    а также убираем **жирное** и -----, чтобы не триггерить markdown-вывод.
    """
    tpl = get_prompt_tpl()
    tpl = tpl.replace("**", "").replace("-----", "")
    return tpl.replace("{CTX}", ctx).replace("{FILENAME}", filename)


def build_ctx(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for d in docs:
        tag = "STAT" if d["doc_type"] == "stat" else "OP"
        parts.append(
            f"[{tag} id={d['model_id']} score={d['score']:.3f}]\n"
            f"{d['content'][:1200]}"
        )
    return "\n\n".join(parts) or "No matching documents."


def generate_one(temp: float) -> Dict[str, Any]:
    # ── готовим контекст ───────────────────────────────────
    docs   = retrieve_docs(USER_QUERY, top_k=20)
    ctx    = build_ctx(docs)
    prompt = make_user_prompt(ctx, Path(IMAGE_PATH).name)

    # ── готовим сообщение ─────────────────────────────────
    img  = load_image(IMAGE_PATH)

    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user",
        "content": [
            {"type": "image"},                     # только «заглушка»
            {"type": "text", "text": prompt}
        ]}
    ]

    chat = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    inp = processor(
        text=[chat],
        images=[img],    #  гарантированно тот же порядок
        return_tensors="pt"
    ).to(llm.device)

    gc.collect()
    torch.cuda.empty_cache()

    # ── генерация ─────────────────────────────────────────
    eos_id = processor.tokenizer.eos_token_id
    out = llm.generate(
        **inp,
        max_new_tokens=1600,
        do_sample=True,
        temperature=temp,
        top_p=0.95,
        eos_token_id=eos_id,
        pad_token_id=eos_id
    )

    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    return first_json(raw)


def sim(a: str, b: str) -> float:
    e = embedder.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(e[0], e[1]))

# ─────────────────── main ──────────────────
# ─────────────────── main ──────────────────
if __name__ == "__main__":
    from pathlib import Path

    results: Dict[float, Dict[str, Any]] = {}

    # Output directory for JSON results
    output_dir = Path(r"C:\Users\culic\RAG_toy-3\abc_pipline\outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for t in TEMPERATURES:
        print(f"\n🚀  Generating (temperature={t}) …")
        try:
            res = generate_one(t)
        except Exception as exc:
            print(f"❌  generation failed: {exc}")
            continue

        # Build base filename: PDF stem + temperature as three digits
        stem     = Path(IMAGE_PATH).stem               # e.g. "00005418"
        temp_str = f"{int(t * 100):03d}"               # e.g. 0.25 -> "025"
        base     = f"{stem}_{temp_str}"                # "00005418_025"

        # Determine output path, bumping counter on name collisions
        out_path = output_dir / f"{base}.json"
        if out_path.exists():
            counter = 1
            while True:
                candidate = output_dir / f"RAG_{base}_{counter}.json"
                if not candidate.exists():
                    out_path = candidate
                    break
                counter += 1

        # Write JSON to the chosen path
        out_path.write_text(
            json.dumps(res, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"✔️  saved → {out_path}")
        results[t] = res

    # ── при необходимости считаем матрицу сходства ─────────
    if EVAL_MATRIX and len(results) > 1:
        ks = sorted(results)
        print("\nCosine similarity matrix")
        print("     " + "  ".join(f"{k:>6.2f}" for k in ks))
        for i in ks:
            row = [
                sim(
                    json.dumps(results[i], ensure_ascii=False),
                    json.dumps(results[j], ensure_ascii=False)
                )
                for j in ks
            ]
            print(f"{i:>4.2f} " + "  ".join(f"{v:6.3f}" for v in row))
