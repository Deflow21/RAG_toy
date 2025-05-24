import os
import random
import time
import json

import torch
import pandas as pd
from tqdm import tqdm
from jsonschema import validate, ValidationError

# наш генератор для Phi-3-Vision-128k-Instruct
from generate_json_phi3 import generate_json_from_image_with_phi3 as gen_phi3

# --- папки и настройки ---
INPUT_DIR   = os.path.join(os.getcwd(), "чертежи")
OUTPUT_ROOT = os.path.join(os.getcwd(), "outputs", "comparisons", "Phi-3-Vision-128k-Instruct")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- JSON-Schema для валидации ---
SCHEMA = {
    "type": "object",
    "required": ["File name", "Name of operation", "Part information", "Steps"],
    "properties": {
        "File name":         {"type": "string"},
        "Name of operation": {"type": "string"},
        "Part information": {
            "type": "object",
            "required": ["Length", "Width", "Height/thickness", "Weight"]
        },
        "Steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["Step number", "Action", "Equipment", "Significance"],
                "properties": {
                    "Step number": {"type": "integer"},
                    "Action":      {"type": "string"},
                    "Equipment":   {"type": "array"},
                    "Significance": {
                        "type": "object",
                        "required": ["Length", "Width", "Height/thickness", "Weight"]
                    }
                }
            }
        }
    }
}

def main(sample_size: int = 5):
    # 1) Выбор случайных файлов
    files    = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    selected = random.sample(files, min(sample_size, len(files)))
    print(f"Выбрано файлов: {selected}\n")

    records = []

    # 2) Генерация JSON и сохранение
    for img in tqdm(selected, desc="Phi-3", leave=False):
        img_path = os.path.join(INPUT_DIR, img)
        torch.cuda.empty_cache()

        start = time.time()
        try:
            result  = gen_phi3(img_path)
            success = True
        except Exception as e:
            result  = json.dumps({"error": str(e)}, ensure_ascii=False)
            success = False
        elapsed = time.time() - start

        out_path = os.path.join(OUTPUT_ROOT, img + ".json")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(result)

        records.append({
            "model":       "Phi-3-Vision-128k-Instruct",
            "image":       img,
            "output_path": out_path,
            "time_sec":    round(elapsed, 2),
            "length_chars": len(result),
            "success":     success
        })

    # 3) Проверка валидности JSON и подсчёт шагов
    for rec in tqdm(records, desc="Quality eval", leave=False):
        try:
            data = json.loads(open(rec["output_path"], encoding='utf-8').read())
            validate(instance=data, schema=SCHEMA)
            rec["valid_json"] = True
            rec["num_steps"]  = len(data.get("Steps", []))
        except (json.JSONDecodeError, ValidationError):
            rec["valid_json"] = False
            rec["num_steps"]  = 0

    # 4) Вывод и экспорт метрик
    df = pd.DataFrame(records).sort_values("time_sec")
    print("\nМетрики для Phi-3-Vision-128k-Instruct:")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUTPUT_ROOT, "phi3_metrics.csv"), index=False)

if __name__ == "__main__":
    main(sample_size=5)
