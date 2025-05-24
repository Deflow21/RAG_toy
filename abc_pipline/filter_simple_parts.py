# filter_simple_parts.py
"""
Находит ID моделей, у которых stats.yml содержит "#parts: 1".
Сохраняет список в data/simple_parts.txt
"""

import yaml
from pathlib import Path
from config import UNZIP_DIR

OUT = Path("data/simple_parts.txt")

def is_simple(stats_file: Path) -> bool:
    with open(stats_file, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return y.get("#parts") == 1

def main():
    simple_ids = []
    #   ищем все stats.yml во всех stat-архивах
    for stats_yml in UNZIP_DIR.rglob("abc_*_stat_v00/*/stats.yml"):
        part_id = stats_yml.parent.name      # ← 00000050_80d90bf6
        if is_simple(stats_yml):
            simple_ids.append(part_id)

    OUT.write_text("\n".join(simple_ids))
    print(f"Simple parts: {len(simple_ids)} записано → {OUT}")

if __name__ == "__main__":
    main()