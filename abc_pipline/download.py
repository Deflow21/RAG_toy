import requests
from pathlib import Path
from tqdm.auto import tqdm
from config import URL_BATCHES, RAW_DIR

def download_file(url: str, fname: str, dest_dir: Path):
    """
    Скачивает один файл по url в папку dest_dir с именем fname.
    Отображает прогресс в байтах через tqdm.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / fname

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B", unit_scale=True,
            desc=fname, leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    return out_path

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(URL_BATCHES), desc="Archives", unit="file") as outer_pbar:
        for url, fname in URL_BATCHES:
            try:
                download_file(url, fname, RAW_DIR)
            except Exception as e:
                tqdm.write(f"[ERROR] {url} → {e}")
            outer_pbar.update(1)

if __name__ == "__main__":
    main()
