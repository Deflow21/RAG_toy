import py7zr
from pathlib import Path

archive_path = Path(r"C:\Users\culic\RAG_toy-3\data\raw\abc_0000_ofs_v00.7z")

with py7zr.SevenZipFile(archive_path, 'r') as z:
    files = z.getnames()
    print(f"Файлы в архиве ({len(files)}):")
    for f in files[:10]:  # первые 10 файлов
        print(f"  - {f}")
        if "ofs" in f.lower():
            print("    ⚠️ Содержит 'ofs' в имени")
