import os
import shutil

# Папка, в которой будем удалять папки
folder = "stat"

# Получаем список всех подпапок в `feat/`
all_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

# Оставляем только первые 10, остальные удаляем
dirs_to_delete = all_dirs[1000:]  # Берём всё, кроме первых 10

print(f"Всего папок в '{folder}': {len(all_dirs)}")
print(f"Удаляем {len(dirs_to_delete)} папок...")

for dir_name in dirs_to_delete:
    dir_path = os.path.join(folder, dir_name)
    shutil.rmtree(dir_path)  # Удаляем папку со всем содержимым
    print(f"[✓] Удалено: {dir_path}")

print("Готово! Остались только первые 10 папок.")
