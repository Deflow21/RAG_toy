import os
import json
from generate_json_rag import generate_json_from_image_with_rag
from tqdm import tqdm  # Импорт tqdm для прогресс-бара

def process_single_file(file_path, file_name, output_file):
    try:
        output = generate_json_from_image_with_rag(file_path)
        # Если output уже строка JSON, то json.dumps не нужен:
        # formatted_output = json.dumps(output, ensure_ascii=False, indent=4)
        formatted_output = output  # Используем результат без дополнительного преобразования
        result = f"Файл: {file_name}\nРезультат:\n{formatted_output}\n{'='*100}\n\n"
    except Exception as e:
        result = f"Файл: {file_name}\nОшибка при обработке: {e}\n{'='*100}\n\n"
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(result)
    
    print(f"[OK] Готово: {file_name}")

def process_all_drawings(input_dir, output_file):
    # Получаем список файлов с изображениями
    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"Найдено файлов: {len(files)}")

    # Последовательная обработка файлов с отображением прогресса
    for file in tqdm(files, desc="Обработка файлов"):
        process_single_file(os.path.join(input_dir, file), file, output_file)

if __name__ == "__main__":
    drawings_directory = os.path.join(os.getcwd(), "чертежи")
    output_results_file = "model_outputs_readable.txt"

    # Очищаем файл результатов перед началом обработки
    open(output_results_file, 'w', encoding='utf-8').close()

    process_all_drawings(drawings_directory, output_results_file)
