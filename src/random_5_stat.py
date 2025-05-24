import os
import json
import random
from geberate_with_RAG_stat import generate_json_from_image_with_rag_stat
from tqdm import tqdm

def process_single_file(file_path, file_name, output_file, handler):
    try:
        output = handler(file_path)
        formatted_output = output
        result = f"Файл: {file_name}\nРезультат:\n{formatted_output}\n{'='*100}\n\n"
    except Exception as e:
        result = f"Файл: {file_name}\nОшибка при обработке: {e}\n{'='*100}\n\n"
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(result)
    print(f"[OK] Готово: {file_name}")

def process_random_drawings(input_dir, output_file, handler, sample_size=5):
    # Список всех изображений
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"Всего файлов: {len(files)}. Выберем случайных {sample_size}.")
    
    # Выбираем случайную подвыборку без повторений
    selected = random.sample(files, min(sample_size, len(files)))
    
    for file in tqdm(selected, desc="Обработка выбранных файлов"):
        process_single_file(os.path.join(input_dir, file), file, output_file, handler)

if __name__ == "__main__":
    drawings_directory = os.path.join(os.getcwd(), "чертежи")
    output_results_file = "model_outputs_readable.txt"
    open(output_results_file, 'w', encoding='utf-8').close()
    
    # handler — функция-обёртка над нужной моделью
    process_random_drawings(drawings_directory, output_results_file, generate_json_from_image_with_rag_stat)
