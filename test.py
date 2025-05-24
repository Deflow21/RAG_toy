# import os
# import glob
# from pdf2image import convert_from_path
# from PIL import Image

# def convert_pdf_to_jpg(folder_path, dpi=90, quality=75, max_dimension=900, crop_percent=0.05):
#     """
#     Находит все PDF-файлы в folder_path, конвертирует каждую страницу каждого PDF в JPG
#     с заданным DPI, обрезает края изображения на заданный процент (crop_percent) и сохраняет их в ту же папку.
    
#     Если полученное изображение превышает по ширине или высоте max_dimension пикселей,
#     оно уменьшается с сохранением пропорций.
    
#     Имена выходных файлов формируются как:
#         <имя_файла>_page_<номер_страницы>.jpg
#     """
#     # Получаем список PDF-файлов в указанной папке
#     pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
#     if not pdf_files:
#         print(f"В папке {folder_path} не найдено PDF-файлов.")
#         return

#     for pdf_file in pdf_files:
#         print(f"Обработка PDF: {pdf_file} ...")
#         try:
#             # Конвертируем PDF в список изображений (одна страница — один объект Image)
#             pages = convert_from_path(pdf_file, dpi=dpi)
#             base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            
#             for i, page in enumerate(pages, start=1):
#                 width, height = page.size
#                 # Расчитываем отступы для обрезки: crop_percent с каждой стороны
#                 left = int(width * crop_percent)
#                 upper = int(height * crop_percent)
#                 right = width - left
#                 lower = height - upper
                
#                 # Применяем обрезку
#                 page = page.crop((left, upper, right, lower))
#                 print(f"  Страница {i} обрезана: ({left}, {upper}, {right}, {lower})")
                
#                 # Обновляем размеры после обрезки
#                 width, height = page.size
#                 if width > max_dimension or height > max_dimension:
#                     scaling_factor = min(max_dimension / width, max_dimension / height)
#                     new_width = int(width * scaling_factor)
#                     new_height = int(height * scaling_factor)
#                     page = page.resize((new_width, new_height), Image.LANCZOS)
#                     print(f"  Изображение уменьшено до {new_width}x{new_height} пикселей")
                
#                 output_filename = f"{base_name}_page_{i}.jpg"
#                 output_path = os.path.join(folder_path, output_filename)
#                 page.save(output_path, "JPEG", quality=quality)
#                 print(f"  Страница {i} сохранена как: {output_filename}")
#         except Exception as e:
#             print(f"Ошибка при обработке {pdf_file}: {e}")

# if __name__ == "__main__":
#     # Путь к папке "чертежи" (убедитесь, что она существует в текущем каталоге)
#     folder = os.path.join(os.getcwd(), "чертежи")
    
#     if os.path.isdir(folder):
#         convert_pdf_to_jpg(folder, dpi=100, quality=75, max_dimension=900, crop_percent=0.06)
#     else:
#         print(f"Папка не найдена: {folder}")


import json

def unescape_json_strings(obj):
    if isinstance(obj, str):
        # Заменяем литеральные последовательности "\n" на реальные переводы строк
        return obj.replace("\\n", "\n")
    elif isinstance(obj, list):
        return [unescape_json_strings(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: unescape_json_strings(value) for key, value in obj.items()}
    else:
        return obj

# Открываем и загружаем JSON-файл
with open('json_edit.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Преобразуем все строки внутри JSON
data = unescape_json_strings(data)

# Выводим каждую пару ключ-значение (каждый словарь) отдельно
for key, value in data.items():
    print(f"Dictionary for key: {key}")
    print(json.dumps(value, indent=4, ensure_ascii=False))
    print("\n" + "=" * 40 + "\n")
