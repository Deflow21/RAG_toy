import os
import re
import fitz  # PyMuPDF
import psycopg2
from sentence_transformers import SentenceTransformer

# Очистка текста: убираем лишние пробелы, спецсимволы и дубликаты строк
def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")  # Убираем переводы строк и табуляцию
    text = re.sub(r'\s+', ' ', text).strip()  # Убираем лишние пробелы
    return text

# Извлечение текста из PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ' '.join([page.get_text() for page in doc])
    return clean_text(text)

# Очистка БД перед загрузкой (если нужно)
def clear_database():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="113245",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
    
    cur.execute("DELETE FROM gost_documents;")  # Полная очистка таблицы
    conn.commit()
    cur.close()
    conn.close()
    print("🗑️ База данных очищена перед загрузкой!")

# Папка с файлами ГОСТов
folder_path = r"C:\Users\culic\Desktop\ГОСТЫ"
gost_texts = {}

# Обрабатываем все файлы в папке
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.pdf'):
        file_path = os.path.join(folder_path, filename)
        text = extract_text(file_path)

        if len(text) > 50:  # Фильтруем пустые или слишком короткие файлы
            gost_texts[filename] = text
            print(f"✅ {filename} - текст успешно извлечён.")
        else:
            print(f"⚠️ {filename} - текст слишком короткий, пропускаем.")

# Подключаем модель для генерации embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gost_embeddings = {
    filename: model.encode(text).tolist()
    for filename, text in gost_texts.items()
}

print("✅ Embeddings успешно сгенерированы.")

# Запись в БД
def insert_into_db(filename, content, embedding):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="113245",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    # Проверяем, есть ли этот файл в базе (чтобы не было дубликатов)
    cur.execute("SELECT COUNT(*) FROM gost_documents WHERE filename = %s;", (filename,))
    result = cur.fetchone()
    
    if result[0] == 0:
        cur.execute("""
            INSERT INTO gost_documents (filename, content, embedding)
            VALUES (%s, %s, %s);
        """, (filename, content, embedding))
        conn.commit()
        print(f"✅ {filename} добавлен в БД.")
    else:
        print(f"⚠️ {filename} уже есть в БД, пропускаем.")

    cur.close()
    conn.close()

# Очищаем базу перед записью (если нужно)
clear_database()

# Записываем ГОСТы в базу
for filename, embedding in gost_embeddings.items():
    insert_into_db(filename, gost_texts[filename], embedding)

print("✅ Все ГОСТы успешно загружены в БД.")
