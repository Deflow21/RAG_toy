# RAG_toy
Игрушечный RAG для ВКР

![](pablo-james-pablo.gif)

### Структура проекта
- `GOSTS_BD.py` – Извлекает текст из PDF-файлов ГОСТов, очищает его и загружает в базу данных PostgreSQL с использованием векторного поиска (pgvector).
- `generate_json_rag.py` – Использует RAG для генерации JSON-описания технологического процесса обработки детали на основе чертежа и релевантных ГОСТов.
- `model.py` – Тестовый скрипт для генерации JSON-описания технологического процесса без RAG, используя только модель Qwen2.5-VL-7B-Instruct.

---

## Установка и настройка

### 1. Установка необходимых библиотек
```sh
pip install torch transformers sentence-transformers psycopg2 fitz[pymupdf] pillow
```

### 2. Настройка базы данных PostgreSQL
Перед началом работы необходимо настроить базу данных PostgreSQL. Убедитесь, что установлен расширенный поиск по векторным данным (pgvector):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
Создайте таблицу для хранения ГОСТов:
```sql
CREATE TABLE gost_documents (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    content TEXT,
    embedding VECTOR(384) -- Используемая модель даёт 384-мерные вектора
);
```

<p><span style="color: red; font-weight: bold;">Настройте параметры подключения к базе данных в файлах <code>GOSTS_BD.py</code> и <code>generate_json_rag.py</code>:</span></p>

<pre style="background-color: black; color: lime; padding: 10px;">
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="ВАШ_ПАРОЛЬ",
    host="localhost",
    port=5432
)
</pre>


---

## Использование

### 1. Загрузка ГОСТов в базу данных
Запустите `GOSTS_BD.py`, чтобы извлечь текст из PDF-файлов и сохранить их в PostgreSQL:
```sh
python GOSTS_BD.py
```

### 2. Генерация JSON с использованием RAG
Запустите `generate_json_rag.py`, передав изображение с чертежом:
```sh
python generate_json_rag.py
```

Скрипт:
- Извлекает релевантные ГОСТы из БД.
- Использует модель `Qwen/Qwen2.5-VL-7B-Instruct` для генерации JSON-описания процесса обработки детали.


### 3. Тестирование модели без RAG
Можно протестировать модель отдельно с помощью `model.py`:
```sh
python model.py
```

Этот скрипт просто передаёт изображение модели и получает JSON без поиска ГОСТов в БД.

