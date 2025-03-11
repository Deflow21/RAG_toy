from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import psycopg2
from sentence_transformers import SentenceTransformer

# Инициализация модели и процессора
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    device_map='cuda', 
    torch_dtype=torch.bfloat16
)

# Функция поиска ГОСТов
def retrieve_from_db(query_text, top_k=3):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="113245",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = model_embed.encode(query_text).tolist()

    cur.execute("""
        SELECT filename, content FROM gost_documents
        ORDER BY embedding <-> %s::vector LIMIT %s;
    """, (query_embedding, top_k))

    results = cur.fetchall()
    conn.close()
    return results

def generate_json_from_image_with_rag(image_path, model, processor):
    image = Image.open(image_path).convert('RGB')

    # Шаг RAG: получаем релевантные ГОСТы
    relevant_docs = retrieve_from_db("технологический процесс изготовления детали")

    # исправляем здесь переменную
    gost_context = "\n\n".join([f"{filename}:\n{content[:1500]}" for filename, content in relevant_docs])

    prompt = '''
     Учитывая следующие ГОСТы:{} 

Создай подробное описание технологического процесса изготовления изделия 
на основе технического чертежа 3D модели в формате JPG. 
Процесс должен быть описан в формате JSON и включать в себя все этапы механической обработки, 
начиная с анализа чертежа и заканчивая получением готового изделия. 
JSON должен содержать следующую информацию:

1. Наименование файла: название входного файла с чертежом.
2. Название операции: тип основной операции (например, токарная, фрезерная, шлифовальная и т.д.).
3. Информация о детали: исходные параметры детали (длина, ширина, высота/толщина, вес).
4. Шаги: последовательность действий, выполняемых для обработки детали. Каждый шаг должен включать:
   - Номер шага.
   - Описание действия (например, "черновая обработка", "чистовая обработка", "сверление отверстий" и т.д.).
   - Используемое оборудование (например, токарный станок, фрезерный станок, измерительный инструмент и т.д.).
   - Изменения параметров детали (длина, ширина, высота/толщина, вес) после выполнения шага. 
     Если параметр не изменяется, укажите значение -1.

Дополнительно:
5. ГОСТы: список номеров ГОСТов, которые могут быть использованы при изготовлении этой детали.
   Перечисли их в массиве "ГОСТы" с номерами и кратким описанием их содержания.

Пример JSON:

{{
    "Наименование файла": "Название файла на входе",
    "Название операции": "Токарная",
    "Информация о детали": {{
        "Длина": 10,
        "Ширина": 10,
        "Высота/толщина": 10,
        "Вес": 10
    }},
    "Шаги": [
        {{
            "Номер шага": 1,
            "Действие": "Черновая обработка заготовки",
            "Оборудование": ["Токарный станок"],
            "Значение": {{
                "Длина": 9,
                "Ширина": 9,
                "Высота/толщина": 9,
                "Вес": 8
            }}
        }},
        {{
            "Номер шага": 2,
            "Действие": "Чистовая обработка поверхности",
            "Оборудование": ["Токарный станок"],
            "Значение": {{
                "Длина": 8.5,
                "Ширина": 8.5,
                "Высота/толщина": 8.5,
                "Вес": 7.5
            }}
        }}
    ],
    "ГОСТы": [
        {{
            "Номер": "ГОСТ 21488-97",
            "Описание": "Припуски на обработку деталей из металлов."
        }},
        {{
            "Номер": "ГОСТ 2789-73",
            "Описание": "Классы шероховатости поверхностей. Параметры и характеристики."
        }}
    ]
}}

Требования:
- Добавь поле "ГОСТы" со списком найденных ГОСТов и кратким описанием каждого.
- Описание каждого шага должно быть максимально точным и соответствовать реальным процессам механической обработки.
- Используемое оборудование должно быть указано с учетом специфики операции.
- Изменения параметров детали должны быть рассчитаны с учетом реальных технологических допусков.
- В ответе должен быть **только JSON**, без дополнительного текста.
    '''.format(gost_context)

    prompt += (
    "\nОтвет строго должен завершаться символом '}'. "
    "Не добавляй больше текста после закрывающей фигурной скобки."
)

    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if isinstance(text_prompt, list):
        text_prompt = text_prompt[0]

    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    for key, tensor in inputs.items():
        inputs[key] = tensor.to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    json_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return json_response


# Вызов функции
result = generate_json_from_image_with_rag('чертеж.jpg', model, processor)
print(result)