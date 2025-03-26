from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# === 1) ИНИЦИАЛИЗАЦИЯ МОДЕЛИ QWEN2.5-VL ===
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    device_map='auto', 
    torch_dtype=torch.bfloat16
)

# === 2) ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА БЕЗ RAG ===
def generate_json_from_image(image_path):
    """
    Генерирует JSON-описание технологического процесса изготовления детали,
    используя только исходный промпт и изображение.
    """
    # 2.1. Загружаем изображение
    image = Image.open(image_path).convert("RGB")
    
    # 2.2. Исходный промпт
    original_prompt = (
        "Create a detailed description of the technological process of manufacturing a product "
        "based on an image providing three dimensions of a 3D model in JPG format. The process "
        "should be described in JSON format and include all stages of machining, starting from "
        "analysing the drawing and ending with obtaining the finished product. The response should "
        "be JSON only, with no additional text. JSON should contain the following information:\n"
        "1. File name: the name of the input file with the drawing.\n"
        "2. Operation name: the type of the main operation (e.g. turning, milling, grinding, etc.).\n"
        "3. part information: initial parameters of the part (length, width, height/thickness, weight).\n"
        "4. Steps: the sequence of operations performed to machine the part. Each step must include:\n"
        "   4.1. Step number.\n"
        "   4.2 Description of the step (e.g., ‘roughing’, ‘finishing’, ‘drilling holes’, etc.).\n"
        "   4.3 The equipment used (e.g. lathe, milling machine, measuring tool, etc.).\n"
        "   4.4 Changes in the part parameters (length, width, height/thickness, weight) after the step "
        "has been executed. If the parameter does not change, specify a value of -1.\n\n"
        "JSON example:\n\n"
        "{\n"
        '  "File name": "Input file name",\n'
        '  "Name of operation": "Lathe",\n'
        '  "Part information": {\n'
        '    "Length": 10,\n'
        '    "Width": 10,\n'
        '    "Height/thickness": 10,\n'
        '    "Weight": 10\n'
        "  },\n"
        '  "Steps": [\n'
        "    {\n"
        '      "Step number": 1,\n'
        '      "Action": "Roughing the workpiece",\n'
        '      "Equipment": ["Turning lathe"],\n'
        '      "Significance": {\n'
        '         "Length": 9,\n'
        '         "Width": 9,\n'
        '         "Height/thickness": 9,\n'
        '         "Weight": 8\n'
        "      }\n"
        "    },\n"
        "    {\n"
        '      "Step number": 2,\n'
        '      "Action": "Finishing of the surface",\n'
        '      "Equipment": ["Turning lathe"],\n'
        '      "Significance": {\n'
        '         "Length": 8.5,\n'
        '         "Width": 8.5,\n'
        '         "Height/thickness": 8.5,\n'
        '         "Weight": 7.5\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Requirements:\n"
        "(a) The description of each step shall be as accurate as possible and shall correspond to actual machining processes.\n"
        "(b) The equipment used must be specified with respect to the specifics of the operation.\n"
        "(c) Changes in part parameters must be calculated to realistic process tolerances.\n"
        "(d) The response should be JSON only, with no additional text."
    )
    
    # Добавляем имя файла и требование завершения JSON символом '}'
    prompt = (
        f"{original_prompt}\n\n"
        f"File name (input): {image_path}\n\n"
        "Important:\n"
        "- Provide only JSON, ending with the '}' character (no extra text).\n"
        "Your answer must end with '}'. No more text after the closing brace.\n"
    )
    
    # 2.3. Формируем сообщения для Qwen2.5-VL
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]]
    
    # Преобразуем сообщения в текстовый промпт для модели
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(text_prompt, list):
        text_prompt = text_prompt[0]
    
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    for key, tensor in inputs.items():
        inputs[key] = tensor.to(model.device)
    
    # 2.4. Генерация ответа
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # 2.5. Декодируем ответ
    json_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return json_response

# === 3) Пример запуска ===
if __name__ == "__main__":
    # Допустим, у вас есть файл fng.jpg (2D проекция)
    result = generate_json_from_image("fng.jpg")
    print(result)
