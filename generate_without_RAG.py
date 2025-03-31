from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

# === 1) ИНИЦИАЛИЗАЦИЯ МОДЕЛИ QWEN2.5-VL ===
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    device_map='auto', 
    torch_dtype=torch.bfloat16
)

def generate_json_from_image(image_path):
    """
    Генерирует JSON-описание технологического процесса изготовления детали
    на основе изображения 3D-модели.
    
    RAG отключён — промпт не дополняется информацией из внешних источников.
    """
    # 1. Загружаем изображение
    image = Image.open(image_path).convert("RGB")
    
    # 2. Формируем основной промпт (без контекста из MongoDB)
    original_english_prompt = """Create a detailed description of the technological process of manufacturing a product based on an image providing three dimensions of a 3D model in JPG format. The process should be described in JSON format and include all stages of machining, starting from analysing the drawing and ending with obtaining the finished product. The response should be JSON only, with no additional text. JSON should contain the following information:
1. File name: the name of the input file with the drawing.
2. Operation name: the type of the main operation (e.g. turning, milling, grinding, etc.).
3. Part information: initial parameters of the part (length, width, height/thickness, weight).
4. Steps: the sequence of operations performed to machine the part. Each step must include:
    4.1. Step number.
    4.2. Description of the step (e.g., 'roughing', 'finishing', 'drilling holes', etc.).
    4.3. The equipment used (e.g. lathe, milling machine, measuring tool, etc.).
    4.4. Changes in the part parameters (length, width, height/thickness, weight) after the step has been executed. If the parameter does not change, specify a value of -1.

JSON example:

{
    "File name": "Input file name",
    "Name of operation": "Lathe",
    "Part information": {
        "Length": 10,
        "Width": 10,
        "Height/thickness": 10,
        "Weight": 10
    },
    "Steps": [
        {
            "Step number": 1,
            "Action": "Roughing the workpiece",
            "Equipment": ["Turning lathe"],
            "Significance": {
                "Length": 9,
                "Width": 9,
                "Height/thickness": 9,
                "Weight": 8
            }
        },
        {
            "Step number": 2,
            "Action": "Finishing of the surface",
            "Equipment": ["Turning lathe"],
            "Significance": {
                "Length": 8.5,
                "Width": 8.5,
                "Height/thickness": 8.5,
                "Weight": 7.5
            }
        }
    ]
}

Requirements:
(a) The description of each step shall be as accurate as possible and shall correspond to actual machining processes.
(b) The equipment used must be specified with respect to the specifics of the operation.
(c) Changes in part parameters must be calculated to realistic process tolerances.
(d) The response should be JSON only, with no additional text.
"""

    # 3. Формируем финальный промпт, добавляя информацию о файле и виде чертежа
    prompt = (
        "Here is a technical drawing showing multiple orthographic views of the part:\n"
        " - FrontView006 (top-left)\n"
        " - BackView006 (top-center)\n"
        " - LeftView006 (right side)\n"
        " - RightView006 (left side)\n"
        " - TopView006 (center)\n"
        " - BottomView006 (bottom)\n\n"
        "Now follow these instructions:\n"
        f"{original_english_prompt}\n\n"
        f"File name (input): {image_path}\n\n"
        "Important:\n"
        "- Provide only JSON, ending with the '}' character (no extra text).\n"
        "Your answer must end with '}'. No more text after the closing brace.\n"
    )

    # 4. Формируем сообщения для модели Qwen2.5-VL
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

    # 5. Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    # 6. Декодируем результат
    json_response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return json_response

# === 7) Пример запуска ===
if __name__ == "__main__":
    # Предполагается, что имеется файл "fng.jpg"
    result = generate_json_from_image("fng.jpg")
    print(result)
