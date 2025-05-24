import asyncio
import os
import glob

# Предполагаем, что '7z' доступен из PATH:
SEVEN_Z_EXE = r"E:\7-Zip\7z.exe"

# Относительные пути к папкам (можно изменить под себя).
SOURCE_DIR = "./feat"
DEST_DIR = "./feat_unzip"

# Максимальное число распаковок одновременно
MAX_CONCURRENT_EXTRACT = 10

sem = asyncio.Semaphore(MAX_CONCURRENT_EXTRACT)

async def extract_7z(file_path: str, out_dir: str):
    """
    Асинхронно распаковать один .7z-файл в out_dir через подпроцесс 7z.exe.
    """
    async with sem:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Формируем команду для subprocess
        cmd = [
            SEVEN_Z_EXE,
            "x",                 # команда "извлечь"
            file_path,
            f"-o{out_dir}",      # куда распаковать
            "-y"                 # авто-подтвердить все диалоги
        ]
        print(f"Распаковка: {os.path.basename(file_path)}")

        # Запускаем 7z как асинхронный процесс
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"[ОШИБКА] Файл: {file_path}\n{stderr.decode('utf-8')}")
        else:
            print(f"  -> Готово: {os.path.basename(file_path)}")


async def main():
    # Ищем все .7z в исходной папке
    pattern = os.path.join(SOURCE_DIR, "*.7z")
    archives = glob.glob(pattern)

    if not archives:
        print("Не найдено ни одного .7z в папке", SOURCE_DIR)
        return

    # Формируем задачи на распаковку
    tasks = []
    for file_path in archives:
        tasks.append(asyncio.create_task(extract_7z(file_path, DEST_DIR)))

    # Запускаем все задачи параллельно (ограничивает semaphore)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
