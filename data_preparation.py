import edge_tts
import uuid
import os
import asyncio
import json
import aiofiles
import base64
from data_ml import data_set
 

prefix = "voice_data"
_prefix = len(prefix)+1

async def process_chunk(voice, text_chunk, output_file):
    communicate = edge_tts.Communicate(text_chunk, voice)
    await communicate.save(output_file)


async def synthesis(data, prefix=""):
    voice = 'ru-RU-SvetlanaNeural'  # Установите желаемый голос
    unique_id = uuid.uuid4()
    created_files = []  # Список для хранения созданных файлов

    # Создаем и запускаем асинхронные потоки для каждого фрагмента текста
    output_file = os.path.join(prefix, f"data_{unique_id}_0.mp3")  # Уникальный путь к файлу
    task = asyncio.create_task(process_chunk(voice, data, output_file))

    # Дождемся завершения всех задач
    await asyncio.gather(task)

    # Соберем имена созданных файлов
    output_file = os.path.join(prefix, f"data_{unique_id}_0.mp3")
    created_files.append(output_file)

    return created_files, unique_id


async def read_files(files, prefix=""):
    # Словарь для хранения содержимого файлов в формате base64
    file_contents = {}
    for file in files:
        file_path = os.path.join(prefix, file)
        async with aiofiles.open(file_path, mode='rb') as f:
            content = await f.read()
            file_contents[file] = base64.b64encode(content).decode("utf-8")
    return file, file_contents


async def main():
    # Создание пустого словаря для хранения результатов
    results = {}

    # Цикл по ключам словаря data_set
    for key, value in data_set.items():
        # Вызов функции synthesis для каждого значения
        file, id = await synthesis(value, prefix=prefix)
        file_name, file_content = await read_files(file)
        # Удаляем ключи и сохраняем только содержимое файлов
        file_content = list(file_content.values())
        # Добавление результатов в словарь
        results[key] = {"text": value, "id": str(id), "file_path": file_name, "file_content": file_content[0]}

    # Запись результатов в файл JSON
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    print(asyncio.run(main()))
