import os
import base64
import uuid
import asyncio
import json
import logging

import uvicorn
import edge_tts
import aiofiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from data_ml import data_set
from data_gpt import texts
import g4f


# Настройка логгера
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация переменных и настроек
MAX_CACHE_SIZE = 1000  # Максимальный размер кэша
MAX_QUESTION_HISTORY_SIZE = 1000  # Максимальный размер списка
prefix_gpt = "voice_data_gpt"  # Префикс для путей к файлам для GPT
prefix_ml = "voice_data"  # Префикс для путей к файлам для ML
my_threshold = 0.7  # Пороговое значение для сравнения сходства векторов

# Создаем пустой словарь для кэширования результатов
cache = {}
# Создаем пустой список для хранения значений переменной question
question_history = []

# Инициализация FastAPI приложения
app = FastAPI()


# Логируем успешную инициализацию переменных и настроек
logging.info("Успешно инициализированы переменные и настройки.")


# Создание объекта TfidfVectorizer для векторизации ключей из data_set
tfidf_vectorizer = TfidfVectorizer()
# Преобразование ключей из data_set в векторную форму
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(list(data_set.keys()))
    logging.info("Успешно выполнено векторизация ключей из data_set")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации ключей из data_set: {e}")

# Создание объекта TfidfVectorizer для векторизации текстов из texts
texts_vectorizer = TfidfVectorizer()
# Преобразование текстов из texts в векторную форму
try:
    texts_vectors = texts_vectorizer.fit_transform(texts)
    logging.info("Успешно выполнено векторизация текстов из texts")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации текстов из texts: {e}")


def read_json(file_path):
    """
    Считывает JSON-файл и возвращает его содержимое в виде словаря.

    Параметры:
    - file_path (str): Путь к JSON-файлу.

    Возвращает:
    dict: Содержимое JSON-файла в виде словаря.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        logging.info(f"JSON-файл {file_path} успешно прочитан.")
        return data
    except FileNotFoundError:
        logging.error(f"JSON-файл {file_path} не найден.")
        return None
    except Exception as e:
        logging.error(f"Ошибка при чтении JSON-файла {file_path}: {e}")
        return None

# Считываем базу данных
data_json = read_json("data.json")


async def process_chunk(voice, text_chunk, output_file):
    """
    Обработка фрагмента текста и генерация речи.

    Параметры:
    - voice (str): Голос для синтеза речи.
    - text_chunk (str): Фрагмент текста для синтеза речи.
    - output_file (str): Путь к файлу, в который будет сохранена речь.

    Возвращает:
    Ничего.
    """
    communicate = edge_tts.Communicate(text_chunk, voice)
    await communicate.save(output_file)


async def synthesis(data, prefix=""):
    """
    Синтез речи на основе входных данных.

    Параметры:
    - data (str): Входные данные для синтеза речи.
    - prefix (str): Префикс для путей к файлам (по умолчанию пустая строка).

    Возвращает:
    created_files (list): Список созданных файлов с речью.
    unique_id (uuid.UUID): Уникальный идентификатор синтеза речи.
    """
    voice = 'ru-RU-SvetlanaNeural'  # Установите желаемый голос
    unique_id = uuid.uuid4()
    created_files = []  # Список для хранения созданных файлов

    # Разделите входной текст на части
    chunks = data.split()
    chunk_size = 50
    chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]

    # Создаем и запускаем асинхронные потоки для каждого фрагмента текста
    tasks = []
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(prefix, f"data_{unique_id}_{i}.mp3")  # Уникальный путь к файлу
        task = asyncio.create_task(process_chunk(voice, " ".join(chunk), output_file))
        tasks.append(task)

    # Дождемся завершения всех задач
    await asyncio.gather(*tasks)

    # Соберем имена созданных файлов
    for i, _ in enumerate(chunks):
        output_file = os.path.join(prefix, f"data_{unique_id}_{i}.mp3")
        created_files.append(output_file)

    return created_files, unique_id


async def vectorize(question, tfidf_vectorizer, tfidf_matrix):
    """
    Векторизация вопроса и сравнение его с данными из базы.

    Параметры:
    - question (str): Вопрос, который необходимо векторизовать и сравнить.
    - tfidf_vectorizer (TfidfVectorizer): Объект TfidfVectorizer для векторизации текста.
    - tfidf_matrix (sparse matrix): Матрица TF-IDF, представляющая данные из базы.

    Возвращает:
    - most_similar_index (int): Индекс наиболее похожего вопроса из базы.
    - similarity (float): Процент сходства между вопросом и наиболее похожим вопросом из базы.
    """
    # Векторизация вопроса
    question_vector = tfidf_vectorizer.transform([question])

    # Вычисление косинусного сходства между вопросом и данными из базы
    cosine_similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    # Находим индекс наиболее похожего вопроса
    most_similar_index = cosine_similarities.argmax()
    
    # Возвращаем индекс и процент сходства
    return most_similar_index, cosine_similarities[most_similar_index]


# Функция для дополнения файла новыми значениями переменной question
async def append_question_history_to_file():
    async with aiofiles.open("question_history.txt", "a", encoding="utf-8") as file:
        for question in question_history:
            await file.write(question + "\n")


async def read_files(files, prefix=""):
    """
    Асинхронно читает содержимое файлов и кодирует его в формат base64.

    Параметры:
    - files (list): Список имен файлов, которые нужно прочитать.
    - prefix (str): Префикс для путей к файлам (по умолчанию пустая строка).

    Возвращает:
    - file_contents (dict): Словарь, содержащий содержимое файлов в формате base64.
    """
    # Словарь для хранения содержимого файлов в формате base64
    file_contents = {}
    for file in files:
        file_path = os.path.join(prefix, file)
        async with aiofiles.open(file_path, mode='rb') as f:
            content = await f.read()
            file_contents[file] = base64.b64encode(content).decode("utf-8")
    return file_contents


async def remove_files(files, prefix=""):
    """
    Асинхронно удаляет файлы из указанного каталога.

    Параметры:
    - files (list): Список имен файлов, которые нужно удалить.
    - prefix (str): Префикс для путей к файлам (по умолчанию пустая строка).

    Возвращает:
    Ничего.
    """
    try:
        # Удаление каждого файла
        for file in files:
            file_path = os.path.join(prefix, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Файл {file_path} удален успешно.")
            else:
                logging.warning(f"Файл {file_path} не существует.")
    except Exception as e:
        # Логирование ошибки, если что-то пошло не так
        logging.error(f"Ошибка при удалении файлов: {e}")


# Функция для логирования остановки сервера
@app.on_event("shutdown")
async def shutdown_event():
    await append_question_history_to_file()
    logging.info("Сервер остановлен.")


@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        # Здесь вы должны указать путь к файлу на сервере
        file_path = f"{prefix_ml}/{filename}"
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Логируем успешную загрузку файла
        logging.info(f"Файл {filename} был успешно загружен")

        return FileResponse(file_path)
    
    except Exception as e:
        # Обработка ошибок
        logging.error(f"Произошла ошибка при загрузке файла {filename}: {e}")
        return {"error": str(e)}


@app.get("/get_response")
async def get_response(question: str):
    try:
        start = time.time()
        # Добавляем значение question в историю
        question_history.append(question)

        # Если размер списка достиг максимального значения, записываем его в файл и очищаем
        if len(question_history) >= MAX_QUESTION_HISTORY_SIZE:
            asyncio.create_task(append_question_history_to_file())
            question_history.clear()

        if question in cache:
            most_similar_index, threshold = cache[question]
            # Логируем успешное получение ответа из кэша
            logging.info(f"Успешно получен ответ из кэша для вопроса: {question}")
        else:
            # выводим значение ключа с наибольшим сходством и сохраняем результат в кэше
            most_similar_index, threshold = await vectorize(question, tfidf_vectorizer, tfidf_matrix)
            cache[question] = (most_similar_index, threshold)

            # Удаляем старые записи, если кэш превышает максимальный размер
            if len(cache) > MAX_CACHE_SIZE:
                # Удаляем самую старую запись из кэша
                oldest_question = min(cache, key=cache.get)
                del cache[oldest_question]

        if threshold >= my_threshold:
            # Если порог сходства превышен, получаем ответ из базы данных
            most_similar_key = list(data_set.keys())[most_similar_index]
            response = data_set[most_similar_key]

            # Проверяем, существует ли файл для этого ответа в папке voice_data
            json_item = data_json[most_similar_key]
            logging.info(f"Успешно получен ответ для вопроса: {question}")

            # Возвращаем ответ с данными и метаданными
            return {
                    "question": question,
                    "response": response,
                    "time": time.time() - start,
                    "file_path": json_item["file_path"], 
                    "file_content": json_item["file_content"],
                }
        
        else:
            # Если порог сходства не превышен, формируем запрос к GPT-3
            context = texts[cosine_similarity(texts_vectors, texts_vectorizer.transform([question])).argmax()]
            template = f"""Ты - полезный ИИ ассистент для нашего колледжа комтехно (comtehno).
                Используйте следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
                Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумывать ответ.
                Сначала убедитесь, что прикрепленный текст имеет отношение к вопросу.
                Если текст не имеет отношения к вопросу, просто скажите, что текст не имеет отношения.
                Используйте максимум три предложения. Держите ответ как можно более кратким. Отвечай с уважением.
                Context: {context}
                Question: {question}
                Helpful Answer:"""

            # Генерация речевого ответа и сохранение его в файлы
            response = await g4f.ChatCompletion.create_async(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": template}]
            )
            files, unique_id  = await synthesis(response, prefix=prefix_gpt)

            # Чтение содержимого файлов асинхронно
            files_contents = await read_files(files)

            # Удаление файлов асинхронно
            await remove_files(files)

            logging.info(f"Успешно получен GPT ответ для вопроса: {question}")

            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": response,
                "time": time.time() - start,
                "files": files,
                "files_content": files_contents
            }
        
    except Exception as e:
        # Логируем ошибку
        logging.error(f"Произошла ошибка при обработке вопроса: {question}. Ошибка: {e}")
        return {"error": str(e)}


def main():
    pass

if __name__ == "__main__":
    main()
    uvicorn.run(app, host="127.0.0.1", port=8000)