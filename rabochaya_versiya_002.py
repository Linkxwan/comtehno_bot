import os
import base64
import uuid
import asyncio

import uvicorn
import edge_tts
import aiofiles
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from data_ml import data_set
from data_gpt import texts
import g4f


# Установка префикса для путей к файлам (по умолчанию пустая строка)
prefix = ""
# Пороговое значение для сравнения сходства векторов
my_threshold = 0.7

app = FastAPI() # Создание экземпляра FastAPI приложения

# Создание объекта TfidfVectorizer для векторизации ключей из data_set
tfidf_vectorizer = TfidfVectorizer()
# Преобразование ключей из data_set в векторную форму
tfidf_matrix = tfidf_vectorizer.fit_transform(list(data_set.keys()))

# Создание объекта TfidfVectorizer для векторизации текстов из texts
texts_vectorizer = TfidfVectorizer()
# Преобразование текстов из texts в векторную форму
texts_vectors = texts_vectorizer.fit_transform(texts)


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
    # Удаление каждого файла
    for file in files:
        file_path = os.path.join(prefix, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Файл {file_path} удален успешно.")
        else:
            print(f"Файл {file_path} не существует.")


@app.get("/get_response")
async def get_response(question: str):
    try:
        start = time.time()
        # Выводим значение ключа с наибольшим сходством
        most_similar_index, threshold = await vectorize(question, tfidf_vectorizer, tfidf_matrix)

        if threshold >= my_threshold:
            # Если порог сходства превышен, получаем ответ из базы данных
            most_similar_key = list(data_set.keys())[most_similar_index]
            response = data_set[most_similar_key]

            # Генерация речевого ответа и сохранение его в файлы
            files, unique_id  = await synthesis(response, prefix=prefix)

            # Чтение содержимого файлов асинхронно
            files_contents = await read_files(files, prefix=prefix)

            # Удаление файлов асинхронно
            await remove_files(files, prefix=prefix)

            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": response,
                "time": time.time() - start,
                "files": files,
                "files_content": files_contents
            }
        
        else:
            # Если порог сходства не превышен, формируем запрос к GPT-3
            context = texts[cosine_similarity(texts_vectors, texts_vectorizer.transform([question])).argmax()]
            template = f"""Ты - полезный ИИ ассистент для нашего колледжа комтехно.
                Используйте следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
                Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумывать ответ.
                Сначала убедитесь, что прикрепленный текст имеет отношение к вопросу.
                Если текст не имеет отношения к вопросу, просто скажите, что текст не имеет отношения.
                Используйте максимум три предложения. Держите ответ как можно более кратким.
                Context: {context}
                Question: {question}
                Helpful Answer:"""

            # Генерация речевого ответа и сохранение его в файлы
            response = await g4f.ChatCompletion.create_async(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": template}]
            )
            files, unique_id  = await synthesis(response, prefix=prefix)

            # Чтение содержимого файлов асинхронно
            files_contents = await read_files(files, prefix=prefix)

            # Удаление файлов асинхронно
            await remove_files(files, prefix=prefix)

            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": response,
                "time": time.time() - start,
                "files": files,
                "files_content": files_contents
            }
        
    except Exception as e:
        # Обработка ошибок
        print(f"Произошла ошибка: {e}")
        return {"error": str(e)}


def main():
    pass

if __name__ == "__main__":
    main()
    uvicorn.run(app, host="127.0.0.1", port=8000) 