# TO DO: 1.сделать тупую версию ответов
#        2.выполнить краш тест, много запросов одновременно

import os
import subprocess
import threading
import time
import base64
import uuid

from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_gpt import texts
import g4f
 

app = FastAPI()


print("Loading...")
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(texts) 
print("Data is done!")


def speak(data):
    voice1 = 'ru-RU-SvetlanaNeural'
    rate, volume, pitch = "+0%", "+0%", "+0Hz"
    # Разделите входной текст на части
    chunks = data.split()
    chunk_size = 50
    chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
    
    unique_id = uuid.uuid4()
    # Определите функцию для обработки каждой части
    def process_chunk(chunk, index):
        filename = f"data_{unique_id}_{index}.wav"  # Используйте уникальные имена файлов
        text = ' '.join(chunk)
        command = f'edge-tts --rate={rate} --volume={volume} --pitch={pitch} --voice "{voice1}" --text "{text}" --write-media "{filename}"'

        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        process.communicate()  # Ждем завершения процесса

    # Создаем и запускаем поток для каждого фрагмента
    threads = []
    for i, chunk in enumerate(chunks):
        thread = threading.Thread(target=process_chunk, args=(chunk, i))
        thread.start()
        threads.append(thread)

    # Подождем, пока все потоки завершатся
    for thread in threads:
        thread.join()

    return unique_id


@app.get("/get_response")
async def get_response(question):
    try:
        start = time.time()
        context = texts[cosine_similarity(text_vectors, vectorizer.transform([question])).argmax()]
        # шаблон
        template = f"""Ты - полезный ИИ ассистент для нашего колледжа комтехно.
            Используйте следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
            Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумывать ответ.
            Сначала убедитесь, что прикрепленный текст имеет отношение к вопросу.
            Если текст не имеет отношения к вопросу, просто скажите, что текст не имеет отношения.
            Используйте максимум три предложения. Держите ответ как можно более кратким.
            Context: {context}
            Question: {question}
            Helpful Answer:"""

        # обращяемся по api к chat GPT
        response = await g4f.ChatCompletion.create_async(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": template}],
        )

        unique_id = speak(response)
        # Получение списка файлов с UUID в имени
        files = [filename for filename in os.listdir(".") if str(unique_id) in filename.split('_')]

        # Прочитайте содержимое каждого файла в формате base64
        file_contents = {}
        for file in files:
            with open(file, "rb") as f:
                file_contents[file] = base64.b64encode(f.read()).decode("utf-8")

        # Префикс для полного пути к файлам
        prefix = ""

        # Удаление каждого файла
        for file in files:
            file_path = os.path.join(prefix, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Файл {file_path} удален успешно.")
            else:
                print(f"Файл {file_path} не существует.")

        print(time.time() - start)

        return {
            "question": question,
            "response": response,
            "files": file_contents
        }


    except Exception as e:
        # Обработка ошибок
        print(f"Произошла ошибка: {e}") 
        

def main():
    pass


if __name__ == "__main__":
    main()
    # uvicorn rabochaya_versiya_001:app --reload    