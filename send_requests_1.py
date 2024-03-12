import aiohttp
import asyncio
import time
import requests


start = time.time()

async def _async_request(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                print(f"Request to {url} was successful")
                return await response.text()  # Получаем текст ответа
            else:
                print(f"Request to {url} failed with status code {response.status}")
                return None
    except Exception as e:
        print(f"Request to {url} failed with error: {str(e)}")
        return None


async def async_request_ml():
    api_url = "http://127.0.0.1:8000/get_response"
    urls = [f"{api_url}?question=цена" for _ in range(1000)]
    responses = []

    async with aiohttp.ClientSession() as session:
        tasks = [_async_request(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
    
    return responses


async def async_request_gpt():
    api_url = "http://127.0.0.1:8000/get_response"
    urls = [f"{api_url}?question=какие специальности есть в колледже" for _ in range(5)]
    responses = []

    async with aiohttp.ClientSession() as session:
        tasks = [_async_request(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
    
    return responses


def sync_request_ml():
    # URL API и параметры запроса
    url = 'http://127.0.0.1:8000/get_response'
    params = {'question': 'цена'}

    # Отправка синхронного GET запроса с параметрами
    response = requests.get(url, params=params)

    # Проверка статуса ответа
    if response.status_code == 200:
        # Вывод ответа
        print(response.json())
    else:
        print('Ошибка при отправке запроса:', response.status_code)


def sync_request_gpt():
    # URL API и параметры запроса
    url = 'http://127.0.0.1:8000/get_response'
    params = {'question': 'какие специальности есть в колледже'}

    # Отправка синхронного GET запроса с параметрами
    response = requests.get(url, params=params)

    # Проверка статуса ответа
    if response.status_code == 200:
        # Вывод ответа
        print(response.status_code)
    else:
        print('Ошибка при отправке запроса:', response.status_code)


async def main():
    responses = await async_request_gpt()
    # Запись ответов в файл
    with open("ahaaha.txt", "w", encoding="utf-8") as f:
        for response in responses:
            if response:
                f.write(response + '\n')


if __name__ == "__main__":
    for i in range(100):
        sync_request_gpt()
        print("=======",i,"=======")
        for i in range(3):
            print("-" ,i, "==")
            time.sleep(1)
    # asyncio.run(main())
    # print(time.time() - start)
