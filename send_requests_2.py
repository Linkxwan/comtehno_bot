import aiohttp
import asyncio

async def _async_request(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                print(f"Request to {url} was successful")
                data = await response.json()  # Получаем JSON ответ
                return data.get('response', None)  # Возвращаем только часть ответа
            else:
                print(f"Request to {url} failed with status code {response.status}")
                return None
    except Exception as e:
        print(f"Request to {url} failed with error: {str(e)}")
        return None

async def async_request_and_write_to_file(urls, file_name):
    async with aiohttp.ClientSession() as session:
        tasks = [_async_request(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)

    # Запись ответов в файл
    with open(file_name, "w", encoding="utf-8") as f:
        for response in responses:
            if response:
                f.write(response + '\n\n\n')

async def main():
    api_url = "http://127.0.0.1:8000/get_response"
    urls = [f"{api_url}?question=кто такие пришельцы" for _ in range(30)]
    await async_request_and_write_to_file(urls, "responses.txt")

asyncio.run(main())