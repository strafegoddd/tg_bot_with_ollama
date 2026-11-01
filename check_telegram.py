import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def main():
    token = os.getenv("BOT_TOKEN")

    if not token:
        print("Ошибка: BOT_TOKEN не найден в .env")
        return

    url = f"https://api.telegram.org/bot{token}/getMe"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20) as response:
                print("HTTP статус:", response.status)
                print("Ответ Telegram:")
                print(await response.text())

    except Exception as error:
        print("Ошибка соединения:", repr(error))

asyncio.run(main())