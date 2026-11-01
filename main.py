
import asyncio
import json
import os
import sqlite3
from rag import search_documents
from datetime import datetime
from agent import run_agent

import aiohttp

from aiogram import Bot, Dispatcher, F, BaseMiddleware
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiohttp import web
from aiogram.webhook.aiohttp_server import (
    SimpleRequestHandler,
    setup_application
)

from dotenv import load_dotenv


load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

DB_PATH = "users.db"
RESULTS_FILE = "survey_results.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

MAX_HISTORY_MESSAGES = 10

conversation_history = {}

if not BOT_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в файле .env")


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


class Survey(StatesGroup):
    waiting_for_name = State()
    waiting_for_age = State()
    waiting_for_occupation = State()


main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Начать опрос")]
    ],
    resize_keyboard=True
)


def init_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def register_user(user):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT OR IGNORE INTO users (
                user_id,
                username,
                first_name,
                last_name
            )
            VALUES (?, ?, ?, ?)
        """, (
            user.id,
            user.username,
            user.first_name,
            user.last_name
        ))


def get_user(user_id):
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        return connection.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()


def get_users_count():
    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]


def get_all_users():
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        return connection.execute("""
            SELECT user_id, username, first_name, last_name, registered_at
            FROM users
            ORDER BY registered_at DESC
        """).fetchall()


def save_result(result):
    results = []

    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as file:
                results = json.load(file)
        except (json.JSONDecodeError, OSError):
            results = []

    results.append(result)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


class RegistrationMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: Message, data: dict):
        if event.from_user:
            register_user(event.from_user)

        return await handler(event, data)


dp.message.outer_middleware(RegistrationMiddleware())

async def ask_ollama_with_context(user_id, user_text):
    documents = await search_documents(user_text, limit=4)

    if not documents:
        return (
            "В базе документации пока нет информации для ответа. "
            "Проверьте, что документы добавлены и проиндексированы."
        )

    context_parts = []
    sources = []

    for document in documents:
        source = document["source"]
        chunk = document["chunk"]

        context_parts.append(
            f"Источник: {source}, фрагмент: {chunk}\n"
            f"{document['text']}"
        )

        source_label = f"{source} (фрагмент {chunk})"

        if source_label not in sources:
            sources.append(source_label)

    context = "\n\n".join(context_parts)

    system_prompt = (
        "Ты помощник, который отвечает на вопросы по документации.\n"
        "Используй исключительно информацию из переданного контекста.\n"
        "Не добавляй сведения из собственных знаний.\n"
        "Если контекст не содержит ответа, прямо скажи, "
        "что в предоставленной документации недостаточно информации.\n"
        "Не придумывай источники и факты.\n"
        "Отвечай на русском языке."
    )

    history = conversation_history.get(user_id, [])

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        *history,
        {
            "role": "user",
            "content": (
                f"Контекст документации:\n{context}\n\n"
                f"Вопрос пользователя:\n{user_text}"
            )
        }
    ]

    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": messages,
                "stream": False
            }
        ) as response:
            response.raise_for_status()
            data = await response.json()

    answer = data["message"]["content"].strip()

    if not answer:
        raise ValueError("Модель вернула пустой ответ.")

    history.extend([
        {
            "role": "user",
            "content": user_text
        },
        {
            "role": "assistant",
            "content": answer
        }
    ])

    conversation_history[user_id] = history[-10:]

    answer += "\n\nИсточники:\n"
    answer += "\n".join(f"- {source}" for source in sources)

    return answer


async def ask_ollama(user_id, user_text):
    history = conversation_history.get(user_id, [])

    messages = [
        {
            "role": "system",
            "content": (
                "Ты полезный русскоязычный помощник. "
                "Отвечай понятно, грамотно и по существу."
            )
        },
        *history,
        {
            "role": "user",
            "content": user_text
        }
    ]

    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }
        ) as response:
            response.raise_for_status()
            data = await response.json()

    answer = data["message"]["content"].strip()

    if not answer:
        raise ValueError("Модель вернула пустой ответ.")

    history.extend([
        {
            "role": "user",
            "content": user_text
        },
        {
            "role": "assistant",
            "content": answer
        }
    ])

    conversation_history[user_id] = history[-MAX_HISTORY_MESSAGES:]

    return answer


@dp.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()

    await message.answer(
        "Здравствуйте! Я Марат-бот.\n\n"
        "Я умею отвечать на вопросы с помощью искусственного интеллекта.\n\n"
        "Доступные команды:\n"
        "/survey — начать опрос\n"
        "/profile — посмотреть профиль\n"
        "/stats — статистика пользователей\n"
        "/myid — узнать свой Telegram ID\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить опрос",
        reply_markup=main_keyboard
    )


@dp.message(Command("myid"))
async def get_my_id(message: Message):
    await message.answer(
        f"Твой Telegram ID: {message.from_user.id}"
    )


@dp.message(Command("stats"))
async def stats(message: Message):
    count = get_users_count()

    await message.answer(
        f"Количество зарегистрированных пользователей: {count}"
    )


@dp.message(Command("profile"))
async def profile(message: Message):
    user = get_user(message.from_user.id)

    if user is None:
        await message.answer("Профиль не найден.")
        return

    username = (
        f"@{user['username']}"
        if user["username"]
        else "Не указан"
    )

    first_name = user["first_name"] or "Не указано"
    last_name = user["last_name"] or "Не указана"

    await message.answer(
        "Ваш профиль:\n\n"
        f"Telegram ID: {user['user_id']}\n"
        f"Username: {username}\n"
        f"Имя: {first_name}\n"
        f"Фамилия: {last_name}\n"
        f"Дата регистрации: {user['registered_at']}"
    )


@dp.message(Command("users"))
async def users(message: Message):
    if ADMIN_ID == 0 or message.from_user.id != ADMIN_ID:
        await message.answer(
            "У вас нет доступа к этой команде."
        )
        return

    all_users = get_all_users()

    if not all_users:
        await message.answer(
            "В базе данных пока нет пользователей."
        )
        return

    await message.answer(
        f"Всего пользователей: {len(all_users)}"
    )

    chunk = ""

    for user in all_users:
        username = (
            f"@{user['username']}"
            if user["username"]
            else "Не указан"
        )

        first_name = user["first_name"] or "Не указано"
        last_name = user["last_name"] or "Не указана"

        line = (
            f"ID: {user['user_id']}\n"
            f"Username: {username}\n"
            f"Имя: {first_name}\n"
            f"Фамилия: {last_name}\n"
            f"Регистрация: {user['registered_at']}\n\n"
        )

        if len(chunk) + len(line) > 3500:
            await message.answer(chunk)
            chunk = ""

        chunk += line

    if chunk:
        await message.answer(chunk)


@dp.message(Command("survey"))
@dp.message(F.text == "Начать опрос")
async def start_survey(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(Survey.waiting_for_name)

    await message.answer("Как вас зовут?")


@dp.message(Survey.waiting_for_name)
async def process_name(message: Message, state: FSMContext):
    name = message.text.strip() if message.text else ""

    if not name:
        await message.answer("Введите имя текстом.")
        return

    await state.update_data(name=name)
    await state.set_state(Survey.waiting_for_age)

    await message.answer("Сколько вам лет?")


@dp.message(Survey.waiting_for_age)
async def process_age(message: Message, state: FSMContext):
    age_text = message.text.strip() if message.text else ""

    if not age_text.isdigit() or int(age_text) <= 0:
        await message.answer(
            "Введите корректный возраст числом."
        )
        return

    await state.update_data(age=int(age_text))
    await state.set_state(Survey.waiting_for_occupation)

    await message.answer("Чем вы занимаетесь?")


@dp.message(Survey.waiting_for_occupation)
async def process_occupation(message: Message, state: FSMContext):
    occupation = message.text.strip() if message.text else ""

    if not occupation:
        await message.answer(
            "Введите свой род занятий."
        )
        return

    await state.update_data(occupation=occupation)

    data = await state.get_data()

    result = {
        "user_id": message.from_user.id,
        "username": message.from_user.username,
        "name": data["name"],
        "age": data["age"],
        "occupation": occupation,
        "date": datetime.now().isoformat(timespec="seconds")
    }

    try:
        save_result(result)
    except (OSError, TypeError, ValueError):
        await message.answer(
            "Не удалось сохранить результаты опроса."
        )
        await state.clear()
        return

    await state.clear()

    await message.answer(
        "Опрос завершён!\n\n"
        f"Имя: {result['name']}\n"
        f"Возраст: {result['age']}\n"
        f"Занятие: {result['occupation']}\n\n"
        "Спасибо за участие!"
    )


@dp.message(Command("cancel"))
async def cancel_survey(message: Message, state: FSMContext):
    current_state = await state.get_state()

    if current_state is None:
        await message.answer(
            "Сейчас опрос не проводится."
        )
        return

    await state.clear()

    await message.answer(
        "Опрос отменён. Для повторного запуска отправьте /survey."
    )


@dp.message(Command("clear"))
async def clear_history(message: Message):
    conversation_history.pop(message.from_user.id, None)

    await message.answer(
        "История диалога очищена. Можем начать разговор заново."
    )


@dp.message(F.text & ~F.text.startswith("/"))
async def chat_with_ai(message: Message):
    user_text = message.text.strip()

    if not user_text:
        return

    try:
        answer = await run_agent(user_text)

        if not answer:
            answer = "Не удалось получить ответ от агента."

        for i in range(0, len(answer), 4000):
            await message.answer(answer[i:i + 4000])

    except asyncio.TimeoutError:
        await message.answer(
            "Агент слишком долго выполняет запрос. Попробуйте ещё раз."
        )

    except aiohttp.ClientConnectorError:
        await message.answer(
            "Не удалось подключиться к Ollama. Проверьте, запущена ли Ollama."
        )

    except aiohttp.ClientResponseError as error:
        await message.answer(
            f"Ошибка Ollama API: HTTP {error.status}."
        )

    except (KeyError, ValueError):
        await message.answer(
            "Произошла ошибка при обработке ответа агента."
        )

    except aiohttp.ClientError:
        await message.answer(
            "Ошибка при обращении к Ollama."
        )



WEBHOOK_PATH = "/webhook"


async def health(request):
    return web.json_response({"status": "ok"})


async def on_startup(bot: Bot):
    domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if not domain:
        raise RuntimeError("Не задан RAILWAY_PUBLIC_DOMAIN")

    webhook_url = f"https://{domain}{WEBHOOK_PATH}"

    await bot.set_webhook(
        url=webhook_url,
        secret_token=os.getenv("WEBHOOK_SECRET")
    )

    logging.info("Webhook установлен: %s", webhook_url)


def main():
    init_database()

    app = web.Application()
    app.router.add_get("/health", health)

    dp.startup.register(on_startup)

    webhook_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        secret_token=os.getenv("WEBHOOK_SECRET")
    )

    webhook_handler.register(
        app,
        path=WEBHOOK_PATH
    )

    setup_application(app, dp, bot=bot)

    port = int(os.getenv("PORT", "8080"))

    web.run_app(
        app,
        host="0.0.0.0",
        port=port
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()



if __name__ == "__main__":
    asyncio.run(main())
