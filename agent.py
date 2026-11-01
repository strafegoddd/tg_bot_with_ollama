
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

MAX_AGENT_STEPS = 5


def get_weather(city: str) -> str:
    weather_data = {
        "москва": "Температура: +12°C. Облачно.",
        "санкт-петербург": "Температура: +9°C. Дождь.",
        "рига": "Температура: +10°C. Переменная облачность.",
        "минск": "Температура: +11°C. Ясно.",
        "лондон": "Температура: +8°C. Дождь.",
    }

    normalized_city = city.strip().lower().replace("ё", "е")

    if normalized_city in weather_data:
        return weather_data[normalized_city]

    aliases = {
        "riga": "рига",
        "латвия": "рига",
        "санкт петербург": "санкт-петербург",
        "петербург": "санкт-петербург",
    }

    normalized_city = aliases.get(normalized_city, normalized_city)

    if normalized_city in weather_data:
        return weather_data[normalized_city]

    return f"Демонстрационные данные о погоде для города {city} отсутствуют."


def get_time(timezone: str) -> str:
    try:
        current_time = datetime.now(ZoneInfo(timezone))
        return current_time.strftime("%d.%m.%Y, %H:%M:%S")
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f"Неизвестный часовой пояс: {timezone}. "
            "Используйте формат, например Europe/Riga или Europe/Moscow."
        )


def calculate(expression: str) -> str:
    if len(expression) > 200:
        return "Выражение слишком длинное."

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Разрешены только числа.")
            return node.value

        if isinstance(node, ast.BinOp):
            operation = allowed_operators.get(type(node.op))
            if operation is None:
                raise ValueError("Недопустимая операция.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Слишком большая степень.")

            result = operation(left, right)

            if abs(result) > 10**100:
                raise ValueError("Слишком большое значение.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = allowed_operators.get(type(node.op))
            if operation is None:
                raise ValueError("Недопустимая операция.")

            return operation(evaluate(node.operand))

        raise ValueError("Недопустимое выражение.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "Ошибка: деление на ноль."
    except (ValueError, SyntaxError, OverflowError) as error:
        return f"Не удалось вычислить выражение: {error}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить демонстрационные данные о погоде в городе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Получить текущее время в часовом поясе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Часовой пояс, например Europe/Riga."
                    }
                },
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Вычислить математическое выражение.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Математическое выражение, например (15 + 5) * 2."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "get_time": get_time,
    "calculate": calculate,
}


async def run_agent(user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Ты Марат-бот, интеллектуальный помощник. "
                "У тебя есть инструменты для получения погоды, "
                "текущего времени и математических вычислений. "
                "Если запрос требует инструмента, обязательно вызови его. "
                "Не придумывай результаты инструментов. "
                "Для сложного запроса можешь вызвать несколько инструментов "
                "последовательно или одновременно. "
                "После получения результатов дай краткий ответ на русском языке."
            )
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(MAX_AGENT_STEPS):
            async with session.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "tools": TOOLS,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()

            assistant_message = data["message"]
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                return assistant_message.get(
                    "content",
                    "Не удалось сформировать ответ."
                ).strip()

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                function = AVAILABLE_FUNCTIONS.get(function_name)

                if function is None:
                    result = "Ошибка: неизвестный инструмент."
                else:
                    try:
                        result = function(**arguments)
                    except (TypeError, ValueError) as error:
                        result = f"Ошибка выполнения инструмента: {error}"

                messages.append({
                    "role": "tool",
                    "tool_name": function_name,
                    "content": str(result)
                })

    return "Не удалось завершить выполнение запроса: превышен лимит шагов агента."
