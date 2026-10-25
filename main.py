import asyncio
import os
import sys
import chromadb
from chromadb.utils import embedding_functions

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioMcpToolAdapter, StdioServerParams
from autogen_core.tools import FunctionTool
from autogen_core.models import ModelInfo

# Настройка клиента OpenRouter
model_client = OpenAIChatCompletionClient(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_info=ModelInfo(
        vision=False, function_calling=True, json_output=True,
        family="unknown", structured_output=True
    )
)

# База знаний и RAG
chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(name="it_faq", embedding_function=default_ef)


# Ищем решение в базе знаний
async def search_faq(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=1)
    if results['documents'] and results['documents'][0]:
        return "\n".join(results['documents'][0])
    return "Решение не найдено в базе знаний."


rag_tool = FunctionTool(search_faq, description="Поиск решения в FAQ")


async def main():
    print("IT-ПОДДЕРЖКА")

    # Подключение MCP-сервера
    server_params = StdioServerParams(
        command=sys.executable,
        args=["mcp_server.py"],
    )

    print("Подключение к MCP-серверу...")
    try:
        get_server_status_tool = await StdioMcpToolAdapter.from_server_params(
            server_params, "get_server_status"
        )
        mcp_tools = [get_server_status_tool]
        print("MCP-сервер успешно подключен.")
    except Exception as e:
        print(f"Не удалось подключить MCP-сервер: {e}")
        mcp_tools = []

    # Собираем все инструменты для агентов
    all_tools = [rag_tool] + mcp_tools

    # Агент 1 - классификатор (определяет проблему)
    classifier = AssistantAgent(
        name="Classifier",
        model_client=model_client,
        system_message="""Ты классификатор IT-заявок. Проанализируй запрос пользователя.
        Определи категорию (VPN, Почта, Оборудование и т.д.) и приоритет (Низкий, Средний, Высокий, Критический).
        Если проблема критическая (например, "у всего отдела") или данных мало (например, "всё сломалось"), 
        напиши 'НУЖНА ЭСКАЛАЦИЯ' или 'НУЖНО УТОЧНЕНИЕ'.
        В остальных случаях напиши 'СТАНДАРТНАЯ ЗАЯВКА' и передай ход следующему агенту."""
    )

    # Агент 2 - решатель (ищет в базе знаний и использует MCP)
    solver = AssistantAgent(
        name="Solver",
        model_client=model_client,
        tools=all_tools,
        system_message="""Ты решатель проблем. Если предыдущий агент написал 'СТАНДАРТНАЯ ЗАЯВКА', 
        используй инструмент search_faq, чтобы найти решение.
        Если нужна дополнительная информация о статусе сервера или сотрудника, используй MCP-инструменты (get_server_status, get_employee_status).
        Найди подходящую статью и кратко перескажи суть решения. Передай ход следующему агенту."""
    )

    # Агент 3 - коммуникатор (пишет финальный ответ)
    communicator = AssistantAgent(
        name="Communicator",
        model_client=model_client,
        system_message="""Ты коммуникатор. На основе всей предыдущей переписки сформируй финальный ответ пользователю.
        Если была эскалация или уточнение - вежливо сообщи об этом.
        Если было найдено решение - напиши пошаговую инструкцию на русском языке.
        В конце ответа ОБЯЗАТЕЛЬНО добавь слово 'КОНЕЦ', чтобы остановить работу команды."""
    )

    # Настройка линейной команды
    team = RoundRobinGroupChat(
        participants=[classifier, solver, communicator],
        termination_condition=TextMentionTermination("КОНЕЦ"),
        max_turns=5
    )

    # Главный цикл программы
    while True:
        query = input("\nВведите описание вашей проблемы (или 'exit' для выхода):\n> ")

        if query.lower().strip() == 'exit':
            print("Выход из программы.")
            break

        print(f"\n[Запуск агентов]")

        try:
            final_answer = ""

            # Выводим все шаги агентов в терминал
            result = await Console(team.run_stream(task=query))

            # Берем финальный ответ от последнего агента
            final_answer = result.messages[-1].content

            # Проверяем, просит ли система уточнение
            needs_clarification = "уточн" in final_answer.lower() or "уточн" in str(result.messages).lower()

            if needs_clarification:
                print("\nСистеме не хватает данных. Пожалуйста, опишите проблему подробнее.")
                continue  # Возвращаемся в начало цикла
            else:
                # Сохраняем финальный ответ в result.txt
                with open("result.txt", "w", encoding="utf-8") as f:
                    f.write(f"Запрос: {query}\n\n")
                    f.write(f"Ответ системы:\n{final_answer}")

                print(f"\nОтвет сохранен в result.txt")
                break  # Выходим из программы

        except Exception as e:
            print(f"\nОшибка: {e}")
            # Продолжаем цикл, чтобы можно было ввести запрос заново
            continue


if __name__ == "__main__":
    asyncio.run(main())