import os
from typing import Dict, List, Any
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

from .retriever import PDDRetriever


class RAGPipelineAPI:
    """
    Упрощенная версия RAG pipeline с использованием Mistral API.
    Не требует локальной загрузки модели - использует API.
    """

    def __init__(self, pdd_path: str = "data/pdd.json", cache_dir: str = "faiss_langchain_cache"):
        """
        Инициализирует RAG pipeline с Mistral API.

        Args:
            pdd_path (str): Путь к данным ПДД.
            cache_dir (str): Директория для FAISS кеша.
        """
        print("Initializing Law RAG Pipeline (Mistral API)...")

        # 1. Инициализация retriever (поиск по документам)
        self.retriever = PDDRetriever(pdd_path=pdd_path, cache_dir=cache_dir)

        # 2. Получение API ключа
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY не найден в переменных окружения. "
                "Получите ключ на https://console.mistral.ai/ и добавьте в .env файл"
            )

        # 3. Инициализация Mistral LLM через API
        model_name = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        print(f"Connecting to Mistral API (model: {model_name})...")

        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0.2,
            max_tokens=2048,
            api_key=api_key,
        )

        # 4. Создание промпта для генерации ответов
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Ты - помощник по Правилам дорожного движения РФ (ПДД).

Твоя задача:
1. Внимательно изучить предоставленный контекст из ПДД
2. Ответить на вопрос пользователя на основе ТОЛЬКО этого контекста
3. Обязательно указать ссылки на конкретные пункты ПДД
4. Если в контексте нет информации для ответа, честно сказать об этом

Важно:
- Отвечай на русском языке
- Будь точным и лаконичным
- Не выдумывай информацию, которой нет в контексте
- Используй официальную терминологию ПДД"""),
            ("human", """Контекст из ПДД:
{context}

Вопрос пользователя: {query}

Ответь на вопрос, опираясь только на предоставленный контекст. Обязательно укажи ссылки на пункты ПДД.""")
        ])

        print("RAG Pipeline (API) ready.")

    def run(self, query: str, top_k_content: int = 5, include_trace: bool = False) -> Dict[str, Any]:
        """
        Выполняет полный RAG pipeline для запроса.

        Args:
            query (str): Вопрос пользователя.
            top_k_content (int): Количество документов для поиска.
            include_trace (bool): Включить ли trace (не используется в API версии).

        Returns:
            Словарь с ответом, контекстом и источниками.
        """
        # 1. Поиск релевантных документов
        initial_context_docs = self.retriever.search(query, k=top_k_content)

        if not initial_context_docs:
            return {
                "answer": "К сожалению, по вашему запросу ничего не найдено.",
                "context": [],
                "trace": []
            }

        # 2. Формирование контекста для LLM
        context_text = "\n\n".join([
            f"[{doc.metadata.get('source', 'Источник не указан')}]\n{doc.page_content}"
            for doc in initial_context_docs
        ])

        # 3. Формирование промпта и вызов LLM
        messages = self.prompt_template.format_messages(
            context=context_text,
            query=query
        )

        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            print(f"Ошибка при вызове Mistral API: {e}")
            return {
                "answer": f"Извините, произошла ошибка при генерации ответа: {str(e)}",
                "context": [
                    {
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "Источник не указан"),
                        "score": doc.metadata.get("combined_score", 0.0)
                    } for doc in initial_context_docs
                ],
                "trace": []
            }

        # 4. Формирование результата
        output = {
            "answer": answer,
            "context": [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "Источник не указан"),
                    "score": doc.metadata.get("combined_score", 0.0)
                } for doc in initial_context_docs
            ]
        }

        if include_trace:
            output["trace"] = []

        return output
