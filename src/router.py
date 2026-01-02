from typing import Literal, Dict
import re


QueryType = Literal["simple", "complex", "comparative"]


class QueryRouter:
    """
    Классифицирует запросы пользователя для выбора оптимальной стратегии обработки.

    Типы запросов:
    - simple: прямой вопрос о конкретном пункте ПДД
    - complex: требует анализа нескольких пунктов или итеративного поиска
    - comparative: требует сравнения или объяснения различий
    """

    def __init__(self):
        self.simple_patterns = [
            r'\bможно\s+ли\b',
            r'\bразрешено\s+ли\b',
            r'\bкакой\s+(штраф|пункт)\b',
            r'\bчто\s+говорит\b',
            r'\bкаков\s+(размер|срок)\b',
        ]

        self.complex_patterns = [
            r'\bпочему\b',
            r'\bкак\s+правильно\b',
            r'\bчто\s+делать\s+если\b',
            r'\bв\s+каких\s+случаях\b',
            r'\bкакие\s+есть\b',
            r'\bперечисли\b',
        ]

        self.comparative_patterns = [
            r'\bразниц[аы]\s+между\b',
            r'\bчем\s+отличается\b',
            r'\bили\b.*\bили\b',
            r'\bвместо\b',
            r'\bа\s+не\b',
        ]

    def route(self, query: str) -> QueryType:
        query_lower = query.lower()

        # Проверяем comparative (приоритет выше)
        for pattern in self.comparative_patterns:
            if re.search(pattern, query_lower):
                return "comparative"

        # Проверяем complex
        for pattern in self.complex_patterns:
            if re.search(pattern, query_lower):
                return "complex"

        # Проверяем simple
        for pattern in self.simple_patterns:
            if re.search(pattern, query_lower):
                return "simple"

        # Эвристики на основе длины и структуры
        words = query_lower.split()
        if len(words) <= 6 and '?' in query:
            return "simple"
        elif len(words) > 15 or len(query) > 100:
            return "complex"

        # По умолчанию считаем complex (безопаснее)
        return "complex"

    def get_retrieval_params(self, query_type: QueryType) -> Dict:
        """
        Возвращает параметры поиска в зависимости от типа запроса.
        """
        if query_type == "simple":
            return {
                "top_k": 3,
                "use_sgr": False,  # Не нужен полный SGR для простых запросов
                "max_iterations": 0
            }
        elif query_type == "comparative":
            return {
                "top_k": 8,
                "use_sgr": True,
                "max_iterations": 2
            }
        else:  # complex
            return {
                "top_k": 5,
                "use_sgr": True,
                "max_iterations": 1
            }