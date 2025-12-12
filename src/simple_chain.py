from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser


def create_simple_qa_chain(llm: HuggingFacePipeline):
    """
    Создает упрощенную цепочку для прямых вопросов.
    Без итеративного поиска и self-correction - быстрее в 3-4 раза.
    """

    prompt = PromptTemplate(
        template="""Ты - юридический ассистент по ПДД РФ.
Ответь кратко и точно на вопрос, используя только предоставленный контекст.
Обязательно укажи номер пункта ПДД.

Контекст:
{context}

Вопрос: {query}

Ответ:""",
        input_variables=["context", "query"]
    )

    def format_context(inputs: dict) -> dict:
        docs = inputs["context"]
        context_str = "\n".join([f"- [{doc.metadata.get('p_num', 'N/A')}] {doc.page_content}" for doc in docs])
        return {
            "context": context_str,
            "query": inputs["query"]
        }

    def parse_output(output: str) -> dict:
        # Убираем артефакты генерации
        answer = output.strip()
        if answer.startswith("Ответ:"):
            answer = answer[6:].strip()
        return {"answer": answer, "trace": []}

    chain = (
        RunnableLambda(format_context)
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_output)
    )

    return chain
