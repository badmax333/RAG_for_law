import json
import re
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError, BaseModel

from .schemas import (
    InitialGenerationOutput,
    CritiqueOutput,
    SynthesisOutput,
    CorrectionOutput,
    TraceStep
)
from .retriever import PDDRetriever

class RobustJsonOutputParser(PydanticOutputParser):
    """
    A robust JSON output parser that can extract JSON from various text formats.
    """
    def parse(self, text: str) -> BaseModel:
        """
        Parses the output from an LLM, handling markdown code blocks and other noise.
        """
        # print(f"--- DEBUG: Parser received text (len={len(text)}) ---\n{text[:500]}\n--- END DEBUG ---")
        
        json_str = None
        try:
            # 1. Try to find a JSON block within a markdown code block (handles 3 or 4 backticks)
            match = re.search(r'```{3,4}json\s*(.*?)\s*```{3,4}', text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # 2. Fallback: Try to find any JSON-like structure in the text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    json_str = match.group(0)
            
            if json_str:
                # print(f"--- DEBUG: Extracted JSON ---\n{json_str}\n--- END DEBUG ---")
                return super().parse(json_str)
            else:
                # 3. Fallback: Try parsing the whole text if no JSON structure is found
                # print("--- DEBUG: No JSON found, attempting to parse full text. ---")
                return super().parse(text)
        except (ValidationError, json.JSONDecodeError) as e:
            # Re-raise as OutputParserException for consistency with LangChain
            from langchain_core.exceptions import OutputParserException
            raise OutputParserException(
                f"Failed to parse output as JSON. Error: {e}\n"
                f"Output was (len={len(text)}): {text}\n"
                f"Extracted JSON was (len={len(json_str) if json_str else 0}): {json_str}"
            )


def create_sgr_chain(llm: HuggingFacePipeline):
    """
    Creates the Schema-Guided Reasoning (SGR) chain using LangChain Expression Language.
    This chain performs initial generation, critique, iterative retrieval, synthesis, and correction.
    """
    
    # Output Parsers
    initial_parser = RobustJsonOutputParser(pydantic_object=InitialGenerationOutput)
    critique_parser = RobustJsonOutputParser(pydantic_object=CritiqueOutput)
    synthesis_parser = RobustJsonOutputParser(pydantic_object=SynthesisOutput)
    correction_parser = RobustJsonOutputParser(pydantic_object=CorrectionOutput)
    
    # Prompts for each step
    initial_prompt = PromptTemplate(
        template="""Ты - юридический ассистент, специализирующийся на правилах дорожного движения (ПДД) РФ.
        Ответь подробно на вопрос пользователя, основываясь ТОЛЬКО на предоставленном контексте.
        Сформулируй развернутый и исчерпывающий ответ.
        В своем ответе обязательно укажи номер раздела и пункта ПДД, на котором он основан, в формате 'Согласно пункту X.Y раздела Z...'.

        Контекст из ПДД:
        {context}

        Вопрос пользователя:
        {query}

        ВАЖНО: Твой ответ должен быть ТОЛЬКО в формате JSON, без каких-либо дополнительных комментариев, объяснений или текста до или после него. Выведи только JSON объект.

        {format_instructions}""",
        input_variables=["context", "query"],
        partial_variables={"format_instructions": initial_parser.get_format_instructions()}
    )

    critique_prompt = PromptTemplate(
        template="""Ты - критик, анализирующий ответ юридического ассистента.
        Проанализируй исходный вопрос, контекст и ответ. Найди пробелы, неточности или упущения в ответе.
        На основе анализа сформулируй 1-3 новых, более конкретных поисковых запроса на русском языке,
        чтобы заполнить эти пробелы. Если ответ полный и точный, верни пустой список для "new_queries".

        Исходный вопрос:
        {query}

        Контекст:
        {context}

        Предварительный ответ:
        {initial_answer}

        ВАЖНО: Твой ответ должен быть ТОЛЬКО в формате JSON, без каких-либо дополнительных комментариев, объяснений или текста до или после него. Выведи только JSON объект.

        {format_instructions}""",
        input_variables=["query", "context", "initial_answer"],
        partial_variables={"format_instructions": critique_parser.get_format_instructions()}
    )

    synthesis_prompt = PromptTemplate(
        template="""Ты - опытный юридический ассистент. Твоя задача - дать исчерпывающий и точный ответ на вопрос пользователя.
        У тебя есть исходный контекст, дополнительный контекст, полученный в результате уточняющего поиска, и предварительный ответ.
        Используй ВСЮ предоставленную информацию, чтобы исправить неточности и дополнить предварительный ответ.
        Сформулируй финальный ответ, который будет полным, точным и хорошо структурированным. Если информация в контексте противоречива, укажи на это.
        Не добавляй информацию из общих знаний, которой нет в предоставленном контекста.
        В своем ответе обязательно укажи номер раздела и пункта ПДД, на котором он основан, в формате 'Согласно пункту X.Y раздела Z...'.

        Исходный вопрос:
        {query}

        Весь доступный контекст:
        {final_context}

        Предварительный ответ:
        {initial_answer}

        ВАЖНО: Твой ответ должен быть ТОЛЬКО в формате JSON, без каких-либо дополнительных комментариев, объяснений или текста до или после него. Выведи только JSON объект.

        {format_instructions}""",
        input_variables=["query", "final_context", "initial_answer"],
        partial_variables={"format_instructions": synthesis_parser.get_format_instructions()}
    )

    correction_prompt = PromptTemplate(
        template="""Ты - юридический ассистент. Твоя задача - финализировать ответ на вопрос пользователя, используя ТОЛЬКО предоставленный контекст из ПДД.
        Проанализируй предварительный ответ и контекст. Сформулируй финальный, точный и полный ответ.
        Если предварительный ответ неточен или неполон, исправь его на основе контекста.
        Если он точен и полон, просто повтори его.
        ВАЖНО: В поле 'final_answer' должен быть только финальный ответ на вопрос пользователя, без каких-либо комментариев или объяснений.

        Контекст из ПДД:
        {final_context}

        Предварительный ответ:
        {answer_to_check}

        {format_instructions}""",
        input_variables=["final_context", "answer_to_check"],
        partial_variables={"format_instructions": correction_parser.get_format_instructions()}
    )

    # Chains for each step
    initial_chain = initial_prompt | llm | initial_parser
    critique_chain = critique_prompt | llm | critique_parser
    synthesis_chain = synthesis_prompt | llm | synthesis_parser
    correction_chain = correction_prompt | llm | correction_parser

    # The main SGR chain logic
    def sgr_logic(inputs: dict) -> dict:
        """
        The main logic for the SGR chain, orchestrating the steps in a Cascade pattern.
        """
        query = inputs["query"]
        initial_context_docs = inputs["context"]
        retriever: PDDRetriever = inputs["retriever"]
        
        reasoning_trace: List[TraceStep] = []
        
        # Helper to format docs
        def format_docs(docs):
            return "\n".join([f"- {d.page_content}" for d in docs])

        # Step 1: Initial Generation
        initial_context_str = format_docs(initial_context_docs)
        try:
            initial_result: InitialGenerationOutput = initial_chain.invoke({"context": initial_context_str, "query": query})
            reasoning_trace.append(TraceStep(step="Initial Generation", result=initial_result.dict()))
        except (ValidationError, Exception) as e:
            print(f"Error in Initial Generation: {e}")
            return {"answer": "Не удалось сформировать ответ на начальном этапе.", "trace": [ts.dict() for ts in reasoning_trace]}
        
        initial_answer = initial_result.answer

        # Step 2: Critique and Decomposition
        try:
            critique_result: CritiqueOutput = critique_chain.invoke({
                "query": query, 
                "context": initial_context_str, 
                "initial_answer": initial_answer
            })
            reasoning_trace.append(TraceStep(step="Critique and Decomposition", result=critique_result.dict()))
        except (ValidationError, Exception) as e:
            print(f"Error in Critique: {e}")
            # Fallback to initial answer if critique fails
            return {"answer": initial_answer, "trace": [ts.dict() for ts in reasoning_trace]}

        new_queries = critique_result.new_queries
        
        # Routing Logic: Decide whether to perform iterative retrieval and synthesis
        if not new_queries:
            print("\nCritique found no gaps. Skipping iterative retrieval and synthesis.")
            final_answer_for_correction = initial_answer
            final_context_for_correction = initial_context_str
        else:
            # Step 3: Iterative Retrieval
            all_context_docs = initial_context_docs
            for new_query in new_queries:
                additional_docs = retriever.search(new_query, k=2)
                all_context_docs.extend(additional_docs)
            
            # Deduplicate documents based on page_content
            unique_docs = {doc.page_content: doc for doc in all_context_docs}.values()
            final_context_str = format_docs(unique_docs)

            # Step 4: Final Synthesis
            try:
                synthesis_result: SynthesisOutput = synthesis_chain.invoke({
                    "query": query,
                    "final_context": final_context_str,
                    "initial_answer": initial_answer
                })
                reasoning_trace.append(TraceStep(step="Final Synthesis", result=synthesis_result.dict()))
                final_answer_for_correction = synthesis_result.synthesized_answer
            except (ValidationError, Exception) as e:
                print(f"Error in Synthesis: {e}")
                # Fallback to initial answer if synthesis fails
                final_answer_for_correction = initial_answer

            final_context_for_correction = final_context_str

        # Step 5: Factual Verification and Correction
        try:
            correction_result: CorrectionOutput = correction_chain.invoke({
                "final_context": final_context_for_correction,
                "answer_to_check": final_answer_for_correction
            })
            reasoning_trace.append(TraceStep(step="Factual Verification and Correction", result=correction_result.dict()))
            final_answer = correction_result.final_answer
        except (ValidationError, Exception) as e:
            print(f"Error in Correction: {e}")
            # Fallback to the answer before correction if correction fails
            final_answer = final_answer_for_correction
        
        return {
            "answer": final_answer if final_answer else "Не удалось сформировать ответ на основе предоставленного контекста.",
            "trace": [ts.dict() for ts in reasoning_trace]
        }

    return RunnableLambda(sgr_logic)
