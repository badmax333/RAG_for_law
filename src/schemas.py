from pydantic import BaseModel, Field
from typing import List, Literal

# Step 1: Initial Generation
class InitialGenerationOutput(BaseModel):
    reasoning: str = Field(description="Краткое описание того, как ты пришел к ответу на основе контекста.")
    answer: str = Field(description="Твой ответ на вопрос.")

# Step 2: Critique and Decomposition
class CritiqueOutput(BaseModel):
    reasoning: str = Field(description="Анализ ответа, выявление пробелов и обоснование необходимости новых запросов.")
    gaps_found: str = Field(description="Описание найденных пробелов или 'Пробелов не найдено'.")
    new_queries: List[str] = Field(description="Список из 1-3 новых, более конкретных поисковых запросов на русском языке.")

# Step 3: Final Synthesis
class SynthesisOutput(BaseModel):
    reasoning: str = Field(description="Описание того, как был синтезирован финальный ответ из всей доступной информации.")
    synthesized_answer: str = Field(description="Финальный, синтезированный ответ.")

# Step 4: Factual Verification and Correction
class CorrectionOutput(BaseModel):
    final_answer: str = Field(description="Финальный, проверенный и исправленный ответ на вопрос пользователя, основанный на контексте.")

# Main SGR Trace Structure
class TraceStep(BaseModel):
    step: Literal["Initial Generation", "Critique and Decomposition", "Final Synthesis", "Factual Verification and Correction"]
    result: dict # Can hold any of the above output models

class SGROutput(BaseModel):
    answer: str
    trace: List[TraceStep]
