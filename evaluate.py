import json
import argparse
import re
import os
import sys
import time

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

from src.pipeline import RAGPipeline

class LLMJudge:
    """
    Uses Mistral AI API to act as a judge for evaluating RAG outputs.
    """
    def __init__(self, model_name: str = "mistral-large-latest"):
        """
        Initializes the judge with Mistral AI API.

        Args:
            model_name (str): The name of the Mistral model to use via API.
        """
        self.model_name = model_name
        
        if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
            raise ValueError(
                "MISTRAL_API_KEY not set. Please set it in .env file or config.py. "
                "Get your API key at https://console.mistral.ai/"
            )
        
        print(f"Initializing Mistral AI API judge with model: {model_name}...")
        # Инициализация LLM Judge через Mistral API
        self.llm = ChatMistralAI(
            model=model_name,
            mistral_api_key=MISTRAL_API_KEY,
            temperature=0.0,  # Use deterministic output for evaluation
            max_tokens=500,
        )
        print("Judge model initialized via API.")

    def _generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Helper function to generate text from a prompt using API."""
        try:
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            answer = response.content.strip()
            return answer
        except Exception as e:
            print(f"Error generating response from API: {e}")
            return ""

    def evaluate(self, query: str, context: list, generated_answer: str, reference_answer: str) -> dict:
        """
        Evaluates a generated answer based on multiple criteria.

        Args:
            query (str): The original user query.
            context (list): The list of retrieved context snippets.
            generated_answer (str): The answer from the RAG pipeline.
            reference_answer (str): The ideal answer from the validation set.

        Returns:
            A dictionary with scores for each criterion.
        """
        context_str = "\n".join([f"- {c}" for c in context])
        
        prompt = f"""
        Ты - экспертный оценщик качества ответов, сгенерированных RAG-системой.
        Твоя задача - оценить ответ на вопрос по правилам дорожного движения (ПДД) по нескольким критериям.
        Оцени ответ от 1 до 10, где 1 - очень плохо, 10 - отлично.

        Вопрос пользователя:
        {query}

        Контекст, предоставленный RAG-системе:
        {context_str}

        Сгенерированный ответ:
        {generated_answer}

        Эталонный ответ (для сравнения):
        {reference_answer}

        Оцени сгенерированный ответ по следующим критериям:
        1. Релевантность: Насколько хорошо ответ отвечает на вопрос пользователя?
        2. Добросовестность (Faithfulness): Насколько точно ответ основан на предоставленном контексте? Есть ли выдуманные факты?
        3. Полнота: Охватывает ли ответ все ключевые аспекты из эталонного ответа?
        4. Ясность: Насколько ответ понятен, логичен и хорошо структурирован?

        Предоставь оценку в формате JSON, например:
        {{
        "relevance": 8,
        "faithfulness": 9,
        "completeness": 7,
        "clarity": 8,
        "justification": "Краткое обоснование оценок."
        }}
        JSON:
        """
        
        output = self._generate(prompt, max_new_tokens=500)
        
        try:
            # Сначала пытаемся извлечь JSON из markdown блока
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Если markdown блока нет, ищем JSON напрямую
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    print(f"Could not find JSON in judge output: {output[:200]}...")
                    return self._get_default_scores()
            
            json_str = json_str.strip()
            
            # Пытаемся распарсить JSON напрямую
            try:
                scores = json.loads(json_str)
            except json.JSONDecodeError:
                # Если не получилось (из-за управляющих символов), используем regex для извлечения значений
                # Это более надежный подход для невалидного JSON
                relevance_match = re.search(r'"relevance"\s*:\s*(\d+)', json_str)
                faithfulness_match = re.search(r'"faithfulness"\s*:\s*(\d+)', json_str)
                completeness_match = re.search(r'"completeness"\s*:\s*(\d+)', json_str)
                clarity_match = re.search(r'"clarity"\s*:\s*(\d+)', json_str)
                
                # Для justification ищем значение между кавычками, включая многострочные
                # Ищем от "justification": " до закрывающей кавычки перед запятой или }
                justification_value = None
                justification_start = json_str.find('"justification"')
                if justification_start != -1:
                    # Находим открывающую кавычку значения
                    quote_start = json_str.find('"', justification_start + len('"justification"'))
                    if quote_start != -1:
                        # Ищем закрывающую кавычку, которая идет перед запятой или }
                        # Пропускаем экранированные кавычки
                        i = quote_start + 1
                        while i < len(json_str):
                            if json_str[i] == '"' and json_str[i-1] != '\\':
                                # Проверяем, что после кавычки идет запятая, } или пробел+}
                                j = i + 1
                                while j < len(json_str) and json_str[j] in [' ', '\n', '\t', '\r']:
                                    j += 1
                                if j >= len(json_str) or json_str[j] in [',', '}']:
                                    justification_value = json_str[quote_start + 1:i]
                                    break
                            i += 1
                
                scores = {
                    "relevance": int(relevance_match.group(1)) if relevance_match else 0,
                    "faithfulness": int(faithfulness_match.group(1)) if faithfulness_match else 0,
                    "completeness": int(completeness_match.group(1)) if completeness_match else 0,
                    "clarity": int(clarity_match.group(1)) if clarity_match else 0,
                    "justification": justification_value if justification_value else "No justification provided."
                }
            
            return {
                "relevance": float(scores.get("relevance", 0)),
                "faithfulness": float(scores.get("faithfulness", 0)),
                "completeness": float(scores.get("completeness", 0)),
                "clarity": float(scores.get("clarity", 0)),
                "justification": scores.get("justification", "No justification provided.")
            }
        except Exception as e:
            print(f"Error parsing judge output: {e}\nOutput was: {output[:500]}...")
            return self._get_default_scores()

    def _get_default_scores(self):
        """Returns default scores in case of parsing errors."""
        return {
            "relevance": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "justification": "Failed to parse judge output."
        }


def load_validation_set(path: str):
    """Loads the validation set from a .jsonl file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_with_rouge(pipeline, validation_data):
    """Evaluates the pipeline using ROUGE scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_f_scores = []
    rougeL_f_scores = []
    
    print("\n--- Starting ROUGE Evaluation ---")
    for i, example in enumerate(validation_data):
        print(f"Evaluating {i+1}/{len(validation_data)}: '{example['query']}'")
        
        result = pipeline.run(example['query'], top_k_content=3)
        generated_answer = result['answer']
        reference_answer = example['expected_answer']
        
        scores = scorer.score(reference_answer, generated_answer)
        rouge1_f_scores.append(scores['rouge1'].fmeasure)
        rougeL_f_scores.append(scores['rougeL'].fmeasure)
        
        print(f"  Generated: {generated_answer}")
        print(f"  Reference: {reference_answer}")
        print(f"  ROUGE-1 F: {scores['rouge1'].fmeasure:.4f}")
        print("-" * 20)

    avg_rouge1_f = sum(rouge1_f_scores) / len(rouge1_f_scores)
    avg_rougeL_f = sum(rougeL_f_scores) / len(rougeL_f_scores)

    return {
        "avg_rouge1_f": avg_rouge1_f,
        "avg_rougeL_f": avg_rougeL_f,
        "num_samples": len(validation_data)
    }

def evaluate_retriever_precision_recall(pipeline, validation_data, top_k: int = 5):
    """
    Evaluates the retriever using precision and recall metrics.
    
    Args:
        pipeline: The RAG pipeline with a retriever.
        validation_data: List of validation examples, each with 'query' and 'source_p_num'.
        top_k: Number of documents to retrieve for evaluation.
    
    Returns:
        Dictionary with precision@k, recall@k, and other metrics.
    """
    print(f"\n--- Starting Retriever Evaluation (Precision/Recall @{top_k}) ---")
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    precision_scores = []
    recall_scores = []
    
    # Access retriever directly to get full metadata
    retriever = pipeline.retriever
    
    for i, example in enumerate(validation_data):
        query = example['query']
        expected_p_num = example.get('source_p_num', 'N/A')
        
        if expected_p_num == 'N/A':
            print(f"Skipping {i+1}/{len(validation_data)}: '{query}' (no source_p_num)")
            continue
        
        print(f"Evaluating {i+1}/{len(validation_data)}: '{query}'")
        print(f"  Expected source: {expected_p_num}")
        
        # Get retrieved documents directly from retriever to access full metadata
        retrieved_docs = retriever.search(query, k=top_k)
        
        # Check if any retrieved document matches the expected source
        found_relevant = False
        retrieved_p_nums = []
        
        for doc in retrieved_docs:
            # Extract p_num from metadata
            doc_p_num = doc.metadata.get('p_num', '')
            retrieved_p_nums.append(doc_p_num)
            
            # Check if this document matches the expected p_num
            if doc_p_num == expected_p_num:
                found_relevant = True
        
        if found_relevant:
            true_positives += 1
            # Precision@k: fraction of retrieved docs that are relevant
            # Since we only have one relevant doc, precision = 1/k if found
            precision_scores.append(1.0 / len(retrieved_docs) if retrieved_docs else 0.0)
            recall_scores.append(1.0)  # Found the relevant document
        else:
            false_negatives += 1
            precision_scores.append(0.0)  # No relevant docs found
            recall_scores.append(0.0)  # Did not find the relevant document
            false_positives += len(retrieved_docs)  # All retrieved docs are false positives
        
        print(f"  Retrieved {len(retrieved_docs)} documents")
        print(f"  Found relevant: {found_relevant}")
        print(f"  Retrieved p_nums: {retrieved_p_nums}")
        print("-" * 20)
    
    # Calculate metrics
    num_valid_samples = len([ex for ex in validation_data if ex.get('source_p_num', 'N/A') != 'N/A'])
    
    if num_valid_samples == 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_samples": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    # Also calculate traditional precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "precision": precision,
        "recall": recall,
        "num_samples": num_valid_samples,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def evaluate_with_llm_judge(pipeline, validation_data, judge, delay_between_requests: float = 0.3):
    """
    Evaluates the pipeline using an LLM as a judge.
    
    Args:
        pipeline: The RAG pipeline.
        validation_data: List of validation examples.
        judge: LLMJudge instance.
        delay_between_requests: Delay in seconds between API requests (default: 0.3s).
                                Recommended: 0.1-0.5 seconds to avoid rate limits.
    """
    all_scores = {
        "relevance": [],
        "faithfulness": [],
        "completeness": [],
        "clarity": []
    }

    print(f"\n--- Starting LLM Judge Evaluation (delay: {delay_between_requests}s between requests) ---")
    for i, example in enumerate(validation_data):
        print(f"Evaluating {i+1}/{len(validation_data)}: '{example['query']}'")
        
        # Добавляем задержку перед запросом к pipeline (кроме первого запроса)
        if i > 0:
            time.sleep(delay_between_requests)
        
        result = pipeline.run(example['query'], top_k_content=3)
        
        # Добавляем задержку после pipeline.run() и перед judge.evaluate()
        time.sleep(delay_between_requests)
        
        scores = judge.evaluate(
            query=example['query'],
            context=result['context'],
            generated_answer=result['answer'],
            reference_answer=example['expected_answer']
        )
        
        print(f"  Generated: {result['answer']}")
        print(f"  Scores: {scores}")
        print("-" * 20)

        for key in all_scores:
            all_scores[key].append(scores[key])
        
        # Добавляем задержку после запроса к API judge (кроме последнего запроса)
        if i < len(validation_data) - 1:
            time.sleep(delay_between_requests)

    # Calculate average scores
    avg_scores = {key: sum(values) / len(values) for key, values in all_scores.items()}
    avg_scores["num_samples"] = len(validation_data)
    
    return avg_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG for Law pipeline.")
    parser.add_argument("--validation", type=str, default="data/validation_set.jsonl", help="Path to the validation set.")
    parser.add_argument("--use_rouge", action="store_true", help="Use ROUGE for evaluation instead of LLM judge.")
    parser.add_argument("--evaluate_retriever", action="store_true", help="Evaluate retriever precision and recall.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve for retriever evaluation.")
    parser.add_argument("--api_delay", type=float, default=0.3, help="Delay in seconds between API requests for LLM judge (default: 0.3, recommended: 0.1-0.5).")
    args = parser.parse_args()

    print("Initializing RAG Pipeline for evaluation...")
    rag_pipeline = RAGPipeline()
    
    print(f"Loading validation set from {args.validation}...")
    validation_data = load_validation_set(args.validation)

    # Evaluate retriever if requested
    if args.evaluate_retriever:
        retriever_metrics = evaluate_retriever_precision_recall(rag_pipeline, validation_data, top_k=args.top_k)
        
        print("\n--- Retriever Evaluation Results ---")
        print(f"Number of samples: {retriever_metrics['num_samples']}")
        print(f"Precision@{args.top_k}: {retriever_metrics['precision_at_k']:.4f}")
        print(f"Recall@{args.top_k}: {retriever_metrics['recall_at_k']:.4f}")
        print(f"Precision: {retriever_metrics['precision']:.4f}")
        print(f"Recall: {retriever_metrics['recall']:.4f}")
        print(f"True Positives: {retriever_metrics['true_positives']}")
        print(f"False Positives: {retriever_metrics['false_positives']}")
        print(f"False Negatives: {retriever_metrics['false_negatives']}")
        print("------------------------------------")

    if args.use_rouge:
        metrics = evaluate_with_rouge(rag_pipeline, validation_data)
        
        print("\n--- ROUGE Evaluation Results ---")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Average ROUGE-1 F-score: {metrics['avg_rouge1_f']:.4f}")
        print(f"Average ROUGE-L F-score: {metrics['avg_rougeL_f']:.4f}")
        print("---------------------------------")
    else:
        judge = LLMJudge()
        metrics = evaluate_with_llm_judge(rag_pipeline, validation_data, judge, delay_between_requests=args.api_delay)
        
        print("\n--- LLM Judge Evaluation Results ---")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Average Relevance Score (1-10): {metrics['relevance']:.2f}")
        print(f"Average Faithfulness Score (1-10): {metrics['faithfulness']:.2f}")
        print(f"Average Completeness Score (1-10): {metrics['completeness']:.2f}")
        print(f"Average Clarity Score (1-10): {metrics['clarity']:.2f}")
        print("------------------------------------")

if __name__ == "__main__":
    main()
