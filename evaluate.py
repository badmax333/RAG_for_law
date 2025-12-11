import json
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.pipeline import RAGPipeline

class LLMJudge:
    """
    Uses a smaller LLM to act as a judge for evaluating RAG outputs.
    """
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the judge with a specified model.

        Args:
            model_name (str): The name of the model on Hugging Face Hub.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.generator = None
        self._load_model()

    def _load_model(self):
        """Loads the tokenizer and the model for the judge."""
        print(f"Loading judge model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Judge model loaded.")

    def _generate(self, prompt: str, max_new_tokens: int = 250) -> str:
        """Helper function to generate text from a prompt."""
        generated_sequences = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False, # Use greedy decoding for more deterministic output
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_output = generated_sequences[0]['generated_text']
        answer = full_output.replace(prompt, "").strip()
        return answer

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
        
        output = self._generate(prompt, max_new_tokens=300)
        
        try:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                json_str = match.group(0)
                scores = json.loads(json_str)
                return {
                    "relevance": float(scores.get("relevance", 0)),
                    "faithfulness": float(scores.get("faithfulness", 0)),
                    "completeness": float(scores.get("completeness", 0)),
                    "clarity": float(scores.get("clarity", 0)),
                    "justification": scores.get("justification", "No justification provided.")
                }
            else:
                print(f"Could not find JSON in judge output: {output}")
                return self._get_default_scores()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing judge output: {e}\nOutput was: {output}")
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

def evaluate_with_llm_judge(pipeline, validation_data, judge):
    """Evaluates the pipeline using an LLM as a judge."""
    all_scores = {
        "relevance": [],
        "faithfulness": [],
        "completeness": [],
        "clarity": []
    }

    print("\n--- Starting LLM Judge Evaluation ---")
    for i, example in enumerate(validation_data):
        print(f"Evaluating {i+1}/{len(validation_data)}: '{example['query']}'")
        
        result = pipeline.run(example['query'], top_k_content=3)
        
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

    # Calculate average scores
    avg_scores = {key: sum(values) / len(values) for key, values in all_scores.items()}
    avg_scores["num_samples"] = len(validation_data)
    
    return avg_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG for Law pipeline.")
    parser.add_argument("--validation", type=str, default="data/validation_set.jsonl", help="Path to the validation set.")
    parser.add_argument("--use_llm_judge", action="store_true", help="Use LLM as a judge for evaluation instead of ROUGE.")
    args = parser.parse_args()

    print("Initializing RAG Pipeline for evaluation...")
    rag_pipeline = RAGPipeline()
    
    print(f"Loading validation set from {args.validation}...")
    validation_data = load_validation_set(args.validation)
    
    if args.use_llm_judge:
        judge = LLMJudge()
        metrics = evaluate_with_llm_judge(rag_pipeline, validation_data, judge)
        
        print("\n--- LLM Judge Evaluation Results ---")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Average Relevance Score (1-10): {metrics['relevance']:.2f}")
        print(f"Average Faithfulness Score (1-10): {metrics['faithfulness']:.2f}")
        print(f"Average Completeness Score (1-10): {metrics['completeness']:.2f}")
        print(f"Average Clarity Score (1-10): {metrics['clarity']:.2f}")
        print("------------------------------------")
    else:
        metrics = evaluate_with_rouge(rag_pipeline, validation_data)
        
        print("\n--- ROUGE Evaluation Results ---")
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Average ROUGE-1 F-score: {metrics['avg_rouge1_f']:.4f}")
        print(f"Average ROUGE-L F-score: {metrics['avg_rougeL_f']:.4f}")
        print("---------------------------------")

if __name__ == "__main__":
    main()
