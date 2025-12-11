import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from typing import Dict, List, Any

from .retriever import PDDRetriever
from .chains import create_sgr_chain

class RAGPipeline:
    """
    A complete Corrective RAG pipeline built with LangChain.
    It uses a retriever for advanced retrieval and a complex SGR chain for generation.
    """
    def __init__(self, pdd_path: str = "data/pdd.json", cache_dir: str = "faiss_langchain_cache"):
        """
        Initializes the RAG pipeline.

        Args:
            pdd_path (str): Path to the structured pdd.json file.
            cache_dir (str): Directory for FAISS cache.
        """
        print("Initializing Law RAG Pipeline...")
        
        # 1. Initialize the retriever
        self.retriever = PDDRetriever(pdd_path=pdd_path, cache_dir=cache_dir)
        
        # 2. Initialize the LLM
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"Loading LLM: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("Model loaded with 4-bit quantization (BitsAndBytesConfig).")
        
        except (ImportError, ValueError) as e:
            print(f"4-bit quantization failed ({type(e).__name__}: {e}). Loading model in float16.")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            except Exception as fallback_error:
                print(f"Fallback to float16 also failed: {fallback_error}. Trying without dtype...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                )

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("LLM loaded.")

        # 3. Create the main SGR chain
        self.sgr_chain = create_sgr_chain(self.llm)
        print("Pipeline ready.")

    def run(self, query: str, top_k_content: int = 5, include_trace: bool = False) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline for a given query.

        Args:
            query (str): The user's question.
            top_k_content (int): Number of documents to retrieve for the initial context.
            include_trace (bool): If True, includes the full reasoning trace in the output.

        Returns:
            A dictionary containing the answer, the retrieved context, and optionally the trace.
        """
        # 1. Retrieve initial documents
        initial_context_docs = self.retriever.search(query, k=top_k_content)
        
        if not initial_context_docs:
            return {
                "answer": "К сожалению, по вашему запросу ничего не найдено.",
                "context": [],
                "trace": []
            }

        # 2. Prepare inputs for the chain
        chain_inputs = {
            "query": query,
            "context": initial_context_docs,
            "retriever": self.retriever # Pass the retriever instance for iterative search
        }
        
        # 3. Invoke the SGR chain
        generation_result = self.sgr_chain.invoke(chain_inputs)
        
        # 4. Format the output
        output = {
            "answer": generation_result["answer"],
            "context": [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "Источник не указан"),
                    "score": doc.metadata.get("combined_score", 0.0) # Use the combined score from the retriever
                } for doc in initial_context_docs
            ]
        }

        if include_trace:
            output["trace"] = generation_result.get("trace", [])
        
        return output
