import os
import sys
from typing import Dict, List, Any
import time

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI

from .retriever import PDDRetriever
from .chains import create_sgr_chain
from .simple_chain import create_simple_qa_chain
from .router import QueryRouter
from .cache import QueryCache

class RAGPipeline:
    """
    A complete Corrective RAG pipeline built with LangChain.
    It uses a retriever for advanced retrieval and a complex SGR chain for generation.
    """
    def __init__(self, pdd_path: str = "data/pdd.json", cache_dir: str = "faiss_langchain_cache",
                 use_cache: bool = True, cache_ttl: int = 3600):
        """
        Initializes the RAG pipeline.

        Args:
            pdd_path (str): Path to the structured pdd.json file.
            cache_dir (str): Directory for FAISS cache.
            use_cache (bool): Enable query result caching.
            cache_ttl (int): Cache TTL in seconds (default: 1 hour).
        """
        print("Initializing Law RAG Pipeline...")

        # 0. Initialize cache
        self.use_cache = use_cache
        self.cache = QueryCache(max_size=100, ttl_seconds=cache_ttl) if use_cache else None

        # 1. Initialize the retriever
        self.retriever = PDDRetriever(pdd_path=pdd_path, cache_dir=cache_dir)
        
        # 2. Initialize the LLM
        # Check if CUDA is available to choose appropriate model
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            # Use larger model with GPU quantization
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            print(f"Loading LLM with GPU: {model_name}...")
        else:
            # Use smaller model optimized for CPU
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"No GPU detected. Loading CPU-optimized LLM: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Try GPU quantization first (only if CUDA available)
        if has_cuda:
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
                    low_cpu_mem_usage=True,
                )
                print("Model loaded with 4-bit quantization (GPU).")
            except Exception as e:
                print(f"GPU quantization failed: {e}")
                print("Falling back to CPU mode...")
                has_cuda = False

        # Load model for CPU
        if not has_cuda:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True,
                )
                print(f"Model loaded on CPU (using {model_name}).")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model: {e}\n\n"
                    "Возможные решения:\n"
                    "1. Освободите больше RAM (закройте другие программы)\n"
                    "2. Используйте API вместо локальной модели (OpenAI, Claude и т.д.)\n"
                    "3. Установите GPU для использования более мощных моделей"
                )

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.2,
            max_tokens=2048,
        )
        print("Mistral AI API client initialized.")

        # 3. Create chains
        self.sgr_chain = create_sgr_chain(self.llm)
        self.simple_chain = create_simple_qa_chain(self.llm)
        self.router = QueryRouter()
        print("Pipeline ready.")

    def run(self, query: str, top_k_content: int = None, include_trace: bool = False,
            force_sgr: bool = False, use_cache: bool = None) -> Dict[str, Any]:
        """
        Runs the RAG pipeline with automatic query routing.

        Args:
            query (str): The user's question.
            top_k_content (int): Number of documents to retrieve. If None, router decides.
            include_trace (bool): If True, includes the full reasoning trace in the output.
            force_sgr (bool): Force using SGR chain even for simple queries.
            use_cache (bool): Override instance cache setting.

        Returns:
            A dictionary containing the answer, context, query_type, and optionally trace.
        """
        # Check cache first
        should_use_cache = use_cache if use_cache is not None else self.use_cache

        if should_use_cache and self.cache:
            cache_key = {"top_k": top_k_content, "force_sgr": force_sgr}
            cached = self.cache.get(query, **cache_key)
            if cached:
                cached["from_cache"] = True
                return cached

        start_time = time.time()

        # 1. Route query
        query_type = self.router.route(query)
        params = self.router.get_retrieval_params(query_type)

        # Override params if user specified
        if top_k_content is not None:
            params["top_k"] = top_k_content
        if force_sgr:
            params["use_sgr"] = True

        # 2. Retrieve documents
        initial_context_docs = self.retriever.search(query, k=params["top_k"])

        if not initial_context_docs:
            return {
                "answer": "К сожалению, по вашему запросу ничего не найдено.",
                "context": [],
                "trace": [],
                "query_type": query_type,
                "latency_ms": (time.time() - start_time) * 1000
            }

        # 3. Choose chain based on routing
        if params["use_sgr"]:
            chain_inputs = {
                "query": query,
                "context": initial_context_docs,
                "retriever": self.retriever
            }
            generation_result = self.sgr_chain.invoke(chain_inputs)
        else:
            # Use simple chain
            chain_inputs = {
                "query": query,
                "context": initial_context_docs
            }
            generation_result = self.simple_chain.invoke(chain_inputs)

        # 4. Format output
        output = {
            "answer": generation_result["answer"],
            "context": [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "Источник не указан"),
                    "score": doc.metadata.get("combined_score", 0.0)
                } for doc in initial_context_docs
            ],
            "query_type": query_type,
            "latency_ms": (time.time() - start_time) * 1000,
            "from_cache": False
        }

        if include_trace:
            output["trace"] = generation_result.get("trace", [])

        # Save to cache
        if should_use_cache and self.cache:
            cache_key = {"top_k": top_k_content, "force_sgr": force_sgr}
            self.cache.set(query, output, **cache_key)

        return output

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}

    def clear_cache(self):
        """Clear the query cache"""
        if self.cache:
            self.cache.clear()
