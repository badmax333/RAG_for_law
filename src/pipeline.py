import os
import sys
from typing import Dict, List, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI

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
        
        # 2. Initialize the LLM via Mistral API
        if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
            raise ValueError(
                "MISTRAL_API_KEY not set. Please set it in .env file or config.py. "
                "Get your API key at https://console.mistral.ai/"
            )
        
        print("Initializing Mistral AI API client...")
        # Инициализация LLM через Mistral API
        # Примечание: таймаут настраивается автоматически библиотекой
        self.llm = ChatMistralAI(
            model="mistral-large-latest",  # Используем большую модель через API
            mistral_api_key=MISTRAL_API_KEY,
            temperature=0.2,
            max_tokens=2048,
        )
        print("Mistral AI API client initialized.")

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
