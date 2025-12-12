import os
import pickle
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain_core.documents import Document

class PDDRetriever:
    """
    Retrieves relevant sections and points from the Russian Road Traffic Rules (ПДД)
    using a three-level semantic search with FAISS.

    This class consolidates the advanced retrieval logic, replacing the simpler
    vector store. It performs a two-stage search:
    1. Finds the most relevant sections based on their name and description.
    2. Finds the most relevant content points within those top sections.
    3. Re-ranks the content points by combining section and content scores.

    The search method returns a list of LangChain Document objects, making it
    compatible with LangChain pipelines.
    """
    def __init__(self, pdd_path: str = "data/pdd.json", cache_dir: str = "faiss_langchain_cache",
                 embedding_model: str = "intfloat/multilingual-e5-base"):
        """
        Initializes the retriever by loading data and FAISS indexes.

        Args:
            pdd_path (str): Path to the structured pdd.json file.
            cache_dir (str): Directory where FAISS indexes and metadata are stored.
            embedding_model (str): HuggingFace embedding model name.
        """
        self.pdd_path = pdd_path
        self.cache_dir = cache_dir
        self.model_name = embedding_model

        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        self.sections: List[Dict] = []
        self.all_content_texts: List[str] = []
        self.all_content_metadata: List[Dict] = []
        self.section_index: Any = None
        self.content_index: Any = None

        self._load_data()
        self._load_or_create_indexes()

    def _load_data(self):
        """Loads the structured PDD data from the JSON file."""
        try:
            with open(self.pdd_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.sections = [item['section'] for item in data]
            print(f"Loaded {len(self.sections)} sections from {self.pdd_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.pdd_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {self.pdd_path}")

    def _load_or_create_indexes(self):
        """Loads FAISS indexes from cache or creates them if they don't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)
        section_index_path = os.path.join(self.cache_dir, 'sections_index.faiss')
        content_index_path = os.path.join(self.cache_dir, 'content_index.faiss')
        metadata_path = os.path.join(self.cache_dir, 'content_metadata.pkl')

        if all(os.path.exists(p) for p in [section_index_path, content_index_path, metadata_path]):
            print("Loading FAISS indexes from cache...")
            self.section_index = faiss.read_index(section_index_path)
            self.content_index = faiss.read_index(content_index_path)
            with open(metadata_path, 'rb') as f:
                self.all_content_metadata = pickle.load(f)
            self.all_content_texts = [item['text'] for item in self.all_content_metadata]
        else:
            print("Cache not found. Creating FAISS indexes...")
            self._create_and_save_indexes(section_index_path, content_index_path, metadata_path)

    def _create_and_save_indexes(self, section_path: str, content_path: str, metadata_path: str):
        """Creates, normalizes, and saves FAISS indexes and metadata."""
        # 1. Vectorize sections
        section_texts = [f"{sec['name']}. {sec.get('description', '')}" for sec in self.sections]

        # E5 models need "passage: " prefix for documents
        if "e5" in self.model_name.lower():
            section_texts = [f"passage: {text}" for text in section_texts]

        section_embeddings = self.embedding_model.encode(section_texts, convert_to_numpy=True)
        faiss.normalize_L2(section_embeddings)
        
        self.section_index = faiss.IndexFlatIP(section_embeddings.shape[1])
        self.section_index.add(section_embeddings)
        faiss.write_index(self.section_index, section_path)

        # 2. Vectorize content (points and sub-points) and prepare metadata
        for sec_idx, sec in enumerate(self.sections):
            for point in sec['content']:
                self.all_content_metadata.append({
                    'text': point['full_text'], 
                    'sec_num': sec['sec_num'], 
                    'p_num': point['p_num'],
                    'sec_name': sec['name'],
                    'sec_idx': sec_idx # Store index for fast lookup
                })
                if 'p_sup' in point:
                    for sub_point in point['p_sup']:
                        self.all_content_metadata.append({
                            'text': sub_point['full_text'], 
                            'sec_num': sec['sec_num'], 
                            'p_num': sub_point['p_num'],
                            'sec_name': sec['name'],
                            'sec_idx': sec_idx
                        })
        
        self.all_content_texts = [item['text'] for item in self.all_content_metadata]

        # E5 models need "passage: " prefix for documents
        content_texts_to_encode = self.all_content_texts
        if "e5" in self.model_name.lower():
            content_texts_to_encode = [f"passage: {text}" for text in self.all_content_texts]

        content_embeddings = self.embedding_model.encode(content_texts_to_encode, convert_to_numpy=True)
        faiss.normalize_L2(content_embeddings)

        self.content_index = faiss.IndexFlatIP(content_embeddings.shape[1])
        self.content_index.add(content_embeddings)
        faiss.write_index(self.content_index, content_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.all_content_metadata, f)
        print("Indexes created and saved to cache.")

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs a three-level search to find the most relevant PDD points.

        Args:
            query (str): The user's query.
            k (int): The number of final content points to return.

        Returns:
            A list of LangChain Document objects, sorted by relevance.
        """
        # Level 1: Search by sections to get relevance scores for all sections
        # E5 models need "query: " prefix for queries
        query_text = query
        if "e5" in self.model_name.lower():
            query_text = f"query: {query}"

        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Get scores for all sections
        section_scores, section_ids = self.section_index.search(query_embedding, self.section_index.ntotal)
        section_score_map = {
            section_id: score 
            for section_id, score in zip(section_ids[0], section_scores[0])
        }
        
        # Level 2: Search all content points
        content_scores, content_ids = self.content_index.search(query_embedding, self.content_index.ntotal)

        # Level 3: Filter and re-rank
        results_with_scores = []
        for score, idx in zip(content_scores[0], content_ids[0]):
            metadata = self.all_content_metadata[idx]
            sec_idx = metadata['sec_idx']
            
            # Find the section score for this content item
            section_score = 0.0
            for s_id, s_score in section_score_map.items():
                if s_id == sec_idx:
                    section_score = s_score
                    break
            
            # Normalize scores to [0, 1] range for combination
            norm_section_score = (section_score + 1) / 2
            norm_content_score = (score + 1) / 2
            combined_score = norm_section_score * norm_content_score
            
            results_with_scores.append({
                "metadata": metadata,
                "content_score": score,
                "section_score": section_score,
                "combined_score": combined_score
            })

        # Sort by combined score and take top k
        results_with_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = results_with_scores[:k]

        # Convert to LangChain Document format
        documents = []
        for res in top_results:
            metadata = res['metadata']
            doc_metadata = {
                "sec_num": metadata['sec_num'],
                "sec_name": metadata['sec_name'],
                "p_num": metadata['p_num'],
                "source": f"Раздел {metadata['sec_num']} ({metadata['sec_name']}), пункт {metadata['p_num']}",
                "section_score": float(res['section_score']),
                "content_score": float(res['content_score']),
                "combined_score": float(res['combined_score'])
            }
            documents.append(Document(page_content=metadata['text'], metadata=doc_metadata))

        return documents
