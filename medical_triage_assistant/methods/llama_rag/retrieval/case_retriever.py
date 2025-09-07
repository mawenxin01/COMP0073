#!/usr/bin/env python3
"""
Medical Case Retrieval Module
Manages retrieval of similar medical cases for triage decision support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add project root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from medical_triage_assistant.methods.llama_rag.vector_store.faiss_store import FAISSVectorStore


class CaseRetriever:
    """Medical Case Retrieval System"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize case retriever
        
        Args:
            embedding_model: Model used for text embeddings
        """
        self.vector_store = FAISSVectorStore(embedding_model=embedding_model)
        self.initialized = False
        
        print(f"🔍 Case Retriever initialized")
        print(f"   Embedding model: {embedding_model}")
    
    def initialize_from_data(self, data_path: str, save_path: str = "case_retriever_index", 
                           force_rebuild: bool = False) -> Dict:
        """
        Initialize retriever from training data
        
        Args:
            data_path: Path to training data CSV file
            save_path: Path to save FAISS index
            force_rebuild: Whether to force rebuild the index
            
        Returns:
            Dict: Initialization status information
        """
        
        try:
            print(f"📖 Loading training data from: {data_path}")
            
            # Load data
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            data = pd.read_csv(data_path, low_memory=False)
            print(f"✅ Data loaded successfully: {len(data)} cases")
            
            # Prepare case database
            print(f"🔍 Preparing case database...")
            prepared_data = self.vector_store.prepare_case_database(data)
            
            # Build or load FAISS index
            print(f"🔍 Building/loading FAISS index...")
            was_cached = self.vector_store.build_or_load_faiss_index(
                prepared_data, 
                save_path=save_path, 
                force_rebuild=force_rebuild
            )
            
            self.initialized = True
            
            return {
                "status": "success",
                "message": "Case retriever initialized successfully",
                "data_file": data_path,
                "total_cases": len(prepared_data),
                "index_file": save_path,
                "was_cached": was_cached,
                "feature_dimension": self.vector_store.faiss_index.d if self.vector_store.faiss_index else 0
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize case retriever: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data_file": data_path
            }
    
    def retrieve_similar_cases(self, query_case: Dict, k: int = 5, use_hybrid: bool = False, 
                             use_reranking: bool = False) -> Tuple[List[Dict], List[float], Dict]:
        """
        Retrieve similar cases for a query case
        
        Args:
            query_case: Query case dictionary
            k: Number of similar cases to retrieve
            use_hybrid: Whether to use hybrid BM25 + vector retrieval
            use_reranking: Whether to use cross-encoder reranking (only if use_hybrid=True)
            
        Returns:
            Tuple containing similar cases, similarities, and metadata
        """
        
        if not self.initialized:
            raise ValueError("Case retriever not initialized. Please call initialize_from_data first.")
        
        try:
            # Convert to pandas Series for processing
            query_series = pd.Series(query_case)
            
            # Retrieve similar cases with hybrid options
            similar_cases, similarities = self.vector_store.retrieve_similar_cases(
                query_series, k=k, use_hybrid=use_hybrid, use_reranking=use_reranking
            )
            
            # Prepare metadata
            metadata = {
                "query_processed": True,
                "retrieved_count": len(similar_cases),
                "similarity_scores": similarities,
                "avg_similarity": np.mean(similarities) if similarities else 0.0,
                "max_similarity": np.max(similarities) if similarities else 0.0,
                "min_similarity": np.min(similarities) if similarities else 0.0,
                "retrieval_method": "hybrid" if use_hybrid else "vector_only",
                "reranking_used": use_reranking if use_hybrid else False
            }
            
            return similar_cases, similarities, metadata
            
        except Exception as e:
            print(f"❌ Failed to retrieve similar cases: {e}")
            return [], [], {"error": str(e)}
    
    def get_status(self) -> Dict:
        """Get current status of the case retriever"""
        
        return {
            "initialized": self.initialized,
            "status": "initialized" if self.initialized else "not_initialized",
            "total_cases": self.vector_store.faiss_index.ntotal if self.vector_store.faiss_index else 0,
            "feature_dimension": self.vector_store.faiss_index.d if self.vector_store.faiss_index else 0
        }
    
    def create_case_description(self, case_data: pd.Series) -> str:
        """Provide case description creation functionality for external use"""
        
        return self.vector_store._create_case_description(case_data)


# Convenience function for quick retrieval
def quick_retrieve(query_case: Dict, data_path: str, k: int = 5, 
                  save_path: str = "quick_retriever_index", use_hybrid: bool = False,
                  use_reranking: bool = False) -> Tuple[List[Dict], List[float]]:
    """
    Quick retrieval convenience function for similar cases
    
    Args:
        query_case: Query case dictionary
        data_path: Training data path
        k: Number of similar cases to return
        save_path: Index save path
        use_hybrid: Whether to use hybrid BM25 + vector retrieval
        use_reranking: Whether to use cross-encoder reranking (only if use_hybrid=True)
        
    Returns:
        Tuple of similar cases and similarity scores
    """
    
    retriever = CaseRetriever()
    
    # Initialize
    init_result = retriever.initialize_from_data(data_path, save_path)
    if init_result["status"] != "success":
        print(f"❌ Quick retrieval initialization failed: {init_result['message']}")
        return [], []
    
    # Retrieve
    similar_cases, similarities, metadata = retriever.retrieve_similar_cases(
        query_case, k, use_hybrid=use_hybrid, use_reranking=use_reranking
    )
    
    if "error" in metadata:
        print(f"❌ Quick retrieval failed: {metadata['error']}")
        return [], []
    
    return similar_cases, similarities