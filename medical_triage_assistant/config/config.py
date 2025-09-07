"""
Main configuration file for Medical Triage Assistant
医疗分诊助手主配置文件
"""

import os
from typing import Dict, Any

class Config:
    """配置管理类"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        # 基础配置
        self.base_config = {
            "project_name": "Medical Triage Assistant",
            "version": "1.0.0",
            "log_level": "INFO"
        }
        
        # 数据处理配置
        self.data_config = {
            "raw_data_path": "data_processing/raw_data",
            "processed_data_path": "data_processing/processed_data",
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "supported_formats": [".txt", ".csv", ".json", ".xml"]
        }
        
        # 关键词提取配置
        self.keyword_config = {
            "min_keyword_length": 2,
            "max_keywords_per_doc": 50,
            "stop_words_language": "chinese"
        }
        
        # TF-IDF + ML 配置
        self.tfidf_ml_config = {
            "vectorizer_params": {
                "max_features": 10000,
                "ngram_range": (1, 2),
                "min_df": 2
            },
            "model_params": {
                "random_state": 42,
                "test_size": 0.2
            }
        }
        
        # ChatGPT 配置
        self.chatgpt_config = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        # RedFlag 配置
        self.redflag_config = {
            "sensitivity_threshold": 0.8,
            "rule_weights": {
                "symptom_severity": 0.4,
                "vital_signs": 0.3,
                "patient_history": 0.3
            }
        }
        
        # GPT + RAG 配置
        self.gpt_rag_config = {
            "vector_store_type": "faiss",  # 或 "chroma", "pinecone"
            "embedding_model": "text-embedding-ada-002",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k_retrieval": 5
        }
        
        # WatsonX 配置 (如果使用)
        self.watsonx_config = {
            "api_key": os.getenv("WATSONX_API_KEY", ""),
            "endpoint": os.getenv("WATSONX_ENDPOINT", ""),
            "model": "meta-llama/llama-2-70b-chat",
            "max_tokens": 1000
        }
    
    def get_config(self, config_type: str) -> Dict[str, Any]:
        """获取指定类型的配置"""
        config_map = {
            "base": self.base_config,
            "data": self.data_config,
            "keyword": self.keyword_config,
            "tfidf_ml": self.tfidf_ml_config,
            "chatgpt": self.chatgpt_config,
            "redflag": self.redflag_config,
            "gpt_rag": self.gpt_rag_config,
            "watsonx": self.watsonx_config
        }
        return config_map.get(config_type, {})
    
    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """更新指定类型的配置"""
        if hasattr(self, f"{config_type}_config"):
            getattr(self, f"{config_type}_config").update(updates) 