#!/usr/bin/env python3
"""
OpenAI API 配置文件
管理 OpenAI 相关的 API 密钥和模型参数
"""

import os
from typing import Dict, Any, Optional
from .base import BaseConfig


class OpenAIConfig(BaseConfig):
    """OpenAI API 配置类"""
    
    # OpenAI API 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")
    
    # 模型配置
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    GPT_MODEL: str = "gpt-3.5-turbo"
    GPT4_MODEL: str = "gpt-4"
    
    # 嵌入向量配置
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-ada-002 的维度
    MAX_TOKENS_PER_REQUEST: int = 8000  # GPT-3.5-turbo 的最大 token 数
    
    # API 调用配置
    API_BATCH_SIZE: int = 100
    API_SLEEP_TIME: float = 0.1  # API 调用间隔时间（秒）
    GPT_SLEEP_TIME: float = 1.0  # GPT 调用间隔时间（秒）
    MAX_RETRIES: int = 3
    
    # 聚类配置
    DEFAULT_N_CLUSTERS: int = 50
    DEFAULT_SAMPLES_PER_CLUSTER: int = 1000
    
    # 温度和其他生成参数
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000
    DEFAULT_TOP_P: float = 1.0
    DEFAULT_FREQUENCY_PENALTY: float = 0.0
    DEFAULT_PRESENCE_PENALTY: float = 0.0
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """验证 API Key 是否设置"""
        if not cls.OPENAI_API_KEY:
            print("❌ 未设置 OPENAI_API_KEY 环境变量")
            print("   请运行: export OPENAI_API_KEY='your-api-key-here'")
            return False
        return True
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """获取嵌入配置"""
        return {
            "model": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "batch_size": cls.API_BATCH_SIZE,
            "sleep_time": cls.API_SLEEP_TIME,
            "max_retries": cls.MAX_RETRIES
        }
    
    @classmethod
    def get_gpt_config(cls) -> Dict[str, Any]:
        """获取 GPT 配置"""
        return {
            "model": cls.GPT_MODEL,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_tokens": cls.DEFAULT_MAX_TOKENS,
            "top_p": cls.DEFAULT_TOP_P,
            "frequency_penalty": cls.DEFAULT_FREQUENCY_PENALTY,
            "presence_penalty": cls.DEFAULT_PRESENCE_PENALTY,
            "sleep_time": cls.GPT_SLEEP_TIME,
            "max_retries": cls.MAX_RETRIES
        }
    
    @classmethod
    def get_clustering_config(cls) -> Dict[str, Any]:
        """获取聚类配置"""
        return {
            "n_clusters": cls.DEFAULT_N_CLUSTERS,
            "samples_per_cluster": cls.DEFAULT_SAMPLES_PER_CLUSTER,
            "random_state": cls.RANDOM_STATE,
            "batch_size": cls.API_BATCH_SIZE
        } 