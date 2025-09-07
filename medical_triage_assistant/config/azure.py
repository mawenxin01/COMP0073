#!/usr/bin/env python3
"""
Azure OpenAI 配置文件
管理 Azure OpenAI 相关的 API 密钥和端点配置
"""

import os
from typing import Dict, Any, Optional
from .base import BaseConfig


class AzureConfig(BaseConfig):
    """Azure OpenAI 配置类"""
    
    # 主要 Azure OpenAI 配置
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    # 嵌入模型配置
    EMBEDDING_API_KEY: str = os.getenv("AZURE_EMBEDDING_API_KEY", "")
    EMBEDDING_ENDPOINT: str = os.getenv("AZURE_EMBEDDING_ENDPOINT", "")
    EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    # GPT 模型配置
    GPT_API_KEY: str = os.getenv("AZURE_GPT_API_KEY", "")
    GPT_ENDPOINT: str = os.getenv("AZURE_GPT_ENDPOINT", "")
    GPT_DEPLOYMENT_NAME: str = os.getenv("AZURE_GPT_DEPLOYMENT", "gpt-35-turbo-instruct")
    
    # GPT-4 配置（备用）
    GPT4_API_KEY: str = os.getenv("AZURE_GPT4_API_KEY", "")
    GPT4_ENDPOINT: str = os.getenv("AZURE_GPT4_ENDPOINT", "")
    GPT4_DEPLOYMENT_NAME: str = os.getenv("AZURE_GPT4_DEPLOYMENT", "gpt-4")
    
    # API 调用配置
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    RETRY_DELAY: float = 1.0
    
    # 模型参数配置
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000
    DEFAULT_TOP_P: float = 1.0
    
    @classmethod
    def validate_azure_config(cls) -> bool:
        """验证 Azure 配置是否完整"""
        required_configs = [
            (cls.AZURE_OPENAI_API_KEY, "AZURE_OPENAI_API_KEY"),
            (cls.AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT"),
        ]
        
        missing_configs = []
        for config_value, config_name in required_configs:
            if not config_value:
                missing_configs.append(config_name)
        
        if missing_configs:
            print("❌ 缺少以下 Azure 配置环境变量:")
            for config in missing_configs:
                print(f"   - {config}")
            print("\n请设置这些环境变量或在配置文件中直接指定值")
            return False
        
        return True
    
    @classmethod
    def get_embedding_client_config(cls) -> Dict[str, Any]:
        """获取嵌入模型客户端配置"""
        return {
            "api_key": cls.EMBEDDING_API_KEY or cls.AZURE_OPENAI_API_KEY,
            "azure_endpoint": cls.EMBEDDING_ENDPOINT or cls.AZURE_OPENAI_ENDPOINT,
            "api_version": cls.AZURE_OPENAI_API_VERSION,
            "deployment_name": cls.EMBEDDING_DEPLOYMENT_NAME,
            "max_retries": cls.MAX_RETRIES,
            "timeout": cls.REQUEST_TIMEOUT
        }
    
    @classmethod
    def get_gpt_client_config(cls) -> Dict[str, Any]:
        """获取 GPT 模型客户端配置"""
        return {
            "api_key": cls.GPT_API_KEY or cls.AZURE_OPENAI_API_KEY,
            "azure_endpoint": cls.GPT_ENDPOINT or cls.AZURE_OPENAI_ENDPOINT,
            "api_version": cls.AZURE_OPENAI_API_VERSION,
            "deployment_name": cls.GPT_DEPLOYMENT_NAME,
            "max_retries": cls.MAX_RETRIES,
            "timeout": cls.REQUEST_TIMEOUT,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_tokens": cls.DEFAULT_MAX_TOKENS,
            "top_p": cls.DEFAULT_TOP_P
        }
    
    @classmethod
    def get_gpt4_client_config(cls) -> Dict[str, Any]:
        """获取 GPT-4 模型客户端配置"""
        return {
            "api_key": cls.GPT4_API_KEY or cls.AZURE_OPENAI_API_KEY,
            "azure_endpoint": cls.GPT4_ENDPOINT or cls.AZURE_OPENAI_ENDPOINT,
            "api_version": cls.AZURE_OPENAI_API_VERSION,
            "deployment_name": cls.GPT4_DEPLOYMENT_NAME,
            "max_retries": cls.MAX_RETRIES,
            "timeout": cls.REQUEST_TIMEOUT,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_tokens": cls.DEFAULT_MAX_TOKENS,
            "top_p": cls.DEFAULT_TOP_P
        }
    
    @classmethod
    def get_failover_config(cls) -> Dict[str, Any]:
        """获取故障转移配置"""
        return {
            "primary": cls.get_gpt_client_config(),
            "fallback": cls.get_gpt4_client_config() if cls.GPT4_API_KEY else None,
            "retry_delay": cls.RETRY_DELAY,
            "max_retries": cls.MAX_RETRIES
        } 