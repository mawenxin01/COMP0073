"""
Configuration module for Medical Triage Assistant
医疗分诊助手配置模块
"""

import os
from typing import Type, Union, Any

# 导入所有配置类
from .base import BaseConfig
from .openai import OpenAIConfig
from .azure import AzureConfig
from .models import ModelConfig
from .logging import LoggingConfig, setup_logging, get_logger
from .database import DatabaseConfig
from .development import DevelopmentConfig
from .production import ProductionConfig


# 配置类映射
CONFIG_MAP = {
    "base": BaseConfig,
    "openai": OpenAIConfig,
    "azure": AzureConfig,
    "models": ModelConfig,
    "logging": LoggingConfig,
    "database": DatabaseConfig,
    "development": DevelopmentConfig,
    "production": ProductionConfig
}


def get_environment() -> str:
    """获取当前环境"""
    return os.getenv("ENVIRONMENT", "development").lower()


def get_config_class() -> Type[Union[DevelopmentConfig, ProductionConfig]]:
    """根据环境获取配置类"""
    env = get_environment()
    
    if env == "production":
        return ProductionConfig
    elif env == "development":
        return DevelopmentConfig
    else:
        print(f"⚠️  未知环境 '{env}'，使用开发环境配置")
        return DevelopmentConfig


def get_config() -> Union[DevelopmentConfig, ProductionConfig]:
    """获取当前环境的配置实例"""
    config_class = get_config_class()
    return config_class()


def setup_environment():
    """设置当前环境"""
    env = get_environment()
    config_class = get_config_class()
    
    print(f"🌍 当前环境: {env}")
    
    if env == "production":
        config_class.setup_production_environment()
    else:
        config_class.setup_development_environment()


def validate_config() -> bool:
    """验证当前环境配置"""
    env = get_environment()
    config_class = get_config_class()
    
    if env == "production":
        return config_class.validate_production_config()
    else:
        config_class.validate_development_config()
        return True


# 便捷的配置获取函数
def get_api_config(provider: str = "auto") -> dict:
    """获取 API 配置"""
    config = get_config()
    
    if provider == "openai":
        return config.get_gpt_config()
    elif provider == "azure":
        return config.get_gpt_client_config()
    elif provider == "auto":
        # 自动选择可用的 API
        if config.AZURE_OPENAI_API_KEY:
            return config.get_gpt_client_config()
        elif config.OPENAI_API_KEY:
            return config.get_gpt_config()
        else:
            raise ValueError("未找到可用的 API 配置")
    else:
        raise ValueError(f"不支持的 API 提供商: {provider}")


def get_model_config(model_name: str = None) -> dict:
    """获取模型配置"""
    config = get_config()
    if model_name:
        return config.get_model_config(model_name)
    else:
        return {
            "random_forest": config.RANDOM_FOREST_CONFIG,
            "xgboost": config.XGBOOST_CONFIG,
            "clustering": config.CLUSTERING_CONFIG
        }


def get_database_config(db_type: str = None) -> dict:
    """获取数据库配置"""
    config = get_config()
    db_type = db_type or getattr(config, 'DATABASE_TYPE', 'sqlite')
    
    return {
        "url": config.get_database_url(db_type),
        "type": db_type,
        "batch_config": config.BATCH_CONFIG
    }


def get_vector_db_config(vector_db: str = "faiss") -> dict:
    """获取向量数据库配置"""
    config = get_config()
    return config.get_vector_db_config(vector_db)


# 导出主要接口
__all__ = [
    # 配置类
    "BaseConfig",
    "OpenAIConfig", 
    "AzureConfig",
    "ModelConfig",
    "LoggingConfig",
    "DatabaseConfig",
    "DevelopmentConfig",
    "ProductionConfig",
    
    # 核心函数
    "get_environment",
    "get_config_class",
    "get_config",
    "setup_environment",
    "validate_config",
    
    # 便捷函数
    "get_api_config",
    "get_model_config", 
    "get_database_config",
    "get_vector_db_config",
    
    # 日志函数
    "setup_logging",
    "get_logger",
    
    # 配置映射
    "CONFIG_MAP"
]


# 模块级别的配置实例（懒加载）
_config_instance = None


def config() -> Union[DevelopmentConfig, ProductionConfig]:
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_config()
    return _config_instance


# 添加到导出列表
__all__.append("config")


# 模块初始化时的提示
def _module_init():
    """模块初始化"""
    env = get_environment()
    print(f"📦 配置模块已加载 - 环境: {env}")


# 执行初始化
_module_init() 