#!/usr/bin/env python3
"""
开发环境配置文件
开发和调试时使用的配置
"""

from .base import BaseConfig
from .openai import OpenAIConfig
from .azure import AzureConfig
from .models import ModelConfig
from .logging import LoggingConfig
from .database import DatabaseConfig


class DevelopmentConfig(BaseConfig):
    """开发环境配置类"""
    
    # 环境标识
    ENVIRONMENT = "development"
    DEBUG = True
    
    # 日志配置（开发环境更详细）
    LOG_LEVEL = "DEBUG"
    CONSOLE_LOG_LEVEL = "DEBUG"
    FILE_LOG_LEVEL = "DEBUG"
    
    # 数据库配置（开发环境使用 SQLite）
    DATABASE_TYPE = "sqlite"
    ENABLE_SQL_ECHO = True  # 显示 SQL 语句
    
    # API 配置（开发环境更宽松）
    API_TIMEOUT = 60  # 更长的超时时间用于调试
    MAX_RETRIES = 5
    
    # 模型配置（开发环境使用更小的模型/数据集）
    DEFAULT_N_CLUSTERS = 10  # 更少的聚类数用于快速测试
    DEFAULT_SAMPLES_PER_CLUSTER = 100  # 更少的样本
    
    # 批处理配置（开发环境更小的批次）
    API_BATCH_SIZE = 10
    INSERT_BATCH_SIZE = 100
    
    # 缓存配置（开发环境使用内存缓存）
    CACHE_TYPE = "memory"
    CACHE_TTL = 300  # 5分钟缓存
    
    # 测试配置
    ENABLE_UNIT_TESTS = True
    TEST_DATA_SIZE = 1000  # 测试时使用的数据量
    
    # 前端配置
    FRONTEND_HOST = "localhost"
    FRONTEND_PORT = 5000
    FLASK_DEBUG = True
    
    # 性能监控配置
    ENABLE_PROFILING = True
    ENABLE_MEMORY_TRACKING = True
    
    # 安全配置（开发环境较宽松）
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5000"]
    SECRET_KEY = "dev-secret-key-change-in-production"
    
    @classmethod
    def setup_development_environment(cls):
        """设置开发环境"""
        # 设置日志
        cls.setup_logging(cls.CONSOLE_LOG_LEVEL, cls.FILE_LOG_LEVEL)
        
        # 创建必要目录
        cls.create_directories()
        cls.create_database_directories()
        
        # 验证配置
        cls.validate_development_config()
        
        print("🔧 开发环境配置完成")
    
    @classmethod
    def validate_development_config(cls):
        """验证开发环境配置"""
        print("🔍 验证开发环境配置...")
        
        # 检查数据库配置
        if not cls.validate_database_config(cls.DATABASE_TYPE):
            print("⚠️  数据库配置有问题，但开发环境可以继续")
        
        # 检查 API 配置（可选）
        if cls.OPENAI_API_KEY:
            print("✅ OpenAI API Key 已设置")
        else:
            print("⚠️  OpenAI API Key 未设置，某些功能可能无法使用")
        
        if cls.AZURE_OPENAI_API_KEY:
            print("✅ Azure OpenAI API Key 已设置")
        else:
            print("⚠️  Azure OpenAI API Key 未设置，某些功能可能无法使用")
    
    @classmethod
    def get_flask_config(cls):
        """获取 Flask 开发配置"""
        return {
            "DEBUG": cls.FLASK_DEBUG,
            "HOST": cls.FRONTEND_HOST,
            "PORT": cls.FRONTEND_PORT,
            "SECRET_KEY": cls.SECRET_KEY,
            "CORS_ORIGINS": cls.CORS_ORIGINS
        } 