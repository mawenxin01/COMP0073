#!/usr/bin/env python3
"""
生产环境配置文件
生产部署时使用的配置
"""

import os
from .base import BaseConfig
from .openai import OpenAIConfig
from .azure import AzureConfig
from .models import ModelConfig
from .logging import LoggingConfig
from .database import DatabaseConfig


class ProductionConfig(BaseConfig):
    """生产环境配置类"""
    
    # 环境标识
    ENVIRONMENT = "production"
    DEBUG = False
    
    # 日志配置（生产环境较简洁）
    LOG_LEVEL = "INFO"
    CONSOLE_LOG_LEVEL = "WARNING"
    FILE_LOG_LEVEL = "INFO"
    
    # 数据库配置（生产环境推荐 PostgreSQL）
    DATABASE_TYPE = os.getenv("DATABASE_TYPE", "postgresql")
    ENABLE_SQL_ECHO = False  # 生产环境不显示 SQL 语句
    
    # API 配置（生产环境更严格）
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # 模型配置（生产环境使用完整配置）
    DEFAULT_N_CLUSTERS = 50
    DEFAULT_SAMPLES_PER_CLUSTER = 1000
    
    # 批处理配置（生产环境优化性能）
    API_BATCH_SIZE = 100
    INSERT_BATCH_SIZE = 1000
    
    # 缓存配置（生产环境使用 Redis）
    CACHE_TYPE = "redis"
    CACHE_TTL = 3600  # 1小时缓存
    
    # 前端配置
    FRONTEND_HOST = "0.0.0.0"
    FRONTEND_PORT = int(os.getenv("PORT", "8080"))
    FLASK_DEBUG = False
    
    # 性能监控配置
    ENABLE_PROFILING = False
    ENABLE_MEMORY_TRACKING = False
    
    # 安全配置（生产环境严格）
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
    SECRET_KEY = os.getenv("SECRET_KEY", "")
    
    # SSL 配置
    SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", "")
    SSL_KEY_PATH = os.getenv("SSL_KEY_PATH", "")
    
    # 监控和告警配置
    MONITORING_CONFIG = {
        "enable_health_check": True,
        "health_check_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "sentry_dsn": os.getenv("SENTRY_DSN", ""),
        "datadog_api_key": os.getenv("DATADOG_API_KEY", "")
    }
    
    # 限流配置
    RATE_LIMITING = {
        "enabled": True,
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
    }
    
    # 工作进程配置
    WORKER_CONFIG = {
        "workers": int(os.getenv("WORKERS", "4")),
        "worker_class": "gevent",
        "worker_connections": 1000,
        "timeout": 30,
        "keepalive": 2
    }
    
    @classmethod
    def setup_production_environment(cls):
        """设置生产环境"""
        # 验证必需的配置
        if not cls.validate_production_config():
            raise ValueError("生产环境配置验证失败")
        
        # 设置日志
        cls.setup_logging(cls.CONSOLE_LOG_LEVEL, cls.FILE_LOG_LEVEL)
        
        # 创建必要目录
        cls.create_directories()
        cls.create_database_directories()
        
        print("🚀 生产环境配置完成")
    
    @classmethod
    def validate_production_config(cls) -> bool:
        """验证生产环境配置"""
        print("🔍 验证生产环境配置...")
        
        required_configs = []
        
        # 检查必需的安全配置
        if not cls.SECRET_KEY:
            required_configs.append("SECRET_KEY")
        
        # 检查数据库配置
        if cls.DATABASE_TYPE != "sqlite":
            if not cls.validate_database_config(cls.DATABASE_TYPE):
                required_configs.append("数据库配置")
        
        # 检查 API 密钥
        if not cls.OPENAI_API_KEY and not cls.AZURE_OPENAI_API_KEY:
            required_configs.append("OpenAI 或 Azure OpenAI API Key")
        
        # 检查缓存配置
        if cls.CACHE_TYPE == "redis":
            redis_config = cls.get_cache_client_config("redis")
            if not redis_config["host"]:
                required_configs.append("Redis 配置")
        
        if required_configs:
            print("❌ 生产环境缺少以下必需配置:")
            for config in required_configs:
                print(f"   - {config}")
            return False
        
        print("✅ 生产环境配置验证通过")
        return True
    
    @classmethod
    def get_gunicorn_config(cls):
        """获取 Gunicorn 生产配置"""
        return {
            "bind": f"{cls.FRONTEND_HOST}:{cls.FRONTEND_PORT}",
            "workers": cls.WORKER_CONFIG["workers"],
            "worker_class": cls.WORKER_CONFIG["worker_class"],
            "worker_connections": cls.WORKER_CONFIG["worker_connections"],
            "timeout": cls.WORKER_CONFIG["timeout"],
            "keepalive": cls.WORKER_CONFIG["keepalive"],
            "max_requests": 1000,
            "max_requests_jitter": 100,
            "preload_app": True,
            "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
            "keyfile": cls.SSL_KEY_PATH if cls.SSL_KEY_PATH else None,
            "certfile": cls.SSL_CERT_PATH if cls.SSL_CERT_PATH else None
        }
    
    @classmethod
    def get_flask_config(cls):
        """获取 Flask 生产配置"""
        return {
            "DEBUG": cls.FLASK_DEBUG,
            "SECRET_KEY": cls.SECRET_KEY,
            "CORS_ORIGINS": cls.CORS_ORIGINS,
            "ENV": "production"
        } 