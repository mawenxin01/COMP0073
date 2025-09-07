#!/usr/bin/env python3
"""
数据库配置文件
管理数据库连接和存储配置
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from .base import BaseConfig


class DatabaseConfig(BaseConfig):
    """数据库配置类"""
    
    # SQLite 配置（默认）
    SQLITE_DB_PATH = BaseConfig.PROJECT_ROOT / "data" / "triage.db"
    SQLITE_TIMEOUT = 30
    
    # PostgreSQL 配置（可选）
    POSTGRES_CONFIG = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "triage_db"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "sslmode": os.getenv("POSTGRES_SSL", "prefer")
    }
    
    # MySQL 配置（可选）
    MYSQL_CONFIG = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "database": os.getenv("MYSQL_DB", "triage_db"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "charset": "utf8mb4"
    }
    
    # 连接池配置
    CONNECTION_POOL_CONFIG = {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True
    }
    
    # 缓存配置
    CACHE_CONFIG = {
        "type": "memory",  # 或 "redis", "memcached"
        "redis": {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD"),
            "decode_responses": True,
            "socket_timeout": 30
        },
        "memory": {
            "max_size": 1000,
            "ttl": 3600  # 1小时
        }
    }
    
    # 向量数据库配置（FAISS）
    VECTOR_DB_CONFIG = {
        "faiss": {
            "index_dir": BaseConfig.PROJECT_ROOT / "data" / "indexes",
            "index_type": "IndexFlatL2",
            "dimension": 1536,
            "nlist": 100,  # 用于 IVF 索引
            "nprobe": 10   # 查询时的探针数
        },
        "pinecone": {
            "api_key": os.getenv("PINECONE_API_KEY", ""),
            "environment": os.getenv("PINECONE_ENV", ""),
            "index_name": os.getenv("PINECONE_INDEX", "triage-embeddings"),
            "dimension": 1536,
            "metric": "cosine"
        },
        "chroma": {
            "persist_directory": str(BaseConfig.PROJECT_ROOT / "data" / "chroma"),
            "collection_name": "triage_embeddings"
        }
    }
    
    # 批处理配置
    BATCH_CONFIG = {
        "insert_batch_size": 1000,
        "update_batch_size": 500,
        "select_batch_size": 10000,
        "commit_interval": 1000
    }
    
    @classmethod
    def get_database_url(cls, db_type: str = "sqlite") -> str:
        """获取数据库连接 URL"""
        if db_type == "sqlite":
            cls.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{cls.SQLITE_DB_PATH}"
        
        elif db_type == "postgresql":
            config = cls.POSTGRES_CONFIG
            return (f"postgresql://{config['user']}:{config['password']}@"
                   f"{config['host']}:{config['port']}/{config['database']}")
        
        elif db_type == "mysql":
            config = cls.MYSQL_CONFIG
            return (f"mysql+pymysql://{config['user']}:{config['password']}@"
                   f"{config['host']}:{config['port']}/{config['database']}")
        
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    @classmethod
    def get_cache_client_config(cls, cache_type: str = "memory") -> Dict[str, Any]:
        """获取缓存客户端配置"""
        if cache_type == "redis":
            return cls.CACHE_CONFIG["redis"]
        elif cache_type == "memory":
            return cls.CACHE_CONFIG["memory"]
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}")
    
    @classmethod
    def get_vector_db_config(cls, vector_db: str = "faiss") -> Dict[str, Any]:
        """获取向量数据库配置"""
        if vector_db not in cls.VECTOR_DB_CONFIG:
            raise ValueError(f"不支持的向量数据库类型: {vector_db}")
        
        config = cls.VECTOR_DB_CONFIG[vector_db].copy()
        
        # 确保 FAISS 索引目录存在
        if vector_db == "faiss":
            Path(config["index_dir"]).mkdir(parents=True, exist_ok=True)
        
        return config
    
    @classmethod
    def validate_database_config(cls, db_type: str = "sqlite") -> bool:
        """验证数据库配置"""
        if db_type == "sqlite":
            # 检查 SQLite 文件目录是否可写
            try:
                cls.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                return True
            except Exception as e:
                print(f"❌ SQLite 数据库路径无法创建: {e}")
                return False
        
        elif db_type == "postgresql":
            config = cls.POSTGRES_CONFIG
            if not all([config["host"], config["database"], config["user"]]):
                print("❌ PostgreSQL 配置不完整")
                return False
        
        elif db_type == "mysql":
            config = cls.MYSQL_CONFIG
            if not all([config["host"], config["database"], config["user"]]):
                print("❌ MySQL 配置不完整")
                return False
        
        return True
    
    @classmethod
    def create_database_directories(cls) -> None:
        """创建数据库相关目录"""
        directories = [
            cls.SQLITE_DB_PATH.parent,
            cls.VECTOR_DB_CONFIG["faiss"]["index_dir"],
            Path(cls.VECTOR_DB_CONFIG["chroma"]["persist_directory"])
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True) 