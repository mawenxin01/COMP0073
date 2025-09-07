"""
Settings and environment configuration
设置和环境配置
"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 环境配置
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

# API 配置
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_triage.db")

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"

# 模型存储路径
MODEL_STORAGE_PATH = BASE_DIR / "models"

# 数据存储路径
DATA_STORAGE_PATH = BASE_DIR / "data_processing"

# 缓存配置
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1小时

# 安全配置
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))  # 每分钟请求数 