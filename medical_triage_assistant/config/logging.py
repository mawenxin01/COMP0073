#!/usr/bin/env python3
"""
日志配置文件
管理应用程序的日志设置
"""

import os
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any
from .base import BaseConfig


class LoggingConfig(BaseConfig):
    """日志配置类"""
    
    # 日志级别配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 日志文件配置
    LOG_DIR = BaseConfig.PROJECT_ROOT / "logs"
    LOG_FILE = LOG_DIR / "triage_analysis.log"
    ERROR_LOG_FILE = LOG_DIR / "error.log"
    
    # 日志格式配置
    DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    SIMPLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    CONSOLE_FORMAT = "%(levelname)-8s | %(name)-20s | %(message)s"
    
    # 日志轮转配置
    MAX_BYTES = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    
    @classmethod
    def setup_logging(cls, console_level: str = None, file_level: str = None) -> None:
        """设置日志配置"""
        # 创建日志目录
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 确定日志级别
        console_level = console_level or cls.LOG_LEVEL
        file_level = file_level or cls.LOG_LEVEL
        
        # 配置字典
        config = cls.get_logging_config(console_level, file_level)
        
        # 应用配置
        logging.config.dictConfig(config)
    
    @classmethod
    def get_logging_config(cls, console_level: str = "INFO", file_level: str = "INFO") -> Dict[str, Any]:
        """获取日志配置字典"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": cls.DETAILED_FORMAT,
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "simple": {
                    "format": cls.SIMPLE_FORMAT,
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "console": {
                    "format": cls.CONSOLE_FORMAT,
                    "datefmt": "%H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": console_level,
                    "formatter": "console",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": file_level,
                    "formatter": "detailed",
                    "filename": str(cls.LOG_FILE),
                    "maxBytes": cls.MAX_BYTES,
                    "backupCount": cls.BACKUP_COUNT,
                    "encoding": "utf-8"
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": str(cls.ERROR_LOG_FILE),
                    "maxBytes": cls.MAX_BYTES,
                    "backupCount": cls.BACKUP_COUNT,
                    "encoding": "utf-8"
                }
            },
            "loggers": {
                "triage": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "openai": {
                    "level": "WARNING",
                    "handlers": ["file"],
                    "propagate": False
                },
                "azure": {
                    "level": "WARNING", 
                    "handlers": ["file"],
                    "propagate": False
                },
                "httpx": {
                    "level": "WARNING",
                    "handlers": ["file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console", "file"]
            }
        }
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        return logging.getLogger(f"triage.{name}")
    
    @classmethod
    def configure_external_loggers(cls) -> None:
        """配置第三方库的日志级别"""
        # 降低第三方库的日志级别
        external_loggers = [
            "openai",
            "azure.core",
            "azure.identity", 
            "httpx",
            "urllib3",
            "requests",
            "matplotlib",
            "PIL"
        ]
        
        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


# 便捷的日志记录器获取函数
def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return LoggingConfig.get_logger(name)


# 快速设置日志的函数
def setup_logging(console_level: str = "INFO", file_level: str = "INFO") -> None:
    """快速设置日志的便捷函数"""
    LoggingConfig.setup_logging(console_level, file_level)
    LoggingConfig.configure_external_loggers() 