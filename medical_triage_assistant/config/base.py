#!/usr/bin/env python3
"""
基础配置文件
包含项目的核心配置和默认值
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class BaseConfig:
    """基础配置类，包含所有默认配置"""
    
    # 项目基础路径
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 数据目录配置
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    FINAL_RESULTS_DIR = DATA_DIR / "final_results"
    
    # 输出目录配置
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    EXAMPLES_DIR = PROJECT_ROOT / "examples"
    EVALUATION_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "logs" / "triage_analysis.log"
    
    # 文件编码配置
    DEFAULT_ENCODING = "utf-8"
    
    # 随机种子配置
    RANDOM_STATE = 42
    
    # 批处理配置
    DEFAULT_BATCH_SIZE = 100
    
    # 超时配置
    DEFAULT_TIMEOUT = 30  # 秒
    
    # 医疗术语配置
    MEDICAL_ABBREVIATIONS: Dict[str, str] = {
        "n/v/d": "nausea vomiting diarrhea",
        "n/v": "nausea vomiting", 
        "s/p": "status post",
        "abd": "abdominal",
        "lethagic": "lethargy",
        "b pedal edema": "bilateral pedal edema",
        " l ": " left ",
        " r ": " right ",
        " luq ": " left upper quadrant ",
        " llq ": " left lower quadrant ",
        " ruq ": " right upper quadrant ",
        " rlq ": " right lower quadrant "
    }
    
    # 分诊等级配置
    TRIAGE_LEVELS = {
        1: "危重 (Critical)",
        2: "严重 (Urgent)", 
        3: "中等 (Semi-urgent)",
        4: "轻微 (Non-urgent)"
    }
    
    @classmethod
    def create_directories(cls) -> None:
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INTERIM_DATA_DIR,
            cls.FINAL_RESULTS_DIR,
            cls.MODELS_DIR,
            cls.OUTPUT_DIR,
            cls.EVALUATION_RESULTS_DIR,
            cls.LOG_FILE.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            "project_root": str(cls.PROJECT_ROOT),
            "data_dir": str(cls.DATA_DIR),
            "models_dir": str(cls.MODELS_DIR),
            "output_dir": str(cls.OUTPUT_DIR),
            "log_level": cls.LOG_LEVEL,
            "random_state": cls.RANDOM_STATE,
            "batch_size": cls.DEFAULT_BATCH_SIZE,
            "timeout": cls.DEFAULT_TIMEOUT
        } 