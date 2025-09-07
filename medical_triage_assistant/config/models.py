#!/usr/bin/env python3
"""
机器学习模型配置文件
管理各种 ML 模型的参数配置
"""

from typing import Dict, Any, List
from .base import BaseConfig


class ModelConfig(BaseConfig):
    """机器学习模型配置类"""
    
    # 通用模型配置
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    N_JOBS = -1  # 使用所有 CPU 核心
    
    # Random Forest 配置
    RANDOM_FOREST_CONFIG = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "class_weight": "balanced"  # 处理类别不平衡
    }
    
    # XGBoost 配置
    XGBOOST_CONFIG = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "eval_metric": "mlogloss"
    }
    
    # LightGBM 配置
    LIGHTGBM_CONFIG = {
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "class_weight": "balanced"
    }
    
    # SVM 配置
    SVM_CONFIG = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    }
    
    # KNN 配置
    KNN_CONFIG = {
        "n_neighbors": 5,
        "weights": "distance",
        "algorithm": "auto",
        "metric": "cosine",  # 适合嵌入向量
        "n_jobs": N_JOBS
    }
    
    # BERT 嵌入配置
    BERT_CONFIG = {
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "batch_size": 16,
        "device": "cpu"  # 如果有 GPU 可设置为 "cuda"
    }
    
    # FAISS 索引配置
    FAISS_CONFIG = {
        "index_type": "IndexFlatL2",  # 或 "IndexIVFFlat" 用于大数据集
        "nlist": 100,  # 仅用于 IVF 索引
        "metric_type": "L2",  # 或 "IP" (内积)
        "dimension": 1536  # text-embedding-ada-002 的维度
    }
    
    # PCA 配置
    PCA_CONFIG = {
        "n_components": 50,
        "random_state": RANDOM_STATE,
        "svd_solver": "auto"
    }
    
    # 聚类配置
    CLUSTERING_CONFIG = {
        "kmeans": {
            "n_clusters": 50,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS
        },
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "n_jobs": N_JOBS
        }
    }
    
    # 两阶段分类配置
    TWO_STAGE_CONFIG = {
        "stage1": {
            "model_type": "random_forest",
            "target": "binary_critical",  # 是否危重
            "config": RANDOM_FOREST_CONFIG
        },
        "stage2_critical": {
            "model_type": "xgboost",
            "target": "critical_subclass",  # 危重子类别
            "config": XGBOOST_CONFIG
        },
        "stage2_noncritical": {
            "model_type": "random_forest",
            "target": "noncritical_subclass",  # 非危重子类别
            "config": RANDOM_FOREST_CONFIG
        }
    }
    
    # 特征工程配置
    FEATURE_CONFIG = {
        "text_features": {
            "tfidf_max_features": 5000,
            "tfidf_ngram_range": (1, 2),
            "tfidf_min_df": 2,
            "tfidf_max_df": 0.95
        },
        "numerical_features": {
            "age_bins": [0, 18, 35, 50, 65, 100],
            "vitals_outlier_threshold": 3  # 标准差
        },
        "embedding_features": {
            "dimension": 1536,
            "normalize": True,
            "use_pca": True,
            "pca_components": 50
        }
    }
    
    # 评估指标配置
    EVALUATION_CONFIG = {
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "cv_folds": 5,
        "scoring": "f1_weighted",  # 主要评估指标
        "threshold_tuning": True,
        "class_weights": True
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        config_map = {
            "random_forest": cls.RANDOM_FOREST_CONFIG,
            "xgboost": cls.XGBOOST_CONFIG,
            "lightgbm": cls.LIGHTGBM_CONFIG,
            "svm": cls.SVM_CONFIG,
            "knn": cls.KNN_CONFIG
        }
        return config_map.get(model_name, {})
    
    @classmethod
    def get_hyperparameter_grid(cls, model_name: str) -> Dict[str, List]:
        """获取模型超参数搜索网格"""
        grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            },
            "svm": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["rbf", "linear", "poly"]
            }
        }
        return grids.get(model_name, {}) 