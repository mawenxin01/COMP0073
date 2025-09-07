"""
Model-specific configurations
模型特定配置
"""

# TF-IDF 配置
TFIDF_CONFIG = {
    "vectorizer": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "stop_words": None
    },
    "classifier": {
        "type": "RandomForest",  # 或 "SVM", "LogisticRegression"
        "params": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1
        }
    }
}

# ChatGPT 配置
CHATGPT_CONFIG = {
    "models": {
        "gpt-3.5-turbo": {
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "gpt-4": {
            "max_tokens": 2000,
            "temperature": 0.5,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    },
    "prompts": {
        "triage": "你是一个专业的医疗分诊助手。请根据患者的症状描述，判断紧急程度并给出分诊建议。",
        "explanation": "请解释为什么给出这样的分诊建议。",
        "follow_up": "请提供后续需要注意的事项。"
    }
}

# RedFlag 配置
REDFLAG_CONFIG = {
    "critical_symptoms": [
        "胸痛", "呼吸困难", "意识丧失", "大出血", "严重创伤"
    ],
    "urgent_symptoms": [
        "高烧", "剧烈腹痛", "持续呕吐", "严重头痛", "肢体麻木"
    ],
    "thresholds": {
        "critical": 0.9,
        "urgent": 0.7,
        "normal": 0.3
    }
}

# GPT + RAG 配置
RAG_CONFIG = {
    "embedding": {
        "model": "text-embedding-ada-002",
        "dimension": 1536,
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "vector_store": {
        "type": "faiss",  # faiss, chroma, pinecone
        "index_type": "Flat",  # 对于faiss
        "similarity_metric": "cosine"
    },
    "retrieval": {
        "top_k": 5,
        "score_threshold": 0.7
    },
    "generation": {
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.9
    }
}

# WatsonX 配置
WATSONX_CONFIG = {
    "models": {
        "llama-2-70b": {
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        "flan-t5-xxl": {
            "max_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9
        }
    },
    "endpoints": {
        "text_generation": "/v1/text/generation",
        "text_embedding": "/v1/text/embeddings"
    }
} 