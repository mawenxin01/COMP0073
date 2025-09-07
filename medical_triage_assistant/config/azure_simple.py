#!/usr/bin/env python3
"""
Azure OpenAI 简单配置文件
基于用户提供的Azure OpenAI信息
"""

# Azure OpenAI GPT-4 配置（使用环境变量）
import os

GPT4_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-api-key-here")
GPT4_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.services.ai.azure.com/")
GPT4_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

# IBM Watson配置（使用环境变量）
IBM_API_KEY = os.getenv("IBM_API_KEY", "your-ibm-api-key-here")
IBM_ENDPOINT = os.getenv("IBM_ENDPOINT", "https://eu-gb.ml.cloud.ibm.com")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "your-project-id-here")
IBM_MODEL_NAME = os.getenv("IBM_MODEL_NAME", "meta-llama/llama-3-3-70b-instruct")

# 备注：
# - 这些是基于您提供的Azure OpenAI配置信息
# - 确保API密钥的安全性，不要将其提交到版本控制系统
# - 建议使用环境变量来管理敏感信息
