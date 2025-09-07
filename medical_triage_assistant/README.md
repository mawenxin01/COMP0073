# Medical Triage Assistant (医疗分诊助手)

一个基于多种 AI 方法的医疗分诊助手系统，支持 TF-IDF+ML、ChatGPT、RedFlag 规则和 GPT+RAG 等多种方法。

## 项目结构

```
medical_triage_assistant/
├── config/                     # 配置文件目录
│   ├── __init__.py
│   ├── config.py              # 主配置类
│   ├── settings.py            # 环境设置
│   ├── model_configs.py       # 模型特定配置
│   └── env_example.txt        # 环境变量示例
├── data_processing/           # 数据处理模块
│   ├── raw_data/             # 原始数据
│   ├── processed_data/        # 处理后的数据
│   ├── preprocessing/         # 数据预处理
│   └── validation/           # 数据验证
├── keyword_extraction/        # 关键词提取模块
│   ├── extractors/           # 提取器
│   ├── models/               # 关键词模型
│   └── evaluation/           # 评估
├── methods/                   # 分诊方法模块
│   ├── tfidf_ml/            # TF-IDF + 机器学习
│   ├── chatgpt/             # ChatGPT方法
│   ├── redflag/             # RedFlag规则方法
│   └── gpt_rag/             # GPT + RAG方法
├── utils/                     # 工具函数
├── models/                    # 训练好的模型
├── logs/                      # 日志文件
├── tests/                     # 测试文件
├── examples/                  # 示例代码
│   └── watsonx_demo.py       # WatsonX使用示例
├── docs/                      # 文档
├── main.py                    # 主程序入口
└── requirements.txt           # 依赖包
```

## 配置使用方法

### 1. 基础配置

```python
from config.config import Config

# 创建配置实例
config = Config()

# 获取特定类型的配置
data_config = config.get_config("data")
chatgpt_config = config.get_config("chatgpt")
```

### 2. 环境设置

```python
from config.settings import API_HOST, API_PORT, DEBUG

print(f"API地址: {API_HOST}:{API_PORT}")
print(f"调试模式: {DEBUG}")
```

### 3. 模型配置

```python
from config.model_configs import WATSONX_CONFIG, TFIDF_CONFIG

# 获取WatsonX模型参数
watsonx_params = WATSONX_CONFIG["models"]["llama-2-70b"]

# 获取TF-IDF参数
tfidf_params = TFIDF_CONFIG["vectorizer"]
```

### 4. WatsonX 配置示例

```python
# 在watsonx_demo.py中
class WatsonXDemo:
    def __init__(self):
        # 使用配置类
        self.config = Config()
        self.watsonx_config = self.config.get_config("watsonx")

        # 或者直接使用模型配置
        self.model_configs = WATSONX_CONFIG
```

## 环境变量配置

1. 复制 `config/env_example.txt` 为 `.env`
2. 填入实际的 API 密钥和配置值

```bash
cp config/env_example.txt .env
# 编辑 .env 文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python examples/watsonx_demo.py
```

## 主要特性

- **多种分诊方法**: 支持传统 ML、大语言模型和规则引擎
- **灵活配置**: 统一的配置管理系统
- **模块化设计**: 易于扩展和维护
- **中文支持**: 专门针对中文医疗文本优化
- **配置驱动**: 所有参数都可以通过配置文件调整

## 配置优势

使用配置文件的好处：

1. **集中管理**: 所有配置在一个地方管理
2. **环境隔离**: 开发、测试、生产环境使用不同配置
3. **参数调优**: 无需修改代码即可调整模型参数
4. **团队协作**: 配置模板便于团队共享
5. **部署灵活**: 支持容器化部署和云服务配置
