# AI 医疗分诊系统前端

基于 RAG + GPT 的智能急诊分诊预测系统前端界面。

## 功能特点

- 🏥 **实时分诊预测**: 输入患者症状和生命体征，获得 AI 分诊建议
- 🤖 **RAG 增强**: 结合历史相似病例进行智能推理
- 📊 **可视化结果**: 清晰展示分诊等级、置信度和分析推理
- 📱 **响应式设计**: 支持桌面和移动设备
- ⚡ **快速响应**: 优化的 API 调用和结果展示

## 系统架构

```
前端界面 (HTML/CSS/JS)
    ↓
Flask后端 (Python)
    ↓
RAG系统 (GPT-4 + FAISS)
    ↓
分诊预测结果
```

## 安装和运行

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 Azure OpenAI

确保你的项目根目录中有 `openai_api/azure_config.py` 文件，包含以下配置：

```python
# Azure OpenAI配置
GPT4_API_KEY = "your-azure-openai-api-key"
GPT4_ENDPOINT = "https://your-resource.openai.azure.com/"
GPT4_DEPLOYMENT_NAME = "gpt-4"
```

### 3. 启动应用

```bash
# 进入前端目录
cd triage_frontend

# 启动Flask应用
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用说明

### 1. 输入患者信息

- **主诉症状**: 详细描述患者的主要症状
- **基本信息**: 年龄、性别
- **生命体征**: 心率、血压、体温、血氧、疼痛评分

### 2. 获取分诊结果

点击"开始分诊预测"按钮，系统将：

1. 分析患者症状和生命体征
2. 检索历史相似病例
3. 使用 GPT-4 进行综合评估
4. 返回分诊等级和建议

### 3. 理解分诊等级

- **1 级 (危重)**: 需要立即医疗干预
- **2 级 (严重)**: 需要及时医疗评估
- **3 级 (中等)**: 需要医疗评估但非紧急
- **4 级 (轻微)**: 可以等待常规医疗评估

## API 接口

### 预测分诊

**POST** `/api/predict`

请求体：

```json
{
  "chief_complaint": "胸痛和呼吸困难",
  "age": 65,
  "gender": "Male",
  "heart_rate": 110,
  "sbp": 140,
  "dbp": 90,
  "temperature": 98.6,
  "o2sat": 92,
  "pain": 8
}
```

响应：

```json
{
    "success": true,
    "prediction": {
        "triage_level": 1,
        "confidence": 0.85,
        "reasoning": "患者表现出危重症状...",
        "response_time": 0.63,
        "similar_cases": [...]
    }
}
```

### 健康检查

**GET** `/api/health`

### 获取示例

**GET** `/api/examples`

## 技术栈

- **前端**: HTML5, CSS3, JavaScript, Bootstrap 5
- **后端**: Flask, Flask-CORS
- **AI 模型**: GPT-4, Sentence Transformers, FAISS
- **数据处理**: Pandas, NumPy, Scikit-learn

## 文件结构

```
triage_frontend/
├── app.py                 # Flask后端应用
├── requirements.txt       # Python依赖
├── README.md             # 说明文档
└── templates/
    └── index.html        # 主页面模板
```

## 故障排除

### 1. RAG 系统加载失败

如果看到"使用模拟模式"的提示：

- 检查 Azure 配置是否正确
- 确认数据文件路径存在
- 验证 FAISS 索引文件是否完整

### 2. API 调用失败

- 检查网络连接
- 验证 Azure OpenAI 服务状态
- 确认 API 密钥和端点配置

### 3. 页面显示异常

- 清除浏览器缓存
- 检查 JavaScript 控制台错误
- 确认 Flask 服务正常运行

## 开发模式

### 启用调试模式

```bash
export FLASK_ENV=development
python app.py
```

### 自定义配置

可以修改 `app.py` 中的配置：

```python
# 修改端口
app.run(debug=True, host='0.0.0.0', port=8080)

# 修改样本数量
sample_size = 50  # 减少测试样本数量
```

## 部署建议

### 生产环境

1. 使用 Gunicorn 或 uWSGI 作为 WSGI 服务器
2. 配置 Nginx 作为反向代理
3. 启用 HTTPS
4. 设置适当的 CORS 策略
5. 配置日志记录

### 性能优化

1. 启用 FAISS 索引缓存
2. 使用 Redis 缓存预测结果
3. 实现请求限流
4. 优化数据库查询

## 许可证

本项目仅供医疗专业人员参考使用，不构成医疗建议。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进系统。
