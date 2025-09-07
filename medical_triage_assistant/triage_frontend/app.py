#!/usr/bin/env python3
"""
Medical Triage Frontend Application - Flask Backend
Provides real-time triage prediction API using RAG + GPT models
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入RAG系统
try:
    from methods.llama_rag.llama_rag_triage import LlamaRAGTriageSystem
    LlamaRAGTriage = LlamaRAGTriageSystem
except ImportError as e:
    print(f"⚠️ Unable to import Llama RAG system: {e}")
    print("⚠️ Will use simulation mode")
    LlamaRAGTriage = None

# 导入传统ML系统需要的库
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 导入必要的库
import joblib
import numpy as np
import sys
import os

# 添加methods目录到路径，以便导入CustomXGBClassifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methods', 'tfidf_ml'))

# 导入CustomXGBClassifier
try:
    from methods.tfidf_ml.nosvd import CustomXGBClassifier
    print("✅ CustomXGBClassifier imported successfully")
except ImportError as e:
    print(f"⚠️ Unable to import CustomXGBClassifier: {e}")
    CustomXGBClassifier = None

# 导入SHAP解释服务
try:
    from methods.llama_rag.shap_explainer_service import get_shap_explanation_for_prediction, initialize_shap_service
    print("✅ SHAP explanation service imported successfully")
except ImportError as e:
    print(f"⚠️ Unable to import SHAP explanation service: {e}")
    get_shap_explanation_for_prediction = None
    initialize_shap_service = None

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 辅助函数
def _format_time_display(arrival_time):
    """将完整时间格式转换为只显示时分的格式"""
    if not arrival_time:
        return None
    try:
        dt = pd.to_datetime(arrival_time)
        return dt.strftime('%H:%M')
    except:
        return arrival_time

# 全局变量
llama_rag_system = None
xgboost_model = None
tfidf_vectorizer = None  
svd_model = None  # SVD降维模型
feature_scaler = None
case_database = None
watsonx_client = None  # Watsonx客户端

class MockRAGSystem:
    """Mock RAG system for demonstration"""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        
    def predict_triage_level(self, patient_info, k=5):
        """Mock triage prediction"""
        
        # 简单的规则引擎
        chief_complaint = patient_info.get('chief_complaint', '').lower()
        heart_rate = patient_info.get('heart_rate', 80)
        sbp = patient_info.get('sbp', 120)
        dbp = patient_info.get('dbp', 80)
        temperature = patient_info.get('temperature', 98.6)
        o2sat = patient_info.get('o2sat', 98)
        pain = patient_info.get('pain', 5)
        age = patient_info.get('age', 50)
        
        # 危重症状关键词
        critical_keywords = ['chest pain', 'shortness of breath', 'unconscious', 'cardiac arrest', 
                           'stroke', 'seizure', 'trauma', 'bleeding', 'shock']
        
        # 严重症状关键词
        severe_keywords = ['fever', 'vomiting', 'diarrhea', 'headache', 'abdominal pain', 
                          'dizziness', 'weakness', 'nausea']
        
        # 生命体征评估
        vital_score = 0
        if heart_rate > 100 or heart_rate < 60:
            vital_score += 1
        if sbp < 90 or sbp > 180:
            vital_score += 1
        if temperature > 100.4:
            vital_score += 1
        if o2sat < 95:
            vital_score += 1
        if pain > 7:
            vital_score += 1
        
        # 年龄风险
        age_risk = 0
        if age > 65:
            age_risk = 1
        
        # 分诊等级判断
        if any(keyword in chief_complaint for keyword in critical_keywords) or vital_score >= 3:
            triage_level = 1  # 危重
            confidence = 0.85
            reasoning = "患者表现出危重症状或生命体征异常，需要立即医疗干预"
        elif any(keyword in chief_complaint for keyword in severe_keywords) or vital_score >= 2 or age_risk:
            triage_level = 2  # 严重
            confidence = 0.75
            reasoning = "患者症状严重或存在风险因素，需要及时医疗评估"
        elif vital_score >= 1 or pain > 5:
            triage_level = 3  # 中等
            confidence = 0.70
            reasoning = "患者症状中等，需要医疗评估但非紧急"
        else:
            triage_level = 4  # 轻微
            confidence = 0.80
            reasoning = "患者症状轻微，可以等待常规医疗评估"
        
        # 返回与真实LlamaRAGTriageSystem相同的格式：(triage_level, reasoning, similar_cases, similarities)
        similar_cases = [
            {'chief_complaint': 'chest pain', 'triage_level': 1, 'outcome': 'MI confirmed'},
            {'chief_complaint': 'fever and cough', 'triage_level': 2, 'outcome': 'Pneumonia'},
            {'chief_complaint': 'headache', 'triage_level': 3, 'outcome': 'Tension headache'}
        ]
        similarities = [0.85, 0.75, 0.65]  # 模拟相似度分数
        
        return triage_level, reasoning, similar_cases, similarities

def initialize_systems():
    """初始化所有预测系统"""
    global llama_rag_system, xgboost_model, feature_scaler, case_database
    
    # 1. 初始化Llama RAG系统 - 直接加载预构建的索引
    try:
        if LlamaRAGTriage:
            print("🚀 Initializing Llama RAG system...")
            llama_rag_system = LlamaRAGTriage()
            
            # 直接加载预构建的索引，不涉及原始数据
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "..")
            
            # 尝试加载已训练好的索引（按优先级）
            possible_index_paths = [
                os.path.join(project_root, "methods/llama_rag/llama_rag_train_only_index_stable"),  # 稳定版本索引
                # 如果索引不存在，则提示用户先运行训练
            ]
            
            loaded = False
            for index_path in possible_index_paths:
                print(f"🔍 Attempting to load index: {index_path}")
                # Check if FAISS index files exist (need both .index and .pkl files)
                faiss_index_path = f"{index_path}_faiss.index"
                metadata_path = f"{index_path}_metadata.pkl"
                
                if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                    print(f"✅ Found pre-built index: {index_path}")
                    try:
                        # 初始化真实的Llama RAG系统
                        from methods.llama_rag.llama_rag_triage import LlamaRAGTriageSystem
                        llama_rag_system = LlamaRAGTriageSystem()
                        # 加载预构建索引
                        llama_rag_system.case_retriever.vector_store.load_index(index_path)
                        
                        # 标记系统为已初始化
                        llama_rag_system.case_retriever.initialized = True
                        llama_rag_system.is_initialized = True
                        
                        print(f"✅ Successfully loaded pre-built index: {index_path}")
                        loaded = True
                        
                        # 检查索引数据量，警告潜在的数据泄漏
                        if hasattr(llama_rag_system.case_retriever, 'vector_store') and hasattr(llama_rag_system.case_retriever.vector_store, 'case_database'):
                            indexed_count = len(llama_rag_system.case_retriever.vector_store.case_database)
                            expected_train_size = int(189780 * 0.8)  # 约151,824
                            print(f"📊 Loaded index contains {indexed_count} cases")
                            
                            if indexed_count > (expected_train_size + 5000):
                                print(f"⚠️ Data leakage warning: Index data size ({indexed_count}) exceeds expected training size ({expected_train_size})")
                                print(f"   May contain test data, evaluation results may be inflated")
                            elif indexed_count < (expected_train_size - 5000):
                                print(f"⚠️ Data size too small: Index data size ({indexed_count}) less than expected training size")
                            else:
                                print(f"✅ Index data size reasonable: {indexed_count} cases (expected ~{expected_train_size})")
                        break
                        
                    except Exception as e:
                        print(f"❌ Failed to load index: {e}")
                        continue
                else:
                    print(f"⚠️ Index does not exist or failed to load: {index_path}")
            
            if not loaded:
                print("❌ No available pre-built index found")
                print("💡 Please run the following commands to build index:")
                print("   cd medical_triage_assistant/methods/llama_rag")
                print("   python llama_rag_triage.py")
                print("⚠️ Temporarily using simulation mode")
                llama_rag_system = MockRAGSystem()
            else:
                print("✅ Llama RAG system initialization complete (using pre-built index)")
                
        else:
            print("⚠️ 使用模拟Llama RAG系统")
            llama_rag_system = MockRAGSystem()
            
    except Exception as e:
        print(f"❌ Llama RAG系统初始化失败: {e}")
        print("⚠️ 使用模拟模式")
        llama_rag_system = MockRAGSystem()
    
    # 2. 初始化TFIDF + Random Forest模型
    global xgboost_model, tfidf_vectorizer, svd_model, watsonx_client
    try:
        print("🎯 初始化TFIDF + Random Forest模型...")
        
        # 确保project_root已定义
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..")
        
        # 加载TFIDF + Random Forest模型
        xgboost_path = os.path.join(project_root, "models", "TFIDF-rf.pkl")
        vectorizer_path = os.path.join(project_root, "models", "TFIDF-rf-vectorizer.pkl")
        svd_path = os.path.join(project_root, "models", "TFIDF-rf-svd.pkl")
        
        if os.path.exists(xgboost_path):
            xgboost_model = joblib.load(xgboost_path)
            print("✅ TFIDF + Random Forest模型加载成功")
            
            # 加载TF-IDF向量化器
            if os.path.exists(vectorizer_path):
                tfidf_vectorizer = joblib.load(vectorizer_path)
                print("✅ TF-IDF向量化器加载成功")
            else:
                print(f"❌ TF-IDF向量化器文件不存在: {vectorizer_path}")
                tfidf_vectorizer = None
            
            # 加载SVD模型
            if os.path.exists(svd_path):
                svd_model = joblib.load(svd_path)
                print("✅ SVD模型加载成功")
            else:
                print(f"❌ SVD模型文件不存在: {svd_path}")
                svd_model = None
                
        else:
            print(f"❌ TFIDF + Random Forest模型文件不存在: {xgboost_path}")
            xgboost_model = None
            tfidf_vectorizer = None
            svd_model = None
        
        # 初始化Watsonx客户端
        try:
            from medical_triage_assistant.config.azure_simple import IBM_API_KEY, IBM_ENDPOINT, IBM_PROJECT_ID, IBM_MODEL_NAME
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Meta
            from ibm_watsonx_ai import Credentials
            
            credentials = Credentials(api_key=IBM_API_KEY, url=IBM_ENDPOINT)
            watsonx_client = ModelInference(
                model_id=IBM_MODEL_NAME,
                credentials=credentials,
                project_id=IBM_PROJECT_ID,
                params={Meta.MAX_NEW_TOKENS: 100, Meta.TEMPERATURE: 0.1}
            )
            print("✅ Watsonx客户端初始化成功")
        except Exception as e:
            print(f"⚠️ Watsonx客户端初始化失败: {e}")
            watsonx_client = None
            
    except Exception as e:
        print(f"❌ TFIDF + Random Forest模型初始化失败: {e}")
        print(f"错误详情: {str(e)}")
        xgboost_model = None
        watsonx_client = None
    
    # 3. 初始化SHAP解释服务
    try:
        print("🔍 初始化SHAP解释服务...")
        if initialize_shap_service:
            if initialize_shap_service():
                print("✅ SHAP解释服务初始化成功")
            else:
                print("⚠️ SHAP解释服务初始化失败")
        else:
            print("⚠️ SHAP解释服务不可用")
    except Exception as e:
        print(f"❌ SHAP解释服务初始化失败: {e}")
        


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_triage():
    """分诊预测API"""
    
    try:
        # 确保系统已初始化
        global llama_rag_system, xgboost_model, watsonx_client
        if llama_rag_system is None or xgboost_model is None:
            initialize_systems()
            
        # 获取患者信息
        data = request.get_json()
        
        # 验证必需字段
        required_fields = ['chief_complaint', 'age', 'heart_rate', 'sbp', 'dbp', 'temperature', 'o2sat', 'pain']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # 记录请求时间
        start_time = time.time()
        
        # 获取选择的模型
        selected_model = data.get('model', 'gpt_rag')
        
        # 构建患者信息
        patient_info = {
            'chiefcomplaint': data['chief_complaint'],
            'chief_complaint': data['chief_complaint'],  # 为传统ML保留
            'age_at_visit': data['age'],
            'age': data['age'],  # 为传统ML保留
            'heartrate': data['heart_rate'],
            'heart_rate': data['heart_rate'],  # 为传统ML保留
            'sbp': data['sbp'],
            'dbp': data['dbp'],
            'temperature': data['temperature'],
            'o2sat': data['o2sat'],
            'pain': data['pain'],
            'gender': data.get('gender', 'Unknown'),
            'arrival_transport': data.get('arrival_transport', 'WALK IN'),
            'arrival_time': pd.to_datetime(data.get('arrival_time', None)).strftime('%H:%M') if data.get('arrival_time', None) else None  # 只显示时分
        }
        
        # 根据选择的模型进行预测
        print(f"🔍 Selected model: {selected_model}")
        
        if selected_model == 'tfidf_rf':
            # 使用TFIDF + Random Forest模型进行预测
            print("📊 Using TFIDF + Random Forest model for prediction")
            
            try:
                if xgboost_model is not None:
                    if tfidf_vectorizer is not None and svd_model is not None:
                        print("✅ Using TFIDF + Random Forest model (complete 33 features)")
                    
                    # 1. 提取关键词
                    chief_complaint = patient_info.get('chief_complaint', '')
                    keywords = chief_complaint  # 默认使用原始文本
                    
                    if watsonx_client is not None:
                        try:
                            prompt = f"""
                            作为医疗专家，请从以下主诉中提取最重要的医学关键词，用逗号分隔：

                            主诉: {chief_complaint}

                            关键词:"""
                            
                            response = watsonx_client.generate_text(prompt=prompt)
                            keywords = response.strip()
                            print(f"🔍 Keywords extracted by Watsonx: {keywords}")
                        except Exception as e:
                            print(f"⚠️ Watsonx keyword extraction failed: {e}")
                    
                    # 2. 处理文本特征（使用真正的TF-IDF + SVD）
                    print(f"🔍 Processing keywords: {keywords}")
                    
                    # TF-IDF变换
                    keywords_tfidf = tfidf_vectorizer.transform([keywords])
                    print(f"✅ TF-IDF feature shape: {keywords_tfidf.shape}")
                    
                    # SVD降维
                    keywords_svd = svd_model.transform(keywords_tfidf)
                    print(f"✅ SVD feature shape: {keywords_svd.shape}")
                    
                    # 3. 处理时间特征
                    arrival_time = patient_info.get('arrival_time', '2024-01-15 12:00:00')
                    try:
                        dt = pd.to_datetime(arrival_time)
                        hour = dt.hour
                        if 0 <= hour < 6:
                            time_period = 0  # 凌晨
                        elif 6 <= hour < 12:
                            time_period = 1  # 上午
                        elif 12 <= hour < 18:
                            time_period = 2  # 下午
                        else:
                            time_period = 3  # 晚上
                    except:
                        time_period = 2  # 默认下午
                    
                    # 4. 构建数值特征（8个基础数值特征）
                    numerical_features = [
                        float(patient_info.get('age', 50)),
                        float(patient_info.get('pain', 0)),
                        float(patient_info.get('temperature', 98.6)),
                        float(patient_info.get('heart_rate', 80)),
                        float(patient_info.get('sbp', 120)),
                        float(patient_info.get('dbp', 80)),
                        float(patient_info.get('o2sat', 98)),
                        float(time_period)
                    ]
                    
                    # 5. 处理分类特征（5个one-hot编码特征）
                    gender = patient_info.get('gender', 'Unknown').upper()
                    gender_m = 1 if gender == 'M' else 0
                    
                    transport = patient_info.get('arrival_transport', 'WALK IN').upper()
                    transport_helicopter = 1 if transport == 'HELICOPTER' else 0
                    transport_other = 1 if transport == 'OTHER' else 0
                    transport_unknown = 1 if transport == 'UNKNOWN' else 0
                    transport_walk_in = 1 if transport == 'WALK IN' else 0
                    
                    categorical_features = [
                        gender_m,
                        transport_helicopter,
                        transport_other,
                        transport_unknown,
                        transport_walk_in
                    ]
                    
                    # 6. 合并所有特征（33个特征：8数值 + 5分类 + 20 SVD）
                    all_features = np.concatenate([
                        numerical_features,        # 8个数值特征
                        categorical_features,      # 5个分类特征  
                        keywords_svd.flatten()     # 20个SVD特征
                    ])
                    
                    print(f"✅ Final feature vector shape: {all_features.shape} (should be 33)")
                    print(f"   Numerical features: {len(numerical_features)}")
                    print(f"   Categorical features: {len(categorical_features)}")
                    print(f"   SVD features: {keywords_svd.shape[1]}")
                    
                    # 7. 使用TFIDF + Random Forest模型预测
                    prediction = xgboost_model.predict([all_features])[0]
                    prediction_proba = xgboost_model.predict_proba([all_features])[0]
                    
                    # 转换回原始分诊等级 (0-4 -> 1-5)
                    triage_level = int(prediction) + 1
                    confidence = float(np.max(prediction_proba))
                    
                    # 分诊等级含义
                    level_meanings = {
                        1: "Critical (立即)",
                        2: "Severe (急诊)",
                        3: "Moderate (紧急)",
                        4: "Mild (较急)",
                        5: "Minor (非急)"
                    }
                    
                    # 获取SHAP解释
                    shap_explanation = None
                    if get_shap_explanation_for_prediction:
                        try:
                            shap_explanation = get_shap_explanation_for_prediction(patient_info)
                            print("✅ SHAP解释生成成功")
                        except Exception as e:
                            print(f"⚠️ SHAP解释生成失败: {e}")
                    
                    # TFIDF模型不提供reasoning，只返回分诊级别
                    reasoning = ""
                    
                    # 不提供相似案例
                    similar_cases = []
                        
                else:
                    print("⚠️ TFIDF + Random Forest模型不可用，使用模拟模式")
                    triage_level = 3
                    confidence = 0.5
                    reasoning = ""
                    similar_cases = []
                
            except Exception as e:
                print(f"❌ TFIDF + Random Forest预测失败: {e}")
                triage_level = 3
                confidence = 0.0
                reasoning = ""
                similar_cases = []
                
            model_used = 'tfidf_rf'
            
        else:
            # 使用Llama RAG系统（默认）
            print(f"🤖 使用Llama RAG系统进行预测")
            if llama_rag_system is not None and hasattr(llama_rag_system, 'predict_triage_level'):
                # 真实Llama RAG系统
                print("✅ 使用真实Llama RAG系统（Vector-Only检索 - 最快模式）")
                try:
                    # 转换为字典格式
                    case_dict = {
                        'chief_complaint': patient_info.get('chief_complaint', ''),
                        'age_at_visit': patient_info.get('age', 50),
                        'gender': patient_info.get('gender', 'Unknown'),
                        'heartrate': patient_info.get('heart_rate', 80),
                        'sbp': patient_info.get('sbp', 120),
                        'dbp': patient_info.get('dbp', 80),
                        'temperature': patient_info.get('temperature', 98.6),
                        'o2sat': patient_info.get('o2sat', 98),
                        'pain': patient_info.get('pain', 5),
                        'arrival_transport': patient_info.get('arrival_method', 'Unknown'),
                        'time_period': patient_info.get('time_period', 2)
                    }
                    
                    # 调用LLaMA RAG系统（使用Vector-Only检索 - 最快模式）
                    # 性能配置：Vector-Only模式，最快的检索速度
                    k_retrieve = 5  # Vector-Only检索 (~30-50ms，最快)
                    
                    start_retrieval = time.time()
                    triage_level, reasoning, similar_cases_raw, similarities = llama_rag_system.predict_triage_level(case_dict, k=k_retrieve)
                    retrieval_time = time.time() - start_retrieval
                    print(f"⏱️ RAG检索+生成时间: {retrieval_time:.3f}s")
                    
                    confidence = 0.8  # 默认置信度
                    if similarities:
                        confidence = min(0.95, max(0.6, np.mean(similarities)))
                    
                    # 转换相似案例格式
                    similar_cases = []
                    if similar_cases_raw:
                        for i, case in enumerate(similar_cases_raw[:3]):  # 取前3个
                            # 构建完整的案例信息
                            age = case.get('age_at_visit', case.get('age', 'N/A'))
                            gender = case.get('gender', 'N/A')
                            heartrate = case.get('heartrate', case.get('heart_rate', 'N/A'))
                            sbp = case.get('sbp', 'N/A')
                            dbp = case.get('dbp', 'N/A')
                            temperature = case.get('temperature', 'N/A')
                            o2sat = case.get('o2sat', 'N/A')
                            pain = case.get('pain', 'N/A')
                            
                            # 构建详细信息字符串
                            vitals = f"Age: {age}, Gender: {gender}, HR: {heartrate}, BP: {sbp}/{dbp}, Temp: {temperature}°F, O2Sat: {o2sat}%, Pain: {pain}/10"
                            similarity_info = f"{similarities[i]:.3f}" if i < len(similarities) else 'N/A'
                            
                            # Debug: 打印案例信息
                            print(f"🔍 Similar case {i+1}: {case.keys()}")
                            print(f"   Vitals: {vitals}")
                            
                            similar_cases.append({
                                'chief_complaint': case.get('chiefcomplaint', 'N/A'),
                                'triage_level': f"Level {case.get('acuity', 'Unknown')}",
                                'vitals': vitals,
                                'outcome': similarity_info
                            })
                    
                except Exception as e:
                    print(f"❌ Llama RAG系统调用失败: {e}")
                    # 回退到模拟模式
                    llama_rag_system = MockRAGSystem()
                    prediction_result = llama_rag_system.predict_triage_level(patient_info)
                    triage_level = prediction_result['triage_level']
                    confidence = prediction_result['confidence']
                    reasoning = prediction_result['reasoning']
                    similar_cases = prediction_result['similar_cases']
                
            else:
                # 模拟RAG系统
                print("⚠️ 使用模拟Llama RAG系统")
                if llama_rag_system is None:
                    llama_rag_system = MockRAGSystem()
                prediction_result = llama_rag_system.predict_triage_level(patient_info)
                triage_level = prediction_result['triage_level']
                confidence = prediction_result['confidence']
                reasoning = prediction_result['reasoning']
                similar_cases = prediction_result['similar_cases']
            
            model_used = 'llama_rag'
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 构建响应
        response = {
            'success': True,
            'prediction': {
                'triage_level': triage_level,
                'confidence': confidence,
                'reasoning': reasoning,
                'response_time': round(response_time, 2),
                'similar_cases': similar_cases,
                'model': model_used
            },
            'patient_info': patient_info,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 添加SHAP解释数据到响应
        if selected_model == 'tfidf_rf' and shap_explanation and shap_explanation.get('success'):
            response['shap_explanation'] = {
                'feature_contributions': shap_explanation['feature_contributions'][:10],
                'total_positive_contribution': shap_explanation['total_positive_contribution'],
                'total_negative_contribution': shap_explanation['total_negative_contribution'],
                'net_contribution': shap_explanation['net_contribution']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/shap_explanation', methods=['POST'])
def get_shap_explanation():
    """获取SHAP解释API"""
    
    try:
        # 确保SHAP服务已初始化
        if not get_shap_explanation_for_prediction:
            return jsonify({
                'success': False,
                'error': 'SHAP解释服务不可用'
            }), 400
        
        # 获取患者信息
        data = request.get_json()
        
        # 验证必需字段
        required_fields = ['chief_complaint', 'age', 'heart_rate', 'sbp', 'dbp', 'temperature', 'o2sat', 'pain']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # 构建患者信息
        patient_info = {
            'chief_complaint': data['chief_complaint'],
            'age_at_visit': data['age'],
            'heartrate': data['heart_rate'],
            'sbp': data['sbp'],
            'dbp': data['dbp'],
            'temperature': data['temperature'],
            'o2sat': data['o2sat'],
            'pain': data['pain'],
            'gender': data.get('gender', 'Unknown'),
            'arrival_transport': data.get('arrival_transport', 'WALK IN'),
            'arrival_time': data.get('arrival_time', '2024-01-15 12:00:00')
        }
        
        # 获取SHAP解释
        shap_explanation = get_shap_explanation_for_prediction(patient_info)
        
        return jsonify({
            'success': True,
            'shap_explanation': shap_explanation,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查API"""
    return jsonify({
        'status': 'healthy',
        'llama_rag_system_loaded': llama_rag_system is not None,
        'xgboost_model_loaded': xgboost_model is not None,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """获取示例病例"""
    examples = [
        {
            'chief_complaint': 'Chest pain and shortness of breath',
            'age': 65,
            'heart_rate': 110,
            'sbp': 140,
            'dbp': 90,
            'temperature': 98.6,
            'o2sat': 92,
            'pain': 8,
            'gender': 'Male',
            'arrival_transport': 'WALK IN',
            'arrival_time': '13:32'
        },
        {
            'chief_complaint': 'Fever and cough for 3 days',
            'age': 45,
            'heart_rate': 95,
            'sbp': 130,
            'dbp': 85,
            'temperature': 101.2,
            'o2sat': 96,
            'pain': 4,
            'gender': 'Female',
            'arrival_transport': 'WALK IN',
            'arrival_time': '21:34'
        },
        {
            'chief_complaint': 'Headache and dizziness',
            'age': 30,
            'heart_rate': 75,
            'sbp': 120,
            'dbp': 80,
            'temperature': 98.4,
            'o2sat': 98,
            'pain': 6,
            'gender': 'Female',
            'arrival_transport': 'AMBULANCE',
            'arrival_time': '08:40'
        },
        {
            'chief_complaint': 'Minor finger laceration',
            'age': 25,
            'heart_rate': 70,
            'sbp': 115,
            'dbp': 75,
            'temperature': 98.2,
            'o2sat': 99,
            'pain': 2,
            'gender': 'Male',
            'arrival_transport': 'WALK IN',
            'arrival_time': '15:54'
        }
    ]
    
    return jsonify({'examples': examples})

if __name__ == '__main__':
    # 初始化所有系统
    initialize_systems()
    
    # 启动Flask应用
    print("🌐 Starting medical triage frontend application...")
    print("📱 Access URL: http://localhost:5003")
    app.run(debug=True, host='0.0.0.0', port=5003) 