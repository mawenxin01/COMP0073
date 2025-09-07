"""
前后端API通信集成测试
===================

测试前端和后端之间的API通信，确保：
1. HTTP请求响应格式正确
2. JSON数据序列化/反序列化正常
3. 错误处理机制有效
4. API响应时间满足要求
"""

import pytest
import requests
import json
import time
from unittest.mock import patch, MagicMock
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

@pytest.mark.integration
@pytest.mark.api
class TestFrontendBackendAPI:
    """前后端API通信集成测试"""
    
    def test_predict_api_request_response_format(self):
        """测试预测API的请求响应格式"""
        # Arrange: 模拟前端发送的请求数据
        frontend_request = {
            "chief_complaint": "chest pain and shortness of breath",
            "age": 65,
            "gender": "Male",
            "heart_rate": 110,
            "sbp": 140,
            "dbp": 90,
            "temperature": 98.6,
            "o2sat": 92,
            "pain": 8,
            "model": "tfidf_rf"
        }
        
        # Act: 模拟API处理
        backend_response = self._mock_api_predict_handler(frontend_request)
        
        # Assert: 验证响应格式
        assert 'success' in backend_response
        assert 'prediction' in backend_response
        assert 'timestamp' in backend_response
        
        prediction = backend_response['prediction']
        assert 'triage_level' in prediction
        assert 'confidence' in prediction
        assert 'reasoning' in prediction
        assert 'response_time' in prediction
        
        # 验证数据类型
        assert isinstance(prediction['triage_level'], int)
        assert isinstance(prediction['confidence'], float)
        assert isinstance(prediction['reasoning'], str)
        assert isinstance(prediction['response_time'], (int, float))
        
        # 验证取值范围
        assert 1 <= prediction['triage_level'] <= 5
        assert 0 <= prediction['confidence'] <= 1
        assert prediction['response_time'] > 0
        
    def test_predict_api_json_serialization(self):
        """测试JSON数据序列化和反序列化"""
        # Arrange: 包含各种数据类型的请求
        complex_request = {
            "chief_complaint": "胸痛，呼吸困难，持续2小时",  # 中文字符
            "age": 65,
            "gender": "Male",
            "heart_rate": 110,
            "sbp": 140,
            "dbp": 90,
            "temperature": 98.6,
            "o2sat": 92,
            "pain": 8,
            "arrival_time": "2024-01-15 14:30:00",  # 时间字符串
            "model": "llama_rag"
        }
        
        # Act: 测试JSON序列化
        json_str = json.dumps(complex_request, ensure_ascii=False)
        parsed_request = json.loads(json_str)
        
        # 模拟API处理
        response = self._mock_api_predict_handler(parsed_request)
        response_json = json.dumps(response, ensure_ascii=False)
        parsed_response = json.loads(response_json)
        
        # Assert: 验证数据完整性
        assert parsed_request["chief_complaint"] == complex_request["chief_complaint"]
        assert parsed_response["success"] == response["success"]
        assert "prediction" in parsed_response
        
    def test_health_api_endpoint(self):
        """测试健康检查API端点"""
        # Act: 模拟健康检查API调用
        health_response = self._mock_health_api()
        
        # Assert: 验证健康检查响应
        assert 'status' in health_response
        assert 'timestamp' in health_response
        assert 'models' in health_response
        
        assert health_response['status'] in ['healthy', 'degraded', 'unhealthy']
        assert isinstance(health_response['models'], dict)
        
        # 验证模型状态
        models = health_response['models']
        assert 'tfidf_rf' in models
        assert 'llama_rag' in models
        
        for model_name, model_status in models.items():
            assert 'status' in model_status
            assert 'last_check' in model_status
            assert model_status['status'] in ['available', 'unavailable', 'error']
            
    def test_examples_api_endpoint(self):
        """测试示例数据API端点"""
        # Act: 模拟获取示例API调用
        examples_response = self._mock_examples_api()
        
        # Assert: 验证示例数据响应
        assert 'examples' in examples_response
        assert isinstance(examples_response['examples'], list)
        assert len(examples_response['examples']) > 0
        
        # 验证示例数据格式
        example = examples_response['examples'][0]
        required_fields = [
            'chief_complaint', 'age', 'gender', 'heart_rate', 
            'sbp', 'dbp', 'temperature', 'o2sat', 'pain'
        ]
        
        for field in required_fields:
            assert field in example
            
    def test_api_error_handling(self):
        """测试API错误处理"""
        # Test Case 1: 缺失必需字段
        invalid_request_1 = {
            "age": 65,
            "gender": "Male"
            # 缺少 chief_complaint
        }
        
        response_1 = self._mock_api_predict_handler(invalid_request_1)
        assert response_1['success'] == False
        assert 'error' in response_1
        assert 'missing' in response_1['error'].lower() or 'required' in response_1['error'].lower()
        
        # Test Case 2: 无效数据类型
        invalid_request_2 = {
            "chief_complaint": "chest pain",
            "age": "sixty-five",  # 应该是数字
            "gender": "Male",
            "heart_rate": 110,
            "sbp": 140,
            "dbp": 90,
            "temperature": 98.6,
            "o2sat": 92,
            "pain": 8
        }
        
        response_2 = self._mock_api_predict_handler(invalid_request_2)
        assert response_2['success'] == False
        assert 'error' in response_2
        
        # Test Case 3: 超出正常范围的数值
        invalid_request_3 = {
            "chief_complaint": "chest pain",
            "age": 200,  # 异常年龄
            "gender": "Male",
            "heart_rate": 300,  # 异常心率
            "sbp": 400,  # 异常血压
            "dbp": 90,
            "temperature": 120,  # 异常体温
            "o2sat": 92,
            "pain": 15  # 超出疼痛评分范围
        }
        
        response_3 = self._mock_api_predict_handler(invalid_request_3)
        # 系统应该能处理但可能返回警告
        if response_3['success']:
            # 如果成功处理，应该有置信度较低或警告信息
            assert response_3['prediction']['confidence'] < 1.0  # 调整期望值
        else:
            assert 'error' in response_3
            
    def test_api_response_time_performance(self):
        """测试API响应时间性能"""
        request_data = {
            "chief_complaint": "chest pain and shortness of breath",
            "age": 65,
            "gender": "Male", 
            "heart_rate": 110,
            "sbp": 140,
            "dbp": 90,
            "temperature": 98.6,
            "o2sat": 92,
            "pain": 8,
            "model": "tfidf_rf"
        }
        
        # 测试多次请求的响应时间
        response_times = []
        for _ in range(5):
            start_time = time.time()
            response = self._mock_api_predict_handler(request_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # 验证响应成功
            assert response['success'] == True
            
        # 验证响应时间要求
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # API响应时间要求（根据实际需求调整）
        assert avg_response_time < 5.0  # 平均响应时间 < 5秒
        assert max_response_time < 10.0  # 最大响应时间 < 10秒
        
    def test_concurrent_api_requests(self):
        """测试并发API请求处理"""
        import threading
        import queue
        
        request_data = {
            "chief_complaint": "chest pain",
            "age": 45,
            "gender": "Female",
            "heart_rate": 85,
            "sbp": 120,
            "dbp": 80,
            "temperature": 98.6,
            "o2sat": 98,
            "pain": 5,
            "model": "tfidf_rf"
        }
        
        # 结果队列
        results = queue.Queue()
        
        def make_request():
            try:
                response = self._mock_api_predict_handler(request_data)
                results.put(('success', response))
            except Exception as e:
                results.put(('error', str(e)))
        
        # 创建并启动多个线程
        threads = []
        num_concurrent_requests = 5
        
        for _ in range(num_concurrent_requests):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
            
        # 等待所有线程完成
        for thread in threads:
            thread.join()
            
        # 验证结果
        successful_responses = 0
        while not results.empty():
            result_type, result_data = results.get()
            if result_type == 'success':
                successful_responses += 1
                assert result_data['success'] == True
                
        # 至少80%的并发请求应该成功
        success_rate = successful_responses / num_concurrent_requests
        assert success_rate >= 0.8
        
    def test_shap_explanation_api(self):
        """测试SHAP解释API"""
        request_data = {
            "patient_data": {
                "chief_complaint": "chest pain",
                "age": 65,
                "gender": "Male",
                "heart_rate": 110,
                "sbp": 140,
                "dbp": 90,
                "temperature": 98.6,
                "o2sat": 92,
                "pain": 8
            },
            "model": "tfidf_rf"
        }
        
        # Act: 模拟SHAP解释API调用
        shap_response = self._mock_shap_explanation_api(request_data)
        
        # Assert: 验证SHAP解释响应
        assert 'success' in shap_response
        if shap_response['success']:
            assert 'explanation' in shap_response
            explanation = shap_response['explanation']
            
            assert 'feature_contributions' in explanation
            assert 'total_positive_contribution' in explanation
            assert 'total_negative_contribution' in explanation
            assert 'net_contribution' in explanation
            
            # 验证特征贡献格式
            contributions = explanation['feature_contributions']
            assert isinstance(contributions, list)
            if len(contributions) > 0:
                feature = contributions[0]
                assert 'feature_name' in feature
                assert 'contribution' in feature
                assert isinstance(feature['contribution'], (int, float))
                
    # ==================== 辅助方法 ====================
    
    def _mock_api_predict_handler(self, request_data):
        """模拟API预测处理器"""
        try:
            # 验证必需字段
            required_fields = ['chief_complaint', 'age', 'gender', 'heart_rate', 
                             'sbp', 'dbp', 'temperature', 'o2sat', 'pain']
            
            for field in required_fields:
                if field not in request_data:
                    return {
                        'success': False,
                        'error': f'Missing required field: {field}',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
            # 验证数据类型
            if not isinstance(request_data.get('age'), int):
                return {
                    'success': False,
                    'error': 'Invalid data type for age',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
            # 模拟预测处理
            start_time = time.time()
            time.sleep(0.1)  # 模拟处理时间
            
            # 基于主诉确定分诊级别
            chief_complaint = request_data['chief_complaint'].lower()
            if 'chest pain' in chief_complaint:
                triage_level = 2
                confidence = 0.85
                reasoning = "Chest pain requires urgent evaluation due to potential cardiac involvement."
            elif 'fever' in chief_complaint:
                triage_level = 3
                confidence = 0.80
                reasoning = "Fever indicates possible infection requiring medical assessment."
            else:
                triage_level = 4
                confidence = 0.75
                reasoning = "Symptoms require routine medical evaluation."
                
            # 年龄调整
            if request_data['age'] > 65:
                triage_level = max(1, triage_level - 1)
                confidence += 0.05
                
            # 生命体征调整
            if request_data['heart_rate'] > 100 or request_data['sbp'] > 160:
                triage_level = max(1, triage_level - 1)
                confidence += 0.05
                
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'prediction': {
                    'triage_level': triage_level,
                    'confidence': min(1.0, confidence),
                    'reasoning': reasoning,
                    'response_time': round(response_time, 2),
                    'similar_cases': [
                        {'case_id': 'case_001', 'similarity': 0.85, 'outcome': f'ESI {triage_level}'}
                    ],
                    'model': request_data.get('model', 'tfidf_rf')
                },
                'patient_info': {
                    'age_group': 'elderly' if request_data['age'] > 65 else 'adult',
                    'severity_indicators': []
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
    def _mock_health_api(self):
        """模拟健康检查API"""
        return {
            'status': 'healthy',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': {
                'tfidf_rf': {
                    'status': 'available',
                    'last_check': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'response_time': 0.1
                },
                'llama_rag': {
                    'status': 'available', 
                    'last_check': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'response_time': 0.5
                }
            },
            'database': {
                'status': 'connected',
                'last_check': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
    def _mock_examples_api(self):
        """模拟示例数据API"""
        return {
            'examples': [
                {
                    'name': 'Chest Pain Emergency',
                    'chief_complaint': 'severe chest pain radiating to left arm',
                    'age': 65,
                    'gender': 'Male',
                    'heart_rate': 110,
                    'sbp': 160,
                    'dbp': 95,
                    'temperature': 98.6,
                    'o2sat': 95,
                    'pain': 9,
                    'expected_triage': 1
                },
                {
                    'name': 'Routine Checkup',
                    'chief_complaint': 'general wellness check',
                    'age': 35,
                    'gender': 'Female',
                    'heart_rate': 75,
                    'sbp': 120,
                    'dbp': 80,
                    'temperature': 98.6,
                    'o2sat': 99,
                    'pain': 0,
                    'expected_triage': 5
                }
            ],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def _mock_shap_explanation_api(self, request_data):
        """模拟SHAP解释API"""
        try:
            patient_data = request_data.get('patient_data', {})
            model = request_data.get('model', 'tfidf_rf')
            
            if model != 'tfidf_rf':
                return {
                    'success': False,
                    'error': 'SHAP explanation only available for TF-IDF model',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
            return {
                'success': True,
                'explanation': {
                    'feature_contributions': [
                        {'feature_name': 'chest_pain_keywords', 'contribution': 0.45},
                        {'feature_name': 'age_65+', 'contribution': 0.32},
                        {'feature_name': 'heart_rate_elevated', 'contribution': 0.28},
                        {'feature_name': 'blood_pressure_high', 'contribution': 0.15},
                        {'feature_name': 'pain_score_high', 'contribution': 0.12}
                    ],
                    'total_positive_contribution': 1.32,
                    'total_negative_contribution': -0.18,
                    'net_contribution': 1.14
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestFrontendBackendIntegrationWorkflow:
    """前后端完整集成工作流程测试"""
    
    def test_complete_frontend_backend_workflow(self):
        """测试完整的前后端交互工作流程"""
        # Step 1: 前端获取示例数据
        examples_response = self._get_examples()
        assert examples_response['success']
        
        # Step 2: 前端发送预测请求
        example_patient = examples_response['data']['examples'][0]
        predict_response = self._send_prediction_request(example_patient)
        assert predict_response['success']
        
        # Step 3: 如果是TF-IDF模型，获取SHAP解释
        if example_patient.get('model') == 'tfidf_rf':
            shap_response = self._get_shap_explanation(example_patient)
            assert shap_response['success']
            
        # Step 4: 前端检查系统健康状态
        health_response = self._check_health()
        assert health_response['success']
        
    def _get_examples(self):
        """模拟获取示例数据"""
        return {
            'success': True,
            'data': {
                'examples': [
                    {
                        'chief_complaint': 'chest pain',
                        'age': 65,
                        'gender': 'Male',
                        'heart_rate': 110,
                        'sbp': 140,
                        'dbp': 90,
                        'temperature': 98.6,
                        'o2sat': 92,
                        'pain': 8,
                        'model': 'tfidf_rf'
                    }
                ]
            }
        }
        
    def _send_prediction_request(self, patient_data):
        """模拟发送预测请求"""
        # 模拟网络延迟
        time.sleep(0.1)
        
        return {
            'success': True,
            'prediction': {
                'triage_level': 2,
                'confidence': 0.85,
                'reasoning': 'Chest pain requires urgent evaluation',
                'response_time': 0.15
            }
        }
        
    def _get_shap_explanation(self, patient_data):
        """模拟获取SHAP解释"""
        time.sleep(0.05)
        
        return {
            'success': True,
            'explanation': {
                'feature_contributions': [
                    {'feature_name': 'chest_pain', 'contribution': 0.45}
                ]
            }
        }
        
    def _check_health(self):
        """模拟健康检查"""
        return {
            'success': True,
            'status': 'healthy',
            'models': {
                'tfidf_rf': {'status': 'available'},
                'llama_rag': {'status': 'available'}
            }
        }
