"""
医疗分诊系统集成测试
==================

这个例子展示如何为医疗分诊系统编写实际的集成测试，
测试从数据输入到预测输出的完整流程。
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
import sys
import os
import json
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.integration
@pytest.mark.model
class TestTriageSystemIntegration:
    """医疗分诊系统完整集成测试"""
    
    def test_complete_triage_workflow(self, realistic_patient_data):
        """测试完整的分诊工作流程"""
        # Arrange: 准备真实的患者数据
        patient = realistic_patient_data[0]
        
        # Act: 执行完整的分诊流程
        result = self._run_triage_system(patient)
        
        # Assert: 验证分诊结果
        assert 'triage_level' in result
        assert 'confidence' in result
        assert 'processing_time' in result
        assert 'model_used' in result
        
        # 验证分诊级别有效性
        assert 1 <= result['triage_level'] <= 5
        assert 0 <= result['confidence'] <= 1
        assert result['processing_time'] > 0
        
        # 验证高优先级情况
        if 'chest pain' in patient['chiefcomplaint'].lower():
            assert result['triage_level'] <= 3  # 胸痛应该是高优先级
            
    def test_batch_triage_processing(self, realistic_patient_data):
        """测试批量分诊处理"""
        # Arrange: 准备多个患者数据
        patients = realistic_patient_data[:5]
        
        # Act: 批量处理
        results = []
        for patient in patients:
            result = self._run_triage_system(patient)
            results.append(result)
            
        # Assert: 验证批量处理结果
        assert len(results) == len(patients)
        
        # 验证每个结果的格式一致性
        for result in results:
            assert 'triage_level' in result
            assert 'confidence' in result
            assert isinstance(result['triage_level'], int)
            assert isinstance(result['confidence'], float)
            
        # 验证不同严重程度的分诊结果
        triage_levels = [r['triage_level'] for r in results]
        assert min(triage_levels) >= 1
        assert max(triage_levels) <= 5
        
    def _run_triage_system(self, patient_data):
        """运行完整的分诊系统"""
        import time
        start_time = time.time()
        
        # 1. 数据预处理
        processed_data = self._preprocess_patient_data(patient_data)
        
        # 2. 特征提取
        features = self._extract_features(processed_data)
        
        # 3. 模型预测
        prediction = self._predict_triage_level(features, processed_data)
        
        # 4. 后处理
        result = self._post_process_prediction(prediction, processed_data)
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        return result
        
    def _preprocess_patient_data(self, data):
        """预处理患者数据"""
        processed = data.copy()
        
        # 文本预处理
        if 'chiefcomplaint' in processed:
            processed['chiefcomplaint'] = processed['chiefcomplaint'].lower().strip()
            
        # 数值验证和处理
        vital_signs = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
        for vital in vital_signs:
            if vital in processed and processed[vital] is not None:
                # 简单的异常值检查
                if vital == 'temperature' and (processed[vital] < 90 or processed[vital] > 110):
                    processed[f'{vital}_abnormal'] = True
                elif vital == 'heartrate' and (processed[vital] < 40 or processed[vital] > 200):
                    processed[f'{vital}_abnormal'] = True
                    
        return processed
        
    def _extract_features(self, data):
        """提取特征"""
        features = {}
        
        # 文本特征（简化的TF-IDF模拟）
        if 'chiefcomplaint' in data:
            text = data['chiefcomplaint']
            high_priority_keywords = ['chest pain', 'difficulty breathing', 'severe', 'emergency']
            features['high_priority_text'] = any(keyword in text for keyword in high_priority_keywords)
            
        # 数值特征
        vital_features = []
        for vital in ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']:
            if vital in data and data[vital] is not None:
                vital_features.append(data[vital])
        features['vital_signs'] = vital_features
        
        # 年龄特征
        if 'age' in data:
            features['age_group'] = 'elderly' if data['age'] > 65 else 'adult' if data['age'] > 18 else 'pediatric'
            
        return features
        
    def _predict_triage_level(self, features, original_data):
        """预测分诊级别"""
        # 基于规则的简化预测逻辑（实际中会使用训练好的模型）
        
        # 高优先级条件
        if features.get('high_priority_text', False):
            if features.get('age_group') == 'elderly':
                return {'triage_level': 2, 'confidence': 0.9, 'reason': 'high_priority_elderly'}
            else:
                return {'triage_level': 3, 'confidence': 0.85, 'reason': 'high_priority_symptoms'}
                
        # 基于生命体征的判断
        vital_signs = features.get('vital_signs', [])
        if vital_signs:
            avg_vitals = sum(vital_signs) / len(vital_signs)
            if avg_vitals > 100:  # 简化的异常判断
                return {'triage_level': 3, 'confidence': 0.75, 'reason': 'abnormal_vitals'}
                
        # 默认情况
        return {'triage_level': 4, 'confidence': 0.7, 'reason': 'standard_case'}
        
    def _post_process_prediction(self, prediction, original_data):
        """后处理预测结果"""
        result = prediction.copy()
        result['model_used'] = 'rule_based_demo'
        result['patient_age'] = original_data.get('age', 'unknown')
        
        # 添加解释
        explanations = {
            'high_priority_elderly': 'High priority symptoms in elderly patient',
            'high_priority_symptoms': 'Symptoms indicate urgent care needed',
            'abnormal_vitals': 'Vital signs outside normal range',
            'standard_case': 'Standard triage case'
        }
        result['explanation'] = explanations.get(result.get('reason', ''), 'Standard triage assessment')
        
        return result


@pytest.mark.integration
@pytest.mark.slow
class TestModelComparison:
    """测试不同模型的集成比较"""
    
    def test_tfidf_vs_rag_integration(self, realistic_patient_data):
        """测试TF-IDF模型与RAG模型的集成比较"""
        patient = realistic_patient_data[0]
        
        # 测试TF-IDF模型
        tfidf_result = self._mock_tfidf_model_integration(patient)
        
        # 测试RAG模型
        rag_result = self._mock_rag_model_integration(patient)
        
        # 验证两个模型都能正常工作
        for result in [tfidf_result, rag_result]:
            assert 'triage_level' in result
            assert 'confidence' in result
            assert 1 <= result['triage_level'] <= 5
            assert 0 <= result['confidence'] <= 1
            
        # 验证RAG模型提供更详细的信息
        assert 'reasoning' in rag_result
        assert 'similar_cases' in rag_result
        
        # 验证处理时间差异
        assert tfidf_result['processing_time'] < rag_result['processing_time']
        
    def _mock_tfidf_model_integration(self, patient_data):
        """模拟TF-IDF模型集成"""
        import time
        time.sleep(0.1)  # 模拟快速处理
        
        # 基于文本的简单分类
        text = patient_data['chiefcomplaint'].lower()
        if 'chest pain' in text:
            triage_level = 2
            confidence = 0.85
        elif 'fever' in text:
            triage_level = 3
            confidence = 0.80
        else:
            triage_level = 4
            confidence = 0.75
            
        return {
            'triage_level': triage_level,
            'confidence': confidence,
            'model_type': 'tfidf_rf',
            'processing_time': 0.1,
            'features_used': ['text_features', 'vital_signs']
        }
        
    def _mock_rag_model_integration(self, patient_data):
        """模拟RAG模型集成"""
        import time
        time.sleep(0.5)  # 模拟较慢的处理
        
        # 模拟RAG的详细推理过程
        text = patient_data['chiefcomplaint'].lower()
        
        if 'chest pain' in text:
            triage_level = 2
            confidence = 0.90
            reasoning = "Based on analysis of similar chest pain cases, this patient requires urgent evaluation."
            similar_cases = [
                {"case_id": "case_001", "similarity": 0.92, "outcome": "ESI 2"},
                {"case_id": "case_015", "similarity": 0.87, "outcome": "ESI 2"}
            ]
        else:
            triage_level = 3
            confidence = 0.85
            reasoning = "Standard triage case based on symptom analysis."
            similar_cases = [
                {"case_id": "case_045", "similarity": 0.78, "outcome": "ESI 3"}
            ]
            
        return {
            'triage_level': triage_level,
            'confidence': confidence,
            'model_type': 'llama_rag',
            'processing_time': 0.5,
            'reasoning': reasoning,
            'similar_cases': similar_cases,
            'retrieval_quality': 0.85
        }


@pytest.mark.integration
class TestSystemReliability:
    """测试系统可靠性"""
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 测试各种错误情况
        error_cases = [
            {'chiefcomplaint': None},  # 空文本
            {'age': -5},  # 无效年龄
            {'temperature': 200},  # 异常生命体征
            {},  # 空数据
        ]
        
        for error_case in error_cases:
            try:
                result = self._robust_triage_system(error_case)
                # 系统应该能处理错误并返回合理结果
                assert 'triage_level' in result
                assert 'error_handled' in result
                assert result['error_handled'] == True
            except Exception as e:
                # 如果抛出异常，应该是预期的异常类型
                assert isinstance(e, (ValueError, TypeError))
                
    def _robust_triage_system(self, patient_data):
        """健壮的分诊系统"""
        try:
            # 数据验证
            if not patient_data:
                raise ValueError("Empty patient data")
                
            # 基本的数据清理和验证
            cleaned_data = {}
            if 'chiefcomplaint' in patient_data and patient_data['chiefcomplaint']:
                cleaned_data['chiefcomplaint'] = str(patient_data['chiefcomplaint'])
            else:
                cleaned_data['chiefcomplaint'] = "no complaint provided"
                
            if 'age' in patient_data:
                age = patient_data['age']
                if age < 0 or age > 120:
                    cleaned_data['age'] = 30  # 默认年龄
                else:
                    cleaned_data['age'] = age
                    
            # 执行预测
            return {
                'triage_level': 4,  # 保守的默认级别
                'confidence': 0.6,
                'error_handled': True,
                'data_quality': 'low'
            }
            
        except Exception as e:
            # 最后的安全网
            return {
                'triage_level': 5,  # 最低优先级
                'confidence': 0.5,
                'error_handled': True,
                'error_type': type(e).__name__
            }


# 集成测试专用的fixtures
@pytest.fixture
def realistic_patient_data():
    """真实的患者数据样本"""
    return [
        {
            'patient_id': 'P001',
            'chiefcomplaint': 'chest pain and shortness of breath for 2 hours',
            'age': 65,
            'gender': 'M',
            'temperature': 98.6,
            'heartrate': 95,
            'resprate': 20,
            'o2sat': 96,
            'sbp': 140,
            'dbp': 90,
            'pain_score': 8
        },
        {
            'patient_id': 'P002', 
            'chiefcomplaint': 'fever and cough for 3 days',
            'age': 34,
            'gender': 'F',
            'temperature': 102.1,
            'heartrate': 105,
            'resprate': 22,
            'o2sat': 97,
            'sbp': 120,
            'dbp': 80,
            'pain_score': 3
        },
        {
            'patient_id': 'P003',
            'chiefcomplaint': 'headache and nausea',
            'age': 28,
            'gender': 'F',
            'temperature': 99.2,
            'heartrate': 78,
            'resprate': 16,
            'o2sat': 99,
            'sbp': 115,
            'dbp': 75,
            'pain_score': 6
        }
    ]


@pytest.fixture(scope="session")
def test_database():
    """测试数据库会话"""
    # 创建临时测试数据库
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    
    # 初始化测试数据
    test_data = {
        'predictions': [],
        'models': ['tfidf_rf', 'llama_rag'],
        'performance_metrics': {}
    }
    
    with open(temp_db.name, 'w') as f:
        json.dump(test_data, f)
        
    yield temp_db.name
    
    # 清理
    os.unlink(temp_db.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

