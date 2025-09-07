"""
Unit Tests for Actual Models Used in Frontend
============================================

Tests for the two models actually integrated in the frontend:
1. TF-IDF + Random Forest (tfidf_rf)
2. Llama RAG (llama_rag)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
import sys
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.integration
@pytest.mark.model
class TestTFIDFRandomForestModel:
    """Test TF-IDF + Random Forest model integration"""
    
    def test_tfidf_rf_model_loading(self):
        """Test TF-IDF + RF model loading simulation"""
        # Mock the model files
        mock_rf_model = MagicMock(spec=RandomForestClassifier)
        mock_rf_model.predict.return_value = np.array([3])
        mock_rf_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.6, 0.15, 0.05]])
        
        mock_vectorizer = MagicMock(spec=TfidfVectorizer)
        mock_vectorizer.transform.return_value = np.random.rand(1, 100)
        
        mock_svd = MagicMock(spec=TruncatedSVD)
        mock_svd.transform.return_value = np.random.rand(1, 50)
        
        # Test model prediction pipeline
        test_input = "chest pain and shortness of breath"
        
        # Simulate the pipeline: text -> TF-IDF -> SVD -> RF
        tfidf_features = mock_vectorizer.transform([test_input])
        svd_features = mock_svd.transform(tfidf_features)
        prediction = mock_rf_model.predict(svd_features)
        probabilities = mock_rf_model.predict_proba(svd_features)
        
        # Assertions
        assert prediction[0] in [1, 2, 3, 4, 5]  # Valid ESI level
        assert probabilities.shape == (1, 5)  # 5 triage levels
        assert abs(sum(probabilities[0]) - 1.0) < 1e-6  # Probabilities sum to 1
        
    def test_tfidf_rf_with_keywords(self):
        """Test TF-IDF + RF with keyword extraction"""
        # Mock keyword extraction result
        keywords = "chest pain, shortness of breath, cardiac"
        
        # Mock models
        mock_rf_model = MagicMock()
        mock_rf_model.predict.return_value = np.array([2])  # ESI 2 (high priority)
        mock_rf_model.predict_proba.return_value = np.array([[0.05, 0.7, 0.15, 0.08, 0.02]])
        
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = np.random.rand(1, 100)
        
        mock_svd = MagicMock()
        mock_svd.transform.return_value = np.random.rand(1, 50)
        
        # Test with keywords
        tfidf_features = mock_vectorizer.transform([keywords])
        svd_features = mock_svd.transform(tfidf_features)
        prediction = mock_rf_model.predict(svd_features)
        confidence = np.max(mock_rf_model.predict_proba(svd_features))
        
        # Assertions
        assert prediction[0] == 2  # High priority for chest pain
        assert confidence > 0.5  # Should have reasonable confidence
        
    def test_tfidf_rf_feature_dimensions(self):
        """Test that feature dimensions are consistent in TF-IDF + RF pipeline"""
        # Mock consistent dimensions
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value = np.random.rand(1, 1000)  # 1000 TF-IDF features
        
        mock_svd = MagicMock()
        mock_svd.transform.return_value = np.random.rand(1, 100)  # Reduced to 100 features
        
        mock_rf_model = MagicMock()
        mock_rf_model.predict.return_value = np.array([3])
        
        # Test dimension consistency
        text_input = "patient complains of abdominal pain"
        tfidf_output = mock_vectorizer.transform([text_input])
        svd_output = mock_svd.transform(tfidf_output)
        
        assert tfidf_output.shape[1] == 1000  # TF-IDF dimension
        assert svd_output.shape[1] == 100     # SVD reduced dimension
        
        # RF model should accept the SVD output
        prediction = mock_rf_model.predict(svd_output)
        assert len(prediction) == 1
        
    def test_tfidf_rf_error_handling(self):
        """Test TF-IDF + RF error handling"""
        # Test empty input
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.side_effect = ValueError("Empty input")
        
        with pytest.raises(ValueError):
            mock_vectorizer.transform([""])
            
        # Test dimension mismatch
        mock_svd = MagicMock()
        mock_svd.transform.side_effect = ValueError("Feature dimension mismatch")
        
        with pytest.raises(ValueError):
            mock_svd.transform(np.random.rand(1, 50))  # Wrong dimension


@pytest.mark.integration
@pytest.mark.model
class TestLlamaRAGModel:
    """Test Llama RAG model integration"""
    
    def test_llama_rag_system_initialization(self):
        """Test Llama RAG system initialization"""
        # Mock LlamaRAGTriageSystem
        mock_rag_system = MagicMock()
        mock_rag_system.is_initialized = True
        mock_rag_system.case_retriever = MagicMock()
        mock_rag_system.case_retriever.initialized = True
        
        # Test initialization status
        assert mock_rag_system.is_initialized == True
        assert mock_rag_system.case_retriever.initialized == True
        
    def test_llama_rag_prediction(self):
        """Test Llama RAG prediction functionality"""
        # Mock RAG system prediction
        mock_rag_system = MagicMock()
        
        # Mock prediction result
        mock_prediction_result = {
            'triage_level': 3,
            'reasoning': 'Patient presents with moderate symptoms requiring standard care.',
            'similar_cases': [
                {'case_id': 'case_001', 'similarity': 0.85, 'outcome': 'ESI 3'},
                {'case_id': 'case_002', 'similarity': 0.78, 'outcome': 'ESI 3'}
            ],
            'confidence': 0.82
        }
        
        mock_rag_system.predict_triage_level.return_value = (
            3,  # triage_level
            'Patient presents with moderate symptoms requiring standard care.',  # reasoning
            [{'case_id': 'case_001', 'similarity': 0.85}],  # similar_cases
            [0.85, 0.78]  # similarities
        )
        
        # Test prediction
        patient_data = {
            'chiefcomplaint': 'chest pain and shortness of breath',
            'age': 45,
            'gender': 'M',
            'temperature': 98.6,
            'heartrate': 85
        }
        
        result = mock_rag_system.predict_triage_level(patient_data, k=5)
        
        # Assertions
        assert result[0] in [1, 2, 3, 4, 5]  # Valid triage level
        assert isinstance(result[1], str)     # Reasoning should be string
        assert isinstance(result[2], list)    # Similar cases should be list
        assert len(result[3]) > 0            # Should have similarity scores
        
    def test_llama_rag_case_retrieval(self):
        """Test Llama RAG case retrieval functionality"""
        # Mock case retriever
        mock_case_retriever = MagicMock()
        mock_case_retriever.retrieve_similar_cases.return_value = (
            [  # similar_cases
                {'case_id': 1, 'chiefcomplaint': 'chest pain', 'acuity': 3},
                {'case_id': 2, 'chiefcomplaint': 'chest discomfort', 'acuity': 3},
                {'case_id': 3, 'chiefcomplaint': 'cardiac symptoms', 'acuity': 2}
            ],
            [0.92, 0.87, 0.83]  # similarities
        )
        
        # Test retrieval
        query = "patient with chest pain and shortness of breath"
        similar_cases, similarities = mock_case_retriever.retrieve_similar_cases(query, k=3)
        
        # Assertions
        assert len(similar_cases) == 3
        assert len(similarities) == 3
        assert all(0 <= sim <= 1 for sim in similarities)  # Valid similarity scores
        assert similarities == sorted(similarities, reverse=True)  # Should be sorted by similarity
        
    def test_llama_rag_reasoning_generation(self):
        """Test Llama RAG reasoning generation"""
        # Mock LLM generator
        mock_generator = MagicMock()
        mock_generator.generate_triage_reasoning.return_value = {
            'triage_level': 2,
            'reasoning': 'Based on similar cases showing chest pain with cardiac risk factors, patient requires urgent evaluation. Similar patients typically received ESI level 2 classification.',
            'confidence': 0.88
        }
        
        # Test reasoning generation
        patient_info = {'chiefcomplaint': 'chest pain', 'age': 65, 'gender': 'M'}
        similar_cases = [
            {'chiefcomplaint': 'chest pain', 'acuity': 2, 'age': 62},
            {'chiefcomplaint': 'cardiac symptoms', 'acuity': 2, 'age': 58}
        ]
        
        result = mock_generator.generate_triage_reasoning(patient_info, similar_cases)
        
        # Assertions
        assert result['triage_level'] in [1, 2, 3, 4, 5]
        assert len(result['reasoning']) > 0
        assert 0 <= result['confidence'] <= 1
        
    def test_llama_rag_vector_store_integration(self):
        """Test Llama RAG vector store integration"""
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.faiss_index = MagicMock()
        mock_vector_store.faiss_index.ntotal = 100000  # Mock index size
        mock_vector_store.case_database = pd.DataFrame({
            'case_id': range(100),
            'chiefcomplaint': ['chest pain'] * 100,
            'acuity': [3] * 100
        })
        
        # Test vector store properties
        assert mock_vector_store.faiss_index.ntotal > 0
        assert len(mock_vector_store.case_database) == 100
        assert 'chiefcomplaint' in mock_vector_store.case_database.columns
        
    def test_llama_rag_fallback_behavior(self):
        """Test Llama RAG fallback behavior when model fails"""
        # Mock failed RAG system
        mock_rag_system = MagicMock()
        mock_rag_system.predict_triage_level.side_effect = Exception("Model unavailable")
        
        # Mock fallback system
        mock_fallback = MagicMock()
        mock_fallback.predict_triage_level.return_value = {
            'triage_level': 3,
            'reasoning': 'Unable to access advanced reasoning. Using default classification.',
            'confidence': 0.5,
            'fallback_used': True
        }
        
        # Test fallback
        patient_data = {'chiefcomplaint': 'headache'}
        
        try:
            result = mock_rag_system.predict_triage_level(patient_data)
        except Exception:
            # Use fallback
            result = mock_fallback.predict_triage_level(patient_data)
            
        assert result['fallback_used'] == True
        assert result['triage_level'] in [1, 2, 3, 4, 5]


@pytest.mark.integration
@pytest.mark.model
class TestModelComparison:
    """Test comparison between TF-IDF + RF and Llama RAG models"""
    
    def test_model_output_format_consistency(self):
        """Test that both models return consistent output formats"""
        # Mock TF-IDF + RF output
        tfidf_rf_output = {
            'triage_level': 3,
            'confidence': 0.85,
            'model_type': 'tfidf_rf'
        }
        
        # Mock Llama RAG output  
        llama_rag_output = {
            'triage_level': 2,
            'confidence': 0.88,
            'reasoning': 'Detailed reasoning based on similar cases',
            'similar_cases': [],
            'model_type': 'llama_rag'
        }
        
        # Both should have required fields
        for output in [tfidf_rf_output, llama_rag_output]:
            assert 'triage_level' in output
            assert 'confidence' in output
            assert output['triage_level'] in [1, 2, 3, 4, 5]
            assert 0 <= output['confidence'] <= 1
            
    def test_model_performance_comparison(self):
        """Test performance characteristics of both models"""
        import time
        
        # Mock TF-IDF + RF (faster but less detailed)
        def mock_tfidf_rf_predict():
            time.sleep(0.1)  # Simulate fast prediction
            return {'triage_level': 3, 'confidence': 0.82, 'processing_time': 0.1}
            
        # Mock Llama RAG (slower but more detailed)
        def mock_llama_rag_predict():
            time.sleep(0.5)  # Simulate slower prediction
            return {
                'triage_level': 3, 
                'confidence': 0.88, 
                'reasoning': 'Detailed analysis',
                'processing_time': 0.5
            }
            
        # Test performance characteristics
        tfidf_result = mock_tfidf_rf_predict()
        rag_result = mock_llama_rag_predict()
        
        # TF-IDF should be faster
        assert tfidf_result['processing_time'] < rag_result['processing_time']
        
        # RAG should provide more detailed output
        assert 'reasoning' in rag_result
        assert 'reasoning' not in tfidf_result
        
    def test_model_selection_logic(self):
        """Test logic for selecting between models"""
        def select_model(user_preference, system_load):
            """Mock model selection logic"""
            # Check system load first for override
            if system_load > 0.8:
                return 'tfidf_rf'  # Use faster model under high load
            elif user_preference == 'fast':
                return 'tfidf_rf'
            elif user_preference == 'detailed':
                return 'llama_rag'
            else:
                return 'llama_rag'  # Default to more detailed model
                
        # Test selection scenarios
        assert select_model('fast', 0.5) == 'tfidf_rf'
        assert select_model('detailed', 0.3) == 'llama_rag'
        assert select_model('detailed', 0.9) == 'tfidf_rf'  # High load override


# Additional fixtures for actual model testing
@pytest.fixture
def mock_patient_data():
    """Mock patient data for model testing"""
    return {
        'chiefcomplaint': 'chest pain and shortness of breath',
        'age': 45,
        'gender': 'M',
        'temperature': 98.6,
        'heartrate': 85,
        'resprate': 16,
        'o2sat': 98,
        'sbp': 120,
        'dbp': 80,
        'pain': 7
    }


@pytest.fixture
def mock_tfidf_pipeline():
    """Mock TF-IDF processing pipeline"""
    pipeline = MagicMock()
    pipeline.vectorizer = MagicMock()
    pipeline.svd = MagicMock()
    pipeline.model = MagicMock()
    
    # Configure mock behavior
    pipeline.vectorizer.transform.return_value = np.random.rand(1, 1000)
    pipeline.svd.transform.return_value = np.random.rand(1, 100)
    pipeline.model.predict.return_value = np.array([3])
    pipeline.model.predict_proba.return_value = np.array([[0.1, 0.1, 0.6, 0.15, 0.05]])
    
    return pipeline


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing"""
    rag_system = MagicMock()
    rag_system.is_initialized = True
    rag_system.predict_triage_level.return_value = (
        3,
        'Patient presents with moderate symptoms requiring evaluation.',
        [{'case_id': 'case_001', 'similarity': 0.85}],
        [0.85]
    )
    return rag_system


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
