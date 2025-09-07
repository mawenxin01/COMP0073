"""
Pytest Configuration and Fixtures
================================

Shared fixtures and test configuration for the medical triage assistant tests.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return pd.DataFrame({
        'subject_id': [1001, 1002, 1003, 1004, 1005],
        'chiefcomplaint': [
            'chest pain and SOB',
            'abd pain w/ N&V',
            'h/a and dizziness',
            'fever and cough',
            'back pain r/o fracture'
        ],
        'temperature': [98.6, 101.2, 99.1, 102.5, 98.0],
        'heartrate': [80, 110, 75, 120, 85],
        'resprate': [16, 22, 14, 24, 18],
        'o2sat': [98, 95, 99, 92, 97],
        'sbp': [120, 140, 110, 130, 125],
        'dbp': [80, 90, 70, 85, 82],
        'pain': [5, 8, 3, 2, 7],
        'acuity': [3, 2, 4, 2, 3],
        'age': [45, 67, 23, 34, 56],
        'gender': ['M', 'F', 'F', 'M', 'M']
    })

@pytest.fixture
def sample_text_data():
    """Sample text data with medical abbreviations"""
    return [
        "Patient c/o chest pain and SOB",
        "Pt presents w/ abd pain, N&V x 2 days", 
        "H/A and dizziness, r/o migraine",
        "Fever and cough, ? pneumonia",
        "Back pain s/p fall, r/o fx"
    ]

@pytest.fixture
def expected_expanded_text():
    """Expected text after abbreviation expansion"""
    return [
        "Patient complains of chest pain and shortness of breath",
        "Patient presents with abdominal pain, nausea and vomiting times 2 days",
        "Headache and dizziness, rule out migraine", 
        "Fever and cough, question pneumonia",
        "Back pain status post fall, rule out fracture"
    ]

@pytest.fixture
def mock_tfidf_model():
    """Mock TF-IDF model for testing"""
    mock_model = MagicMock()
    mock_model.transform.return_value = np.random.rand(5, 1000)  # 5 samples, 1000 features
    mock_model.get_feature_names_out.return_value = [f"feature_{i}" for i in range(1000)]
    return mock_model

@pytest.fixture
def mock_svd_model():
    """Mock SVD model for testing"""
    mock_model = MagicMock()
    mock_model.transform.return_value = np.random.rand(5, 100)  # 5 samples, 100 components
    mock_model.n_components = 100
    return mock_model

@pytest.fixture
def sample_api_response():
    """Sample API response structure"""
    return {
        "success": True,
        "prediction": {
            "triage_level": 3,
            "confidence": 0.85,
            "processing_time": 1.23
        },
        "explanation": {
            "method": "TF-IDF + Random Forest",
            "key_features": ["chest pain", "shortness of breath", "heart rate"],
            "similar_cases": []
        },
        "metadata": {
            "model_version": "v1.0",
            "timestamp": "2025-01-04T12:00:00Z"
        }
    }

@pytest.fixture
def test_config():
    """Test configuration settings"""
    return {
        "TFIDF_MAX_FEATURES": 1000,
        "SVD_COMPONENTS": 10,  # Reduced to work with small test datasets
        "VALID_TRIAGE_LEVELS": [1, 2, 3, 4, 5],
        "CONFIDENCE_THRESHOLD": 0.0,
        "MAX_PROCESSING_TIME": 30.0
    }
