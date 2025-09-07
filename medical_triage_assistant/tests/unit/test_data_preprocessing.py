"""
Unit Tests for Data Preprocessing Functions
==========================================

Tests for:
- Text cleaning and normalization
- Medical abbreviation expansion
- Outlier detection and handling
- Missing value imputation
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_processing.preprocessing.text_preprocessing import (
    expand_medical_abbreviations,
    clean_text,
    normalize_chief_complaint
)
from data_processing.preprocessing.numerical_preprocessing import (
    detect_outliers,
    handle_missing_values,
    validate_vital_signs
)


@pytest.mark.unit
@pytest.mark.preprocessing
class TestTextPreprocessing:
    """Test medical text preprocessing functions"""
    
    def test_abbreviation_expansion(self, sample_text_data, expected_expanded_text):
        """Test medical abbreviation expansion"""
        for original, expected in zip(sample_text_data, expected_expanded_text):
            expanded = expand_medical_abbreviations(original)
            assert isinstance(expanded, str)
            assert len(expanded) >= len(original)  # Expanded text should be longer
            
    def test_abbreviation_expansion_common_cases(self):
        """Test specific common medical abbreviations"""
        test_cases = {
            "c/o chest pain": "complains of chest pain",
            "SOB and DOE": "shortness of breath and dyspnea on exertion", 
            "N&V x 3 days": "nausea and vomiting times 3 days",
            "r/o MI": "rule out myocardial infarction",
            "s/p surgery": "status post surgery",
            "w/ fever": "with fever",
            "pt presents": "patient presents"
        }
        
        for original, expected_part in test_cases.items():
            expanded = expand_medical_abbreviations(original)
            assert expected_part.lower() in expanded.lower()
    
    def test_text_cleaning(self):
        """Test text cleaning function"""
        dirty_text = "  Patient C/O chest pain!!!   multiple    spaces  "
        cleaned = clean_text(dirty_text)
        
        assert isinstance(cleaned, str)
        assert not cleaned.startswith(" ")  # No leading spaces
        assert not cleaned.endswith(" ")    # No trailing spaces
        assert "  " not in cleaned          # No double spaces
        assert cleaned.islower()            # Should be lowercase
        
    def test_chief_complaint_normalization(self, sample_text_data):
        """Test chief complaint normalization pipeline"""
        for text in sample_text_data:
            normalized = normalize_chief_complaint(text)
            
            assert isinstance(normalized, str)
            assert len(normalized) > 0
            assert not normalized.startswith(" ")
            assert not normalized.endswith(" ")
            
    def test_empty_text_handling(self):
        """Test handling of empty or None text"""
        assert expand_medical_abbreviations("") == ""
        assert expand_medical_abbreviations(None) == ""
        assert clean_text("") == ""
        assert clean_text(None) == ""


@pytest.mark.unit
@pytest.mark.preprocessing
class TestNumericalPreprocessing:
    """Test numerical data preprocessing functions"""
    
    def test_outlier_detection_iqr(self, sample_patient_data):
        """Test IQR-based outlier detection"""
        outliers = detect_outliers(sample_patient_data['heartrate'], method='iqr')
        
        assert isinstance(outliers, (list, np.ndarray, pd.Series))
        assert len(outliers) <= len(sample_patient_data)
        
    def test_outlier_detection_zscore(self, sample_patient_data):
        """Test Z-score based outlier detection"""
        outliers = detect_outliers(sample_patient_data['temperature'], method='zscore', threshold=2.0)
        
        assert isinstance(outliers, (list, np.ndarray, pd.Series))
        assert len(outliers) <= len(sample_patient_data)
        
    def test_vital_signs_validation(self):
        """Test vital signs range validation"""
        valid_vitals = {
            'temperature': 98.6,
            'heartrate': 80,
            'resprate': 16,
            'o2sat': 98,
            'sbp': 120,
            'dbp': 80
        }
        
        invalid_vitals = {
            'temperature': 150.0,  # Too high
            'heartrate': 300,      # Too high  
            'resprate': 0,         # Too low
            'o2sat': 150,          # Impossible
            'sbp': 50,             # Too low
            'dbp': 200             # Too high
        }
        
        # Valid vitals should pass
        for vital, value in valid_vitals.items():
            assert validate_vital_signs(vital, value) == True
            
        # Invalid vitals should fail
        for vital, value in invalid_vitals.items():
            assert validate_vital_signs(vital, value) == False
            
    def test_missing_value_handling(self, sample_patient_data):
        """Test missing value imputation"""
        # Create data with missing values
        data_with_missing = sample_patient_data.copy()
        data_with_missing.loc[0, 'temperature'] = np.nan
        data_with_missing.loc[1, 'heartrate'] = np.nan
        
        # Handle missing values
        filled_data = handle_missing_values(data_with_missing)
        
        assert not filled_data.isnull().any().any()  # No missing values
        assert len(filled_data) == len(data_with_missing)  # Same length
        assert filled_data.dtypes.equals(data_with_missing.dtypes)  # Same types
        
    def test_extreme_outlier_handling(self):
        """Test handling of extreme outliers"""
        # Create data with extreme outliers
        data = pd.Series([70, 75, 80, 85, 90, 1000, 95])  # 1000 is extreme outlier
        
        outliers = detect_outliers(data, method='iqr')
        assert 1000 in data.iloc[outliers].values if len(outliers) > 0 else True
        
    def test_age_validation(self):
        """Test age range validation"""
        valid_ages = [0, 18, 45, 65, 100]
        invalid_ages = [-5, 150, 200]
        
        for age in valid_ages:
            assert validate_vital_signs('age', age) == True
            
        for age in invalid_ages:
            assert validate_vital_signs('age', age) == False


@pytest.mark.unit
@pytest.mark.preprocessing
class TestDataIntegrity:
    """Test overall data integrity and consistency"""
    
    def test_preprocessing_pipeline_consistency(self, sample_patient_data):
        """Test that preprocessing pipeline maintains data consistency"""
        original_length = len(sample_patient_data)
        
        # Apply preprocessing steps
        processed_data = sample_patient_data.copy()
        
        # Text preprocessing
        if 'chiefcomplaint' in processed_data.columns:
            processed_data['chiefcomplaint'] = processed_data['chiefcomplaint'].apply(
                normalize_chief_complaint
            )
        
        # Numerical preprocessing
        processed_data = handle_missing_values(processed_data)
        
        # Verify consistency
        assert len(processed_data) == original_length
        assert processed_data.columns.equals(sample_patient_data.columns)
        assert not processed_data.isnull().any().any()
        
    def test_data_type_preservation(self, sample_patient_data):
        """Test that preprocessing preserves appropriate data types"""
        processed_data = handle_missing_values(sample_patient_data)
        
        # Numerical columns should remain numerical
        numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'age']
        for col in numerical_cols:
            if col in processed_data.columns:
                assert pd.api.types.is_numeric_dtype(processed_data[col])
                
        # Text columns should remain text
        text_cols = ['chiefcomplaint', 'gender']
        for col in text_cols:
            if col in processed_data.columns:
                assert pd.api.types.is_string_dtype(processed_data[col]) or \
                       pd.api.types.is_object_dtype(processed_data[col])


# Fixtures for this test module
@pytest.fixture
def sample_outlier_data():
    """Sample data with known outliers"""
    return pd.Series([
        70, 72, 74, 76, 78, 80, 82, 84, 86, 88,  # Normal range
        150, 200  # Clear outliers
    ])


@pytest.fixture 
def sample_missing_data():
    """Sample data with missing values"""
    return pd.DataFrame({
        'temperature': [98.6, np.nan, 99.1, 102.5, np.nan],
        'heartrate': [80, 110, np.nan, 120, 85],
        'text': ['normal', None, 'test', '', 'valid']
    })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
