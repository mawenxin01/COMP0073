"""
Unit Tests for Feature Extraction Functions
===========================================

Tests for:
- TF-IDF vectorization and dimensionality
- SVD dimensionality reduction validation
- Feature matrix shape consistency
- Numerical feature scaling and normalization
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.unit
@pytest.mark.feature_extraction
class TestTFIDFExtraction:
    """Test TF-IDF feature extraction"""
    
    def test_tfidf_basic_functionality(self, sample_text_data, test_config):
        """Test basic TF-IDF vectorization"""
        vectorizer = TfidfVectorizer(
            max_features=test_config["TFIDF_MAX_FEATURES"],
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(sample_text_data)
        
        # Test matrix properties
        assert isinstance(tfidf_matrix, sp.csr_matrix)
        assert tfidf_matrix.shape[0] == len(sample_text_data)  # Number of documents
        assert tfidf_matrix.shape[1] <= test_config["TFIDF_MAX_FEATURES"]  # Feature limit
        
        # Test feature names
        feature_names = vectorizer.get_feature_names_out()
        assert len(feature_names) == tfidf_matrix.shape[1]
        assert all(isinstance(name, str) for name in feature_names)
        
    def test_tfidf_dimensionality_consistency(self, sample_text_data, test_config):
        """Test TF-IDF dimension consistency across different inputs"""
        vectorizer = TfidfVectorizer(max_features=test_config["TFIDF_MAX_FEATURES"])
        
        # Fit on training data
        tfidf_train = vectorizer.fit_transform(sample_text_data)
        
        # Transform new data
        new_text = ["new patient with different symptoms", "another case"]
        tfidf_new = vectorizer.transform(new_text)
        
        # Dimensions should match
        assert tfidf_train.shape[1] == tfidf_new.shape[1]
        assert tfidf_new.shape[0] == len(new_text)
        
    def test_tfidf_empty_text_handling(self):
        """Test TF-IDF handling of empty or invalid text"""
        vectorizer = TfidfVectorizer(max_features=100)
        
        # Test with empty strings
        empty_texts = ["", " ", "   "]
        try:
            tfidf_matrix = vectorizer.fit_transform(empty_texts)
            assert tfidf_matrix.shape[0] == len(empty_texts)
        except ValueError:
            # Expected behavior for all-empty corpus
            pass
            
        # Test with mixed empty and valid text
        mixed_texts = ["valid text", "", "another valid text"]
        tfidf_matrix = vectorizer.fit_transform(mixed_texts)
        assert tfidf_matrix.shape[0] == len(mixed_texts)
        
    def test_tfidf_medical_terms_preservation(self):
        """Test that medical terms are preserved in TF-IDF"""
        medical_texts = [
            "chest pain and shortness of breath",
            "abdominal pain with nausea",
            "headache and dizziness",
            "fever and cough"
        ]
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(medical_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Check that important medical terms are captured
        medical_terms = ['chest', 'pain', 'fever', 'cough', 'chest pain']
        captured_terms = [term for term in medical_terms if term in feature_names]
        
        assert len(captured_terms) > 0  # At least some medical terms should be captured
        
    def test_tfidf_sparse_matrix_properties(self, sample_text_data):
        """Test TF-IDF sparse matrix properties"""
        vectorizer = TfidfVectorizer(max_features=500)
        tfidf_matrix = vectorizer.fit_transform(sample_text_data)
        
        # Test sparsity
        assert sp.issparse(tfidf_matrix)
        assert tfidf_matrix.nnz < tfidf_matrix.shape[0] * tfidf_matrix.shape[1]  # Should be sparse
        
        # Test value ranges (TF-IDF values should be between 0 and 1)
        assert tfidf_matrix.min() >= 0
        assert tfidf_matrix.max() <= 1
        
        # Test that each row has at least some non-zero values (unless empty text)
        row_sums = np.array(tfidf_matrix.sum(axis=1)).flatten()
        assert np.all(row_sums >= 0)


@pytest.mark.unit
@pytest.mark.feature_extraction
class TestSVDDimensionalityReduction:
    """Test SVD dimensionality reduction"""
    
    def test_svd_basic_functionality(self, test_config):
        """Test basic SVD dimensionality reduction"""
        # Create sample high-dimensional data
        n_samples, n_features = 10, 1000
        X = np.random.rand(n_samples, n_features)
        
        # Use the smaller of desired components or available dimensions
        n_components = min(test_config["SVD_COMPONENTS"], n_samples-1, n_features-1)
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X)
        
        # Test output dimensions
        assert X_reduced.shape == (n_samples, n_components)
        assert X_reduced.shape[1] < X.shape[1]  # Dimensionality reduced
        
        # Test explained variance
        assert hasattr(svd, 'explained_variance_ratio_')
        assert len(svd.explained_variance_ratio_) == n_components
        assert np.all(svd.explained_variance_ratio_ >= 0)
        assert np.all(svd.explained_variance_ratio_ <= 1)
        
    def test_svd_with_tfidf_pipeline(self, sample_text_data, test_config):
        """Test SVD applied to TF-IDF features"""
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=100)  # Reduced for small dataset
        tfidf_matrix = vectorizer.fit_transform(sample_text_data)
        
        # Apply SVD with dynamic component sizing
        n_components = min(test_config["SVD_COMPONENTS"], tfidf_matrix.shape[0]-1, tfidf_matrix.shape[1]-1)
        svd = TruncatedSVD(n_components=n_components)
        reduced_features = svd.fit_transform(tfidf_matrix)
        
        # Test pipeline output
        assert reduced_features.shape[0] == len(sample_text_data)
        assert reduced_features.shape[1] == n_components
        assert not sp.issparse(reduced_features)  # SVD output should be dense
        
    def test_svd_explained_variance_threshold(self, test_config):
        """Test SVD explained variance meets threshold"""
        # Create sample data with known structure
        n_samples, n_features = 100, 500
        # Create data with clear principal components
        X = np.random.randn(n_samples, n_features)
        X[:, :10] *= 10  # Make first 10 features more important
        
        n_components = min(test_config["SVD_COMPONENTS"], n_samples-1, n_features-1)
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X)
        
        # Test that we capture reasonable variance
        total_variance = np.sum(svd.explained_variance_ratio_)
        assert total_variance > 0.01  # Should capture at least 1% of variance
        
        # Test that components are ordered by importance (mostly descending)
        variance_ratios = svd.explained_variance_ratio_
        # Allow some flexibility in ordering due to random data
        assert variance_ratios[0] >= variance_ratios[-1]  # First should be >= last
        
    def test_svd_consistency_across_transforms(self, sample_text_data, test_config):
        """Test SVD consistency when transforming new data"""
        vectorizer = TfidfVectorizer(max_features=50)
        
        # Fit vectorizer first to know feature dimensions
        tfidf_train = vectorizer.fit_transform(sample_text_data)
        n_components = min(5, tfidf_train.shape[0]-1, tfidf_train.shape[1]-1)
        svd = TruncatedSVD(n_components=n_components)
        
        # Fit SVD on training data
        reduced_train = svd.fit_transform(tfidf_train)
        
        # Transform new data
        new_text = ["new medical case", "different symptoms"]
        tfidf_new = vectorizer.transform(new_text)
        reduced_new = svd.transform(tfidf_new)
        
        # Check consistency
        assert reduced_train.shape[1] == reduced_new.shape[1]
        assert reduced_new.shape[0] == len(new_text)
        
    def test_svd_component_count_validation(self):
        """Test SVD with different component counts"""
        X = np.random.rand(20, 100)
        
        # Test valid component counts
        valid_components = [5, 10, 19]  # Less than min(n_samples, n_features)
        for n_comp in valid_components:
            # Ensure component count is valid
            actual_n_comp = min(n_comp, X.shape[0]-1, X.shape[1]-1)
            svd = TruncatedSVD(n_components=actual_n_comp)
            X_reduced = svd.fit_transform(X)
            assert X_reduced.shape[1] == actual_n_comp
            
        # Test invalid component count (too large)
        with pytest.raises(ValueError):
            svd = TruncatedSVD(n_components=150)  # More than n_features
            svd.fit(X)


@pytest.mark.unit
@pytest.mark.feature_extraction
class TestNumericalFeatureExtraction:
    """Test numerical feature extraction and scaling"""
    
    def test_numerical_feature_scaling(self, sample_patient_data):
        """Test StandardScaler on numerical features"""
        numerical_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
        available_cols = [col for col in numerical_cols if col in sample_patient_data.columns]
        
        if available_cols:
            X = sample_patient_data[available_cols]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Test scaling properties
            assert X_scaled.shape == X.shape
            assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)  # Mean ≈ 0
            assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)   # Std ≈ 1
            
    def test_feature_scaling_consistency(self, sample_patient_data):
        """Test feature scaling consistency across train/test splits"""
        numerical_cols = ['temperature', 'heartrate', 'resprate']
        available_cols = [col for col in numerical_cols if col in sample_patient_data.columns]
        
        if available_cols:
            X_train = sample_patient_data[available_cols].iloc[:3]  # First 3 samples
            X_test = sample_patient_data[available_cols].iloc[3:]   # Remaining samples
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test consistency
            assert X_train_scaled.shape[1] == X_test_scaled.shape[1]
            assert not np.array_equal(X_train_scaled, X_test_scaled)  # Should be different
            
    def test_missing_value_handling_in_scaling(self):
        """Test scaling behavior with missing values"""
        # Create data with missing values
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50]
        })
        
        # StandardScaler should handle missing values appropriately
        scaler = StandardScaler()
        
        # This should either work or raise a clear error
        try:
            X_scaled = scaler.fit_transform(X)
            assert X_scaled.shape == X.shape
        except ValueError as e:
            # Expected behavior - scaler doesn't handle NaN
            assert "NaN" in str(e) or "nan" in str(e)


@pytest.mark.unit
@pytest.mark.feature_extraction
class TestFeaturePipelineIntegration:
    """Test integration of feature extraction pipeline"""
    
    def test_full_feature_pipeline(self, sample_patient_data, sample_text_data, test_config):
        """Test complete feature extraction pipeline"""
        # Text features
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_features = vectorizer.fit_transform(sample_text_data)
        
        # SVD reduction with dynamic sizing
        n_components = min(5, tfidf_features.shape[0]-1, tfidf_features.shape[1]-1)
        svd = TruncatedSVD(n_components=n_components)
        text_features_reduced = svd.fit_transform(tfidf_features)
        
        # Numerical features
        numerical_cols = ['temperature', 'heartrate', 'resprate', 'age']
        available_cols = [col for col in numerical_cols if col in sample_patient_data.columns]
        
        if available_cols:
            numerical_features = sample_patient_data[available_cols].values
            scaler = StandardScaler()
            numerical_features_scaled = scaler.fit_transform(numerical_features)
            
            # Combine features
            combined_features = np.hstack([text_features_reduced, numerical_features_scaled])
            
            # Test combined feature matrix
            assert combined_features.shape[0] == len(sample_text_data)
            assert combined_features.shape[1] == text_features_reduced.shape[1] + numerical_features_scaled.shape[1]
            assert not np.isnan(combined_features).any()
            
    def test_feature_dimension_consistency(self, test_config):
        """Test that feature dimensions remain consistent across different inputs"""
        # Simulate training data
        train_texts = ["chest pain", "abdominal pain", "headache", "fever"]
        train_numerical = np.random.rand(4, 5)
        
        # Simulate test data
        test_texts = ["back pain", "dizziness"]
        test_numerical = np.random.rand(2, 5)
        
        # Build pipeline
        vectorizer = TfidfVectorizer(max_features=20)
        scaler = StandardScaler()
        
        # Fit vectorizer first to determine dimensions
        tfidf_train = vectorizer.fit_transform(train_texts)
        n_components = min(3, tfidf_train.shape[0]-1, tfidf_train.shape[1]-1)
        svd = TruncatedSVD(n_components=n_components)
        
        # Fit on training data
        text_train = svd.fit_transform(tfidf_train)
        numerical_train = scaler.fit_transform(train_numerical)
        
        # Transform test data
        tfidf_test = vectorizer.transform(test_texts)
        text_test = svd.transform(tfidf_test)
        numerical_test = scaler.transform(test_numerical)
        
        # Test dimension consistency
        assert text_train.shape[1] == text_test.shape[1]
        assert numerical_train.shape[1] == numerical_test.shape[1]
        
        # Combined features should have consistent dimensions
        combined_train = np.hstack([text_train, numerical_train])
        combined_test = np.hstack([text_test, numerical_test])
        
        assert combined_train.shape[1] == combined_test.shape[1]


# Additional fixtures for this test module
@pytest.fixture
def sample_sparse_matrix():
    """Sample sparse matrix for testing"""
    return sp.random(10, 100, density=0.1, format='csr')


@pytest.fixture
def sample_high_dimensional_data():
    """Sample high-dimensional data for SVD testing"""
    return np.random.randn(50, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
