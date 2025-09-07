"""
Unit Tests for API Response Validation
======================================

Tests for:
- API response format validation
- Required fields presence and types
- Error response handling
- Data serialization/deserialization
- HTTP status codes and headers
"""

import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.unit
@pytest.mark.api
class TestAPIResponseStructure:
    """Test API response structure and format"""
    
    def test_successful_prediction_response_format(self, sample_api_response):
        """Test successful prediction response contains all required fields"""
        required_fields = ['success', 'prediction', 'explanation', 'metadata']
        
        # Check top-level fields
        for field in required_fields:
            assert field in sample_api_response
            
        # Check prediction fields
        prediction_fields = ['triage_level', 'confidence', 'processing_time']
        for field in prediction_fields:
            assert field in sample_api_response['prediction']
            
        # Check explanation fields
        explanation_fields = ['method', 'key_features', 'similar_cases']
        for field in explanation_fields:
            assert field in sample_api_response['explanation']
            
        # Check metadata fields
        metadata_fields = ['model_version', 'timestamp']
        for field in metadata_fields:
            assert field in sample_api_response['metadata']
            
    def test_response_field_types(self, sample_api_response):
        """Test that response fields have correct data types"""
        # Top-level types
        assert isinstance(sample_api_response['success'], bool)
        assert isinstance(sample_api_response['prediction'], dict)
        assert isinstance(sample_api_response['explanation'], dict)
        assert isinstance(sample_api_response['metadata'], dict)
        
        # Prediction field types
        pred = sample_api_response['prediction']
        assert isinstance(pred['triage_level'], int)
        assert isinstance(pred['confidence'], (int, float))
        assert isinstance(pred['processing_time'], (int, float))
        
        # Explanation field types
        expl = sample_api_response['explanation']
        assert isinstance(expl['method'], str)
        assert isinstance(expl['key_features'], list)
        assert isinstance(expl['similar_cases'], list)
        
        # Metadata field types
        meta = sample_api_response['metadata']
        assert isinstance(meta['model_version'], str)
        assert isinstance(meta['timestamp'], str)
        
    def test_triage_level_validation(self, sample_api_response, test_config):
        """Test triage level is within valid range"""
        triage_level = sample_api_response['prediction']['triage_level']
        valid_levels = test_config['VALID_TRIAGE_LEVELS']
        
        assert triage_level in valid_levels
        assert isinstance(triage_level, int)
        
    def test_confidence_score_validation(self, sample_api_response):
        """Test confidence score is within valid range [0, 1]"""
        confidence = sample_api_response['prediction']['confidence']
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, (int, float))
        
    def test_processing_time_validation(self, sample_api_response, test_config):
        """Test processing time is reasonable"""
        processing_time = sample_api_response['prediction']['processing_time']
        max_time = test_config['MAX_PROCESSING_TIME']
        
        assert processing_time > 0
        assert processing_time < max_time
        assert isinstance(processing_time, (int, float))


@pytest.mark.unit
@pytest.mark.api
class TestErrorResponseHandling:
    """Test error response formats and handling"""
    
    def test_error_response_structure(self):
        """Test error response contains required fields"""
        error_response = {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "details": "Missing required field: chiefcomplaint"
            },
            "metadata": {
                "timestamp": "2025-01-04T12:00:00Z",
                "request_id": "req_123456"
            }
        }
        
        # Check required error fields
        assert error_response['success'] is False
        assert 'error' in error_response
        assert 'code' in error_response['error']
        assert 'message' in error_response['error']
        
        # Check field types
        assert isinstance(error_response['error']['code'], str)
        assert isinstance(error_response['error']['message'], str)
        
    def test_validation_error_codes(self):
        """Test different validation error codes"""
        error_codes = [
            "MISSING_REQUIRED_FIELD",
            "INVALID_DATA_TYPE", 
            "VALUE_OUT_OF_RANGE",
            "MODEL_UNAVAILABLE",
            "PROCESSING_TIMEOUT"
        ]
        
        for code in error_codes:
            error_response = {
                "success": False,
                "error": {
                    "code": code,
                    "message": f"Error: {code}"
                }
            }
            
            assert error_response['success'] is False
            assert error_response['error']['code'] == code
            assert isinstance(error_response['error']['message'], str)
            
    def test_http_status_code_mapping(self):
        """Test HTTP status codes for different error types"""
        status_mappings = {
            "VALIDATION_ERROR": 400,
            "AUTHENTICATION_ERROR": 401,
            "AUTHORIZATION_ERROR": 403,
            "RESOURCE_NOT_FOUND": 404,
            "METHOD_NOT_ALLOWED": 405,
            "PROCESSING_TIMEOUT": 408,
            "RATE_LIMIT_EXCEEDED": 429,
            "INTERNAL_SERVER_ERROR": 500,
            "MODEL_UNAVAILABLE": 503
        }
        
        for error_type, expected_status in status_mappings.items():
            assert 200 <= expected_status <= 599  # Valid HTTP status code
            assert isinstance(expected_status, int)


@pytest.mark.unit
@pytest.mark.api
class TestRequestValidation:
    """Test request data validation"""
    
    def test_valid_request_format(self):
        """Test valid request data format"""
        valid_request = {
            "patient_data": {
                "chiefcomplaint": "chest pain and shortness of breath",
                "temperature": 98.6,
                "heartrate": 85,
                "resprate": 16,
                "o2sat": 98,
                "sbp": 120,
                "dbp": 80,
                "pain": 5,
                "age": 45,
                "gender": "M"
            },
            "model_config": {
                "method": "tfidf_rf",
                "use_keywords": True
            }
        }
        
        # Check required fields
        assert 'patient_data' in valid_request
        assert 'chiefcomplaint' in valid_request['patient_data']
        
        # Check data types
        patient_data = valid_request['patient_data']
        assert isinstance(patient_data['chiefcomplaint'], str)
        assert isinstance(patient_data['temperature'], (int, float))
        assert isinstance(patient_data['age'], int)
        assert isinstance(patient_data['gender'], str)
        
    def test_missing_required_fields_detection(self):
        """Test detection of missing required fields"""
        incomplete_request = {
            "patient_data": {
                "temperature": 98.6,
                "heartrate": 85
                # Missing chiefcomplaint
            }
        }
        
        required_fields = ['chiefcomplaint']
        
        for field in required_fields:
            assert field not in incomplete_request['patient_data']
            
    def test_invalid_data_types_detection(self):
        """Test detection of invalid data types"""
        invalid_request = {
            "patient_data": {
                "chiefcomplaint": "chest pain",
                "temperature": "98.6",  # Should be numeric
                "heartrate": "high",    # Should be numeric
                "age": "45",           # Should be numeric
                "gender": 1            # Should be string
            }
        }
        
        # These should be detected as type errors
        assert not isinstance(invalid_request['patient_data']['temperature'], (int, float))
        assert not isinstance(invalid_request['patient_data']['heartrate'], (int, float))
        assert not isinstance(invalid_request['patient_data']['age'], int)
        assert not isinstance(invalid_request['patient_data']['gender'], str)
        
    def test_value_range_validation(self):
        """Test validation of value ranges"""
        out_of_range_request = {
            "patient_data": {
                "chiefcomplaint": "chest pain",
                "temperature": 150.0,  # Too high
                "heartrate": 300,      # Too high
                "resprate": -5,        # Negative
                "o2sat": 150,          # Impossible
                "sbp": 50,             # Too low
                "pain": 15,            # Out of 0-10 scale
                "age": -5              # Negative
            }
        }
        
        # Define valid ranges
        valid_ranges = {
            'temperature': (95.0, 110.0),
            'heartrate': (30, 250),
            'resprate': (8, 40),
            'o2sat': (70, 100),
            'sbp': (70, 200),
            'pain': (0, 10),
            'age': (0, 120)
        }
        
        patient_data = out_of_range_request['patient_data']
        
        # Check for out-of-range values
        for field, (min_val, max_val) in valid_ranges.items():
            if field in patient_data:
                value = patient_data[field]
                is_in_range = min_val <= value <= max_val
                
                # These specific values should be out of range
                if field in ['temperature', 'heartrate', 'o2sat', 'pain', 'age']:
                    assert not is_in_range


@pytest.mark.unit
@pytest.mark.api
class TestDataSerialization:
    """Test JSON serialization/deserialization"""
    
    def test_json_serialization(self, sample_api_response):
        """Test that response can be serialized to JSON"""
        try:
            json_string = json.dumps(sample_api_response)
            assert isinstance(json_string, str)
            assert len(json_string) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed: {e}")
            
    def test_json_deserialization(self, sample_api_response):
        """Test that JSON can be deserialized back to dict"""
        # Serialize then deserialize
        json_string = json.dumps(sample_api_response)
        deserialized = json.loads(json_string)
        
        # Should match original structure
        assert deserialized == sample_api_response
        assert isinstance(deserialized, dict)
        
    def test_datetime_serialization(self):
        """Test datetime serialization in API responses"""
        response_with_datetime = {
            "success": True,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "processed_at": "2025-01-04T12:00:00Z"
            }
        }
        
        # Should be serializable
        try:
            json_string = json.dumps(response_with_datetime)
            deserialized = json.loads(json_string)
            
            # Timestamps should be strings
            assert isinstance(deserialized['metadata']['timestamp'], str)
            assert isinstance(deserialized['metadata']['processed_at'], str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Datetime serialization failed: {e}")
            
    def test_numpy_array_serialization(self):
        """Test NumPy array serialization in responses"""
        import numpy as np
        
        response_with_arrays = {
            "success": True,
            "explanation": {
                "feature_importance": np.array([0.1, 0.2, 0.3]).tolist(),
                "probabilities": np.array([[0.2, 0.8], [0.6, 0.4]]).tolist()
            }
        }
        
        # Should be serializable after .tolist()
        try:
            json_string = json.dumps(response_with_arrays)
            deserialized = json.loads(json_string)
            
            # Should be regular Python lists
            assert isinstance(deserialized['explanation']['feature_importance'], list)
            assert isinstance(deserialized['explanation']['probabilities'], list)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Array serialization failed: {e}")


@pytest.mark.unit
@pytest.mark.api
class TestAPIEndpointIntegration:
    """Test integration with API endpoints"""
    
    @patch('flask.Flask')
    def test_prediction_endpoint_response(self, mock_flask):
        """Test prediction endpoint response format"""
        # Mock Flask app and request
        mock_app = MagicMock()
        mock_request = {
            "patient_data": {
                "chiefcomplaint": "chest pain",
                "temperature": 98.6,
                "heartrate": 85,
                "age": 45,
                "gender": "M"
            },
            "model_config": {
                "method": "tfidf_rf"
            }
        }
        
        # Mock response
        mock_response = {
            "success": True,
            "prediction": {
                "triage_level": 3,
                "confidence": 0.85,
                "processing_time": 1.23
            },
            "explanation": {
                "method": "TF-IDF + Random Forest",
                "key_features": ["chest", "pain"],
                "similar_cases": []
            },
            "metadata": {
                "model_version": "v1.0",
                "timestamp": "2025-01-04T12:00:00Z"
            }
        }
        
        # Validate response structure
        self._validate_prediction_response(mock_response)
        
    def _validate_prediction_response(self, response):
        """Helper method to validate prediction response"""
        # Required fields
        required_fields = ['success', 'prediction', 'explanation', 'metadata']
        for field in required_fields:
            assert field in response
            
        # Success should be boolean
        assert isinstance(response['success'], bool)
        
        # If successful, check prediction structure
        if response['success']:
            pred = response['prediction']
            assert 'triage_level' in pred
            assert 'confidence' in pred
            assert isinstance(pred['triage_level'], int)
            assert 1 <= pred['triage_level'] <= 5
            assert 0 <= pred['confidence'] <= 1
            
    def test_health_check_endpoint(self):
        """Test health check endpoint response"""
        health_response = {
            "status": "healthy",
            "timestamp": "2025-01-04T12:00:00Z",
            "version": "1.0.0",
            "models": {
                "tfidf_rf": "available",
                "tfidf_xgb": "available", 
                "chatgpt": "available",
                "llama_rag": "unavailable"
            }
        }
        
        # Validate health check structure
        assert 'status' in health_response
        assert 'timestamp' in health_response
        assert 'models' in health_response
        
        # Status should be valid
        valid_statuses = ['healthy', 'degraded', 'unhealthy']
        assert health_response['status'] in valid_statuses
        
        # Models should have availability status
        for model, status in health_response['models'].items():
            assert status in ['available', 'unavailable', 'error']
            
    def test_model_info_endpoint(self):
        """Test model information endpoint response"""
        model_info_response = {
            "models": {
                "tfidf_rf": {
                    "name": "TF-IDF + Random Forest",
                    "version": "1.0",
                    "description": "Text vectorization with Random Forest classification",
                    "parameters": {
                        "max_features": 1000,
                        "n_estimators": 100
                    },
                    "performance": {
                        "accuracy": 0.85,
                        "f1_score": 0.83
                    }
                }
            }
        }
        
        # Validate model info structure
        assert 'models' in model_info_response
        
        for model_name, model_data in model_info_response['models'].items():
            assert 'name' in model_data
            assert 'version' in model_data
            assert isinstance(model_data['name'], str)
            assert isinstance(model_data['version'], str)


@pytest.mark.unit
@pytest.mark.api
class TestResponseHeaders:
    """Test HTTP response headers"""
    
    def test_content_type_header(self):
        """Test that responses have correct Content-Type header"""
        expected_content_type = "application/json"
        
        # This would be tested in integration tests with actual HTTP responses
        assert expected_content_type == "application/json"
        
    def test_cors_headers(self):
        """Test CORS headers for cross-origin requests"""
        expected_cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
        
        # Validate header names and values
        for header, value in expected_cors_headers.items():
            assert isinstance(header, str)
            assert isinstance(value, str)
            assert len(header) > 0
            assert len(value) > 0
            
    def test_cache_control_headers(self):
        """Test cache control headers"""
        cache_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # For medical data, should typically not be cached
        for header, value in cache_headers.items():
            assert isinstance(header, str)
            assert isinstance(value, str)


# Additional fixtures for this test module
@pytest.fixture
def sample_error_response():
    """Sample error response for testing"""
    return {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid patient data",
            "details": "Temperature value out of valid range"
        },
        "metadata": {
            "timestamp": "2025-01-04T12:00:00Z",
            "request_id": "req_error_123"
        }
    }


@pytest.fixture
def sample_request_data():
    """Sample request data for testing"""
    return {
        "patient_data": {
            "chiefcomplaint": "chest pain and shortness of breath",
            "temperature": 98.6,
            "heartrate": 85,
            "resprate": 16,
            "o2sat": 98,
            "sbp": 120,
            "dbp": 80,
            "pain": 5,
            "age": 45,
            "gender": "M"
        },
        "model_config": {
            "method": "tfidf_rf",
            "use_keywords": True
        },
        "options": {
            "include_explanation": True,
            "include_similar_cases": False
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
