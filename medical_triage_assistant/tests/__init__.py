"""
Medical Triage Assistant - Test Suite
====================================

Comprehensive unit and integration tests for the medical triage system.

Test Categories:
- Data Preprocessing: Text cleaning, abbreviation expansion, outlier detection
- Feature Extraction: TF-IDF, SVD dimensionality validation
- Model Predictions: Output range validation, performance consistency
- API Validation: Response format, error handling
"""

import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
