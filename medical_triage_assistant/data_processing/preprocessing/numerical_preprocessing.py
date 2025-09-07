"""
Numerical Data Preprocessing Functions
=====================================

Functions for handling numerical medical data preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from scipy import stats


def detect_outliers(data: Union[pd.Series, np.ndarray, List], 
                   method: str = 'iqr', 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in numerical data.
    
    Args:
        data: Input numerical data
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Array of outlier indices
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.values
    
    # Remove NaN values for calculation
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return np.array([])
    
    if method == 'iqr':
        Q1 = np.percentile(clean_data, 25)
        Q3 = np.percentile(clean_data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(clean_data, nan_policy='omit'))
        # Map z_scores back to original data indices
        outlier_mask = np.zeros(len(data), dtype=bool)
        clean_indices = ~np.isnan(data)
        outlier_mask[clean_indices] = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.where(outlier_mask)[0]


def validate_vital_signs(vital_name: str, value: Union[int, float]) -> bool:
    """
    Validate if a vital sign value is within reasonable range.
    
    Args:
        vital_name: Name of the vital sign
        value: Value to validate
        
    Returns:
        True if value is valid, False otherwise
    """
    if pd.isna(value):
        return False
    
    # Define valid ranges for common vital signs
    valid_ranges = {
        'temperature': (95.0, 110.0),  # Fahrenheit
        'heartrate': (30, 250),
        'resprate': (8, 40),
        'o2sat': (70, 100),
        'sbp': (70, 200),  # Systolic blood pressure
        'dbp': (40, 120),  # Diastolic blood pressure
        'pain': (0, 10),
        'age': (0, 120)
    }
    
    if vital_name.lower() not in valid_ranges:
        return True  # Unknown vital signs pass validation
    
    min_val, max_val = valid_ranges[vital_name.lower()]
    return min_val <= value <= max_val


def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'median',
                         fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in numerical data.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'fill')
        fill_value: Value to use if strategy is 'fill'
        
    Returns:
        DataFrame with missing values handled
    """
    df_filled = df.copy()
    
    # Get numerical columns
    numerical_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if df_filled[col].isnull().any():
            if strategy == 'median':
                fill_val = df_filled[col].median()
            elif strategy == 'mean':
                fill_val = df_filled[col].mean()
            elif strategy == 'mode':
                fill_val = df_filled[col].mode().iloc[0] if not df_filled[col].mode().empty else 0
            elif strategy == 'fill':
                fill_val = fill_value if fill_value is not None else 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col].fillna(fill_val, inplace=True)
    
    # Handle text columns with empty strings
    text_cols = df_filled.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_filled[col].fillna('', inplace=True)
    
    return df_filled

