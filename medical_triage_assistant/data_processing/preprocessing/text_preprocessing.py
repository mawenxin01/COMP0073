"""
Text Preprocessing Functions
===========================

Functions for cleaning and preprocessing medical text data.
"""

import re
import string
from typing import Optional


def expand_medical_abbreviations(text: Optional[str]) -> str:
    """
    Expand common medical abbreviations in text.
    
    Args:
        text: Input text containing medical abbreviations
        
    Returns:
        Text with abbreviations expanded
    """
    if not text:
        return ""
    
    # Common medical abbreviations mapping
    abbreviations = {
        r'\bc/o\b': 'complains of',
        r'\bSOB\b': 'shortness of breath',
        r'\bDOE\b': 'dyspnea on exertion',
        r'\bN&V\b': 'nausea and vomiting',
        r'\bn&v\b': 'nausea and vomiting',
        r'\br/o\b': 'rule out',
        r'\bR/O\b': 'rule out',
        r'\bs/p\b': 'status post',
        r'\bS/P\b': 'status post',
        r'\bw/': 'with',  # Remove word boundary to match "w/ fever"
        r'\bW/': 'with',
        r'\bpt\b': 'patient',
        r'\bPt\b': 'patient',
        r'\bPT\b': 'patient',
        r'\babd\b': 'abdominal',
        r'\bAbd\b': 'abdominal',
        r'\bh/a\b': 'headache',
        r'\bH/A\b': 'headache',
        r'\bfx\b': 'fracture',
        r'\bFx\b': 'fracture',
        r'\bMI\b': 'myocardial infarction',
        r'\bx\s+(\d+)\s+days?\b': r'times \1 days',
        r'\bx\s+(\d+)\b': r'times \1'
    }
    
    expanded_text = text
    for abbrev, expansion in abbreviations.items():
        expanded_text = re.sub(abbrev, expansion, expanded_text, flags=re.IGNORECASE)
    
    return expanded_text


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    cleaned = text.lower()
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Remove excessive punctuation (keep single instances)
    cleaned = re.sub(r'([!?.]){2,}', r'\1', cleaned)
    
    return cleaned


def normalize_chief_complaint(text: Optional[str]) -> str:
    """
    Normalize chief complaint text through complete preprocessing pipeline.
    
    Args:
        text: Raw chief complaint text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Apply abbreviation expansion
    expanded = expand_medical_abbreviations(text)
    
    # Apply text cleaning
    cleaned = clean_text(expanded)
    
    return cleaned
