#!/usr/bin/env python3
"""
Llama-3.3-70B Generation Module
Uses IBM Watson's Llama-3.3-70B to generate triage decisions based on retrieved similar cases
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

# Dynamically import IBM Watson configuration
try:
    import importlib.util
    config_path = os.path.join(os.path.dirname(__file__), '../../../config/azure_simple.py')
    if os.path.exists(config_path):
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        IBM_API_KEY = getattr(config, 'IBM_API_KEY', None)
        IBM_ENDPOINT = getattr(config, 'IBM_ENDPOINT', None)
        IBM_PROJECT_ID = getattr(config, 'IBM_PROJECT_ID', None)
    else:
        print("⚠️ Warning: config/azure_simple.py not found")
        IBM_API_KEY = None
        IBM_ENDPOINT = None
        IBM_PROJECT_ID = None
except Exception as e:
    print(f"❌ Failed to import configuration: {e}")
    IBM_API_KEY = None
    IBM_ENDPOINT = None
    IBM_PROJECT_ID = None

try:
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Meta
    from ibm_watsonx_ai import Credentials
except ImportError:
    print("❌ ibm_watsonx_ai package not installed, please run: pip install ibm-watsonx-ai")
    ModelInference = None
    Meta = None
    Credentials = None


class LlamaGenerator:
    """Llama-3.3-70B Triage Decision Generator"""
    
    def __init__(self, model_name: str = "meta-llama/llama-3-3-70b-instruct"):
        """Initialize Llama generator"""
        
        self.model_name = model_name
        self.llama_model = None
        
        # Error statistics and retry mechanism
        self.error_stats = {
            'config_errors': 0,
            'network_errors': 0,
            'api_errors': 0,
            'response_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial retry delay
        
        # Initialize Llama client
        self._initialize_llama_client()
    
    def _initialize_llama_client(self):
        """Initialize Llama client"""
        
        if not all([IBM_API_KEY, IBM_ENDPOINT, IBM_PROJECT_ID]):
            print("❌ IBM Watson configuration incomplete, cannot initialize Llama client")
            print("   Please ensure config/azure_simple.py contains necessary IBM configuration")
            return
        
        if ModelInference is None:
            print("❌ IBM Watson package not installed, cannot initialize Llama client")
            return
        
        try:
            credentials = Credentials(api_key=IBM_API_KEY, url=IBM_ENDPOINT)
            
            self.llama_model = ModelInference(
                model_id=self.model_name,
                credentials=credentials,
                project_id=IBM_PROJECT_ID,
                params={Meta.MAX_NEW_TOKENS: 500, Meta.TEMPERATURE: 0.1}
            )
            print(f"✅ Llama client initialized successfully")
            print(f"   Endpoint: {IBM_ENDPOINT}")
            print(f"   Project ID: {IBM_PROJECT_ID}")
            print(f"   Model: {self.model_name}")
        except Exception as e:
            print(f"❌ Llama client initialization failed: {e}")
            self.llama_model = None
    
    def create_case_description(self, row: pd.Series, include_acuity: bool = True) -> str:
        """Create textual description of medical case"""
        
        description = f"Chief Complaint: {row.get('chiefcomplaint', row.get('complaint_keywords', 'Unknown'))}\n"
        description += f"Age: {row.get('age_at_visit', 'Unknown')} years\n"
        description += f"Gender: {row.get('gender', 'Unknown')}\n"
        
        # Vital signs
        description += "Vital Signs:\n"
        description += f"- Heart Rate: {row.get('heartrate', 'Unknown')} bpm\n"
        description += f"- Blood Pressure: {row.get('sbp', 'Unknown')}/{row.get('dbp', 'Unknown')} mmHg\n"
        description += f"- Temperature: {row.get('temperature', 'Unknown')} °F\n"
        description += f"- Oxygen Saturation: {row.get('o2sat', 'Unknown')}%\n"
        description += f"- Pain Score: {row.get('pain', 'Unknown')}/10\n"
        
        # Arrival method
        if 'arrival_transport' in row and pd.notna(row['arrival_transport']):
            description += f"Arrival Method: {row['arrival_transport']}\n"
        
        # Time period information
        if 'time_period' in row and pd.notna(row['time_period']):
            time_period_map = {0: "Night (0-6AM)", 1: "Morning (6AM-12PM)", 
                             2: "Afternoon (12PM-6PM)", 3: "Evening (6PM-12AM)"}
            time_period_name = time_period_map.get(int(row['time_period']), "Unknown")
            description += f"Arrival Time Period: {time_period_name}\n"
            
        # Symptom keywords
        if 'complaint_keywords' in row and pd.notna(row['complaint_keywords']):
            description += f"Symptom Keywords: {row['complaint_keywords']}\n"
            
        # Actual acuity (only include for few-shot examples, not for current patient)
        if include_acuity and 'acuity' in row and pd.notna(row['acuity']):
            description += f"Actual Triage Level: ESI {row['acuity']}\n"
            
        return description
    
    def create_llama_prompt(self, current_case: pd.Series, similar_cases: List[Dict], 
                           similarities: List[float]) -> str:
        """Create Llama prompt"""
        
        # Current patient information (without acuity to prevent data leakage)
        current_description = self.create_case_description(current_case, include_acuity=False)
        
        # Build similar cases information (with acuity for few-shot learning)
        similar_cases_info = ""
        for i, (case, sim) in enumerate(zip(similar_cases, similarities)):
            case_desc = self.create_case_description(pd.Series(case), include_acuity=True)
            similar_cases_info += f"\nSimilar Case {i+1} (Similarity: {sim:.3f}):\n{case_desc}\n"
        
        prompt = f"""You are an experienced emergency medicine doctor. Based on patient information and similar historical cases, perform triage assessment.

Current Patient Information:
{current_description}

Similar Historical Cases for Reference:
{similar_cases_info}

Based on the above information, please assess the current patient for triage. Triage level explanations:
- Level 1 (Critical): Requires immediate resuscitation, unstable vital signs
- Level 2 (Emergent): Requires urgent treatment, but vital signs relatively stable  
- Level 3 (Urgent): Requires timely treatment, condition relatively stable
- Level 4 (Less Urgent): Can wait for treatment
- Level 5 (Non-urgent): Can wait for treatment

Please provide your assessment in the following format:
TRIAGE LEVEL: [number 1-5]
REASONING: [brief clinical reasoning in 2-3 sentences]"""

        return prompt
    
    def generate_triage_decision(self, current_case: pd.Series, similar_cases: List[Dict], 
                               similarities: List[float]) -> Tuple[int, str]:
        """
        Generate triage decision
        
        Args:
            current_case: Current patient case
            similar_cases: List of similar cases
            similarities: List of similarity scores
            
        Returns:
            Tuple[int, str]: Triage level and decision reasoning
        """
        
        # Create prompt
        prompt = self.create_llama_prompt(current_case, similar_cases, similarities)
        
        # Call Llama model
        return self._call_llama_with_retry(prompt)
    
    def _call_llama_with_retry(self, prompt: str) -> Tuple[int, str]:
        """Llama API call with retry mechanism"""
        
        self.error_stats['total_requests'] += 1
        
        for attempt in range(self.max_retries):
            try:
                # Layer 1: Configuration error handling
                if self.llama_model is None:
                    self.error_stats['config_errors'] += 1
                    print(f"❌ Configuration error: Llama client not initialized")
                    return 3, "Error: Llama client not initialized"
                
                # Layer 2: Network error handling
                response = self.llama_model.generate_text(prompt=prompt)
                
                # Layer 3: API error handling (response status check)
                if not response or response.strip() == "":
                    self.error_stats['api_errors'] += 1
                    print(f"❌ API error: empty response")
                    if attempt < self.max_retries - 1:
                        print(f"🔄 Retry {attempt + 1}/{self.max_retries}")
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    return 3, "Empty response from Llama"
                
                response_text = response.strip()
                
                # Layer 4: Response error handling (content validation)
                import re
                numbers = re.findall(r'\b[1-5]\b', response_text)
                
                if numbers:
                    decision = int(numbers[0])
                    self.error_stats['successful_requests'] += 1
                    return decision, response_text
                else:
                    self.error_stats['response_errors'] += 1
                    print(f"⚠️ Response error: abnormal Llama return format - {response_text}")
                    if attempt < self.max_retries - 1:
                        print(f"🔄 Retry {attempt + 1}/{self.max_retries}")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return 3, f"Invalid Response: {response_text}"
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Network errors: connection timeout, network issues
                if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'unreachable']):
                    self.error_stats['network_errors'] += 1
                    print(f"❌ Network error: {e}")
                    if attempt < self.max_retries - 1:
                        print(f"🔄 Retry {attempt + 1}/{self.max_retries}")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return 3, f"Network Error: {e}"
                
                # API errors: rate limit, quota exceeded, authentication failed
                elif any(keyword in error_msg for keyword in ['rate limit', 'quota', '429', '401', '403']):
                    self.error_stats['api_errors'] += 1
                    print(f"❌ API error: {e}")
                    if attempt < self.max_retries - 1:
                        print(f"🔄 Retry {attempt + 1}/{self.max_retries}")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return 3, f"API Error: {e}"
                
                # Other unknown errors
                else:
                    print(f"❌ Unknown error: {e}")
                    if attempt < self.max_retries - 1:
                        print(f"🔄 Retry {attempt + 1}/{self.max_retries}")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return 3, f"Unknown Error: {e}"
        
        # All retries failed
        return 3, "Error: All retries failed"
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics"""
        
        total_errors = (self.error_stats['config_errors'] + 
                       self.error_stats['network_errors'] + 
                       self.error_stats['api_errors'] + 
                       self.error_stats['response_errors'])
        
        success_rate = (self.error_stats['successful_requests'] / 
                       self.error_stats['total_requests']) if self.error_stats['total_requests'] > 0 else 0
        
        return {
            'total_requests': self.error_stats['total_requests'],
            'successful_requests': self.error_stats['successful_requests'],
            'success_rate': success_rate,
            'config_errors': self.error_stats['config_errors'],
            'network_errors': self.error_stats['network_errors'],
            'api_errors': self.error_stats['api_errors'],
            'response_errors': self.error_stats['response_errors'],
            'total_errors': total_errors
        }