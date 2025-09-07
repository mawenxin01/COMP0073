#!/usr/bin/env python3
"""
Step 4: Two-stage symptom classification
First stage: Symptom group classification
Second stage: Severity assessment
Use Llama-3.3-70B to improve classification accuracy
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import sys
from datetime import datetime
import time
from tqdm import tqdm
import argparse

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../..')
sys.path.insert(0, project_root)

# 导入Llama生成器
try:
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Meta
    from ibm_watsonx_ai import Credentials
except ImportError:
    print("❌ 未安装ibm_watsonx_ai包，请运行: pip install ibm-watsonx-ai")
    ModelInference = None
    Meta = None
    Credentials = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoStageSymptomClassifier:
    """Two-stage symptom classifier"""
    
    def __init__(self):
        """Initialize classifier"""
        # Disable HTTP request logging
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        
        # Get script directory, then build data directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Parent directory is project root
        
        self.data_dir = os.path.join(script_dir, "data")
        self.output_dir = os.path.join(self.data_dir, "two_stage_classification")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化Llama客户端
        self._initialize_llama_client()
        
        # Load symptom group standards from the specific file
        self.symptom_groups = self.load_symptom_groups_from_txt()
        self.severity_rules = self.load_severity_rules()
        
        logger.info(f"✅ Using symptom groups from grouping_summary_20250831_163519.txt")
        
        logger.info(f"✅ Two-stage symptom classifier initialized")
        logger.info(f"🤖 Using model: Llama-3.3-70B")
    
    def _initialize_llama_client(self):
        """初始化Llama客户端"""
        try:
            # 动态导入IBM Watson配置
            import importlib.util
            config_path = os.path.join(project_root, 'config/azure_simple.py')
            if os.path.exists(config_path):
                spec = importlib.util.spec_from_file_location("config", config_path)
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                
                IBM_API_KEY = getattr(config, 'IBM_API_KEY', None)
                IBM_ENDPOINT = getattr(config, 'IBM_ENDPOINT', None)
                IBM_PROJECT_ID = getattr(config, 'IBM_PROJECT_ID', None)
                IBM_MODEL_NAME = getattr(config, 'IBM_MODEL_NAME', 'meta-llama/llama-3-3-70b-instruct')
            else:
                print("⚠️ 警告: config/azure_simple.py 未找到")
                return
            
            if not all([IBM_API_KEY, IBM_ENDPOINT, IBM_PROJECT_ID]):
                print("❌ IBM Watson配置不完整，无法初始化Llama客户端")
                self.llama_model = None
                return
            
            if ModelInference is None:
                print("❌ IBM Watson包未安装，无法初始化Llama客户端")
                self.llama_model = None
                return
            
            credentials = Credentials(api_key=IBM_API_KEY, url=IBM_ENDPOINT)
            
            self.llama_model = ModelInference(
                model_id=IBM_MODEL_NAME,
                credentials=credentials,
                project_id=IBM_PROJECT_ID,
                params={Meta.MAX_NEW_TOKENS: 300, Meta.TEMPERATURE: 0.1}
            )
            self.model_name = IBM_MODEL_NAME
            print(f"✅ Llama客户端初始化成功")
            
        except Exception as e:
            print(f"❌ Llama客户端初始化失败: {e}")
            self.llama_model = None
            self.model_name = "Llama-3.3-70B (Failed to initialize)"
    
    def load_symptom_groups_from_txt(self):
        """Load symptom groups from the specific TXT file"""
        try:
            txt_file = os.path.join(self.data_dir, "symptom_groups", "grouping_summary_20250831_163519.txt")
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the TXT file content
            symptom_groups = []
            lines = content.split('\n')
            
            current_group = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a group header (e.g., "1. Abdominal Pain and Related Symptoms")
                if line[0].isdigit() and '. ' in line:
                    if current_group:
                        symptom_groups.append(current_group)
                    
                    # Extract group number and name
                    group_num = int(line.split('. ', 1)[0])
                    group_name = line.split('. ', 1)[1]
                    current_group = {
                        'group_id': group_num,
                        'group_name': group_name,
                        'name': group_name,
                        'description': '',
                        'severity_range': [],
                        'included_clusters': [],
                        'typical_symptoms': []
                    }
                
                elif current_group:
                    if line.startswith('Description:'):
                        current_group['description'] = line.split('Description: ', 1)[1]
                    elif line.startswith('Severity range:'):
                        severity_str = line.split('Severity range: ', 1)[1]
                        # Parse [2, 5] format
                        severity_str = severity_str.strip('[]')
                        current_group['severity_range'] = [int(x.strip()) for x in severity_str.split(',')]
                    elif line.startswith('Included clusters:'):
                        clusters_str = line.split('Included clusters: ', 1)[1]
                        # Parse [0, 24, 35] format
                        clusters_str = clusters_str.strip('[]')
                        current_group['included_clusters'] = [int(x.strip()) for x in clusters_str.split(',')]
                    elif line.startswith('Typical symptoms:'):
                        symptoms_str = line.split('Typical symptoms: ', 1)[1]
                        current_group['typical_symptoms'] = [s.strip() for s in symptoms_str.split(',')]
            
            # Add the last group
            if current_group:
                symptom_groups.append(current_group)
            
            logger.info(f"✅ Loaded {len(symptom_groups)} symptom groups from TXT file")
            return symptom_groups
            
        except Exception as e:
            logger.error(f"❌ Failed to load symptom groups from TXT file: {e}")
            raise
    
    def load_symptom_groups(self):
        """Load symptom group definitions"""
        try:
            latest_file = os.path.join(self.data_dir, "symptom_groups", "latest_manual_grouping.json")
            with open(latest_file, 'r') as f:
                metadata = json.load(f)
            
            groups_file = metadata['groups_file']
            # If relative path, convert to absolute path
            if groups_file.startswith('../'):
                # Parse relative path from project root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                groups_file = os.path.join(project_root, groups_file[3:])  # Remove '../' prefix
            elif not os.path.isabs(groups_file):
                groups_file = os.path.join(self.data_dir, groups_file)
                
            with open(groups_file, 'r', encoding='utf-8') as f:
                groups_data = json.load(f)
            
            return groups_data['symptom_groups']
        
        except Exception as e:
            logger.error(f"❌ Failed to load symptom group definitions: {e}")
            raise
    
    def load_severity_rules(self):
        """Load severity rules - use default rules since step 03 no longer generates them"""
        try:
            # Try to load from external file first (if exists from old runs)
            latest_file = os.path.join(self.data_dir, "symptom_groups", "latest_manual_grouping.json")
            if os.path.exists(latest_file):
                with open(latest_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if rules_file exists in metadata (for backward compatibility)
                if 'rules_file' in metadata:
                    rules_file = metadata['rules_file']
                    # If relative path, convert to absolute path
                    if rules_file.startswith('../'):
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(script_dir)
                        rules_file = os.path.join(project_root, rules_file[3:])
                    elif not os.path.isabs(rules_file):
                        rules_file = os.path.join(self.data_dir, rules_file)
                        
                    if os.path.exists(rules_file):
                        with open(rules_file, 'r', encoding='utf-8') as f:
                            rules_data = json.load(f)
                        logger.info("✅ Loaded severity rules from external file")
                        return rules_data
            
            # Fall back to default rules (step 03 no longer generates severity rules)
            logger.info("📋 Using built-in default severity rules (step 03 no longer generates them)")
            return self._get_default_severity_rules()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load external severity rules: {e}")
            logger.info("📋 Using built-in default severity rules")
            return self._get_default_severity_rules()
    
    def _get_default_severity_rules(self):
        """Get default severity rules (consistent with hospital acuity system: 1 most severe, 5 least severe)"""
        return {
            "severity_levels": {
                "1": {
                    "name": "Critical",
                    "definition": "Life-threatening, highest priority, immediate treatment required",
                    "keywords": ["arrest", "unconscious", "critical", "emergency", "cardiac", "stroke"],
                    "wait_time": "Immediate"
                },
                "2": {
                    "name": "Severe", 
                    "definition": "Immediate treatment required, may be life-threatening",
                    "keywords": ["severe", "acute", "difficulty", "bleeding", "trauma"],
                    "wait_time": "Immediate"
                },
                "3": {
                    "name": "Moderate",
                    "definition": "Immediate treatment required, within 15 minutes", 
                    "keywords": ["pain", "nausea", "vomiting", "fever", "headache"],
                    "wait_time": "15 minutes"
                },
                "4": {
                    "name": "Mild",
                    "definition": "Requires attention but not urgent, can be handled within 30 minutes",
                    "keywords": ["sore", "ache", "discomfort", "concern", "worry"],
                    "wait_time": "30 minutes"
                },
                "5": {
                    "name": "Minor",
                    "definition": "Symptom is mild, can be delayed, not urgent",
                    "keywords": ["mild", "minor", "routine", "follow up", "check"],
                    "wait_time": "Can be delayed"
                }
            },
            "assessment_rules": {
                "keyword_scoring": "Based on keyword matching for preliminary scoring, then adjusted based on combination rules",
                "combination_rules": [
                    "Chest pain + Dyspnea = 1-2 levels",
                    "Change in consciousness = 1-2 levels",
                    "Severe trauma = 1-2 levels", 
                    "Cardiac symptoms = 2-3 levels"
                ],
                "special_cases": [
                    "cardiac arrest = 1 level",
                    "stroke symptoms = 1 level",
                    "major trauma = 1-2 levels",
                    "sepsis = 1-2 levels"
                ]
            }
        }
    
    def load_unique_complaints(self, limit=1000):
        """Load unique chief complaint data"""
        logger.info(f"📂 Loading unique chief complaint data (limit: {limit} records)...")
        
        try:
            # Load latest embedding data file
            latest_embeddings_file = os.path.join(self.data_dir, "embeddings", "latest_embeddings.json")
            with open(latest_embeddings_file, 'r') as f:
                embeddings_metadata = json.load(f)
            
            embedding_data_file = os.path.join(self.data_dir, "..", embeddings_metadata['data_file'])
            df = pd.read_csv(embedding_data_file)
            
            unique_complaints = df[['chiefcomplaint_clean']].drop_duplicates()
            
            if len(unique_complaints) > limit:
                unique_complaints = unique_complaints.head(limit)
            
            logger.info(f"✅ Loading completed, {len(unique_complaints)} unique chief complaints")
            return unique_complaints.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to load chief complaint data: {e}")
            raise
    
    def create_group_classification_prompt(self):
        """Create symptom group classification prompt"""
        groups_description = []
        for group in self.symptom_groups:
            group_desc = f"""
{group['group_id']}. {group['group_name']}
    Description: {group['description']}
    Typical symptoms: {', '.join(group['typical_symptoms'][:6])}
"""
            groups_description.append(group_desc)
        
        prompt = f"""You are a senior emergency department doctor. Please classify the given chief complaint based on the following 10 symptom group standards.

Symptom Group Standards:
{''.join(groups_description)}

Please carefully analyze the following chief complaint and select the most appropriate symptom group:

Chief Complaint: {{complaint}}

Please only return the symptom group number (1-10) and confidence (1-5):
Group: [Symptom Group Number]
Confidence: [1-5]

Example:
Group: 1
Confidence: 4
"""
        return prompt
    
    def create_severity_assessment_prompt(self):
        """Create severity assessment prompt"""
        severity_levels = self.severity_rules['severity_levels']
        
        levels_description = []
        for level, info in severity_levels.items():
            level_desc = f"""
{level}级-{info['name']}: {info['definition']}
    Keywords: {', '.join(info['keywords'][:5])}
    Treatment time: {info['wait_time']}
"""
            levels_description.append(level_desc)
        
        prompt = f"""You are an emergency triage expert. Please assess the severity of the given chief complaint based on the following severity standards.

Severity Standards (1 most severe, 5 least severe):
{''.join(levels_description)}

Please carefully assess the severity of the following chief complaint:

Chief Complaint: {{complaint}}
Symptom Group: {{group_name}}

Please only return the severity (1-5) and confidence (1-5):
Severity: [1-5]
Confidence: [1-5]

Example:
Severity: 2
Confidence: 4
"""
        return prompt
    
    def create_enhanced_severity_assessment_prompt(self, group_name, data_driven_range=None):
        """Create enhanced severity assessment prompt that uses data-driven ranges as guidance"""
        severity_levels = self.severity_rules['severity_levels']
        
        levels_description = []
        for level, info in severity_levels.items():
            level_desc = f"""
{level}级-{info['name']}: {info['definition']}
    Keywords: {', '.join(info['keywords'][:5])}
    Treatment time: {info['wait_time']}
"""
            levels_description.append(level_desc)
        
        # Add data-driven guidance if available (not constraint)
        guidance_note = ""
        if data_driven_range:
            guidance_note = f"""
CLINICAL GUIDANCE: Based on real hospital data for {group_name}, 
this symptom group typically receives severity levels {data_driven_range[0]}-{data_driven_range[1]}.
This is GUIDANCE to understand typical patterns, not a restriction. Use your clinical judgment:
- If presentation is more severe than typical, assign lower severity (1-2)
- If presentation is less severe than typical, assign higher severity (4-5)
- Most cases will likely fall in the typical range, but exceptions are important
"""
        
        prompt = f"""You are an emergency triage expert. Please assess the severity of the given chief complaint based on the following severity standards.

Severity Standards (1 most severe, 5 least severe):
{''.join(levels_description)}

{guidance_note}

Please carefully assess the severity of the following chief complaint:

Chief Complaint: {{complaint}}
Symptom Group: {{group_name}}

IMPORTANT: Use the hospital data as REFERENCE for typical patterns, but prioritize clinical assessment of the specific presentation. A severe presentation should get severity 1-2 even if the group typically sees 3-4.

Please only return the severity (1-5) and confidence (1-5):
Severity: [1-5]
Confidence: [1-5]

Example:
Severity: 2
Confidence: 4
"""
        return prompt
    
    def load_enhanced_symptom_groups(self):
        """Load enhanced symptom groups with data-driven severity ranges"""
        try:
            # Try to load enhanced groups first
            enhanced_file = os.path.join(self.data_dir, "symptom_groups", "latest_enhanced_grouping.json")
            if os.path.exists(enhanced_file):
                with open(enhanced_file, 'r') as f:
                    metadata = json.load(f)
                
                groups_file = metadata['groups_file']
                with open(groups_file, 'r', encoding='utf-8') as f:
                    enhanced_data = json.load(f)
                
                logger.info("✅ Loaded enhanced symptom groups with data-driven severity ranges")
                return enhanced_data['symptom_groups'], True
            
            # Fall back to regular groups
            return self.load_symptom_groups(), False
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load enhanced groups: {e}")
            return self.load_symptom_groups(), False
    
    def get_data_driven_range(self, group_id):
        """Get data-driven severity range for a group"""
        for group in self.symptom_groups:
                if group['group_id'] == group_id:
                  return group.get('severity_range')
        return None
    
    def classify_symptom_group(self, complaint, group_prompt_template):
        """First stage: Classify symptom group using Llama"""
        max_retries = 3
        base_delay = 1.0
        
        if self.llama_model is None:
            return {"group_id": 0, "confidence": 1, "error": "Llama model not initialized"}
        
        for attempt in range(max_retries):
            try:
                prompt = group_prompt_template.replace("{complaint}", complaint)
                
                response = self.llama_model.generate_text(prompt=prompt)
                
                result_text = response.strip()
                return self.parse_group_result(result_text)
                
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    # Handle API rate limit errors
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        return {"group_id": 0, "confidence": 1, "error": "API rate limit timeout"}
                else:
                    # Silently handle errors to keep output clean
                    return {"group_id": 0, "confidence": 1, "error": str(e)}
    
    def assess_severity(self, complaint, group_name, severity_prompt_template):
        """Second stage: Assess severity using Llama"""
        max_retries = 3
        base_delay = 1.0
        
        if self.llama_model is None:
            return {"severity_score": 3, "confidence": 1, "error": "Llama model not initialized"}
        
        for attempt in range(max_retries):
            try:
                prompt = severity_prompt_template.replace("{complaint}", complaint).replace("{group_name}", group_name)
                
                response = self.llama_model.generate_text(prompt=prompt)
                
                result_text = response.strip()
                return self.parse_severity_result(result_text)
                
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    # Handle API rate limit errors
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    if attempt == max_retries - 1:
                        return {"severity": 3, "confidence": 1, "error": "API rate limit timeout"}
                else:
                    # Silently handle errors to keep output clean
                    return {"severity": 3, "confidence": 1, "error": str(e)}
    
    def parse_group_result(self, result_text):
        """Parse symptom group classification result"""
        try:
            lines = result_text.split('\n')
            result = {"group_id": 0, "confidence": 1}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Group:'):
                    result['group_id'] = int(line.split(':')[1].strip())
                elif line.startswith('Confidence:'):
                    result['confidence'] = int(line.split(':')[1].strip())
            
            return result
            
        except Exception as e:
            # Silently handle parse errors to keep output clean
            return {"group_id": 0, "confidence": 1, "parse_error": str(e)}
    
    def parse_severity_result(self, result_text):
        """Parse severity assessment result"""
        try:
            lines = result_text.split('\n')
            result = {"severity": 3, "confidence": 1}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Severity:'):
                    result['severity'] = int(line.split(':')[1].strip())
                elif line.startswith('Confidence:'):
                    result['confidence'] = int(line.split(':')[1].strip())
            
            return result
            
        except Exception as e:
            # Silently handle parse errors to keep output clean
            return {"severity": 3, "confidence": 1, "parse_error": str(e)}
    
    def get_group_name(self, group_id):
        """Get group name by group ID"""
        for group in self.symptom_groups:
            if group['group_id'] == group_id:
                return group['group_name']
        return "Unclassified"
    
    def classify_two_stage(self, complaints_df, batch_size=100, start_index=0, timestamp=None, existing_results=None):
        """Process two-stage classification"""
        logger.info(f"🎯 Starting classification: {len(complaints_df)} records (from index {start_index})")
        
        # Prepare prompt templates
        group_prompt_template = self.create_group_classification_prompt()
        
        # Use standard severity assessment
        severity_prompt_template = self.create_severity_assessment_prompt()
        
        # Initialize results and timestamp
        if existing_results:
            results = existing_results
            logger.info(f"🔄 Resuming from {len(existing_results)} processed records")
        else:
            results = []
        
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process one by one
        for idx in tqdm(range(start_index, len(complaints_df)), 
                       desc="Classifying", 
                       total=len(complaints_df),
                       initial=start_index):
            complaint = complaints_df.iloc[idx]['chiefcomplaint_clean']
            
            # First stage: Symptom group classification
            group_result = self.classify_symptom_group(complaint, group_prompt_template)
            group_id = group_result['group_id']
            group_name = self.get_group_name(group_id)
            
            # Second stage: Severity assessment
            severity_result = self.assess_severity(complaint, group_name, severity_prompt_template)
            
            # Combine results
            result = {
                'index': idx,
                'chiefcomplaint_clean': complaint,
                'group_id': group_id,
                'group_name': group_name,
                'group_confidence': group_result['confidence'],
                'severity': severity_result['severity'],
                'severity_confidence': severity_result['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add error messages
            if 'error' in group_result:
                result['group_error'] = group_result['error']
            if 'parse_error' in group_result:
                result['group_parse_error'] = group_result['parse_error']
            if 'error' in severity_result:
                result['severity_error'] = severity_result['error']
            if 'parse_error' in severity_result:
                result['severity_parse_error'] = severity_result['parse_error']
            
            results.append(result)
            
            # Save progress periodically
            if (idx + 1) % batch_size == 0 or idx == len(complaints_df) - 1:
                self.save_progress(results, timestamp, idx + 1)
                
                # Add a short delay to avoid API rate limits
                time.sleep(0.1)
        
        return results, timestamp
    
    def save_progress(self, results, timestamp, processed_count):
        """Save progress"""
        results_df = pd.DataFrame(results)
        results_file = os.path.join(self.output_dir, f"two_stage_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        progress_metadata = {
            'timestamp': timestamp,
            'processed_count': processed_count,
            'model_used': self.model_name,
            'results_file': results_file,
            'total_groups': len(self.symptom_groups),
            'method': 'two_stage'
        }
        
        metadata_file = os.path.join(self.output_dir, f"progress_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(progress_metadata, f, indent=2)
    
    def generate_classification_summary(self, results, timestamp):
        """Generate classification summary report"""
        results_df = pd.DataFrame(results)
        
        # Statistical analysis
        group_counts = results_df['group_id'].value_counts().sort_index()
        severity_counts = results_df['severity'].value_counts().sort_index()
        group_confidence_stats = results_df['group_confidence'].describe()
        severity_confidence_stats = results_df['severity_confidence'].describe()
        
        # Error statistics
        group_errors = results_df['group_error'].notna().sum() if 'group_error' in results_df.columns else 0
        severity_errors = results_df['severity_error'].notna().sum() if 'severity_error' in results_df.columns else 0
        
        # Generate report
        summary_file = os.path.join(self.output_dir, f"two_stage_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Two-stage symptom classification summary report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Using model: {self.model_name}\n")
            f.write(f"Total processed: {len(results)}\n")
            f.write(f"Symptom group classification errors: {group_errors}\n")
            f.write(f"Severity assessment errors: {severity_errors}\n\n")
            
            f.write("Symptom group distribution:\n")
            f.write("-" * 30 + "\n")
            for group_id, count in group_counts.items():
                group_name = "Unclassified" if group_id == 0 else next(
                    (g['group_name'] for g in self.symptom_groups if g['group_id'] == group_id), 
                    f"Group {group_id}"
                )
                f.write(f"Group {group_id} ({group_name}): {count} ({count/len(results)*100:.1f}%)\n")
            
            f.write(f"\nSeverity distribution:\n")
            f.write("-" * 30 + "\n")
            severity_names = {1: "Critical", 2: "Severe", 3: "Moderate", 4: "Mild", 5: "Mildest"}
            for severity, count in severity_counts.items():
                name = severity_names.get(severity, f"Level {severity}")
                f.write(f"{severity} level ({name}): {count} ({count/len(results)*100:.1f}%)\n")
            
            f.write(f"\nSymptom group classification confidence:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average confidence: {group_confidence_stats['mean']:.2f}\n")
            f.write(f"Minimum confidence: {group_confidence_stats['min']:.0f}\n")
            f.write(f"Maximum confidence: {group_confidence_stats['max']:.0f}\n")
            
            f.write(f"\nSeverity assessment confidence:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average confidence: {severity_confidence_stats['mean']:.2f}\n")
            f.write(f"Minimum confidence: {severity_confidence_stats['min']:.0f}\n")
            f.write(f"Maximum confidence: {severity_confidence_stats['max']:.0f}\n")
        
        logger.info(f"📋 Summary report saved: {summary_file}")
        return summary_file
    
    def process_two_stage_classification(self, limit=1000, batch_size=100, start_index=0):
        """Complete two-stage classification processing flow"""
        logger.info("🎯 Starting two-stage symptom classification")
        logger.info(f"🤖 Using model: {self.model_name}")
        logger.info("=" * 60)
        
        # Step 1: Load unique chief complaints
        complaints_df = self.load_unique_complaints(limit)
        total_count = len(complaints_df)
        
        # Step 2: Check for incomplete progress
        existing_timestamp, processed_count = self.load_existing_progress()
        
        # Determine starting position and timestamp
        if existing_timestamp and processed_count > 0 and processed_count < total_count:
            # Found incomplete task, ask if resuming
            if self.ask_resume_confirmation(processed_count, total_count):
                # Resume progress
                logger.info(f"🔄 Resuming from record {processed_count + 1}")
                start_index = processed_count
                timestamp = existing_timestamp
                existing_results = self.load_existing_results(timestamp)
            else:
                # Start fresh
                logger.info("🆕 Starting new task")
                start_index = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                existing_results = []
        else:
            # No incomplete tasks or already completed
            if processed_count >= total_count:
                logger.info("✅ Found completed processing results")
                print(f"✅ Task completed! Processed {processed_count} records")
                
                # Return completed metadata
                return {
                    'timestamp': existing_timestamp,
                    'total_processed': processed_count,
                    'model_used': self.model_name,
                    'method': 'two_stage',
                    'results_file': os.path.join(self.output_dir, f"two_stage_results_{existing_timestamp}.csv"),
                    'summary_file': os.path.join(self.output_dir, f"two_stage_summary_{existing_timestamp}.txt"),
                    'symptom_groups_count': len(self.symptom_groups),
                    'api_calls_made': processed_count * 2
                }
            
            logger.info("🆕 Starting new task")
            start_index = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            existing_results = []
        
        # Step 3: Two-stage classification
        results, final_timestamp = self.classify_two_stage(
            complaints_df, batch_size, start_index, timestamp, existing_results
        )
        
        # Step 4: Generate summary report
        summary_file = self.generate_classification_summary(results, final_timestamp)
        
        # Save final metadata
        final_metadata = {
            'timestamp': final_timestamp,
            'total_processed': len(results),
            'model_used': self.model_name,
            'method': 'two_stage',
            'results_file': os.path.join(self.output_dir, f"two_stage_results_{final_timestamp}.csv"),
            'summary_file': summary_file,
            'symptom_groups_count': len(self.symptom_groups),
            'api_calls_made': len(results) * 2
        }
        
        final_metadata_file = os.path.join(self.output_dir, f"final_metadata_{final_timestamp}.json")
        with open(final_metadata_file, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        # Update latest pointer
        latest_file = os.path.join(self.output_dir, "latest_two_stage.json")
        with open(latest_file, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("🎉 Two-stage symptom classification completed!")
        logger.info(f"📊 Classification statistics:")
        logger.info(f"    Total processed: {len(results)}")
        logger.info(f"   API calls made: {final_metadata['api_calls_made']}")
        logger.info(f"    Number of symptom groups: {len(self.symptom_groups)}")
        logger.info(f"📂 Main output files:")
        logger.info(f"    Classification results: {final_metadata['results_file']}")
        logger.info(f"    Summary report: {summary_file}")
        
        return final_metadata

    def load_existing_progress(self):
        """Check for incomplete progress that can be resumed"""
        try:
            # Find the latest progress file
            progress_files = []
            if os.path.exists(self.output_dir):
                for file in os.listdir(self.output_dir):
                    if file.startswith('progress_') and file.endswith('.json'):
                        progress_files.append(file)
            
            if not progress_files:
                return None, 0
            
            # Get the latest progress file
            latest_progress = sorted(progress_files)[-1]
            progress_file = os.path.join(self.output_dir, latest_progress)
            
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Check if the corresponding results file exists
            results_file = progress_data.get('results_file')
            if results_file and os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                
                # Get the number of processed records
                processed_count = len(results_df)
                timestamp = progress_data.get('timestamp')
                
                logger.info(f"🔄 Found incomplete progress: {processed_count} records")
                logger.info(f"📁 Progress file: {progress_file}")
                logger.info(f"📁 Results file: {results_file}")
                
                return timestamp, processed_count
            
            return None, 0
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking progress: {e}")
            return None, 0
    
    def ask_resume_confirmation(self, processed_count, total_count):
        """Ask user if they want to resume progress"""
        remaining = total_count - processed_count
        percentage = (processed_count / total_count) * 100
        
        print(f"\n🔄 Found incomplete task:")
        print(f"    Already processed: {processed_count:,} / {total_count:,} ({percentage:.1f}%)")
        print(f"    Remaining: {remaining:,} records")
        print(f"    Estimated remaining time: {remaining * 2 / 60:.1f} minutes")
        
        while True:
            choice = input("\nPlease choose:\n1. Resume processing (r)\n2. Start fresh (n)\n3. Exit (q)\nEnter choice: ").strip().lower()
            
            if choice in ['r', '1', 'resume', '恢复']:
                return True
            elif choice in ['n', '2', 'new', '重新']:
                return False
            elif choice in ['q', '3', 'quit', '退出']:
                print("👋 Exiting program")
                exit(0)
            else:
                print("❌ Invalid choice, please try again")
    
    def load_existing_results(self, timestamp):
        """Load existing results"""
        try:
            results_file = os.path.join(self.output_dir, f"two_stage_results_{timestamp}.csv")
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                return results_df.to_dict('records')
            return []
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing results: {e}")
            return []


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Two-stage symptom classification tool')
    parser.add_argument('--limit', type=int, default=1000, help='Limit of chief complaints to process')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index')
    
    args = parser.parse_args()
    
    print("🏥 Two-stage symptom classification tool")
    print("🤖 Using Llama-3.3-70B for classification") 
    print("🎯 First stage: Symptom group classification")
    print("📊 Second stage: Severity assessment")
    print(f"📊 Processing limit: {args.limit} records")
    print("=" * 60)
    
    try:
        # Create classifier
        classifier = TwoStageSymptomClassifier()
        
        # Execute two-stage classification
        result = classifier.process_two_stage_classification(
            limit=args.limit,
            batch_size=args.batch_size,
            start_index=args.start_index
        )
        
        print("\n" + "=" * 60)
        print("✅ Two-stage classification completed!")
        print(f"📊 Successfully classified {result['total_processed']} chief complaints")
        print(f"🔢 Total API calls made: {result['api_calls_made']}")
        print(f"📋 Detailed report: {result['summary_file']}")
        print("\n💡 Next steps:")
        print("   1. View classification results and summary report")
        print("   2. Analyze accuracy of each stage")
        print("   3. Compare with one-stage method")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error encountered during processing: {e}")
        print("💡 Please ensure:")
        print("   1. Llama-3.3-70B model is correctly configured")
        print("   2. IBM Watsonx configuration is correct")
        print("   3. Symptom group standards file exists")


if __name__ == "__main__":
    main() 

# ┌────────────────────────────────────────────────────────────┐
# │               Step 4: Two-Stage Symptom Classification     │
# └────────────────────────────────────────────────────────────┘

# 📁 输入：
#   - 清洗后的主诉文本（chiefcomplaint_clean）
#   - Step 3 输出的 symptom group 定义（latest_manual_grouping.json）
#   - 默认或已有的 severity rules

#        ↓
# 1. 加载主诉数据（load_unique_complaints）
#    - 最多加载 50,000 条唯一主诉文本
#    - 来源于 Step 1 embedding 阶段生成的 CSV 文件

#        ↓
# 2. 加载症状组定义 + 严重程度规则（load_symptom_groups + load_severity_rules）
#    - 从 Step 3 输出的 JSON 文件中加载 10~20 个标准的症状组
#    - 加载默认的 5 级严重程度规则（若无专属定义）

#        ↓
# 3. 准备两阶段 GPT 提示词（create_group_classification_prompt + create_severity_assessment_prompt）
#    - 第一阶段 prompt：根据主诉选择最合适的 symptom group
#    - 第二阶段 prompt：根据主诉 + 症状组评估 1~5 的严重程度

#        ↓
# 4. 遍历主诉进行两阶段推理（classify_two_stage）
#    For 每条主诉:
#      ├─ 第一阶段：症状组分类（classify_symptom_group）
#      │     → 输出 group_id, confidence（置信度）
#      └─ 第二阶段：严重程度评估（assess_severity）
#            → 输出 severity, confidence

#        ↓
# 5. 自动保存进度（save_progress）
#    - 每处理完一个 batch（默认 100 条）自动写入 CSV & JSON
#    - 支持断点续跑（load_existing_progress）

#        ↓
# 6. 生成分析报告（generate_classification_summary）
#    - 统计各 group/severity 的分布
#    - 平均置信度、错误率、异常条目数量等

#        ↓
# 🎉 最终输出：
#   - two_stage_results_<timestamp>.csv → 每条主诉的分组 + 严重等级
#   - two_stage_summary_<timestamp>.txt → 总体统计报告
#   - final_metadata_<timestamp>.json   → 元数据记录
 