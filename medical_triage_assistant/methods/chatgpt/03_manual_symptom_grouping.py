#!/usr/bin/env python3
"""
Step 3: Manual symptom grouping and annotation
Use GPT-3.5-turbo to manually analyze 1000 representative samples, define 10-20 medical symptom groups
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from openai import AzureOpenAI
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManualSymptomGroupAnalyzer:
    """Manual symptom grouping analyzer"""
    
    def __init__(self):
        """Initialize analyzer"""
        # Get script directory, then build data directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Parent directory is project root
        
        self.data_dir = os.path.join(script_dir, "data", "clustering")
        self.output_dir = os.path.join(script_dir, "data", "symptom_groups")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Azure OpenAI client for GPT-4
        import sys
        import importlib.util
        
        # Load azure_simple config - fix path to go up two levels
        config_path = os.path.join(project_root, '..', 'config', 'azure_simple.py')
        spec = importlib.util.spec_from_file_location("azure_simple", config_path)
        azure_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(azure_config)
        
        GPT4_API_KEY = azure_config.GPT4_API_KEY
        GPT4_ENDPOINT = azure_config.GPT4_ENDPOINT
        GPT4_DEPLOYMENT_NAME = azure_config.GPT4_DEPLOYMENT_NAME
        self.client = AzureOpenAI(
            api_key=GPT4_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=GPT4_ENDPOINT
        )
        self.gpt_model = GPT4_DEPLOYMENT_NAME
        
        logger.info(f"✅ Manual symptom grouping analyzer initialized, using model: {self.gpt_model}")
    
    def load_representative_samples(self):
        """Load representative sample data"""
        logger.info("📂 Loading representative samples...")
        
        # Read latest clustering results
        latest_file = f"{self.data_dir}/latest_clustering.json"
        
        if not os.path.exists(latest_file):
            raise FileNotFoundError(f"Clustering result file not found: {latest_file}")
        
        with open(latest_file, 'r') as f:
            metadata = json.load(f)
        
        # Load representative samples - use absolute path
        csv_path = metadata['output_file']
        # If relative path, convert to absolute path
        if not os.path.isabs(csv_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            csv_path = os.path.join(project_root, csv_path)
        samples_df = pd.read_csv(csv_path)
        logger.info(f"✅ Loading completed, {len(samples_df)} representative samples distributed across {samples_df['cluster'].nunique()} clusters")
        
        return samples_df, metadata
    
    def prepare_comprehensive_analysis(self, samples_df: pd.DataFrame):
        """Prepare comprehensive analysis data"""
        logger.info("📋 Preparing comprehensive symptom analysis data...")
        
        # Organize data by cluster
        cluster_summary = {}
        
        for cluster_id in sorted(samples_df['cluster'].unique()):
            cluster_data = samples_df[samples_df['cluster'] == cluster_id]
            
            # Get all symptoms for this cluster (sorted by frequency)
            symptoms = cluster_data['chiefcomplaint_clean'].tolist()
            
            # If frequency info available, sort by frequency
            if 'frequency' in cluster_data.columns:
                cluster_data_sorted = cluster_data.sort_values('frequency', ascending=False)
                symptoms = cluster_data_sorted['chiefcomplaint_clean'].tolist()
            
            cluster_summary[cluster_id] = {
                'count': len(cluster_data),
                'symptoms': symptoms[:30],  # Take top 15 most representative symptoms
                'sample_indices': cluster_data.index.tolist()
            }
        
        logger.info(f"✅ Data preparation completed, covering {len(cluster_summary)} original clusters")
        return cluster_summary
    
    def analyze_all_symptoms_with_gpt(self, cluster_summary: dict):
        """Use GPT to analyze all symptoms and define symptom groups"""
        logger.info("🤖 Starting GPT-3.5-turbo for comprehensive symptom analysis...")
        
        # Build a comprehensive list of all symptoms
        all_symptoms_by_cluster = []
        
        for cluster_id, data in cluster_summary.items():
            cluster_info = f"Original cluster {cluster_id} ({data['count']} samples):\n"
            cluster_info += "\n".join([f"  - {symptom}" for symptom in data['symptoms'][:10]])
            all_symptoms_by_cluster.append(cluster_info)
        
        # Build comprehensive analysis prompt
        comprehensive_prompt = f"""
You are helping build an emergency triage system. We have collected 1000 representative patient chief complaints from the emergency department.

Each complaint is a short phrase in natural language, describing the patient's main symptom (e.g., "chest pain", "sudden vomiting", "dizziness", "vaginal bleeding during pregnancy", etc.).

These complaints have been grouped by machine-learning clusters as follows:

{chr(10).join(all_symptoms_by_cluster)}

Your task:

**CRITICAL REQUIREMENTS:**
1. You MUST output 10–20 groups, not fewer than 10. Even though the example shows 2 groups, that is only an example.
2. You have 60 original clusters — you MAY merge them into broader categories if clinically appropriate.
3. **Do NOT produce fewer than 10 groups.**
4. Each group should represent a clinically meaningful triage category that could realistically guide emergency decision-making.
5. Prefer **moderate merging** (combine closely related clusters), but do not collapse everything into overly broad buckets.
6. If in doubt, split a large category into 2–3 subgroups to maintain the required 10–20 range.

**Grouping Strategy:**
- Organize by **major body system or triage-relevant categories**.
- Merge clusters that are clinically overlapping, but keep important distinctions (e.g., abdominal pain vs. chest pain).
- Balance granularity: not as fine as 60 clusters, not as broad as 10 categories.
- Target a total of **15 groups** (acceptable range 10–20).

**Examples of groups:**
- Abdominal Pain
- Headache and Dizziness
- Chest Pain / Cardiac Symptoms
- Shortness of Breath / Respiratory Symptoms
- Trauma and Injury
- Pregnancy-Related Issues
- Neurological Deficits

Please return the result in the following JSON format:

{{
  "symptom_groups": [
    {{
      "group_id": 1,
      "group_name": "Chest Pain / Cardiac Symptoms",
      "medical_description": "Symptoms related to acute or chronic cardiac issues, including ischemia and arrhythmias.",
      "severity_range": [2, 5],
      "included_clusters": [2, 7],
      "typical_symptoms": ["chest tightness", "sudden chest pain", "palpitations", "chest pressure", "heart discomfort"]
    }},
    {{
      "group_id": 2,
      "group_name": "Neurological Symptoms",
      "medical_description": "Symptoms indicating central or peripheral neurological dysfunction, may involve brain, nerves, or balance systems.",
      "severity_range": [2, 5],
      "included_clusters": [3, 16],
      "typical_symptoms": ["migraine", "severe dizziness", "slurred speech", "facial numbness", "weakness in limbs"]
    }}
  ],
  "grouping_notes": "Explain your reasoning briefly"
}}


"""

        
        try:
            logger.info("   Performing comprehensive symptom analysis...")
            logger.info(f"   Number of original clusters sent to GPT: {len(cluster_summary)}")
            
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an emergency department chief physician with 20 years of experience, specializing in hospital triage standard development. You are proficient in various disease symptom presentations and urgency assessments."
                    },
                    {
                        "role": "user", 
                        "content": comprehensive_prompt
                    }
                ],
                temperature=0.2,  # Lower temperature for consistency
                max_tokens=4000   # Increase token limit
            )
            
            # Parse GPT response
            gpt_response = response.choices[0].message.content.strip()
            logger.info("✅ GPT analysis completed, parsing results...")
            
            try:
                # Handle possible code block wrapping
                if gpt_response.startswith("```json"):
                    gpt_response = gpt_response[7:]  # Remove ```json
                if gpt_response.endswith("```"):
                    gpt_response = gpt_response[:-3]  # Remove ```
                
                gpt_response = gpt_response.strip()
                
                analysis_result = json.loads(gpt_response)
                symptom_groups = analysis_result.get('symptom_groups', [])
                
                logger.info(f"✅ Successfully defined {len(symptom_groups)} symptom groups")
                
                # Display grouping results
                for group in symptom_groups:
                    severity_info = group.get('severity_range', 'N/A')
                    logger.info(f"   Symptom group {group['group_id']}: {group['group_name']} (severity: {severity_info})")
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing failed: {e}")
                logger.info("Original GPT response:")
                logger.info(gpt_response)
                
                # Return fallback analysis
                return self._create_fallback_groups(cluster_summary)
            except Exception as e:
                logger.error(f"❌ Unexpected error during JSON processing: {e}")
                logger.info("Original GPT response:")
                logger.info(gpt_response)
                
                # Return fallback analysis
                return self._create_fallback_groups(cluster_summary)
                
        except Exception as e:
            logger.error(f"❌ GPT analysis failed: {e}")
            return self._create_fallback_groups(cluster_summary)
    
    def _create_fallback_groups(self, cluster_summary: dict):
        """Create fallback symptom group definitions"""
        logger.info("📋 Creating fallback symptom group definitions...")
        
        # Fallback grouping based on common medical classifications
        fallback_groups = []
        
        # Basic grouping logic
        group_mapping = [
            {"group_id": 1, "group_name": "Cardiovascular Symptoms", "clusters": [2, 16], "severity": [3, 5]},
            {"group_id": 2, "group_name": "Respiratory Symptoms", "clusters": [5], "severity": [2, 4]},
            {"group_id": 3, "group_name": "Digestive Symptoms", "clusters": [6], "severity": [2, 4]},
            {"group_id": 4, "group_name": "Neurological Symptoms", "clusters": [11], "severity": [2, 5]},
            {"group_id": 5, "group_name": "Trauma Related", "clusters": [7, 8], "severity": [2, 4]},
            {"group_id": 6, "group_name": "Substance Related", "clusters": [0], "severity": [2, 5]},
            {"group_id": 7, "group_name": "Other Symptoms", "clusters": list(set(cluster_summary.keys()) - {0,2,5,6,7,8,11,16}), "severity": [1, 4]}
        ]
        
        for mapping in group_mapping:
            # Collect typical symptoms for this group
            typical_symptoms = []
            for cluster_id in mapping["clusters"]:
                if cluster_id in cluster_summary:
                    typical_symptoms.extend(cluster_summary[cluster_id]["symptoms"][:3])
            
            fallback_groups.append({
                "group_id": mapping["group_id"],
                "group_name": mapping["group_name"],
                "medical_description": f"Clinical manifestations related to {mapping['group_name']}",
                "severity_range": mapping["severity"],
                "included_clusters": mapping["clusters"],
                "typical_symptoms": typical_symptoms[:8],
                "triage_recommendation": "Determine priority based on specific symptoms"
            })
        
        return {
            "symptom_groups": fallback_groups,
            "grouping_rationale": "Fallback grouping based on medical system classification"
        }
    
    def create_mapping_table(self, samples_df: pd.DataFrame, analysis_result: dict):
        """Create a mapping table from original clusters to new symptom groups"""
        logger.info("📋 Creating cluster mapping table...")
        
        # Create mapping relationships
        cluster_to_group = {}
        for group in analysis_result['symptom_groups']:
            for cluster_id in group['included_clusters']:
                cluster_to_group[cluster_id] = {
                    'group_id': group['group_id'],
                    'group_name': group['group_name'],
                    'severity_range': group.get('severity_range', 'N/A')
                }
        
        # Apply mapping to sample data
        samples_with_groups = samples_df.copy()
        samples_with_groups['symptom_group_id'] = samples_with_groups['cluster'].map(
            lambda x: cluster_to_group.get(x, {'group_id': 0})['group_id']
        )
        samples_with_groups['symptom_group_name'] = samples_with_groups['cluster'].map(
            lambda x: cluster_to_group.get(x, {'group_name': 'Unclassified'})['group_name']
        )
        
        logger.info(f"✅ Mapping table created, covering {samples_with_groups['symptom_group_id'].nunique()} symptom groups")
        
        return samples_with_groups, cluster_to_group
    
    def save_results_without_severity(self, analysis_result: dict, 
                    samples_with_groups: pd.DataFrame, cluster_mapping: dict, metadata: dict, cluster_summary: dict):
        """Save analysis results without severity rules (handled in step 04)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save symptom group definitions
        groups_file = f"{self.output_dir}/manual_symptom_groups_{timestamp}.json"
        with open(groups_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Symptom group definitions saved: {groups_file}")
        
        # 2. Save mapping table
        mapping_file = f"{self.output_dir}/cluster_mapping_{timestamp}.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_mapping, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"💾 Cluster mapping table saved: {mapping_file}")
        
        # 3. Save samples with groups
        samples_file = f"{self.output_dir}/samples_with_groups_{timestamp}.csv"
        samples_with_groups.to_csv(samples_file, index=False)
        logger.info(f"💾 Samples with groups data saved: {samples_file}")
        
        # 4. Create summary report
        summary_file = f"{self.output_dir}/grouping_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Manual Symptom Grouping Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model used: {self.gpt_model}\n")
            f.write(f"Original clusters: {len(cluster_summary)}\n")
            f.write(f"Final symptom groups: {len(analysis_result['symptom_groups'])} groups\n\n")
            
            f.write("Symptom Group Details:\n")
            f.write("-" * 30 + "\n")
            for group in analysis_result['symptom_groups']:
                f.write(f"\n{group['group_id']}. {group['group_name']}\n")
                f.write(f"    Description: {group.get('medical_description', 'N/A')}\n")
                f.write(f"    Severity range: {group.get('severity_range', 'N/A')}\n")
                f.write(f"    Included clusters: {group.get('included_clusters', [])}\n")
                f.write(f"    Typical symptoms: {', '.join(group.get('typical_symptoms', [])[:5])}\n")
            
            f.write(f"\nNote: Severity assessment rules are generated in step 04_two_stage_classification.py\n")
        
        logger.info(f"📋 Summary report saved: {summary_file}")
        
        # 5. Save metadata (without severity rules)
        result_metadata = {
            'timestamp': timestamp,
            'model_used': self.gpt_model,
            'total_groups': len(analysis_result['symptom_groups']),
            'groups_file': groups_file,
            'mapping_file': mapping_file,
            'samples_file': samples_file,
            'summary_file': summary_file,
            'source_clustering': metadata['output_file'],
            'note': 'Severity rules generated in step 04_two_stage_classification.py'
        }
        
        metadata_file = f"{self.output_dir}/manual_grouping_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(result_metadata, f, indent=2, default=str)
        
        # Save latest result info
        with open(f"{self.output_dir}/latest_manual_grouping.json", 'w') as f:
            json.dump(result_metadata, f, indent=2, default=str)
        
        logger.info(f"📋 Metadata saved: {metadata_file}")
        
        return result_metadata
    
    def process_manual_grouping(self):
        """Complete manual symptom grouping processing flow"""
        logger.info("🎯 Starting manual symptom grouping analysis")
        logger.info(f"🤖 Using model: {self.gpt_model}")
        logger.info("=" * 60)
        
        # Step 1: Load representative samples
        samples_df, metadata = self.load_representative_samples()
        
        # Step 2: Prepare comprehensive analysis data
        cluster_summary = self.prepare_comprehensive_analysis(samples_df)
        
        # Step 3: GPT comprehensive analysis
        analysis_result = self.analyze_all_symptoms_with_gpt(cluster_summary)
        
        # Note: Severity rules creation removed - handled in 04_two_stage_classification.py
        # Step 4: Create mapping table
        samples_with_groups, cluster_mapping = self.create_mapping_table(samples_df, analysis_result)
        
        # Step 5: Save results (without severity rules)
        result_metadata = self.save_results_without_severity(
            analysis_result, samples_with_groups, cluster_mapping, metadata, cluster_summary
        )
        
        logger.info("=" * 60)
        logger.info("🎉 Manual symptom grouping analysis completed!")
        logger.info(f"📊 Analysis statistics:")
        logger.info(f"    Original clusters: {len(cluster_summary)}")
        logger.info(f"    Final symptom groups: {result_metadata['total_groups']} groups")
        logger.info(f"    Processed samples: {len(samples_df)}")
        logger.info(f"🎯 Main output files:")
        logger.info(f"    Symptom group definitions: {result_metadata['groups_file']}")
        logger.info(f"    Summary report: {result_metadata['summary_file']}")
        logger.info("💡 Severity assessment will be handled in step 04_two_stage_classification.py")
        
        return result_metadata


def main():
    """Main function"""
    print("🏥 Manual Symptom Grouping Analysis Tool")
    print("🤖 Using GPT-3.5-turbo for professional medical analysis")
    print("=" * 60)
    
    try:
        # Create analyzer
        analyzer = ManualSymptomGroupAnalyzer()
        
        # Execute manual grouping analysis
        result = analyzer.process_manual_grouping()
        
        print("\n" + "=" * 60)
        print("✅ Manual grouping analysis completed!")
        print(f"📊 Successfully defined {result['total_groups']} symptom groups")
        print(f"📂 Detailed report: {result['summary_file']}")
        print("\n💡 Next steps:")
        print("   1. View symptom group definitions and summary report")
        print("   2. Run batch classification script")
        print("   3. Apply to all 400,000 data")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Please ensure:")
        print("   1. Clustering analysis has been run")
        print("   2. GPT-3.5-turbo model is properly deployed")
        print("   3. Azure OpenAI configuration is correct")


if __name__ == "__main__":
    main() 


# ┌────────────────────────────────────────────────────────────┐
# │                   Step 3: Manual Symptom Grouping          │
# └────────────────────────────────────────────────────────────┘

# 📁 输入：从前一阶段输出的代表性样本（CSV），每条有聚类标签 cluster

#        ↓
# 1. 加载代表样本（load_representative_samples）
#    - 从 latest_clustering.json 获取 CSV 路径
#    - 加载数据集，包含样本文本、embedding 聚类结果等

#        ↓
# 2. 准备综合分析数据（prepare_comprehensive_analysis）
#    - 按 cluster 汇总每类的症状文本（chief complaint）
#    - 每类最多保留 top 15 个代表性主诉（按频率排序）

#        ↓
        # 3. 用 Llama 分析症状群组（analyze_all_symptoms_with_llama）
#    - 构建包含所有 cluster 的症状描述 prompt
#    - 让 GPT 按医疗标准将这些聚类重新归类为 10~20 个“症状组”
#    - 每组包含：
#       • group_id、group_name、severity_range
#       • typical symptoms、included_clusters、triage_recommendation

#        ↓
# 4. 生成 cluster → symptom group 的映射（create_mapping_table）
#    - 将聚类编号映射到 Llama 分析出的 symptom group
#    - 为每条样本添加 symptom_group_id 和 symptom_group_name

#        ↓
# 5. 保存分析结果（save_results_without_severity）
#    - 保存：
#       • Llama 返回的分组 JSON
#       • cluster 到 symptom group 的映射表
#       • 样本数据（包含分组标签）
#       • 汇总报告 summary.txt（症状组分布、代表症状等）
#       • metadata（供后续步骤使用）

#        ↓
# 💡 提示：
#   后续的 Step 4 会单独调用 Llama，为每条主诉评估 1~5 的严重等级（severity）
 