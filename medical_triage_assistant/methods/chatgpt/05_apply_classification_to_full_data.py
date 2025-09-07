#!/usr/bin/env python3
"""
Step 5: Apply Llama classification results to full dataset
Map Llama-3.3-70B classification results of unique chief complaints to original records
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassificationMapper:
    """Classification result mapper"""
    
    def __init__(self):
        """Initialize mapper"""
        # Get script directory, then build data directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        self.data_dir = os.path.join(project_root, "data")
        self.output_dir = os.path.join(self.data_dir, "final_results")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("✅ Classification result mapper initialized")
    
    def load_original_data(self):
        """Load original 400k data"""
        logger.info("📂 Loading original data...")
        
        try:
            original_file = os.path.join(self.data_dir, "processed_data", "triage_final.csv")
            df = pd.read_csv(original_file)
            
            logger.info(f"✅ Original data loaded: {len(df):,} records")
            logger.info(f"   Unique chief complaints: {df['chiefcomplaint_clean'].nunique():,}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load original data: {e}")
            raise
    
    def load_classification_results(self):
        """Load classification results"""
        logger.info("📂 Loading classification results...")
        
        try:
            # Find latest classification results
            two_stage_dir = os.path.join(self.data_dir, "two_stage_classification")
            latest_file = os.path.join(two_stage_dir, "latest_two_stage.json")
            
            if os.path.exists(latest_file):
                with open(latest_file, 'r') as f:
                    metadata = json.load(f)
                
                results_file = metadata['results_file']
                results_df = pd.read_csv(results_file)
                
                logger.info(f"✅ Classification results loaded: {len(results_df):,} unique chief complaints")
                logger.info(f"   Results file: {results_file}")
                
                return results_df, metadata
            else:
                raise FileNotFoundError("Classification results file not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load classification results: {e}")
            raise
    
    def create_classification_lookup(self, classification_df):
        """Create classification lookup dictionary"""
        logger.info("🔍 Creating classification lookup table...")
        
        # Create mapping dictionary from chief complaint to classification results
        lookup_dict = {}
        
        for _, row in classification_df.iterrows():
            complaint = row['chiefcomplaint_clean']
            lookup_dict[complaint] = {
                'group_id': row['group_id'],
                'group_name': row['group_name'],
                'group_confidence': row['group_confidence'],
                'severity': row['severity'],
                'severity_confidence': row['severity_confidence']
            }
        
        logger.info(f"✅ Lookup table created: {len(lookup_dict):,} mapping relationships")
        
        return lookup_dict
    
    def apply_classification_to_full_data(self, original_df, lookup_dict):
        """Apply classification results to full dataset"""
        logger.info("🎯 Starting to map classification results to full dataset...")
        
        # Copy original data
        result_df = original_df.copy()
        
        # Add classification result columns
        result_df['group_id'] = None
        result_df['group_name'] = None
        result_df['group_confidence'] = None
        result_df['severity'] = None
        result_df['severity_confidence'] = None
        result_df['classification_found'] = False
        
        # Statistics variables
        found_count = 0
        missing_count = 0
        missing_complaints = set()
        
        # Map row by row
        logger.info("   Performing data mapping...")
        for idx, row in result_df.iterrows():
            complaint = row['chiefcomplaint_clean']
            
            if complaint in lookup_dict:
                # Found matching classification results
                classification = lookup_dict[complaint]
                result_df.at[idx, 'group_id'] = classification['group_id']
                result_df.at[idx, 'group_name'] = classification['group_name']
                result_df.at[idx, 'group_confidence'] = classification['group_confidence']
                result_df.at[idx, 'severity'] = classification['severity']
                result_df.at[idx, 'severity_confidence'] = classification['severity_confidence']
                result_df.at[idx, 'classification_found'] = True
                found_count += 1
            else:
                # No matching classification results found
                missing_count += 1
                missing_complaints.add(complaint)
            
            # Progress update
            if (idx + 1) % 50000 == 0:
                logger.info(f"   Processed: {idx + 1:,}/{len(result_df):,} ({(idx + 1)/len(result_df)*100:.1f}%)")
        
        # Mapping completion statistics
        logger.info("✅ Data mapping completed!")
        logger.info(f"   Successfully mapped: {found_count:,} records ({found_count/len(result_df)*100:.2f}%)")
        logger.info(f"   Classification not found: {missing_count:,} records ({missing_count/len(result_df)*100:.2f}%)")
        logger.info(f"   Missing unique chief complaints: {len(missing_complaints):,}")

        
        return result_df, missing_complaints
    
    def generate_mapping_statistics(self, result_df, missing_complaints, metadata):
        """Generate mapping statistics report"""
        logger.info("📊 Generating statistics report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Basic statistics
        total_records = len(result_df)
        mapped_records = result_df['classification_found'].sum()
        unmapped_records = total_records - mapped_records
        
        # Symptom group distribution
        group_stats = result_df[result_df['classification_found']].groupby(['group_id', 'group_name']).size().sort_values(ascending=False)
        
        # Severity distribution (with robust handling for mixed types)
        severity_series = result_df[result_df['classification_found']]['severity']
        
        # Convert to numeric, coercing non-numeric values to NaN
        severity_numeric = pd.to_numeric(severity_series, errors='coerce')
        
        # Drop NaN values and calculate statistics
        severity_stats = severity_numeric.dropna().value_counts().sort_index()
        
        # Generate report
        report_file = os.path.join(self.output_dir, f"mapping_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Classification Results Mapping Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Classification results used: {metadata.get('results_file', 'Unknown')}\n")
            f.write(f"Classification model: {metadata.get('model_used', 'Unknown')}\n\n")
            
            f.write("Mapping statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total original records: {total_records:,}\n")
            f.write(f"Successfully mapped records: {mapped_records:,} ({mapped_records/total_records*100:.2f}%)\n")
            f.write(f"Unmapped records: {unmapped_records:,} ({unmapped_records/total_records*100:.2f}%)\n")
            f.write(f"Missing unique chief complaints: {len(missing_complaints):,}\n\n")
            
            f.write("Symptom group distribution (mapped records):\n")
            f.write("-" * 30 + "\n")
            for (group_id, group_name), count in group_stats.items():
                percentage = count / mapped_records * 100
                f.write(f"Group {group_id} ({group_name}): {count:,} ({percentage:.1f}%)\n")
            
            f.write(f"\nSeverity distribution (mapped records):\n")
            f.write("-" * 30 + "\n")
            severity_names = {1: "Critical", 2: "Severe", 3: "Moderate", 4: "Mild", 5: "Minor"}
            for severity, count in severity_stats.items():
                name = severity_names.get(severity, f"Level {severity}")
                percentage = count / mapped_records * 100
                f.write(f"{severity} level ({name}): {count:,} ({percentage:.1f}%)\n")
            
            if missing_complaints:
                f.write(f"\nMissing chief complaint examples (first 20):\n")
                f.write("-" * 30 + "\n")
                for i, complaint in enumerate(list(missing_complaints)[:20], 1):
                    f.write(f"{i}. {complaint}\n")
        
        logger.info(f"📋 Statistics report saved: {report_file}")
        return report_file
    
    def save_final_results(self, result_df, missing_complaints, metadata):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        full_results_file = os.path.join(self.output_dir, f"triage_with_classification_{timestamp}.csv")
        result_df.to_csv(full_results_file, index=False)
        logger.info(f"💾 Complete results saved: {full_results_file}")
        
        # Save only successfully mapped records
        mapped_results_file = os.path.join(self.output_dir, f"triage_mapped_only_{timestamp}.csv")
        mapped_df = result_df[result_df['classification_found']].copy()
        mapped_df = mapped_df.drop('classification_found', axis=1)  # Remove helper column
        mapped_df.to_csv(mapped_results_file, index=False)
        logger.info(f"💾 Mapped records saved: {mapped_results_file}")
        
        # Save missing chief complaints list
        if missing_complaints:
            missing_file = os.path.join(self.output_dir, f"missing_complaints_{timestamp}.txt")
            with open(missing_file, 'w', encoding='utf-8') as f:
                f.write("Chief complaints without classification results\n")
                f.write("=" * 30 + "\n")
                # Convert all items to string before sorting to avoid TypeError with mixed types (e.g., str and float NaN)
                sorted_complaints = sorted([str(c) for c in missing_complaints])
                for complaint in sorted_complaints:
                    f.write(f"{complaint}\n")
            logger.info(f"📋 Missing chief complaints list saved: {missing_file}")
        
        # Generate statistics report
        report_file = self.generate_mapping_statistics(result_df, missing_complaints, metadata)
        
        # Save metadata
        final_metadata = {
            'timestamp': timestamp,
            'source_classification': metadata.get('results_file'),
            'original_records': len(result_df),
            # Cast numpy.int64 to standard python int for JSON serialization
            'mapped_records': int(result_df['classification_found'].sum()),
            'unmapped_records': int(len(result_df) - result_df['classification_found'].sum()),
            'missing_complaints_count': len(missing_complaints),
            'full_results_file': full_results_file,
            'mapped_only_file': mapped_results_file,
            'report_file': report_file,
            'model_used': metadata.get('model_used'),
            'method': metadata.get('method')
        }
        
        metadata_file = os.path.join(self.output_dir, f"mapping_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        # Update latest pointer
        latest_file = os.path.join(self.output_dir, "latest_mapping.json")
        with open(latest_file, 'w') as f:
            json.dump(final_metadata, f, indent=2)
        
        logger.info(f"📋 Metadata saved: {metadata_file}")
        
        return final_metadata
    
    def process_full_mapping(self):
        """Complete mapping processing flow"""
        logger.info("🎯 Starting to map classification results to full dataset")
        logger.info("=" * 60)
        
        # Step 1: Load original data
        original_df = self.load_original_data()
        
        # Step 2: Load classification results
        classification_df, metadata = self.load_classification_results()
        
        # Step 3: Create lookup table
        lookup_dict = self.create_classification_lookup(classification_df)
        
        # Step 4: Apply classification to full dataset
        result_df, missing_complaints = self.apply_classification_to_full_data(original_df, lookup_dict)
        
        # Step 5: Save results
        final_metadata = self.save_final_results(result_df, missing_complaints, metadata)
        
        logger.info("=" * 60)
        logger.info("🎉 Classification result mapping completed!")
        logger.info(f"📊 Mapping statistics:")
        logger.info(f"   Original records: {final_metadata['original_records']:,}")
        logger.info(f"   Successfully mapped: {final_metadata['mapped_records']:,} ({final_metadata['mapped_records']/final_metadata['original_records']*100:.2f}%)")
        logger.info(f"   Unmapped records: {final_metadata['unmapped_records']:,}")
        logger.info(f"📂 Main output files:")
        logger.info(f"   Complete results: {final_metadata['full_results_file']}")
        logger.info(f"   Mapped records: {final_metadata['mapped_only_file']}")
        logger.info(f"   Statistics report: {final_metadata['report_file']}")
        
        return final_metadata


def main():
    """Main function"""
    print("🏥 Classification Results Mapping Tool")
    print("📊 Apply unique chief complaint classification results to full 400k dataset")
    print("=" * 60)
    
    try:
        # Create mapper
        mapper = ClassificationMapper()
        
        # Execute mapping processing
        result = mapper.process_full_mapping()
        
        print("\n" + "=" * 60)
        print("✅ Mapping processing completed!")
        print(f"📊 Successfully processed {result['original_records']:,} records")
        print(f"📈 Mapping success rate: {result['mapped_records']/result['original_records']*100:.2f}%")
        print(f"📂 Detailed report: {result['report_file']}")
        print("\n💡 Now you can:")
        print("   1. View mapping statistics report")
        print("   2. Analyze symptom group and severity distribution")
        print("   3. Handle unmapped records")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        print("💡 Please ensure:")
        print("   1. Unique chief complaint classification has been completed")
        print("   2. Original data file exists")
        print("   3. Sufficient disk space available")


if __name__ == "__main__":
    main() 

# 📁 Step 5: 映射分类结果到完整数据集（ClassificationMapper）

# ┌───────────────────────────────────────────────┐
# │               process_full_mapping()         │
# └───────────────────────────────────────────────┘
#                      ↓
# 1️⃣ Load 原始主诉数据（400k）
#    └── original_df = load_original_data()

#                      ↓
# 2️⃣ Load 两阶段分类结果（55k 唯一主诉）
#    └── classification_df, metadata = load_classification_results()

#                      ↓
# 3️⃣ 构建映射表 lookup_dict
#    └── chiefcomplaint_clean → group_id, severity 等
#    └── lookup_dict = create_classification_lookup(classification_df)

#                      ↓
# 4️⃣ 对原始数据逐条映射（apply_classification_to_full_data）
#    ┌─ 遍历每一行 chiefcomplaint_clean:
#    │    ├─ 如果能在 lookup_dict 中找到 → 赋值 group_id、severity 等
#    │    └─ 否则计入 missing_complaints
#    └─ 输出 result_df 和 missing chief complaints

#                      ↓
# 5️⃣ 保存结果（save_final_results）
#    ├─ 保存完整数据（triage_with_classification_xxx.csv）
#    ├─ 保存已映射子集（triage_mapped_only_xxx.csv）
#    ├─ 保存缺失主诉列表（missing_complaints_xxx.txt）
#    └─ 保存统计报告（mapping_report_xxx.txt）

#                      ↓
# 6️⃣ 输出最终 metadata & 显示映射概况

# ✅ 最终结果：
#    - 完整映射数据文件（含 group_id + severity）
#    - 可分析症状分布、严重程度分布
#    - 可追踪缺失主诉原因
 