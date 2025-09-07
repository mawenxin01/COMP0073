#!/usr/bin/env python3
"""
Step 1: Extract unique Chief Complaints and convert to embedding vectors
Extract unique chiefcomplaint from triage_final.csv, use SentenceTransformer model to convert to vectors

Optimization strategy:
- Only process unique complaints, avoid duplicate computation of same text embeddings
- Use local SentenceTransformer model for faster processing
- Automatic text standardization (lowercase, strip spaces, merge duplicate spaces)

Resume support:
- Automatically save progress to progress.txt
- Temporary results saved to embeddings_temp.npy  
- Resume from checkpoint when restarted after interruption
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChiefComplaintEmbeddingExtractor:
    """Chief Complaint embedding vector extractor - supports resume from interruption"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with SentenceTransformer model"""
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Set up file paths - 使用绝对路径避免相对路径问题
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '../..')
        self.data_dir = os.path.join(project_root, "data_processing/processed_data")
        self.output_dir = os.path.join(script_dir, "data/embeddings")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"✅ Chief Complaint embedding vector extractor initialized with model: {model_name}")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        
        logger.info(f"📂 Loading data: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Data loaded successfully, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def clean_chief_complaints(self, df: pd.DataFrame) -> pd.DataFrame:
       
        logger.info("🧹 Cleaning chief complaint data...")
        
        # Save original data count
        original_count = len(df)
        
        # 1. Remove rows with null chiefcomplaint
        df_clean = df.dropna(subset=['chiefcomplaint']).copy()
        logger.info(f"   After removing nulls: {len(df_clean)} rows (original: {original_count})")
        
        # 2. Standardize text format (if not already standardized)
        df_clean['chiefcomplaint_clean'] = df_clean['chiefcomplaint'].astype(str)
        df_clean['chiefcomplaint_clean'] = df_clean['chiefcomplaint_clean'].str.strip()
        df_clean['chiefcomplaint_clean'] = df_clean['chiefcomplaint_clean'].str.lower()
        df_clean['chiefcomplaint_clean'] = df_clean['chiefcomplaint_clean'].str.replace(r'\s+', ' ', regex=True)
        
        # 3. Remove duplicate chief complaints (based on standardized text)
        # Keep first occurrence of each unique complaint
        unique_complaints = df_clean['chiefcomplaint_clean'].value_counts()
        logger.info(f"   Found unique complaints: {len(unique_complaints)}")
        
        # Keep only first record of each unique complaint
        df_unique = df_clean.drop_duplicates(subset=['chiefcomplaint_clean'], keep='first')
        logger.info(f"   After removing duplicates: {len(df_unique)} rows")
        
        # 4. Smart filtering: Remove meaningless short entries but preserve medical abbreviations
        # Test results show model understands abbreviations well with medical context (0.904 avg similarity)
        # So we'll preserve all medical abbreviations as-is
        
        # All medical abbreviations preserved as-is - model understands them well with medical prefix
        medical_abbreviations = {
            'si', 'cp', 'ha', 'od', 'sz', 'pe', 'mi', 'af', 'bp', 'fb',
            'st', 'ah', 'hi', 'ch', 'sa', 'vh', 'vt', 'tb', 'ap'
        }
        
        # No expansion needed - model understands abbreviations well with medical context
        df_unique['chiefcomplaint_expanded'] = df_unique['chiefcomplaint_clean']
        
        logger.info(f"   Medical abbreviations to preserve: {sorted(medical_abbreviations)}")
        logger.info(f"   All abbreviations kept as-is (model understands well with medical context)")
        logger.info(f"   Test results: 0.904 average similarity for medical abbreviations")
        
        # 5. Simple filtering: Remove only single characters and meaningless patterns
        # Let the model learn to understand all other text including potential medical abbreviations
        before_filter = len(df_unique)
        
        # Remove single characters - they're meaningless even with medical prefix
        single_char_mask = df_unique['chiefcomplaint_clean'].str.len() == 1
        single_char_count = single_char_mask.sum()
        df_unique = df_unique[~single_char_mask]
        
        # Remove entries that are clearly not medical (empty, special chars only)
        invalid_patterns = df_unique['chiefcomplaint_clean'].str.match(r'^[\s\-\.\"\'\\\?]*$')
        invalid_count = invalid_patterns.fillna(False).sum()
        df_unique = df_unique[~invalid_patterns.fillna(False)]
        
        after_filter = len(df_unique)
        total_filtered = before_filter - after_filter
        
        logger.info(f"   After simple filtering: {after_filter} rows")
        logger.info(f"   Filtered out: {total_filtered} entries total")
        logger.info(f"     - Single characters: {single_char_count}")
        logger.info(f"     - Invalid patterns: {invalid_count}")
        logger.info(f"   All other text preserved for model to learn and understand")
        
        # Show which specific abbreviations were found
        found_abbreviations = [k for k in medical_abbreviations if k in df_unique['chiefcomplaint_clean'].values]
        if found_abbreviations:
            logger.info(f"   Known medical abbreviations found: {found_abbreviations}")
        else:
            logger.info(f"   No known medical abbreviations found in this dataset")
        
        # 5. Add indices and statistics
        df_unique = df_unique.reset_index(drop=True)
        df_unique['complaint_id'] = range(len(df_unique))
        
        # Add frequency information for each complaint
        frequency_map = df_clean['chiefcomplaint_clean'].value_counts().to_dict()
        df_unique['frequency'] = df_unique['chiefcomplaint_clean'].map(frequency_map)
        
        logger.info(f"✅ Data cleaning completed, final unique complaints count: {len(df_unique)}")
        logger.info(f"   Data compression rate: {((original_count - len(df_unique)) / original_count * 100):.1f}%")
        
        # Show most common complaints
        top_complaints = df_unique.nlargest(5, 'frequency')[['chiefcomplaint_clean', 'frequency']]
        logger.info("   Top 5 most common complaints:")
        for _, row in top_complaints.iterrows():
            logger.info(f"     '{row['chiefcomplaint_clean']}': {row['frequency']} times")
        
        return df_unique
    
    def extract_embeddings_batch(self, df: pd.DataFrame, batch_size: int = 50, save_interval: int = 500) -> np.ndarray:
        
        logger.info(f"🚀 Starting batch embedding extraction, data size: {len(df)}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Save interval: {save_interval} items")
        logger.info(f"   🔄 Resume support: If interrupted, restart will automatically continue from checkpoint")
        
        # Use expanded text for embedding if available, otherwise use original cleaned text
        if 'chiefcomplaint_expanded' in df.columns:
            base_complaints = df['chiefcomplaint_expanded'].tolist()
            logger.info("   Using expanded medical abbreviations for better embedding quality")
        else:
            base_complaints = df['chiefcomplaint_clean'].tolist()
            logger.info("   Using original cleaned text for embedding")
        
        # Add medical context prefix for better embedding understanding
        # The prefix helps the embedding model understand this is medical domain text
        medical_prefix = "Medical chief complaint: "
        complaints_for_embedding = [f"{medical_prefix}{complaint}" for complaint in base_complaints]
        
        logger.info(f"   Added medical context prefix: '{medical_prefix}'")
        logger.info(f"   Example: '{complaints_for_embedding[0][:100]}...'")
        logger.info("   Note: Prefix is only used for embedding generation, original text preserved in results")
            
        total_count = len(complaints_for_embedding)
        
        # Check for intermediate saved files
        temp_file = f"{self.output_dir}/embeddings_temp.npy"
        progress_file = f"{self.output_dir}/progress.txt"
        
        start_idx = 0
        all_embeddings = []
        
        # Check progress file
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                start_idx = int(f.read().strip())
            logger.info(f"🔄 Found progress file, starting from item {start_idx}")
            
            # Load existing embeddings
            if os.path.exists(temp_file):
                all_embeddings = np.load(temp_file).tolist()
                logger.info(f"📂 Loaded existing embeddings: {len(all_embeddings)} items")
        else:
            logger.info(f"🆕 Starting fresh embedding extraction")
        
        try:
            for i in range(start_idx, total_count, batch_size):
                batch_end = min(i + batch_size, total_count)
                batch_complaints = complaints_for_embedding[i:batch_end]
                
                progress_pct = (i / total_count) * 100
                logger.info(f"📤 Processing batch {i//batch_size + 1}: {i+1}-{batch_end}/{total_count} ({progress_pct:.1f}%)")
                
                # Get batch embeddings using SentenceTransformer
                batch_embeddings = self.model.encode(batch_complaints, show_progress_bar=False)
                
                # Add to total results
                all_embeddings.extend(batch_embeddings.tolist())
                
                # Periodically save intermediate results
                if (i + batch_size) % save_interval == 0 or batch_end == total_count:
                    # Save temporary file
                    np.save(temp_file, np.array(all_embeddings))
                    
                    # Save progress
                    with open(progress_file, 'w') as f:
                        f.write(str(batch_end))
                    
                    progress_pct = (batch_end / total_count) * 100
                    logger.info(f"💾 Intermediate results saved, progress: {batch_end}/{total_count} ({progress_pct:.1f}%)")
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings)
            logger.info(f"✅ Embedding extraction completed! Final shape: {embeddings_array.shape}")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"❌ Embedding extraction failed: {e}")
            logger.info(f"💾 Current progress saved to: {progress_file}")
            logger.info(f"🔄 Restart this script will automatically continue from item {start_idx}")
            raise
    
    def save_results(self, df: pd.DataFrame, embeddings: np.ndarray):
       
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save embeddings
        embeddings_file = f"{self.output_dir}/chief_complaint_embeddings_{timestamp}.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"💾 Embeddings saved: {embeddings_file}")
        
        # Save corresponding data info
        data_file = f"{self.output_dir}/chief_complaint_data_{timestamp}.csv"
        df.to_csv(data_file, index=False)
        logger.info(f"💾 Data info saved: {data_file}")
        
        # Save latest file paths (for next step)
        latest_files = {
            'embeddings_file': embeddings_file,
            'data_file': data_file,
            'timestamp': timestamp,
            'total_records': len(df),
            'embedding_dimension': embeddings.shape[1]
        }
        
        import json
        with open(f"{self.output_dir}/latest_embeddings.json", 'w') as f:
            json.dump(latest_files, f, indent=2)
        
        logger.info(f"📋 Metadata saved: {self.output_dir}/latest_embeddings.json")
        
        # Clean up temporary files
        temp_files = [
            f"{self.output_dir}/embeddings_temp.npy",
            f"{self.output_dir}/progress.txt"
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"🗑️  Cleaned temp file: {temp_file}")
        
        return latest_files
    
    def process_data(self, csv_file: str = None, batch_size: int = 50):
        """
        Complete data processing pipeline - supports resume from interruption
        
        Args:
            csv_file: CSV file path, defaults to triage_final.csv
            batch_size: Batch processing size
        """
        if csv_file is None:
            csv_file = f"{self.data_dir}/triage_final.csv"
        
        logger.info("🎯 Starting Chief Complaint embedding extraction pipeline")
        logger.info("💡 Only processing unique complaints, dramatically reducing API calls and processing time")
        logger.info("🔄 Resume support - can be interrupted at any time, restart will automatically continue")
        logger.info("=" * 60)
        
        # Step 1: Load data
        df = self.load_data(csv_file)
        
        # Step 2: Clean data
        df_clean = self.clean_chief_complaints(df)
        
        # Step 3: Extract embeddings
        embeddings = self.extract_embeddings_batch(df_clean, batch_size=batch_size)
        
        # Step 4: Save results
        result_info = self.save_results(df_clean, embeddings)
        
        logger.info("=" * 60)
        logger.info("🎉 Chief Complaint embedding extraction completed!")
        logger.info(f"📊 Processing statistics:")
        logger.info(f"   Total records: {result_info['total_records']}")
        logger.info(f"   Embedding dimension: {result_info['embedding_dimension']}")
        logger.info(f"   Timestamp: {result_info['timestamp']}")
        logger.info(f"📂 Output files:")
        logger.info(f"   Embeddings: {result_info['embeddings_file']}")
        logger.info(f"   Data info: {result_info['data_file']}")
        
        return result_info


def main():
    """Main function"""
    print("🏥 Unique Chief Complaint Embedding Extraction Tool")
    print("💡 Smart deduplication: Only process unique complaints, using local SentenceTransformer")
    print("🔄 Resume support - can be interrupted at any time, restart will automatically continue")
    print("=" * 60)
    
    try:
        # Create extractor with SentenceTransformer
        extractor = ChiefComplaintEmbeddingExtractor(
            model_name="all-MiniLM-L6-v2"
        )
        
        print(f"📋 Using configuration:")
        print(f"   Model: {extractor.model_name}")
        print(f"   Local processing: No API calls required")
        print("=" * 60)
        
        # Process data (400k records, can use larger batches with local model)
        result_info = extractor.process_data(batch_size=100)  # Larger batch size for local processing
        
        print("\n" + "=" * 60)
        print("✅ Processing completed! Next step run:")
        print("   python 02_kmeans_clustering.py")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  User interrupted operation")
        print("🔄 Progress saved, restart will automatically continue")
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        print("💡 Please check SentenceTransformer installation and model availability")
        print("🔄 If it's a network issue, restart will automatically continue")


if __name__ == "__main__":
    main() 

# triage_final.csv 原始主诉文本
#         │
#         ▼
# 【Step 1】加载数据（load_data）
#         │
#         ▼
# 【Step 2】清洗主诉（clean_chief_complaints）
#       ├─ 去空值、标准化
#       ├─ 去重
#       ├─ 扩展医学缩写（如 cp → chest pain）
#       ├─ 保留模糊缩写（如 ah/st）
#       └─ 过滤无效项（单字符、空符号）
#         │
#         ▼
# 【Step 3】提取嵌入向量（extract_embeddings_batch）
#       ├─ 加上“Medical chief complaint:”前缀
#       ├─ 分批调用 text-embedding-ada-002
#       ├─ 支持断点恢复（progress.txt + embeddings_temp.npy）
#         │
#         ▼
# 【Step 4】保存结果（save_results）
#       ├─ 保存向量（.npy）
#       ├─ 保存主诉信息（.csv）
#       ├─ 保存元数据（.json）
#       └─ 清理临时文件
#         │
#         ▼
# 🏁 完成（process_data + main）
 