#!/usr/bin/env python3
"""
Step 2: KMeans clustering and select representative data
Perform KMeans clustering on chief complaint embeddings, and select representative data from each cluster, finally select 1000 samples
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set font for plots (if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class ChiefComplaintClustering:
    """Chief Complaint clustering analyzer"""
    
    def __init__(self, target_samples: int = 1000):
       
        self.target_samples = target_samples
        
        # 使用绝对路径避免相对路径问题
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data/embeddings")
        self.output_dir = os.path.join(script_dir, "data/clustering")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"✅ Clustering analyzer initialized, target samples: {target_samples}")
        logger.info(f"📁 Embeddings directory: {self.data_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_embeddings_data(self):
       
        logger.info("📂 Loading embedding data...")
        
        # Read latest embedding file info
        latest_file = f"{self.data_dir}/latest_embeddings.json"
        
        if not os.path.exists(latest_file):
            raise FileNotFoundError(f"Embedding metadata file not found: {latest_file}")
        
        with open(latest_file, 'r') as f:
            metadata = json.load(f)
        
        # Load embeddings
        embeddings = np.load(metadata['embeddings_file'])
        logger.info(f"✅ Embeddings loaded, shape: {embeddings.shape}")
        
        # Load corresponding data
        df = pd.read_csv(metadata['data_file'])
        logger.info(f"✅ Data info loaded, shape: {df.shape}")
        
        return embeddings, df, metadata
    
    def determine_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 50) -> int:
       
        logger.info("🔍 Determining optimal number of clusters...")
        
        # Consider target sample count, cluster number should be reasonable
        # If selecting 1000 samples, 20-100 clusters is reasonable
        min_clusters = max(10, self.target_samples // 100)  # At least 10 clusters
        max_clusters = min(max_clusters, self.target_samples // 5)  # At least 5 samples per cluster
        
        logger.info(f"   Cluster range: {min_clusters} - {max_clusters}")
        
        # Calculate evaluation metrics for different cluster numbers
        inertias = []
        silhouette_scores = []
        k_range = range(min_clusters, max_clusters + 1, 2)  # Test every 2
        
        for k in k_range:
            logger.info(f"   Testing k={k}...")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette coefficient (sample for large datasets)
            if len(embeddings) > 10000:
                sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
                sample_embeddings = embeddings[sample_indices]
                sample_labels = cluster_labels[sample_indices]
                sil_score = silhouette_score(sample_embeddings, sample_labels)
            else:
                sil_score = silhouette_score(embeddings, cluster_labels)
            
            silhouette_scores.append(sil_score)
            
            logger.info(f"      Silhouette score: {sil_score:.3f}")
        
        # Choose k with highest silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[best_k_idx]
        
        logger.info(f"✅ Optimal cluster number: {optimal_k} (silhouette score: {silhouette_scores[best_k_idx]:.3f})")
        
        # Save evaluation plot
        self._plot_cluster_evaluation(k_range, inertias, silhouette_scores, optimal_k)
        
        return optimal_k
    
    def _plot_cluster_evaluation(self, k_range, inertias, silhouette_scores, optimal_k):
        """Plot cluster evaluation graphs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow method plot
        ax1.plot(k_range, inertias, 'bo-')
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score plot
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"{self.output_dir}/cluster_evaluation_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Cluster evaluation plot saved: {plot_file}")
        
        plt.close()
    
    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Perform KMeans clustering
        
        Args:
            embeddings: Embedding vector matrix
            n_clusters: Number of clusters
            
        Returns:
            np.ndarray: Cluster labels
        """
        logger.info(f"🎯 Starting KMeans clustering, clusters: {n_clusters}")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering quality metrics
        inertia = kmeans.inertia_
        
        # Calculate silhouette coefficient (sample for large datasets)
        if len(embeddings) > 10000:
            sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            sample_labels = cluster_labels[sample_indices]
            silhouette_avg = silhouette_score(sample_embeddings, sample_labels)
        else:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        logger.info(f"✅ Clustering completed!")
        logger.info(f"   Inertia: {inertia:.2f}")
        logger.info(f"   Silhouette score: {silhouette_avg:.3f}")
        
        # Save clustering model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = f"{self.output_dir}/kmeans_model_{timestamp}.joblib"
        
        import joblib
        joblib.dump(kmeans, model_file)
        logger.info(f"💾 Clustering model saved: {model_file}")
        
        return cluster_labels
    
    def select_representative_samples(self, embeddings: np.ndarray, df: pd.DataFrame, 
                                   cluster_labels: np.ndarray) -> pd.DataFrame:
        """
        Select representative samples from each cluster
        
        Args:
            embeddings: Embedding vector matrix
            df: Dataframe
            cluster_labels: Cluster labels
            
        Returns:
            pd.DataFrame: Selected representative samples
        """
        logger.info(f"🎯 Starting representative sample selection, target count: {self.target_samples}")
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Analyze cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        logger.info(f"📊 Cluster size distribution:")
        for cluster_id, size in cluster_sizes.items():
            logger.info(f"   Cluster {cluster_id}: {size} samples")
        
        # Calculate how many samples each cluster should select
        total_samples = len(df)
        samples_per_cluster = {}
        
        for cluster_id, size in cluster_sizes.items():
            # Allocate proportionally, but ensure at least 1 per cluster
            proportion = size / total_samples
            target_count = max(1, int(self.target_samples * proportion))
            # Cannot exceed total count of the cluster
            target_count = min(target_count, size)
            samples_per_cluster[cluster_id] = target_count
        
        # Adjust selection count to reach target total
        current_total = sum(samples_per_cluster.values())
        if current_total != self.target_samples:
            # Sort by cluster size, adjust large clusters first
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            
            if current_total < self.target_samples:
                # Need to add samples
                need_more = self.target_samples - current_total
                for cluster_id, size in sorted_clusters:
                    if need_more <= 0:
                        break
                    can_add = min(need_more, size - samples_per_cluster[cluster_id])
                    samples_per_cluster[cluster_id] += can_add
                    need_more -= can_add
            else:
                # Need to reduce samples
                need_less = current_total - self.target_samples
                for cluster_id, size in sorted_clusters:
                    if need_less <= 0:
                        break
                    can_remove = min(need_less, samples_per_cluster[cluster_id] - 1)
                    samples_per_cluster[cluster_id] -= can_remove
                    need_less -= can_remove
        
        logger.info(f"📋 Selection plan for each cluster:")
        for cluster_id in sorted(samples_per_cluster.keys()):
            logger.info(f"   Cluster {cluster_id}: select {samples_per_cluster[cluster_id]} / {cluster_sizes[cluster_id]} samples")
        
        # Select representative samples from each cluster
        selected_samples = []
        
        for cluster_id in sorted(samples_per_cluster.keys()):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_embeddings = embeddings[cluster_mask]
            cluster_df = df_with_clusters[cluster_mask].copy()
            
            if len(cluster_df) == 0:
                continue
            
            target_count = samples_per_cluster[cluster_id]
            
            if target_count >= len(cluster_df):
                # Select all samples
                selected_indices = range(len(cluster_df))
            else:
                # Select samples closest to cluster center
                cluster_center = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distance from each sample to cluster center
                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                
                # Select closest samples
                selected_indices = np.argsort(distances)[:target_count]
            
            selected_cluster_samples = cluster_df.iloc[selected_indices].copy()
            selected_cluster_samples['distance_to_center'] = np.linalg.norm(
                cluster_embeddings[selected_indices] - np.mean(cluster_embeddings, axis=0), 
                axis=1
            )
            
            selected_samples.append(selected_cluster_samples)
            
            logger.info(f"   Cluster {cluster_id}: actually selected {len(selected_cluster_samples)} samples")
        
        # Merge all selected samples
        result_df = pd.concat(selected_samples, ignore_index=True)
        
        logger.info(f"✅ Representative sample selection completed!")
        logger.info(f"   Total selected: {len(result_df)} samples")
        logger.info(f"   Target count: {self.target_samples}")
        
        return result_df
    
    def analyze_clusters(self, df_selected: pd.DataFrame):
        """
        Analyze clustering results
        
        Args:
            df_selected: Selected representative samples
        """
        logger.info("📊 Starting cluster analysis...")
        
        # 1. Cluster distribution analysis
        cluster_analysis = df_selected.groupby('cluster').agg({
            'chiefcomplaint_clean': ['count', lambda x: list(x)[:3]],  # Count and first 3 examples
            'acuity': ['mean', 'std'] if 'acuity' in df_selected.columns else 'count',
            'age_at_visit': ['mean', 'std'] if 'age_at_visit' in df_selected.columns else 'count'
        }).round(2)
        
        logger.info("📈 Cluster statistics:")
        print(cluster_analysis)
        
        # 2. Typical chief complaints for each cluster
        logger.info("\n📝 Typical Chief Complaints for each cluster:")
        for cluster_id in sorted(df_selected['cluster'].unique()):
            cluster_data = df_selected[df_selected['cluster'] == cluster_id]
            complaints = cluster_data['chiefcomplaint_clean'].head(5).tolist()
            
            logger.info(f"\nCluster {cluster_id} (total {len(cluster_data)} samples):")
            for i, complaint in enumerate(complaints, 1):
                logger.info(f"  {i}. {complaint}")
        
        # 3. Save detailed analysis report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"{self.output_dir}/cluster_analysis_{timestamp}.txt"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("Chief Complaint Clustering Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total samples: {len(df_selected)}\n")
            f.write(f"Number of clusters: {df_selected['cluster'].nunique()}\n\n")
            
            # Detailed info for each cluster
            for cluster_id in sorted(df_selected['cluster'].unique()):
                cluster_data = df_selected[df_selected['cluster'] == cluster_id]
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(f"  Sample count: {len(cluster_data)}\n")
                
                if 'acuity' in cluster_data.columns:
                    f.write(f"  Average acuity: {cluster_data['acuity'].mean():.2f}\n")
                
                if 'age_at_visit' in cluster_data.columns:
                    f.write(f"  Average age: {cluster_data['age_at_visit'].mean():.1f}\n")
                
                f.write("  Typical symptoms:\n")
                complaints = cluster_data['chiefcomplaint_clean'].head(10).tolist()
                for i, complaint in enumerate(complaints, 1):
                    f.write(f"    {i}. {complaint}\n")
        
        logger.info(f"📋 Detailed analysis report saved: {analysis_file}")
    
    def visualize_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                          df_selected: pd.DataFrame):
        """
        Visualize clustering results
        
        Args:
            embeddings: Original embeddings
            cluster_labels: Cluster labels
            df_selected: Selected representative samples
        """
        logger.info("📊 Generating cluster visualization...")
        
        # Use PCA to reduce to 2D
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All data points clustering results
        scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='tab20', alpha=0.6, s=20)
        ax1.set_title('All Data Points - Cluster Visualization')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Plot 2: Selected representative samples
        selected_indices = df_selected.index.tolist()
        selected_embeddings_2d = embeddings_2d[selected_indices]
        selected_labels = cluster_labels[selected_indices]
        
        scatter2 = ax2.scatter(selected_embeddings_2d[:, 0], selected_embeddings_2d[:, 1],
                             c=selected_labels, cmap='tab20', alpha=0.8, s=50)
        ax2.set_title('Selected Representative Samples')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        
        # Add legends
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"{self.output_dir}/cluster_visualization_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Cluster visualization saved: {plot_file}")
        
        plt.close()
    
    def save_results(self, df_selected: pd.DataFrame, metadata: dict):
        """
        Save clustering and selection results
        
        Args:
            df_selected: Selected representative samples
            metadata: Original data metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save selected samples
        output_file = f"{self.output_dir}/representative_samples_{timestamp}.csv"
        df_selected.to_csv(output_file, index=False)
        logger.info(f"💾 Representative samples saved: {output_file}")
        
        # Save metadata
        result_metadata = {
            'timestamp': timestamp,
            'total_selected_samples': len(df_selected),
            'target_samples': self.target_samples,
            'n_clusters': df_selected['cluster'].nunique(),
            'source_embeddings': metadata['embeddings_file'],
            'source_data': metadata['data_file'],
            'output_file': output_file
        }
        
        metadata_file = f"{self.output_dir}/clustering_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(result_metadata, f, indent=2)
        
        # Save latest result info
        with open(f"{self.output_dir}/latest_clustering.json", 'w') as f:
            json.dump(result_metadata, f, indent=2)
        
        logger.info(f"📋 Clustering metadata saved: {metadata_file}")
        
        return result_metadata
    
    def process_clustering(self, auto_clusters: bool = True, n_clusters: int = None):
        """
        Complete clustering processing pipeline
        
        Args:
            auto_clusters: Whether to automatically determine cluster count
            n_clusters: Manually specified cluster count (used when auto_clusters=False)
        """
        logger.info("🎯 Starting Chief Complaint clustering analysis pipeline")
        logger.info("=" * 60)
        
        # Check if clustering results already exist
        latest_clustering_file = f"{self.output_dir}/latest_clustering.json"
        if os.path.exists(latest_clustering_file):
            with open(latest_clustering_file, 'r') as f:
                existing_result = json.load(f)
            
            logger.info("🔍 Found existing clustering results:")
            logger.info(f"   Timestamp: {existing_result['timestamp']}")
            logger.info(f"   Sample count: {existing_result['total_selected_samples']}")
            logger.info(f"   Cluster count: {existing_result['n_clusters']}")
            logger.info(f"   Result file: {existing_result['output_file']}")
            
            if os.path.exists(existing_result['output_file']):
                logger.info("✅ Clustering result file exists and is complete")
                logger.info("💡 To re-cluster, please delete latest_clustering.json file first")
                logger.info("=" * 60)
                return existing_result
            else:
                logger.info("⚠️  Result file missing, will re-perform clustering analysis")
        
        # Step 1: Load data
        embeddings, df, metadata = self.load_embeddings_data()
        
        # Step 2: Determine cluster count
        if auto_clusters:
            optimal_clusters = self.determine_optimal_clusters(embeddings)
        else:
            optimal_clusters = n_clusters or 30
            logger.info(f"🎯 Using manually specified cluster count: {optimal_clusters}")
        
        # Step 3: Perform clustering
        cluster_labels = self.perform_clustering(embeddings, optimal_clusters)
        
        # Step 4: Select representative samples
        df_selected = self.select_representative_samples(embeddings, df, cluster_labels)
        
        # Step 5: Analyze results
        self.analyze_clusters(df_selected)
        
        # Step 6: Visualize
        self.visualize_clusters(embeddings, cluster_labels, df_selected)
        
        # Step 7: Save results
        result_metadata = self.save_results(df_selected, metadata)
        
        logger.info("=" * 60)
        logger.info("🎉 Clustering analysis completed!")
        logger.info(f"📊 Result statistics:")
        logger.info(f"   Original data size: {len(df)}")
        logger.info(f"   Cluster count: {optimal_clusters}")
        logger.info(f"   Selected samples: {len(df_selected)}")
        logger.info(f"   Target samples: {self.target_samples}")
        logger.info(f"📂 Output file: {result_metadata['output_file']}")
        
        return result_metadata


def main():
    """Main function"""
    print("🔬 Chief Complaint Clustering Analysis Tool")
    print("=" * 60)
    
    # Set target sample count
    TARGET_SAMPLES = 1000
    
    try:
        # Create clustering analyzer
        clustering = ChiefComplaintClustering(target_samples=TARGET_SAMPLES)
        
        # Execute clustering analysis
        # Execute clustering analysis with fixed cluster number
        result = clustering.process_clustering(auto_clusters=False, n_clusters=60)

        
        print("\n" + "=" * 60)
        print("✅ Clustering analysis completed!")
        print(f"📊 Selected {result['total_selected_samples']} representative samples")
        print(f"📂 Result file: {result['output_file']}")
        print("\n💡 Next steps:")
        print("   1. Review clustering analysis report")
        print("   2. Use selected samples for further analysis")
        print("   3. Train classification models")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        print("💡 Please ensure 01_extract_embeddings.py has been run to generate embeddings")


if __name__ == "__main__":
    main() 


# 📁 已有输入：chief complaint 的 embedding + 原始文本数据

#        ↓
# 1. 加载数据（load_embeddings_data）

#        ↓
# 2. 自动寻找最佳聚类数 K（determine_optimal_clusters）
#     - 遍历 K ∈ [min_k, max_k]
#     - 记录每个 K 的 inertia、silhouette score
#     - 可视化 K 的选择效果（保存图）

#        ↓
# 3. 用最佳 K 进行 KMeans 聚类（perform_clustering）
#     - 输出每条数据的 cluster 标签

#        ↓
# 4. 每个 cluster 中选择代表样本（select_representative_samples）
#     - 每类按样本占比分配数量
#     - 再从中挑离中心最近的点作为“代表”

#        ↓
# 5. 分析聚类结果（analyze_clusters）
#     - 每类中前几个主诉、平均年龄、acuity 等统计
#     - 生成分析报告 txt 文件

#        ↓
# 6. 聚类可视化（visualize_clusters）
#     - PCA 降维后展示整体分布 + 代表样本

#        ↓
# 7. 保存代表样本和元数据（save_results）
#     - 保存选中样本 CSV
#     - 保存聚类元数据 JSON

#        ↓
# 🎉 完成聚类分析
