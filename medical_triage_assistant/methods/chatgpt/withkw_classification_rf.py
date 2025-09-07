#!/usr/bin/env python3
"""
RandomForest Classification with Keywords for Medical Triage
Perform RandomForest classification on medical triage data with keyword features
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, make_scorer, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set font for plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class ClassificationRandomForestTrainer:
    """RandomForest classification trainer for medical triage"""
    
    def __init__(self, force_retrain=False):
        """Initialize trainer with paths and settings"""
        # Set up directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data/two_stage_classification")
        self.models_dir = os.path.join(script_dir, "models")
        
        # Setup reports directory
        self.project_root = os.path.join(script_dir, '../..')
        self.reports_dir = os.path.join(self.project_root, "reports")
        self._setup_reports_directories()
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Define model paths
        self.model_path = os.path.join(self.models_dir, "GPT-rf.pkl")
        
        # Check if model already exists
        if not force_retrain and self._check_existing_model():
            print("✅ Model already exists and will be skipped!")
            exit(0)
        
        # Initialize preprocessing objects
        self.scaler = None
        self.one_hot_encoder = None
        self.onehot_feature_names = []
        self.label_encoders = {}
        self.feature_names = None
        
        logger.info(f"✅ RandomForest trainer initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Models directory: {self.models_dir}")
    
    def _check_existing_model(self):
        """Check if model files already exist"""
        # Define all expected files
        per_class_report_path = os.path.join(self.models_dir, "per_class_performance_rf.csv")
        confusion_matrix_path = os.path.join(self.models_dir, "confusion_matrix_rf.png")
        detailed_analysis_path = os.path.join(self.models_dir, "detailed_performance_analysis_rf.csv")
        
        files_to_check = [
            (self.model_path, "Model"),
            (per_class_report_path, "Per-class performance report"),
            (confusion_matrix_path, "Confusion matrix"),
            (detailed_analysis_path, "Detailed performance analysis")
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                existing_files.append((file_path, description))
            else:
                missing_files.append((file_path, description))
        
        if existing_files and not missing_files:
            print(f"✅ All model files already exist:")
            for file_path, description in existing_files:
                print(f"   - {description}: {file_path}")
            print("🔄 To retrain the model, use --force flag")
            print("   Example: python withkw_classification_rf.py --force")
            return True
        elif existing_files and missing_files:
            print(f"⚠️ Some model files exist but others are missing:")
            print(f"   Existing files:")
            for file_path, description in existing_files:
                print(f"   - {description}: {file_path}")
            print(f"   Missing files:")
            for file_path, description in missing_files:
                print(f"   - {description}: {file_path}")
            print("🔄 Will retrain to generate all required files...")
            return False
        else:
            print("📝 No existing model files found. Starting training...")
            return False
    
    def _setup_reports_directories(self):
        """Setup reports directories"""
        os.makedirs(os.path.join(self.reports_dir, "evaluation_results"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "model_performance"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "visualization"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "experiments"), exist_ok=True)
    
    def load_data(self):
        """Load training data"""
        print("📂 Loading training data...")
        
        # Load the processed data with keywords
        data_path = os.path.join(self.data_dir, "triage_with_classification_filtered_20250831_231646.csv")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {len(df):,} samples, {len(df.columns)} features")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for RandomForest training"""
        print("🔧 Preprocessing data...")
        
        # Separate features and target
        target_col = 'acuity'
        
        # Exclude future information, text features, and redundant features
        exclude_cols = [
            'acuity',                # Target variable
            'outtime',              # Future: 出院时间 (患者还未入院)
            'disposition',          # Future: 处置结果 (治疗后的结果)
            'intime',               # Future: 入院时间 (分诊时还未入院)
            'subject_id',           # Not predictive: 患者ID (无预测价值)
            'chiefcomplaint',       # Text feature: 原始主诉 (已通过ChatGPT特征提取语义)
            'chiefcomplaint_clean', # Text feature: 清理后主诉 (已通过ChatGPT特征提取语义)
            'group_id',             # Redundant: 与group_name表示相同信息
            'severity_confidence',  # Auxiliary: 置信度信息，简化模型只用核心分类结果
            'classification_found'  # Auxiliary: 成功标志，简化模型只用核心分类结果
        ]
        
        # Only use core structured features available at triage time:
        # - Vital signs: pain, temperature, heartrate, sbp, dbp, o2sat
        # - Demographics: age_at_visit, gender  
        # - Arrival info: arrival_transport
        # - ChatGPT hierarchical classification:
        #   * group_name: 症状组类别 (如"心血管疾病", "消化系统疾病")
        #   * severity: 在该症状组内的严重程度等级 (1-5)
        #   组合示例: "心血管疾病" + severity=3 → "心血管疾病：严重程度3"
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"🚫 Excluded columns to prevent data leakage: {exclude_cols}")
        print(f"✅ Using {len(feature_cols)} features for training")
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical features with mixed encoding
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        import pandas as pd
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"📊 数值特征 ({len(numerical_cols)}): {numerical_cols}")
        print(f"📊 分类特征 ({len(categorical_cols)}): {categorical_cols}")
        
        # 分离不同类型的分类特征
        onehot_features = ['gender', 'arrival_transport']  # 无序分类变量
        label_features = ['group_name', 'severity']        # 有序/层次化分类变量
        
        # 实际存在的独热编码特征
        onehot_cols = [col for col in onehot_features if col in categorical_cols]
        label_cols = [col for col in label_features if col in categorical_cols]
        
        print(f"📊 独热编码特征: {onehot_cols}")
        print(f"📊 标签编码特征: {label_cols}")
        
        # 1. 处理独热编码特征
        if onehot_cols:
            X_onehot = X[onehot_cols]
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            X_onehot_encoded = encoder.fit_transform(X_onehot)
            
            # 获取编码后的特征名
            onehot_feature_names = encoder.get_feature_names_out(onehot_cols)
            X_onehot_df = pd.DataFrame(X_onehot_encoded, 
                                     columns=onehot_feature_names,
                                     index=X.index)
            
            print(f"📊 独热编码后特征数: {X_onehot_df.shape[1]}")
            print(f"📊 独热编码特征名: {list(onehot_feature_names)}")
            
            self.one_hot_encoder = encoder
            self.onehot_feature_names = onehot_cols
        else:
            X_onehot_df = pd.DataFrame(index=X.index)
            self.one_hot_encoder = None
            self.onehot_feature_names = []
        
        # 2. 处理标签编码特征
        self.label_encoders = {}
        if label_cols:
            X_label = X[label_cols].copy()
            for col in label_cols:
                le = LabelEncoder()
                X_label[col] = le.fit_transform(X_label[col].astype(str))
                self.label_encoders[col] = le
                print(f"   标签编码特征: {col} -> {X_label[col].nunique()} 个类别")
        else:
            X_label = pd.DataFrame(index=X.index)
        
        # 3. 合并所有特征
        X_numerical = X[numerical_cols] if numerical_cols else pd.DataFrame(index=X.index)
        
        # 按顺序合并：数值特征 + 独热编码特征 + 标签编码特征
        X = pd.concat([X_numerical, X_onehot_df, X_label], axis=1)
        
        print(f"📊 最终特征矩阵形状: {X.shape}")
        print(f"📊 最终特征列: {list(X.columns)}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert target to 0-based indexing (ESI 1-5 -> 0-4)
        y = y - 1
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Target distribution:")
        print(y.value_counts().sort_index())
        
        return X, y
    
    def train_model(self, X, y):
        """Train RandomForest model with GridSearchCV"""
        print("🚀 Training RandomForest model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        print(f"📊 Training set: {len(X_train):,} samples")
        print(f"📊 Test set: {len(X_test):,} samples")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # -------------------------------
        # 🔍 Define parameter search space
        # -------------------------------
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "class_weight": [
                None,
                "balanced",
                {0: 10, 1: 8, 2: 3, 3: 2, 4: 1}
            ]
        }

        # Base model
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            criterion="gini",
            min_samples_split=2,
            min_samples_leaf=1
        )

        # Setup GridSearchCV
        f1_weighted = make_scorer(f1_score, average="weighted")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=f1_weighted,
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        print(f"🚀 Running GridSearchCV with {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['class_weight'])} combinations...")
        grid_search.fit(X_train_scaled, y_train)

        print(f"✅ Best parameters found:")
        print(f"   {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")

        # Save best model
        self.model = grid_search.best_estimator_
        self.grid_search_results = pd.DataFrame(grid_search.cv_results_)

        # -------------------------------
        # 📊 Evaluate model
        # -------------------------------
        y_pred = self.model.predict(X_test_scaled)

        # Convert back to ESI 1-5
        y_test_esi = y_test + 1
        y_pred_esi = y_pred + 1

        # Accuracy & Weighted F1
        accuracy = accuracy_score(y_test_esi, y_pred_esi)
        weighted_f1 = f1_score(y_test_esi, y_pred_esi, average="weighted")

        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Weighted F1: {weighted_f1:.3f}")

        # Per-Class Performance Report (Table 5.4 style)
        report_dict = classification_report(
            y_test_esi,
            y_pred_esi,
            target_names=[f"ESI {i}" for i in range(1, 6)],
            output_dict=True,
            digits=2
        )
        
        # Create detailed performance table
        performance_data = []
        for i in range(1, 6):
            esi_key = f"ESI {i}"
            if esi_key in report_dict:
                performance_data.append({
                    'ESI Level': f'ESI {i}',
                    'Precision': report_dict[esi_key]['precision'],
                    'Recall': report_dict[esi_key]['recall'],
                    'F1': report_dict[esi_key]['f1-score']
                })
        
        # Add overall metrics
        performance_data.append({
            'ESI Level': 'Accuracy',
            'Precision': '',
            'Recall': '',
            'F1': accuracy
        })
        performance_data.append({
            'ESI Level': 'Macro Avg',
            'Precision': report_dict['macro avg']['precision'],
            'Recall': report_dict['macro avg']['recall'],
            'F1': report_dict['macro avg']['f1-score']
        })
        performance_data.append({
            'ESI Level': 'Weighted Avg',
            'Precision': report_dict['weighted avg']['precision'],
            'Recall': report_dict['weighted avg']['recall'],
            'F1': report_dict['weighted avg']['f1-score']
        })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Save detailed performance report
        report_path = os.path.join(self.models_dir, "per_class_performance_rf.csv")
        performance_df.to_csv(report_path, index=False)
        print(f"✅ Per-class performance saved: {report_path}")
        
        # Also save to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_perf_path = os.path.join(self.reports_dir, "model_performance", f"per_class_performance_rf_chatgpt_{timestamp}.csv")
        performance_df.to_csv(reports_perf_path, index=False)
        print(f"✅ Per-class performance also saved to reports: {reports_perf_path}")
        
        # Print formatted performance table
        print(f"\n📊 Table 5.4: Per-class performance on the held-out test set (Scheme A, RandomForest)")
        print("=" * 80)
        print(f"{'ESI Level':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        for _, row in performance_df.iterrows():
            if row['ESI Level'] == 'Accuracy':
                print(f"{row['ESI Level']:<12} {'':<10} {'':<10} {row['F1']:<10.2f}")
            else:
                print(f"{row['ESI Level']:<12} {row['Precision']:<10.2f} {row['Recall']:<10.2f} {row['F1']:<10.2f}")
        print("=" * 80)

        # Enhanced Confusion Matrix (Best RandomForest style)
        cm = confusion_matrix(y_test_esi, y_pred_esi)
        cm_path = os.path.join(self.models_dir, "confusion_matrix_rf.png")

        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better visualization
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with better proportions
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap with enhanced styling
        heatmap = sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=[f"ESI {i}" for i in range(1, 6)],
            yticklabels=[f"ESI {i}" for i in range(1, 6)],
            cbar_kws={'label': 'Count'},
            ax=ax
        )
        
        # Customize labels and title
        plt.xlabel("Predicted Triage Level", fontsize=12, fontweight='bold')
        plt.ylabel("True Triage Level", fontsize=12, fontweight='bold')
        plt.title("Confusion Matrix - Best RandomForest with Keywords", fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Add grid lines for better readability
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save to original location (for backward compatibility)
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Enhanced confusion matrix saved: {cm_path}")
        
        # Also save to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_cm_path = os.path.join(self.reports_dir, "visualization", f"confusion_matrix_rf_chatgpt_{timestamp}.png")
        plt.savefig(reports_cm_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix also saved to reports: {reports_cm_path}")
        
        plt.close()
        
        # Print confusion matrix summary
        print(f"\n📊 Confusion Matrix Summary:")
        print("=" * 50)
        total_correct = np.sum(np.diag(cm))
        total_samples = np.sum(cm)
        print(f"Total samples: {total_samples:,}")
        print(f"Correct predictions: {total_correct:,}")
        print(f"Accuracy: {total_correct/total_samples:.3f}")
        
        # Per-class correct predictions
        print(f"\nPer-class correct predictions:")
        for i in range(5):
            print(f"ESI {i+1}: {cm[i,i]:,} correct out of {np.sum(cm[i,:]):,} total")

        # Generate detailed analysis report
        detailed_report = self.generate_detailed_analysis_report(y_test_esi, y_pred_esi, cm)

        return {
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
            "per_class_report": performance_df,
            "confusion_matrix": cm,
            "detailed_report": detailed_report
        }

    def generate_detailed_analysis_report(self, y_test_esi, y_pred_esi, cm):
        """Generate detailed analysis report similar to the images"""
        print("\n📊 Generating detailed analysis report...")
        
        # Calculate additional metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_esi, y_pred_esi, average=None, labels=range(1, 6)
        )
        
        # Create comprehensive report
        report_data = []
        for i in range(5):
            esi_level = i + 1
            true_pos = cm[i, i]
            false_pos = np.sum(cm[:, i]) - true_pos
            false_neg = np.sum(cm[i, :]) - true_pos
            
            report_data.append({
                'ESI Level': f'ESI {esi_level}',
                'Precision': precision[i],
                'Recall': recall[i],
                'F1': f1[i],
                'Support': support[i],
                'True Positives': true_pos,
                'False Positives': false_pos,
                'False Negatives': false_neg
            })
        
        # Save detailed report
        detailed_report_path = os.path.join(self.models_dir, "detailed_performance_analysis_rf.csv")
        detailed_df = pd.DataFrame(report_data)
        detailed_df.to_csv(detailed_report_path, index=False)
        print(f"✅ Detailed analysis report saved: {detailed_report_path}")
        
        # Also save to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_detailed_path = os.path.join(self.reports_dir, "model_performance", f"detailed_performance_analysis_rf_chatgpt_{timestamp}.csv")
        detailed_df.to_csv(reports_detailed_path, index=False)
        print(f"✅ Detailed analysis also saved to reports: {reports_detailed_path}")
        
        # Print detailed analysis
        print(f"\n📊 Detailed Performance Analysis:")
        print("=" * 100)
        print(f"{'ESI Level':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 100)
        for _, row in detailed_df.iterrows():
            print(f"{row['ESI Level']:<10} {row['Precision']:<10.2f} {row['Recall']:<10.2f} {row['F1']:<10.2f} {row['Support']:<10.0f} {row['True Positives']:<8} {row['False Positives']:<8} {row['False Negatives']:<8}")
        print("=" * 100)
        
        return detailed_df

    def save_model(self):
        """Save trained model and preprocessing objects"""
        print("💾 Saving model and preprocessing objects...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"randomforest_with_classification_{timestamp}.pkl")
        joblib.dump(self.model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f"scaler_rf_classification_{timestamp}.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save one-hot encoder
        if hasattr(self, 'one_hot_encoder') and self.one_hot_encoder is not None:
            encoder_path = os.path.join(self.models_dir, f"onehot_encoder_rf_classification_{timestamp}.pkl")
            joblib.dump(self.one_hot_encoder, encoder_path)
            print(f"✅ One-hot encoder saved: {encoder_path}")
            
            # Save onehot feature names
            onehot_features_path = os.path.join(self.models_dir, f"onehot_features_rf_classification_{timestamp}.pkl")
            joblib.dump(self.onehot_feature_names, onehot_features_path)
            print(f"✅ One-hot feature names saved: {onehot_features_path}")
        else:
            print("⚠️ No one-hot features to save encoder for")
        
        # Save label encoders
        if hasattr(self, 'label_encoders') and self.label_encoders:
            label_encoders_path = os.path.join(self.models_dir, f"label_encoders_rf_classification_{timestamp}.pkl")
            joblib.dump(self.label_encoders, label_encoders_path)
            print(f"✅ Label encoders saved: {label_encoders_path}")
        else:
            print("⚠️ No label encoders to save")
        
        # Save feature names
        features_path = os.path.join(self.models_dir, f"feature_names_rf_classification_{timestamp}.txt")
        with open(features_path, 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        print(f"✅ Feature names saved: {features_path}")
        
        # Save GridSearchCV results
        if hasattr(self, 'grid_search_results'):
            gridsearch_path = os.path.join(self.models_dir, f"gridsearch_results_rf_classification_{timestamp}.csv")
            self.grid_search_results.to_csv(gridsearch_path, index=False)
            print(f"✅ GridSearchCV results saved: {gridsearch_path}")
            
            # Also save to reports directory
            reports_grid_path = os.path.join(self.reports_dir, "experiments", f"gridsearch_results_rf_chatgpt_{timestamp}.csv")
            self.grid_search_results.to_csv(reports_grid_path, index=False)
            print(f"✅ GridSearchCV results also saved to reports: {reports_grid_path}")
        
        # Save best model as default
        best_model_path = os.path.join(self.models_dir, "GPT-rf.pkl")
        joblib.dump(self.model, best_model_path)
        print(f"✅ Best model saved: {best_model_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path,
            'best_model_path': best_model_path
        }
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("🏥 Classification RandomForest Training Pipeline")
        print("=" * 60)
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess data
            X, y = self.preprocess_data(df)
            
            # Train model
            results = self.train_model(X, y)
            
            # Save model
            saved_paths = self.save_model()
            
            print("\n" + "=" * 60)
            print("🎉 Training completed successfully!")
            print(f"📊 Final Results:")
            print(f"   Accuracy: {results['accuracy']:.3f}")
            print(f"   Weighted F1: {results['weighted_f1']:.3f}")
            print(f"📂 Model files saved in: {self.models_dir}")
            print("=" * 60)
            
            return results, saved_paths
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ChatGPT + Random Forest classification model')
    parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
    args = parser.parse_args()
    
    trainer = ClassificationRandomForestTrainer(force_retrain=args.force)
    results, saved_paths = trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
