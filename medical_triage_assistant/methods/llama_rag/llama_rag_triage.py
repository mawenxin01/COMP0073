#!/usr/bin/env python3
"""
Llama-3.3-70B RAG-Enhanced Triage System
Integrates retrieval and generation modules for complete Llama-based triage solution
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from medical_triage_assistant.methods.llama_rag.retrieval.case_retriever import CaseRetriever
from medical_triage_assistant.methods.llama_rag.generation.llama_generator import LlamaGenerator


class LlamaRAGTriageSystem:
    """Llama-3.3-70B RAG-Enhanced Triage System"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", llama_model: str = "meta-llama/llama-3-3-70b-instruct"):
        """
        Initialize Llama RAG triage system
        
        Args:
            embedding_model: Model for text embeddings
            llama_model: Llama model name
        """
        self.case_retriever = CaseRetriever(embedding_model)
        self.llama_generator = LlamaGenerator(llama_model)
        self.is_initialized = False
        
        # Setup reports directory
        self.project_root = os.path.join(os.path.dirname(__file__), '../..')
        self.reports_dir = os.path.join(self.project_root, "reports")
        self._setup_reports_directories()
        
        print(f"🦙 Llama RAG Triage System initialized successfully")
        print(f"   Embedding model: {embedding_model}")
        print(f"   Llama model: {llama_model}")
    
    def _setup_reports_directories(self):
        """Setup reports directories"""
        os.makedirs(os.path.join(self.reports_dir, "evaluation_results"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "model_performance"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "visualization"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "experiments"), exist_ok=True)
    
    def initialize_system(self, train_data: pd.DataFrame, save_path: str = "llama_rag_system_index", 
                          force_rebuild: bool = False):
        """
        Initialize Llama RAG triage system
        
        Args:
            train_data: Training data (should be training set, not including test data)
            save_path: FAISS index save path
            force_rebuild: Whether to force rebuild the index
        """
        
        print(f"🔧 Initializing Llama RAG system...")
        print(f"   Training data size: {len(train_data)}")
        print(f"   Index save path: {save_path}")
        
        # Validate training data doesn't include test data
        self._validate_training_data(train_data)
        
        # Initialize case retriever directly with DataFrame
        print(f"🔍 Initializing case retriever...")
        try:
            # Prepare case database
            prepared_data = self.case_retriever.vector_store.prepare_case_database(train_data)
            
            # Build or load FAISS index
            was_cached = self.case_retriever.vector_store.build_or_load_faiss_index(
                prepared_data, 
                save_path=save_path, 
                force_rebuild=force_rebuild
            )
            
            # Mark as initialized
            self.case_retriever.initialized = True
            self.is_initialized = True
            
            print(f"✅ Llama RAG system initialized successfully")
            print(f"   Total training cases: {len(prepared_data)}")
            print(f"   Index cached: {was_cached}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Llama RAG system: {e}")
            self.is_initialized = False
    
    def _validate_training_data(self, data: pd.DataFrame):
        """Validate training data to prevent data leakage"""
        
        print(f"🔍 Validating training data...")
        
        expected_train_size = 246943  # Based on 80% of ~308K valid cases
        tolerance = 5000  # Allow 5K tolerance
        
        if len(data) > (expected_train_size + tolerance):
            print(f"⚠️ Warning: Training data size ({len(data)}) larger than expected ({expected_train_size})")
            print(f"   This might indicate data leakage (test data included in training)")
        elif len(data) < (expected_train_size - tolerance):
            print(f"⚠️ Warning: Training data size ({len(data)}) smaller than expected ({expected_train_size})")
        else:
            print(f"✅ Training data size validation passed: {len(data)} cases")
    
    @staticmethod
    def extract_time_features(time_str: str) -> Dict:
        """Extract useful time features from time string - uses unified time processor"""
        
        # Since time_period is already in the data, we just return it directly
        # This method is kept for compatibility but not used
        return {
            'time_period': 2,  # Default to afternoon
        }
    
    def predict_triage_level(self, case_data: Dict, k: int = 5) -> Tuple[int, str, List[Dict], List[float]]:
        """
        Predict triage level for a case
        
        Args:
            case_data: Patient case data dictionary
            k: Number of similar cases to retrieve
            
        Returns:
            Tuple containing triage level, reasoning, similar cases, and similarities
        """
        
        if not self.is_initialized:
            raise ValueError("System not initialized. Please call initialize_system first.")
        
        start_time = time.time()
        
        # Field mapping for compatibility
        field_mapping = {
            'patient_id': 'subject_id',
            'chief_complaint': 'chiefcomplaint',
            'complaint_keywords': 'complaint_keywords',
            'age': 'age_at_visit',
            'heart_rate': 'heartrate',
            'blood_pressure_systolic': 'sbp',
            'blood_pressure_diastolic': 'dbp',
            'temperature': 'temperature',
            'oxygen_saturation': 'o2sat',
            'pain_score': 'pain',
            'gender': 'gender',
            'arrival_method': 'arrival_transport',
            'arrival_time': 'intime',
            'time_period': 'time_period'  # Add time period mapping
        }
        
        # Normalize field names
        normalized_case = {}
        for api_field, db_field in field_mapping.items():
            if api_field in case_data:
                normalized_case[db_field] = case_data[api_field]
        
        # Add any additional fields that weren't mapped
        for key, value in case_data.items():
            if key not in field_mapping:
                normalized_case[key] = value
        
        # Process time features if intime is provided but time_period isn't
        if 'intime' in case_data and 'time_period' not in normalized_case:
            time_features = self.extract_time_features(case_data['intime'])
            normalized_case.update(time_features)
        elif 'time_period' not in normalized_case:
            # If no time information, use default values
            time_features = self.extract_time_features(None)
            normalized_case.update(time_features)
        
        try:
            # Retrieve similar cases with vector-only retrieval (fastest)
            similar_cases, similarities, metadata = self.case_retriever.retrieve_similar_cases(
                normalized_case, k=k, use_hybrid=False, use_reranking=False
            )
            
            if "error" in metadata:
                return 3, f"Retrieval error: {metadata['error']}", [], []
            
            # Generate triage decision using Llama
            triage_level, reasoning = self.llama_generator.generate_triage_decision(
                pd.Series(normalized_case), similar_cases, similarities
            )
            
            processing_time = time.time() - start_time
            print(f"⏱️ Prediction completed in {processing_time:.2f}s")
            
            return triage_level, reasoning, similar_cases, similarities
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return 3, f"Error: {str(e)}", [], []
    
    def run_full_pipeline(self, data_path: str, test_size: int = 100, 
                         save_path: str = "shared_rag_train_index",
                         force_rebuild: bool = False) -> Dict:
        """
        Run complete pipeline: data loading, train/test split, initialization, evaluation
        
        Args:
            data_path: Path to complete dataset
            test_size: Number of test samples
            save_path: Index save path
            force_rebuild: Whether to force rebuild index
            
        Returns:
            Dict: Evaluation results
        """
        
        print(f"🚀 Running Llama RAG full pipeline...")
        print(f"   Data path: {data_path}")
        print(f"   Test size: {test_size}")
        print(f"   Save path: {save_path}")
        
        # Load data
        print(f"📖 Loading data...")
        data = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Data loaded: {len(data)} total cases")
        
        # Filter valid cases (non-null acuity and required fields)
        initial_count = len(data)
        data = data.dropna(subset=['acuity', 'chiefcomplaint'])
        valid_count = len(data)
        print(f"📊 Valid cases: {valid_count}/{initial_count} ({valid_count/initial_count:.1%})")
        
        # Stratified train/test split (80/20)
        print(f"🔄 Performing stratified train/test split...")
        train_data, test_data = train_test_split(
            data, 
            test_size=0.2,  # 20% for testing
            random_state=42, 
            stratify=data['acuity']
        )
        
        print(f"📊 Data split completed:")
        print(f"   Training set: {len(train_data)} cases")
        print(f"   Test set: {len(test_data)} cases")
        print(f"   Test ratio: {len(test_data)/len(data):.1%}")
        
        # Use only the specified test_size for evaluation
        if len(test_data) > test_size:
            test_data = test_data.sample(n=test_size, random_state=42)
            print(f"📊 Using {test_size} test samples for evaluation")
        
        # Initialize system with training data only
        self.initialize_system(train_data, save_path=save_path, force_rebuild=force_rebuild)
        
        if not self.is_initialized:
            return {"error": "System initialization failed"}
        
        # Run evaluation
        print(f"🧪 Running evaluation on {len(test_data)} test cases...")
        
        # Choose evaluation mode
        if test_size >= 100:  # For larger test sets, use multi-config evaluation
            print(f"🔍 Using multi-config evaluation (vector_only, hybrid_only, hybrid_rerank)")
            return self._evaluate_system_multi(test_data)
        else:
            print(f"🔍 Using single-config evaluation (hybrid_only)")
            return self._evaluate_system(test_data)
    
    def _evaluate_system(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate system performance on test data"""
        
        predictions = []
        actual_labels = []
        processing_times = []
        error_count = 0
        
        print(f"🔍 Evaluating {len(test_data)} test cases...")
        
        for idx, (_, case) in enumerate(test_data.iterrows()):
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{len(test_data)} ({idx/len(test_data):.1%})")
            
            try:
                start_time = time.time()
                
                # Convert case to dictionary
                case_dict = case.to_dict()
                
                # Predict with hybrid retrieval only (no reranking)
                predicted_level, reasoning, similar_cases, similarities = self.predict_triage_level(
                    case_dict, k=5
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                predictions.append(predicted_level)
                actual_labels.append(int(case['acuity']))
                
            except Exception as e:
                print(f"❌ Error processing case {idx}: {e}")
                error_count += 1
                predictions.append(3)  # Default to moderate
                actual_labels.append(int(case['acuity']))
                processing_times.append(0)
        
        # Calculate metrics
        accuracy = np.mean([p == a for p, a in zip(predictions, actual_labels)])
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Get error statistics from Llama generator
        error_stats = self.llama_generator.get_error_statistics()
        
        results = {
            "evaluation_completed": True,
            "test_cases": len(test_data),
            "successful_predictions": len(predictions) - error_count,
            "errors": error_count,
            "accuracy": accuracy,
            "avg_processing_time": avg_processing_time,
            "predictions": predictions,
            "actual_labels": actual_labels,
            "classification_report": classification_report(actual_labels, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(actual_labels, predictions).tolist(),
            "llama_error_stats": error_stats
        }
        
        print(f"📊 Evaluation Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Avg processing time: {avg_processing_time:.2f}s")
        print(f"   Successful predictions: {len(predictions) - error_count}/{len(test_data)}")
        print(f"   Llama success rate: {error_stats['success_rate']:.3f}")
        
        # Display detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print("-" * 50)
        
        # Get classification report as string for display
        report_str = classification_report(actual_labels, predictions, target_names=[f'ESI {i}' for i in range(1, 6)])
        print(report_str)
        
        # Display confusion matrix
        print(f"\n🔢 Confusion Matrix:")
        print("-" * 30)
        cm = confusion_matrix(actual_labels, predictions)
        print("Predicted →")
        print("Actual ↓")
        print("     ", end="")
        for i in range(1, 6):
            print(f"ESI{i:>6}", end="")
        print()
        
        for i in range(5):
            print(f"ESI{i+1:>5}", end="")
            for j in range(5):
                print(f"{cm[i][j]:>6}", end="")
            print()
        
        # Display F1 scores by class
        print(f"\n🎯 F1 Scores by ESI Level:")
        print("-" * 30)
        for i in range(1, 6):
            if i in actual_labels:
                # Calculate F1 for this class
                precision = results['classification_report'][f'{i}']['precision']
                recall = results['classification_report'][f'{i}']['recall']
                f1 = results['classification_report'][f'{i}']['f1-score']
                support = results['classification_report'][f'{i}']['support']
                print(f"   ESI {i}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, Support={support}")
            else:
                print(f"   ESI {i}: No samples in test set")
        
        # Display macro and weighted averages
        macro_f1 = results['classification_report']['macro avg']['f1-score']
        weighted_f1 = results['classification_report']['weighted avg']['f1-score']
        print(f"\n📊 Overall F1 Scores:")
        print(f"   Macro F1: {macro_f1:.3f}")
        print(f"   Weighted F1: {weighted_f1:.3f}")
        
        # Generate enhanced reports and visualizations
        print(f"\n🎨 Generating enhanced reports and visualizations...")
        
        # Generate confusion matrix plot
        cm_plot = self.generate_confusion_matrix_plot(actual_labels, predictions)
        
        # Generate formatted classification report
        formatted_report = self.generate_formatted_classification_report(actual_labels, predictions)
        
        # Generate detailed analysis report
        detailed_report = self.generate_detailed_analysis_report(actual_labels, predictions)
        
        # Add enhanced reports to results
        results.update({
            'confusion_matrix_plot': cm_plot,
            'formatted_classification_report': formatted_report,
            'detailed_analysis_report': detailed_report
        })
        
        return results

    def generate_confusion_matrix_plot(self, actual_labels: List[int], predictions: List[int], 
                                     save_path: str = "confusion_matrix_llama_rag.png"):
        """Generate and save confusion matrix visualization"""
        print(f"📊 Generating confusion matrix plot...")
        
        # Set style for better visualization
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create confusion matrix
        cm = confusion_matrix(actual_labels, predictions)
        
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
        plt.title("Confusion Matrix - Llama RAG Triage System", fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Add grid lines for better readability
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save to original location (for backward compatibility)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
        
        # Also save to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_save_path = os.path.join(self.reports_dir, "visualization", f"confusion_matrix_llama_rag_{timestamp}.png")
        plt.savefig(reports_save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix also saved to reports: {reports_save_path}")
        
        plt.close()
        
        return cm

    def generate_formatted_classification_report(self, actual_labels: List[int], predictions: List[int]) -> Dict:
        """Generate formatted classification report similar to Table 5.4"""
        print(f"📋 Generating formatted classification report...")
        
        # Get classification report
        report_dict = classification_report(
            actual_labels, predictions,
            target_names=[f"ESI {i}" for i in range(1, 6)],
            output_dict=True,
            digits=2
        )
        
        # Calculate accuracy
        accuracy = np.mean([p == a for p, a in zip(predictions, actual_labels)])
        
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
        
        # Print formatted performance table
        print(f"\n📊 Table 5.4: Per-class performance on the held-out test set (Scheme A, Llama RAG)")
        print("=" * 80)
        print(f"{'ESI Level':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        for _, row in performance_df.iterrows():
            if row['ESI Level'] == 'Accuracy':
                print(f"{row['ESI Level']:<12} {'':<10} {'':<10} {row['F1']:<10.2f}")
            else:
                print(f"{row['ESI Level']:<12} {row['Precision']:<10.2f} {row['Recall']:<10.2f} {row['F1']:<10.2f}")
        print("=" * 80)
        
        # Save performance report to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        performance_path = os.path.join(self.reports_dir, "model_performance", f"per_class_performance_llama_rag_{timestamp}.csv")
        performance_df.to_csv(performance_path, index=False)
        print(f"✅ Per-class performance saved to reports: {performance_path}")
        
        return {
            'performance_df': performance_df,
            'report_dict': report_dict,
            'accuracy': accuracy
        }

    def generate_detailed_analysis_report(self, actual_labels: List[int], predictions: List[int]) -> pd.DataFrame:
        """Generate detailed analysis report with TP/FP/FN metrics"""
        print(f"\n📊 Generating detailed analysis report...")
        
        # Calculate additional metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            actual_labels, predictions, average=None, labels=range(1, 6)
        )
        
        # Create confusion matrix for detailed analysis
        cm = confusion_matrix(actual_labels, predictions)
        
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
        
        detailed_df = pd.DataFrame(report_data)
        
        # Print detailed analysis
        print(f"\n📊 Detailed Performance Analysis:")
        print("=" * 100)
        print(f"{'ESI Level':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 100)
        for _, row in detailed_df.iterrows():
            print(f"{row['ESI Level']:<10} {row['Precision']:<10.2f} {row['Recall']:<10.2f} {row['F1']:<10.2f} {row['Support']:<10.0f} {row['True Positives']:<8} {row['False Positives']:<8} {row['False Negatives']:<8}")
        print("=" * 100)
        
        # Save detailed analysis report to reports directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_path = os.path.join(self.reports_dir, "model_performance", f"detailed_performance_analysis_llama_rag_{timestamp}.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"✅ Detailed performance analysis saved to reports: {detailed_path}")
        
        return detailed_df

    def _evaluate_system_multi(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate system performance under different retrieval configs"""
        
        configs = {
            "vector_only": {"use_hybrid": False, "use_reranking": False},
            "hybrid_only": {"use_hybrid": True, "use_reranking": False},
            "hybrid_rerank": {"use_hybrid": True, "use_reranking": True}
        }
        
        all_results = {}
        
        for name, cfg in configs.items():
            print(f"\n🚀 Evaluating config: {name}")
            print("=" * 50)
            predictions, actual_labels, processing_times = [], [], []
            error_count = 0
            
            for idx, (_, case) in enumerate(test_data.iterrows()):
                if idx % 50 == 0:
                    print(f"   Progress: {idx}/{len(test_data)} ({idx/len(test_data):.1%})")
                
                try:
                    start_time = time.time()
                    case_dict = case.to_dict()
                    
                    # Call different retrieval methods
                    similar_cases, similarities, metadata = self.case_retriever.retrieve_similar_cases(
                        case_dict, k=5,
                        use_hybrid=cfg["use_hybrid"],
                        use_reranking=cfg["use_reranking"]
                    )
                    
                    # Llama generation
                    predicted_level, reasoning = self.llama_generator.generate_triage_decision(
                        pd.Series(case_dict), similar_cases, similarities
                    )
                    
                    predictions.append(predicted_level)
                    actual_labels.append(int(case['acuity']))
                    processing_times.append(time.time() - start_time)
                    
                except Exception as e:
                    print(f"❌ Error processing case {idx}: {e}")
                    error_count += 1
                    predictions.append(3)
                    actual_labels.append(int(case['acuity']))
                    processing_times.append(0)
            
            accuracy = np.mean([p == a for p, a in zip(predictions, actual_labels)])
            avg_time = np.mean(processing_times) if processing_times else 0
            error_stats = self.llama_generator.get_error_statistics()
            
            all_results[name] = {
                "accuracy": accuracy,
                "avg_time": avg_time,
                "errors": error_count,
                "predictions": predictions,
                "actual_labels": actual_labels,
                "classification_report": classification_report(actual_labels, predictions, output_dict=True),
                "confusion_matrix": confusion_matrix(actual_labels, predictions).tolist(),
                "llama_error_stats": error_stats
            }

            # Generate enhanced reports for this config
            print(f"\n🎨 Generating enhanced reports for {name}...")
            
            # Generate confusion matrix plot
            cm_plot = self.generate_confusion_matrix_plot(actual_labels, predictions, f"confusion_matrix_llama_rag_{name}.png")
            
            # Generate formatted classification report
            formatted_report = self.generate_formatted_classification_report(actual_labels, predictions)
            
            # Generate detailed analysis report
            detailed_report = self.generate_detailed_analysis_report(actual_labels, predictions)
            
            # Add enhanced reports to results
            all_results[name].update({
                'confusion_matrix_plot': cm_plot,
                'formatted_classification_report': formatted_report,
                'detailed_analysis_report': detailed_report
            })

            print(f"📊 {name} Results: Accuracy={accuracy:.3f}, AvgTime={avg_time:.2f}s")
            
            # Display detailed results for each config
            self._display_config_results(name, all_results[name])
        
        return all_results

    def _display_config_results(self, config_name: str, results: Dict):
        """Display detailed results for a specific configuration"""
        
        print(f"\n📋 {config_name} - Detailed Classification Report:")
        print("-" * 50)
        
        # Auto-detect actual ESI levels
        actual_labels = results.get('actual_labels', [])
        predictions = results.get('predictions', [])
        
        if not actual_labels or not predictions:
            return
            
        unique_labels = sorted(set(actual_labels))
        target_names = [f'ESI {i}' for i in unique_labels]
        report_str = classification_report(actual_labels, predictions, target_names=target_names)
        print(report_str)
        
        # Display confusion matrix
        print(f"\n🔢 {config_name} - Confusion Matrix:")
        print("-" * 30)
        cm = confusion_matrix(actual_labels, predictions, labels=unique_labels)
        print("Predicted →")
        print("Actual ↓")
        print("     ", end="")
        for label in unique_labels:
            print(f"ESI{label:>6}", end="")
        print()
        
        for i, actual_label in enumerate(unique_labels):
            print(f"ESI{actual_label:>5}", end="")
            for j, pred_label in enumerate(unique_labels):
                print(f"{cm[i][j]:>6}", end="")
            print()
        
        # Display F1 scores
        print(f"\n🎯 {config_name} - F1 Scores by ESI Level:")
        print("-" * 30)
        for label in unique_labels:
            if str(label) in results['classification_report']:
                precision = results['classification_report'][str(label)]['precision']
                recall = results['classification_report'][str(label)]['recall']
                f1 = results['classification_report'][str(label)]['f1-score']
                support = results['classification_report'][str(label)]['support']
                print(f"   ESI {label}: F1={f1:.3f}, Accuracy={precision:.3f}, Recall={recall:.3f}, Support={support}")
        
        # Overall metrics
        macro_f1 = results['classification_report']['macro avg']['f1-score']
        weighted_f1 = results['classification_report']['weighted avg']['f1-score']
        print(f"\n📊 {config_name} - Overall F1 Scores:")
        print(f"   Macro F1: {macro_f1:.3f}")
        print(f"   Weighted F1: {weighted_f1:.3f}")
    
    def save_evaluation_results(self, results: Dict, save_path: str = None):
        """Save evaluation results to JSON file"""
        
        if save_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"llama_rag_evaluation_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = results.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        if 'confusion_matrix_plot' in serializable_results:
            serializable_results['confusion_matrix_plot'] = serializable_results['confusion_matrix_plot'].tolist()
        
        # Convert DataFrames to dictionaries
        if 'formatted_classification_report' in serializable_results:
            report_data = serializable_results['formatted_classification_report']
            if 'performance_df' in report_data:
                serializable_results['formatted_classification_report']['performance_df'] = report_data['performance_df'].to_dict('records')
        
        if 'detailed_analysis_report' in serializable_results:
            serializable_results['detailed_analysis_report'] = serializable_results['detailed_analysis_report'].to_dict('records')
        
        try:
            # Save to original location (for backward compatibility)
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Evaluation results saved to: {save_path}")
            
            # Also save to reports directory
            reports_json_path = os.path.join(self.reports_dir, "evaluation_results", os.path.basename(save_path))
            with open(reports_json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Evaluation results also saved to reports: {reports_json_path}")
            
            # Also save performance report as CSV
            if 'formatted_classification_report' in results:
                report_data = results['formatted_classification_report']
                if 'performance_df' in report_data:
                    csv_path = save_path.replace('.json', '_performance_report.csv')
                    report_data['performance_df'].to_csv(csv_path, index=False)
                    print(f"💾 Performance report saved to: {csv_path}")
            
            # Save detailed analysis as CSV
            if 'detailed_analysis_report' in results:
                detailed_csv_path = save_path.replace('.json', '_detailed_analysis.csv')
                results['detailed_analysis_report'].to_csv(detailed_csv_path, index=False)
                print(f"💾 Detailed analysis saved to: {detailed_csv_path}")
                
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


# Standalone evaluation function
def evaluate_llama_rag_system(data_path: str, test_size: int = 100, 
                             save_index_path: str = "llama_rag_train_only_index_stable",
                             force_rebuild: bool = False):
    """
    Standalone function to evaluate Llama RAG system
    
    Args:
        data_path: Path to dataset
        test_size: Number of test samples
        save_index_path: Path to save/load index
        force_rebuild: Whether to force rebuild index
    """
    
    print(f"🦙 Starting Llama RAG System Evaluation")
    print(f"=" * 50)
    
    # Initialize system
    system = LlamaRAGTriageSystem()
    
    # Run pipeline
    results = system.run_full_pipeline(
        data_path=data_path,
        test_size=test_size,
        save_path=save_index_path,
        force_rebuild=force_rebuild
    )
    
    if "error" not in results:
        # Save results
        system.save_evaluation_results(results)
        
        # Save test data for analysis
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        test_data_path = f"test_data_llama_rag_{timestamp}.csv"
        
        print(f"🎉 Evaluation completed successfully!")
        
        # Check if this is multi-config evaluation
        if isinstance(results, dict) and any(key in results for key in ["vector_only", "hybrid_only", "hybrid_rerank"]):
            # Multi-config results
            print(f"📊 Multi-Config Evaluation Results:")
            print("=" * 50)
            
            # Find best performing config
            best_config = max(results.items(), key=lambda x: x[1]['accuracy'])
            worst_config = min(results.items(), key=lambda x: x[1]['accuracy'])
            
            print(f"🏆 Best Config: {best_config[0]} - Accuracy: {best_config[1]['accuracy']:.3f}")
            print(f"⚠️ Worst Config: {worst_config[0]} - Accuracy: {worst_config[1]['accuracy']:.3f}")
            
            # Show all config results
            print(f"\n📈 All Configurations:")
            for config_name, config_results in results.items():
                accuracy = config_results['accuracy']
                avg_time = config_results['avg_time']
                print(f"   {config_name:15}: Accuracy={accuracy:.3f}, Time={avg_time:.2f}s")
            
            # Show improvement analysis
            vector_acc = results.get('vector_only', {}).get('accuracy', 0)
            hybrid_acc = results.get('hybrid_only', {}).get('accuracy', 0)
            rerank_acc = results.get('hybrid_rerank', {}).get('accuracy', 0)
            
            if vector_acc > 0 and hybrid_acc > 0:
                hybrid_improvement = ((hybrid_acc - vector_acc) / vector_acc) * 100
                print(f"\n📊 Hybrid vs Vector: {hybrid_improvement:+.1f}% improvement")
            
            if hybrid_acc > 0 and rerank_acc > 0:
                rerank_improvement = ((rerank_acc - hybrid_acc) / hybrid_acc) * 100
                print(f"📊 Rerank vs Hybrid: {rerank_improvement:+.1f}% improvement")
                
        else:
            # Single config results
            print(f"📊 Final Results:")
            print(f"   Accuracy: {results['accuracy']:.3f}")
            print(f"   Avg Processing Time: {results['avg_processing_time']:.2f}s")
            print(f"   Llama Success Rate: {results['llama_error_stats']['success_rate']:.3f}")
            
            # Display key F1 metrics
            if 'classification_report' in results:
                macro_f1 = results['classification_report']['macro avg']['f1-score']
                weighted_f1 = results['classification_report']['weighted avg']['f1-score']
                print(f"   Macro F1: {macro_f1:.3f}")
                print(f"   Weighted F1: {weighted_f1:.3f}")
                
                # Show best and worst performing ESI levels
                f1_by_class = {}
                for i in range(1, 6):
                    if str(i) in results['classification_report']:
                        f1_by_class[i] = results['classification_report'][str(i)]['f1-score']
                
                if f1_by_class:
                    best_esi = max(f1_by_class, key=f1_by_class.get)
                    worst_esi = min(f1_by_class, key=f1_by_class.get)
                    print(f"   Best ESI: {best_esi} (F1={f1_by_class[best_esi]:.3f})")
                    print(f"   Worst ESI: {worst_esi} (F1={f1_by_class[worst_esi]:.3f})")
    else:
            print(f"❌ Evaluation failed: {results['error']}")


if __name__ == "__main__":
    # Example usage
    # Get the project root directory (medical_triage_assistant)
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(project_root, "data_processing", "processed_data", "triage_with_keywords.csv")
    
    if os.path.exists(data_path):
        # Use existing index without rebuilding
        evaluate_llama_rag_system(
            data_path=data_path,
            test_size=1000,
            save_index_path="llama_rag_train_only_index_stable",
            force_rebuild=False  # Explicitly use existing index
        )
    else:
        print(f"❌ Data file not found: {data_path}")
        print(f"   Please run from the methods/llama_rag directory or adjust the path")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Project root: {project_root}")
        print(f"   Looking for: {data_path}")