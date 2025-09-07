import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer, precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# 🔧 Command Line Arguments
# -------------------------------
parser = argparse.ArgumentParser(description='Train TFIDF + Random Forest model')
parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
args = parser.parse_args()

# -------------------------------
# 📂 Path Settings
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../..')
data_path = os.path.join(project_root, "data_processing/processed_data/triage_with_keywords.csv")
reports_dir = os.path.join(project_root, "reports")
model_path = os.path.join(project_root, "models", "TFIDF-rf.pkl")
vectorizer_path = os.path.join(project_root, "models", "TFIDF-rf-vectorizer.pkl")
svd_path = os.path.join(project_root, "models", "TFIDF-rf-svd.pkl")

# Create directories
os.makedirs(os.path.join(reports_dir, "model_performance"), exist_ok=True)
os.makedirs(os.path.join(reports_dir, "visualization"), exist_ok=True)
os.makedirs(os.path.join(reports_dir, "experiments"), exist_ok=True)
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

# -------------------------------
# 🔍 Check if model already exists
# -------------------------------
def check_existing_models():
    """Check if all model files and reports already exist"""
    # Define all expected files
    files_to_check = [
        (model_path, "Random Forest model"),
        (vectorizer_path, "TF-IDF vectorizer"),
        (svd_path, "SVD model")
    ]
    
    # Check for latest report files in reports directory
    import glob
    confusion_matrix_files = glob.glob(os.path.join(reports_dir, "visualization", "confusion_matrix_rf_tfidf_keywords_*.png"))
    detailed_analysis_files = glob.glob(os.path.join(reports_dir, "model_performance", "detailed_performance_analysis_rf_tfidf_keywords_*.csv"))
    per_class_files = glob.glob(os.path.join(reports_dir, "model_performance", "per_class_performance_rf_tfidf_keywords_*.csv"))
    
    if confusion_matrix_files:
        latest_cm = max(confusion_matrix_files, key=os.path.getmtime)
        files_to_check.append((latest_cm, "Confusion matrix"))
    
    if detailed_analysis_files:
        latest_detailed = max(detailed_analysis_files, key=os.path.getmtime)
        files_to_check.append((latest_detailed, "Detailed performance analysis"))
        
    if per_class_files:
        latest_per_class = max(per_class_files, key=os.path.getmtime)
        files_to_check.append((latest_per_class, "Per-class performance report"))
    
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
        print("   Example: python withkw-randomforest.py --force")
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

if not args.force and check_existing_models():
    exit(0)

print(f"📂 Data file path: {data_path}")
if not os.path.exists(data_path):
    print(f"❌ File not found: {data_path}")
    exit(1)

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv(data_path, low_memory=False)

print("⏰ Using preprocessed time_period feature...")
if 'time_period' not in data.columns:
    print("❌ Warning: 'time_period' column missing. Defaulting to afternoon (2).")
    data['time_period'] = 2
else:
    data['time_period'] = pd.to_numeric(data['time_period'], errors='coerce').fillna(2)

# One-hot encoding
cat_cols = ["gender", "arrival_transport"]
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# TF-IDF
data = data[data["complaint_keywords"].notna() & (data["complaint_keywords"].str.strip() != "")]
tfidf = TfidfVectorizer()
X_keywords_tfidf = tfidf.fit_transform(data["complaint_keywords"].fillna(""))

# SVD dimensionality reduction
svd = TruncatedSVD(n_components=20, random_state=42)
X_keywords_svd = svd.fit_transform(X_keywords_tfidf)
for i in range(X_keywords_svd.shape[1]):
    data[f"kw_svd_{i}"] = X_keywords_svd[:, i]

# Drop unnecessary columns
drop_cols = ["subject_id", "intime", "outtime", "chiefcomplaint", "complaint_keywords", "disposition"]
data = data.drop(columns=drop_cols)

# -------------------------------
# 2. Features & Labels
# -------------------------------
X = data.drop(columns=["acuity"])
y = data["acuity"].astype(int)  # ESI levels 1–5

X["pain"] = pd.to_numeric(X["pain"], errors="coerce")
X = X.fillna(0)

bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

if 'time_period' in X.columns:
    X['time_period'] = X['time_period'].astype(int)

# Convert all to float32
numeric_cols = X.select_dtypes(include=['number']).columns
X[numeric_cols] = X[numeric_cols].astype("float32")

print(f"✅ Feature count: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
print(f"   Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# -------------------------------
# 3. GridSearchCV Settings
# -------------------------------
f1_weighted = make_scorer(f1_score, average="weighted")

class_weight_options = [
    None,
    "balanced",
  
    {1: 10, 2: 8, 3: 3, 4: 2, 5: 1},  
]

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": class_weight_options
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring=f1_weighted,
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("\n🔍 Running GridSearchCV for RandomForest...")
grid_search.fit(X_train, y_train)

print("\n✅ Best parameters found:")
print(grid_search.best_params_)
print(f"Best Weighted F1: {grid_search.best_score_:.4f}")

# -------------------------------
# 4. Best Model Evaluation
# -------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Save models
joblib.dump(best_model, model_path)
joblib.dump(tfidf, vectorizer_path)
joblib.dump(svd, svd_path)

print(f"💾 Best model saved to: {model_path}")
print(f"💾 TF-IDF vectorizer saved to: {vectorizer_path}")
print(f"💾 SVD model saved to: {svd_path}")

# Output report
print("\n📊 Classification report:")
class_report = classification_report(y_test, y_pred, target_names=[f"ESI {i}" for i in range(1, 6)], output_dict=True)
print(classification_report(y_test, y_pred, target_names=[f"ESI {i}" for i in range(1, 6)]))

# Calculate detailed metrics
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"ESI {i}" for i in range(1, 6)],
            yticklabels=[f"ESI {i}" for i in range(1, 6)],
            cbar_kws={'label': 'Count'})
plt.xlabel("Predicted Triage Level", fontsize=12, fontweight='bold')
plt.ylabel("True Triage Level", fontsize=12, fontweight='bold')
plt.title(f"Confusion Matrix - Best RandomForest with Keywords\nAccuracy: {accuracy:.3f}, Weighted F1: {weighted_f1:.3f}", 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save confusion matrix
cm_path = os.path.join(reports_dir, "visualization", f"confusion_matrix_rf_tfidf_keywords_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {cm_path}")
plt.show()

# Save detailed performance analysis
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=range(1, 6)
)

performance_data = []
for i in range(5):
    esi_level = i + 1
    true_pos = cm[i, i]
    false_pos = np.sum(cm[:, i]) - true_pos
    false_neg = np.sum(cm[i, :]) - true_pos
    
    performance_data.append({
        'ESI Level': f'ESI {esi_level}',
        'Precision': precision[i],
        'Recall': recall[i],
        'F1': f1[i],
        'Support': support[i],
        'True Positives': true_pos,
        'False Positives': false_pos,
        'False Negatives': false_neg
    })

performance_df = pd.DataFrame(performance_data)
performance_path = os.path.join(reports_dir, "model_performance", f"detailed_performance_analysis_rf_tfidf_keywords_{timestamp}.csv")
performance_df.to_csv(performance_path, index=False)
print(f"✅ Detailed performance analysis saved to: {performance_path}")

# Save per-class performance summary
summary_data = []
for i in range(5):
    summary_data.append({
        'ESI Level': f'ESI {i+1}',
        'Precision': precision[i],
        'Recall': recall[i],
        'F1': f1[i]
    })

# Add overall metrics
summary_data.append({
    'ESI Level': 'Accuracy',
    'Precision': '',
    'Recall': '',
    'F1': accuracy
})
summary_data.append({
    'ESI Level': 'Weighted Avg',
    'Precision': np.average(precision, weights=support),
    'Recall': np.average(recall, weights=support),
    'F1': weighted_f1
})

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(reports_dir, "model_performance", f"per_class_performance_rf_tfidf_keywords_{timestamp}.csv")
summary_df.to_csv(summary_path, index=False)
print(f"✅ Per-class performance summary saved to: {summary_path}")

# -------------------------------
# 5. Save GridSearch Report
# -------------------------------
results = pd.DataFrame(grid_search.cv_results_)
report = results[["params", "mean_test_score", "std_test_score", "rank_test_score"]].sort_values("rank_test_score")

# Save to reports directory
gridsearch_report_path = os.path.join(reports_dir, "experiments", f"gridsearch_f1_report_rf_tfidf_keywords_{timestamp}.csv")
report.to_csv(gridsearch_report_path, index=False)
print(f"✅ GridSearch report saved to: {gridsearch_report_path}")

# Also save to models directory for backward compatibility
models_report_path = os.path.join(project_root, "models/randomforest_gridsearch_report.csv")
report.to_csv(models_report_path, index=False)
print(f"💾 GridSearch report also saved to: {models_report_path}")

print(f"\n📊 Final Results Summary:")
print(f"   Accuracy: {accuracy:.3f}")
print(f"   Weighted F1: {weighted_f1:.3f}")
print(f"   Best GridSearch F1: {grid_search.best_score_:.3f}")
print(f"\n📁 All reports saved to: {reports_dir}")

