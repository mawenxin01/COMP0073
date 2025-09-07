import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# 🔧 Command Line Arguments
# -------------------------------
parser = argparse.ArgumentParser(description='Train TFIDF + XGBoost model')
parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
args = parser.parse_args()

# Add project root directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Define absolute paths to avoid working directory issues
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../..')
data_path = os.path.join(project_root, "data_processing/processed_data/triage_with_keywords.csv")
reports_dir = os.path.join(project_root, "reports")
model_path = os.path.join(project_root, "models", "TFIDF-xgb.pkl")

# Create reports directories
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
        (model_path, "XGBoost model")
    ]
    
    # Check for latest report files in reports directory
    import glob
    confusion_matrix_files = glob.glob(os.path.join(reports_dir, "visualization", "confusion_matrix_xgb_tfidf_keywords_*.png"))
    detailed_analysis_files = glob.glob(os.path.join(reports_dir, "model_performance", "detailed_performance_analysis_xgb_tfidf_keywords_*.csv"))
    per_class_files = glob.glob(os.path.join(reports_dir, "model_performance", "per_class_performance_xgb_tfidf_keywords_*.csv"))
    
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
        print("   Example: python withkw-xgb.py --force")
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
    print(f"❌ Data file not found: {data_path}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Script directory: {script_dir}")
    print(f"   Project root: {project_root}")
    exit(1)

# 1. Load dataset
data = pd.read_csv(data_path, low_memory=False)

print("⏰ Using preprocessed time_period feature (0=night, 1=morning, 2=afternoon, 3=evening)...")
print(f"   Distribution: {data['time_period'].value_counts().sort_index().to_dict()}")
print(f"   Total samples: {len(data)}")

# Handle missing or malformed time_period
if 'time_period' not in data.columns:
    print("❌ Warning: time_period column missing, assigning default value (2=afternoon)...")
    data['time_period'] = 2
else:
    data['time_period'] = pd.to_numeric(data['time_period'], errors="coerce").fillna(2)
    print(f"✅ time_period column loaded, dtype: {data['time_period'].dtype}")

# 2. One-hot encode categorical features
cat_cols = ["gender", "arrival_transport"]
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# 3. Apply TF-IDF and SVD on keyword text
data = data[data["complaint_keywords"].notna() & (data["complaint_keywords"].str.strip() != "")]
tfidf = TfidfVectorizer()
X_keywords_tfidf = tfidf.fit_transform(data["complaint_keywords"].fillna(""))

svd = TruncatedSVD(n_components=200, random_state=42)
X_keywords_svd = svd.fit_transform(X_keywords_tfidf)

# Append reduced features to main DataFrame
for i in range(X_keywords_svd.shape[1]):
    data[f"kw_svd_{i}"] = X_keywords_svd[:, i]

# 4. Drop irrelevant columns
drop_cols = ["subject_id", "intime", "outtime", "chiefcomplaint", "complaint_keywords", "disposition"]
data = data.drop(columns=drop_cols)

# 5. Prepare features and labels
X = data.drop(columns=["acuity"])
y = data["acuity"].astype(int) - 1  # Convert to 0-based index

X["pain"] = pd.to_numeric(X["pain"], errors="coerce")
X = X.fillna(0)

bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

if 'time_period' in X.columns:
    X['time_period'] = X['time_period'].astype(int)
    print(f"✅ time_period dtype: {X['time_period'].dtype}")

print(f"\n🔍 Data type check:")
for col in X.columns:
    dtype = X[col].dtype
    unique_count = X[col].nunique()
    print(f"   {col}: {dtype} (unique: {unique_count})")
    if dtype == 'object':
        print(f"      Sample values: {X[col].dropna().head(3).tolist()}")

print(f"\n📊 Feature summary:")
print(f"   Total features: {X.shape[1]}")
print(f"   Numeric features: {X.select_dtypes(include=['number']).columns.tolist()}")
print(f"   Non-numeric features: {X.select_dtypes(exclude=['number']).columns.tolist()}")

# Convert to float32
numeric_cols = X.select_dtypes(include=['number']).columns
X[numeric_cols] = X[numeric_cols].astype("float32")
print(f"✅ All numeric columns converted to float32")

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# -------------------------------
# 🔍 Add GridSearch
# -------------------------------
f1_weighted = make_scorer(f1_score, average="weighted")

class_weight_options = [
    None,
    "balanced",
    {0: 10, 1: 8, 2: 3, 3: 2, 4: 1},
]

# Custom wrapper to make class_weight work in GridSearch
class CustomXGBClassifier(XGBClassifier):
    def __init__(self, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None, **kwargs):
        if self.class_weight is not None:
            if self.class_weight == "balanced":
                # Automatically calculate balanced weights
                classes = np.unique(y)
                class_counts = np.bincount(y)
                total = len(y)
                weights = {cls: total / (len(classes) * count)
                           for cls, count in enumerate(class_counts)}
                sample_weight = pd.Series(y).map(weights).values
                print(f"🔄 Using balanced weights: {weights}")
            elif isinstance(self.class_weight, dict):
                # Use manually specified weights
                sample_weight = pd.Series(y).map(self.class_weight).values
                print(f"🔄 Using custom weights: {self.class_weight}")
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "class_weight": class_weight_options
}

grid_model = CustomXGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    num_class=len(np.unique(y)),
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

grid_search = GridSearchCV(
    estimator=grid_model,
    param_grid=param_grid,
    scoring=f1_weighted,
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("\n🔍 Running GridSearchCV...")
grid_search.fit(X_train, y_train)


print("\n✅ Best parameters found:")
print(grid_search.best_params_)
print(f"Best Weighted F1: {grid_search.best_score_:.4f}")

# -------------------------------
# 📊 Output all parameter combinations for comparison
# -------------------------------
results = pd.DataFrame(grid_search.cv_results_)

# Keep only the most relevant information
report = results[[
    "params",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]].sort_values("rank_test_score")

print("\n📊 GridSearchCV Results (sorted by F1 score):")
print(report)

# Save to CSV for further analysis or inclusion in the thesis
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(reports_dir, "experiments", f"gridsearch_f1_report_xgb_tfidf_{timestamp}.csv")
report.to_csv(report_path, index=False)
print(f"💾 GridSearch report saved to: {report_path}")

# Also save to models directory for backward compatibility
models_report_path = os.path.join(project_root, "models/gridsearch_f1_report.csv")
report.to_csv(models_report_path, index=False)
print(f"💾 GridSearch report also saved to: {models_report_path}")


print("\n✅ Best parameters found:")
print(grid_search.best_params_)
print(f"Best Weighted F1: {grid_search.best_score_:.4f}")

# -------------------------------
# Use best model for prediction
# -------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Save best model
joblib.dump(best_model, model_path)
print(f"💾 Best model saved to: {model_path}")

# Evaluation
print("\n📊 Classification report:")
class_report = classification_report(y_test + 1, y_pred + 1, target_names=[f"Level {i}" for i in range(1, 6)], output_dict=True)
print(classification_report(y_test + 1, y_pred + 1, target_names=[f"Level {i}" for i in range(1, 6)]))

# Calculate detailed metrics
y_test_esi = y_test + 1  # Convert to ESI levels 1-5
y_pred_esi = y_pred + 1
accuracy = accuracy_score(y_test_esi, y_pred_esi)
weighted_f1 = f1_score(y_test_esi, y_pred_esi, average='weighted')

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Level {i+1}" for i in range(5)],
            yticklabels=[f"Level {i+1}" for i in range(5)],
            cbar_kws={'label': 'Count'})
plt.xlabel("Predicted Triage Level", fontsize=12, fontweight='bold')
plt.ylabel("True Triage Level", fontsize=12, fontweight='bold')
plt.title(f"Confusion Matrix - Best XGBoost with Keywords\nAccuracy: {accuracy:.3f}, Weighted F1: {weighted_f1:.3f}", 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save confusion matrix
cm_path = os.path.join(reports_dir, "visualization", f"confusion_matrix_xgb_tfidf_keywords_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {cm_path}")
plt.show()

# Save detailed performance analysis
precision, recall, f1, support = precision_recall_fscore_support(
    y_test_esi, y_pred_esi, average=None, labels=range(1, 6)
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
performance_path = os.path.join(reports_dir, "model_performance", f"detailed_performance_analysis_xgb_tfidf_keywords_{timestamp}.csv")
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
summary_path = os.path.join(reports_dir, "model_performance", f"per_class_performance_xgb_tfidf_keywords_{timestamp}.csv")
summary_df.to_csv(summary_path, index=False)
print(f"✅ Per-class performance summary saved to: {summary_path}")

print(f"\n📊 Final Results Summary:")
print(f"   Accuracy: {accuracy:.3f}")
print(f"   Weighted F1: {weighted_f1:.3f}")
print(f"   Best GridSearch F1: {grid_search.best_score_:.3f}")
print(f"\n📁 All reports saved to: {reports_dir}")
