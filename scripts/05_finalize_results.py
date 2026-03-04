# scripts/05_finalize_results.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Sklearn Models
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import catboost as cb

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_group_id

# --- Safe CatBoost Wrapper (Required for Pipeline) ---
class SafeCatBoostClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, iterations=100, verbose=0, auto_class_weights='Balanced'):
        self.iterations = iterations
        self.verbose = verbose
        self.auto_class_weights = auto_class_weights
        self.model = None

    def fit(self, X, y):
        self.model = cb.CatBoostClassifier(
            iterations=self.iterations, 
            verbose=self.verbose, 
            auto_class_weights=self.auto_class_weights,
            allow_writing_files=False,
            thread_count=1
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def get_model_instance(name):
    """Returns a fresh instance of the model based on name."""
    if 'HistGradientBoosting' in name:
        return HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
    elif 'CatBoost' in name:
        return SafeCatBoostClassifier(iterations=100, verbose=0, auto_class_weights='Balanced')
    elif 'ExtraTrees_100' in name:
        return ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    elif 'ExtraTrees_200' in name:
        return ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    elif 'LightGBM' in name:
        return lgb.LGBMClassifier(class_weight='balanced', verbose=-1, random_state=42)
    elif 'MLP_1Layer' in name:
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    elif 'RandomForest' in name:
        return RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    else:
        # Fallback / Placeholder
        print(f"Warning: No specific config for {name}, defaulting to HistGradientBoosting")
        return HistGradientBoostingClassifier(class_weight='balanced')

def main():
    print("--- 1. Loading & Splitting Data (80/20 Group-Aware) ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    groups = np.array([get_group_id(f) for f in meta['filename'].values])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(y, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- 2. Process CSV & Select Top 5 ---
    csv_path = os.path.join(base_dir, 'output', 'reports', 'benchmark_grouped_80_20.csv')
    df = pd.read_csv(csv_path)
    
    # Sort by Balanced Accuracy to find top 5
    df_sorted = df.sort_values(by='CV_Bal_Acc', ascending=False).reset_index(drop=True)
    top_5_names = df_sorted.head(5)['Model'].values
    print(f"Top 5 Models selected: {top_5_names}")

    # --- 3. Format CSV (Combine Mean & Std) ---
    print("\n--- Reformatting CSV ---")
    metrics = ['CV_Bal_Acc', 'CV_Acc', 'CV_Sens', 'CV_Spec', 'CV_AUROC']
    
    df_formatted = df_sorted.copy()
    
    # Create combined columns
    for m in metrics:
        mean_col = m
        std_col = f"{m}_Std"
        
        # Handle CatBoost NaN stds if any (fill with 0 for formatting)
        df_formatted[std_col] = df_formatted[std_col].fillna(0)
        
        df_formatted[m] = df_formatted.apply(
            lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}", axis=1
        )
        
        # Drop std column
        df_formatted.drop(columns=[std_col], inplace=True)
        
    save_csv_path = os.path.join(base_dir, 'output', 'reports', 'benchmark_grouped_80_20_formatted.csv')
    df_formatted.to_csv(save_csv_path, index=False)
    print(f"Saved formatted CSV to {save_csv_path}")

    # --- 4. Train, Save, & Validate Top 5 ---
    print("\n--- Training & Validating Top 5 Models ---")
    
    models_dir = os.path.join(base_dir, 'output', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    mad_results = []
    
    plt.figure(figsize=(10, 8))
    
    for name in top_5_names:
        print(f"Processing {name}...")
        
        # Instantiate
        model = get_model_instance(name)
        pipeline = make_pipeline(StandardScaler(), model)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Save (This addresses "save CatBoost as pkl")
        safe_name = name.replace(" ", "_").lower()
        joblib.dump(pipeline, os.path.join(models_dir, f"{safe_name}.pkl"))
        
        # Predict on Test Set (External Validation)
        try:
            y_probs = pipeline.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_probs)
            
            # Get CV AUC (Raw value from sorted original df)
            cv_auc_val = df_sorted[df_sorted['Model'] == name]['CV_AUROC'].values[0]
            
            # Calculate MAD
            mad = abs(cv_auc_val - test_auc)
            
            mad_results.append({
                'Model': name,
                'CV_AUROC': cv_auc_val,
                'Test_AUROC': test_auc,
                'MAD': mad
            })
            
            # Plot ROC
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {test_auc:.4f})')
            
        except Exception as e:
            print(f"Failed to validate {name}: {e}")

    # Finalize Plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - Top 5 Models (External Validation)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(base_dir, 'output', 'plots', 'top_5_external_roc.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved ROC Plot to {plot_path}")
    
    # Select best model (lowest MAD + High AUC)
    df_mad = pd.DataFrame(mad_results)
    df_mad = df_mad.sort_values(by='MAD') # Lowest MAD = Best Generalization
    
    mad_csv_path = os.path.join(base_dir, 'output', 'reports', 'top_5_mad_ranking.csv')
    df_mad.to_csv(mad_csv_path, index=False)
    print(f"Saved MAD Ranking to {mad_csv_path}")
    
    print("\n--- Final MAD Ranking ---")
    print(df_mad)

if __name__ == "__main__":
    main()
