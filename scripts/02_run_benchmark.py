# scripts/02_run_benchmark.py
import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import warnings

# Sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, roc_auc_score
import joblib

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models_config import get_models

# Suppress ConvergenceWarnings etc
warnings.filterwarnings("ignore")

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def main():
    print("--- Step 2: Model Benchmarking ---")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        print(f"Loaded Data: {X.shape}")
    except FileNotFoundError:
        print("Data not found. Run scripts/01_process_data.py first.")
        return

    # 2. Setup Benchmark
    models = get_models()
    print(f"Loaded {len(models)} models configuration.")
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc'
    }
    
    results = []
    
    print("Running Stratified 10-Fold Cross-Validation...")
    
    for name, model in tqdm(models):
        try:
            # Scaling is important for many models
            # Tree-based models don't strictly need it, but it doesn't hurt much
            pipeline = make_pipeline(StandardScaler(), model)
            
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score='raise')
            
            res = {
                'Model': name,
                'Balanced Accuracy': np.mean(scores['test_balanced_acc']),
                'Balanced Acc Std': np.std(scores['test_balanced_acc']),
                'Sensitivity (TB)': np.mean(scores['test_sensitivity']),
                'Sensitivity Std': np.std(scores['test_sensitivity']),
                'Specificity (NTM)': np.mean(scores['test_specificity']),
                'Specificity Std': np.std(scores['test_specificity']),
                'AUROC': np.mean(scores['test_roc_auc']),
                'AUROC Std': np.std(scores['test_roc_auc'])
            }
            results.append(res)
            
        except Exception as e:
            print(f"Failed {name}: {e}")
            # Try capturing error to result
            res = {
                'Model': name,
                'Balanced Accuracy': 0, 'Balanced Acc Std': 0,
                'Sensitivity (TB)': 0, 'Sensitivity Std': 0,
                'Specificity (NTM)': 0, 'Specificity Std': 0,
                'AUROC': 0, 'AUROC Std': 0
            }
            results.append(res)

    # 3. Save Results
    df = pd.DataFrame(results)
    df = df.sort_values(by='Balanced Accuracy', ascending=False)
    
    output_path = os.path.join(base_dir, 'output', 'reports', 'final_benchmark_42_models.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print nice table (subset of cols)
    print_cols = ['Model', 'Balanced Accuracy', 'Sensitivity (TB)', 'Specificity (NTM)', 'AUROC']
    print(tabulate(df[print_cols], headers='keys', tablefmt='github', floatfmt=".4f"))

    # --- 4. Save Top 5 Models ---
    print("\nTraining and Saving Top 5 Models...")
    models_dir = os.path.join(base_dir, 'output', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    top_5_names = df.head(5)['Model'].tolist()
    
    # Reload fresh models to train on full dataset
    all_models = get_models()
    
    for name, model in all_models:
        if name in top_5_names:
            print(f"  Retraining {name} on full dataset...")
            try:
                # Use pipeline with scaler if needed, same as in benchmark
                # Note: In benchmark we used make_pipeline(StandardScaler(), model)
                # We should save the PIPELINE, not just the model, so normalization is applied at inference.
                full_pipeline = make_pipeline(StandardScaler(), model)
                full_pipeline.fit(X, y)
                
                safe_name = name.replace(" ", "_").lower()
                save_path = os.path.join(models_dir, f"{safe_name}.pkl")
                joblib.dump(full_pipeline, save_path)
                print(f"    Saved to {save_path}")
            except Exception as e:
                print(f"    Failed to save {name}: {e}")

if __name__ == "__main__":
    main()
