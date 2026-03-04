# scripts/test_missing_models.py
import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, accuracy_score

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models_config import get_models
from src.utils import get_group_id

def main():
    print("--- Testing CatBoost and QDA Only ---")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    groups = np.array([get_group_id(f) for f in meta['filename'].values])

    # 2. Split (Same Seed as before)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(y, y, groups))
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    
    # 3. Filter Models
    all_models = get_models()
    target_models = [m for m in all_models if m[0] in ['CatBoost', 'QDA']]
    
    print(f"Testing {len(target_models)} models: {[m[0] for m in target_models]}")
    
    cv = StratifiedGroupKFold(n_splits=10)
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc'
    }
    
    
    results = []
    
    for name, model in target_models:
        print(f"Running {name}...")
        try:
            pipeline = make_pipeline(StandardScaler(), model)
            scores = cross_validate(pipeline, X_train, y_train, groups=groups_train, 
                                  cv=cv, scoring=scoring, n_jobs=1) # n_jobs=1 for debug safety
            
            results.append({
                'Model': name,
                'CV_Bal_Acc': np.mean(scores['test_balanced_acc']),
                'CV_Acc': np.mean(scores['test_accuracy']),
                'CV_Sens': np.mean(scores['test_sensitivity']),
                'CV_Spec': np.mean(scores['test_specificity']),
                'CV_AUROC': np.mean(scores['test_roc_auc']),
                # Std Devs
                'CV_Bal_Acc_Std': np.std(scores['test_balanced_acc']),
                'CV_Acc_Std': np.std(scores['test_accuracy']),
                'CV_Sens_Std': np.std(scores['test_sensitivity']),
                'CV_Spec_Std': np.std(scores['test_specificity']),
                'CV_AUROC_Std': np.std(scores['test_roc_auc'])
            })
            print(f"  Success! BalAcc: {np.mean(scores['test_balanced_acc']):.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            
    if results:
        print(tabulate(results, headers='keys', tablefmt='github'))

if __name__ == "__main__":
    main()
