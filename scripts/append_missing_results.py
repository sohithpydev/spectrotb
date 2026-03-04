# scripts/append_missing_results.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, recall_score, balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import catboost as cb
import joblib

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import get_group_id

# Fixed Wrapper
class SafeCatBoostClassifier(BaseEstimator, ClassifierMixin):
    # Explicitly state estimator type for sklearn < 1.6 and 1.6+
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
        
    @property
    def feature_importances_(self):
        return self.model.feature_importances_

def main():
    print("--- 1. Loading Data ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    csv_path = os.path.join(base_dir, 'output', 'reports', 'benchmark_grouped_80_20.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    groups = np.array([get_group_id(f) for f in meta['filename'].values])

    # 2. Split (Identical to 03_train_validate_grouped.py)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(y, y, groups))
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    
    print("--- 2. Benchmarking Missing Models ---")
    
    models = [
        ('QDA', QuadraticDiscriminantAnalysis(reg_param=0.5)),
        ('CatBoost', SafeCatBoostClassifier(iterations=100, verbose=0, auto_class_weights='Balanced'))
    ]
    
    cv = StratifiedGroupKFold(n_splits=10)
    
    # Define exact same scorers as 03_train_validate_grouped.py
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'accuracy': 'accuracy',
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(recall_score, pos_label=0),
        'roc_auc': 'roc_auc'  # This relies on predict_proba
    }
    
    new_rows = []
    
    # 1. QDA (Standard Scikit-Learn Model)
    print("Running QDA...")
    try:
        pipeline = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.5))
        scores = cross_validate(pipeline, X_train, y_train, groups=groups_train, 
                              cv=cv, scoring=scoring, n_jobs=1)
        new_rows.append({
            'Model': 'QDA',
            'CV_Bal_Acc': np.mean(scores['test_balanced_acc']),
            'CV_Acc': np.mean(scores['test_accuracy']),
            'CV_Sens': np.mean(scores['test_sensitivity']),
            'CV_Spec': np.mean(scores['test_specificity']),
            'CV_AUROC': np.mean(scores['test_roc_auc']),
            'CV_Bal_Acc_Std': np.std(scores['test_balanced_acc']),
            'CV_Acc_Std': np.std(scores['test_accuracy']),
            'CV_Sens_Std': np.std(scores['test_sensitivity']),
            'CV_Spec_Std': np.std(scores['test_specificity']),
            'CV_AUROC_Std': np.std(scores['test_roc_auc'])
        })
    except Exception as e:
        print(f"QDA Failed: {e}")

    # 2. CatBoost (Manual Loop to avoid Wrapper issues)
    print("Running CatBoost (Manual CV)...")
    cb_bal_acc = []
    cb_acc = []
    cb_sens = []
    cb_spec = []
    cb_auc = []
    
    scaler = StandardScaler()
    
    for fold_i, (train_i, val_i) in enumerate(cv.split(X_train, y_train, groups_train)):
        X_tr, X_val = X_train[train_i], X_train[val_i]
        y_tr, y_val = y_train[train_i], y_train[val_i]
        
        # Scale
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        # Fit
        # Use a fresh instance
        model = cb.CatBoostClassifier(iterations=100, verbose=0, auto_class_weights='Balanced', allow_writing_files=False, thread_count=1)
        model.fit(X_tr_s, y_tr)
        
        # Predict
        probs = model.predict_proba(X_val_s)[:, 1]
        preds = model.predict(X_val_s)
        
        # Metrics
        cb_bal_acc.append(balanced_accuracy_score(y_val, preds))
        cb_acc.append(accuracy_score(y_val, preds))
        cb_sens.append(recall_score(y_val, preds, pos_label=1))
        cb_spec.append(recall_score(y_val, preds, pos_label=0))
        try:
            cb_auc.append(roc_auc_score(y_val, probs))
        except:
            cb_auc.append(np.nan)
            
    new_rows.append({
        'Model': 'CatBoost',
        'CV_Bal_Acc': np.mean(cb_bal_acc),
        'CV_Acc': np.mean(cb_acc),
        'CV_Sens': np.mean(cb_sens),
        'CV_Spec': np.mean(cb_spec),
        'CV_AUROC': np.mean(cb_auc),
        'CV_Bal_Acc_Std': np.std(cb_bal_acc),
        'CV_Acc_Std': np.std(cb_acc),
        'CV_Sens_Std': np.std(cb_sens),
        'CV_Spec_Std': np.std(cb_spec),
        'CV_AUROC_Std': np.std(cb_auc)
    })

    print(f"  > CatBoost: BalAcc={new_rows[-1]['CV_Bal_Acc']:.4f}, AUROC={new_rows[-1]['CV_AUROC']:.4f}")

    if not new_rows:
        print("No new results to append.")
        return

    print("--- 3. Appending Results ---")
    df_existing = pd.read_csv(csv_path)
    
    # Remove existing entries for these models if they exist (to avoid duplicates/old nulls)
    df_existing = df_existing[~df_existing['Model'].isin([r['Model'] for r in new_rows])]
    
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Sort by Bal Index (descending)
    df_combined = df_combined.sort_values(by='CV_Bal_Acc', ascending=False)
    
    df_combined.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")
    print(df_combined[['Model', 'CV_Bal_Acc', 'CV_AUROC']].head(10))

if __name__ == "__main__":
    main()
