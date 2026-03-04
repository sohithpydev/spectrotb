# scripts/03_train_validate.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm
import joblib
import warnings

# Sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (make_scorer, recall_score, balanced_accuracy_score, 
                             accuracy_score, roc_auc_score, roc_curve)

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models_config import get_models

# Suppress warnings
warnings.filterwarnings("ignore")

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def main():
    print("--- 80/20 Training & Validation Pipeline ---")
    
    # 1. Load Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        print(f"Loaded Full Dataset: {X.shape}")
    except FileNotFoundError:
        print("Data not found. Run scripts/01_process_data.py first.")
        return

    # 2. Data Splitting (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"Training Set: {X_train.shape} (Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)})")
    print(f"Test Set:     {X_test.shape} (Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)})")
    
    # 3. Model Benchmark (CV on Training Set)
    models = get_models()
    print(f"\nEvaluating {len(models)} models with Stratified 10-Fold CV on Training Set...")
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc'
    }
    
    results = []
    
    for name, model in tqdm(models):
        try:
            pipeline = make_pipeline(StandardScaler(), model)
            scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            res = {
                'Model': name,
                # Means
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
            }
            results.append(res)
        except Exception as e:
            # print(f"Error {name}: {e}")
            pass
            
    # Save CV Results
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='CV_Bal_Acc', ascending=False)
    
    os.makedirs(os.path.join(base_dir, 'output', 'reports'), exist_ok=True)
    cv_report_path = os.path.join(base_dir, 'output', 'reports', 'benchmark_80_20_CV_results.csv')
    df_results.to_csv(cv_report_path, index=False)
    print(f"\nCV Results saved to {cv_report_path}")
    
    # 4. Top 5 Analysis
    top_5_models = df_results.head(5)
    print("\n--- Top 5 Models Analysis (External Validation) ---")
    print(tabulate(top_5_models[['Model', 'CV_Bal_Acc', 'CV_AUROC']], headers='keys', tablefmt='simple'))
    
    os.makedirs(os.path.join(base_dir, 'output', 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'output', 'plots'), exist_ok=True)
    
    top5_metrics = []
    plt.figure(figsize=(10, 8))
    
    # Re-instantiate models to get fresh ones
    all_models_dict = dict(get_models())
    
    for _, row in top_5_models.iterrows():
        name = row['Model']
        model = all_models_dict[name]
        
        print(f"Retraining {name} on full Training Set...")
        
        # Train
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        
        # Predict on Test
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_probs = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_probs = pipeline.decision_function(X_test)
            
        # Metrics
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        test_sens = recall_score(y_test, y_pred, pos_label=1)
        test_spec = specificity_score(y_test, y_pred)
        test_auroc = roc_auc_score(y_test, y_probs)
        
        # MAD (Mean Absolute Difference in AUROC)
        cv_auroc = row['CV_AUROC']
        mad_auroc = abs(cv_auroc - test_auroc)
        
        top5_metrics.append({
            'Model': name,
            'Test_Bal_Acc': test_bal_acc,
            'Test_Acc': test_acc,
            'Test_Sens': test_sens,
            'Test_Spec': test_spec,
            'Test_AUROC': test_auroc,
            'CV_AUROC': cv_auroc,
            'MAD_AUROC': mad_auroc
        })
        
        # Save Model
        safe_name = name.replace(" ", "_").lower()
        joblib.dump(pipeline, os.path.join(base_dir, 'output', 'models', f"{safe_name}.pkl"))
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {test_auroc:.3f})")

    # Finalize Plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - Top 5 Models (External Validation Set)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(base_dir, 'output', 'plots', 'top_5_roc_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"ROC Curves saved to {plot_path}")
    
    # Save Top 5 Metrics
    df_top5 = pd.DataFrame(top5_metrics)
    top5_path = os.path.join(base_dir, 'output', 'reports', 'top_5_validation_results.csv')
    df_top5.to_csv(top5_path, index=False)
    print(f"Top 5 Validation Results saved to {top5_path}")
    
    print("\nTop 5 Generalization Gap (MAD):")
    print(tabulate(df_top5[['Model', 'Test_AUROC', 'CV_AUROC', 'MAD_AUROC']], headers='keys', tablefmt='github', floatfmt=".4f"))

if __name__ == "__main__":
    main()
