
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from tqdm import tqdm

# Sklearn
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, balanced_accuracy_score, accuracy_score, roc_auc_score, make_scorer

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models_config import get_models
from src.utils import get_group_id

# Suppress warnings
warnings.filterwarnings("ignore")

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def main():
    print("--- Biomarker Group Experiments (10-Fold CV) ---")
    
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    exp_dir = os.path.join(base_dir, 'output', 'biomarker_experiments')
    os.makedirs(exp_dir, exist_ok=True)
    
    # 2. Load Data
    try:
        X_full = np.load(os.path.join(data_dir, 'X.npy'))
        y_full = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        filenames = meta['filename'].values
        
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = np.array([line.strip() for line in f.readlines()])
            
        print(f"Loaded Dataset: {X_full.shape}")
        print(f"Features: {len(feature_names)}")
    except FileNotFoundError:
        print("Data not found.")
        return

    # 3. Group Split (80/20) - Maintain consistent split
    groups = np.array([get_group_id(f) for f in filenames])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_full, y_full, groups))
    
    X_train_all, X_test_all = X_full[train_idx], X_full[test_idx]
    y_train_all, y_test_all = y_full[train_idx], y_full[test_idx]
    groups_train = groups[train_idx]
    
    print(f"Train/Test Split: {len(train_idx)}/{len(test_idx)} spectra")
    
    # 4. Define Experiments
    cfp10_mask = np.char.startswith(feature_names, 'CFP-10')
    esat6_mask = np.char.startswith(feature_names, 'ESAT-6')
    
    experiments = [
        {
            'name': 'CFP-10_Only',
            'indices': np.where(cfp10_mask)[0]
        },
        {
            'name': 'ESAT-6_Only',
            'indices': np.where(esat6_mask)[0]
        },
        {
            'name': 'Combined',
            'indices': np.arange(len(feature_names))
        }
    ]

    # CV Setup
    cv = StratifiedGroupKFold(n_splits=10)
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc'
    }

    # 5. Run Loop
    models_def = get_models()
    
    for exp in experiments:
        exp_name = exp['name']
        print(f"\nrunning experiment: {exp_name}")
        
        # Filter Data
        indices = exp['indices']
        X_train = X_train_all[:, indices]
        # X_test = X_test_all[:, indices] # Not using test set metrics in CSV this time, but training final model
        
        # Setup Output Dirs
        curr_exp_dir = os.path.join(exp_dir, exp_name)
        models_dir = os.path.join(curr_exp_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        results = []
        
        for model_name, model_inst in tqdm(models_def):
            try:
                pipeline = make_pipeline(StandardScaler(), model_inst)
                
                # 10-Fold CV
                scores = cross_validate(pipeline, X_train, y_train_all, groups=groups_train,
                                      cv=cv, scoring=scoring, n_jobs=-1)
                
                # Aggregate Metrics
                metrics = {}
                metrics['Model'] = model_name
                
                # Helper to format
                def fmt(key):
                    mean = np.mean(scores[f"test_{key}"])
                    std = np.std(scores[f"test_{key}"])
                    return f"{mean:.4f} ± {std:.4f}", mean, std
                
                str_ba, m_ba, s_ba = fmt('balanced_acc')
                str_acc, m_acc, s_acc = fmt('accuracy')
                str_sens, m_sens, s_sens = fmt('sensitivity')
                str_spec, m_spec, s_spec = fmt('specificity')
                str_auc, m_auc, s_auc = fmt('roc_auc')

                metrics['Balanced_Acc'] = str_ba
                metrics['Accuracy'] = str_acc
                metrics['Sensitivity'] = str_sens
                metrics['Specificity'] = str_spec
                metrics['AUROC'] = str_auc
                
                # Hidden numeric values for sorting/mean calculation
                metrics['_raw_ba_mean'] = m_ba
                metrics['_raw_acc_mean'] = m_acc
                metrics['_raw_sens_mean'] = m_sens
                metrics['_raw_spec_mean'] = m_spec
                metrics['_raw_auc_mean'] = m_auc
                
                metrics['_raw_ba_std'] = s_ba
                metrics['_raw_acc_std'] = s_acc
                metrics['_raw_sens_std'] = s_sens
                metrics['_raw_spec_std'] = s_spec
                metrics['_raw_auc_std'] = s_auc

                results.append(metrics)
                
                # Train Final Model on all Training Data and Save
                pipeline.fit(X_train, y_train_all)
                safe_name = model_name.replace(" ", "_").lower()
                joblib.dump(pipeline, os.path.join(models_dir, f"{safe_name}.pkl"))
                
            except Exception as e:
                # print(f"Error {model_name}: {e}")
                pass
        
        # DataFrame formatting
        df = pd.DataFrame(results)
        df = df.sort_values(by='_raw_ba_mean', ascending=False)
        
        # Calculate Mean Row (Average of Means ± Average of Stds)
        non_dummy = df[~df['Model'].str.contains('Dummy', case=False)]
        
        mean_metrics = {'Model': 'Mean'}
        cols = ['Balanced_Acc', 'Accuracy', 'Sensitivity', 'Specificity', 'AUROC']
        raw_map = {
            'Balanced_Acc': ('_raw_ba_mean', '_raw_ba_std'),
            'Accuracy': ('_raw_acc_mean', '_raw_acc_std'),
            'Sensitivity': ('_raw_sens_mean', '_raw_sens_std'),
            'Specificity': ('_raw_spec_mean', '_raw_spec_std'),
            'AUROC': ('_raw_auc_mean', '_raw_auc_std'),
        }
        
        for col in cols:
            m_col, s_col = raw_map[col]
            avg_mean = non_dummy[m_col].mean()
            avg_std = non_dummy[s_col].mean()
            mean_metrics[col] = f"{avg_mean:.4f} ± {avg_std:.4f}"
            
        mean_row = pd.DataFrame([mean_metrics])
        
        # Clean up raw columns for CSV
        df_final = pd.concat([df, mean_row], ignore_index=True)
        cols_to_keep = ['Model'] + cols
        df_final = df_final[cols_to_keep]
        
        csv_path = os.path.join(curr_exp_dir, 'cv_metrics.csv')
        df_final.to_csv(csv_path, index=False)
        print(f"Saved CV metrics to {csv_path}")

    print("\nAll CV experiments completed.")

if __name__ == "__main__":
    main()
