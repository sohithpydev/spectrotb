
import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
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
    print("--- Fixing CatBoost Metrics (10-Fold CV) ---")
    
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    exp_dir = os.path.join(base_dir, 'output', 'biomarker_experiments')
    
    # 2. Load Data
    try:
        X_full = np.load(os.path.join(data_dir, 'X.npy'))
        y_full = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        filenames = meta['filename'].values
        
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = np.array([line.strip() for line in f.readlines()])
    except FileNotFoundError:
        print("Data not found.")
        return

    # 3. Group Split (80/20) - Maintain consistent split key
    groups = np.array([get_group_id(f) for f in filenames])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_full, y_full, groups))
    
    X_train_all = X_full[train_idx]
    y_train_all = y_full[train_idx]
    groups_train = groups[train_idx]
    
    # 4. Define Experiments
    cfp10_mask = np.char.startswith(feature_names, 'CFP-10')
    esat6_mask = np.char.startswith(feature_names, 'ESAT-6')
    
    # Ensure masks are boolean
    # If feature_names is string array, startswith works.
    
    experiments = [
        { 'name': 'CFP-10_Only', 'indices': np.where(cfp10_mask)[0] },
        { 'name': 'ESAT-6_Only', 'indices': np.where(esat6_mask)[0] },
        { 'name': 'Combined', 'indices': np.arange(len(feature_names)) }
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

    # Get CatBoost Model Only
    print("Loading CatBoost config...")
    all_models = get_models()
    catboost_model = None
    for name, model in all_models:
        if name == 'CatBoost':
            catboost_model = (name, model)
            break
            
    if not catboost_model:
        print("CatBoost model not found in config.")
        return

    model_name, model_inst = catboost_model
    print(f"Model Found: {model_name}")

    for exp in experiments:
        exp_name = exp['name']
        print(f"\nProcessing Experiment: {exp_name}")
        
        indices = exp['indices']
        X_train = X_train_all[:, indices]
        
        curr_exp_dir = os.path.join(exp_dir, exp_name)
        models_dir = os.path.join(curr_exp_dir, 'models')
        csv_path = os.path.join(curr_exp_dir, 'cv_metrics.csv')
        
        # Run CV
        print("Running CV for CatBoost...")
        # Note: CatBoost handles scaling internally well, but pipeline uses StandardScaler for consistency
        pipeline = make_pipeline(StandardScaler(), model_inst)
        
        try:
            scores = cross_validate(pipeline, X_train, y_train_all, groups=groups_train,
                                  cv=cv, scoring=scoring, n_jobs=-1)
            
            # Format Metrics
            def fmt(key):
                vals = scores[f"test_{key}"]
                mean = np.mean(vals)
                std = np.std(vals)
                return f"{mean:.4f} ± {std:.4f}"
            
            new_row_data = {
                'Model': model_name,
                'Balanced_Acc': fmt('balanced_acc'),
                'Accuracy': fmt('accuracy'),
                'Sensitivity': fmt('sensitivity'),
                'Specificity': fmt('specificity'),
                'AUROC': fmt('roc_auc')
            }
            
            print(f"New Scores: {new_row_data}")
            
            # Update CSV
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Check column consistency
                # Load existing columns
                existing_cols = df.columns.tolist()
                
                # Filter out old CatBoost row
                df = df[df['Model'] != 'CatBoost']
                
                # Append new row
                df_new_row = pd.DataFrame([new_row_data])
                
                # Concatenate
                df = pd.concat([df, df_new_row], ignore_index=True)
                
                # If 'Mean' row exists, remove and maybe re-append (or ignore update for now)
                # To be clean, valid Mean requires full data. Let's just keep 'Mean' as is (from other models)
                # But 'CatBoost' row is now updated.
                
                # Move 'Mean' to bottom if exists
                if 'Mean' in df['Model'].values:
                    mean_row = df[df['Model'] == 'Mean']
                    df = df[df['Model'] != 'Mean']
                    df = pd.concat([df, mean_row], ignore_index=True)

                df.to_csv(csv_path, index=False)
                print(f"Updated {csv_path}")
                
            # Re-train and Save Model
            print("Retraining Final Model...")
            pipeline.fit(X_train, y_train_all)
            safe_name = model_name.replace(" ", "_").lower()
            joblib.dump(pipeline, os.path.join(models_dir, f"{safe_name}.pkl"))
            print("Saved Model.")
            
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

if __name__ == "__main__":
    main()
