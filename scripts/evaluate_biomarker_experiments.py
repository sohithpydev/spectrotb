
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import get_group_id

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    print("--- Evaluating Biomarker Experiments (Test Set & MAD) ---")
    print("--- Metric: AUROC ---")
    
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    exp_dir = os.path.join(base_dir, 'output', 'biomarker_experiments')
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 2. Load Data and Split
    try:
        X_full = np.load(os.path.join(data_dir, 'X.npy'))
        y_full = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = np.array([line.strip() for line in f.readlines()])
    except:
        print("Data parsing error")
        return

    groups = np.array([get_group_id(f) for f in meta['filename'].values])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_full, y_full, groups))
    
    # Test Data
    X_test_all = X_full[test_idx]
    y_test = y_full[test_idx]
    
    print(f"Test Set: {len(y_test)} samples")
    
    # 3. Define Experiments
    cfp10_mask = np.char.startswith(feature_names, 'CFP-10')
    esat6_mask = np.char.startswith(feature_names, 'ESAT-6')
    
    experiments = [
        { 'name': 'CFP-10_Only', 'indices': np.where(cfp10_mask)[0], 'label': 'CFP-10' },
        { 'name': 'ESAT-6_Only', 'indices': np.where(esat6_mask)[0], 'label': 'ESAT-6' },
        { 'name': 'Combined', 'indices': np.arange(len(feature_names)), 'label': 'Combined' }
    ]
    
    # Data structure for MAD
    # We need to store: {ModelName: {ExpLabel: {Gap, CV_Score, Test_Score}}}
    model_data = {}
    
    # 4. Evaluate Loops & Plotting
    
    for i, exp in enumerate(experiments):
        exp_name = exp['name']
        label = exp['label']
        print(f"\nEvaluating Experiment: {exp_name}")
        
        curr_exp_dir = os.path.join(exp_dir, exp_name)
        models_dir = os.path.join(curr_exp_dir, 'models')
        cv_csv = os.path.join(curr_exp_dir, 'cv_metrics.csv')
        

        # Load CV Metrics (AUROC)
        if not os.path.exists(cv_csv):
            print(f"Skipping {exp_name}, no CV metrics found.")
            continue
            
        cv_df = pd.read_csv(cv_csv)
        cv_score_map = {}
        for _, row in cv_df.iterrows():
            m = row['Model']
            if m == 'Mean': continue
            # Look for AUROC column
            val_str = str(row['AUROC'])
            try:
                mean_val = float(val_str.split('±')[0].strip())
            except:
                mean_val = np.nan
            cv_score_map[m] = mean_val

        # Filter Test Data
        indices = exp['indices']
        X_test_exp = X_test_all[:, indices]
        
        # Setup Plot (Keep plotting ROCs because they look good)
        plt.figure(figsize=(12, 10))
        models_to_plot = []
        
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.pkl'):
                    safe_name = f.replace('.pkl', '')
                    original_name = None
                    cv_score = np.nan
                    
                    for name in cv_score_map.keys():
                        if name.replace(" ", "_").lower() == safe_name:
                            original_name = name
                            cv_score = cv_score_map[name]
                            break
                    
                    if not original_name:
                        continue 
                        
                    try:
                        pipeline = joblib.load(os.path.join(models_dir, f))
                        
                        # Get Probas for Plotting & AUC
                        if hasattr(pipeline, "predict_proba"):
                            try:
                                probas = pipeline.predict_proba(X_test_exp)[:, 1]
                            except:
                                probas = pipeline.decision_function(X_test_exp)
                        else:
                            probas = pipeline.decision_function(X_test_exp)
                        
                        # Metrics (Revert to AUROC)
                        test_auc = roc_auc_score(y_test, probas)
                        
                        # Store Data for MAD (Using AUROC)
                        if not np.isnan(cv_score):
                            gap = test_auc - cv_score
                            
                            if original_name not in model_data:
                                model_data[original_name] = {}
                                
                            model_data[original_name][label] = {
                                'Gap': gap,
                                'CV_AUC': cv_score, # Renamed key
                                'Test_AUC': test_auc # Renamed key
                            }
                        
                        # Store for Plot (ROC)
                        models_to_plot.append({
                            'name': original_name,
                            'auc': test_auc,
                            'probas': probas
                        })
                            
                    except Exception as e:
                         pass
        

        # Sort and Plot ROC
        models_to_plot.sort(key=lambda x: x['auc'], reverse=True)
        for m in models_to_plot:
            fpr, tpr, _ = roc_curve(y_test, m['probas'])
            plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f"{m['name']} (AUC={m['auc']:.3f})")
            
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {label} (All Models)')
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_name = f"roc_curve_{exp_name}.png"
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path, dpi=300)
        plt.close() 

    # Generate CSV with AUROC Metrics (Reverted to AUROC as requested)
    rows = []
    labels = ['CFP-10', 'ESAT-6', 'Combined']
    
    for model, data in model_data.items():
        if all(lbl in data for lbl in labels):
            row = {'Model': model}
            abs_gaps = []
            mean_cv_auc = 0
            
            for lbl in labels:
                exp_data = data[lbl]
                gap = exp_data['Gap']
                row[f"{lbl}_Gap"] = round(gap, 4)
                row[f"{lbl}_Test_AUC"] = round(exp_data['Test_AUC'], 4)
                row[f"{lbl}_CV_AUC"] = round(exp_data['CV_AUC'], 4)
                abs_gaps.append(abs(gap))
                mean_cv_auc += exp_data['CV_AUC']
            
            mad = sum(abs_gaps) / 3
            row['MAD'] = round(mad, 4)
            row['Mean_CV_AUC'] = round(mean_cv_auc / 3, 4)
            rows.append(row)
            
    df_mad = pd.DataFrame(rows)
    
    if not df_mad.empty:
        # 1. Save Full List (Sorted by MAD)
        df_mad['Rank'] = df_mad['MAD'].rank(method='min', ascending=True)
        df_mad = df_mad.sort_values(by='Rank')
        
        cols = ['Model', 'MAD', 'Rank']
        for lbl in labels:
            cols.extend([f"{lbl}_Test_AUC", f"{lbl}_CV_AUC", f"{lbl}_Gap"])
            
        df_mad_full = df_mad[cols]
        out_csv = os.path.join(exp_dir, 'biomarker_generalization_mad.csv')
        df_mad_full.to_csv(out_csv, index=False)
        print(f"Saved Full MAD table to {out_csv}")
        
        # 2. Save Top 10 by AUROC (Sorted by Mean CV AUROC, then MAD)
        # Sort by Mean CV AUC Descending
        df_top = df_mad.sort_values(by='Mean_CV_AUC', ascending=False).head(10).copy()
        
        # Recalculate Rank based on MAD within this Top 10? Or just show them?
        # User said "do mad ranking in seperate csv". So rank these 10 by MAD.
        df_top['MAD_Rank_in_Top10'] = df_top['MAD'].rank(method='min', ascending=True)
        df_top = df_top.sort_values(by='MAD_Rank_in_Top10')
        
        cols_top = ['Model', 'MAD', 'MAD_Rank_in_Top10', 'Mean_CV_AUC']
        for lbl in labels:
            cols_top.extend([f"{lbl}_Test_AUC", f"{lbl}_CV_AUC", f"{lbl}_Gap"])
            
        df_top = df_top[cols_top]
        out_top_csv = os.path.join(exp_dir, 'biomarker_top10_mad.csv')
        df_top.to_csv(out_top_csv, index=False)
        print(f"Saved Top 10 (by AUROC) MAD table to {out_top_csv}")
        print(df_top.head(10))
        
    else:
        print("No valid models found.")

if __name__ == "__main__":
    main()
