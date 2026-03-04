import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (make_scorer, recall_score, balanced_accuracy_score, 
                             accuracy_score, roc_auc_score, roc_curve)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset_files, load_spectrum
from src.preprocessing import preprocess_spectrum
from src.features import extract_features, get_feature_names
from src.models_config import get_models
from src.utils import get_group_id

warnings.filterwarnings("ignore")

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def format_metric(mean_val, std_val):
    return f"{mean_val:.2f} ± {std_val:.2f}"

def run_evaluation_for_subset(subset_name, X_train_sub, X_test_sub, y_train, y_test, 
                              groups_train, groups_test, output_dir_reports, output_dir_plots):
    print(f"\n{'='*50}")
    print(f"Running Evaluation for Subset: {subset_name}")
    print(f"Features: {X_train_sub.shape[1]}")
    print(f"{'='*50}")
    
    models = get_models()
    cv = StratifiedGroupKFold(n_splits=10)
    
    scoring = {
        'balanced_acc': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(specificity_score),
        'roc_auc': 'roc_auc'
    }
    
    results = []
    
    # 1. Cross Validation on 80%
    print(f"Running 10-Fold CV on 80% Training Set for {subset_name}...")
    for name, model in tqdm(models):
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_validate(pipeline, X_train_sub, y_train, groups=groups_train, 
                              cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_bal_acc = np.mean(scores['test_balanced_acc'])
        cv_auroc = np.mean(scores['test_roc_auc'])
        
        # Format the rest as text for the final CSV as requested (rounded to 2 point, with +-)
        acc_str = format_metric(np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']))
        bal_acc_str = format_metric(cv_bal_acc, np.std(scores['test_balanced_acc']))
        sens_str = format_metric(np.mean(scores['test_sensitivity']), np.std(scores['test_sensitivity']))
        spec_str = format_metric(np.mean(scores['test_specificity']), np.std(scores['test_specificity']))
        auroc_str = format_metric(cv_auroc, np.std(scores['test_roc_auc']))
        
        results.append({
            'Model': name,
            '_sort_metric': cv_bal_acc, # hidden for sorting
            'CV_AUROC_num': cv_auroc,   # hidden for test comparison
            'Accuracy': acc_str,
            'Balanced Accuracy': bal_acc_str,
            'Sensitivity': sens_str,
            'Specificity': spec_str,
            'AUROC': auroc_str
        })
        
    df_results = pd.DataFrame(results).sort_values(by='_sort_metric', ascending=False)
    
    # Save CSV
    out_csv = os.path.join(output_dir_reports, f'clsa_cv_metrics_{subset_name}.csv')
    df_results.drop(columns=['_sort_metric', 'CV_AUROC_num']).to_csv(out_csv, index=False)
    print(f"CV Metrics saved to: {out_csv}")
    
    # 2. Test Set Validation & ROC
    print(f"Evaluating ALL Models on 20% Test Set for {subset_name}...")
    models_to_eval = df_results
    
    plt.figure(figsize=(14, 12))
    test_results = []
    
    all_models_dict = dict(get_models())
    
    for _, row in models_to_eval.iterrows():
        name = row['Model']
        model = all_models_dict[name]
        
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train_sub, y_train)
        
        y_pred = pipeline.predict(X_test_sub)
        if hasattr(pipeline, "predict_proba"):
            try:
                y_probs = pipeline.predict_proba(X_test_sub)[:, 1]
            except: 
                 y_probs = pipeline.decision_function(X_test_sub)
        else:
            y_probs = pipeline.decision_function(X_test_sub)
            
        test_auroc = roc_auc_score(y_test, y_probs)
        mad_auroc = abs(row['CV_AUROC_num'] - test_auroc)
        
        test_results.append({
            'Model': name,
            'Test_AUROC': f"{test_auroc:.2f}",
            'CV_AUROC_Mean': f"{row['CV_AUROC_num']:.2f}",
            'MAD_AUROC': f"{mad_auroc:.2f}"
        })
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {test_auroc:.2f}, MAD = {mad_auroc:.2f})")
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves on 20% External Setup (CLSA) - {subset_name}')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='x-small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir_plots, f'clsa_test_roc_{subset_name}_all.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"ROC Curves saved to: {plot_path}")
    
    df_test = pd.DataFrame(test_results)
    print("\nTest Generalization Gap (MAD AUROC):")
    print(df_test.to_string(index=False))
    
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.dirname(base_dir) # Data/
    
    output_dir_reports = os.path.join(base_dir, 'output', 'reports')
    output_dir_plots = os.path.join(base_dir, 'output', 'plots')
    os.makedirs(output_dir_reports, exist_ok=True)
    os.makedirs(output_dir_plots, exist_ok=True)
    
    # 1. Load completely fresh from directories (external_tb, external_ntm, ntm, tb)
    print("--- Step 1: Loading & Real-time CLSA Preprocessing ---")
    dataset_files = load_dataset_files(data_dir)
    label_map = {
        'ntm': 0, 'tb': 1,
        'external_ntm': 0, 'external_tb': 1
    }
    
    all_files = []
    all_labels = []
    
    for key, label in label_map.items():
        files = dataset_files.get(key, [])
        for f in files:
            all_files.append(f)
            all_labels.append(label)
            
    print(f"Total files located: {len(all_files)}")
    
    # 2. Process Data
    cache_fX = os.path.join(output_dir_reports, 'clsa_X.npy')
    cache_fy = os.path.join(output_dir_reports, 'clsa_y.npy')
    cache_fg = os.path.join(output_dir_reports, 'clsa_groups.npy')
    
    if os.path.exists(cache_fX):
        print("Loading cached CLSA features...")
        X = np.load(cache_fX)
        y = np.load(cache_fy)
        groups = np.load(cache_fg)
    else:
        X = []
        y = []
        groups = []
        
        print("Extracting features (this may take a short moment due to rolling operations)...")
        for i, filepath in enumerate(tqdm(all_files)):
            try:
                mz, intensity = load_spectrum(filepath)
                if mz is None or len(mz) == 0: continue
                
                # Use CLSA method explicitly!
                mz_proc, int_proc = preprocess_spectrum(mz, intensity, baseline_method='clsa')
                feats = extract_features(mz_proc, int_proc)
                
                X.append(feats)
                y.append(all_labels[i])
                groups.append(get_group_id(os.path.basename(filepath)))
                
            except Exception as e:
                pass
                
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)
        np.save(cache_fX, X)
        np.save(cache_fy, y)
        np.save(cache_fg, groups)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    feature_names = np.array(get_feature_names())
    
    # 3. Patient Split (80/20)
    print("\n--- Step 2: Patient Group Splitting ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    
    print(f"Training Set: 80% -> {X_train.shape[0]} spectra, {len(np.unique(groups_train))} patients")
    print(f"Test Set: 20% -> {X_test.shape[0]} spectra, {len(np.unique(groups_test))} patients")
    
    # 4. Filter Feature Subsets
    # Find column indices for 'CFP-10' and 'ESAT-6'
    cfp10_cols = [idx for idx, name in enumerate(feature_names) if 'CFP-10' in name]
    esat6_cols = [idx for idx, name in enumerate(feature_names) if 'ESAT-6' in name]
    combined_cols = list(range(len(feature_names))) # All
    
    subsets = {
        'CFP-10': cfp10_cols,
        'ESAT-6': esat6_cols,
        'CFP-10_and_ESAT-6': combined_cols
    }
    
    # 5. Evaluate
    for subset_name, mask_cols in subsets.items():
        X_train_sub = X_train[:, mask_cols]
        X_test_sub = X_test[:, mask_cols]
        
        run_evaluation_for_subset(
            subset_name, 
            X_train_sub, X_test_sub, 
            y_train, y_test, 
            groups_train, groups_test, 
            output_dir_reports, output_dir_plots
        )
        
if __name__ == "__main__":
    main()
