# scripts/analyze_biomarker_groups.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
import catboost as cb
from sklearn.base import BaseEstimator, ClassifierMixin

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_group_id

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
            allow_writing_files=False
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def __sklearn_tags__(self):
        return super().__sklearn_tags__()

def load_data(data_root):
    """Loads feature matrix X and labels y from metadata."""
    from src.data_loader import load_spectrum
    from src.preprocessing import preprocess_spectrum
    from src.features import extract_features
    
    print(f"Loading data from: {data_root}")
    
    features_list = []
    labels = []
    groups = []
    
    # Folders
    folders = [
        ('tb', 1), ('ntm', 0), 
        ('external_tb', 1), ('external_ntm', 0)
    ]
    
    for folder, label in folders:
        path = os.path.join(data_root, folder)
        if not os.path.exists(path):
            print(f"Skipping {path} (not found)")
            continue
            
        print(f"Processing {folder}...")
        for f in os.listdir(path):
            if f.endswith('.txt'):
                fpath = os.path.join(path, f)
                mz, intensity = load_spectrum(fpath)
                mz_proc, int_proc = preprocess_spectrum(mz, intensity)
                feats = extract_features(mz_proc, int_proc)
                
                features_list.append(feats)
                labels.append(label)
                groups.append(get_group_id(f))
                
    X = np.array(features_list)
    y = np.array(labels)
    groups = np.array(groups)
    
    print(f"Data Loaded: {X.shape}")
    return X, y, groups

def evaluate_ablation(model, X_test, y_test, modality_name, model_name, feature_indices=None):
    """
    Evaluates model with specific features active (others zeroed).
    If feature_indices is None, use all.
    Otherwise, zero out all indices NOT in feature_indices.
    """
    X_mod = X_test.copy()
    
    if feature_indices is not None:
        # Identify columns to zero (Inverse of kept indices)
        all_cols = set(range(X_mod.shape[1]))
        keep_cols = set(feature_indices)
        zero_cols = list(all_cols - keep_cols)
        
        X_mod[:, zero_cols] = 0.0
        
    y_pred = model.predict(X_mod)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Modality': modality_name,
        'Sensitivity': sens,
        'Specificity': spec,
        'Accuracy': acc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }, confusion_matrix(y_test, y_pred)

def plot_confusion_matrices(results_dict, output_dir):
    """Plots 3x2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = ['HistGradientBoosting', 'CatBoost']
    modalities = ['Full Data', 'CFP-10 Only', 'ESAT-6 Only']
    
    for i, model_name in enumerate(models):
        for j, mode in enumerate(modalities):
            key = (model_name, mode)
            if key in results_dict:
                cm = results_dict[key]
                ax = axes[i, j]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
                ax.set_title(f"{model_name}\n{mode}")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_xticklabels(['NTM', 'TB'])
                ax.set_yticklabels(['NTM', 'TB'])
                
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plots', 'biomarker_group_confusion_matrices.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved matrices to {out_path}")

def main():
    print("--- Biomarker Group Ablation Analysis (ESAT-6 vs CFP-10) ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'output')
    
    # 1. Load Data
    project_root = os.path.dirname(base_dir) # Data/
    X, y, groups = load_data(project_root)
    
    # 2. Split (Reproduce 80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Test Set: {len(y_test)} samples")
    
    # 3. Load Models
    models_dir = os.path.join(out_dir, 'models')
    hgb = joblib.load(os.path.join(models_dir, 'histgradientboosting.pkl'))
    cat = joblib.load(os.path.join(models_dir, 'catboost.pkl'))
    
    models = {
        'HistGradientBoosting': hgb,
        'CatBoost': cat
    }
    
    # 4. Define Feature Indices (Based on src/features.py)
    # 0: CFP-10, 1: CFP-10*
    # 2: ESAT-6_1, 3: ESAT-6_2, 4: ESAT-6*_1, 5: ESAT-6*_2
    # 6: CFP-10_z2, 7: CFP-10*_z2
    # 8-11: ESAT-6 z2 variants
    
    idx_cfp10 = [0, 1, 6, 7]
    idx_esat6 = [2, 3, 4, 5, 8, 9, 10, 11]
    
    results = []
    cm_dict = {}
    
    for name, model in models.items():
        # A. Full
        res, cm = evaluate_ablation(model, X_test, y_test, 'Full Data', name, None)
        results.append(res)
        cm_dict[(name, 'Full Data')] = cm
        
        # B. CFP-10 Only
        res, cm = evaluate_ablation(model, X_test, y_test, 'CFP-10 Only', name, idx_cfp10)
        results.append(res)
        cm_dict[(name, 'CFP-10 Only')] = cm
        
        # C. ESAT-6 Only
        res, cm = evaluate_ablation(model, X_test, y_test, 'ESAT-6 Only', name, idx_esat6)
        results.append(res)
        cm_dict[(name, 'ESAT-6 Only')] = cm
        
    # 5. Save Results
    df = pd.DataFrame(results)
    print(df[['Model', 'Modality', 'Sensitivity', 'Specificity', 'Accuracy']])
    
    csv_path = os.path.join(out_dir, 'reports', 'biomarker_group_analysis.csv')
    df.to_csv(csv_path, index=False)
    
    # 6. Plot
    plot_confusion_matrices(cm_dict, out_dir)

if __name__ == "__main__":
    main()
