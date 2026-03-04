# scripts/generate_shap_plots.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import GroupShuffleSplit
import catboost as cb
from sklearn.base import BaseEstimator, ClassifierMixin

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_group_id

# Fix for SafeCatBoostClassifier unpickling
class SafeCatBoostClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, iterations=100, verbose=0, auto_class_weights='Balanced'):
        self.iterations = iterations
        self.verbose = verbose
        self.auto_class_weights = auto_class_weights
        self.model = None
    def fit(self, X, y):
        self.model = cb.CatBoostClassifier(iterations=self.iterations, verbose=self.verbose, auto_class_weights=self.auto_class_weights, allow_writing_files=False)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def __sklearn_tags__(self): return super().__sklearn_tags__()

def load_data(data_root):
    from src.data_loader import load_spectrum
    from src.preprocessing import preprocess_spectrum
    from src.features import extract_features
    
    print(f"Loading data from: {data_root}")
    features_list = []
    labels = []
    groups = []
    
    folders = [('tb', 1), ('ntm', 0), ('external_tb', 1), ('external_ntm', 0)]
    
    for folder, label in folders:
        path = os.path.join(data_root, folder)
        if not os.path.exists(path): continue
        for f in os.listdir(path):
            if f.endswith('.txt'):
                fpath = os.path.join(path, f)
                mz, intensity = load_spectrum(fpath)
                mz_proc, int_proc = preprocess_spectrum(mz, intensity)
                feats = extract_features(mz_proc, int_proc)
                features_list.append(feats)
                labels.append(label)
                groups.append(get_group_id(f))
                
    return np.array(features_list), np.array(labels), np.array(groups)

def main():
    print("--- Generatng SHAP Plots ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'output')
    project_root = os.path.dirname(base_dir)
    
    # 1. Load Data & Split
    X, y, groups = load_data(project_root)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    feature_names = [
        "CFP-10", "CFP-10*", 
        "ESAT-6_1", "ESAT-6_2", "ESAT-6*_1", "ESAT-6*_2",
        "CFP-10_z2", "CFP-10*_z2",
        "ESAT-6_1_z2", "ESAT-6_2_z2", "ESAT-6*_1_z2", "ESAT-6*_2_z2"
    ]
    
    # 2. Load Models
    models_dir = os.path.join(out_dir, 'models')
    hgb = joblib.load(os.path.join(models_dir, 'histgradientboosting.pkl'))
    cat = joblib.load(os.path.join(models_dir, 'catboost.pkl'))
    
    # 3. SHAP for CatBoost
    print("Generating SHAP for CatBoost...")
    try:
        # Check if it's a grid search result
        if hasattr(cat, 'best_estimator_'):
            cat = cat.best_estimator_
            
        # Check if it's a Pipeline and prepare TRANSFORMED data
        X_shap_cb = X_test
        cb_model = cat
        transformer_cb = None
        
        if hasattr(cat, 'named_steps'):
            print(f"CatBoost object is a Pipeline with steps: {cat.named_steps.keys()}")
            # Extract transformer
            transformer_cb = cat[:-1]
            try:
                X_shap_cb = transformer_cb.transform(X_test)
                print("Transformed X_test for CatBoost SHAP.")
            except Exception as e:
                print(f"Failed to transform X_test: {e}")
            
            # Extract model
            cb_model = cat.steps[-1][1]
        
        # Access the underlying CatBoost model from the wrapper if needed
        if hasattr(cb_model, 'model'):
            cb_model_inner = cb_model.model
        else:
            cb_model_inner = cb_model
            
        print(f"Using CatBoost model type: {type(cb_model_inner)}")

        explainer_cb = shap.TreeExplainer(cb_model_inner)
        shap_values_cb = explainer_cb.shap_values(X_shap_cb)
        
        # Convert to Explanation object for modern plotting
        # shap_values_cb is (N, Features)
        expl_cb = shap.Explanation(
            values=shap_values_cb,
            data=X_shap_cb,
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(expl_cb, show=False, max_display=12)
        plt.title("SHAP Beeswarm: CatBoost (Test Set)", fontsize=14)
        plt.savefig(os.path.join(out_dir, 'plots', 'shap_catboost_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved CatBoost SHAP plot.")
    except Exception as e:
        print(f"Error generating CatBoost SHAP: {e}")
        import traceback
        traceback.print_exc()

    # 4. SHAP for HistGradientBoosting
    print("Generating SHAP for HistGradientBoosting...")
    try:
        # Check if GridSearch
        if hasattr(hgb, 'best_estimator_'):
            hgb = hgb.best_estimator_

        # Handling Pipeline for HGB
        X_shap_hgb = X_test
        X_bg_hgb = X[train_idx]
        hgb_model = hgb
        
        if hasattr(hgb, 'named_steps'):
            transformer_hgb = hgb[:-1]
            try:
                X_shap_hgb = transformer_hgb.transform(X_test)
                X_bg_hgb = transformer_hgb.transform(X[train_idx])
                print("Transformed data for HGB SHAP.")
            except Exception as e:
                print(f"Failed to transform for HGB: {e}")
            hgb_model = hgb.steps[-1][1]
        
        # Only use background summary as kernel explainer is slow
        X_background = shap.kmeans(X_bg_hgb, 10) 
        
        # Using KernelExplainer on the ESTIMATOR, not the pipeline, with TRANSFORMED data
        explainer_k = shap.KernelExplainer(hgb_model.predict_proba, X_background)
        # nsamples='auto' or distinct number
        shap_values_k = explainer_k.shap_values(X_shap_hgb, nsamples=100)
        
        # KernelExplainer returns list for classes [class0_shap, class1_shap] or (N, M, C) array
        # We want class 1 (TB)
        shap_values_k_arr = np.array(shap_values_k)
        
        if isinstance(shap_values_k, list):
            sv = shap_values_k[1]
        elif len(shap_values_k_arr.shape) == 3 and shap_values_k_arr.shape[2] == 2:
            # shape (N, Features, 2) -> take index 1 for class 1
            sv = shap_values_k_arr[:, :, 1]
        else:
            sv = shap_values_k
            
        print(f"HGB SHAP shape after selection: {np.array(sv).shape}")
        
        # Ensure 2D
        if len(np.array(sv).shape) == 3:
             sv = np.array(sv).squeeze()

        # Convert to Explanation object
        expl_hgb = shap.Explanation(
            values=sv,
            data=X_shap_hgb,
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(expl_hgb, show=False, max_display=12)
        plt.title("SHAP Beeswarm: HistGradientBoosting (Test Set)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'plots', 'shap_hgb_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved HGB SHAP plot.")
    except Exception as e:
        print(f"Error generating HGB SHAP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
