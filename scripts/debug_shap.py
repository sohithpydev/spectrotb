
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
import catboost as cb

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
    # Simplified loader for txt files in structure
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    def preprocess_spectrum(intensity):
        # Simplified ALS
        L = len(intensity)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(10):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + 1000 * D.dot(D.transpose())
            z = spsolve(Z, w*intensity)
            w = 0.001 * (1 - np.exp(z - intensity)) # Approx
            w[w < 0] = 0.001
            # Simple fallback if this fails
        return intensity - z if 'z' in locals() else intensity

    # Actually we just need to load pickles or re-read raw
    # Re-reading raw is safer if imports are broken
    # But let's copy logic from analyze_charge_states.py
    folders = [('tb', 1), ('ntm', 0), ('external_tb', 1), ('external_ntm', 0)]
    X_list = []
    y_list = []
    groups_list = []
    
    # Feature extraction (peaks)
    target_masses = [
        10111, 10111, 10660, 10660,
        9813, 9813, 9786, 9786,
        7931, 7931, 7974, 7974
    ]
    # We need the full logic or just load the features if saved?
    # We didn't save feature matrix X as a file.
    # So we must re-extract.
    
    # ... copying full extraction logic is tedious and error prone here.
    # Let's try to fix path one last time.
    
    pass

# Forget redefining load_data, just fix path
import sys
# If running from Data/
sys.path.insert(0, os.path.join(os.getcwd(), 'pipeline'))
try:
    from scripts.analyze_charge_states import load_data, SafeCatBoostClassifier
except ImportError:
    print("STILL IMPORT ERROR")
    sys.exit(1)

def main():
    print("--- Debugging SHAP Distribution ---")
    data_dir = "/Users/sohith/Documents/NDHU/Data"
    pipeline_dir = os.path.join(data_dir, "pipeline")
    
    # 1. Load Data
    project_root = data_dir
    X, y, groups = load_data(project_root)
    feature_names = [
        "CFP-10 (10111)", "CFP-10 (10111) [Low]", "CFP-10 (10660)", "CFP-10 (10660) [Low]",
        "ESAT-6 (9813)", "ESAT-6 (9813) [Low]", "ESAT-6 (9786)", "ESAT-6 (9786) [Low]",
        "ESAT-6 (7931)", "ESAT-6 (7931) [Low]", "ESAT-6 (7974)", "ESAT-6 (7974) [Low]"
    ]
    
    # Split (Same seed)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_test = X[test_idx]
    
    # 2. Load CatBoost
    cb_path = os.path.join(pipeline_dir, "output", "models", "catboost.pkl")
    try:
        cb_pipeline = joblib.load(cb_path)
        if hasattr(cb_pipeline, 'steps'):
            cb_model = cb_pipeline.named_steps['safecatboostclassifier'].model
        else:
            cb_model = cb_pipeline
    except Exception as e:
        print(f"Failed to load CatBoost: {e}")
        return

    # 3. Calculate SHAP
    # IMPORTANT: If pipeline, we MUST transform X_test using the preprocessing steps
    # The TreeExplainer explains the boosting model, which expects TRANSFORMED features.
    X_shap = X_test
    if hasattr(cb_pipeline, 'steps'):
        # Extract all steps except the last (the estimator)
        transformer = cb_pipeline[:-1]
        try:
            X_shap = transformer.transform(X_test)
            print("Applied pipeline transformations to X_test for SHAP.")
        except Exception as e:
            print(f"Warning: Failed to transform X_test with pipeline: {e}")
            X_shap = X_test

    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer.shap_values(X_shap)
    
    print(f"SHAP Values Shape: {shap_values.shape}")
    print(f"X_shap Shape: {X_shap.shape}")
    
    # 4. Analyze Distribution
    # Check for sparsity in SHAP values
    zeros = np.isclose(shap_values, 0).sum()
    total = shap_values.size
    print(f"Sparsity (Zero SHAP values): {zeros/total:.2%}")
    
    # Check distinct values per feature
    print("\nDistinct SHAP values per feature (top 5 max):")
    for i in range(min(5, shap_values.shape[1])):
        distinct = len(np.unique(shap_values[:, i].round(5)))
        print(f"Feature {feature_names[i]}: {distinct} distinct SHAP values")
        
    print("\nDistinct X (transformed) values per feature (top 5 max):")
    for i in range(min(5, X_shap.shape[1])):
        distinct_x = len(np.unique(X_shap[:, i].round(5)))
        print(f"Feature {feature_names[i]}: {distinct_x} distinct X values")
        print(f"X range: {X_shap[:, i].min()} - {X_shap[:, i].max()}")

    # Check model predictions
    print("\nModel Predictions on X_test:")
    # Use the full pipeline if it exists, otherwise use the model
    predictor = cb_pipeline if hasattr(cb_pipeline, 'predict_proba') else cb_model
    
    preds = predictor.predict(X_test)
    probs = predictor.predict_proba(X_test)
    print(f"Unique predictions: {np.unique(preds)}")
    print(f"Unique probabilities (rounded): {np.unique(probs.round(4), axis=0)}")
    
    # explainer was defined as 'explainer' above, not 'explainer_cb'
    print(f"Base value (expected value): {explainer.expected_value}")

        
    # Check if distributions are bimodal
    # If distinct values is low (<10), it will look like vertical lines
    
    # 5. Generate a 'force spread' plot to see if it changes perception
    # We can try adjustments
    expl = shap.Explanation(
        values=shap_values,
        data=X_shap,
        feature_names=feature_names
    )
    
    plt.figure(figsize=(14, 8)) # Wider
    shap.plots.beeswarm(expl, show=False, max_display=12)
    plt.title("SHAP Beeswarm: Debug (Wider)", fontsize=14)
    plt.savefig(os.path.join(pipeline_dir, "output", "plots", "shap_debug_wide.png"), dpi=300, bbox_inches='tight')
    print("Saved shap_debug_wide.png")


if __name__ == "__main__":
    main()
