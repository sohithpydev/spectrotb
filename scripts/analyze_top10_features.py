
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Add parent directory to path to ensure 'src' is importable
# Script is in pipeline/scripts/
# We want pipeline/ to be in path
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(current_dir)
sys.path.append(pipeline_dir)

# Now we can import from src if needed, or allow pickle to find 'src'
import src # Trigger import to verify

def get_feature_importance(model, model_name, X, y):
    # Unwrap pipeline if needed
    if hasattr(model, 'steps'):
        classifier = model.steps[-1][1]
    else:
        classifier = model

    importances = None
    
    # 1. Tree-based models (RF, ET, GBM, XGB, LGBM, CatBoost)
    # Note: HGB in sklearn < 1.0 might not have it, or it might be permutation based only.
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
    # 2. Bagging Classifier (Bagging_Tree)
    elif hasattr(classifier, 'estimators_'):
        if hasattr(classifier.estimators_[0], 'feature_importances_'):
            imps = np.array([est.feature_importances_ for est in classifier.estimators_])
            importances = np.mean(imps, axis=0)
            
    # 3. Fallback: Permutation Importance validation (especially for HGB)
    if importances is None:
        print(f"[{model_name}] Calculating Permutation Importance (n_repeats=5)...")
        # Use a subset for speed if X is huge, but X is small here
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = r.importances_mean
            
    if importances is None:
        print(f"[{model_name}] No feature importances found.")
        return None
        
    # Normalize to sum to 1
    total = np.sum(importances)
    if total > 0:
        return importances / total
    return importances

def main():
    print("--- Analyzing Shared Features of Top 10 Models ---")
    
    # Paths (Already defined above as pipeline_dir)
    base_dir = os.path.dirname(pipeline_dir) # Data root? No, pipeline_dir is .../NDHU/Data/pipeline
    # Data is in pipeline/output/data
    data_dir = os.path.join(pipeline_dir, 'output', 'data')
    models_dir = os.path.join(pipeline_dir, 'output', 'biomarker_experiments', 'Combined', 'models')
    output_dir = os.path.join(pipeline_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data (Needed for Permutation Importance)
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = np.array([line.strip() for line in f.readlines()])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Define Top 10 Models
    top_10 = [
        'HistGradientBoosting', 'XGBoost', 'LightGBM', 'GradientBoosting',
        'RandomForest_200', 'RandomForest_100', 'ExtraTrees_100', 
        'CatBoost', 'ExtraTrees_200', 'Bagging_Tree'
    ]
    
    # 3. Extract Importances
    imp_data = {'Feature': feature_names}
    valid_models = []
    
    for name in top_10:
        safe_name = name.replace(" ", "_").lower()
        model_path = os.path.join(models_dir, f"{safe_name}.pkl")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        try:
            model = joblib.load(model_path)
            imps = get_feature_importance(model, name, X, y)
            
            if imps is not None:
                imp_data[name] = imps
                valid_models.append(name)
        except Exception as e:
            print(f"Error loading {name}: {e}")

    if not valid_models:
        print("No valid feature importances extracted.")
        return

    # 4. Create DataFrame & Calculate Mean
    df_imp = pd.DataFrame(imp_data)
    df_imp['Mean_Importance'] = df_imp[valid_models].mean(axis=1)
    df_imp = df_imp.sort_values(by='Mean_Importance', ascending=False)
    
    # 5. Save CSV
    csv_path = os.path.join(output_dir, 'top10_shared_features.csv')
    df_imp.to_csv(csv_path, index=False)
    print(f"Saved feature importance data to {csv_path}")
    print(df_imp[['Feature', 'Mean_Importance']])
    
    # 6. Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_imp, x='Mean_Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Shared Feature Importance (Top 10 MAD Models)', fontsize=14)
    plt.xlabel('Mean Normalized Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'top10_feature_importance.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
