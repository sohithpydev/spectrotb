
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from matplotlib_venn import venn3, venn2
from matplotlib.colors import ListedColormap

# Add pipelines path
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.dirname(current_dir)
sys.path.append(pipeline_dir)
import src 

def get_feature_importance(model, model_name, X, y):
    # Unwrap pipeline if needed
    if hasattr(model, 'steps'):
        classifier = model.steps[-1][1]
    else:
        classifier = model

    importances = None
    
    # 1. Tree-based / Linear coefficients
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # For linear models, use absolute coefficients
        if classifier.coef_.ndim > 1:
            importances = np.mean(np.abs(classifier.coef_), axis=0)
        else:
            importances = np.abs(classifier.coef_)
    elif hasattr(classifier, 'estimators_'):
        # Bagging
        if hasattr(classifier.estimators_[0], 'feature_importances_'):
            imps = np.array([est.feature_importances_ for est in classifier.estimators_])
            importances = np.mean(imps, axis=0)
            
    # 2. Fallback: Permutation Importance
    if importances is None:
        # Optimization: Use smaller n_repeats for 44 models to be fast
        try:
            r = permutation_importance(model, X, y, n_repeats=3, random_state=42, n_jobs=-1)
            importances = r.importances_mean
        except Exception as e:
            print(f"Permutation importance failed for {model_name}: {e}")
            return None
            
    if importances is None:
        return None
        
    # Normalize
    total = np.sum(np.abs(importances))
    if total > 0:
        return np.abs(importances) / total
    return importances

def main():
    print("--- Analyzing Feature Consensus (All 44 Models) ---")
    
    base_dir = os.path.dirname(pipeline_dir)
    data_dir = os.path.join(pipeline_dir, 'output', 'data')
    models_dir = os.path.join(pipeline_dir, 'output', 'biomarker_experiments', 'Combined', 'models')
    output_dir = os.path.join(pipeline_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis_all')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data/Features
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = np.array([line.strip() for line in f.readlines()])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load Metadata to find all model names (excluding Dummy)
    # Actually, we can just list .pkl files in models_dir
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'dummy' not in f.lower()]
    print(f"Found {len(model_files)} models.")
    
    imp_data = {'Feature': feature_names}
    valid_models = []
    
    # 3. Extract Importances
    for i, file in enumerate(model_files):
        name = file.replace('.pkl', '').replace('_', ' ') # Readability
        path = os.path.join(models_dir, file)
        
        print(f"Processing {i+1}/{len(model_files)}: {name}...", end='\r')
        
        try:
            model = joblib.load(path)
            imps = get_feature_importance(model, name, X, y)
            
            if imps is not None and len(imps) == len(feature_names):
                imp_data[name] = imps
                valid_models.append(name)
        except Exception as e:
            pass # Skip failed loads
            
    print(f"\nSuccessfully extracted importance for {len(valid_models)} models.")
    
    df_imp = pd.DataFrame(imp_data)
    df_imp['Mean_Importance'] = df_imp[valid_models].mean(axis=1)
    df_imp = df_imp.sort_values(by='Mean_Importance', ascending=False)
    
    # Save CSV
    df_imp.to_csv(os.path.join(output_dir, 'all_models_feature_importance.csv'), index=False)
    
    # 4. Consensus Ranking
    print("\n--- Top Features by Consensus Support ---")
    # For each feature, count how many models have it in Top 3
    
    top3_counts = {}
    for feat in feature_names:
        count = 0
        for model in valid_models:
            # Check if feat is in top 3 for this model
            top_feats = df_imp.sort_values(by=model, ascending=False).head(3)['Feature'].values
            if feat in top_feats:
                count += 1
        top3_counts[feat] = count
        
    df_counts = pd.DataFrame(list(top3_counts.items()), columns=['Feature', 'In_Top_3'])
    df_counts['Percent'] = (df_counts['In_Top_3'] / len(valid_models)) * 100
    df_counts = df_counts.sort_values(by='In_Top_3', ascending=False)
    
    print(df_counts)
    
    # 5. Consensus Matrix Plot
    # Rows = Features (Sorted by Consensus)
    # Cols = Models (Sorted by Family/Type ideally, but simple sort for now)
    
    matrix_data = []
    sorted_features = df_counts['Feature'].tolist()
    sorted_models = sorted(valid_models)
    
    for feat in sorted_features:
        row = []
        for model in sorted_models:
            # Is feature in Top 3 for this model?
            top3 = df_imp.sort_values(by=model, ascending=False).head(3)['Feature'].tolist()
            if feat in top3:
                row.append(1)
            else:
                row.append(0)
        matrix_data.append(row)
        
    df_matrix = pd.DataFrame(matrix_data, columns=sorted_models, index=sorted_features)
    
    plt.figure(figsize=(20, 10))
    cmap = ListedColormap(['#f5f5f5', '#2c7bb6']) # White / Blue
    sns.heatmap(df_matrix, cmap=cmap, cbar=False, linewidths=0.5, linecolor='white')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(f'Feature Consensus Matrix (All {len(valid_models)} Models)\nFeature in Top 3 = Blue', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_models_consensus_matrix.png'), dpi=300)
    print("Saved matrix plot.")

if __name__ == "__main__":
    main()
