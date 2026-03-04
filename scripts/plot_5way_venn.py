
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from venn import venn

def main():
    print("--- Generating 5-Way Grouped Venn Diagram ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis')
    csv_path = os.path.join(analysis_dir, 'top10_shared_features.csv')
    
    if not os.path.exists(csv_path):
        print("Feature importance CSV not found.")
        return

    df = pd.read_csv(csv_path)
    feature_col = 'Feature'
    
    # Define Model Groups (Family mapping based on Top 10 list)
    # 1. HGB / LightGBM (Histogram-based)
    # 2. XGBoost / CatBoost (Advanced Boosting)
    # 3. GradientBoosting (Standard Boosting)
    # 4. Random Forest (RF_100, RF_200)
    # 5. Extra Trees / Bagging (ExtraTrees_100, 200, Bagging_Tree)
    
    groups = {
        "Hist-Boosting": ["HistGradientBoosting", "LightGBM"],
        "Adv. Boosting": ["XGBoost", "CatBoost"],
        "Grad. Boosting": ["GradientBoosting"],
        "Random Forest": ["RandomForest_200", "RandomForest_100"],
        "Rand. Ensembles": ["ExtraTrees_100", "ExtraTrees_200", "Bagging_Tree"]
    }
    
    sets = {}
    
    for group_name, models in groups.items():
        group_features = set()
        for model in models:
            if model in df.columns:
                # Union of Top 5 features from all models in this group
                top5 = df.sort_values(by=model, ascending=False).head(5)[feature_col].tolist()
                group_features.update(top5)
            else:
                print(f"Warning: {model} not found in CSV.")
        
        sets[group_name] = group_features
        print(f"Group '{group_name}' has {len(group_features)} features: {sorted(list(group_features))}")

    # Generate Venn
    plt.figure(figsize=(12, 12))
    venn(sets)
    plt.title("Feature Overlap (Top 5 Features Union per Model Family)", fontsize=16)
    
    plot_path = os.path.join(analysis_dir, '5way_feature_venn.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 5-way Venn to {plot_path}")

if __name__ == "__main__":
    main()
