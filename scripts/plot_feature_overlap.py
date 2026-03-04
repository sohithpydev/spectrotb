
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

def main():
    print("--- Visualizing Feature Overlap ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis')
    csv_path = os.path.join(analysis_dir, 'top10_shared_features.csv')
    
    if not os.path.exists(csv_path):
        print("Feature importance CSV not found. Run analyze_top10_features.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Feature Rank Heatmap (All 10 Models)
    # We need access to the individual model columns. 
    # The CSV has columns: Feature, Model1, Model2... Mean_Importance
    # Let's identify model columns
    feature_col = 'Feature'
    mean_col = 'Mean_Importance'
    model_cols = [c for c in df.columns if c not in [feature_col, mean_col]]
    
    # Create Rank DataFrame
    # Rank 1 = Highest Importance
    df_rank = df.copy()
    for col in model_cols:
        df_rank[col] = df_rank[col].rank(ascending=False)
        
    df_rank = df_rank.set_index('Feature')
    df_rank = df_rank[model_cols]
    
    # Sort features by Mean Rank across models
    df_rank['Mean_Rank'] = df_rank.mean(axis=1)
    df_rank = df_rank.sort_values(by='Mean_Rank')
    df_rank = df_rank.drop(columns=['Mean_Rank'])
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(df_rank, annot=True, fmt=".0f", cmap="viridis_r", linewidths=.5)
    plt.title("Feature Rank Heatmap (Top 10 Models)\n(Lower Rank = More Important)", fontsize=14)
    plt.tight_layout()
    
    heatmap_path = os.path.join(analysis_dir, 'feature_rank_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    print(f"Saved Heatmap to {heatmap_path}")
    
    # 2. Venn Diagram (Top 3 Model Families: HGB, XGB, CatBoost)
    # Define sets of Top 5 features for each
    targets = ['HistGradientBoosting', 'XGBoost', 'CatBoost']
    sets = {}
    
    for model in targets:
        if model in df.columns:
            # Get Top 5 features for this model
            top5 = df.nlargest(5, model)['Feature'].tolist()
            sets[model] = set(top5)
            print(f"Top 5 {model}: {top5}")
        else:
            print(f"Model {model} not found in CSV.")
            sets[model] = set()

    plt.figure(figsize=(10, 8))
    venn = venn3([sets[targets[0]], sets[targets[1]], sets[targets[2]]], 
          set_labels=(targets[0], targets[1], targets[2]))
          
    plt.title("Shared Top 5 Features (Boosting Models)", fontsize=16)
    
    venn_path = os.path.join(analysis_dir, 'top3_feature_venn.png')
    plt.savefig(venn_path, dpi=300)
    print(f"Saved Venn Diagram to {venn_path}")

if __name__ == "__main__":
    main()
