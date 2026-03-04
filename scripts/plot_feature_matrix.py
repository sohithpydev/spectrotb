
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def main():
    print("--- Generating Feature Overlap Matrix (10 Models) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis')
    csv_path = os.path.join(analysis_dir, 'top10_shared_features.csv')
    
    if not os.path.exists(csv_path):
        print("Feature importance CSV not found.")
        return

    df = pd.read_csv(csv_path)
    feature_col = 'Feature'
    model_cols = [c for c in df.columns if c not in ['Feature', 'Mean_Importance']]
    
    # 1. Identify Top 5 Features for each model
    top5_per_model = {}
    all_selected_features = set()
    
    for model in model_cols:
        top5 = df.sort_values(by=model, ascending=False).head(5)[feature_col].tolist()
        top5_per_model[model] = set(top5)
        all_selected_features.update(top5)
        
    sorted_features = sorted(list(all_selected_features))
    
    # 2. Build Binary Matrix (Features x Models)
    matrix_data = []
    
    for feat in sorted_features:
        row = []
        for model in model_cols:
            if feat in top5_per_model[model]:
                row.append(1)
            else:
                row.append(0)
        matrix_data.append(row)
        
    df_matrix = pd.DataFrame(matrix_data, columns=model_cols, index=sorted_features)
    
    # Sort features by frequency (Row Sum)
    df_matrix['Freq'] = df_matrix.sum(axis=1)
    df_matrix = df_matrix.sort_values(by='Freq', ascending=False)
    df_matrix = df_matrix.drop(columns=['Freq'])
    
    # 3. Plot Feature Intersection Matrix (Alternative to 10-way Venn)
    plt.figure(figsize=(14, 10))
    
    # Custom cmap: 0=White, 1=Teal
    cmap = ListedColormap(['#f0f0f0', '#008080'])
    
    ax = sns.heatmap(df_matrix, annot=False, cmap=cmap, cbar=False, linewidths=1, linecolor='white')
    
    # Customize tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.title('Feature Intersection Matrix (Top 10 Models)\n(Teal = Feature is in Top 5 for Model)', fontsize=16)
    plt.ylabel('Features (Union of Top 5s)', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    
    # Annotate cells with "X" or checkmark approximation
    for y in range(df_matrix.shape[0]):
        for x in range(df_matrix.shape[1]):
            val = df_matrix.iloc[y, x]
            if val == 1:
                plt.text(x + 0.5, y + 0.5, '●', 
                         ha='center', va='center', color='white', fontsize=14)

    plt.tight_layout()
    
    plot_path = os.path.join(analysis_dir, 'feature_overlap_matrix.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Overlap Matrix to {plot_path}")
    
    # 4. Attempt a "Flower" Venn (Petal Diagram) for visual style
    # This places features in center (intersection) and others in petals
    # Not exact, but visually similar to what user might want for "many circles"
    
    # Actually, let's create a textual Venn representation on the plot
    # Calculate stats again
    n_models = len(model_cols)
    feat_counts = df_matrix.sum(axis=1)
    
    core = feat_counts[feat_counts == n_models].index.tolist()
    shared_9 = feat_counts[feat_counts == 9].index.tolist()
    
    print("\n--- Core Intersection ---")
    print(f"Present in 10/10: {core}")
    print(f"Present in 9/10: {shared_9}")

if __name__ == "__main__":
    main()
