
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import plot as upset_plot
from upsetplot import from_contents

def main():
    print("--- Generating UpSet Plot for Feature Intersection ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis')
    csv_path = os.path.join(analysis_dir, 'top10_shared_features.csv')
    
    if not os.path.exists(csv_path):
        print("Feature importance CSV not found.")
        return

    df = pd.read_csv(csv_path)
    feature_col = 'Feature'
    
    # Identify model columns (exclude Feature and Mean_Importance)
    model_cols = [c for c in df.columns if c not in ['Feature', 'Mean_Importance']]
    
    # We define "Used" as being in the Top 5 most important features for that model
    contents = {}
    
    print(f"Analyzing intersection of Top 5 features across {len(model_cols)} models...")
    
    for model in model_cols:
        # Get Top 5 features for this model
        # Sort by model column descending
        top5 = df.sort_values(by=model, ascending=False).head(5)[feature_col].tolist()
        contents[model] = top5
        
    # Plotting Fallback: Feature Frequency Bar Chart
    # Since upsetplot is failing due to environment issues, we visualize the consensus
    # by showing how many of the Top 10 models selected each feature in their Top 5.
    
    from collections import Counter
    all_features = []
    for model in model_cols:
        all_features.extend(contents[model])
        
    counts = Counter(all_features)
    
    # Prepare data for plotting
    features = []
    freqs = []
    for feat, count in counts.most_common():
        features.append(feat)
        freqs.append(count)
        
    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, freqs, color='teal')
    plt.xlabel('Number of Models (out of 10) containing feature in Top 5', fontsize=12)
    plt.title('Feature Consensus: Frequency in Top 10 Models', fontsize=14)
    plt.gca().invert_yaxis() # Highest frequency on top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add counts at end of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}/10', ha='left', va='center', fontweight='bold')
                 
    plt.xlim(0, 11)
    plt.tight_layout()
    
    # Save as the "upset" plot path to maintain compatibility with report expectations, 
    # or save as new name. Let's save as new and I'll update report.
    plot_path = os.path.join(analysis_dir, 'feature_consensus_frequency.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Frequency Consensus plot to {plot_path}")

    # Identify Shared Features (Intersection of All)
    shared_features = set(contents[model_cols[0]])
    for model in model_cols[1:]:
        shared_features = shared_features.intersection(set(contents[model]))
        
    print("\n--- Shared Features (Intersection of ALL 10 Models) ---")
    if shared_features:
        for f in shared_features:
            print(f"- {f}")
    else:
        print("No single feature is in the Top 5 for ALL 10 models.")
    
    print("\n--- Feature Frequency (Present in Top 5 of X Models) ---")
    for feat, count in counts.most_common():
        print(f"{feat}: {count}/10")

if __name__ == "__main__":
    main()
