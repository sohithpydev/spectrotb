
import pandas as pd
import numpy as np
import os
import re

def parse_val_std(val_str):
    """Parses '0.9412 ± 0.05' into (0.9412, 0.05). Returns (None, None) if fail."""
    if not isinstance(val_str, str):
        return None, None
    match = re.match(r"([\d\.]+)\s*±\s*([\d\.]+)", val_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def format_val_std(mean, std):
    """Formats (0.9412, 0.05) into '0.9412 ± 0.0500'"""
    return f"{mean:.4f} ± {std:.4f}"

def main():
    csv_path = '/Users/sohith/Documents/NDHU/Data/pipeline/output/reports/all_experiments_cv_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Columns to recalculate
    metric_cols = ['Balanced Accuracy', 'Accuracy', 'Sensitivity', 'Specificity', 'AUROC']
    # Handle column name mismatch if any (e.g. Balanced_Acc vs Balanced Accuracy)
    # The file view showed 'Balanced Accuracy' in line 1 header of previous view? 
    # Wait, Step 1327 showed "Balanced_Acc". Step 1355 showed "Balanced Accuracy".
    # I will check columns dynamically.
    
    actual_cols = [c for c in metric_cols if c in df.columns]
    if not actual_cols:
        # Try finding them
        mapping = {'Balanced_Acc': 'Balanced Accuracy', 'Balanced Accuracy': 'Balanced Accuracy'}
        # Just use whatever is in df.columns that looks like a metric
        actual_cols = [c for c in df.columns if c not in ['Experiment', 'Model']]
    
    experiments = df['Experiment'].unique()
    new_dfs = []

    for exp in experiments:
        print(f"Processing Experiment: {exp}")
        exp_df = df[df['Experiment'] == exp].copy()
        
        # Saparate rows
        # 1. Real Models (excluding Dummy and Mean)
        # 2. Dummy Row
        # 3. Old Mean Row (discard)
        
        models_df = exp_df[~exp_df['Model'].isin(['Mean', 'Dummy', 'DummyClassifier'])]
        dummy_df = exp_df[exp_df['Model'].isin(['Dummy', 'DummyClassifier'])]
        
        # Calculate New Mean from models_df
        new_mean_row = {'Experiment': exp, 'Model': 'Mean'}
        
        for col in actual_cols:
            vals = []
            for v_str in models_df[col]:
                m, s = parse_val_std(v_str)
                if m is not None:
                    vals.append(m)
            
            if vals:
                # Calculate mean of means and std of means (to show variation across models)
                # Or mean of means and Pooled Std? 
                # Usually "Mean" row in these tables implies "Average performance of all classifiers".
                # So Mean = mean(vals), Std = std(vals).
                mu = np.mean(vals)
                sigma = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                new_mean_row[col] = format_val_std(mu, sigma)
            else:
                new_mean_row[col] = "0.0000 ± 0.0000"

        # Create DataFrame for Mean Row
        mean_df = pd.DataFrame([new_mean_row])
        
        # Sort Models by AUROC (descending)
        # Parse AUROC for sorting
        if 'AUROC' in actual_cols:
             sort_col = 'AUROC'
        elif 'Mean CV AUROC' in actual_cols:
             sort_col = 'Mean CV AUROC'
        else:
             sort_col = actual_cols[-1]

        # Add temp sort var
        models_df['sort_val'] = models_df[sort_col].apply(lambda x: parse_val_std(x)[0] if isinstance(x, str) else 0)
        models_df = models_df.sort_values(by='sort_val', ascending=False).drop(columns=['sort_val'])
        
        # Combine: Models -> Dummy -> Mean
        parts = [models_df]
        if not dummy_df.empty:
            parts.append(dummy_df)
        parts.append(mean_df)
        
        new_exp_df = pd.concat(parts, ignore_index=True)
        new_dfs.append(new_exp_df)

    final_df = pd.concat(new_dfs, ignore_index=True)
    final_df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
