
import pandas as pd
import os
import re

def parse_auroc_mean(val):
    """Extracts the mean AUROC from string '0.9511 ± 0.0349'"""
    if isinstance(val, str):
        match = re.match(r"([\d\.]+)", val)
        if match:
            return float(match.group(1))
    return 0.0

def main():
    csv_path = '/Users/sohith/Documents/NDHU/Data/pipeline/output/reports/all_experiments_cv_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Identify distinct experiments
    # Preserving the order they appear in the file might be good, 
    # or just unique(). unique() returns in order of appearance usually.
    experiments = df['Experiment'].unique()
    
    sorted_dfs = []

    for exp in experiments:
        sub_df = df[df['Experiment'] == exp].copy()
        
        # Split into 'Mean' row and 'Others'
        mean_row = sub_df[sub_df['Model'] == 'Mean']
        other_rows = sub_df[sub_df['Model'] != 'Mean']
        
        # Sort 'Others' by AUROC descending
        # We need a temporary column for sorting
        other_rows['AUROC_Value'] = other_rows['AUROC'].apply(parse_auroc_mean)
        other_rows = other_rows.sort_values(by='AUROC_Value', ascending=False)
        other_rows = other_rows.drop(columns=['AUROC_Value'])
        
        # Combine: Others + Mean
        # If mean row exists
        if not mean_row.empty:
            sorted_group = pd.concat([other_rows, mean_row])
        else:
            sorted_group = other_rows
            
        sorted_dfs.append(sorted_group)
        
    # Concatenate all groups
    final_df = pd.concat(sorted_dfs, ignore_index=True)
    
    # Save
    final_df.to_csv(csv_path, index=False)
    print(f"Successfully re-sorted metrics for {len(experiments)} experiments.")
    print(f"Saved to {csv_path}")

if __name__ == "__main__":
    main()
