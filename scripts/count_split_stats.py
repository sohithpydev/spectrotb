# scripts/count_split_stats.py
import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import GroupShuffleSplit

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import get_group_id

def main():
    print("--- Comprehensive 80/20 & CV Split Statistics ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    
    try:
        y = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        filenames = meta['filename'].values
    except FileNotFoundError:
        print("Data not found.")
        return

    groups = np.array([get_group_id(f) for f in filenames])
    
    # Replicate Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(y, y, groups))
    
    def get_stats(indices, stage_name):
        y_sub = y[indices]
        g_sub = groups[indices]
        
        # Patients
        df_sub = pd.DataFrame({'group': g_sub, 'label': y_sub})
        total_pat = df_sub['group'].nunique()
        tb_pat = df_sub[df_sub['label']==1]['group'].nunique()
        ntm_pat = df_sub[df_sub['label']==0]['group'].nunique()
        
        # Spectra
        total_spec = len(indices)
        tb_spec = np.sum(y_sub == 1)
        ntm_spec = np.sum(y_sub == 0)
        
        return {
            'Stage': stage_name,
            'Patients (Total)': total_pat,
            'Patients (TB)': tb_pat,
            'Patients (NTM)': ntm_pat,
            'Spectra (Total)': total_spec,
            'Spectra (TB)': tb_spec,
            'Spectra (NTM)': ntm_spec
        }

    rows = []
    
    # 1. Total
    rows.append(get_stats(np.arange(len(y)), "1. Total Dataset (100%)"))
    
    # 2. Test
    rows.append(get_stats(test_idx, "2. External Validation Set (20%)"))
    
    # 3. Train
    train_stats = get_stats(train_idx, "3. Training Set (80%)")
    rows.append(train_stats)
    
    # 4. CV Fold (Avg)
    # Calculate directly to avoid type mixing issues
    train_vals = list(train_stats.values())[1:] # Skip 'Stage' string
    cv_vals = [round(x/10, 1) for x in train_vals]
    
    cv_fold_stats = {
        'Stage': "   └─ Valid. Fold in 10-Fold CV (~10%)",
        'Patients (Total)': cv_vals[0],
        'Patients (TB)': cv_vals[1],
        'Patients (NTM)': cv_vals[2],
        'Spectra (Total)': cv_vals[3],
        'Spectra (TB)': cv_vals[4],
        'Spectra (NTM)': cv_vals[5]
    }
            
    rows.append(cv_fold_stats)
    
    df = pd.DataFrame(rows)
    print(tabulate(df, headers='keys', tablefmt='simple'))
    
    out_path = os.path.join(base_dir, 'output', 'reports', 'overall_split_hierarchy.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
