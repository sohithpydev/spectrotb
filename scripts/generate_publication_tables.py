
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import re

def main():
    print("--- Generating Publication Tables ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'output', 'data')
    report_path = os.path.join(base_dir, 'output', 'reports', 'publication_tables.md')
    
    # 1. Load Data
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        
        # Load feature names
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
            
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['Label'] = y
        df['Filename'] = meta['filename']
        # Map labels
        df['Group'] = df['Label'].map({0: 'NTM', 1: 'TB'})
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ---------------------------------------------------------
    # Table 1: Cohort Characteristics (Derived from Metadata)
    # ---------------------------------------------------------
    print("Generating Table 1...")
    
    # Extract Year from filename (e.g., 20211015 -> 2021)
    # Pattern: 4 digits starting with 202 (2020-2029) anywhere in string
    def extract_year(fname):
        match = re.search(r'(202[0-9])', str(fname))
        if match:
            return match.group(1)
        return "Unknown"
        
    df['Year'] = df['Filename'].apply(extract_year)
    
    # Analyze Demographics
    n_ntm = len(df[df['Group'] == 'NTM'])
    n_tb = len(df[df['Group'] == 'TB'])
    total = len(df)
    
    table1 = []
    table1.append(f"| Variable | NTM (N={n_ntm}) | TB (N={n_tb}) | P-value* |")
    table1.append("|:---|:---|:---|---:|")
    
    # 1. Total Count (Row 1)
    # table1.append(f"| Total Samples | {n_ntm} ({n_ntm/total:.1%}) | {n_tb} ({n_tb/total:.1%}) | - |")
    
    # 2. Collection Year (Categorical)
    years = sorted(df['Year'].unique())
    if 'Unknown' in years: # Move to end
        years.remove('Unknown')
        years.append('Unknown')
        
    for yr in years:
        n_yr_ntm = len(df[(df['Group'] == 'NTM') & (df['Year'] == yr)])
        n_yr_tb = len(df[(df['Group'] == 'TB') & (df['Year'] == yr)])
        
        # Format: N (%)
        val_ntm = f"{n_yr_ntm} ({n_yr_ntm/n_ntm*100:.1f}%)"
        val_tb = f"{n_yr_tb} ({n_yr_tb/n_tb*100:.1f}%)"
        
        table1.append(f"| Collection Year: {yr} | {val_ntm} | {val_tb} | |")

    # ---------------------------------------------------------
    # Table 2: Metabolite/Biomarker Comparison
    # ---------------------------------------------------------
    print("Generating Table 2...")
    
    table2 = []
    table2.append(f"| Biomarker (m/z) | NTM % Pos | TB % Pos | NTM Median (IQR) | TB Median (IQR) | P-value | Fold Change |")
    table2.append("|:---|---:|---:|:---|:---|---:|---:|")
    
    # Loop features
    for feat in feature_names:
        ntm_vals = df[df['Group'] == 'NTM'][feat].values
        tb_vals = df[df['Group'] == 'TB'][feat].values
        
        # Detection Rate
        pos_ntm = np.sum(ntm_vals > 0)
        pct_ntm = (pos_ntm / n_ntm) * 100
        
        pos_tb = np.sum(tb_vals > 0)
        pct_tb = (pos_tb / n_tb) * 100
        
        # Stats
        med_ntm = np.median(ntm_vals)
        q1_ntm = np.percentile(ntm_vals, 25)
        q3_ntm = np.percentile(ntm_vals, 75)
        
        med_tb = np.median(tb_vals)
        q1_tb = np.percentile(tb_vals, 25)
        q3_tb = np.percentile(tb_vals, 75)
        
        # Test
        try:
            u_stat, p_val = mannwhitneyu(ntm_vals, tb_vals, alternative='two-sided')
        except:
            p_val = 1.0
            
        # Format P-value
        if p_val < 0.001:
            p_str = "**< 0.001**"
        elif p_val < 0.05:
            p_str = f"**{p_val:.3f}**"
        else:
            p_str = f"{p_val:.3f}"
            
        # Fold Change (Median Ratio)
        if med_ntm == 0:
            if med_tb == 0:
                fc_str = "1.00" # Both zero
            else:
                fc_str = "> 100" # Infinite
        else:
            fc = med_tb / med_ntm
            fc_str = f"{fc:.2f}"
        
        # Format Cells: Use 4 decimal places given small values
        fmt = ".4f"
        
        cell_ntm = f"{med_ntm:{fmt}} ({q1_ntm:{fmt}}, {q3_ntm:{fmt}})"
        cell_tb = f"{med_tb:{fmt}} ({q1_tb:{fmt}}, {q3_tb:{fmt}})"
        
        row_str = f"| {feat} | {pct_ntm:.1f}% | {pct_tb:.1f}% | {cell_ntm} | {cell_tb} | {p_str} | {fc_str} |"
        table2.append(row_str)
        
    # Save to CSV
    # Parse markdown table rows into list of lists
    csv_data = []
    headers = [c.strip() for c in table2[0].split('|')[1:-1]]
    
    for row in table2[2:]:
        # Split by pipe and strip whitespace
        cols = [c.strip() for c in row.split('|')[1:-1]]
        csv_data.append(cols)
        
    df_table2 = pd.DataFrame(csv_data, columns=headers)
    # Clean up markdown formatting for CSV
    # Remove ** bolding
    df_table2 = df_table2.replace(r'\*\*', '', regex=True)
    csv_path = os.path.join(base_dir, 'output', 'reports', 'biomarker_table2.csv')
    df_table2.to_csv(csv_path, index=False)
    print(f"Table 2 CSV saved to {csv_path}")

    # Write Report
    with open(report_path, 'w') as f:
        f.write("# Publication Tables\n\n")
        
        f.write("## Table 1: Dataset Characteristics (Derived)\n")
        f.write("*Note: Demographic data was unavailable. Characteristics derived from filename metadata.*\n\n")
        f.write("\n".join(table1))
        f.write("\n\n")
        
        f.write("## Table 2: Quantitative Biomarker Profile\n")
        f.write("Comparison of extracted feature intensities between groups. **% Pos** indicates the detection rate (samples with non-zero intensity). Values presented as **Median (IQR)**. P-values calculated via Mann-Whitney U test.\n\n")
        f.write("\n".join(table2))
        f.write("\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
