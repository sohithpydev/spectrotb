
import os
import pandas as pd
import numpy as np

def update_mean_row(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Remove existing Mean row
    df = df[df['Model'] != 'Mean']
    
    # Filter for calculation (Exclude Dummy)
    df_calc = df[df['Model'] != 'Dummy']
    
    if df_calc.empty:
        print(f"No models found in {file_path} to calculate mean.")
        return

    cols = ['Balanced_Acc', 'Accuracy', 'Sensitivity', 'Specificity', 'AUROC']
    mean_data = {'Model': 'Mean'}
    
    for col in cols:
        if col not in df.columns:
            continue
            
        values = []
        stds = []
        
        for val_str in df_calc[col]:
            try:
                # Format is "0.9256 ± 0.0677"
                parts = str(val_str).split('±')
                mean_val = float(parts[0].strip())
                std_val = float(parts[1].strip()) if len(parts) > 1 else 0.0
                
                values.append(mean_val)
                stds.append(std_val)
            except:
                pass
        
        if values:
            avg_mean = np.mean(values)
            avg_std = np.mean(stds)
            mean_data[col] = f"{avg_mean:.4f} ± {avg_std:.4f}"
        else:
            mean_data[col] = "NaN"
            
    # Append Mean Row
    df_mean = pd.DataFrame([mean_data])
    df_final = pd.concat([df, df_mean], ignore_index=True)
    
    df_final.to_csv(file_path, index=False)
    print(f"Updated Mean row in {file_path}")
    print(df_final.tail(3))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(base_dir, 'output', 'biomarker_experiments')
    
    experiments = ['CFP-10_Only', 'ESAT-6_Only', 'Combined']
    
    for exp in experiments:
        csv_path = os.path.join(exp_dir, exp, 'cv_metrics.csv')
        update_mean_row(csv_path)

if __name__ == "__main__":
    main()
