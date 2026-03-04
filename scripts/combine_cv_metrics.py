
import os
import pandas as pd

def main():
    print("--- Combining CV Metrics from Biomarker Experiments ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiments_dir = os.path.join(base_dir, 'output', 'biomarker_experiments')
    output_path = os.path.join(base_dir, 'output', 'reports', 'all_experiments_cv_metrics.csv')
    
    # Define experiments and their paths
    experiments = {
        'Combined': os.path.join(experiments_dir, 'Combined', 'cv_metrics.csv'),
        'CFP-10 Only': os.path.join(experiments_dir, 'CFP-10_Only', 'cv_metrics.csv'),
        'ESAT-6 Only': os.path.join(experiments_dir, 'ESAT-6_Only', 'cv_metrics.csv')
    }
    
    all_metrics = []
    
    for exp_name, file_path in experiments.items():
        if os.path.exists(file_path):
            print(f"Loading {exp_name} metrics from: {file_path}")
            df = pd.read_csv(file_path)
            
            # Add Experiment column
            # Insert at the beginning
            df.insert(0, 'Experiment', exp_name)
            
            all_metrics.append(df)
        else:
            print(f"Warning: Metrics file not found for {exp_name}: {file_path}")
    
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Sort by Experiment and then by Test AUROC (descending) if available, else by Mean CV AUROC
        sort_cols = ['Experiment']
        if 'Mean CV AUROC' in combined_df.columns:
             sort_cols.append('Mean CV AUROC')
             ascending = [True, False]
        else:
             ascending = [True]

        combined_df = combined_df.sort_values(by=sort_cols, ascending=ascending)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Combined metrics saved to: {output_path}")
        
    else:
        print("No metrics files found to combine.")

if __name__ == "__main__":
    main()
