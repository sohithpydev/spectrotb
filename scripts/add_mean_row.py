
import pandas as pd
import numpy as np

def main():
    csv_path = 'pipeline/output/reports/benchmark_grouped_80_20_formatted.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    # Separate Dummy from others
    dummy_row = df[df['Model'].str.contains('Dummy', case=False)]
    other_rows = df[~df['Model'].str.contains('Dummy', case=False)]

    if other_rows.empty:
        print("No non-Dummy models found.")
        return

    # Columns to calculate mean for (all except 'Model')
    metric_cols = df.columns[1:]
    
    mean_values = {}
    
    for col in metric_cols:
        # Extract numeric values from "Mean ± Std" string
        # Assuming format "0.9314 ± 0.0601"
        try:
            # Split and take the first part (mean), convert to float
            values = other_rows[col].astype(str).apply(lambda x: float(x.split('±')[0].strip()))
            stds = other_rows[col].astype(str).apply(lambda x: float(x.split('±')[1].strip()) if '±' in x else 0.0)
            
            avg_val = values.mean()
            avg_std = stds.mean()
            
            mean_values[col] = f"{avg_val:.4f} ± {avg_std:.4f}"
        except Exception as e:
            print(f"Could not parse column {col}: {e}")
            mean_values[col] = "N/A"

    # Create the Mean Row
    mean_row = pd.DataFrame([{**{'Model': 'Mean'}, **mean_values}])

    # Concatenate: Others -> Mean -> Dummy
    df_final = pd.concat([other_rows, mean_row, dummy_row], ignore_index=True)

    # Save
    df_final.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with Mean row.")
    print(df_final.tail())

if __name__ == "__main__":
    main()
