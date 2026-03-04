# scripts/count_dataset_stats.py
import os
import sys
import pandas as pd
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import get_group_id

def main():
    print("--- Dataset Statistics (Spectra vs Patients) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # pipeline/
    data_root = os.path.dirname(base_dir) # Data/
    
    folders = [
        {'name': 'tb', 'class': 'TB', 'type': 'Internal'},
        {'name': 'ntm', 'class': 'NTM', 'type': 'Internal'},
        {'name': 'external_tb', 'class': 'TB', 'type': 'External'},
        {'name': 'external_ntm', 'class': 'NTM', 'type': 'External'}
    ]
    
    stats = []
    
    total_spectra = 0
    total_patients = set()
    
    # Track unique patients globally to handle any cross-folder duplicates (already checked, but good for validity)
    global_patient_map = {}
    
    for folder_info in folders:
        folder_path = os.path.join(data_root, folder_info['name'])
        
        folder_spectra_count = 0
        folder_patients = set()
        
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
            folder_spectra_count = len(files)
            
            for f in files:
                pid = get_group_id(f)
                # Make patient ID unique per class to avoid confusion 
                # (though leakage check showed no overlap, good to be explicit for counting)
                # We will trust the raw ID for "Unique Patients" count.
                folder_patients.add(pid)
                
                # For global tracking (e.g. knowing if Patient X is in TB or NTM)
                global_patient_map[pid] = folder_info['class']
                
        stats.append({
            'Dataset': folder_info['type'],
            'Folder': folder_info['name'],
            'Class': folder_info['class'],
            'Spectra Count': folder_spectra_count,
            'Patient Count': len(folder_patients),
            'Avg Spectra/Patient': f"{folder_spectra_count / len(folder_patients):.2f}" if len(folder_patients) > 0 else "0.0"
        })
        
        total_spectra += folder_spectra_count
        total_patients.update(folder_patients)
        
    # Create DataFrame
    df = pd.DataFrame(stats)
    
    # Calculate Totals
    total_tb_patients = sum(1 for p, c in global_patient_map.items() if c == 'TB')
    total_ntm_patients = sum(1 for p, c in global_patient_map.items() if c == 'NTM')
    
    # Add Total Row
    total_row = pd.DataFrame([{
        'Dataset': 'TOTAL',
        'Folder': 'ALL',
        'Class': 'COMBINED',
        'Spectra Count': total_spectra,
        'Patient Count': len(total_patients),
        'Avg Spectra/Patient': f"{total_spectra / len(total_patients):.2f}"
    }])
    
    df_final = pd.concat([df, total_row], ignore_index=True)
    
    # Print
    print(tabulate(df_final, headers='keys', tablefmt='github'))
    print(f"\nBreakdown of Unique Patients:")
    print(f"  TB Patients:  {total_tb_patients}")
    print(f"  NTM Patients: {total_ntm_patients}")
    print(f"  Total:        {len(total_patients)}")
    
    # Save
    out_path = os.path.join(base_dir, 'output', 'reports', 'dataset_statistics.csv')
    df_final.to_csv(out_path, index=False)
    print(f"\nSaved statistics to {out_path}")

if __name__ == "__main__":
    main()
