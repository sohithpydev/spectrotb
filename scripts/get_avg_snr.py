import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.data_loader import load_spectrum
from src.preprocessing import baseline_als, normalize_tic, smoothing_savgol
from src.features import BIOMARKERS, PPM_TOLERANCE

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.dirname(base_dir) # Data/

# List of folders
folders = ['tb', 'ntm', 'external_tb', 'external_ntm']
file_cache = {}
for folder in folders:
    fpath = os.path.join(data_dir, folder)
    if os.path.exists(fpath):
        for f in os.listdir(fpath):
            if f.endswith('.txt'):
                file_cache[f] = os.path.join(fpath, f)

# Load metadata to get labels
meta_path = os.path.join(base_dir, 'output', 'data', 'metadata.csv')
if not os.path.exists(meta_path):
    print("Metadata not found at", meta_path)
    sys.exit(1)

meta = pd.read_csv(meta_path)
# Ensure labels are known: usually 1=TB, 0=NTM
# Let's verify with the file path
rows = []
for idx, row in meta.iterrows():
    fname = row['filename']
    label = row['label']
    if fname in file_cache:
        rows.append((file_cache[fname], label))
    else:
        # Some metadata files might be in subfolders? Just check...
        pass

def process_file(args):
    filepath, label = args
    try:
        mz, raw_int = load_spectrum(filepath)
    except Exception as e:
        return None
    
    smoothed_int = smoothing_savgol(raw_int, window_length=21, polyorder=3)
    
    # Global noise region as done in the previous specific signal S/N calculation
    global_noise_mask = (mz > 18000) & (mz < 19000)
    
    if not np.any(global_noise_mask):
        return None  # Skip if file somehow doesn't have this range
        
    global_noise_raw = np.std(raw_int[global_noise_mask])
    global_noise_smooth = np.std(smoothed_int[global_noise_mask])
    global_median_raw = np.median(raw_int[global_noise_mask])
    global_median_smooth = np.median(smoothed_int[global_noise_mask])
    
    snr_dict = {}
    for name, target_mass in BIOMARKERS.items():
        delta = target_mass * PPM_TOLERANCE / 1e6
        lower_bound = target_mass - delta
        upper_bound = target_mass + delta
        
        mask_signal = (mz >= lower_bound) & (mz <= upper_bound)
        
        if np.any(mask_signal):
            # Raw S/N using global info
            peak_raw_val = np.max(raw_int[mask_signal]) - global_median_raw
            snr_raw = peak_raw_val / global_noise_raw if global_noise_raw > 0 else 0
            snr_dict[f'{name}_raw'] = max(0, snr_raw)
            
            # Smoothed S/N using global info
            peak_smooth_val = np.max(smoothed_int[mask_signal]) - global_median_smooth
            snr_smooth = peak_smooth_val / global_noise_smooth if global_noise_smooth > 0 else 0
            snr_dict[f'{name}_smooth'] = max(0, snr_smooth)
        else:
            snr_dict[f'{name}_raw'] = np.nan
            snr_dict[f'{name}_smooth'] = np.nan
            
    snr_dict['label'] = label
    return snr_dict

if __name__ == '__main__':
    print(f"Processing {len(rows)} files...")
    
    results = []
    with Pool() as pool:
        for res in pool.imap_unordered(process_file, rows):
            if res is not None:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    tb_df = df[df['label'] == 1]
    ntm_df = df[df['label'] == 0]
    
    print("\n--- Average S/N Comparison (Raw vs Smoothed) ---")
    print(f"{'Biomarker':<15} | {'TB Raw':<10} | {'NTM Raw':<10} | {'TB Smth':<10} | {'NTM Smth':<10}")
    print("-" * 68)
    
    for name in BIOMARKERS.keys():
        tb_raw = tb_df[f'{name}_raw'].mean()
        ntm_raw = ntm_df[f'{name}_raw'].mean()
        tb_smth = tb_df[f'{name}_smooth'].mean()
        ntm_smth = ntm_df[f'{name}_smooth'].mean()
        
        print(f"{name:<15} | {tb_raw:<10.2f} | {ntm_raw:<10.2f} | {tb_smth:<10.2f} | {ntm_smth:<10.2f}")
    
    print("-" * 68)

