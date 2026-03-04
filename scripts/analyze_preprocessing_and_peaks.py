# scripts/analyze_preprocessing_and_peaks.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_spectrum
from src.preprocessing import baseline_als, normalize_tic, smoothing_savgol
from src.features import BIOMARKERS, find_peak_in_window

def calculate_snr(intensity, peak_idx, window=50, noise_region_idx=None):
    """Calculates S/N ratio for a specific peak index."""
    peak_signal = intensity[peak_idx]
    
    # Estimate noise from a 'quiet' region if provided, else local
    if noise_region_idx is not None:
        noise_start, noise_end = noise_region_idx
        noise = np.std(intensity[noise_start:noise_end])
    else:
        # Simple local estimate
        start = max(0, peak_idx - 100)
        end = min(len(intensity), peak_idx + 100)
        noise = np.std(intensity[start:end])
        
    if noise == 0: return 0
    return peak_signal / noise

def main():
    print("--- Preprocessing Analysis & Peak Statistics ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.dirname(base_dir) # Data/
    
    # 1. Load Representative TB File
    # Using the one found in previous step
    tb_file_name = "60uL TB (pH3 buffer + 5nm acid ND), 1 uL SA 45%offset, 50%laser Jeff LP_2.1_0_C8_1.txt"
    tb_path = os.path.join(data_dir, 'tb', tb_file_name)
    
    # Fallback if specific file not found (though it should be there)
    if not os.path.exists(tb_path):
        tb_files = [f for f in os.listdir(os.path.join(data_dir, 'tb')) if f.endswith('.txt')]
        tb_path = os.path.join(data_dir, 'tb', tb_files[0])
        print(f"Warning: Specific file not found, using {tb_files[0]}")
    
    mz, raw_int = load_spectrum(tb_path)
    
    # --- Part A: Preprocessing Visualization & S/N ---
    
    # 1. Savitzky-Golay
    smoothed_int = smoothing_savgol(raw_int, window_length=21, polyorder=3)
    
    # 2. ALS Baseline
    baseline = baseline_als(smoothed_int)
    corrected_int = np.maximum(smoothed_int - baseline, 0)
    
    # 3. TIC Normalization
    normalized_int = normalize_tic(corrected_int)
    
    # Plot 1: SavGol Shape Preservation (Zoom on ESAT-6 peak)
    # Target ~9813 Da
    idx_9813 = np.argmin(np.abs(mz - 9813))
    window = 100 # indices
    
    plt.figure(figsize=(10, 6))
    plt.plot(mz[idx_9813-window:idx_9813+window], raw_int[idx_9813-window:idx_9813+window], 
             label='Raw Signal', color='lightgray', linewidth=2)
    plt.plot(mz[idx_9813-window:idx_9813+window], smoothed_int[idx_9813-window:idx_9813+window], 
             label='Savitzky-Golay (w=21, p=3)', color='blue', linewidth=1.5, linestyle='--')
    plt.title("Savitzky-Golay Filter: Shape Preservation (Zoomed on ESAT-6)")
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'output', 'plots', 'preprocess_savgol_zoom.png'), dpi=300)
    plt.close()
    
    # Calculate S/N improvement
    # Define noise region (18000-19000 Da where no peaks usually are)
    noise_mask = (mz > 18000) & (mz < 19000)
    if np.any(noise_mask):
        raw_noise = np.std(raw_int[noise_mask])
        smooth_noise = np.std(smoothed_int[noise_mask])
        
        peak_raw = raw_int[idx_9813] - np.median(raw_int[noise_mask]) # Approx baseline sub for S/N calc
        peak_smooth = smoothed_int[idx_9813] - np.median(smoothed_int[noise_mask])
        
        snr_raw = peak_raw / raw_noise
        snr_smooth = peak_smooth / smooth_noise
        
        print(f"S/N Ratio (Raw): {snr_raw:.2f}")
        print(f"S/N Ratio (Smoothed): {snr_smooth:.2f}")
        print(f"S/N Improvement: {((snr_smooth - snr_raw)/snr_raw)*100:.1f}%")
        
    # Plot 2: ALS Baseline Correction (Full + Zoom)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(mz, smoothed_int, label='Original (Smoothed)', color='black', linewidth=0.8)
    plt.plot(mz, baseline, label='Estimated Baseline (ALS)', color='red', linewidth=1.5)
    plt.title("ALS Baseline Estimation")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(mz, corrected_int, label='Baseline Corrected', color='green', linewidth=0.8)
    plt.title("Corrected Spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'output', 'plots', 'preprocess_als_check.png'), dpi=300)
    plt.close()

    # --- Part B: Biomarker Frequency Stats (Training Set Only) ---
    print("\n--- Analysing Biomarker Frequency (TB Training Set 80%) ---")
    
    # 1. Gather ALL files first to replicate the split exactly
    from sklearn.model_selection import GroupShuffleSplit
    from src.utils import get_group_id
    
    root_dir = os.path.dirname(data_dir) # Documents/NDHU/Data ? No, data_dir is Data
    # data_dir is .../Data from lines 36-37
    
    # helper to get all files map
    all_files = [] 
    all_labels = [] # We need labels to split y? Or just filenames?
    # Actually we just need to know which PATIENTS are in train.
    
    # Let's verify paths
    # data_dir = .../Data
    folders = ['tb', 'ntm', 'external_tb', 'external_ntm']
    
    file_patient_map = {}
    patient_list = []
    
    for folder in folders:
        fpath = os.path.join(data_dir, folder)
        if os.path.exists(fpath):
            for f in os.listdir(fpath):
                if f.endswith('.txt'):
                    full_path = os.path.join(fpath, f)
                    pid = get_group_id(f)
                    file_patient_map[full_path] = pid
                    patient_list.append(pid)

    # Unique patients
    unique_patients = np.array(sorted(list(set(patient_list))))
    
    # But split was done on X, y arrays in pipeline.
    # GroupShuffleSplit splits GROUPS.
    # So if we apply same seed=42 to unique_patients, we get Train Patients.
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    # GSS needs X, y, groups. It splits based on groups.
    # We can just pass dummy X, y of same length as unique_patients? No, GSS splits unique groups.
    # Actually GSS index output is indices of the INPUT arrays.
    # In pipeline: split(X, y, groups) where groups is aligned with X.
    
    # To reproduce exactly, we should load metadata if possible, or reconstruct.
    # Safest: Use metadata.csv which was used in pipeline.
    meta_path = os.path.join(base_dir, 'output', 'data', 'metadata.csv')
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        # meta has 'filename' and 'label'
        # Reconstruct groups
        groups_vec = np.array([get_group_id(f) for f in meta['filename']])
        y_vec = meta['label'].values # 1=TB, 0=NTM
        
        train_idx, test_idx = next(gss.split(y_vec, y_vec, groups_vec))
        
        # Train Filenames
        train_filenames = set(meta.iloc[train_idx]['filename'].values)
        
        # Now filter for TB only
        # We need full paths. meta['filename'] is typically basename.
        # Let's map basename -> fullpath
        
        basename_to_path = {}
        tb_files_train = []
        
        # Scan again to build map and filter
        count_tb_train = 0
        
        for folder in ['tb', 'external_tb']:
            fpath = os.path.join(data_dir, folder)
            if os.path.exists(fpath):
                for f in os.listdir(fpath):
                    if f.endswith('.txt') and f in train_filenames:
                        tb_files_train.append(os.path.join(fpath, f))
                        
        print(f"Found {len(tb_files_train)} TB spectra in Training Set.")
        files = tb_files_train
        
    else:
        print("Error: metadata.csv not found, cannot reproduce split.")
        return

    counts = {k: 0 for k in BIOMARKERS.keys() if not k.endswith('_z2')} 
    total_files = len(files)
    
    primary_markers = [k for k in BIOMARKERS.keys() if not k.endswith('_z2')]
    
    for f_path in files:
        m, i = load_spectrum(f_path)
        # Preprocess logic
        s = smoothing_savgol(i)
        b = baseline_als(s)
        c = np.maximum(s - b, 0)
        n = normalize_tic(c)
        
        for key in primary_markers:
            mass = BIOMARKERS[key]
            val = find_peak_in_window(m, n, mass, 1000)
            if val > 0.0001: 
                counts[key] += 1
                
    # Create Table
    res = []
    for k, v in counts.items():
        res.append({
            'Biomarker': k,
            'Mass': BIOMARKERS[k],
            'Observed Count': v,
            'Total Samples': total_files,
            'Frequency (%)': (v/total_files)*100
        })
        
    df = pd.DataFrame(res)
    df = df.sort_values(by='Frequency (%)', ascending=False)
    print(df)
    
    df.to_csv(os.path.join(base_dir, 'output', 'reports', 'biomarker_frequency.csv'), index=False)

if __name__ == "__main__":
    main()
