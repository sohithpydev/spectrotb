
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.signal import savgol_filter

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data_loader import load_spectrum
    from src.preprocessing import baseline_als, smoothing_savgol
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_loader import load_spectrum
    from src.preprocessing import baseline_als, smoothing_savgol

def get_biomarker_intensity(mz, intensity):
    targets = [9813, 10100]
    total_signal = 0
    for mass in targets:
        mask = (mz > mass - 30) & (mz < mass + 30)
        if np.any(mask):
            total_signal += np.max(intensity[mask])
    return total_signal

def find_top_files(data_dir, n=3):
    """Finds top n TB files with STRONG biomarkers."""
    tb_dir = os.path.join(data_dir, 'tb')
    if not os.path.exists(tb_dir): return []
    
    tb_files = [f for f in os.listdir(tb_dir) if f.endswith('.txt')]
    if not tb_files: return []
    random.shuffle(tb_files)
    
    files_with_scores = []
    
    print(f"Scanning {min(len(tb_files), 100)} TB files for diverse examples...")
    for f in tb_files[:100]:
        path = os.path.join(tb_dir, f)
        mz, inte = load_spectrum(path)
        score = get_biomarker_intensity(mz, inte)
        if score > 0: # Only keep if there is some signal
            files_with_scores.append((path, score))
            
    # Sort by score descending and take top n
    files_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files_with_scores[:n]]

def main():
    print("--- Verifying Baseline Correction on 3 Files ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    tb_files = find_top_files(data_dir, n=3)
    
    if not tb_files:
        print("Error: Could not find suitable files.")
        return

    # Create a figure with 3 rows (files), 2 columns (A: Raw, B: Corrected)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    for i, file_path in enumerate(tb_files):
        print(f"File {i+1}: {os.path.basename(file_path)}")
        mz, intensity_raw = load_spectrum(file_path)
        
        # 1. Smoothing
        intensity_smooth = smoothing_savgol(intensity_raw, window_length=21, polyorder=3)
        
        # 2. Baseline (using new default p=0.0001 from source)
        baseline = baseline_als(intensity_smooth) # Uses updated default
        
        # 3. Correct
        intensity_corrected = intensity_smooth - baseline
        
        # Plot Row i
        # Column 0: Raw + Baseline
        ax_raw = axes[i, 0]
        ax_raw.plot(mz, intensity_raw, color='blue', linewidth=0.8, alpha=0.7, label='Raw')
        ax_raw.plot(mz, baseline, color='black', linestyle='--', linewidth=1.5, label='Baseline')
        ax_raw.set_title(f"File {i+1} - Raw Data", loc='left', fontsize=10)
        ax_raw.set_ylabel('Intensity')
        ax_raw.legend(loc='upper right')
        ax_raw.set_xlim(1000, 13000)
        
        # Column 1: Corrected
        ax_corr = axes[i, 1]
        ax_corr.plot(mz, intensity_corrected, color='blue', linewidth=0.8)
        ax_corr.set_title(f"File {i+1} - Baseline Corrected", loc='left', fontsize=10)
        ax_corr.set_xlim(1000, 13000)
        
        # Add a zero line to check if it cuts below
        ax_corr.axhline(0, color='gray', linestyle=':', alpha=0.5)

    axes[2, 0].set_xlabel('m/z')
    axes[2, 1].set_xlabel('m/z')
    
    plt.tight_layout()
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'reproduction_3files_check.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
