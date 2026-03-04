
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

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

def calculate_drift_metric(mz, intensity_smooth, baseline):
    """
    Higher score is more drift.
    Metric: Mean Absolute Baseline Slope? Or just normalized mean?
    
    A simple metric: Mean Baseline / Mean Intensity (percentage of signal that is baseline).
    Since user wants "too much baseline shift", we want HIGH score.
    """
    mean_baseline = np.mean(baseline)
    mean_intensity = np.mean(intensity_smooth)
    
    if mean_intensity == 0: return 0
    
    return mean_baseline / mean_intensity

def find_high_drift_files(data_dir, n=2):
    """Finds top n TB files with HIGH baseline change."""
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    random.shuffle(files)
    
    scored_files = []
    
    # scan up to 200 files
    print(f"Scanning up to 200 files for significant drift spectra...")
    for f in files[:200]:
        mz, inte = load_spectrum(f)
        
        # Must have some signal
        if np.max(inte) < 3000: continue
            
        # Smooth
        smooth = smoothing_savgol(inte, window_length=21, polyorder=3)
        # Baseline (Using verified params)
        baseline = baseline_als(smooth, lam=100000, p=0.0001)
        
        drift_score = calculate_drift_metric(mz, smooth, baseline)
        
        # We want HIGH drift score
        scored_files.append((f, drift_score))
            
    # Sort by score DESCENDING (highest drift best)
    scored_files.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N paths
    return [x[0] for x in scored_files[:n]]

def main():
    print("--- Verifying Correction on High-Drift Spectra ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    top_files = find_high_drift_files(data_dir, n=2)
    
    if len(top_files) < 2:
        print(f"Warning: Only found {len(top_files)} suitable files.")
        
    # Plotting: 2 Rows (Files), 2 Columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    
    # Handle single file case if needed
    if len(top_files) == 1:
        axes = np.array([axes])
        
    for i, file_path in enumerate(top_files):
        print(f"Processing File {i+1}: {os.path.basename(file_path)}")
        mz, intensity_raw = load_spectrum(file_path)
        
        # 1. Smooth
        intensity_smooth = smoothing_savgol(intensity_raw, window_length=21, polyorder=3)
        
        # 2. Baseline (Corrected, p=0.0001)
        baseline = baseline_als(intensity_smooth) 
        
        # 3. Correct
        intensity_corrected = intensity_smooth - baseline
        
        # Row i, Column 0: Raw + Baseline
        ax1 = axes[i, 0]
        ax1.plot(mz, intensity_raw, color='blue', linewidth=0.8, alpha=0.9, label='Raw Data')
        ax1.plot(mz, baseline, color='black', linestyle='--', linewidth=1.5, label='Baseline')
        # Simplified title
        ax1.set_title(f"File {i+1} - Raw Data (High Drift)", loc='left', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Intensity')
        if i == 0: ax1.legend(loc='upper right')
        ax1.set_xlim(1000, 13000)
        
        # Row i, Column 1: Corrected
        ax2 = axes[i, 1]
        ax2.plot(mz, intensity_corrected, color='blue', linewidth=0.8, label='Corrected')
        ax2.set_title(f"File {i+1} - Baseline Corrected", loc='left', fontweight='bold', fontsize=10)
        ax2.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        ax2.set_xlim(1000, 13000)
    
    axes[-1, 0].set_xlabel('m/z')
    axes[-1, 1].set_xlabel('m/z')
    
    plt.tight_layout()
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'verify_high_drift_2files.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
