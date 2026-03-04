
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

def calculate_slope_and_peak_strength(mz, intensity_smooth, baseline):
    """
    We want:
    1. Visible Background: Max(Baseline) > 2000 (meaning there is some drift)
    2. Strong Peaks: Max(Intensity) > 10000 (so peaks are prominent)
    3. Ratio: Max(Baseline) / Max(Intensity) < 0.3 (so the hump doesn't dominate)
    """
    max_baseline = np.max(baseline)
    max_intensity = np.max(intensity_smooth)
    
    if max_intensity < 8000: return -1 # Skip weak signal
    if max_baseline < 1000: return -1 # Skip too flat (Low Drift)
    
    ratio = max_baseline / max_intensity
    
    # We want visible slope but not overwhelming.
    # Ideally ratio between 0.1 and 0.4
    if 0.1 <= ratio <= 0.4:
        # Score higher for stronger peaks combined with visible baseline
        return max_intensity * (1 - abs(ratio - 0.25)) # Peak near optimal ratio
        
    return 0

def find_medium_drift_files(data_dir, n=5):
    """Finds top n TB files with MEDIUM baseline drift (visible hump/slope)."""
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    random.shuffle(files)
    
    scored_files = []
    
    # scan up to 300 files
    print(f"Scanning up to 300 files for 'decent slope' spectra...")
    for f in files[:300]:
        mz, inte = load_spectrum(f)
        
        # Smooth
        smooth = smoothing_savgol(inte, window_length=21, polyorder=3)
        # Baseline
        baseline = baseline_als(smooth, lam=100000, p=0.0001)
        
        score = calculate_slope_and_peak_strength(mz, smooth, baseline)
        
        if score > 0:
            scored_files.append((f, score))
            
    # Sort by score DESCENDING (best match)
    scored_files.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N paths
    return [x[0] for x in scored_files[:n]]

def main():
    print("--- Verifying Correction on Medium-Drift Spectra ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    top_files = find_medium_drift_files(data_dir, n=5)
    
    if len(top_files) < 5:
        print(f"Warning: Only found {len(top_files)} suitable files.")
        
    # Plotting: 5 Rows, 2 Columns
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True)
    
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
        
        # Calculate stats for title
        raw_max = np.max(intensity_raw)
        base_max = np.max(baseline)
        corr_max = np.max(intensity_corrected)
        drop_pct = ((raw_max - corr_max) / raw_max) * 100
        
        title = f"File {i+1} (Drop: {drop_pct:.1f}%)"
        ax1.set_title(title, loc='left', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Intensity')
        if i == 0: ax1.legend(loc='upper right')
        ax1.set_xlim(1000, 13000)
        
        # Row i, Column 1: Corrected
        ax2 = axes[i, 1]
        ax2.plot(mz, intensity_corrected, color='blue', linewidth=0.8, label='Corrected')
        ax2.set_title(f"Baseline Corrected", loc='left', fontweight='bold', fontsize=10)
        ax2.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        ax2.set_xlim(1000, 13000)
    
    axes[4, 0].set_xlabel('m/z')
    axes[4, 1].set_xlabel('m/z')
    
    plt.tight_layout()
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'verify_medium_drift_5files.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
