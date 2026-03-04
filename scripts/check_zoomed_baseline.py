
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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

def find_representative_file(data_dir):
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    random.shuffle(files)
    
    best_file = None
    max_score = -1
    
    for f in files[:50]:
        mz, inte = load_spectrum(f)
        score = get_biomarker_intensity(mz, inte)
        if score > max_score:
            max_score = score
            best_file = f
    return best_file

def main():
    print("--- Detailed Zoom Analysis ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    file_path = find_representative_file(data_dir)
    if not file_path:
        print("No file found.")
        return

    print(f"File: {os.path.basename(file_path)}")
    mz, intensity_raw = load_spectrum(file_path)
    intensity_smooth = smoothing_savgol(intensity_raw, window_length=21, polyorder=3)
    
    # Compare baselines
    baseline_old = baseline_als(intensity_smooth, lam=100000, p=0.001)    # Old (Problematic)
    baseline_new = baseline_als(intensity_smooth, lam=100000, p=0.0001)   # New (Fixed)
    
    # Plot Zoomed Region (Pick a region with broad peaks, e.g., 2000-8000)
    # Or specifically focus on a major peak complex
    zoom_range = (2000, 8000)
    mask = (mz >= zoom_range[0]) & (mz <= zoom_range[1])
    
    plt.figure(figsize=(12, 6))
    plt.plot(mz[mask], intensity_raw[mask], color='blue', alpha=0.6, label='Raw Data')
    
    # Plot Baselines
    plt.plot(mz[mask], baseline_old[mask], color='red', linestyle='--', linewidth=2, label='Old Baseline (p=0.001)')
    plt.plot(mz[mask], baseline_new[mask], color='green', linestyle='-', linewidth=2, label='New Baseline (p=0.0001)')
    
    plt.title('Zoomed View: Broad Peak Bases vs Baselines')
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'zoomed_baseline_check.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
