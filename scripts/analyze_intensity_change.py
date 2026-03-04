
import os
import sys
import numpy as np
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

def find_representative_file(data_dir):
    tb_dir = os.path.join(data_dir, 'tb')
    tb_files = [f for f in os.listdir(tb_dir) if f.endswith('.txt')]
    if not tb_files: return None
    random.shuffle(tb_files)
    
    best_tb = None
    max_score = -1
    
    for f in tb_files[:50]:
        path = os.path.join(tb_dir, f)
        mz, inte = load_spectrum(path)
        score = get_biomarker_intensity(mz, inte)
        if score > max_score:
            max_score = score
            best_tb = path
    return best_tb

def main():
    print("--- Analyzing Intensity Change ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    tb_file = find_representative_file(data_dir)
    if not tb_file:
        print("Error: Could not find suitable file.")
        return

    print(f"File: {os.path.basename(tb_file)}")
    
    mz, intensity_raw = load_spectrum(tb_file)
    intensity_smooth = smoothing_savgol(intensity_raw, window_length=21, polyorder=3)
    baseline = baseline_als(intensity_smooth, lam=100000, p=0.001)
    intensity_corrected = intensity_smooth - baseline
    
    raw_max = np.max(intensity_raw)
    baseline_mean = np.mean(baseline)
    baseline_max = np.max(baseline)
    corrected_max = np.max(intensity_corrected)
    
    print(f"\nStats:")
    print(f"Raw Max Intensity: {raw_max:.2f}")
    print(f"Baseline Mean Level: {baseline_mean:.2f}")
    print(f"Baseline Max Level: {baseline_max:.2f}")
    print(f"Corrected Max Intensity: {corrected_max:.2f}")
    print(f"Difference (Raw Max - Corrected Max): {raw_max - corrected_max:.2f}")
    print(f"Percentage of signal removed (approx baseline): {(baseline_max/raw_max)*100:.1f}%")

if __name__ == "__main__":
    main()
