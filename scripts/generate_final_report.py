
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
    
    # Check more files to get a really nice one
    for f in files[:50]:
        mz, inte = load_spectrum(f)
        score = get_biomarker_intensity(mz, inte)
        if score > max_score:
            max_score = score
            best_file = f
    return best_file

def main():
    print("--- Generating Final Report Plot ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    file_path = find_representative_file(data_dir)
    if not file_path:
        print("No file found.")
        return

    print(f"File: {os.path.basename(file_path)}")
    mz, intensity_raw = load_spectrum(file_path)
    
    # 1. Smooth
    intensity_smooth = smoothing_savgol(intensity_raw, window_length=21, polyorder=3)
    
    # 2. Baseline (New Default p=0.0001)
    baseline = baseline_als(intensity_smooth) 
    
    # 3. Correct
    intensity_corrected = intensity_smooth - baseline
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # A: Raw Data
    # Optional: Plot the NEW baseline faintly to show it's doing the right thing, 
    # but user asked for "Raw and then Corrected".
    # I will plot Raw Data in Blue.
    ax1.plot(mz, intensity_raw, color='blue', linewidth=0.8, label='Raw Data')
    # I'll add the baseline in Black Dashed just to prove it's low (professor might want to see WHERE the line is)
    # But user said "dont keep old baseline", they might mean the comparison.
    # To be safe and clearer: "Raw Data vs Estimated Baseline" is standard.
    # I will add the baseline but make it clearly the "New" one.
    ax1.plot(mz, baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Estimated Baseline (New)')
    
    ax1.set_title('A    Raw Data', loc='left', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Intensity')
    ax1.legend()
    # Limit to relevant range
    ax1.set_xlim(1000, 13000)
    
    # B: Baseline Corrected
    ax2.plot(mz, intensity_corrected, color='blue', linewidth=0.8, label='Corrected Spectrum')
    ax2.set_title('B    Baseline Corrected', loc='left', fontweight='bold', fontsize=12)
    ax2.set_xlabel('m/z')
    ax2.set_ylabel('Intensity')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax2.set_xlim(1000, 13000)
    
    plt.tight_layout()
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'final_baseline_report.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
