
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

def find_representative_file(data_dir):
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    random.shuffle(files)
    
    best_file = None
    max_score = -1
    
    # Check more files to ensure good signal (like "File 2")
    for f in files[:100]:
        mz, inte = load_spectrum(f)
        score = get_biomarker_intensity(mz, inte)
        if score > max_score:
            max_score = score
            best_file = f
    return best_file

def main():
    print("--- Generating Representative Spectra (File 2) Plot ---")
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
    
    # 2. Baseline (Corrected, p=0.0001)
    baseline = baseline_als(intensity_smooth) 
    
    # 3. Correct
    intensity_corrected = intensity_smooth - baseline
    
    # Plotting: Raw + Baseline Left, Corrected Right. Just for 1 file.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Raw Data + Baseline
    ax1.plot(mz, intensity_raw, color='blue', linewidth=0.8, alpha=0.9, label='Raw Data')
    ax1.plot(mz, baseline, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    ax1.set_title('Raw Data', fontweight='bold', fontsize=12)
    ax1.set_xlabel('m/z')
    ax1.set_ylabel('Intensity')
    ax1.legend(loc='upper right')
    ax1.set_xlim(1000, 13000)
    
    # Baseline Corrected
    ax2.plot(mz, intensity_corrected, color='blue', linewidth=0.8, label='Corrected Spectrum')
    ax2.set_title('Baseline Corrected', fontweight='bold', fontsize=12)
    ax2.set_xlabel('m/z')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax2.set_xlim(1000, 13000)
    
    plt.tight_layout()
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'reproduction_file2_style.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
