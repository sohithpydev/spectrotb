
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
    if not os.path.exists(tb_dir): return None
    
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
    print("--- Tuning Baseline Parameters ---")
    
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
    
    # Parameters to test
    # Current: lam=10^5, p=0.001
    lams = [100000, 1000000, 10000000] # 10^5, 10^6, 10^7
    ps = [0.001, 0.0001] # 10^-3, 10^-4
    
    fig, axes = plt.subplots(len(ps), len(lams), figsize=(18, 10), sharex=True, sharey=True)
    
    for i, p in enumerate(ps):
        for j, lam in enumerate(lams):
            ax = axes[i, j]
            
            # Calculate baseline
            baseline = baseline_als(intensity_smooth, lam=lam, p=p)
            
            # Plot
            ax.plot(mz, intensity_smooth, color='gray', alpha=0.5, label='Smoothed Raw')
            ax.plot(mz, baseline, color='red', linewidth=1.5, label=f'Baseline')
            
            ax.set_title(f"lam={lam:.0e}, p={p:.0e}", fontsize=10)
            
            # Zoom in on relevant region if needed, but keeping full view is good for overall context
            ax.set_xlim(1000, 13000)
            
            if i == 0 and j == 0:
                ax.legend()
                
    plt.suptitle(f"Baseline Parameter Tuning\nFile: {os.path.basename(tb_file)}", fontsize=14)
    plt.tight_layout()
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'baseline_tuning.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
