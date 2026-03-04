import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import smoothing_savgol, baseline_als, baseline_clsa, normalize_tic

def load_spectrum(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except:
        data = np.loadtxt(filepath)
    if data.shape[1] > 2:
        return data[:, 0], data[:, 1]
    return data[:, 0], data[:, 1]

def process_and_collect(files):
    # We will collect the data for plotting
    data_dict = {
        'mz': [],
        'raw': [],
        'smoothed': [],
        'norm_no_baseline': [],
        'norm_als': [],
        'norm_clsa': []
    }
    
    for f in files:
        mz, intensity = load_spectrum(f)
        
        # Apply a low-mass cutoff to remove matrix interference (e.g. m/z < 2000)
        mask = (mz >= 2000) & (mz <= 15000)
        mz = mz[mask]
        intensity = intensity[mask]
        
        if len(mz) == 0:
            continue
            
        smoothed = smoothing_savgol(intensity)
        
        # Baselines
        b_als = baseline_als(smoothed)
        b_clsa = baseline_clsa(mz, smoothed, k=100.0)
        
        # Corrections
        corr_none = np.maximum(smoothed, 0)
        corr_als = np.maximum(smoothed - b_als, 0)
        corr_clsa = np.maximum(smoothed - b_clsa, 0)
        
        # Normalization
        norm_none = normalize_tic(corr_none)
        norm_als = normalize_tic(corr_als)
        norm_clsa = normalize_tic(corr_clsa)
        
        data_dict['mz'].append(mz)
        data_dict['raw'].append(intensity)
        data_dict['smoothed'].append(smoothed)
        data_dict['norm_no_baseline'].append(norm_none)
        data_dict['norm_als'].append(norm_als)
        data_dict['norm_clsa'].append(norm_clsa)

    return data_dict

def main():
    data_dir = '../mof_tb_test'
    if not os.path.exists(data_dir):
        data_dir = 'mof_tb_test'
        
    # Get 10 random files to overlay
    np.random.seed(42)  # for reproducibility
    all_files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    if len(all_files) > 10:
        files_to_test = np.random.choice(all_files, 10, replace=False)
    else:
        files_to_test = all_files
        
    data_dict = process_and_collect(files_to_test)
    
    # Create an overly large plot to see everything clearly
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(files_to_test)))
    
    for i in range(len(data_dict['mz'])):
        mz = data_dict['mz'][i]
        c = colors[i]
        alpha = 0.5
        
        axes[0].plot(mz, data_dict['smoothed'][i], color=c, alpha=alpha, lw=1)
        axes[1].plot(mz, data_dict['norm_no_baseline'][i], color=c, alpha=alpha, lw=1)
        axes[2].plot(mz, data_dict['norm_als'][i], color=c, alpha=alpha, lw=1)
        axes[3].plot(mz, data_dict['norm_clsa'][i], color=c, alpha=alpha, lw=1)
        
    axes[0].set_title("1. Smoothed Spectra Overlay (Raw Scale, No Baseline Correction)\nNotice the huge scale variation and baseline shifts.")
    axes[1].set_title("2. Normalized (Smoothed ONLY)\nAre there artifacts? Is the baseline shift eliminated?")
    axes[2].set_title("3. Normalized (Smoothed + ALS Baseline Corrected)\nFor comparison with standard pipeline.")
    axes[3].set_title("4. Normalized (Smoothed + CLSA Baseline Corrected)\nFor comparison with standard pipeline.")
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Intensity / Normalized Intensity")
        
    axes[-1].set_xlabel("m/z")
    
    plt.tight_layout()
    output_path = 'output/baseline_necessity_check.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Overlay plot saved to {output_path}")

if __name__ == "__main__":
    main()
