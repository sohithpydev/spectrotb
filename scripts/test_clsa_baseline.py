import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import smoothing_savgol, baseline_als, baseline_clsa

def load_spectrum(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except:
        data = np.loadtxt(filepath)
    if data.shape[1] > 2:
        return data[:, 0], data[:, 1]
    return data[:, 0], data[:, 1]

def main():
    data_dir = '../mof_tb_test'
    if not os.path.exists(data_dir):
        data_dir = 'mof_tb_test'
        
    files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    # Let's find files with a poor baseline at the start.
    # We can quantify this by taking the ratio of the mean intensity in the first 1000 m/z
    # to the median intensity of the whole spectrum.
    baseline_scores = []
    data_store = []
    
    for f in files:
        mz, intensity = load_spectrum(f)
        mask = (mz >= 2000) & (mz <= 20000)
        mz = mz[mask]
        intensity = intensity[mask]
        
        early_mask = (mz >= 2000) & (mz <= 4000)
        if np.sum(early_mask) > 0:
            early_mean = np.mean(intensity[early_mask])
            overall_median = np.median(intensity)
            score = early_mean / (overall_median + 1e-5)
        else:
            score = 0
            
        baseline_scores.append((score, f, mz, intensity))
        
    # Sort by descending score (worst baselines at the start)
    baseline_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 3 worst baselines
    top_files = baseline_scores[:3]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex='col')
    
    for i, (score, filepath, mz, intensity) in enumerate(top_files):
        filename = os.path.basename(filepath)
        
        # 1. Smooth
        smoothed = smoothing_savgol(intensity)
        
        # 2. Baselines
        b_als = baseline_als(smoothed)
        b_clsa = baseline_clsa(mz, smoothed, k=100.0)
        
        corrected_als = np.maximum(smoothed - b_als, 0)
        corrected_clsa = np.maximum(smoothed - b_clsa, 0)
        
        # Plot 1: Smoothed + Baselines
        axes[i, 0].plot(mz, smoothed, label='Smoothed', color='black', alpha=0.5, lw=1)
        axes[i, 0].plot(mz, b_als, label='ALS Baseline', color='red', linestyle='--', lw=1.5)
        axes[i, 0].plot(mz, b_clsa, label='CLSA Baseline', color='blue', linestyle='-.', lw=1.5)
        axes[i, 0].set_title(f"{filename}\nScore: {score:.1f} - Baselines")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2: ALS Corrected
        axes[i, 1].plot(mz, corrected_als, label='ALS Corrected', color='red', lw=1)
        axes[i, 1].set_title("ALS Corrected")
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot 3: CLSA Corrected
        axes[i, 2].plot(mz, corrected_clsa, label='CLSA Corrected', color='blue', lw=1)
        axes[i, 2].set_title("CLSA Corrected")
        axes[i, 2].grid(True, alpha=0.3)
        
    axes[2, 0].set_xlabel("m/z")
    axes[2, 1].set_xlabel("m/z")
    axes[2, 2].set_xlabel("m/z")
    
    plt.tight_layout()
    output_path = 'output/poor_baseline_comparison.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot for worst baselines saved to {output_path}")

if __name__ == "__main__":
    main()
