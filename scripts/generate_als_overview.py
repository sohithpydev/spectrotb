import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_spectrum
from src.preprocessing import smoothing_savgol, baseline_als
from src.features import BIOMARKERS

def plot_als_baseline(mz, raw_int, smooth_int, baseline, corrected, filename, file_path_str):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.15)
    
    # Plot 1: Smoothed data + ALS Baseline
    ax1 = axes[0]
    ax1.plot(mz, smooth_int, color='gray', alpha=0.7, label='Smoothed Spectrum')
    ax1.plot(mz, baseline, color='#ff7f0e', linestyle='--', linewidth=2, label='Fitted ALS Baseline\n($\lambda=10^5, p=0.0001$)')
    
    # Let's highlight the low mass region since baseline is steepest there
    ax1.set_xlim(2000, 20000)
    # Calculate a good ylim for the first plot
    max_val = np.max(smooth_int[(mz > 2000) & (mz < 20000)])
    ax1.set_ylim(-0.05 * max_val, max_val * 1.05)
    
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Step 1: Smoothed Spectrum & Estimated Baseline\nFile: {file_path_str}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: After Baseline Correction
    ax2 = axes[1]
    ax2.plot(mz, corrected, color='#1f77b4', linewidth=1.2, label='Corrected Spectrum \n(Smoothed - Baseline)')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_xlim(2000, 20000)
    ax2.set_ylim(-0.05 * max_val, max_val * 1.05)
    
    ax2.set_xlabel('m/z', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title(f'Step 2: After ALS Baseline Correction', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Asymmetric Least Squares (ALS) Baseline Correction", fontsize=16, fontweight='bold', y=1.02)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    # Use the same file from the smoothing visualization
    file_name = '26TB-SA off45% slider 50% 400shots-5nM cDND after 3rd extr by 100nM_0_G2_2.txt'
    file_path = os.path.join(data_dir, 'tb', file_name)
    
    print(f"Loading File: {file_name}")
    
    mz, raw_int = load_spectrum(file_path)
    
    # Apply smoothing
    smooth_int = smoothing_savgol(raw_int)
    
    # Apply ALS baseline 
    # Use the standard parameters from preprocessing (lam=100000, p=0.0001)
    baseline = baseline_als(smooth_int)
    
    # Correct the spectrum, keeping values >= 0 realistically
    corrected = smooth_int - baseline
    
    # Plot side by side
    filename = 'als_baseline_correction_comparison.png'
    plot_als_baseline(mz, raw_int, smooth_int, baseline, corrected, filename, file_name)

if __name__ == "__main__":
    main()

