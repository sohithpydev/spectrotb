import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import smoothing_savgol, baseline_clsa

def load_spectrum(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except:
        data = np.loadtxt(filepath)
    if data.shape[1] > 2:
        return data[:, 0], data[:, 1]
    return data[:, 0], data[:, 1]

def main():
    test_file = '../mof_tb_test/SYN-MOF-1_32566_0_C3_1.txt'
    if not os.path.exists(test_file):
        test_file = 'mof_tb_test/SYN-MOF-1_32566_0_C3_1.txt'
        
    mz, intensity = load_spectrum(test_file)
    mask = (mz >= 2000) & (mz <= 15000)
    mz = mz[mask]
    intensity = intensity[mask]
    
    smoothed = smoothing_savgol(intensity)
    
    # 1. CLSA with small window (k=100) - current undercutting
    b_clsa_100 = baseline_clsa(mz, smoothed, k=100.0)
    
    # 2. CLSA with large window (k=500)
    b_clsa_500 = baseline_clsa(mz, smoothed, k=500.0)
    
    # 3. CLSA with sqrt transformation
    # TOF peak width is often proportional to sqrt(m/z)
    mz_trans = np.sqrt(mz)
    # the width in sqrt domain. If width at m/z=10000 is ~200 m/z.
    # d(mz) = 2 * sqrt(mz) * d(sqrt_mz) -> 200 = 2 * 100 * d(sqrt_mz) -> d(sqrt_mz) = 1
    # Let's try k_trans = 2.0
    b_clsa_trans = baseline_clsa(mz_trans, smoothed, k=2.0)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: The issue with k=100
    axes[0].plot(mz, smoothed, color='black', alpha=0.5, lw=1)
    axes[0].plot(mz, b_clsa_100, label='CLSA (k=100) - Undercutting', color='red', linestyle='--', lw=1.5)
    axes[0].set_title("k=100 (Too small, dips into peaks)")
    axes[0].legend()
    
    # Plot 2: Larger constant window k=500
    axes[1].plot(mz, smoothed, color='black', alpha=0.5, lw=1)
    axes[1].plot(mz, b_clsa_500, label='CLSA (k=500) - Wider window', color='blue', linestyle='-.', lw=1.5)
    axes[1].set_title("k=500 (Wider, prevents undercutting)")
    axes[1].legend()
    
    # Plot 3: Sqrt transformed m/z axis
    axes[2].plot(mz, smoothed, color='black', alpha=0.5, lw=1)
    axes[2].plot(mz, b_clsa_trans, label='CLSA (sqrt transform, k=2.0)', color='green', linestyle=':', lw=2)
    axes[2].set_title("Sqrt Transformed m/z (Dynamic window size)")
    axes[2].legend()
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("m/z")
    
    plt.tight_layout()
    output_path = 'output/window_size_experiments.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
