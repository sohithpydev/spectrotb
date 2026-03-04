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
    data_dir = '../tb'
    if not os.path.exists(data_dir):
        data_dir = 'tb'
        
    files = glob.glob(os.path.join(data_dir, '*.txt'))
    
    # We want to find spectra that have a massive exponentially decaying baseline shift
    # similar to the image provided by the user (huge drop from 1000-4000 m/z, then flattening out).
    # We can score this by looking at the diff between mean intensity at 2000-3000 m/z versus 8000-10000 m/z.
    
    scores = []
    
    for f in files:
        mz, intensity = load_spectrum(f)
        mask = (mz >= 2000) & (mz <= 10000)
        
        if np.sum(mask) < 100:
            continue
            
        mzi = mz[mask]
        inti = intensity[mask]
        
        early_mask = (mzi >= 2000) & (mzi <= 3000)
        late_mask = (mzi >= 8000) & (mzi <= 10000)
        
        if np.sum(early_mask) == 0 or np.sum(late_mask) == 0:
            continue
            
        early_mean = np.mean(inti[early_mask])
        late_mean = np.mean(inti[late_mask])
        
        # We want a massive drop
        drop_ratio = early_mean / (late_mean + 1e-5)
        
        # Check for visible biomarkers (ESAT-6 / CFP-10 region: ~9500 - 11000 m/z)
        biomarker_mask = (mzi >= 9500) & (mzi <= 11500)
        if np.sum(biomarker_mask) > 0:
            # We want a very sharp peak in this specific region relative to the late background
            local_max = np.max(inti[biomarker_mask])
            biomarker_score = local_max / (late_mean + 1e-5)
        else:
            biomarker_score = 0
        
        # We also want to make sure there are actually peaks overall
        peakiness = np.max(inti) / (np.mean(inti) + 1e-5)
        
        # Combine into a score: heavy weight on both massive drop and specific biomarker presence
        score = drop_ratio * biomarker_score * peakiness
        scores.append((score, f, mz, intensity))
        
    # Sort by descending score
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 5 files with the most massive baselines but visible peaks
    top_files = scores[:5]
    
    fig, axes = plt.subplots(5, 3, figsize=(18, 25), sharex='col')
    
    for i, (score, filepath, mz, intensity) in enumerate(top_files):
        filename = os.path.basename(filepath)
        
        mask = (mz >= 1000) & (mz <= 10000)
        mz = mz[mask]
        intensity = intensity[mask]
        
        # 1. Smooth
        smoothed = smoothing_savgol(intensity)
        
        # 2. Baselines
        b_als = baseline_als(smoothed)
        b_clsa = baseline_clsa(mz, smoothed)
        
        corrected_als = np.maximum(smoothed - b_als, 0)
        corrected_clsa = np.maximum(smoothed - b_clsa, 0)
        
        # Plot 1: Smoothed + Baselines
        axes[i, 0].plot(mz, smoothed, label='Smoothed', color='black', alpha=0.5, lw=1)
        axes[i, 0].plot(mz, b_als, label='ALS Baseline', color='red', linestyle='--', lw=1.5)
        axes[i, 0].plot(mz, b_clsa, label='CLSA Baseline', color='blue', linestyle='-.', lw=1.5)
        axes[i, 0].set_title(f"{filename[:30]}...\nScore: {score:.1f}")
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
        
    axes[4, 0].set_xlabel("m/z")
    axes[4, 1].set_xlabel("m/z")
    axes[4, 2].set_xlabel("m/z")
    
    plt.tight_layout()
    output_path = 'output/massive_shift_baseline_comparison.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    main()
