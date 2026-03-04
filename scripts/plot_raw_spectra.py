# scripts/plot_raw_spectra.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_spectrum
from src.features import BIOMARKERS

def plot_spectrum_with_markers(ax, mz, intensity, title, color, mark_peaks=False):
    ax.plot(mz, intensity, color=color, linewidth=0.8, label='Raw Spectrum')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel("Intensity")
    ax.set_xlim(2000, 12000) # Focus on the relevant range for these biomarkers
    
    if mark_peaks:
        # Get y-limit to draw lines appropriately
        ymin, ymax = ax.get_ylim()
        
        # Plot markers
        for name, mass in BIOMARKERS.items():
            # Only label the primary singly charged ones to avoid clutter
            is_primary = not name.endswith('_z2')
            
            # Check if peak is roughly in this spectrum (just visual check logic)
            # Draw vertical line
            ax.axvline(x=mass, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            
            if is_primary:
                # Use annotate for clearer marking with arrow
                ax.annotate(name, 
                           xy=(mass, ymax*0.9), 
                           xytext=(mass, ymax*0.95),
                           rotation=90, 
                           fontsize=14, 
                           color='darkred', 
                           backgroundcolor='white',
                           ha='center',
                           va='bottom',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                           arrowprops=dict(arrowstyle="->", color='red', lw=1.5))

def get_biomarker_intensity(mz, intensity):
    """Calculates sum of intensities at key biomarker locations."""
    # Key markers: ESAT-6 (9813), CFP-10 (10100)
    targets = [9813, 10100]
    total_signal = 0
    
    for mass in targets:
        # Simple window search +/- 30 Da for raw check
        mask = (mz > mass - 30) & (mz < mass + 30)
        if np.any(mask):
            total_signal += np.max(intensity[mask])
            
    return total_signal

def find_representative_files(data_dir):
    """Finds a TB file with STRONG biomarkers and an NTM file WITHOUT them."""
    tb_dir = os.path.join(data_dir, 'tb')
    ntm_dir = os.path.join(data_dir, 'ntm')
    
    # 1. Search TB
    best_tb = None
    max_score = -1
    
    # Limit scan to 50 random files if too many to save time, or scan all
    tb_files = [f for f in os.listdir(tb_dir) if f.endswith('.txt')]
    random.shuffle(tb_files)
    
    print("Scanning TB files for strong signal...")
    for f in tb_files[:100]:
        path = os.path.join(tb_dir, f)
        mz, inte = load_spectrum(path)
        score = get_biomarker_intensity(mz, inte)
        
        if score > max_score:
            max_score = score
            best_tb = path
            
    # 2. Search NTM (cleanest at those spots but valid signal)
    best_ntm = None
    min_score = float('inf')
    
    ntm_files = [f for f in os.listdir(ntm_dir) if f.endswith('.txt')]
    random.shuffle(ntm_files)
    
    print("Scanning NTM files for clean signal...")
    for f in ntm_files[:50]:
        path = os.path.join(ntm_dir, f)
        mz, inte = load_spectrum(path)
        score = get_biomarker_intensity(mz, inte)
        
        # We want low biomarker score, but decent max intensity (not an empty scan)
        if np.max(inte) > 1000 and score < min_score:
            min_score = score
            best_ntm = path
            
    return best_tb, best_ntm

def main():
    print("--- Plotting Representative Spectra (High Quality Selection) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.dirname(base_dir) # Data/
    
    tb_file, ntm_file = find_representative_files(data_dir)
    
    if not tb_file or not ntm_file:
        print("Error: Could not find suitable files.")
        return

    print(f"Selected TB: {os.path.basename(tb_file)}")
    print(f"Selected NTM: {os.path.basename(ntm_file)}")
    
    # Load
    mz_tb, int_tb = load_spectrum(tb_file)
    mz_ntm, int_ntm = load_spectrum(ntm_file)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    
    # TB Plot
    # Use the loaded file names for titles
    plot_spectrum_with_markers(axes[0], mz_tb, int_tb, "MTC", 'royalblue', mark_peaks=True)
    
    # NTM Plot
    plot_spectrum_with_markers(axes[1], mz_ntm, int_ntm, "NTM", 'darkorange', mark_peaks=False)
    
    # Shared X label
    axes[1].set_xlabel("Mass-to-Charge (m/z)", fontsize=14)
    plt.tight_layout()
    
    out_path = os.path.join(base_dir, 'output', 'plots', 'representative_spectra.png')
    plt.savefig(out_path, dpi=600)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
