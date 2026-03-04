import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol
    from src.features import BIOMARKERS, PPM_TOLERANCE
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol
    from src.features import BIOMARKERS, PPM_TOLERANCE

def calculate_snr(mz, intensity, target_mass, local=False):
    """Calculates S/N for a specific target_mass"""
    delta = target_mass * PPM_TOLERANCE / 1e6
    lower_bound = target_mass - delta
    upper_bound = target_mass + delta
    
    mask_signal = (mz >= lower_bound) & (mz <= upper_bound)
    
    if local:
        bg_lower = target_mass - (delta * 5)
        bg_upper = target_mass + (delta * 5)
        mask_bg_all = (mz >= bg_lower) & (mz <= bg_upper)
        mask_noise = mask_bg_all & ~mask_signal
        
        if np.any(mask_signal) and np.any(mask_noise):
            peak_val = np.max(intensity[mask_signal]) - np.median(intensity[mask_bg_all])
            noise_val = np.std(intensity[mask_noise])
            return peak_val / noise_val if noise_val > 0 else 0
        return 0
    else:
        # Global noise
        global_noise_mask = (mz > 18000) & (mz < 19000)
        if np.any(mask_signal) and np.any(global_noise_mask):
            global_noise = np.std(intensity[global_noise_mask])
            global_median = np.median(intensity[global_noise_mask])
            peak_val = np.max(intensity[mask_signal]) - global_median
            return peak_val / global_noise if global_noise > 0 else 0
        return 0

def find_poor_snr_file(data_dir):
    """Finds a TB file with a poor S/N for ESAT-6_1 (raw global)."""
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    
    target_mass = BIOMARKERS['ESAT-6_1']
    
    print(f"Scanning files for poor S/N (focusing on ESAT-6_1)...")
    
    candidates = []
    
    for f in files[:200]: # Scan first 200
        try:
            mz, raw_int = load_spectrum(f)
        except:
            continue
            
        snr_raw = calculate_snr(mz, raw_int, target_mass, local=False) # global
        local_snr = calculate_snr(mz, raw_int, target_mass, local=True)
        
        # We want S/N low but visible (e.g., > 10, < 50 global)
        if 10 < snr_raw < 50 and local_snr > 0.5:
            candidates.append((f, snr_raw))
            
    # Sort by raw S/N (lowest first)
    candidates.sort(key=lambda x: x[1])
    
    if candidates:
        print(f"Found {len(candidates)} candidates. Selecting the one with lowest raw S/N: {candidates[0][1]:.2f}")
        return candidates[0][0]
    
    # Fallback to random if no perfect match
    import random
    return random.choice(files)

def plot_smoothing_comparison(mz, raw_int, smooth_int, filename, file_path_str):
    
    # Calculate global S/Ns for the title
    global_noise_mask = (mz > 18000) & (mz < 19000)
    
    esat6_mass = BIOMARKERS['ESAT-6_1']
    raw_snr = calculate_snr(mz, raw_int, esat6_mass, local=False)
    smooth_snr = calculate_snr(mz, smooth_int, esat6_mass, local=False)
    
    improvement = ((smooth_snr - raw_snr) / raw_snr) * 100 if raw_snr > 0 else 0
    
    # --- Start Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.15)
    
    # Plot 1: Standard full-view (raw vs smoothed)
    ax1 = axes[0]
    # To not make it too messy, let's plot raw with alpha
    ax1.plot(mz, raw_int, color='gray', alpha=0.5, label='Raw Data (Noisy)', linewidth=0.8)
    ax1.plot(mz, smooth_int, color='#d62728', alpha=0.8, label='Smoothed (Savitzky-Golay)', linewidth=1.2)
    
    # Highlight ESAT-6 region
    ax1.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5, label='ESAT-6 (~9813 Da)')
    
    ax1.set_xlim(2000, 20000)
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Overall Spectrum Comparison (m/z 2,000 - 20,000)\nFile: {file_path_str}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Zoomed in ESAT-6
    ax2 = axes[1]
    zoom_span = 80
    mask_zoom = (mz > esat6_mass - zoom_span) & (mz < esat6_mass + zoom_span)
    
    ax2.plot(mz[mask_zoom], raw_int[mask_zoom], color='gray', alpha=0.6, label='Raw Grass', marker='.', markersize=4)
    ax2.plot(mz[mask_zoom], smooth_int[mask_zoom], color='#d62728', linewidth=2.5, label='SG Polynomial Fit')
    ax2.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5)
    
    ax2.set_xlim(esat6_mass - zoom_span, esat6_mass + zoom_span)
    # calculate y bounds
    y_max = max(np.max(raw_int[mask_zoom]), np.max(smooth_int[mask_zoom]))
    y_min = min(np.min(raw_int[mask_zoom]), np.min(smooth_int[mask_zoom]))
    margin = (y_max - y_min) * 0.1
    ax2.set_ylim(y_min - margin, y_max + margin)
    
    ax2.set_xlabel('m/z', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title(f'Inset: ESAT-6 Peak (m/z ~{esat6_mass:.0f})\nRaw S/N: {raw_snr:.1f}  →  Smoothed S/N: {smooth_snr:.1f} (+{improvement:.1f}%)', fontsize=12, fontweight='bold', color='#2ca02c')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Add a global title
    plt.suptitle("Savitzky-Golay Smoothing preserves Structural Integrity while drastically removing high-frequency noise", fontsize=16, fontweight='bold', y=1.02)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {out_path}")
    
    return raw_snr, smooth_snr, improvement

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    file_path = find_poor_snr_file(data_dir)
    print(f"Selected File: {os.path.basename(file_path)}")
    
    mz, raw_int = load_spectrum(file_path)
    
    # Apply smoothing
    smooth_int = smoothing_savgol(raw_int, window_length=21, polyorder=3)
    
    # Plot side by side
    filename = 'overall_smoothing_comparison_with_inset.png'
    plot_smoothing_comparison(mz, raw_int, smooth_int, filename, os.path.basename(file_path))

if __name__ == "__main__":
    main()

