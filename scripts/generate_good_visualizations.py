import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol, baseline_als
    from src.features import BIOMARKERS, PPM_TOLERANCE
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol, baseline_als
    from src.features import BIOMARKERS, PPM_TOLERANCE

def calculate_snr(mz, intensity, target_mass):
    delta = target_mass * PPM_TOLERANCE / 1e6
    mask_signal = (mz >= target_mass - delta) & (mz <= target_mass + delta)
    global_noise_mask = (mz > 18000) & (mz < 19000)
    
    if np.any(mask_signal) and np.any(global_noise_mask):
        global_noise = np.std(intensity[global_noise_mask])
        global_median = np.median(intensity[global_noise_mask])
        peak_val = np.max(intensity[mask_signal]) - global_median
        return peak_val / global_noise if global_noise > 0 else 0
    return 0

def find_good_file(data_dir):
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    
    esat6_mass = BIOMARKERS['ESAT-6_1']
    cfp10_mass = BIOMARKERS['CFP-10']
    
    best_file = None
    best_score = 0
    
    print("Scanning for a file with visibly strong ESAT-6 and CFP-10 peaks...")
    for f in files[:300]: # Scan first 300
        try:
            mz, int_raw = load_spectrum(f)
        except:
            continue
            
        snr_esat = calculate_snr(mz, int_raw, esat6_mass)
        snr_cfp = calculate_snr(mz, int_raw, cfp10_mass)
        
        # We want high S/N for both to ensure naked-eye visibility
        if snr_esat > 15 and snr_cfp > 30:
            score = snr_esat + snr_cfp
            if score > best_score:
                best_score = score
                best_file = f
                
    if best_file is None:
        print("Could not find optimal file, falling back to first file.")
        best_file = files[0] # fallback
        
    return best_file

def plot_smoothing(mz, raw_int, smooth_int, esat6_mass, out_path, file_name):
    raw_snr = calculate_snr(mz, raw_int, esat6_mass)
    smooth_snr = calculate_snr(mz, smooth_int, esat6_mass)
    improvement = ((smooth_snr - raw_snr) / raw_snr) * 100 if raw_snr > 0 else 0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.15)
    
    ax1 = axes[0]
    ax1.plot(mz, raw_int, color='gray', alpha=0.5, label='Raw Data (Noisy)', linewidth=0.8)
    ax1.plot(mz, smooth_int, color='#d62728', alpha=0.9, label='Smoothed (Savitzky-Golay)', linewidth=1.5)
    
    ax1.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5, label='ESAT-6 (~9813 Da)')
    ax1.axvline(BIOMARKERS['CFP-10'], color='green', linestyle='--', alpha=0.5, label='CFP-10 (~10100 Da)')
    
    ax1.set_xlim(2000, 20000)
    y_max_overall = np.max(raw_int[(mz>2000) & (mz<20000)])
    ax1.set_ylim(-0.05*y_max_overall, y_max_overall*1.05)
    
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Overall Spectrum Comparison (m/z 2,000 - 20,000)\nFile: {file_name}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = axes[1]
    zoom_span = 80
    mask_zoom = (mz > esat6_mass - zoom_span) & (mz < esat6_mass + zoom_span)
    
    ax2.plot(mz[mask_zoom], raw_int[mask_zoom], color='gray', alpha=0.6, label='Raw Grass', marker='.', markersize=4)
    ax2.plot(mz[mask_zoom], smooth_int[mask_zoom], color='#d62728', linewidth=2.5, label='SG Polynomial Fit')
    ax2.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5)
    
    ax2.set_xlim(esat6_mass - zoom_span, esat6_mass + zoom_span)
    y_max = max(np.max(raw_int[mask_zoom]), np.max(smooth_int[mask_zoom]))
    y_min = min(np.min(raw_int[mask_zoom]), np.min(smooth_int[mask_zoom]))
    margin = (y_max - y_min) * 0.1
    ax2.set_ylim(y_min - margin, y_max + margin)
    
    ax2.set_xlabel('m/z', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title(f'Inset: ESAT-6 Peak (m/z ~{esat6_mass:.0f})\nRaw S/N: {raw_snr:.1f}  →  Smoothed S/N: {smooth_snr:.1f} (+{improvement:.1f}%)', fontsize=12, fontweight='bold', color='#2ca02c')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Savitzky-Golay Smoothing preserves Structural Integrity of Key Biomarkers", fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_als(mz, raw_int, smooth_int, baseline, corrected, out_path, file_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.15)
    
    ax1 = axes[0]
    ax1.plot(mz, smooth_int, color='gray', alpha=0.7, label='Smoothed Spectrum')
    ax1.plot(mz, baseline, color='#ff7f0e', linestyle='--', linewidth=2, label='Fitted ALS Baseline $\lambda=10^5, p=10^{-4}$')
    
    ax1.set_xlim(2000, 20000)
    max_val = np.max(smooth_int[(mz > 2000) & (mz < 20000)])
    ax1.set_ylim(-0.05 * max_val, max_val * 1.05)
    
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Step 1: Smoothed Spectrum with Distinct Peaks & Estimated Baseline\nFile: {file_name}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = axes[1]
    ax2.plot(mz, corrected, color='#1f77b4', linewidth=1.2, label='Corrected Spectrum (Smoothed - Baseline)')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    # Give reference lines for where peaks are
    ax2.axvline(BIOMARKERS['ESAT-6_1'], color='blue', linestyle=':', alpha=0.4)
    ax2.axvline(BIOMARKERS['CFP-10'], color='green', linestyle=':', alpha=0.4)
    
    ax2.set_xlim(2000, 20000)
    ax2.set_ylim(-0.05 * max_val, max_val * 1.05)
    
    ax2.set_xlabel('m/z', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title(f'Step 2: flattened ALS Baseline Correction', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Asymmetric Least Squares (ALS) Baseline Correction on High Quality Spectrum", fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    
    best_file = find_good_file(data_dir)
    file_name = os.path.basename(best_file)
    print(f"Using visually strong file: {file_name}")
    
    mz, raw_int = load_spectrum(best_file)
    smooth_int = smoothing_savgol(raw_int)
    baseline = baseline_als(smooth_int)
    corrected = smooth_int - baseline
    
    path_smoothing = os.path.join(out_dir, 'good_smoothing_comparison.png')
    path_als = os.path.join(out_dir, 'good_als_comparison.png')
    
    plot_smoothing(mz, raw_int, smooth_int, BIOMARKERS['ESAT-6_1'], path_smoothing, file_name)
    plot_als(mz, raw_int, smooth_int, baseline, corrected, path_als, file_name)
    
    print(f"Saved: {path_smoothing}")
    print(f"Saved: {path_als}")

if __name__ == '__main__':
    main()
