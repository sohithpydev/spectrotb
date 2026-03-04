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

def find_extreme_cfp_file(data_dir):
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    
    esat6_mass = BIOMARKERS['ESAT-6_1']
    cfp10_mass = BIOMARKERS['CFP-10']
    
    candidates = []
    
    print("Scanning for a file with poor baseline, clear ESAT-6, and highly prioritized CFP-10...")
    for f in files[:800]: # Scan first 800 cases
        try:
            mz, int_raw = load_spectrum(f)
        except:
            continue
            
        snr_esat = calculate_snr(mz, int_raw, esat6_mass)
        snr_cfp = calculate_snr(mz, int_raw, cfp10_mass)
        
        # Must have visible peaks, prioritizing CFP-10 heavily
        if snr_esat > 15 and snr_cfp > 40:
            smooth_int = smoothing_savgol(int_raw)
            smooth_snr_esat = calculate_snr(mz, smooth_int, esat6_mass)
            smooth_snr_cfp = calculate_snr(mz, smooth_int, cfp10_mass)
            
            improvement_esat = ((smooth_snr_esat - snr_esat) / snr_esat) * 100 if snr_esat > 0 else 0
            improvement_cfp = ((smooth_snr_cfp - snr_cfp) / snr_cfp) * 100 if snr_cfp > 0 else 0
            
            baseline = baseline_als(smooth_int)
            
            mask_low_mass = (mz > 2000) & (mz < 6000)
            mask_high_mass = (mz > 18000) & (mz < 19000)
            
            if np.any(mask_low_mass) and np.any(mask_high_mass):
                baseline_severity = np.max(baseline[mask_low_mass]) - np.min(baseline[mask_high_mass])
            else:
                baseline_severity = 0
            
            candidates.append({
                'file_path': f,
                'improvement_esat': improvement_esat,
                'improvement_cfp': improvement_cfp,
                'cfp_snr': snr_cfp,
                'baseline_severity': baseline_severity,
                'raw_snr_esat': snr_esat
            })
            
    if not candidates:
        print("No matches with strict criteria. Reverting to fallback.")
        return files[0]
        
    # Get 75th percentile of baseline_severity to define "poor baseline"
    baselines = [c['baseline_severity'] for c in candidates]
    p75 = np.percentile(baselines, 75)
    
    poor_baseline_candidates = [c for c in candidates if c['baseline_severity'] >= p75]
    
    if not poor_baseline_candidates:
        poor_baseline_candidates = candidates
        
    # Sort primarily by how good CFP-10 is, then by improvement
    poor_baseline_candidates.sort(key=lambda x: (x['improvement_cfp'] + x['improvement_esat']) * x['cfp_snr'], reverse=True)
    
    best = poor_baseline_candidates[0]
    print(f"Chosen File: {os.path.basename(best['file_path'])}")
    print(f"CFP-10 SNR: {best['cfp_snr']:.1f}")
    print(f"ESAT-6 Improvement: {best['improvement_esat']:.1f}%")
    print(f"CFP-10 Improvement: {best['improvement_cfp']:.1f}%")
    print(f"Baseline Severity: {best['baseline_severity']}")
    
    return best['file_path']

def plot_smoothing_dual_inset(mz, raw_int, smooth_int, esat6_mass, cfp10_mass, out_path, file_name):
    raw_snr_esat = calculate_snr(mz, raw_int, esat6_mass)
    smooth_snr_esat = calculate_snr(mz, smooth_int, esat6_mass)
    imp_e = ((smooth_snr_esat - raw_snr_esat) / raw_snr_esat) * 100 if raw_snr_esat > 0 else 0
    
    raw_snr_cfp = calculate_snr(mz, raw_int, cfp10_mass)
    smooth_snr_cfp = calculate_snr(mz, smooth_int, cfp10_mass)
    imp_c = ((smooth_snr_cfp - raw_snr_cfp) / raw_snr_cfp) * 100 if raw_snr_cfp > 0 else 0
    
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.35, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Overall
    ax1.plot(mz, raw_int, color='gray', alpha=0.5, label='Raw Data (Noisy)', linewidth=0.8)
    ax1.plot(mz, smooth_int, color='#d62728', alpha=0.9, label='Smoothed (Savitzky-Golay)', linewidth=1.5)
    
    ax1.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5, label='ESAT-6 (~9813 Da)')
    ax1.axvline(cfp10_mass, color='green', linestyle='--', alpha=0.5, label='CFP-10 (~10100 Da)')
    
    ax1.set_xlim(2000, 20000)
    y_max_overall = np.max(raw_int[(mz>2000) & (mz<20000)])
    ax1.set_ylim(-0.05*y_max_overall, y_max_overall*1.05)
    
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Overall Spectrum Comparison (m/z 2,000 - 20,000)\nFile: {file_name}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: ESAT-6
    zoom_span = 80
    mask_e = (mz > esat6_mass - zoom_span) & (mz < esat6_mass + zoom_span)
    ax2.plot(mz[mask_e], raw_int[mask_e], color='gray', alpha=0.6, label='Raw Grass', marker='.', markersize=4)
    ax2.plot(mz[mask_e], smooth_int[mask_e], color='#d62728', linewidth=2.5, label='SG Fit')
    ax2.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5)
    
    ax2.set_xlim(esat6_mass - zoom_span, esat6_mass + zoom_span)
    y_max_e = max(np.max(raw_int[mask_e]), np.max(smooth_int[mask_e]))
    y_min_e = min(np.min(raw_int[mask_e]), np.min(smooth_int[mask_e]))
    margin_e = (y_max_e - y_min_e) * 0.1
    ax2.set_ylim(y_min_e - margin_e, y_max_e + margin_e)
    
    ax2.set_ylabel('Intensity', fontsize=11)
    ax2.set_title(f'ESAT-6 Inset (m/z ~{esat6_mass:.0f})\nRaw S/N: {raw_snr_esat:.1f} → Smoothed: {smooth_snr_esat:.1f} (+{imp_e:.1f}%)', fontsize=11, fontweight='bold', color='blue')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: CFP-10
    mask_c = (mz > cfp10_mass - zoom_span) & (mz < cfp10_mass + zoom_span)
    ax3.plot(mz[mask_c], raw_int[mask_c], color='gray', alpha=0.6, label='Raw Grass', marker='.', markersize=4)
    ax3.plot(mz[mask_c], smooth_int[mask_c], color='#d62728', linewidth=2.5, label='SG Fit')
    ax3.axvline(cfp10_mass, color='green', linestyle='--', alpha=0.5)
    
    ax3.set_xlim(cfp10_mass - zoom_span, cfp10_mass + zoom_span)
    y_max_c = max(np.max(raw_int[mask_c]), np.max(smooth_int[mask_c]))
    y_min_c = min(np.min(raw_int[mask_c]), np.min(smooth_int[mask_c]))
    margin_c = (y_max_c - y_min_c) * 0.1
    ax3.set_ylim(y_min_c - margin_c, y_max_c + margin_c)
    
    ax3.set_xlabel('m/z', fontsize=12)
    ax3.set_ylabel('Intensity', fontsize=11)
    ax3.set_title(f'CFP-10 Inset (m/z ~{cfp10_mass:.0f})\nRaw S/N: {raw_snr_cfp:.1f} → Smoothed: {smooth_snr_cfp:.1f} (+{imp_c:.1f}%)', fontsize=11, fontweight='bold', color='green')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Savitzky-Golay Smoothing preserves Structural Integrity & Max S/N Improvement\n(Dual Biomarker Focus)", fontsize=16, fontweight='bold', y=0.98)
    
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
    ax1.set_title(f'Step 1: Smoothed Spectrum (Poor Baseline Drift) & Estimated Baseline\nFile: {file_name}', fontsize=12)
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
    
    plt.suptitle("Asymmetric Least Squares (ALS) Baseline Correction on Extreme Drift Spectrum", fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    
    best_file = find_extreme_cfp_file(data_dir)
    file_name = os.path.basename(best_file)
    print(f"Using extreme-case file prioritized for CFP-10: {file_name}")
    
    mz, raw_int = load_spectrum(best_file)
    smooth_int = smoothing_savgol(raw_int)
    baseline = baseline_als(smooth_int)
    corrected = smooth_int - baseline
    
    path_smoothing = os.path.join(out_dir, 'max_imp_cfp10_dual_inset_smoothing.png')
    path_als = os.path.join(out_dir, 'max_imp_cfp10_poor_baseline_als.png')
    
    plot_smoothing_dual_inset(mz, raw_int, smooth_int, BIOMARKERS['ESAT-6_1'], BIOMARKERS['CFP-10'], path_smoothing, file_name)
    plot_als(mz, raw_int, smooth_int, baseline, corrected, path_als, file_name)
    
    print(f"Saved: {path_smoothing}")
    print(f"Saved: {path_als}")

if __name__ == '__main__':
    main()
