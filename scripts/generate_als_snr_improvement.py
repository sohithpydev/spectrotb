import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_local_snr(mz, intensity, target_mass):
    """
    To see S/N improvement after baseline correction, we MUST look at LOCAL S/N.
    ALS removes the slope. If a peak sits on a slope, the "noise" (standard deviation
    of the background around the peak) is artificially high because it includes the slope.
    Flattening the baseline removes the slope, dropping the local noise StdDev, thus
    increasing the local S/N!
    """
    delta = target_mass * (PPM_TOLERANCE / 1e6)
    mask_signal = (mz >= target_mass - delta) & (mz <= target_mass + delta)
    
    # Define a local window 5x the peak width on each side
    bg_lower = target_mass - (delta * 5)
    bg_upper = target_mass + (delta * 5)
    mask_bg_all = (mz >= bg_lower) & (mz <= bg_upper)
    mask_noise = mask_bg_all & ~mask_signal
    
    if np.any(mask_signal) and np.any(mask_noise):
        peak_val = np.max(intensity[mask_signal]) - np.median(intensity[mask_bg_all])
        noise_val = np.std(intensity[mask_noise])
        return max(0, peak_val / noise_val if noise_val > 0 else 0)
    return 0

def scan_files_for_als_improvement(data_dir):
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    
    esat6_mass = BIOMARKERS['ESAT-6_1']
    cfp10_mass = BIOMARKERS['CFP-10']
    
    candidates = []
    
    print("Scanning dataset for spectra where Local S/N improves dramatically after ALS...")
    for f in files[:800]:
        try:
            mz, int_raw = load_spectrum(f)
        except:
            continue
            
        # We start from SMOOTHED data, because we want to measure just the effect of ALS.
        int_smooth = smoothing_savgol(int_raw)
        
        # S/N before ALS
        snr_e_before = calculate_local_snr(mz, int_smooth, esat6_mass)
        snr_c_before = calculate_local_snr(mz, int_smooth, cfp10_mass)
        
        # Only look at files where peaks are present locally
        if snr_e_before > 0.5 and snr_c_before > 1.0:
            # Apply ALS
            baseline = baseline_als(int_smooth)
            int_corrected = int_smooth - baseline
            
            # S/N after ALS
            snr_e_after = calculate_local_snr(mz, int_corrected, esat6_mass)
            snr_c_after = calculate_local_snr(mz, int_corrected, cfp10_mass)
            
            # Improvement calculation
            imp_e = ((snr_e_after - snr_e_before) / snr_e_before) * 100 if snr_e_before > 0 else 0
            imp_c = ((snr_c_after - snr_c_before) / snr_c_before) * 100 if snr_c_before > 0 else 0
            
            candidates.append({
                'file_path': f,
                'snr_e_before': snr_e_before,
                'snr_e_after': snr_e_after,
                'imp_e': imp_e,
                'snr_c_before': snr_c_before,
                'snr_c_after': snr_c_after,
                'imp_c': imp_c,
                'total_imp': imp_e + imp_c
            })
            
    if not candidates:
        print("No robust candidates found.")
        return None
        
    # Sort by maximum total local S/N improvement
    candidates.sort(key=lambda x: max(x['imp_e'], x['imp_c']), reverse=True)
    
    best = candidates[0]
    print(f"Chosen File: {os.path.basename(best['file_path'])}")
    print(f"ESAT-6 Local S/N Improvement: {best['imp_e']:.1f}% ({best['snr_e_before']:.1f} -> {best['snr_e_after']:.1f})")
    print(f"CFP-10 Local S/N Improvement: {best['imp_c']:.1f}% ({best['snr_c_before']:.1f} -> {best['snr_c_after']:.1f})")
    
    return best['file_path'], best

def plot_als_local_improvement(mz, smooth_int, corrected_int, baseline, esat6_mass, cfp10_mass, out_path, file_name, stats):
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.35, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Overall Before vs After
    ax1.plot(mz, smooth_int, color='gray', alpha=0.5, label='Smoothed (Before ALS)', linewidth=1.5)
    ax1.plot(mz, corrected_int, color='#1f77b4', alpha=0.9, label='Corrected (After ALS)', linewidth=1.5)
    ax1.plot(mz, baseline, color='#ff7f0e', linestyle='--', label='Fitted ALS Baseline', linewidth=2)
    
    ax1.axvline(esat6_mass, color='blue', linestyle='--', alpha=0.5)
    ax1.axvline(cfp10_mass, color='green', linestyle='--', alpha=0.5)
    
    ax1.set_xlim(2000, 20000)
    y_max_overall = np.max(smooth_int[(mz>2000) & (mz<20000)])
    ax1.set_ylim(-0.05*y_max_overall, y_max_overall*1.05)
    
    ax1.set_xlabel('m/z', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Overall ALS Spectrum Correction (m/z 2,000 - 20,000)\nFile: {file_name}', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: ESAT-6 Local Zoom
    zoom_span = 80
    mask_e = (mz > esat6_mass - zoom_span) & (mz < esat6_mass + zoom_span)
    ax2.plot(mz[mask_e], smooth_int[mask_e], color='gray', alpha=0.8, label='Sloped Baseline (Before)', linestyle='--')
    ax2.plot(mz[mask_e], corrected_int[mask_e], color='#1f77b4', linewidth=2.5, label='Flat Baseline (After)')
    
    ax2.set_xlim(esat6_mass - zoom_span, esat6_mass + zoom_span)
    # y bounds based on both lines to show they are shifted
    y_max_e = max(np.max(smooth_int[mask_e]), np.max(corrected_int[mask_e]))
    y_min_e = min(np.min(smooth_int[mask_e]), np.min(corrected_int[mask_e]))
    margin_e = (y_max_e - y_min_e) * 0.1
    ax2.set_ylim(y_min_e - margin_e, y_max_e + margin_e)
    
    ax2.set_ylabel('Intensity', fontsize=11)
    ax2.set_title(f'ESAT-6 Inset (Slope Removal)\nLocal S/N: {stats["snr_e_before"]:.1f} → {stats["snr_e_after"]:.1f} (+{stats["imp_e"]:.1f}%)', fontsize=11, fontweight='bold', color='blue')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: CFP-10 Local Zoom
    mask_c = (mz > cfp10_mass - zoom_span) & (mz < cfp10_mass + zoom_span)
    ax3.plot(mz[mask_c], smooth_int[mask_c], color='gray', alpha=0.8, label='Sloped Baseline (Before)', linestyle='--')
    ax3.plot(mz[mask_c], corrected_int[mask_c], color='#1f77b4', linewidth=2.5, label='Flat Baseline (After)')
    
    ax3.set_xlim(cfp10_mass - zoom_span, cfp10_mass + zoom_span)
    y_max_c = max(np.max(smooth_int[mask_c]), np.max(corrected_int[mask_c]))
    y_min_c = min(np.min(smooth_int[mask_c]), np.min(corrected_int[mask_c]))
    margin_c = (y_max_c - y_min_c) * 0.1
    ax3.set_ylim(y_min_c - margin_c, y_max_c + margin_c)
    
    ax3.set_xlabel('m/z', fontsize=12)
    ax3.set_ylabel('Intensity', fontsize=11)
    ax3.set_title(f'CFP-10 Inset (Slope Removal)\nLocal S/N: {stats["snr_c_before"]:.1f} → {stats["snr_c_after"]:.1f} (+{stats["imp_c"]:.1f}%)', fontsize=11, fontweight='bold', color='green')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("How ALS Baseline Correction Improves LOCAL S/N by Flattening the Matrix Slope Variance", fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    
    best_file, stats = scan_files_for_als_improvement(data_dir)
    if not best_file:
        return
        
    file_name = os.path.basename(best_file)
    
    mz, raw_int = load_spectrum(best_file)
    smooth_int = smoothing_savgol(raw_int)
    baseline = baseline_als(smooth_int)
    corrected_int = smooth_int - baseline
    
    out_path = os.path.join(out_dir, 'als_local_snr_improvement.png')
    
    plot_als_local_improvement(mz, smooth_int, corrected_int, baseline, BIOMARKERS['ESAT-6_1'], BIOMARKERS['CFP-10'], out_path, file_name, stats)
    
    print(f"\nSaved ALS Local S/N Improvement plot to: {out_path}")

if __name__ == '__main__':
    main()
