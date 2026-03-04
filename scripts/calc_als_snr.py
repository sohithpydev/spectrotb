import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol, baseline_als
    from src.features import BIOMARKERS, PPM_TOLERANCE
except ImportError:
    pass

def calculate_snr(mz, intensity, target_mass):
    delta = target_mass * PPM_TOLERANCE / 1e6
    mask_signal = (mz >= target_mass - delta) & (mz <= target_mass + delta)
    global_noise_mask = (mz > 18000) & (mz < 19000)
    
    if np.any(mask_signal) and np.any(global_noise_mask):
        global_noise = np.std(intensity[global_noise_mask])
        global_median = np.median(intensity[global_noise_mask])
        peak_val = np.max(intensity[mask_signal]) - global_median
        return max(0, peak_val / global_noise if global_noise > 0 else 0)
    return 0

def main():
    data_dir = '/Volumes/My Passport/NDHU/Data'
    file_name = '20230906-NONCRUSHED-MOF(COM)(20uL)+TB 32961(20uL)+(100uL)B+-SA-60%INTENSITY+45%OFFSET_0_C4_1.txt'
    file_path = os.path.join(data_dir, 'tb', file_name)

    mz, raw_int = load_spectrum(file_path)
    smooth_int = smoothing_savgol(raw_int)
    baseline = baseline_als(smooth_int)
    corrected_int = smooth_int - baseline

    esat6_mass = BIOMARKERS['ESAT-6_1']
    cfp10_mass = BIOMARKERS['CFP-10']

    # S/N calculations
    snr_esat_before = calculate_snr(mz, smooth_int, esat6_mass)
    snr_esat_after = calculate_snr(mz, corrected_int, esat6_mass)

    snr_cfp_before = calculate_snr(mz, smooth_int, cfp10_mass)
    snr_cfp_after = calculate_snr(mz, corrected_int, cfp10_mass)

    print("=== ESAT-6 (9813 Da) ===")
    print(f"Before ALS: {snr_esat_before:.2f}")
    print(f"After ALS:  {snr_esat_after:.2f}")
    
    print("\n=== CFP-10 (10100 Da) ===")
    print(f"Before ALS: {snr_cfp_before:.2f}")
    print(f"After ALS:  {snr_cfp_after:.2f}")

if __name__ == '__main__':
    main()
