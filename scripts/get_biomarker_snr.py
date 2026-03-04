import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.data_loader import load_spectrum
from src.preprocessing import baseline_als, normalize_tic, smoothing_savgol
from src.features import BIOMARKERS, PPM_TOLERANCE

tb_path = "../../tb/60uL TB (pH3 buffer + 5nm acid ND), 1 uL SA 45%offset, 50%laser Jeff LP_2.1_0_C8_1.txt"
if not os.path.exists(tb_path):
    print("File not found:", tb_path)
    sys.exit(1)

mz, raw_int = load_spectrum(tb_path)

# Apply Savitzky-Golay smoothing
smoothed_int = smoothing_savgol(raw_int, window_length=21, polyorder=3)

print(f"{'Biomarker':<15} | {'m/z':<8} | {'Raw S/N':<10} | {'Smooth S/N':<10} | {'Improvement'}")
print("-" * 65)

for name, target_mass in BIOMARKERS.items():
    # Calculate local noise region
    # Peak window = target_mass +/- (target_mass * 1000 ppm)
    delta = target_mass * PPM_TOLERANCE / 1e6
    lower_bound = target_mass - delta
    upper_bound = target_mass + delta
    
    # Noise window = 5*delta around peak, EXCLUDING the peak window itself
    bg_lower = target_mass - (delta * 5)
    bg_upper = target_mass + (delta * 5)
    
    mask_signal = (mz >= lower_bound) & (mz <= upper_bound)
    mask_bg_all = (mz >= bg_lower) & (mz <= bg_upper)
    mask_noise = mask_bg_all & ~mask_signal
    
    if np.any(mask_signal) and np.any(mask_noise):
        # Calculate raw SNR
        peak_raw = np.max(raw_int[mask_signal]) - np.median(raw_int[mask_bg_all])
        raw_noise = np.std(raw_int[mask_noise])
        snr_raw = peak_raw / raw_noise if raw_noise > 0 else 0
        
        # Calculate smoothed SNR
        peak_smooth = np.max(smoothed_int[mask_signal]) - np.median(smoothed_int[mask_bg_all])
        smooth_noise = np.std(smoothed_int[mask_noise])
        snr_smooth = peak_smooth / smooth_noise if smooth_noise > 0 else 0
        
        imp = ((snr_smooth - snr_raw) / snr_raw * 100) if snr_raw > 0 else 0
        
        print(f"{name:<15} | {target_mass:<8} | {snr_raw:<10.2f} | {snr_smooth:<10.2f} | {imp:+.1f}%")
    else:
        print(f"{name:<15} | {target_mass:<8} | {'N/A':<10} | {'N/A':<10} | N/A")

print("-" * 65)
