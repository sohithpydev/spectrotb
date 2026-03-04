import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.data_loader import load_spectrum
from src.preprocessing import baseline_als, normalize_tic, smoothing_savgol

tb_path = "../../tb/60uL TB (pH3 buffer + 5nm acid ND), 1 uL SA 45%offset, 50%laser Jeff LP_2.1_0_C8_1.txt"
if not os.path.exists(tb_path):
    print("File not found:", tb_path)
    sys.exit(1)

mz, raw_int = load_spectrum(tb_path)

smoothed_int = smoothing_savgol(raw_int, window_length=21, polyorder=3)

idx_9813 = np.argmin(np.abs(mz - 9813))

noise_mask = (mz > 2500) & (mz < 3000)
if np.any(noise_mask):
    raw_noise = np.std(raw_int[noise_mask])
    smooth_noise = np.std(smoothed_int[noise_mask])
    
    peak_raw = raw_int[idx_9813] - np.median(raw_int[noise_mask])
    peak_smooth = smoothed_int[idx_9813] - np.median(smoothed_int[noise_mask])
    
    snr_raw = peak_raw / raw_noise
    snr_smooth = peak_smooth / smooth_noise
    
    print(f"S/N Ratio (Raw): {snr_raw:.2f}")
    print(f"S/N Ratio (Smoothed): {snr_smooth:.2f}")
    print(f"S/N Improvement: {((snr_smooth - snr_raw)/snr_raw)*100:.1f}%")
else:
    print("No mz in 2500-3000 noise region.")
