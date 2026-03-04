
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import random

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_loader import load_spectrum
    from src.preprocessing import smoothing_savgol

def find_low_intensity_esat6(data_dir):
    """Finds a TB file with a distinct but LOW intensity ESAT-6 peak (approx 9813)."""
    tb_dir = os.path.join(data_dir, 'tb')
    files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if f.endswith('.txt')]
    random.shuffle(files)
    
    target_mass = 9813
    window = 10
    
    best_file = None
    best_score = float('inf') # We want low intensity, but > noise
    
    print(f"Scanning files for low-intensity ESAT-6...")
    
    candidates = []
    
    for f in files[:200]:
        mz, inte = load_spectrum(f)
        
        # Check if peak exists
        mask = (mz > target_mass - window) & (mz < target_mass + window)
        if not np.any(mask): continue
        
        peak_intensity = np.max(inte[mask])
        
        # Define "Low Intensity": Visible but not huge. 
        # Noise floor is often ~0-100 after baseline, but absolute raw might be higher.
        # Let's look for peaks in 500-3000 range.
        if 500 < peak_intensity < 3000:
            candidates.append((f, peak_intensity))
            
    # Sort by intensity (lowest first)
    candidates.sort(key=lambda x: x[1])
    
    if candidates:
        return candidates[0][0] # Return the lowest valid one
    return None

def main():
    print("--- Visualizing Smoothing on ESAT-6 ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.dirname(current_dir)
    data_dir = os.path.dirname(pipeline_dir)
    
    file_path = find_low_intensity_esat6(data_dir)
    if not file_path:
        print("No suitable file found.")
        # Fallback to random if no perfect candidate
        return

    print(f"File: {os.path.basename(file_path)}")
    mz, intensity_raw = load_spectrum(file_path)
    
    # Smooth
    # Default parameters: window_length=21, polyorder=3
    intensity_smooth = smoothing_savgol(intensity_raw, window_length=21, polyorder=3)
    
    # Plot Zoomed Region around ESAT-6 (9813)
    center = 9813
    span = 100
    mask = (mz > center - span) & (mz < center + span)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Scatter for Raw to show noise
    plt.plot(mz[mask], intensity_raw[mask], color='gray', alpha=0.5, label='Raw Data (Noisy)', linewidth=1)
    
    # Plot Smooth line
    plt.plot(mz[mask], intensity_smooth[mask], color='red', linewidth=2, label='Smoothed (Savitzky-Golay)')
    
    # Mark the peak position
    plt.axvline(center, color='blue', linestyle=':', alpha=0.5, label='ESAT-6 (Theoretical)')
    
    plt.title(f"Smoothing Effect on Low Intensity ESAT-6 Peak\nFile: {os.path.basename(file_path)}", fontsize=11)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = os.path.join(pipeline_dir, 'output', 'plots')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    out_path = os.path.join(out_dir, 'preprocess_savgol_zoom.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
