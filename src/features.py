import numpy as np
from typing import List, Dict

# Biomarkers defintions
BIOMARKERS = {
    'CFP-10': 10100,
    'CFP-10*': 10660,
    'ESAT-6_1': 9813,
    'ESAT-6_2': 9786,
    'ESAT-6*_1': 7931,
    'ESAT-6*_2': 7974,
    # Doubly Charged (approximate m/z = mass / 2)
    'CFP-10_z2': 5050,
    'CFP-10*_z2': 5330,
    'ESAT-6_1_z2': 4906.5,
    'ESAT-6_2_z2': 4893,
    'ESAT-6*_1_z2': 3965.5,
    'ESAT-6*_2_z2': 3987
}

PPM_TOLERANCE = 1000

def find_peak_in_window(mz: np.ndarray, intensity: np.ndarray, target_mass: float, ppm: float) -> float:
    """
    Finds the max intensity in window minus the local background (estimated by median).
    """
    delta = target_mass * ppm / 1e6
    # Search window
    lower_bound = target_mass - delta
    upper_bound = target_mass + delta
    
    # Background window (slightly wider: 3x tolerance)
    bg_lower = target_mass - (delta * 5)
    bg_upper = target_mass + (delta * 5)
    
    # Indices
    mask_signal = (mz >= lower_bound) & (mz <= upper_bound)
    mask_bg = (mz >= bg_lower) & (mz <= bg_upper)
    
    if np.any(mask_signal):
        peak_intensity = np.max(intensity[mask_signal])
        
        # Estimate background from the wider local area (excluding the peak itself if possible, but median is robust)
        if np.any(mask_bg):
            background = np.median(intensity[mask_bg])
        else:
            background = 0.0
            
        # Return height relative to local background
        return max(0.0, peak_intensity - background)
    else:
        return 0.0

def extract_features(mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    Extracts intensity features for all defined biomarkers.
    """
    features = []
    # Ensure defined order
    feature_names = get_feature_names()
    
    for key in feature_names:
        mass = BIOMARKERS[key]
        val = find_peak_in_window(mz, intensity, mass, PPM_TOLERANCE)
        features.append(val)
        
    return np.array(features)

def get_feature_names() -> List[str]:
    return [
        'CFP-10', 'CFP-10*', 'ESAT-6_1', 'ESAT-6_2', 'ESAT-6*_1', 'ESAT-6*_2',
        'CFP-10_z2', 'CFP-10*_z2', 'ESAT-6_1_z2', 'ESAT-6_2_z2', 'ESAT-6*_1_z2', 'ESAT-6*_2_z2'
    ]
