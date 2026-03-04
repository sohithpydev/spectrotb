import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y: np.ndarray, lam: float = 100000, p: float = 0.0001, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares Smoothing for baseline correction.
    
    Args:
        y: Intensity array.
        lam: Smoothness parameter (lambda).
        p: Asymmetry parameter.
        niter: Number of iterations.
        
    Returns:
        Estimated baseline.
    """
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def _rolling_min_clsa(x: np.ndarray, f: np.ndarray, k: float) -> np.ndarray:
    """Helper for Continuous Line Segment Algorithm (CLSA) rolling minimum."""
    nx = len(x)
    k0 = k / 2.0
    
    # Compute block indices
    theta = np.floor((x - x[0]) / k).astype(int) + 1
    m = theta[-1] if nx > 0 else 0
    
    theta_pad = np.zeros(nx + 2, dtype=int)
    theta_pad[0] = 0
    theta_pad[-1] = m + 1
    theta_pad[1:-1] = theta
    
    s = pd.Series(f)
    g = s.groupby(theta).cummin().to_numpy()
    
    s_rev = pd.Series(f[::-1])
    h = s_rev.groupby(theta[::-1]).cummin().to_numpy()[::-1]
    
    i_l = np.searchsorted(x, x - k0, side='left')
    i_r = np.searchsorted(x, x + k0, side='right') - 1
    
    r_min = np.minimum(h[i_l], g[i_r])
    
    mask2 = theta_pad[i_l] == theta_pad[i_r + 1]
    r_min[mask2] = h[i_l[mask2]]
    
    mask1 = theta_pad[i_l + 1] == theta_pad[i_r + 2]
    r_min[mask1] = g[i_r[mask1]]
    
    return r_min

def baseline_clsa(x: np.ndarray, y: np.ndarray, k: float = 2.0, transform_mz: bool = True) -> np.ndarray:
    """
    Continuous Line Segment Algorithm (CLSA) baseline correction.
    Applies a top-hat filter (morphological opening) on unevenly spaced data.
    
    Args:
        x: m/z array (unevenly spaced independent variable).
        y: Intensity array.
        k: Structuring element (window) size. If transform_mz is True, this is in sqrt(m/z) units (default 2.0). 
           If False, this is in pure m/z units (usually needs to be much larger, e.g. 500-1000).
        transform_mz: Whether to apply the square-root transformation to the m/z axis, 
                      which normalizes TOF peak widths and prevents undercutting at higher masses.
        
    Returns:
        Estimated baseline.
    """
    if transform_mz:
        x_calc = np.sqrt(x)
    else:
        x_calc = x
        
    # Baseline = rolling_max(rolling_min(intensity))
    # rolling_max(f) = -rolling_min(-f)
    eroded = _rolling_min_clsa(x_calc, y, k)
    baseline = -_rolling_min_clsa(x_calc, -eroded, k)
    return baseline

def normalize_tic(intensity: np.ndarray) -> np.ndarray:
    """
    Total Ion Current (TIC) normalization.
    
    Args:
        intensity: Intensity array.
        
    Returns:
        Normalized intensity array.
    """
    total_ion_current = np.sum(intensity)
    if total_ion_current == 0:
        return intensity
    return intensity / total_ion_current

def smoothing_savgol(intensity: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay smoothing.
    
    Args:
        intensity: Intensity array.
        window_length: Length of the filter window (must be odd).
        polyorder: Order of the polynomial.
        
    Returns:
        Smoothed intensity array.
    """
    # Ensure window_length is odd and not larger than the array
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= len(intensity):
        window_length = len(intensity) - 1 if (len(intensity) - 1) % 2 == 1 else len(intensity) - 2
        
    if window_length < polyorder + 2:
        return intensity # Signal too short to smooth effectively with parameters
        
    return savgol_filter(intensity, window_length, polyorder)

def preprocess_spectrum(mz: np.ndarray, intensity: np.ndarray, baseline_method: str = 'als') -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: Smoothing -> Baseline Correction -> Subtraction -> Normalization.
    
    Args:
        mz: m/z array.
        intensity: raw intensity array.
        baseline_method: 'als' or 'clsa'.
        
    Returns:
        Tuple of (mz, processed_intensity).
    """
    # 1. Smoothing
    smoothed = smoothing_savgol(intensity)
    
    # 2. Baseline Correction
    if baseline_method == 'clsa':
        baseline = baseline_clsa(mz, smoothed) # Uses default transformed k=2.0
    else:
        baseline = baseline_als(smoothed)
    
    # 3. Baseline Subtraction
    corrected = smoothed - baseline
    
    # Clip negative values to 0 (optional but common in MS)
    corrected = np.maximum(corrected, 0)
    
    # 4. Normalization (TIC)
    normalized = normalize_tic(corrected)
    
    return mz, normalized

def preprocess_steps_visualization(mz: np.ndarray, intensity: np.ndarray, baseline_method: str = 'als') -> Dict[str, np.ndarray]:
    """
    Returns intermediate steps for visualization purposes.
    """
    smoothed = smoothing_savgol(intensity)
    if baseline_method == 'clsa':
        baseline = baseline_clsa(mz, smoothed)
    else:
        baseline = baseline_als(smoothed)
    corrected = np.maximum(smoothed - baseline, 0)
    normalized = normalize_tic(corrected)
    
    return {
        'raw': intensity,
        'smoothed': smoothed,
        'baseline': baseline,
        'corrected': corrected,
        'normalized': normalized
    }
