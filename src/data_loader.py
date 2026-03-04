import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def load_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a mass spectrum from a space-separated .txt file.
    
    Args:
        filepath: Path to the .txt file.
        
    Returns:
        Tuple of (mz, intensity) arrays.
    """
    try:
        # Assuming space-separated values, no header based on previous file view
        # Using pandas for robust reading of potentially varying whitespace
        df = pd.read_csv(filepath, sep='\s+', header=None, engine='python')
        
        # Check if we have at least 2 columns
        if df.shape[1] < 2:
            raise ValueError(f"File {filepath} has fewer than 2 columns.")
            
        mz = df.iloc[:, 0].values
        intensity = df.iloc[:, 1].values
        
        return mz, intensity
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([]), np.array([])

def load_dataset_files(data_dir: str) -> Dict[str, List[str]]:
    """
    Scans the data directory and returns a dictionary of class_name -> list of file paths.
    
    Args:
        data_dir: Root data directory containing subfolders (tb, ntm, external_tb, external_ntm).
        
    Returns:
        Dictionary mapping label to list of file paths.
    """
    dataset = {}
    subfolders = ['tb', 'ntm', 'external_tb', 'external_ntm']
    
    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
            
        files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith('.txt')
        ]
        dataset[folder] = files
        print(f"Found {len(files)} files in {folder}")
        
    return dataset
