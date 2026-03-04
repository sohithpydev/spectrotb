# scripts/01_process_data.py
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_dataset_files, load_spectrum
from src.preprocessing import preprocess_spectrum
from src.features import extract_features, get_feature_names

def main():
    print("--- Step 1: Data Processing ---")
    
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # pipeline/
    data_dir = os.path.dirname(base_dir) # Data/
    
    output_dir = os.path.join(base_dir, 'output', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Data Root: {data_dir}")
    print(f"Output Dir: {output_dir}")
    
    # 2. Load File Lists
    dataset_files = load_dataset_files(data_dir)
    
    # Combine all data
    # 0 = NTM, 1 = TB
    label_map = {
        'ntm': 0, 'tb': 1,
        'external_ntm': 0, 'external_tb': 1
    }
    
    all_files = []
    all_labels = []
    all_filenames = []
    
    for key, label in label_map.items():
        files = dataset_files.get(key, [])
        print(f"Found {len(files)} files for {key} (Label {label})")
        
        for f in files:
            all_files.append(f)
            all_labels.append(label)
            all_filenames.append(os.path.basename(f))
            
    print(f"Total Files: {len(all_files)}")
    
    # 3. Process Loops
    X = []
    y = []
    valid_filenames = []
    
    print("Processing files...")
    for i, filepath in enumerate(tqdm(all_files)):
        try:
            # Load
            mz, intensity = load_spectrum(filepath)
            if mz is None or len(mz) == 0:
                continue
                
            # Preprocess
            mz_proc, int_proc = preprocess_spectrum(mz, intensity)
            
            # Extract Features
            feats = extract_features(mz_proc, int_proc)
            
            X.append(feats)
            y.append(all_labels[i])
            valid_filenames.append(all_filenames[i])
            
        except Exception as e:
            # print(f"Error {filepath}: {e}")
            pass
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Final Shape: {X.shape}")
    
    # 4. Save
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    # Save filenames and feature names for reference
    pd.DataFrame({'filename': valid_filenames, 'label': y}).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    feature_names = get_feature_names()
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write("\n".join(feature_names))
        
    # Also save as one big CSV as requested ("save the preprocessed dataset")
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = y
    df['Filename'] = valid_filenames
    df.to_csv(os.path.join(output_dir, 'processed_dataset.csv'), index=False)
    
    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()
