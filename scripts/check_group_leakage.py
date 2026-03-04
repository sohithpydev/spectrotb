import os
import re
import numpy as np
import pandas as pd
from tabulate import tabulate

def get_group_id(filename):
    fname = os.path.basename(filename)

    # 1. remove spot / replicate suffix (_0_F9_1 etc.)
    # Note: original code used split("_")[0]. This might be too aggressive if valid names contain underscores
    # Looking at file list (2_0_C6_1.txt), 2 is presumably sample ID.
    parts = fname.split("_")
    # Heuristic: usually the suffix is like _0_F9_1.txt (last 3-4 parts)
    # But user provided specific code: fname.split("_")[0]
    # Let's trust the user's logic first, but be careful.
    fname = parts[0]

    # 2. remove parentheses content (instrument / volume info)
    fname = re.sub(r"\(.*?\)", "", fname)

    # 3. normalize separators
    fname = fname.replace("+", " ").replace("-", " ")

    # collapse whitespace
    fname = re.sub(r"\s+", " ", fname).strip()

    # -------------------------
    # RULE A: leading pure numeric ID (e.g. 30141, 21)
    # -------------------------
    m = re.match(r"^(\d{2,6})\b", fname, re.IGNORECASE)
    if m:
        return m.group(1)

    # -------------------------
    # RULE B: date-based sample (YYYYMMDD Sample X)
    # -------------------------
    m = re.match(r"^(\d{8}\s+TB\s+Sample\s+\d+)", fname, re.IGNORECASE)
    if m:
        return m.group(1)

    # -------------------------
    # RULE C: reject %-only or instrument-only names
    # -------------------------
    if re.fullmatch(r"[\d.%]+", fname):
        return f"UNKNOWN_{fname}"

    # -------------------------
    # RULE D: fallback – first meaningful phrase
    # -------------------------
    tokens = fname.split(" ")
    return " ".join(tokens[:4])

def main():
    print("--- Checking for DATA LEAKAGE based on Sample IDs ---")
    
    # Load previously processed metadata
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # pipeline/
        meta_path = os.path.join(base_dir, 'output', 'data', 'processed_dataset.csv')
        df = pd.read_csv(meta_path)
    except Exception as e:
        print(f"Could not load metadata: {e}")
        return

    print(f"Total Spectra: {len(df)}")
    
    # Assign Group IDs
    df['GroupID'] = df['Filename'].apply(get_group_id)
    
    # Count unique samples
    unique_groups = df['GroupID'].nunique()
    print(f"Unique Sample Groups: {unique_groups} (avg {len(df)/unique_groups:.1f} spectra per sample)")
    
    # 1. Check overlap between Internal and External folders
    # We need to know which file came from which folder. 
    # processed_dataset.csv just has filenames.
    # Re-scan directories to build a map.
    data_root = os.path.dirname(base_dir) # Data/
    
    file_to_folder = {}
    for folder in ['tb', 'ntm', 'external_tb', 'external_ntm']:
        folder_path = os.path.join(data_root, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".txt"):
                    file_to_folder[f] = folder
                    
    df['Folder'] = df['Filename'].map(file_to_folder)
    
    # Identify Internal vs External
    internal_mask = df['Folder'].isin(['tb', 'ntm'])
    external_mask = df['Folder'].isin(['external_tb', 'external_ntm'])
    
    internal_groups = set(df[internal_mask]['GroupID'])
    external_groups = set(df[external_mask]['GroupID'])
    
    intersection = internal_groups.intersection(external_groups)
    
    print("\n[Check 1] Overlap between Internal (Training) and External (Validation) Folders:")
    if len(intersection) > 0:
        print(f"⚠️  CRITICAL LEAKAGE FOUND: {len(intersection)} samples appear in both sets!")
        print(f"Overlapping IDs: {list(intersection)[:10]}...")
    else:
        print("✅  No overlap found. Internal and External folders contain distinct patients.")
        
    # 2. Check the random 80/20 split we did earlier
    # If we did a random split on FILES, we might have split Sample A (Spectrum 1) into Train and Sample A (Spectrum 2) into Test.
    # This is a classic leakage.
    
    # We need to simulate the split we did in 03_train_validate.py
    # But simpler: let's just recommend GroupKFold if we find multiple spectra per sample.
    
    counts = df['GroupID'].value_counts()
    multi_spectrum_samples = counts[counts > 1]
    
    print(f"\n[Check 2] Multi-spectrum Samples:")
    print(f"number of samples with >1 spectrum: {len(multi_spectrum_samples)} out of {unique_groups}")
    
    if len(multi_spectrum_samples) > 0:
        print("⚠️  WARNING: You have multiple spectra per sample.")
        print("If you used `train_test_split` or `StratifiedKFold` on the RAW FILES, you likely have data leakage.")
        print("(The model learned to recognize Patient X, not TB generally).")
        print("\nRecommendation: Switch to `GroupKFold` or `StratifiedGroupKFold`.")

if __name__ == "__main__":
    main()
