import os
import hashlib
from collections import defaultdict

def hash_file(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    root_dir = "."
    hashes = defaultdict(list)
    
    print("Scanning for duplicate files...")
    count = 0 
    for dirpath, _, filenames in os.walk(root_dir):
        if "pipeline" in dirpath or "venv" in dirpath:
            continue
            
        for f in filenames:
            if not f.endswith(".txt"): continue
            
            full_path = os.path.join(dirpath, f)
            h = hash_file(full_path)
            hashes[h].append(full_path)
            count += 1
            
    print(f"Scanned {count} files.")
    
    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    
    if not duplicates:
        print("No duplicate files found.")
    else:
        print(f"Found {len(duplicates)} sets of duplicates!")
        for h, paths in duplicates.items():
            print(f"\nHash: {h}")
            for p in paths:
                print(f" - {p}")

if __name__ == "__main__":
    main()
