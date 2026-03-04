# scripts/04_predict_new.py
import os
import sys
import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_spectrum
from src.preprocessing import preprocess_spectrum
from src.features import extract_features

def main():
    print("--- Blind Validation on 'mof_tb_test' Data ---")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'mof_tb_test')
    model_path = os.path.join(base_dir, 'output', 'models', 'histgradientboosting.pkl')
    
    # Check paths
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run scripts/03_train_validate_grouped.py first.")
        return
        
    # Load Model
    print(f"Loading Model: {os.path.basename(model_path)}...")
    model = joblib.load(model_path)
    
    # Load Files
    files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
    print(f"Found {len(files)} files to process.")
    
    results = []
    
    print("Processing and Predicting...")
    for filename in tqdm(files):
        try:
            filepath = os.path.join(test_dir, filename)
            
            # 1. Load
            mz, intensity = load_spectrum(filepath)
            
            # 2. Preprocess
            _, processed_int = preprocess_spectrum(mz, intensity)
            
            # 3. Extract Features
            feats = extract_features(mz, processed_int)
            feats = feats.reshape(1, -1) # Reshape for single sample
            
            # 4. Predict
            # Labels: 1=TB, 0=NTM
            pred_class = model.predict(feats)[0]
            pred_prob = model.predict_proba(feats)[0][1] # Probability of Class 1 (TB)
            
            label = "TB" if pred_class == 1 else "NTM"
            
            results.append({
                "Filename": filename,
                "Prediction": label,
                "Confidence": f"{pred_prob*100:.1f}%" if label == "TB" else f"{(1-pred_prob)*100:.1f}%",
                "TB_Score": pred_prob
            })
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    # Display Results
    df = pd.DataFrame(results)
    
    # Sort by Filename for readability
    df = df.sort_values(by="Filename")
    
    print("\n--- Prediction Results ---")
    print(tabulate(df[['Filename', 'Prediction', 'Confidence']], headers='keys', tablefmt='github'))
    
    # Save CSV
    out_path = os.path.join(base_dir, 'output', 'reports', 'mof_blind_test_predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
