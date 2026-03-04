import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'output', 'reports')
    models_dir = os.path.join(base_dir, 'output', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load grouped dataset
    cache_fX = os.path.join(reports_dir, 'clsa_X.npy')
    cache_fy = os.path.join(reports_dir, 'clsa_y.npy')
    
    if not os.path.exists(cache_fX):
        print("Run the clsa pipeline first to cache features.")
        return
        
    X = np.load(cache_fX)
    y = np.load(cache_fy)
    
    print(f"Loaded CLSA features: {X.shape}")
    print("Training the final HistGradientBoostingClassifier on all available data cross CFP-10 + ESAT-6")
    
    # HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
    pipeline = make_pipeline(StandardScaler(), model)
    
    pipeline.fit(X, y)
    
    # Save model
    model_path = os.path.join(models_dir, 'hgb_clsa_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    main()
