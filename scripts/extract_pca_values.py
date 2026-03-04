import os
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import get_group_id

def main():
    print("--- Extracting PC1 and PC2 Values ---")

    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # pipeline/
    data_file = os.path.join(base_dir, 'output', 'data', 'processed_dataset.csv')
    output_csv = os.path.join(base_dir, 'output', 'data', 'pca_values.csv')
    
    # 2. Load Data
    print(f"Loading data from {data_file}...")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run preprocessing first.")
        return

    df = pd.read_csv(data_file)
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    # Extract Features
    metadata_cols = ['Label', 'Filename', 'Group', 'Class']
    feature_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('Unnamed')]
    
    # Add Group/Patient ID
    df['Group'] = df['Filename'].apply(get_group_id)
    
    label_map = {0: 'NTM', 1: 'TB'}
    if 'Label' in df.columns:
         df['Class'] = df['Label'].map(label_map).fillna(df['Label'])
    
    # 3. PCA
    print("Performing PCA...")
    X = df[feature_cols].values
    
    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create output DataFrame
    pca_df = pd.DataFrame()
    pca_df['Filename'] = df['Filename']
    pca_df['Group'] = df['Group']
    pca_df['Class'] = df['Class']
    pca_df['PC1'] = X_pca[:, 0]
    pca_df['PC2'] = X_pca[:, 1]
    
    # Save to CSV
    pca_df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(pca_df)} PC values to {output_csv}")

if __name__ == '__main__':
    main()
