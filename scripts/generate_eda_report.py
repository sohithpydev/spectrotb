
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import get_group_id

def main():
    print("--- Generating Data Overview Report ---")

    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # pipeline/
    data_file = os.path.join(base_dir, 'output', 'data', 'processed_dataset.csv')
    report_dir = os.path.join(base_dir, 'output', 'reports')
    figures_dir = os.path.join(report_dir, 'figures')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # 2. Load Data
    print(f"Loading data from {data_file}...")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run 01_process_data.py first.")
        return

    df = pd.read_csv(data_file)
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    # Extract Features (columns that are numeric and not metadata)
    metadata_cols = ['Label', 'Filename', 'Group', 'Class']
    feature_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('Unnamed')]
    
    # Add Group/Patient ID
    df['Group'] = df['Filename'].apply(get_group_id)
    
    # Map Label to Class Name if needed (assuming 0=NTM, 1=TB based on previous file)
    # But let's check unique values in Label first or rely on metadata
    label_map = {0: 'NTM', 1: 'TB'}
    if 'Label' in df.columns:
         df['Class'] = df['Label'].map(label_map).fillna(df['Label'])
    
    # 3. Calculate Statistics
    print("Calculating statistics...")
    stats = []
    
    for cls in df['Class'].unique():
        cls_df = df[df['Class'] == cls]
        n_spectra = len(cls_df)
        n_patients = cls_df['Group'].nunique()
        stats.append({
            'Class': cls,
            'Spectra': n_spectra,
            'Patients': n_patients,
            'Spectra/Patient': f"{n_spectra/n_patients:.2f}"
        })
        
    stats_df = pd.DataFrame(stats)
    print(stats_df)
    
    # 4. Generate Plots
    sns.set_style("whitegrid")
    
    # 4.1 Class Distribution (Bar Plot)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=stats_df, x='Class', y='Spectra', hue='Class', palette='viridis', legend=False)
    plt.title('Number of Spectra per Class')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=stats_df, x='Class', y='Patients', hue='Class', palette='viridis', legend=False)
    plt.title('Number of Patients per Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'eda_class_distribution.png'), dpi=300)
    plt.close()
    
    # 4.2 Mean Feature Intensity
    print("Generating mean feature intensity plot...")
    plt.figure(figsize=(12, 6))
    
    # Use indices for plotting but label with feature names
    x_indices = np.arange(len(feature_cols))
        
    for cls in df['Class'].unique():
        cls_data = df[df['Class'] == cls][feature_cols].values
        mean_spec = np.mean(cls_data, axis=0)
        std_spec = np.std(cls_data, axis=0)
        
        plt.plot(x_indices, mean_spec, marker='o', label=f'{cls} Mean')
        plt.fill_between(x_indices, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2)
        
    plt.xticks(x_indices, feature_cols, rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Intensity')
    plt.title('Mean Feature Intensity by Class (prediction intervals: +/- 1 std dev)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'eda_mean_spectra.png'), dpi=300)
    plt.close()
    
    # 4.3 PCA
    print("Performing PCA...")
    X = df[feature_cols].values
    y = df['Class'].values
    
    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Class'] = y
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Class', alpha=0.7, palette='viridis')
    plt.title(f'PCA of Spectra (Explained Variance: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
    plt.savefig(os.path.join(figures_dir, 'eda_pca.png'), dpi=300)
    plt.close()

    # 5. Generate Markdown Report
    print("Generating markdown report...")
    
    report_path = os.path.join(report_dir, 'data_overview.md')
    
    with open(report_path, 'w') as f:
        f.write("# Data Overview & EDA Report\n\n")
        f.write("## 1. Dataset Statistics\n\n")
        f.write(stats_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Class Distribution\n")
        f.write("![Class Distribution](files/eda_class_distribution.png)\n\n") # Check relative path
        # Actually standard markdown path from report dir would be ./figures/
        # But if using standard agentic conventions, maybe absolute or relative to root.
        # Let's use relative to the report file.
        f.write("![Class Distribution](figures/eda_class_distribution.png)\n\n")
        
        f.write("## 3. Mean Spectra\n")
        f.write("Visualizing the average spectral signature for each class with standard deviation shaded.\n\n")
        f.write("![Mean Spectra](figures/eda_mean_spectra.png)\n\n")
        
        f.write("## 4. PCA Analysis\n")
        f.write("Principal Component Analysis (PCA) of the full spectral data.\n\n")
        f.write("![PCA](figures/eda_pca.png)\n\n")
        
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
