
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_group_id

def load_data(base_dir):
    data_dir = os.path.join(base_dir, 'output', 'data')
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        
        # Load feature names to filter
        with open(os.path.join(data_dir, 'feature_names.txt'), 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
            
        return X, y, meta, feature_names
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def plot_custom_confusion_matrix(model, X_test, y_test, output_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred) # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_test, y_pred)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Create figure for custom layout
    fig = plt.figure(figsize=(10, 8))
    
    # Colors
    c_good = '#a8dca8' # Light Green
    c_bad = '#ff8c8c'  # Light Red
    c_neut = '#d9d9d9' # Grey

    # Values
    cells = [
        {'text': f"TP\n({tp})", 'val': tp, 'color': c_good, 'x': 0.5, 'y': 2.5},
        {'text': f"FN\n({fn})", 'val': fn, 'color': c_bad,  'x': 1.5, 'y': 2.5},
        {'text': f"Recall\n{sensitivity:.2f}", 'val': sensitivity, 'color': c_neut, 'x': 2.5, 'y': 2.5},
        
        {'text': f"FP\n({fp})", 'val': fp, 'color': c_bad,  'x': 0.5, 'y': 1.5},
        {'text': f"TN\n({tn})", 'val': tn, 'color': c_good, 'x': 1.5, 'y': 1.5},
        {'text': f"Specificity\n{specificity:.2f}", 'val': specificity, 'color': c_neut, 'x': 2.5, 'y': 1.5},
        
        {'text': f"Precision\n{precision:.2f}", 'val': precision, 'color': c_neut, 'x': 0.5, 'y': 0.5},
        {'text': f"NPV\n{npv:.2f}", 'val': npv, 'color': c_neut, 'x': 1.5, 'y': 0.5},
        {'text': f"Accuracy\n{accuracy:.2f}", 'val': accuracy, 'color': c_neut, 'x': 2.5, 'y': 0.5},
    ]

    # Plot
    ax = plt.gca()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    
    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw Cells
    for cell in cells:
        rect = plt.Rectangle((cell['x']-0.5, cell['y']-0.5), 1, 1, facecolor=cell['color'], edgecolor='white', lw=2)
        ax.add_patch(rect)
        ax.text(cell['x'], cell['y'], cell['text'], ha='center', va='center', fontsize=14, fontweight='bold', color='black')

    # Draw Headers (Outside grid)
    # Top Headers (Predicted)
    plt.text(0.5, 3.2, "Predicted Positive", ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))
    plt.text(1.5, 3.2, "Predicted Negative", ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))
    
    # Left Headers (Actual) - Rotated
    plt.text(-0.2, 2.5, "Actual Positive", ha='center', va='center', rotation=90, fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))
    plt.text(-0.2, 1.5, "Actual Negative", ha='center', va='center', rotation=90, fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))

    # No Title as requested
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Custom Confusion Matrix to {output_path}")

def main():
    print("--- Generating Confusion Matrix (CFP-10 Only, HGB) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'CFP-10', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y, meta, feature_names = load_data(base_dir)
    if X is None: return
    
    # 2. Filter for CFP-10 Features
    # 'CFP-10', 'CFP-10*', 'CFP-10_z2', 'CFP-10*_z2'
    cfp10_indices = [i for i, f in enumerate(feature_names) if 'CFP-10' in f]
    selected_features = [feature_names[i] for i in cfp10_indices]
    
    print(f"Filtering features... Keeping {len(cfp10_indices)} features: {selected_features}")
    
    X_filtered = X[:, cfp10_indices]
    
    # 3. Split (80/20) - SAME split as training
    groups = np.array([get_group_id(f) for f in meta['filename']])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_filtered, y, groups))
    
    X_train = X_filtered[train_idx]
    y_train = y[train_idx]
    
    X_test = X_filtered[test_idx]
    y_test = y[test_idx]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Train Model
    print("Training HGB Model...")
    model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
    model.fit(X_train, y_train)
    
    # 5. Generate Plot
    cm_path = os.path.join(output_dir, 'hgb_cfp10_confusion_matrix_styled.png')
    plot_custom_confusion_matrix(model, X_test, y_test, cm_path)
    
    print("Done.")

if __name__ == "__main__":
    main()
