
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_group_id

def load_data(base_dir):
    data_dir = os.path.join(base_dir, 'output', 'data')
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        meta = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
        return X, y, meta
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def plot_cv_roc(X, y, groups, output_path):
    cv = StratifiedGroupKFold(n_splits=10)
    classifier = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(class_weight='balanced', random_state=42))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    for i, (train, test) in enumerate(cv.split(X, y, groups)):
        classifier.fit(X[train], y[train])
        probas_ = classifier.predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (10-Fold CV)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved CV ROC plot to {output_path}")

def plot_test_roc(model, X_test, y_test, output_path):
    probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (External Validation Set)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Test ROC plot to {output_path}")

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
    
    # We will use a grid layout or simple text placement to match the user's request (Image style)
    # The reference image has:
    # Top: Predicted Labels (Positive, Negative)
    # Left: Actual Labels (Positive, Negative)
    # Center 2x2: TP, FN / FP, TN with colors (Green for TP/TN, Red for FP/FN)
    # Right Column: Recall / Specificity (Grey)
    # Bottom Row: Precision / NPV (Grey)
    # Bottom Right: Accuracy (Grey)

    # Colors
    c_good = '#a8dca8' # Light Green
    c_bad = '#ff8c8c'  # Light Red
    c_neut = '#d9d9d9' # Grey
    c_head = '#fce5cd' # Light Orange/Yellow head

    # Grid Setup (3x3)
    # Cells: (0,0) Empty, (0,1) Pred Pos, (0,2) Pred Neg
    # (1,0) Act Pos, (1,1) TP, (1,2) FN, (1,3) Recall
    # (2,0) Act Neg, (2,1) FP, (2,2) TN, (2,3) Specificity
    # (3,0) Empty,   (3,1) Prec, (3,2) NPV, (3,3) Acc
    
    # Let's verify the user image mapping:
    # Columns: Predicted Positive, Predicted Negative
    # Rows: Actual Positive, Actual Negative
    #
    # [TP] [FN] [Recall]
    # [FP] [TN] [Specificity]
    # [Prec][NPV][Accuracy]

    # Note: Scikit-learn CM is [[TN, FP], [FN, TP]] usually?
    # No, sklearn is:
    # [[TN, FP], (Row 0: True Neg)
    #  [FN, TP]] (Row 1: True Pos)
    #
    # Wait, Reference Image has "Actual Positive" as Top Row?
    # User Image:
    # Rows: Positive, Negative
    # Cols: Positive, Negative
    #
    # If Act=Pos (Row 1), Pred=Pos (Col 1) -> TP
    # If Act=Pos (Row 1), Pred=Neg (Col 2) -> FN
    # If Act=Neg (Row 2), Pred=Pos (Col 1) -> FP
    # If Act=Neg (Row 2), Pred=Neg (Col 2) -> TN
    
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

    # plt.title("Confusion Matrix Metrics", fontsize=16, pad=30)
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Custom Confusion Matrix to {output_path}")

def main():
    print("--- Generating Detailed HGB Plots ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y, meta = load_data(base_dir)
    if X is None: return
    
    # 2. Split (80/20) - SAME split as training
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.array([get_group_id(f) for f in meta['filename']])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. CV ROC on Training Set
    print("Generating CV ROC...")
    cv_roc_path = os.path.join(output_dir, 'hgb_cv_roc.png')
    plot_cv_roc(X_train, y_train, groups_train, cv_roc_path)
    
    # 4. Train Final Model
    print("Training Final Model on Full Training Set...")
    model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(class_weight='balanced', random_state=42))
    model.fit(X_train, y_train)
    
    # 5. Test ROC
    print("Generating Test ROC...")
    test_roc_path = os.path.join(output_dir, 'hgb_test_roc.png')
    plot_test_roc(model, X_test, y_test, test_roc_path)
    
    # 6. Custom CM
    print("Generating Confusion Matrix...")
    cm_path = os.path.join(output_dir, 'hgb_confusion_matrix_styled.png')
    plot_custom_confusion_matrix(model, X_test, y_test, cm_path)
    
    print("Done.")

if __name__ == "__main__":
    main()
