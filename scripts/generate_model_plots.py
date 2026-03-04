
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

def get_model(name):
    if name == 'XGBoost':
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif name == 'LightGBM':
        return LGBMClassifier(random_state=42, class_weight='balanced')
    elif name == 'GradientBoosting':
        return GradientBoostingClassifier(random_state=42)
    elif name == 'RandomForest_200':
        return RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")

def plot_cv_roc(model_name, model_factory, X, y, groups, output_path):
    cv = StratifiedGroupKFold(n_splits=10)
    # Re-create pipeline fresh for each fold/call to avoid state issues
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    for i, (train, test) in enumerate(cv.split(X, y, groups)):
        # Instantiate fresh model
        clf = make_pipeline(StandardScaler(), get_model(model_name))
        clf.fit(X[train], y[train])
        
        probas_ = clf.predict_proba(X[test])
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
    plt.title(f'{model_name} - ROC (10-Fold CV)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved CV ROC plot to {output_path}")

def plot_test_roc(model, model_name, X_test, y_test, output_path):
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
    plt.title(f'{model_name} - ROC (Validation Set)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Test ROC plot to {output_path}")

def plot_custom_confusion_matrix(model, model_name, X_test, y_test, output_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_test, y_pred)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    fig = plt.figure(figsize=(10, 8))
    
    c_good = '#a8dca8'
    c_bad = '#ff8c8c'
    c_neut = '#d9d9d9'

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

    ax = plt.gca()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for cell in cells:
        rect = plt.Rectangle((cell['x']-0.5, cell['y']-0.5), 1, 1, facecolor=cell['color'], edgecolor='white', lw=2)
        ax.add_patch(rect)
        ax.text(cell['x'], cell['y'], cell['text'], ha='center', va='center', fontsize=14, fontweight='bold', color='black')

    plt.text(0.5, 3.2, "Predicted Positive", ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))
    plt.text(1.5, 3.2, "Predicted Negative", ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))
    
    plt.text(-0.2, 2.5, "Actual Positive", ha='center', va='center', rotation=90, fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))
    plt.text(-0.2, 1.5, "Actual Negative", ha='center', va='center', rotation=90, fontsize=12, fontweight='bold', bbox=dict(facecolor='#ffd966', edgecolor='white'))

    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Custom Confusion Matrix to {output_path}")

def main():
    print("--- Generating Plots for Multiple Models ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, meta = load_data(base_dir)
    if X is None: return
    
    groups = np.array([get_group_id(f) for f in meta['filename']])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    models_to_run = ['XGBoost', 'LightGBM', 'GradientBoosting', 'RandomForest_200']
    
    for model_name in models_to_run:
        print(f"\nProcessing {model_name}...")
        
        # 1. CV ROC
        cv_path = os.path.join(output_dir, f'{model_name.lower()}_cv_roc.png')
        plot_cv_roc(model_name, None, X_train, y_train, groups_train, cv_path)
        
        # 2. Train Final
        print(f"Training final {model_name}...")
        clf = make_pipeline(StandardScaler(), get_model(model_name))
        clf.fit(X_train, y_train)
        
        # 3. Test ROC
        test_path = os.path.join(output_dir, f'{model_name.lower()}_test_roc.png')
        plot_test_roc(clf, model_name, X_test, y_test, test_path)
        
        # 4. CM
        cm_path = os.path.join(output_dir, f'{model_name.lower()}_confusion_matrix_styled.png')
        plot_custom_confusion_matrix(clf, model_name, X_test, y_test, cm_path)
        
    print("\nAll plots generated.")

if __name__ == "__main__":
    main()
