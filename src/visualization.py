import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_spectrum(mz: np.ndarray, intensity: np.ndarray, title: str = "Mass Spectrum", save_path: str = None):
    """
    Plots a single mass spectrum.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mz, intensity, lw=1)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_preprocessing_steps(mz: np.ndarray, steps_dict: dict, title_prefix: str = "Sample", save_path: str = None):
    """
    Plots the stages of preprocessing: Raw, Smoothed, Baseline, Corrected.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Raw vs Smoothed
    axes[0].plot(mz, steps_dict['raw'], label='Raw', color='gray', alpha=0.5)
    axes[0].plot(mz, steps_dict['smoothed'], label='Smoothed (SG)', color='blue', lw=1)
    axes[0].legend()
    axes[0].set_title(f"{title_prefix} - Smoothing")
    axes[0].set_ylabel('Intensity')
    
    # Smoothed vs Baseline
    axes[1].plot(mz, steps_dict['smoothed'], label='Smoothed', color='blue', lw=1)
    axes[1].plot(mz, steps_dict['baseline'], label='Baseline (ALS)', color='red', linestyle='--', lw=1)
    axes[1].legend()
    axes[1].set_title(f"{title_prefix} - Baseline Est.")
    axes[1].set_ylabel('Intensity')
    
    # Corrected
    axes[2].plot(mz, steps_dict['corrected'], label='Baseline Corrected', color='green', lw=1)
    axes[2].legend()
    axes[2].set_title(f"{title_prefix} - Corrected")
    axes[2].set_xlabel('m/z')
    axes[2].set_ylabel('Intensity')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_names, importances, save_path=None):
    """
    Plots the feature importance bar chart.
    """
    # Sort features by importance
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_imps = [importances[i] for i in indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), sorted_imps, align='center', color='teal')
    plt.yticks(range(len(indices)), sorted_names)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
