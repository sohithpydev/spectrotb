import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models_config import get_models
from src.features import get_feature_names

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'output', 'reports')
    
    # Load grouped dataset
    cache_fX = os.path.join(reports_dir, 'clsa_X.npy')
    cache_fy = os.path.join(reports_dir, 'clsa_y.npy')
    cache_fg = os.path.join(reports_dir, 'clsa_groups.npy')
    
    if not os.path.exists(cache_fX):
        print("Run the clsa pipeline first to cache features.")
        return
        
    X = np.load(cache_fX)
    y = np.load(cache_fy)
    groups = np.load(cache_fg)
    
    # Feature Subsetting (CFP-10 + ESAT-6 = ALL)
    # 80/20 Split identically
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Load CV metrics to get the mean AUROC
    cv_csv = os.path.join(reports_dir, 'clsa_cv_metrics_CFP-10_and_ESAT-6.csv')
    cv_df = pd.read_csv(cv_csv)
    
    # Parse the mean from "0.98 ± 0.02"
    cv_df['CV_AUROC_Mean'] = cv_df['AUROC'].apply(lambda x: float(str(x).split(' ± ')[0]))
    
    models = dict(get_models())
    
    results = []
    
    print("Evaluating Test AUROC and computing MAD for all 45 models...")
    for idx, row in cv_df.iterrows():
        name = row['Model']
        cv_auroc = row['CV_AUROC_Mean']
        
        model = models[name]
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        
        if hasattr(pipeline, "predict_proba"):
            try:
                y_probs = pipeline.predict_proba(X_test)[:, 1]
            except: 
                 y_probs = pipeline.decision_function(X_test)
        else:
            y_probs = pipeline.decision_function(X_test)
            
        test_auroc = roc_auc_score(y_test, y_probs)
        mad = abs(cv_auroc - test_auroc)
        
        results.append({
            'Algorithm': name,
            'CV_AUROC': cv_auroc,
            'Test_AUROC': test_auroc,
            'MAD': mad
        })
        
    # Rank by MAD (ascending - lower is better generalization), then by Test AUROC
    df_mad = pd.DataFrame(results)
    df_mad = df_mad.sort_values(by=['MAD', 'Test_AUROC'], ascending=[True, False])
    
    out_csv = os.path.join(reports_dir, 'clsa_mad_ranking_CFP-10_and_ESAT-6.csv')
    df_mad.to_csv(out_csv, index=False)
    
    print("\nTop 45 External Validation Rankings by MAD AUROC (CFP-10 + ESAT-6):")
    print(tabulate(df_mad, headers='keys', tablefmt='github', floatfmt=".4f", showindex=False))
    print(f"\nFull ranking exported to: {out_csv}")

if __name__ == "__main__":
    main()
