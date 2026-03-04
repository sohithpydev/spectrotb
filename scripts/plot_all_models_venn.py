
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from venn import venn

def main():
    print("--- Generating 5-Way Venn Diagram (All 44 Models Grouped) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(base_dir, 'output', 'biomarker_experiments', 'Combined', 'analysis_all')
    csv_path = os.path.join(analysis_dir, 'all_models_feature_importance.csv')
    
    if not os.path.exists(csv_path):
        print("Feature importance CSV not found.")
        return

    df = pd.read_csv(csv_path)
    feature_col = 'Feature'
    
    # Define Model Families (Grouping all 44 models)
    # We need to map every column (except Feature and Mean) to a group.
    
    all_models = [c for c in df.columns if c not in ['Feature', 'Mean_Importance']]
    
    # 1. Tree Ensembles & Boosting (The heavy hitters)
    # Includes: RF, ExtraTrees, Bagging, AdaBoost, GBM, HGB, XGB, LGB, CatBoost
    group_trees = [m for m in all_models if any(x in m.lower() for x in 
                   ['randomforest', 'extratrees', 'bagging', 'boost', 'gbm', 'lightgbm', 'xgboost', 'catboost'])]
                   
    # 2. Linear Classifiers & Discriminant Analysis
    # Includes: Logistic, Ridge, SGD, PasAgg, LDA, QDA, Perceptron, CalibratedSVC (linear base)
    group_linear = [m for m in all_models if any(x in m.lower() for x in 
                    ['logistic', 'ridge', 'sgd', 'passiveaggressive', 'lda', 'qda', 'perceptron', 'calibrated'])]
                    
    # 3. Support Vector Machines
    # Includes: SVC, LinearSVC, NuSVC
    # Note: Calibrated_LinearSVC captured in linear above, need to be careful not to dupe or miss.
    # Let's be exclusive.
    group_svm = [m for m in all_models if 'svc' in m.lower() and m not in group_linear and m not in group_trees]
    
    # 4. Nearest Neighbors & Naive Bayes
    # Includes: KNN, NearestCentroid, GaussianNB, BernoulliNB
    group_knn_nb = [m for m in all_models if any(x in m.lower() for x in ['knn', 'nearest', 'nb'])]
    
    # 5. Neural Networks & Others (Decision Trees - distinct from Ensembles)
    # Includes: MLP, DecisionTree
    group_other = [m for m in all_models if any(x in m.lower() for x in ['mlp', 'decisiontree'])]
    
    # Check for unassigned models
    assigned = set(group_trees + group_linear + group_svm + group_knn_nb + group_other)
    unassigned = set(all_models) - assigned
    if unassigned:
        print(f"Warning: Unassigned models: {unassigned}")
        # Assign them to 'group_other' as fallback
        group_other.extend(list(unassigned))
        
    groups = {
        "Trees & Boost": group_trees,
        "Linear & DA": group_linear,
        "SVMs": group_svm,
        "KNN & NB": group_knn_nb,
        "Neural & DT": group_other
    }
    
    sets = {}
    
    for group_name, models in groups.items():
        group_features = set()
        print(f"\nGroup '{group_name}' ({len(models)} models):")
        for model in models:
            if model in df.columns:
                # Union of Top 5 features from all models in this group
                top5 = df.sort_values(by=model, ascending=False).head(5)[feature_col].tolist()
                group_features.update(top5)
            else:
                print(f"  Warning: {model} not found in CSV.")
        
        sets[group_name] = group_features
        print(f"  -> Features ({len(group_features)}): {sorted(list(group_features))}")

    # Generate Venn
    plt.figure(figsize=(12, 12))
    venn(sets)
    plt.title("Feature Overlap (Union of Top 5 Features per 44 Models Grouped)", fontsize=16)
    
    plot_path = os.path.join(analysis_dir, 'all_models_5way_venn.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved 44-model grouped Venn to {plot_path}")
    
    # ---------------------------------------------------------
    # Detailed Intersection Analysis
    # ---------------------------------------------------------
    print("\n--- Detailed Intersection Breakdown ---")
    
    # Helper to get intersection of a list of group names
    def get_intersection(group_names):
        if not group_names: return set()
        res = sets[group_names[0]].copy()
        for g in group_names[1:]:
            res &= sets[g]
        return res
        
    names = list(sets.keys())
    # 1. Calculate Full Intersection (Center)
    core = get_intersection(names)
    print(f"\n[CORE] Shared by ALL 5 Families ({len(core)}):")
    for f in sorted(list(core)):
        print(f"  - {f}")
        
    # 2. Calculate Exclusive Intersections (just to give examples of the numbers)
    # It's hard to print all 31 permutations, but let's print unique ones per group
    # or specific interesting ones.
    
    # Let's compute specifically what's unique to the "Trees & Boost" group vs others
    tree_features = sets["Trees & Boost"]
    others = set()
    for n in names:
        if n != "Trees & Boost":
            others.update(sets[n])
            
    unique_trees = tree_features - others
    print(f"\n[UNIQUE] Exclusive to 'Trees & Boost' ({len(unique_trees)}):")
    for f in sorted(list(unique_trees)):
        print(f"  - {f}")
        
    # Let's print the counts for the map
    # We can iterate through all combinations of groups?
    # Actually, let's just create a "membership map" which is cleaner.
    
    all_feats = set()
    for s in sets.values():
        all_feats.update(s)
        
    print("\n--- Feature Membership Map ---")
    print(f"{'Feature':<15} | {'Count':<5} | {'Groups'}")
    print("-" * 60)
    
    feat_map = []
    for f in sorted(list(all_feats)):
        present_in = []
        for name in names:
            if f in sets[name]:
                present_in.append(name)
        feat_map.append((f, len(present_in), present_in))
        
    # Sort by count (descending)
    feat_map.sort(key=lambda x: x[1], reverse=True)
    
    for f, count, groups in feat_map:
        # Abbreviate groups for display
        grp_str = ", ".join([g.split(' ')[0] for g in groups])
        print(f"{f:<15} | {count:<5} | {grp_str}")

if __name__ == "__main__":
    main()
