
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, roc_auc_score
import catboost as cb


# Define the wrapper with fix
class SafeCatBoostClassifier(ClassifierMixin, BaseEstimator): # Swap order?
    _estimator_type = "classifier"
    
    def __init__(self, iterations=100, verbose=0, auto_class_weights='Balanced'):
        self.iterations = iterations
        self.verbose = verbose
        self.auto_class_weights = auto_class_weights
        self.model = None
        
    def fit(self, X, y):
        self.model = cb.CatBoostClassifier(
            iterations=self.iterations, 
            verbose=self.verbose, 
            auto_class_weights=self.auto_class_weights,
            allow_writing_files=False
        )
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = None # Let default handle? Or need to init?
        return tags

# Create Dummy Data
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# CV Setup
cv = StratifiedKFold(n_splits=5)
scoring = {'roc_auc': 'roc_auc'}

from sklearn.base import is_classifier

# Model
clf = SafeCatBoostClassifier()
try:
    print(f"Tags: {clf.__sklearn_tags__()}")
except:
    print("Tags: Error getting tags")
    
print(f"Is Classifier? {is_classifier(clf)}")
print(f"Estimator Type: {getattr(clf, '_estimator_type', 'None')}")

print("Running Cross-Validation...")
try:
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score='raise')
    print("Scores:", scores) 
    print("Mean AUC:", np.mean(scores['test_roc_auc']))
except Exception as e:
    print("CV Failed:", e)
