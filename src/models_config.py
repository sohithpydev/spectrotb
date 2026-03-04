from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

# Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# --- 11. Custom Wrappers (CatBoost Fix) ---
from sklearn.base import BaseEstimator, ClassifierMixin

class SafeCatBoostClassifier(ClassifierMixin, BaseEstimator):
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
        tags.classifier_tags = None
        return tags

def get_models():
    """
    Returns a list of (name, model) tuples.
    Target: 42+ distinct models/configurations.
    """
    models = []
    
    # --- 1. Linear Models (10) ---
    models.append(('LogisticRegression_L2', LogisticRegression(penalty='l2', max_iter=1000, class_weight='balanced')))
    # ElasticNet requires 'saga' solver
    models.append(('LogisticRegression_Elastic', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, class_weight='balanced')))
    models.append(('RidgeClassifier', RidgeClassifier(class_weight='balanced')))
    models.append(('SGD_Hinge', SGDClassifier(loss='hinge', class_weight='balanced')))
    models.append(('SGD_Log', SGDClassifier(loss='log_loss', class_weight='balanced')))
    models.append(('SGD_ModHuber', SGDClassifier(loss='modified_huber', class_weight='balanced')))
    models.append(('SGD_SquaredHinge', SGDClassifier(loss='squared_hinge', class_weight='balanced')))
    models.append(('Perceptron', Perceptron(class_weight='balanced')))
    models.append(('PassiveAggressive', PassiveAggressiveClassifier(class_weight='balanced')))
    models.append(('Calibrated_LinearSVC', CalibratedClassifierCV(LinearSVC(dual='auto', class_weight='balanced'))))

    # --- 2. Support Vector Machines (6) ---
    models.append(('LinearSVC', LinearSVC(dual='auto', class_weight='balanced')))
    models.append(('SVC_RBF', SVC(kernel='rbf', probability=True, class_weight='balanced')))
    models.append(('SVC_Poly', SVC(kernel='poly', probability=True, class_weight='balanced')))
    models.append(('SVC_Sigmoid', SVC(kernel='sigmoid', probability=True, class_weight='balanced')))
    models.append(('NuSVC', NuSVC(probability=True, class_weight='balanced')))
    # Variation of C
    models.append(('LinearSVC_C10', LinearSVC(C=10.0, dual='auto', class_weight='balanced')))

    # --- 3. Nearest Neighbors (5) ---
    models.append(('KNN_3', KNeighborsClassifier(n_neighbors=3)))
    models.append(('KNN_5', KNeighborsClassifier(n_neighbors=5)))
    models.append(('KNN_7', KNeighborsClassifier(n_neighbors=7)))
    models.append(('KNN_9', KNeighborsClassifier(n_neighbors=9)))
    models.append(('NearestCentroid', NearestCentroid()))

    # --- 4. Naive Bayes (2) ---
    models.append(('GaussianNB', GaussianNB()))
    models.append(('BernoulliNB', BernoulliNB()))

    # --- 5. Trees (4) ---
    models.append(('DecisionTree_Gini', DecisionTreeClassifier(criterion='gini', class_weight='balanced')))
    models.append(('DecisionTree_Entropy', DecisionTreeClassifier(criterion='entropy', class_weight='balanced')))
    models.append(('ExtraTree_Gini', ExtraTreeClassifier(criterion='gini', class_weight='balanced')))
    models.append(('ExtraTree_Entropy', ExtraTreeClassifier(criterion='entropy', class_weight='balanced')))

    # --- 6. Ensembles (Scikit-Learn) (10) ---
    models.append(('RandomForest_100', RandomForestClassifier(n_estimators=100, class_weight='balanced')))
    models.append(('RandomForest_200', RandomForestClassifier(n_estimators=200, class_weight='balanced')))
    models.append(('ExtraTrees_100', ExtraTreesClassifier(n_estimators=100, class_weight='balanced')))
    models.append(('ExtraTrees_200', ExtraTreesClassifier(n_estimators=200, class_weight='balanced')))
    models.append(('AdaBoost_50', AdaBoostClassifier(n_estimators=50)))
    models.append(('AdaBoost_100', AdaBoostClassifier(n_estimators=100)))
    models.append(('GradientBoosting', GradientBoostingClassifier(n_estimators=100)))
    models.append(('HistGradientBoosting', HistGradientBoostingClassifier(class_weight='balanced')))
    models.append(('Bagging_SVC', BaggingClassifier(estimator=SVC(class_weight='balanced'), n_estimators=10)))
    models.append(('Bagging_Tree', BaggingClassifier(n_estimators=50)))

    # --- 7. Boosting Libraries (3) ---
    # XGBoost
    models.append(('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')))
    # LightGBM
    models.append(('LightGBM', lgb.LGBMClassifier(class_weight='balanced', verbose=-1)))
    # CatBoost
    models.append(('CatBoost', SafeCatBoostClassifier(iterations=100, verbose=0, auto_class_weights='Balanced')))

    # --- 8. Discriminant Analysis (2) ---
    models.append(('LDA', LinearDiscriminantAnalysis()))
    # Fix Collinearity by adding regularization
    models.append(('QDA', QuadraticDiscriminantAnalysis(reg_param=0.5)))

    # --- 9. Neural Networks (2) ---
    models.append(('MLP_1Layer', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)))
    models.append(('MLP_2Layers', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)))

    # --- 10. Baseline (1) ---
    models.append(('Dummy', DummyClassifier(strategy='stratified')))
    
    # Total should be > 42
    return models
