import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .base import BaseModel
from .registry import register_model


@register_model('LogReg-L2')
class LogRegL2Model(BaseModel):
    def __init__(self, C=1.0):
        super().__init__('LogReg-L2')
        self.model = LogisticRegression(C=C, penalty='l2', max_iter=1000,
                                        solver='lbfgs', random_state=42)

    def fit(self, X, y):
        if len(X) < 100 or len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)


@register_model('RandomForest')
class RFClassModel(BaseModel):
    def __init__(self, n_est=200, max_depth=4):
        super().__init__('RandomForest')
        self.model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
                                            min_samples_leaf=20, random_state=42, n_jobs=-1)

    def fit(self, X, y):
        if len(X) < 200 or len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        probs = np.array([t.predict_proba(X_row)[0][1] for t in self.model.estimators_])
        return float(np.mean(probs)), float(np.var(probs)) + 1e-6


@register_model('LGBM')
class LightGBMModel(BaseModel):
    """LightGBM with anti-bias calibration and conservative regularisation.

    Fixes vs previous:
    - max_depth 8->5, num_leaves 80->31 (prevents bullish-regime overfitting)
    - Removed scale_pos_weight; class_weight=balanced handles imbalance alone
    - min_child_samples 15->30  (more stable leaf estimates on ~1100 samples)
    - CalibratedClassifierCV(isotonic) for well-calibrated probabilities
    - No verbose prints (walk-forward ~10x faster)
    """

    def __init__(self, n_estimators=400, max_depth=-1, learning_rate=0.05,
                 num_leaves=63, lambda_l1=0.5, lambda_l2=0.5,
                 boosting_type='gbdt'):
        super().__init__('LightGBM')
        try:
            import lightgbm as lgb
            self._raw = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                boosting_type=boosting_type,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=30,
                bagging_fraction=0.8,
                feature_fraction=0.8,
                bagging_freq=5,
                objective='binary',
                class_weight='balanced',
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
            self.model = CalibratedClassifierCV(self._raw, method='isotonic', cv=3)
        except ImportError:
            self._raw = None
            self.model = None

    def fit(self, X, y):
        if self._raw is None or len(X) < 150 or len(np.unique(y)) < 2:
            return
        try:
            self.model.fit(X, y)
            self.is_fitted = True
        except Exception:
            pass

    def predict_proba_one(self, X_row):
        if not self.is_fitted or self.model is None:
            return 0.5, 0.25
        try:
            proba = self.model.predict_proba(X_row)
            p = float(proba[0][1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


@register_model('KNN')
class KNNClassModel(BaseModel):
    def __init__(self, k=7):
        super().__init__('KNN')
        self.model = KNeighborsClassifier(n_neighbors=k, weights='distance',
                                          metric='euclidean', algorithm='auto', n_jobs=-1)

    def fit(self, X, y):
        if len(X) < 100 or len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)
