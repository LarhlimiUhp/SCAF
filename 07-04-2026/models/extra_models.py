"""
SCAF Extended Model Registry  (+8 models)
==========================================
Adds the following expert models to the SCAF registry:

  1. XGBoost              – gradient boosting with tree learners
  2. ExtraTrees           – extremely randomised trees (fast, low-variance)
  3. AdaBoost             – adaptive boosting over decision stumps
  4. HistGradientBoosting – sklearn's fast histogram-based GBDT
  5. MLPClassifier        – shallow multilayer perceptron (sklearn)
  6. RidgeClassifier      – L2-penalised linear classifier (fast baseline)
  7. BaggingEnsemble      – bagging wrapper over LogReg for variance reduction
  8. CatBoost             – gradient boosting with native categorical support

All models follow the BaseModel interface (fit / predict_proba_one).
CatBoost and XGBoost are imported inside __init__ so missing packages
cause graceful degradation (model is not fitted) rather than import errors.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .registry import register_model


# ---------------------------------------------------------------------------
# 1. XGBoost
# ---------------------------------------------------------------------------

@register_model('XGBoost')
class XGBoostModel(BaseModel):
    """XGBoost gradient boosting classifier with isotonic calibration."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.1,
    ):
        super().__init__('XGBoost')
        try:
            from xgboost import XGBClassifier
            self._raw = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                verbosity=0,
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
            p = float(self.model.predict_proba(X_row)[0][1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


# ---------------------------------------------------------------------------
# 2. ExtraTrees
# ---------------------------------------------------------------------------

@register_model('ExtraTrees')
class ExtraTreesModel(BaseModel):
    """Extremely Randomised Trees – faster and often lower variance than RF."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 6):
        super().__init__('ExtraTrees')
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        if len(X) < 150 or len(np.unique(y)) < 2:
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        probs = np.array([t.predict_proba(X_row)[0][1] for t in self.model.estimators_])
        return float(np.mean(probs)), float(np.var(probs)) + 1e-6


# ---------------------------------------------------------------------------
# 3. AdaBoost
# ---------------------------------------------------------------------------

@register_model('AdaBoost')
class AdaBoostModel(BaseModel):
    """AdaBoost over decision stumps with SAMME.R algorithm."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.5):
        super().__init__('AdaBoost')
        self._raw = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.model = CalibratedClassifierCV(self._raw, method='sigmoid', cv=3)

    def fit(self, X, y):
        if len(X) < 100 or len(np.unique(y)) < 2:
            return
        try:
            self.model.fit(X, y)
            self.is_fitted = True
        except Exception:
            pass

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        try:
            p = float(self.model.predict_proba(X_row)[0][1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


# ---------------------------------------------------------------------------
# 4. HistGradientBoosting
# ---------------------------------------------------------------------------

@register_model('HistGBT')
class HistGBTModel(BaseModel):
    """sklearn HistGradientBoostingClassifier – fast native histogram GBDT."""

    def __init__(
        self,
        max_iter: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        l2_regularization: float = 1.0,
    ):
        super().__init__('HistGBT')
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            class_weight='balanced',
            random_state=42,
        )

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


# ---------------------------------------------------------------------------
# 5. MLP Classifier (sklearn)
# ---------------------------------------------------------------------------

@register_model('MLP')
class MLPClassModel(BaseModel):
    """Shallow MLP (2 hidden layers) with early stopping and L2 regularisation."""

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        alpha: float = 1e-3,
        max_iter: int = 200,
    ):
        super().__init__('MLP')
        self.sc = StandardScaler()
        self._raw = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        )
        self.model = CalibratedClassifierCV(self._raw, method='isotonic', cv=3)

    def fit(self, X, y):
        if len(X) < 150 or len(np.unique(y)) < 2:
            return
        try:
            X_s = self.sc.fit_transform(X)
            self.model.fit(X_s, y)
            self.is_fitted = True
        except Exception:
            pass

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        try:
            X_s = self.sc.transform(X_row)
            p = float(self.model.predict_proba(X_s)[0][1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


# ---------------------------------------------------------------------------
# 6. Ridge Classifier (linear discriminant via L2 regression)
# ---------------------------------------------------------------------------

@register_model('RidgeClass')
class RidgeClassModel(BaseModel):
    """
    Ridge classifier: fast L2-penalised linear model.
    sklearn's RidgeClassifier does not natively produce probabilities;
    we wrap it with Platt scaling (sigmoid calibration).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__('RidgeClass')
        self.sc = StandardScaler()
        self._raw = RidgeClassifier(alpha=alpha, class_weight='balanced')
        self.model = CalibratedClassifierCV(self._raw, method='sigmoid', cv=3)

    def fit(self, X, y):
        if len(X) < 100 or len(np.unique(y)) < 2:
            return
        try:
            X_s = self.sc.fit_transform(X)
            self.model.fit(X_s, y)
            self.is_fitted = True
        except Exception:
            pass

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        try:
            X_s = self.sc.transform(X_row)
            p = float(self.model.predict_proba(X_s)[0][1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


# ---------------------------------------------------------------------------
# 7. Bagging Ensemble
# ---------------------------------------------------------------------------

@register_model('BaggingLR')
class BaggingLRModel(BaseModel):
    """
    Bagging over Logistic Regression base estimators.
    Reduces variance via bootstrap aggregation.
    """

    def __init__(self, n_estimators: int = 50, max_features: float = 0.7):
        super().__init__('BaggingLR')
        self.sc = StandardScaler()
        base = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs',
                                   random_state=42)
        self.model = BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        if len(X) < 100 or len(np.unique(y)) < 2:
            return
        try:
            X_s = self.sc.fit_transform(X)
            self.model.fit(X_s, y)
            self.is_fitted = True
        except Exception:
            pass

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        try:
            X_s = self.sc.transform(X_row)
            # Average probabilities across all estimators
            probs = np.array([
                est.predict_proba(X_s)[0][1]
                for est in self.model.estimators_
            ])
            p = float(np.mean(probs))
            return p, float(np.var(probs)) + 1e-6
        except Exception:
            return 0.5, 0.25


# ---------------------------------------------------------------------------
# 8. CatBoost
# ---------------------------------------------------------------------------

@register_model('CatBoost')
class CatBoostModel(BaseModel):
    """CatBoost gradient boosting with native handling of ordered boosting."""

    def __init__(
        self,
        iterations: int = 400,
        depth: int = 4,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
    ):
        super().__init__('CatBoost')
        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                loss_function='Logloss',
                class_weights=[1.0, 1.0],
                random_seed=42,
                verbose=False,
            )
        except ImportError:
            self.model = None

    def fit(self, X, y):
        if self.model is None or len(X) < 150 or len(np.unique(y)) < 2:
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
            p = float(self.model.predict_proba(X_row)[0][1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25
