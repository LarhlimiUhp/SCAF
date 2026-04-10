"""
Extra Models for SCAF-LS  (8 additional classifiers → 12 total)

Registers the following models via @register_model:
  XGBoost          — gradient boosting (xgboost library)
  ExtraTrees       — extremely randomised trees
  AdaBoost         — adaptive boosting
  HistGBT          — histogram gradient boosting (sklearn, no extra dep)
  MLP              — multi-layer perceptron
  RidgeClass       — ridge-regression classifier
  Bagging          — bagging of logistic regressors
  CatBoost         — categorical boosting (catboost library, optional)
"""

from __future__ import annotations

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

from .base import BaseModel
from .registry import register_model


# ─────────────────────────── shared helpers ─────────────────────────────── #

def _fit_guard(X, y, min_samples: int = 100) -> bool:
    return len(X) >= min_samples and len(np.unique(y)) >= 2


# ─────────────────────────── XGBoost ────────────────────────────────────── #

@register_model("XGBoost")
class XGBoostModel(BaseModel):
    """XGBoost gradient-boosting classifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        super().__init__("XGBoost")
        try:
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
        except ImportError:
            self.model = None

    def fit(self, X, y):
        if self.model is None or not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted or self.model is None:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)


# ─────────────────────────── ExtraTrees ─────────────────────────────────── #

@register_model("ExtraTrees")
class ExtraTreesModel(BaseModel):
    """Extremely Randomised Trees classifier."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 6):
        super().__init__("ExtraTrees")
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        if not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        probs = np.array([t.predict_proba(X_row)[0][1] for t in self.model.estimators_])
        return float(np.mean(probs)), float(np.var(probs)) + 1e-6


# ─────────────────────────── AdaBoost ───────────────────────────────────── #

@register_model("AdaBoost")
class AdaBoostModel(BaseModel):
    """AdaBoost classifier (SAMME.R algorithm)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.5):
        super().__init__("AdaBoost")
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            algorithm="SAMME",
        )

    def fit(self, X, y):
        if not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)


# ─────────────────────────── HistGBT ────────────────────────────────────── #

@register_model("HistGBT")
class HistGBTModel(BaseModel):
    """Histogram Gradient Boosting Trees (sklearn, no external dependency)."""

    def __init__(
        self,
        max_iter: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        l2_regularization: float = 0.5,
    ):
        super().__init__("HistGBT")
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            random_state=42,
        )

    def fit(self, X, y):
        if not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)


# ─────────────────────────── MLP ────────────────────────────────────────── #

@register_model("MLP")
class MLPModel(BaseModel):
    """Multi-Layer Perceptron classifier."""

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        alpha: float = 1e-3,
        max_iter: int = 500,
    ):
        super().__init__("MLP")
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def fit(self, X, y):
        if not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)


# ─────────────────────────── RidgeClass ─────────────────────────────────── #

@register_model("RidgeClass")
class RidgeClassModel(BaseModel):
    """Ridge regression classifier with probability calibration."""

    def __init__(self, alpha: float = 1.0):
        super().__init__("RidgeClass")
        self._raw = RidgeClassifier(alpha=alpha, class_weight="balanced")
        self.model = CalibratedClassifierCV(self._raw, method="sigmoid", cv=3)

    def fit(self, X, y):
        if not _fit_guard(X, y, min_samples=150):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)


# ─────────────────────────── Bagging ────────────────────────────────────── #

@register_model("Bagging")
class BaggingModel(BaseModel):
    """Bagging ensemble of Logistic Regression classifiers."""

    def __init__(self, n_estimators: int = 50, max_samples: float = 0.8):
        super().__init__("Bagging")
        self.model = BaggingClassifier(
            estimator=LogisticRegression(
                C=1.0, max_iter=500, solver="lbfgs", random_state=42
            ),
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        if not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        probs = np.array(
            [est.predict_proba(X_row)[0][1] for est in self.model.estimators_]
        )
        return float(np.mean(probs)), float(np.var(probs)) + 1e-6


# ─────────────────────────── CatBoost ───────────────────────────────────── #

@register_model("CatBoost")
class CatBoostModel(BaseModel):
    """CatBoost gradient-boosting classifier (catboost library, optional)."""

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 4,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
    ):
        super().__init__("CatBoost")
        try:
            from catboost import CatBoostClassifier

            self.model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=42,
                verbose=0,
            )
        except ImportError:
            self.model = None

    def fit(self, X, y):
        if self.model is None or not _fit_guard(X, y):
            return
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted or self.model is None:
            return 0.5, 0.25
        p = float(self.model.predict_proba(X_row)[0][1])
        return p, p * (1 - p)
