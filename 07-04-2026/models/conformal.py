"""
Conformal Prediction for SCAF-LS

Three classes:
  SplitConformalClassifier
      Marginal coverage ≥ 1−α guaranteed by split-conformal calibration.

  AdaptiveConformalClassifier (ACI)
      Online α_t adaptation for non-i.i.d. time-series
      (Gibbs & Candès 2021, "Adaptive Conformal Inference Under Distribution Shift").

  ConformalEnsemble
      Wraps any list of BaseModel instances; calibrates with CP and
      returns prediction sets at inference time.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────── SplitConformalClassifier ───────────────────── #

class SplitConformalClassifier:
    """Split-conformal classifier wrapper.

    Workflow
    --------
    1. ``fit(X_train, y_train)``   — train the underlying model.
    2. ``calibrate(X_cal, y_cal)`` — compute non-conformity scores on a
       held-out calibration set and store the (1−α) quantile *q_hat*.
    3. ``predict_set(X_row)``      — return the prediction set
       {y : score(x, y) ≤ q_hat}.

    Marginal coverage guarantee: P(Y ∈ C(X)) ≥ 1−α.

    Parameters
    ----------
    base_model:
        Any SCAF-LS BaseModel instance.
    alpha:
        Miscoverage level (default 0.05 → 95 % coverage).
    """

    def __init__(self, base_model: BaseModel, alpha: float = 0.05):
        self.base_model = base_model
        self.alpha = alpha
        self._q_hat: Optional[float] = None
        self._calibrated = False

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplitConformalClassifier":
        self.base_model.fit(X, y)
        return self

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> "SplitConformalClassifier":
        """Compute conformal non-conformity scores on calibration data."""
        n = len(y_cal)
        scores = np.empty(n)
        for i in range(n):
            p, _ = self.base_model.predict_proba_one(X_cal[i : i + 1])
            # Non-conformity score: 1 − P(true class)
            true_prob = p if y_cal[i] == 1 else 1.0 - p
            scores[i] = 1.0 - true_prob

        # Empirical (1−α) quantile with finite-sample correction
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        self._calibrated = True

        # Empirical coverage check
        covered = int(np.sum(scores <= self._q_hat))
        empirical_cov = covered / n
        logger.info(
            "SplitConformal calibrated: α=%.3f, q_hat=%.4f, "
            "empirical_coverage=%.3f (n_cal=%d)",
            self.alpha,
            self._q_hat,
            empirical_cov,
            n,
        )
        return self

    def predict_set(self, X_row: np.ndarray) -> List[int]:
        """Return the prediction set {0, 1} for *X_row*.

        A label *y* is included when its non-conformity score ≤ q_hat.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_set().")
        p, _ = self.base_model.predict_proba_one(X_row)
        prediction_set = []
        for label in (0, 1):
            true_prob = p if label == 1 else 1.0 - p
            score = 1.0 - true_prob
            if score <= self._q_hat:
                prediction_set.append(label)
        return prediction_set

    def predict_proba_one(self, X_row: np.ndarray) -> Tuple[float, float]:
        """Delegate to the underlying model for point predictions."""
        return self.base_model.predict_proba_one(X_row)

    @property
    def q_hat(self) -> Optional[float]:
        return self._q_hat


# ─────────────────────────── AdaptiveConformalClassifier ────────────────── #

class AdaptiveConformalClassifier:
    """Adaptive Conformal Inference (ACI) for time-series classification.

    Adapts the miscoverage level α_t online to maintain valid coverage
    despite distribution shift, following:

        Gibbs, I. & Candès, E. (2021). "Adaptive Conformal Inference Under
        Distribution Shift."  NeurIPS 2021.

    Update rule:
        α_{t+1} = α_t + γ · (α − err_t)
    where err_t = 1 if y_t ∉ C_t(x_t) else 0.

    Parameters
    ----------
    base_model:
        Any SCAF-LS BaseModel instance.
    alpha:
        Target miscoverage level.
    gamma:
        Step-size for the online update (default 0.005).
    alpha_min, alpha_max:
        Clipping bounds for α_t (default [0.01, 0.5]).
    """

    def __init__(
        self,
        base_model: BaseModel,
        alpha: float = 0.05,
        gamma: float = 0.005,
        alpha_min: float = 0.01,
        alpha_max: float = 0.50,
    ):
        self.base_model = base_model
        self.alpha = alpha          # target coverage
        self.alpha_t = alpha        # current (adaptive) α
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self._scores: List[float] = []   # calibration scores seen so far
        self._errors: List[int] = []     # 1 if missed, 0 if covered

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveConformalClassifier":
        self.base_model.fit(X, y)
        return self

    def update(self, X_row: np.ndarray, y_true: int) -> None:
        """Observe a new labelled point, update scores and adapt α_t.

        This is called *after* making a prediction so that we can measure err_t.
        """
        p, _ = self.base_model.predict_proba_one(X_row)
        true_prob = p if y_true == 1 else 1.0 - p
        score = 1.0 - true_prob
        self._scores.append(score)

        # Compute q_hat from current scores at level α_t
        q_hat = self._current_q_hat()
        err_t = 0 if score <= q_hat else 1
        self._errors.append(err_t)

        # ACI update: α_{t+1} = clip(α_t + γ·(α − err_t))
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.alpha - err_t),
                self.alpha_min,
                self.alpha_max,
            )
        )

    def predict_set(self, X_row: np.ndarray) -> List[int]:
        """Return the adaptive prediction set for *X_row*."""
        if not self._scores:
            return [0, 1]  # Before any calibration, return full set

        p, _ = self.base_model.predict_proba_one(X_row)
        q_hat = self._current_q_hat()
        prediction_set = []
        for label in (0, 1):
            true_prob = p if label == 1 else 1.0 - p
            if (1.0 - true_prob) <= q_hat:
                prediction_set.append(label)
        return prediction_set

    def predict_proba_one(self, X_row: np.ndarray) -> Tuple[float, float]:
        return self.base_model.predict_proba_one(X_row)

    def empirical_coverage(self) -> float:
        """Fraction of steps where the true label was in the prediction set."""
        if not self._errors:
            return float("nan")
        return 1.0 - float(np.mean(self._errors))

    def _current_q_hat(self) -> float:
        n = len(self._scores)
        if n == 0:
            return 1.0
        level = np.ceil((n + 1) * (1.0 - self.alpha_t)) / n
        level = min(level, 1.0)
        return float(np.quantile(self._scores, level))


# ─────────────────────────── ConformalEnsemble ──────────────────────────── #

class ConformalEnsemble:
    """Conformal wrapper around a list of BaseModel instances.

    Aggregates point predictions from all models via soft vote, then
    calibrates a single SplitConformalClassifier on their averaged output.

    Parameters
    ----------
    models:
        List of fitted (or unfitted) BaseModel instances.
    alpha:
        Marginal miscoverage level for calibration.
    """

    def __init__(self, models: List[BaseModel], alpha: float = 0.05):
        if not models:
            raise ValueError("models list must be non-empty.")
        self.models = models
        self.alpha = alpha
        self._q_hat: Optional[float] = None
        self._calibrated = False

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalEnsemble":
        for model in self.models:
            model.fit(X, y)
        return self

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> "ConformalEnsemble":
        """Calibrate the ensemble on a held-out set."""
        n = len(y_cal)
        scores = np.empty(n)
        for i in range(n):
            p_mean = self._ensemble_proba(X_cal[i : i + 1])
            true_prob = p_mean if y_cal[i] == 1 else 1.0 - p_mean
            scores[i] = 1.0 - true_prob

        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        self._calibrated = True

        empirical_cov = float(np.mean(scores <= self._q_hat))
        logger.info(
            "ConformalEnsemble calibrated: n_models=%d, α=%.3f, q_hat=%.4f, "
            "empirical_coverage=%.3f",
            len(self.models),
            self.alpha,
            self._q_hat,
            empirical_cov,
        )
        return self

    def predict_set(self, X_row: np.ndarray) -> List[int]:
        """Return the prediction set for *X_row*."""
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_set().")
        p = self._ensemble_proba(X_row)
        prediction_set = []
        for label in (0, 1):
            true_prob = p if label == 1 else 1.0 - p
            if (1.0 - true_prob) <= self._q_hat:
                prediction_set.append(label)
        return prediction_set

    def predict_proba_one(self, X_row: np.ndarray) -> Tuple[float, float]:
        """Ensemble point prediction (mean probability, mean variance)."""
        probs = []
        variances = []
        for model in self.models:
            p, v = model.predict_proba_one(X_row)
            probs.append(p)
            variances.append(v)
        p_mean = float(np.mean(probs))
        # Total variance = mean of individual variances + variance of means
        v_total = float(np.mean(variances) + np.var(probs))
        return p_mean, v_total

    @property
    def q_hat(self) -> Optional[float]:
        return self._q_hat

    # ------------------------------------------------------------------ #

    def _ensemble_proba(self, X_row: np.ndarray) -> float:
        """Compute the mean probability across fitted models."""
        probs = []
        for model in self.models:
            if model.is_fitted:
                p, _ = model.predict_proba_one(X_row)
                probs.append(p)
        if not probs:
            return 0.5
        return float(np.mean(probs))


# ─────────────────────── AdaptiveConformalEnsemble ──────────────────────── #

class AdaptiveConformalEnsemble:
    """Ensemble conformal predictor with online ACI adaptation for time series.

    Combines split-conformal seeding (from a held-out calibration set) with
    the ACI online update rule to maintain marginal coverage ≥ 1−α even
    under distribution shift and non-stationarity.

    Online update rule (Gibbs & Candès 2021):
        α_{t+1} = clip(α_t + γ·(α − err_t), α_min, α_max)
    where err_t = 1 if y_t ∉ Ĉ_t(x_t) else 0.

    Parameters
    ----------
    models:
        List of fitted BaseModel instances.
    alpha:
        Target miscoverage level (default 0.05 → 95 % coverage).
    gamma:
        Step-size for the online α update (default 0.005).
    alpha_min, alpha_max:
        Clipping bounds for α_t.
    """

    def __init__(
        self,
        models: List[BaseModel],
        alpha: float = 0.05,
        gamma: float = 0.005,
        alpha_min: float = 0.005,
        alpha_max: float = 0.50,
    ):
        if not models:
            raise ValueError("models list must be non-empty.")
        self.models = models
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.alpha_t = alpha          # current adaptive miscoverage level
        self._scores: List[float] = []  # all non-conformity scores seen so far
        self._errors: List[int] = []    # 1=miss, 0=covered
        self._q_hat: Optional[float] = None
        self._calibrated: bool = False

    # ------------------------------------------------------------------ #

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> "AdaptiveConformalEnsemble":
        """Seed the running score buffer from the calibration set.

        This gives the predictor a warm-start so that q_hat is meaningful
        from the very first validation step.
        """
        n = len(y_cal)
        for i in range(n):
            p = self._ensemble_proba(X_cal[i : i + 1])
            true_prob = p if y_cal[i] == 1 else 1.0 - p
            self._scores.append(1.0 - true_prob)

        self._q_hat = self._compute_q_hat()
        self._calibrated = True

        empirical_cov = float(np.mean(np.array(self._scores) <= self._q_hat))
        logger.info(
            "AdaptiveConformalEnsemble seeded: n_cal=%d, α_0=%.3f, "
            "q_hat=%.4f, cal_coverage=%.3f",
            n, self.alpha_t, self._q_hat, empirical_cov,
        )
        return self

    def predict_set(self, X_row: np.ndarray) -> List[int]:
        """Return the adaptive prediction set for *X_row* (no state update)."""
        q = self._q_hat if self._q_hat is not None else 1.0
        p = self._ensemble_proba(X_row)
        result = []
        for label in (0, 1):
            true_prob = p if label == 1 else 1.0 - p
            if (1.0 - true_prob) <= q:
                result.append(label)
        return result

    def predict_set_and_update(
        self, X_row: np.ndarray, y_true: int
    ) -> Tuple[List[int], bool]:
        """Predict, observe the true label, update ACI state.

        Returns
        -------
        prediction_set : List[int]
            The prediction set *before* observing y_true.
        covered : bool
            Whether y_true ∈ prediction_set.
        """
        pred_set = self.predict_set(X_row)

        # Compute non-conformity score for this sample
        p = self._ensemble_proba(X_row)
        true_prob = p if y_true == 1 else 1.0 - p
        score = 1.0 - true_prob

        # Check coverage, record error
        covered = y_true in pred_set
        err_t = 0 if covered else 1
        self._errors.append(err_t)
        self._scores.append(score)

        # ACI alpha update: α_{t+1} = clip(α_t + γ·(α − err_t))
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.alpha - err_t),
                self.alpha_min,
                self.alpha_max,
            )
        )

        # Recompute q_hat with updated alpha_t and new score included
        self._q_hat = self._compute_q_hat()

        return pred_set, covered

    def empirical_coverage(self) -> float:
        """Fraction of *validation* steps where the true label was covered."""
        if not self._errors:
            return float("nan")
        return 1.0 - float(np.mean(self._errors))

    @property
    def q_hat(self) -> Optional[float]:
        return self._q_hat

    # ------------------------------------------------------------------ #

    def _compute_q_hat(self) -> float:
        n = len(self._scores)
        if n == 0:
            return 1.0
        level = min(np.ceil((n + 1) * (1.0 - self.alpha_t)) / n, 1.0)
        return float(np.quantile(self._scores, level))

    def _ensemble_proba(self, X_row: np.ndarray) -> float:
        probs = []
        for model in self.models:
            if model.is_fitted:
                p, _ = model.predict_proba_one(X_row)
                probs.append(p)
        return float(np.mean(probs)) if probs else 0.5
