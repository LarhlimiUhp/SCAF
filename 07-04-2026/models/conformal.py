"""
SCAF Conformal Prediction Module
==================================
Provides distribution-free marginal coverage guarantees for the ensemble
prediction set, using Split Conformal Prediction (Venn–Abers / RAPS variant
for classification).

Theory recap
------------
Given a calibration set {(x_i, y_i)}_{i=1}^n and a user-specified
miscoverage level α ∈ (0,1):

  1. Compute nonconformity scores s_i = 1 − ŷ_i[y_i]  (label-conditional score)
  2. Choose threshold  τ = Quantile((1-α)(1 + 1/n); {s_i})
  3. For a new test point x*, the *prediction set* is
       C(x*) = {y : 1 − ŷ*(y) ≤ τ}
  
This guarantees  P(y* ∈ C(x*)) ≥ 1 − α  marginally over the joint
distribution of calibration + test samples.

For time-series data we implement **Adaptive Conformal Inference (ACI)**
(Gibbs & Candès 2021), which dynamically adjusts α_t to handle distribution
shift while maintaining long-run coverage.

Classes
-------
SplitConformalClassifier   – static split-CP for binary classification
AdaptiveConformalClassifier – ACI for non-i.i.d. sequential data
ConformalEnsemble           – wraps any BaseModel list with CP guarantees
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split Conformal Classifier
# ---------------------------------------------------------------------------

class SplitConformalClassifier:
    """
    Split Conformal Prediction for binary classification.

    Parameters
    ----------
    alpha : target miscoverage level (e.g. 0.05 → 95 % coverage guarantee)
    """

    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self._tau: Optional[float] = None
        self._calibration_scores: Optional[np.ndarray] = None
        self._n_calibration: int = 0
        self._empirical_coverage: Optional[float] = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, proba: np.ndarray, y_true: np.ndarray) -> float:
        """
        Fit the conformal threshold on held-out calibration data.

        Parameters
        ----------
        proba  : predicted probabilities of class 1, shape (n,)
        y_true : true binary labels, shape (n,)

        Returns
        -------
        τ  (the conformal threshold)
        """
        proba = np.asarray(proba, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()

        if len(proba) != len(y_true):
            raise ValueError("proba and y_true must have the same length.")

        # Nonconformity score: 1 − p̂(y_i)
        # For binary: if y_i=1 use proba_i, if y_i=0 use 1−proba_i
        scores = np.where(y_true == 1, 1.0 - proba, proba)
        self._calibration_scores = scores
        self._n_calibration = len(scores)

        # Adjusted quantile level q = ⌈(1-α)(1 + 1/n)⌉ / n
        q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / self._n_calibration))
        self._tau = float(np.quantile(scores, q_level, method="higher"))

        # Compute empirical coverage for diagnostics
        self._empirical_coverage = float(np.mean(scores <= self._tau))

        logger.info(
            "Conformal calibration: n=%d, α=%.3f, τ=%.4f, "
            "empirical_coverage=%.3f",
            self._n_calibration, self.alpha, self._tau, self._empirical_coverage,
        )
        return self._tau

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_set(self, proba: np.ndarray) -> np.ndarray:
        """
        Return a binary prediction set indicator for each sample.

        Shape: (n, 2) where column 0 = class 0 included, column 1 = class 1.
        """
        if self._tau is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        proba = np.asarray(proba, dtype=float).ravel()
        include_class1 = (1.0 - proba) <= self._tau   # score for label 1
        include_class0 = proba <= self._tau            # score for label 0
        return np.column_stack([include_class0, include_class1]).astype(int)

    def predict_with_uncertainty(
        self, proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        point_pred  : argmax hard prediction (n,)
        set_size    : size of the prediction set (1 or 2) (n,)
        is_certain  : True when the set contains exactly one label (n,)
        """
        pred_sets = self.predict_set(proba)
        point_pred = (proba >= 0.5).astype(int)
        set_size = pred_sets.sum(axis=1)
        is_certain = (set_size == 1)
        return point_pred, set_size, is_certain

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_report(self, proba: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute realised coverage and average set size on test data."""
        if self._tau is None:
            raise RuntimeError("Call calibrate() before coverage_report().")
        proba = np.asarray(proba, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        pred_sets = self.predict_set(proba)
        covered = np.array([pred_sets[i, int(y_true[i])] for i in range(len(y_true))])
        return {
            "coverage": float(covered.mean()),
            "target_coverage": 1.0 - self.alpha,
            "avg_set_size": float(pred_sets.sum(axis=1).mean()),
            "n_test": len(y_true),
            "tau": self._tau,
            "n_calibration": self._n_calibration,
        }

    @property
    def tau(self) -> Optional[float]:
        return self._tau

    @property
    def is_calibrated(self) -> bool:
        return self._tau is not None


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (ACI) for time series
# ---------------------------------------------------------------------------

class AdaptiveConformalClassifier:
    """
    Adaptive Conformal Inference for sequential (non-i.i.d.) data.

    Gibbs & Candès (2021): dynamically adjusts α_t after each step to
    correct for coverage violations and maintain long-run coverage ≥ 1−α.

    Parameters
    ----------
    alpha    : target miscoverage level
    gamma    : step-size for α update (Gibbs & Candès recommend 0.005–0.05)
    window   : rolling window for empirical coverage estimate
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.01,
        window: int = 200,
    ):
        self.alpha_target = alpha
        self.gamma = gamma
        self.window = window
        self._alpha_t = alpha           # dynamic α
        self._scores: List[float] = []  # calibration nonconformity scores
        self._coverage_history: List[int] = []
        self._tau_history: List[float] = []

    def update_calibration(self, score: float):
        """Append one new nonconformity score to the rolling calibration set."""
        self._scores.append(score)
        if len(self._scores) > self.window:
            self._scores.pop(0)

    def get_threshold(self) -> float:
        """Compute current τ_t from rolling calibration scores and α_t."""
        if not self._scores:
            return 0.5  # uninformative default
        q_level = min(1.0, (1.0 - self._alpha_t) * (1.0 + 1.0 / len(self._scores)))
        return float(np.quantile(self._scores, q_level, method="higher"))

    def observe_outcome(self, score: float, was_covered: bool):
        """
        After observing the true label, update α_t:
          α_{t+1} = α_t + γ · (α_target − 1{not covered})
        """
        err = 0.0 if was_covered else 1.0
        self._alpha_t = float(
            np.clip(self._alpha_t + self.gamma * (self.alpha_target - err), 1e-4, 0.5)
        )
        self._coverage_history.append(int(was_covered))
        self._tau_history.append(self.get_threshold())
        self.update_calibration(score)

    def step(
        self, proba: float, y_true: Optional[int] = None
    ) -> Tuple[List[int], float, bool]:
        """
        One inference step.

        Parameters
        ----------
        proba  : predicted probability of class 1
        y_true : observed label (if available, triggers ACI update)

        Returns
        -------
        pred_set   : list of labels in the prediction set (subset of {0,1})
        tau        : current conformal threshold
        is_certain : True if prediction set has exactly one label
        """
        tau = self.get_threshold()

        # Nonconformity scores for each possible label
        score_c1 = 1.0 - proba   # score if true label is 1
        score_c0 = proba          # score if true label is 0

        pred_set = []
        if score_c1 <= tau:
            pred_set.append(1)
        if score_c0 <= tau:
            pred_set.append(0)
        if not pred_set:  # ensure at least one prediction
            pred_set = [int(proba >= 0.5)]

        is_certain = len(pred_set) == 1

        # Update ACI if true label is observed
        if y_true is not None:
            obs_score = 1.0 - proba if y_true == 1 else proba
            covered = int(y_true) in pred_set
            self.observe_outcome(obs_score, covered)

        return pred_set, tau, is_certain

    @property
    def rolling_coverage(self) -> float:
        if not self._coverage_history:
            return float("nan")
        window = min(self.window, len(self._coverage_history))
        return float(np.mean(self._coverage_history[-window:]))

    @property
    def alpha_t(self) -> float:
        return self._alpha_t

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "alpha_target": self.alpha_target,
            "alpha_current": round(self._alpha_t, 5),
            "rolling_coverage": round(self.rolling_coverage, 4) if self._coverage_history else None,
            "n_observations": len(self._coverage_history),
            "current_tau": round(self.get_threshold(), 5),
        }


# ---------------------------------------------------------------------------
# Conformal Ensemble Wrapper
# ---------------------------------------------------------------------------

class ConformalEnsemble:
    """
    Wraps a list of SCAF BaseModel instances with conformal prediction.

    Usage
    -----
    1. `ce.calibrate(X_cal, y_cal)`  after training the base models on train fold
    2. `ce.predict(X_row)`           at inference time → returns (point_pred, pred_set, is_certain)
    """

    def __init__(
        self,
        models: List[Any],          # List[BaseModel]
        alpha: float = 0.05,
        use_adaptive: bool = True,
    ):
        self.models = models
        self.alpha = alpha
        self.use_adaptive = use_adaptive

        self._static_cp = SplitConformalClassifier(alpha=alpha)
        self._adaptive_cp = AdaptiveConformalClassifier(alpha=alpha) if use_adaptive else None

        self._is_calibrated = False

    # ------------------------------------------------------------------
    # Ensemble prediction helpers
    # ------------------------------------------------------------------

    def _ensemble_proba(self, X_row) -> float:
        """Simple confidence-weighted average over fitted models."""
        probas, confidences = [], []
        for m in self.models:
            if not m.is_fitted:
                continue
            try:
                p, unc = m.predict_proba_one(X_row)
                c = 1.0 / (unc + 1e-6)
                probas.append(p)
                confidences.append(c)
            except Exception:
                continue
        if not probas:
            return 0.5
        w = np.array(confidences)
        w = w / w.sum()
        return float(np.dot(probas, w))

    def _batch_ensemble_proba(self, X) -> np.ndarray:
        """Batch version for calibration."""
        return np.array([
            self._ensemble_proba(X[i : i + 1]) for i in range(len(X))
        ])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Compute conformal thresholds on the calibration fold.
        Call this *after* fitting all base models on the training fold.
        """
        proba_cal = self._batch_ensemble_proba(X_cal)
        self._static_cp.calibrate(proba_cal, y_cal)

        # Seed adaptive CP with calibration nonconformity scores
        if self._adaptive_cp is not None:
            for p, y in zip(proba_cal, y_cal):
                score = (1.0 - p) if y == 1 else p
                self._adaptive_cp.update_calibration(score)

        self._is_calibrated = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        X_row,
        y_true: Optional[int] = None,
    ) -> Tuple[int, List[int], bool, float]:
        """
        Parameters
        ----------
        X_row  : single sample (1 × features)
        y_true : observed label, used only for ACI update

        Returns
        -------
        point_pred   : hard prediction (0 or 1)
        pred_set     : conformal prediction set (subset of {0,1})
        is_certain   : True when |pred_set| = 1
        proba        : raw ensemble probability
        """
        proba = self._ensemble_proba(X_row)
        point_pred = int(proba >= 0.5)

        if not self._is_calibrated:
            return point_pred, [point_pred], True, proba

        if self.use_adaptive and self._adaptive_cp is not None:
            pred_set, _, is_certain = self._adaptive_cp.step(proba, y_true)
        else:
            _, _, is_certain_arr = self._static_cp.predict_with_uncertainty(
                np.array([proba])
            )
            pred_sets = self._static_cp.predict_set(np.array([proba]))
            pred_set = [c for c in range(2) if pred_sets[0, c] == 1]
            is_certain = bool(is_certain_arr[0])

        return point_pred, pred_set, is_certain, proba

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage_report(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate realised coverage on a test set."""
        if not self._is_calibrated:
            return {"error": "not calibrated"}
        proba_test = self._batch_ensemble_proba(X_test)
        report = self._static_cp.coverage_report(proba_test, y_test)
        if self._adaptive_cp is not None:
            report["aci"] = self._adaptive_cp.diagnostics()
        return report
