"""
Tests for models/conformal.py — SplitConformalClassifier.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "07-04-2026"))

from models.conformal import SplitConformalClassifier
from models.sklearn_models import LogRegL2Model


# ─── helpers ──────────────────────────────────────────────────────────────── #

def _make_dataset(n: int = 300, n_features: int = 4, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _fitted_conformal(alpha: float = 0.1):
    X, y = _make_dataset(n=400)
    n_train = 200
    clf = SplitConformalClassifier(LogRegL2Model(), alpha=alpha)
    clf.fit(X[:n_train], y[:n_train])
    clf.calibrate(X[n_train:], y[n_train:])
    return clf, X


# ─── SplitConformalClassifier ─────────────────────────────────────────────── #

class TestSplitConformalClassifier:
    def test_instantiation(self):
        clf = SplitConformalClassifier(LogRegL2Model())
        assert clf.alpha == 0.05
        assert clf.q_hat is None

    def test_custom_alpha(self):
        clf = SplitConformalClassifier(LogRegL2Model(), alpha=0.2)
        assert clf.alpha == 0.2

    def test_fit_does_not_raise(self):
        X, y = _make_dataset()
        clf = SplitConformalClassifier(LogRegL2Model())
        clf.fit(X[:200], y[:200])

    def test_calibrate_sets_q_hat(self):
        clf, _ = _fitted_conformal()
        assert clf.q_hat is not None
        assert 0.0 <= clf.q_hat <= 1.0

    def test_predict_set_before_calibrate_raises(self):
        X, y = _make_dataset()
        clf = SplitConformalClassifier(LogRegL2Model())
        clf.fit(X[:200], y[:200])
        with pytest.raises(RuntimeError):
            clf.predict_set(X[200:201])

    def test_predict_set_returns_list_of_labels(self):
        clf, X = _fitted_conformal()
        pred_set = clf.predict_set(X[300:301])
        assert isinstance(pred_set, list)
        assert all(label in (0, 1) for label in pred_set)

    def test_predict_set_not_empty(self):
        """For a well-calibrated model the prediction set should rarely be empty."""
        clf, X = _fitted_conformal(alpha=0.05)
        non_empty = sum(
            1 for i in range(300, 400)
            if len(clf.predict_set(X[i : i + 1])) > 0
        )
        assert non_empty >= 90   # at least 90 % non-empty

    def test_marginal_coverage(self):
        """Empirical coverage must be ≥ 1 − alpha on the calibration set."""
        X, y = _make_dataset(n=600)
        alpha = 0.1
        clf = SplitConformalClassifier(LogRegL2Model(), alpha=alpha)
        clf.fit(X[:200], y[:200])
        clf.calibrate(X[200:400], y[200:400])

        covered = 0
        n_test = 200
        for i in range(400, 600):
            pred_set = clf.predict_set(X[i : i + 1])
            if y[i] in pred_set:
                covered += 1
        coverage = covered / n_test
        assert coverage >= 1.0 - alpha - 0.05  # small tolerance for finite samples
