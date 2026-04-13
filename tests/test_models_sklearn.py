"""
Tests for models/sklearn_models.py — LogRegL2Model and RFClassModel.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "07-04-2026"))

from models.sklearn_models import LogRegL2Model, RFClassModel


# ─── helpers ──────────────────────────────────────────────────────────────── #

def _make_dataset(n: int = 200, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] > 0).astype(int)          # linearly separable target
    return X, y


# ─── LogRegL2Model ────────────────────────────────────────────────────────── #

class TestLogRegL2Model:
    def test_instantiation(self):
        m = LogRegL2Model()
        assert m.name == "LogReg-L2"
        assert not m.is_fitted

    def test_fit_sets_fitted(self):
        X, y = _make_dataset()
        m = LogRegL2Model()
        m.fit(X, y)
        assert m.is_fitted

    def test_predict_proba_one_range(self):
        X, y = _make_dataset()
        m = LogRegL2Model()
        m.fit(X, y)
        prob, unc = m.predict_proba_one(X[:1])
        assert 0.0 <= prob <= 1.0
        assert unc >= 0.0

    def test_predict_proba_one_unfitted_returns_default(self):
        m = LogRegL2Model()
        prob, unc = m.predict_proba_one(np.zeros((1, 5)))
        assert prob == 0.5
        assert unc == 0.25

    def test_fit_too_few_samples_no_crash(self):
        """fit() with < 100 samples must not raise."""
        m = LogRegL2Model()
        X = np.zeros((50, 3))
        y = np.zeros(50)
        m.fit(X, y)           # should be a no-op
        assert not m.is_fitted

    def test_fit_single_class_no_crash(self):
        """fit() when y contains a single class must not raise."""
        m = LogRegL2Model()
        X = np.ones((200, 3))
        y = np.zeros(200)
        m.fit(X, y)           # should be a no-op
        assert not m.is_fitted


# ─── RFClassModel ─────────────────────────────────────────────────────────── #

class TestRFClassModel:
    def test_instantiation(self):
        m = RFClassModel()
        assert m.name == "RandomForest"
        assert not m.is_fitted

    def test_fit_sets_fitted(self):
        X, y = _make_dataset(n=300)
        m = RFClassModel(n_est=10, max_depth=3)   # small for test speed
        m.fit(X, y)
        assert m.is_fitted

    def test_predict_proba_one_range(self):
        X, y = _make_dataset(n=300)
        m = RFClassModel(n_est=10, max_depth=3)
        m.fit(X, y)
        prob, unc = m.predict_proba_one(X[:1])
        assert 0.0 <= prob <= 1.0
        assert unc >= 0.0

    def test_predict_proba_one_unfitted_returns_default(self):
        m = RFClassModel()
        prob, unc = m.predict_proba_one(np.zeros((1, 5)))
        assert prob == 0.5
        assert unc == 0.25

    def test_fit_too_few_samples_no_crash(self):
        m = RFClassModel()
        X = np.zeros((100, 3))
        y = np.zeros(100)
        m.fit(X, y)
        assert not m.is_fitted
