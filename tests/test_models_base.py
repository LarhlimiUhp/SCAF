"""
Tests for models/base.py — BaseModel and ModelRegistry.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "07-04-2026"))

from models.base import BaseModel, ModelRegistry


# ─── minimal concrete implementation ──────────────────────────────────────── #

class _AlwaysHalfModel(BaseModel):
    """Trivial model that always returns probability 0.5."""

    def fit(self, X, y):
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        return 0.5, 0.25


# ─── BaseModel ────────────────────────────────────────────────────────────── #

class TestBaseModel:
    def test_name_stored(self):
        m = _AlwaysHalfModel(name="test_model")
        assert m.name == "test_model"

    def test_not_fitted_initially(self):
        m = _AlwaysHalfModel(name="m")
        assert m.is_fitted is False

    def test_fit_sets_fitted(self):
        m = _AlwaysHalfModel(name="m")
        m.fit(np.zeros((10, 2)), np.zeros(10))
        assert m.is_fitted is True

    def test_predict_proba_one_returns_tuple(self):
        m = _AlwaysHalfModel(name="m")
        result = m.predict_proba_one(np.zeros((1, 2)))
        assert len(result) == 2

    def test_predict_signal_zero_for_half(self):
        m = _AlwaysHalfModel(name="m")
        signal, _ = m.predict_signal(np.zeros((1, 2)))
        assert abs(signal) < 1e-9

    def test_abstract_fit_not_callable_directly(self):
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel(name="abstract")  # type: ignore[abstract]


# ─── ModelRegistry ────────────────────────────────────────────────────────── #

class TestModelRegistry:
    def _make_registry(self):
        reg = ModelRegistry()
        reg.register("half", lambda: _AlwaysHalfModel("half"))
        return reg

    def test_register_and_names(self):
        reg = self._make_registry()
        assert "half" in reg.names()

    def test_create_known_model(self):
        reg = self._make_registry()
        m = reg.create("half")
        assert isinstance(m, _AlwaysHalfModel)

    def test_create_unknown_raises(self):
        reg = self._make_registry()
        with pytest.raises(KeyError):
            reg.create("unknown_model")

    def test_all_returns_dict(self):
        reg = self._make_registry()
        d = reg.all()
        assert isinstance(d, dict)
        assert "half" in d

    def test_all_is_copy(self):
        """Modifying the returned dict must not affect the registry."""
        reg = self._make_registry()
        d = reg.all()
        d["injected"] = None
        assert "injected" not in reg.names()
