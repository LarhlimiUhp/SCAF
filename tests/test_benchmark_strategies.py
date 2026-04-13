"""
Tests for benchmark/strategies.py — BenchmarkStrategies class.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make the SCAF-LS package importable from the test suite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "07-04-2026"))

from benchmark.strategies import BenchmarkStrategies


# ─── helpers ──────────────────────────────────────────────────────────────── #

def _make_prices(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, name="price")


# ─── buy_and_hold ─────────────────────────────────────────────────────────── #

class TestBuyAndHold:
    def test_returns_series(self):
        prices = _make_prices()
        result = BenchmarkStrategies.buy_and_hold(prices)
        assert isinstance(result, pd.Series)

    def test_same_length(self):
        prices = _make_prices()
        result = BenchmarkStrategies.buy_and_hold(prices)
        assert len(result) == len(prices)

    def test_no_nan(self):
        prices = _make_prices()
        result = BenchmarkStrategies.buy_and_hold(prices)
        assert not result.isna().any()


# ─── momentum ─────────────────────────────────────────────────────────────── #

class TestMomentum:
    def test_returns_series(self):
        prices = _make_prices()
        result = BenchmarkStrategies.momentum(prices)
        assert isinstance(result, pd.Series)

    def test_same_length(self):
        prices = _make_prices()
        result = BenchmarkStrategies.momentum(prices)
        assert len(result) == len(prices)

    def test_positions_bounded(self):
        prices = _make_prices()
        # Positions are 0 or 1, so strategy returns are within log_return bounds
        result = BenchmarkStrategies.momentum(prices)
        log_ret = np.log(prices / prices.shift(1)).fillna(0)
        assert (result.abs() <= log_ret.abs() + 1e-10).all()


# ─── mean_reversion ───────────────────────────────────────────────────────── #

class TestMeanReversion:
    def test_returns_series(self):
        prices = _make_prices()
        result = BenchmarkStrategies.mean_reversion(prices)
        assert isinstance(result, pd.Series)

    def test_same_length(self):
        prices = _make_prices()
        result = BenchmarkStrategies.mean_reversion(prices)
        assert len(result) == len(prices)


# ─── sma_crossover ────────────────────────────────────────────────────────── #

class TestSmaCrossover:
    def test_returns_series(self):
        prices = _make_prices(n=300)
        result = BenchmarkStrategies.sma_crossover(prices, fast=10, slow=30)
        assert isinstance(result, pd.Series)

    def test_same_length(self):
        prices = _make_prices(n=300)
        result = BenchmarkStrategies.sma_crossover(prices, fast=10, slow=30)
        assert len(result) == len(prices)


# ─── all_strategies ───────────────────────────────────────────────────────── #

class TestAllStrategies:
    def test_returns_dict(self):
        strategies = BenchmarkStrategies.all_strategies()
        assert isinstance(strategies, dict)

    def test_fifteen_strategies(self):
        strategies = BenchmarkStrategies.all_strategies()
        assert len(strategies) == 15

    def test_all_callables(self):
        strategies = BenchmarkStrategies.all_strategies()
        for name, fn in strategies.items():
            assert callable(fn), f"Strategy '{name}' is not callable"

    def test_all_produce_series(self):
        prices = _make_prices(n=300)
        strategies = BenchmarkStrategies.all_strategies()
        for name, fn in strategies.items():
            result = fn(prices)
            assert isinstance(result, pd.Series), f"Strategy '{name}' did not return a Series"
            assert len(result) == len(prices), f"Strategy '{name}' returned wrong length"
