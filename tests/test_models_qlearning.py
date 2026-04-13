"""
Tests for models/qlearning_selector.py — helper functions and QLearningSelector.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "07-04-2026"))

from models.qlearning_selector import (
    QLearningSelector,
    _performance_bucket,
    _state_index,
    _compute_reward,
    _build_action_space,
    REGIMES,
    PERF_BUCKETS,
    N_STATES,
)


# ─── helpers ──────────────────────────────────────────────────────────────── #

_MODEL_NAMES = ["LogReg-L2", "RandomForest", "LightGBM", "GBM"]


# ─── _performance_bucket ──────────────────────────────────────────────────── #

class TestPerformanceBucket:
    def test_low_sharpe(self):
        assert _performance_bucket(-1.0) == "low"

    def test_high_sharpe(self):
        assert _performance_bucket(1.0) == "high"

    def test_mid_sharpe(self):
        assert _performance_bucket(0.0) == "mid"

    def test_boundary_low(self):
        assert _performance_bucket(-0.5) == "mid"   # exactly -0.5 → mid (not < -0.5)

    def test_boundary_high(self):
        assert _performance_bucket(0.5) == "mid"    # exactly 0.5 → mid (not > 0.5)


# ─── _state_index ─────────────────────────────────────────────────────────── #

class TestStateIndex:
    def test_returns_int(self):
        idx = _state_index("bull", 1.0)
        assert isinstance(idx, int)

    def test_range(self):
        for regime in REGIMES:
            for sharpe in [-1.0, 0.0, 1.0]:
                idx = _state_index(regime, sharpe)
                assert 0 <= idx < N_STATES

    def test_unknown_regime_defaults_to_sideways(self):
        idx_unknown = _state_index("unknown", 0.0)
        idx_sideways = _state_index("sideways", 0.0)
        assert idx_unknown == idx_sideways


# ─── _compute_reward ──────────────────────────────────────────────────────── #

class TestComputeReward:
    def test_positive_return_positive_reward(self):
        r = _compute_reward(ret=0.01, vol=0.001, drawdown=0.0)
        assert r > 0

    def test_negative_return_negative_reward(self):
        r = _compute_reward(ret=-0.01, vol=0.001, drawdown=0.0)
        assert r < 0

    def test_large_drawdown_penalises(self):
        r_no_dd = _compute_reward(ret=0.01, vol=0.001, drawdown=0.0)
        r_dd = _compute_reward(ret=0.01, vol=0.001, drawdown=1.0)
        assert r_dd < r_no_dd


# ─── _build_action_space ──────────────────────────────────────────────────── #

class TestBuildActionSpace:
    def test_not_empty(self):
        actions = _build_action_space(_MODEL_NAMES, max_subset_size=2)
        assert len(actions) > 0

    def test_max_subset_size_respected(self):
        actions = _build_action_space(_MODEL_NAMES, max_subset_size=2)
        for subset in actions:
            assert len(subset) <= 2

    def test_no_empty_subsets(self):
        actions = _build_action_space(_MODEL_NAMES, max_subset_size=3)
        for subset in actions:
            assert len(subset) >= 1

    def test_single_model(self):
        actions = _build_action_space(["A"], max_subset_size=1)
        assert actions == [("A",)]


# ─── QLearningSelector ────────────────────────────────────────────────────── #

class TestQLearningSelector:
    def _make_agent(self):
        return QLearningSelector(_MODEL_NAMES, max_subset_size=2)

    def test_instantiation(self):
        agent = self._make_agent()
        assert agent is not None

    def test_q_table_shape(self):
        agent = self._make_agent()
        q = agent.get_q_table()
        assert q.shape[0] == N_STATES
        assert q.shape[1] == len(agent._actions)

    def test_select_returns_list(self):
        agent = self._make_agent()
        action = agent.select(regime="bull", recent_sharpe=1.0)
        assert isinstance(action, list)
        assert all(name in _MODEL_NAMES for name in action)

    def test_select_all_regimes(self):
        agent = self._make_agent()
        for regime in REGIMES:
            for sharpe in [-1.0, 0.0, 1.0]:
                action = agent.select(regime=regime, recent_sharpe=sharpe)
                assert len(action) >= 1

    def test_observe_does_not_raise(self):
        agent = self._make_agent()
        agent.observe(
            state_regime="bull",
            state_sharpe=0.5,
            action_models=[_MODEL_NAMES[0]],
            ret=0.01,
            vol=0.005,
            drawdown=0.02,
            next_regime="bull",
            next_sharpe=0.6,
        )

    def test_epsilon_decreases_after_decay(self):
        agent = self._make_agent()
        eps_before = agent.epsilon
        agent._decay_epsilon()
        assert agent.epsilon <= eps_before

    def test_epsilon_minimum_enforced(self):
        agent = self._make_agent()
        for _ in range(10_000):
            agent._decay_epsilon()
        assert agent.epsilon >= agent.epsilon_min
