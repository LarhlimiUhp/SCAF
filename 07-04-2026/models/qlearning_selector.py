"""
Q-Learning Model Selector for SCAF-LS

Tabular Q-learning that learns which subset of SCAF-LS models to activate
for a given market state.

State space  : 12 states = 4 regimes × 3 performance buckets
Action space : all subsets of registered model names (pruned to N actions)
Update rule  : Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,a′) − Q(s,a)]
Reward       : normalised_return / volatility − λ·drawdown²
Exploration  : ε-greedy with exponential decay (ε_min = 0.05)
Data         : experience replay with mini-batch updates
Persistence  : save/load Q-table to/from JSON
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from itertools import combinations
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────── constants ──────────────────────────────────── #

REGIMES = ["bull", "bear", "sideways", "crisis"]          # 4 regimes
PERF_BUCKETS = ["low", "mid", "high"]                     # 3 perf buckets
N_STATES = len(REGIMES) * len(PERF_BUCKETS)               # 12

_REGIME_IDX = {r: i for i, r in enumerate(REGIMES)}
_BUCKET_IDX = {b: i for i, b in enumerate(PERF_BUCKETS)}


# ─────────────────────────── helpers ────────────────────────────────────── #

def _performance_bucket(recent_sharpe: float) -> str:
    """Map a rolling Sharpe ratio to a performance bucket."""
    if recent_sharpe < -0.5:
        return "low"
    if recent_sharpe > 0.5:
        return "high"
    return "mid"


def _state_index(regime: str, recent_sharpe: float) -> int:
    r_idx = _REGIME_IDX.get(regime, 2)          # default: sideways
    b_idx = _BUCKET_IDX[_performance_bucket(recent_sharpe)]
    return r_idx * len(PERF_BUCKETS) + b_idx


def _compute_reward(
    ret: float,
    vol: float,
    drawdown: float,
    lambda_dd: float = 0.5,
) -> float:
    """Sharpe-like reward penalised by squared drawdown.

    r = (ret / (vol + 1e-8)) − λ · drawdown²
    """
    normalised_ret = ret / (vol + 1e-8)
    return normalised_ret - lambda_dd * drawdown ** 2


def _build_action_space(model_names: List[str], max_subset_size: int = 4) -> List[Tuple[str, ...]]:
    """Return all non-empty subsets of *model_names* up to *max_subset_size*."""
    actions: List[Tuple[str, ...]] = []
    for size in range(1, min(max_subset_size, len(model_names)) + 1):
        for subset in combinations(model_names, size):
            actions.append(subset)
    return actions


# ──────────────────────────── QLearningSelector ─────────────────────────── #

class QLearningSelector:
    """Tabular Q-learning agent that selects which SCAF-LS models to activate.

    Parameters
    ----------
    model_names:
        All model names available in the registry.
    alpha:
        Learning rate (default 0.1).
    gamma:
        Discount factor (default 0.9).
    epsilon_start:
        Initial exploration rate (default 1.0).
    epsilon_min:
        Minimum exploration rate (default 0.05).
    epsilon_decay:
        Multiplicative decay applied after each episode (default 0.995).
    lambda_dd:
        Drawdown penalty coefficient in reward function.
    replay_capacity:
        Maximum experiences stored in the replay buffer.
    batch_size:
        Mini-batch size for experience replay updates.
    max_subset_size:
        Maximum number of models in a single action (limits action space).
    """

    def __init__(
        self,
        model_names: List[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lambda_dd: float = 0.5,
        replay_capacity: int = 2000,
        batch_size: int = 32,
        max_subset_size: int = 4,
    ):
        self.model_names = list(model_names)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lambda_dd = lambda_dd
        self.batch_size = batch_size

        # Build finite action space
        self._actions: List[Tuple[str, ...]] = _build_action_space(
            self.model_names, max_subset_size
        )
        n_actions = len(self._actions)

        # Q-table: shape (N_STATES, n_actions), initialised optimistically
        self._q: np.ndarray = np.zeros((N_STATES, n_actions), dtype=np.float64)

        # Experience replay buffer: (state, action_idx, reward, next_state)
        self._replay: Deque[Tuple[int, int, float, int]] = deque(
            maxlen=replay_capacity
        )

        logger.info(
            "QLearningSelector: %d states × %d actions",
            N_STATES,
            n_actions,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def select(self, regime: str, recent_sharpe: float) -> List[str]:
        """Return the model subset to activate (ε-greedy policy).

        Call this at inference time.
        """
        state = _state_index(regime, recent_sharpe)
        if random.random() < self.epsilon:
            action_idx = random.randrange(len(self._actions))
        else:
            action_idx = int(np.argmax(self._q[state]))
        return list(self._actions[action_idx])

    def observe(
        self,
        state_regime: str,
        state_sharpe: float,
        action_models: List[str],
        ret: float,
        vol: float,
        drawdown: float,
        next_regime: str,
        next_sharpe: float,
    ) -> None:
        """Store a transition and trigger a mini-batch update.

        Parameters
        ----------
        state_regime, state_sharpe  : current state descriptor
        action_models               : list of model names that were active
        ret                         : realised portfolio return for the step
        vol                         : realised volatility for the step
        drawdown                    : current drawdown (positive value)
        next_regime, next_sharpe    : next state descriptor
        """
        state = _state_index(state_regime, state_sharpe)
        next_state = _state_index(next_regime, next_sharpe)
        action_idx = self._action_index(tuple(sorted(action_models)))
        reward = _compute_reward(ret, vol, drawdown, self.lambda_dd)

        self._replay.append((state, action_idx, reward, next_state))
        self._replay_update()
        self._decay_epsilon()

    def get_q_table(self) -> np.ndarray:
        """Return a copy of the Q-table."""
        return self._q.copy()

    def save(self, filepath: str) -> None:
        """Persist the Q-table and hyper-parameters to a JSON file."""
        data = {
            "q_table": self._q.tolist(),
            "actions": [list(a) for a in self._actions],
            "model_names": self.model_names,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "lambda_dd": self.lambda_dd,
            "batch_size": self.batch_size,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Q-table saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "QLearningSelector":
        """Load a previously saved Q-table."""
        with open(filepath) as fh:
            data = json.load(fh)
        instance = cls(
            model_names=data["model_names"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
            lambda_dd=data["lambda_dd"],
            batch_size=data["batch_size"],
        )
        instance._q = np.array(data["q_table"])
        instance._actions = [tuple(a) for a in data["actions"]]
        logger.info("Q-table loaded from %s", filepath)
        return instance

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _action_index(self, action_tuple: Tuple[str, ...]) -> int:
        """Map a sorted tuple of model names to its action index."""
        try:
            return self._actions.index(action_tuple)
        except ValueError:
            # Unknown subset — map to the first action
            return 0

    def _replay_update(self) -> None:
        """Sample a mini-batch from the replay buffer and perform Bellman updates."""
        if len(self._replay) < self.batch_size:
            return

        batch = random.sample(self._replay, self.batch_size)
        for state, action_idx, reward, next_state in batch:
            q_current = self._q[state, action_idx]
            q_next_max = float(np.max(self._q[next_state]))
            # Bellman update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)]
            td_target = reward + self.gamma * q_next_max
            td_error = td_target - q_current
            self._q[state, action_idx] += self.alpha * td_error

    def _decay_epsilon(self) -> None:
        """Apply exponential ε decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
