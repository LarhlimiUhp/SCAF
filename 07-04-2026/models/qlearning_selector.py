"""
SCAF Q-Learning Model Selector
================================
Implements a proper Q-Learning agent that learns *which combination of expert
models* to activate in each market regime.

Key design choices
------------------
* **State space** : discretised market regime (bull / bear / sideways / crisis)
  crossed with a rolling-performance bucket for the current ensemble.
  → 4 × 3 = 12 distinct states (small enough for a tabular Q-table).

* **Action space** : every possible subset of registered models, capped at a
  manageable size by pre-selecting the top-K candidate models.

* **Reward** : risk-adjusted return realised in the *next* step.
  r_t = Sharpe-scaled daily PnL − λ · max_drawdown_penalty

* **Update rule** : standard one-step Temporal-Difference (TD) Bellman update
      Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

* **Exploration** : ε-greedy with exponential decay (ε_min = 0.05).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections import defaultdict, deque
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

_REGIME_IDX = {"bull": 0, "bear": 1, "sideways": 2, "crisis": 3}
_PERF_IDX = {"good": 0, "neutral": 1, "bad": 2}


def encode_state(regime: str, recent_sharpe: float) -> int:
    """Map (regime, performance bucket) → integer state index."""
    r = _REGIME_IDX.get(regime, 2)
    if recent_sharpe > 0.5:
        p = 0
    elif recent_sharpe > -0.5:
        p = 1
    else:
        p = 2
    return r * len(_PERF_IDX) + p


NUM_STATES = len(_REGIME_IDX) * len(_PERF_IDX)  # 12


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

def build_action_space(model_names: List[str], max_subset_size: int = 3) -> List[Tuple[str, ...]]:
    """
    Enumerate all non-empty subsets up to *max_subset_size* models.
    For N models and size K the number of actions is Σ C(N,k) for k=1..K.
    With N=10, K=3 → 175 actions — well within a tabular Q-table.
    """
    actions: List[Tuple[str, ...]] = []
    for size in range(1, min(max_subset_size, len(model_names)) + 1):
        for combo in combinations(model_names, size):
            actions.append(combo)
    return actions


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearningModelSelector:
    """
    Tabular Q-Learning agent for adaptive model selection.

    Parameters
    ----------
    model_names : list of registered model names to consider
    alpha       : learning rate (0, 1]
    gamma       : discount factor [0, 1]
    epsilon     : initial exploration rate
    epsilon_min : minimum exploration rate
    epsilon_decay : multiplicative decay per episode
    max_subset_size : maximum number of models in a selected subset
    lambda_dd   : drawdown penalty coefficient in the reward function
    """

    def __init__(
        self,
        model_names: List[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        max_subset_size: int = 3,
        lambda_dd: float = 2.0,
    ):
        self.model_names = list(model_names)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lambda_dd = lambda_dd

        self.actions = build_action_space(self.model_names, max_subset_size)
        self._action_index: Dict[Tuple[str, ...], int] = {
            a: i for i, a in enumerate(self.actions)
        }
        num_actions = len(self.actions)

        # Q-table: shape (NUM_STATES, num_actions)
        self.Q = np.zeros((NUM_STATES, num_actions), dtype=np.float64)

        # Experience replay buffer
        self._replay: deque[Tuple[int, int, float, int]] = deque(maxlen=2000)

        # Tracking
        self._episode = 0
        self._total_steps = 0
        self._last_state: Optional[int] = None
        self._last_action_idx: Optional[int] = None
        self._reward_history: deque[float] = deque(maxlen=500)
        self._action_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_models(
        self,
        regime: str,
        recent_sharpe: float,
        *,
        explore: bool = True,
    ) -> List[str]:
        """
        Choose a model subset for the given state.

        Parameters
        ----------
        regime        : current market regime label
        recent_sharpe : rolling Sharpe ratio of the ensemble (last N steps)
        explore       : whether ε-greedy exploration is active

        Returns
        -------
        List of model names to activate this step.
        """
        state = encode_state(regime, recent_sharpe)
        self._last_state = state

        if explore and np.random.random() < self.epsilon:
            action_idx = np.random.randint(len(self.actions))
        else:
            action_idx = int(np.argmax(self.Q[state]))

        self._last_action_idx = action_idx
        self._action_counts[action_idx] += 1
        self._total_steps += 1

        return list(self.actions[action_idx])

    def update(
        self,
        next_regime: str,
        next_sharpe: float,
        step_pnl: float,
        step_max_dd: float,
    ) -> float:
        """
        Perform a one-step TD update after observing the outcome.

        Parameters
        ----------
        next_regime  : regime in the *next* step (s')
        next_sharpe  : rolling Sharpe in the next step
        step_pnl     : realised portfolio log-return for this step
        step_max_dd  : maximum intra-step drawdown (≥ 0)

        Returns
        -------
        The TD error (δ).
        """
        if self._last_state is None or self._last_action_idx is None:
            return 0.0

        reward = self._compute_reward(step_pnl, step_max_dd)
        self._reward_history.append(reward)

        next_state = encode_state(next_regime, next_sharpe)

        # Bellman target
        q_current = self.Q[self._last_state, self._last_action_idx]
        q_next_max = float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current

        # In-place Q-table update
        self.Q[self._last_state, self._last_action_idx] += self.alpha * td_error

        # Store experience for batch replay
        self._replay.append(
            (self._last_state, self._last_action_idx, reward, next_state)
        )

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(td_error)

    def replay(self, batch_size: int = 32) -> float:
        """
        Mini-batch experience replay (improves sample efficiency).
        Returns mean absolute TD error over the batch.
        """
        if len(self._replay) < batch_size:
            return 0.0

        indices = np.random.choice(len(self._replay), batch_size, replace=False)
        errors = []
        for idx in indices:
            s, a, r, s_next = self._replay[idx]
            q_cur = self.Q[s, a]
            q_next = float(np.max(self.Q[s_next]))
            td_target = r + self.gamma * q_next
            td_err = td_target - q_cur
            self.Q[s, a] += self.alpha * td_err
            errors.append(abs(td_err))
        return float(np.mean(errors))

    def end_episode(self):
        """Call at the end of each walk-forward fold."""
        self._episode += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save Q-table and metadata to disk."""
        state = {
            "Q": self.Q,
            "epsilon": self.epsilon,
            "model_names": self.model_names,
            "episode": self._episode,
            "total_steps": self._total_steps,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Q-table saved to %s", path)

    def load(self, path: str):
        """Load Q-table from disk."""
        if not os.path.exists(path):
            logger.warning("Q-table checkpoint not found at %s – starting fresh.", path)
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.Q = state["Q"]
        self.epsilon = state["epsilon"]
        self._episode = state.get("episode", 0)
        self._total_steps = state.get("total_steps", 0)
        logger.info("Q-table loaded from %s (episode %d)", path, self._episode)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_policy_table(self) -> Dict[str, Any]:
        """Return the greedy policy for each state as a human-readable dict."""
        policy = {}
        for regime, r_idx in _REGIME_IDX.items():
            for perf, p_idx in _PERF_IDX.items():
                state = r_idx * len(_PERF_IDX) + p_idx
                best_action = int(np.argmax(self.Q[state]))
                best_q = float(np.max(self.Q[state]))
                key = f"{regime}/{perf}"
                policy[key] = {
                    "models": list(self.actions[best_action]),
                    "q_value": round(best_q, 4),
                }
        return policy

    def summary(self) -> Dict[str, Any]:
        return {
            "episodes": self._episode,
            "total_steps": self._total_steps,
            "epsilon": round(self.epsilon, 4),
            "num_states": NUM_STATES,
            "num_actions": len(self.actions),
            "mean_recent_reward": (
                round(float(np.mean(self._reward_history)), 4)
                if self._reward_history else 0.0
            ),
            "policy": self.get_policy_table(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reward(self, step_pnl: float, step_max_dd: float) -> float:
        """
        Reward shaping:
          r = pnl_scaled − λ · drawdown_penalty
        where pnl_scaled is the Sharpe-normalised step return.
        """
        # Normalise PnL by recent volatility (Sharpe-like scaling)
        recent = list(self._reward_history)[-20:] if self._reward_history else [0.0]
        vol = float(np.std(recent)) if len(recent) > 1 else 1.0
        pnl_scaled = step_pnl / (vol + 1e-6)

        # Drawdown penalty (convex to penalise large drawdowns more)
        dd_penalty = self.lambda_dd * (step_max_dd ** 2)

        return float(pnl_scaled - dd_penalty)
