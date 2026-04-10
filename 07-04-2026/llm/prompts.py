"""
LLM Prompt Templates for SCAF-LS

Structured prompts for the three LLM orchestration methods:
  - analyze_market()
  - select_models()
  - assess_risk_override()
"""

from typing import Any, Dict, List


class PromptBuilder:
    """Builds structured prompts for each LLM orchestration task."""

    # ------------------------------------------------------------------ #
    #  analyze_market                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def analyze_market(market_snapshot: Dict[str, Any]) -> str:
        """Return a prompt that asks the LLM to identify the market regime,
        recommend an action and suggest a position scalar.

        Args:
            market_snapshot: dict with keys such as ret_1d, vol_20d, vix,
                yield_curve_spread, rsi_14, current_drawdown, etc.
        """
        snapshot_lines = "\n".join(
            f"  {k}: {v}" for k, v in market_snapshot.items()
        )
        return (
            "You are a quantitative risk analyst for the SCAF-LS trading system.\n"
            "Given the following market snapshot, identify the current regime, "
            "recommend a trading action, and suggest a position scalar.\n\n"
            f"Market snapshot:\n{snapshot_lines}\n\n"
            "Respond in strict JSON with these keys:\n"
            '  "regime": one of ["bull", "bear", "sideways", "crisis"],\n'
            '  "action": one of ["long", "short", "flat", "reduce"],\n'
            '  "position_scalar": float in [0.0, 1.5],\n'
            '  "rationale": brief explanation (max 3 sentences).\n'
            "Respond with JSON only — no markdown, no extra text."
        )

    # ------------------------------------------------------------------ #
    #  select_models                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def select_models(
        regime: str,
        available_models: List[str],
        model_performance: Dict[str, Dict[str, float]],
    ) -> str:
        """Return a prompt that asks the LLM to select and weight models
        for the detected regime.

        Args:
            regime: current regime label.
            available_models: list of model names available in the registry.
            model_performance: dict mapping model name → performance metrics
                (e.g. {"LGBM": {"auc": 0.55, "sharpe": 0.8}, ...}).
        """
        perf_lines = "\n".join(
            f"  {m}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            for m, metrics in model_performance.items()
        )
        models_list = ", ".join(available_models)
        return (
            "You are a quantitative portfolio manager for the SCAF-LS trading system.\n"
            f"Current market regime: {regime}\n"
            f"Available models: {models_list}\n\n"
            "Recent performance metrics per model:\n"
            f"{perf_lines}\n\n"
            "Select the best subset of models for this regime and assign each a "
            "weight (weights must sum to 1.0).\n"
            "Respond in strict JSON with this key:\n"
            '  "model_weights": {{"ModelName": weight_float, ...}}\n'
            "Only include models that should be active. "
            "Respond with JSON only — no markdown, no extra text."
        )

    # ------------------------------------------------------------------ #
    #  assess_risk_override                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def assess_risk_override(
        current_drawdown: float,
        current_vol: float,
        position_scalar: float,
        vix: float,
        recent_returns: List[float],
    ) -> str:
        """Return a prompt that asks the LLM whether to override and reduce
        exposure given the current risk metrics.

        Args:
            current_drawdown: current peak-to-trough drawdown (positive value).
            current_vol: annualised realised volatility.
            position_scalar: current position scalar [0, 1.5].
            vix: VIX level (or synthetic proxy).
            recent_returns: last 5 daily returns.
        """
        returns_str = ", ".join(f"{r:.4f}" for r in recent_returns[-5:])
        return (
            "You are a risk manager for the SCAF-LS trading system.\n"
            "Evaluate whether an exposure reduction is warranted given the "
            "following risk metrics:\n\n"
            f"  current_drawdown: {current_drawdown:.4f}\n"
            f"  annualised_volatility: {current_vol:.4f}\n"
            f"  current_position_scalar: {position_scalar:.2f}\n"
            f"  vix: {vix:.2f}\n"
            f"  recent_daily_returns (last 5): [{returns_str}]\n\n"
            "Respond in strict JSON with these keys:\n"
            '  "override": true | false,\n'
            '  "new_position_scalar": float in [0.0, 1.5]  '
            "(use current value if no override),\n"
            '  "reason": brief explanation (max 2 sentences).\n'
            "Respond with JSON only — no markdown, no extra text."
        )

    # ------------------------------------------------------------------ #
    #  system message                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def system_message() -> str:
        """Generic system message injected as the first message for
        chat-completion APIs."""
        return (
            "You are SCAF-LS Assistant, a precise quantitative finance AI. "
            "Always respond with valid JSON only. Never add explanations outside "
            "the JSON structure."
        )
