"""
SCAF LLM Prompt Templates
Provides structured prompt builders for LLM-based cognitive orchestration.
"""

from typing import Dict, Any, List


class PromptBuilder:
    """Builds structured prompts for LLM market analysis and decision orchestration."""

    SYSTEM_PROMPT = """You are SCAF-LLM, a cognitive orchestrator for a multi-agent financial forecasting system.
You analyze market conditions, model performance, and risk signals to guide ensemble decisions.
Always respond in valid JSON. Be concise and data-driven. Never give financial advice."""

    @staticmethod
    def market_regime_analysis(
        market_stats: Dict[str, float],
        model_signals: Dict[str, float],
        risk_signals: Dict[str, float],
    ) -> str:
        """Build prompt for LLM market regime analysis."""
        return f"""Analyze the following real-time market data and return a JSON decision.

MARKET STATISTICS:
- SPX 5-day return: {market_stats.get('ret_5d', 0.0):.4f}
- SPX 20-day volatility (annualised): {market_stats.get('vol_20d_ann', 0.0):.4f}
- VIX level: {market_stats.get('vix', 20.0):.2f}
- Yield curve spread (10Y-3M): {market_stats.get('yield_spread', 0.0):.4f}
- 20-day z-score: {market_stats.get('zscore_20d', 0.0):.3f}

MODEL SIGNALS (probability of up move, 0–1):
{_format_dict(model_signals)}

RISK AGENT AGGREGATED SIGNAL (−1 = max reduce, +1 = max increase):
{_format_dict(risk_signals)}

Return a JSON object with exactly these keys:
{{
  "regime": "<bull|bear|sideways|crisis>",
  "regime_confidence": <0.0–1.0>,
  "recommended_action": "<increase|hold|decrease|exit>",
  "position_scalar": <0.0–1.5>,
  "top_models": ["<model1>", "<model2>"],
  "reasoning": "<one sentence>"
}}"""

    @staticmethod
    def model_selection_guidance(
        model_performance: Dict[str, Dict[str, float]],
        current_regime: str,
        available_models: List[str],
    ) -> str:
        """Build prompt for LLM-assisted model selection."""
        perf_text = "\n".join(
            f"  {name}: AUC={stats.get('auc', 0.5):.3f}, "
            f"Sharpe={stats.get('sharpe', 0.0):.2f}, "
            f"Drawdown={stats.get('max_dd', 0.0):.3f}"
            for name, stats in model_performance.items()
        )
        return f"""Current market regime: {current_regime}

Model performance statistics:
{perf_text}

Available models: {available_models}

Select the optimal subset of models for the current regime and return JSON:
{{
  "selected_models": ["<model1>", ...],
  "weights": {{"<model1>": <weight>, ...}},
  "excluded_models": ["<model>", ...],
  "rationale": "<one sentence>"
}}"""

    @staticmethod
    def risk_override_assessment(
        current_drawdown: float,
        volatility: float,
        crisis_score: float,
        ensemble_signal: float,
    ) -> str:
        """Build prompt for LLM risk override decision."""
        return f"""RISK OVERRIDE ASSESSMENT

Current portfolio drawdown: {current_drawdown:.3f} ({current_drawdown*100:.1f}%)
Realised annualised volatility: {volatility:.3f} ({volatility*100:.1f}%)
Crisis detection score (0–1): {crisis_score:.3f}
Ensemble forecast signal (−1 to +1): {ensemble_signal:.3f}

Should the system apply a risk override? Return JSON:
{{
  "apply_override": <true|false>,
  "override_scalar": <0.0–1.0>,
  "stop_trading": <true|false>,
  "reason": "<one sentence>"
}}"""


def _format_dict(d: Dict[str, Any]) -> str:
    return "\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                     for k, v in d.items())
