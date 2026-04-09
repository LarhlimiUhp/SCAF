"""
SCAF LLM Cognitive Orchestrator
Integrates a real Large Language Model (OpenAI-compatible API) as the top-level
cognitive layer that interprets market conditions and overrides / fine-tunes the
ensemble decision produced by the 300 statistical sub-agents.

Supported backends (in priority order):
  1. OpenAI GPT-4o / GPT-3.5-turbo  (OPENAI_API_KEY env var)
  2. Anthropic Claude                (ANTHROPIC_API_KEY env var)
  3. Any OpenAI-compatible server    (SCAF_LLM_BASE_URL + SCAF_LLM_API_KEY)
  4. Offline stub (returns neutral defaults so the rest of the pipeline keeps working)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Runtime configuration for the LLM orchestrator."""

    # Which backend to prefer ('openai', 'anthropic', 'compatible', 'auto')
    backend: str = "auto"
    model: str = "gpt-4o-mini"          # model name passed to the API
    temperature: float = 0.1            # low temperature for deterministic finance decisions
    max_tokens: int = 512
    timeout: float = 30.0               # seconds
    max_retries: int = 3
    retry_delay: float = 2.0            # seconds between retries

    # OpenAI-compatible server override
    base_url: Optional[str] = None      # e.g. "http://localhost:11434/v1" for Ollama
    api_key: Optional[str] = None       # fallback if env var not set

    # Caching: avoid re-querying the LLM for identical market snapshots
    enable_cache: bool = True
    cache_ttl_seconds: int = 60

    # Whether to fall back gracefully when no API key is available
    allow_offline_fallback: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_api_key(env_var: str, config_key: Optional[str]) -> Optional[str]:
    return os.environ.get(env_var) or config_key


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Extract the first JSON object from a potentially markdown-wrapped response."""
    raw = raw.strip()
    # Strip ```json ... ``` fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------

class _OpenAIAdapter:
    """Wraps the openai Python SDK."""

    def __init__(self, cfg: LLMConfig):
        import openai  # noqa: PLC0415
        key = _load_api_key("OPENAI_API_KEY", cfg.api_key)
        kwargs: Dict[str, Any] = {"api_key": key, "timeout": cfg.timeout}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = openai.OpenAI(**kwargs)
        self._cfg = cfg

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


class _AnthropicAdapter:
    """Wraps the anthropic Python SDK."""

    def __init__(self, cfg: LLMConfig):
        import anthropic  # noqa: PLC0415
        key = _load_api_key("ANTHROPIC_API_KEY", cfg.api_key)
        self._client = anthropic.Anthropic(api_key=key)
        self._cfg = cfg
        # Default to Claude Haiku for cost-efficiency
        self._model = cfg.model if "claude" in cfg.model else "claude-3-haiku-20240307"

    def chat(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text if msg.content else ""


class _OfflineAdapter:
    """Returns neutral defaults when no API key is configured."""

    def chat(self, system: str, user: str) -> str:  # noqa: ARG002
        logger.debug("LLM offline fallback: returning neutral defaults.")
        # Detect which type of response is needed from the prompt content
        if "regime" in user:
            return json.dumps({
                "regime": "sideways",
                "regime_confidence": 0.5,
                "recommended_action": "hold",
                "position_scalar": 1.0,
                "top_models": [],
                "reasoning": "LLM offline – defaulting to neutral.",
            })
        if "selected_models" in user:
            return json.dumps({
                "selected_models": [],
                "weights": {},
                "excluded_models": [],
                "rationale": "LLM offline – no selection.",
            })
        return json.dumps({
            "apply_override": False,
            "override_scalar": 1.0,
            "stop_trading": False,
            "reason": "LLM offline – no override.",
        })


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class LLMOrchestrator:
    """
    Cognitive orchestration layer backed by a real LLM.

    Responsibilities:
      - Interpret current market statistics and return a regime label + confidence.
      - Select the optimal model subset for the current regime.
      - Decide whether to apply a risk override on top of the statistical agents.

    The orchestrator is designed to be called *once per prediction step*
    (i.e. once per trading day in a walk-forward setup), so latency matters.
    Responses are cached for ``cfg.cache_ttl_seconds`` to avoid redundant calls
    within the same timestep.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig()
        self._adapter = self._build_adapter()
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
        self._call_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_market(
        self,
        market_stats: Dict[str, float],
        model_signals: Dict[str, float],
        risk_signals: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to classify the current market regime and recommend
        an ensemble action.

        Returns a dict with keys:
          regime, regime_confidence, recommended_action,
          position_scalar, top_models, reasoning
        """
        prompt = PromptBuilder.market_regime_analysis(market_stats, model_signals, risk_signals)
        defaults = {
            "regime": "sideways",
            "regime_confidence": 0.5,
            "recommended_action": "hold",
            "position_scalar": 1.0,
            "top_models": [],
            "reasoning": "default",
        }
        return self._query_with_defaults(prompt, defaults)

    def select_models(
        self,
        model_performance: Dict[str, Dict[str, float]],
        current_regime: str,
        available_models: List[str],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to select and weight models for the current regime.

        Returns a dict with keys:
          selected_models, weights, excluded_models, rationale
        """
        prompt = PromptBuilder.model_selection_guidance(
            model_performance, current_regime, available_models
        )
        defaults = {
            "selected_models": available_models,
            "weights": {m: 1.0 / max(len(available_models), 1) for m in available_models},
            "excluded_models": [],
            "rationale": "default equal weighting",
        }
        return self._query_with_defaults(prompt, defaults)

    def assess_risk_override(
        self,
        current_drawdown: float,
        volatility: float,
        crisis_score: float,
        ensemble_signal: float,
    ) -> Dict[str, Any]:
        """
        Ask the LLM whether a risk override should be applied.

        Returns a dict with keys:
          apply_override, override_scalar, stop_trading, reason
        """
        prompt = PromptBuilder.risk_override_assessment(
            current_drawdown, volatility, crisis_score, ensemble_signal
        )
        defaults = {
            "apply_override": False,
            "override_scalar": 1.0,
            "stop_trading": False,
            "reason": "default – no override",
        }
        return self._query_with_defaults(prompt, defaults)

    @property
    def stats(self) -> Dict[str, int]:
        return {"calls": self._call_count, "errors": self._error_count}

    def is_online(self) -> bool:
        return not isinstance(self._adapter, _OfflineAdapter)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_adapter(self):
        backend = self.cfg.backend

        # Explicit compatible-server override (e.g. Ollama, LM Studio)
        base_url = self.cfg.base_url or os.environ.get("SCAF_LLM_BASE_URL")
        compat_key = self.cfg.api_key or os.environ.get("SCAF_LLM_API_KEY")
        if base_url and (backend in ("compatible", "auto")):
            try:
                compat_cfg = LLMConfig(
                    backend="compatible",
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout,
                    base_url=base_url,
                    api_key=compat_key or "sk-placeholder",
                )
                adapter = _OpenAIAdapter(compat_cfg)
                logger.info("LLM backend: OpenAI-compatible server at %s", base_url)
                return adapter
            except Exception as exc:
                logger.warning("Compatible server init failed: %s", exc)

        if backend in ("openai", "auto"):
            if _load_api_key("OPENAI_API_KEY", self.cfg.api_key):
                try:
                    adapter = _OpenAIAdapter(self.cfg)
                    logger.info("LLM backend: OpenAI (%s)", self.cfg.model)
                    return adapter
                except Exception as exc:
                    logger.warning("OpenAI init failed: %s", exc)

        if backend in ("anthropic", "auto"):
            if _load_api_key("ANTHROPIC_API_KEY", self.cfg.api_key):
                try:
                    adapter = _AnthropicAdapter(self.cfg)
                    logger.info("LLM backend: Anthropic Claude")
                    return adapter
                except Exception as exc:
                    logger.warning("Anthropic init failed: %s", exc)

        if self.cfg.allow_offline_fallback:
            logger.warning(
                "No LLM API key found. Using offline fallback (neutral defaults). "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable real LLM orchestration."
            )
            return _OfflineAdapter()

        raise RuntimeError(
            "LLM orchestrator: no API key configured and allow_offline_fallback=False."
        )

    def _query_with_defaults(
        self, user_prompt: str, defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an LLM query with retry logic and cache."""
        cache_key = user_prompt[:256]  # use truncated prompt as key
        if self.cfg.enable_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                ts, result = cached
                if time.time() - ts < self.cfg.cache_ttl_seconds:
                    return result

        result = dict(defaults)
        for attempt in range(self.cfg.max_retries):
            try:
                self._call_count += 1
                raw = self._adapter.chat(PromptBuilder.SYSTEM_PROMPT, user_prompt)
                parsed = _parse_llm_json(raw)
                # Merge parsed keys into defaults (only update known keys)
                result.update({k: v for k, v in parsed.items() if k in defaults})
                if self.cfg.enable_cache:
                    self._cache[cache_key] = (time.time(), result)
                return result
            except Exception as exc:
                self._error_count += 1
                logger.warning(
                    "LLM query attempt %d/%d failed: %s",
                    attempt + 1, self.cfg.max_retries, exc,
                )
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(self.cfg.retry_delay)

        logger.error("All LLM retries exhausted – using defaults.")
        return result
