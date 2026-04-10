"""
LLM Orchestrator for SCAF-LS

Supports:
  - OpenAI GPT-4o  (OPENAI_API_KEY)
  - Anthropic Claude (ANTHROPIC_API_KEY)
  - Any OpenAI-compatible server: Ollama, LM Studio … (SCAF_LLM_BASE_URL)
  - Offline / no-key fallback (returns rule-based defaults)

Three orchestration methods
  - analyze_market()      → regime + action + position_scalar
  - select_models()       → weighted model subset per regime
  - assess_risk_override()→ exposure-reduction decision

Features
  - Auto-detection of API keys from environment variables
  - TTL-based response cache (default 300 s)
  - Automatic retry with exponential back-off (3 attempts)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .prompts import PromptBuilder

logger = logging.getLogger(__name__)

# ─────────────────────────── constants ──────────────────────────────────── #

_DEFAULT_TTL = 300          # seconds
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0     # seconds


# ─────────────────────────── helpers ────────────────────────────────────── #

def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object found in *text*."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()
    # Find first { … }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end])


# ──────────────────────── backend detection ─────────────────────────────── #

class _BackendType:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPAT = "openai_compat"  # Ollama / LM Studio / etc.
    OFFLINE = "offline"


def _detect_backend() -> Tuple[str, Optional[str], Optional[str]]:
    """Return (backend_type, api_key, base_url)."""
    base_url = os.environ.get("SCAF_LLM_BASE_URL", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if base_url:
        # Custom OpenAI-compatible server (Ollama, LM Studio …)
        key = openai_key or "ollama"  # Ollama accepts any non-empty key
        return _BackendType.OPENAI_COMPAT, key, base_url
    if openai_key:
        return _BackendType.OPENAI, openai_key, None
    if anthropic_key:
        return _BackendType.ANTHROPIC, anthropic_key, None
    return _BackendType.OFFLINE, None, None


# ──────────────────────────── LLMOrchestrator ───────────────────────────── #

class LLMOrchestrator:
    """High-level interface between SCAF-LS and language model backends.

    Parameters
    ----------
    model_name:
        Override the model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022",
        "llama3.2").  When *None* a sensible default is chosen per backend.
    cache_ttl:
        Seconds to cache identical prompts.  Set to 0 to disable caching.
    max_retries:
        Number of retry attempts on transient errors.
    temperature:
        Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_ttl: int = _DEFAULT_TTL,
        max_retries: int = _MAX_RETRIES,
        temperature: float = 0.0,
    ):
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.temperature = temperature

        self._backend, self._api_key, self._base_url = _detect_backend()
        self._model_name = model_name or self._default_model()
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

        logger.info(
            "LLMOrchestrator initialised: backend=%s model=%s",
            self._backend,
            self._model_name,
        )

    # ------------------------------------------------------------------ #
    #  Public orchestration methods                                         #
    # ------------------------------------------------------------------ #

    def analyze_market(
        self, market_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify the market regime, recommend an action and a position scalar.

        Returns
        -------
        dict with keys: regime, action, position_scalar, rationale
        """
        prompt = PromptBuilder.analyze_market(market_snapshot)
        raw = self._call_with_cache(prompt)
        result = _parse_json_response(raw)
        # Validate / coerce
        result.setdefault("regime", "sideways")
        result.setdefault("action", "flat")
        result["position_scalar"] = float(
            max(0.0, min(1.5, result.get("position_scalar", 1.0)))
        )
        return result

    def select_models(
        self,
        regime: str,
        available_models: List[str],
        model_performance: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Select and weight models appropriate for *regime*.

        Returns
        -------
        dict mapping model_name → weight (sum ≈ 1.0)
        """
        prompt = PromptBuilder.select_models(
            regime, available_models, model_performance
        )
        raw = self._call_with_cache(prompt)
        result = _parse_json_response(raw)
        weights: Dict[str, float] = result.get("model_weights", {})

        # Normalise so they sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback: equal weights for all available models
            n = len(available_models)
            weights = {m: 1.0 / n for m in available_models}

        return weights

    def assess_risk_override(
        self,
        current_drawdown: float,
        current_vol: float,
        position_scalar: float,
        vix: float,
        recent_returns: List[float],
    ) -> Dict[str, Any]:
        """Decide whether to reduce exposure.

        Returns
        -------
        dict with keys: override (bool), new_position_scalar (float), reason (str)
        """
        prompt = PromptBuilder.assess_risk_override(
            current_drawdown, current_vol, position_scalar, vix, recent_returns
        )
        raw = self._call_with_cache(prompt)
        result = _parse_json_response(raw)
        result["override"] = bool(result.get("override", False))
        result["new_position_scalar"] = float(
            max(0.0, min(1.5, result.get("new_position_scalar", position_scalar)))
        )
        return result

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _default_model(self) -> str:
        defaults = {
            _BackendType.OPENAI: "gpt-4o",
            _BackendType.ANTHROPIC: "claude-3-5-sonnet-20241022",
            _BackendType.OPENAI_COMPAT: "llama3.2",
            _BackendType.OFFLINE: "offline",
        }
        return defaults[self._backend]

    def _call_with_cache(self, prompt: str) -> str:
        """Return cached response or issue a fresh LLM call (with retry)."""
        if self.cache_ttl > 0:
            key = _cache_key(prompt)
            if key in self._cache:
                response, ts = self._cache[key]
                if time.time() - ts < self.cache_ttl:
                    logger.debug("LLM cache hit for key %s", key[:8])
                    return json.dumps(response)

        raw = self._call_with_retry(prompt)

        if self.cache_ttl > 0:
            key = _cache_key(prompt)
            try:
                parsed = _parse_json_response(raw)
                self._cache[key] = (parsed, time.time())
            except ValueError:
                pass  # Don't cache unparseable responses

        return raw

    def _call_with_retry(self, prompt: str) -> str:
        """Issue the LLM call, retrying up to *max_retries* times on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._dispatch(prompt)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        logger.error(
            "LLM call failed after %d attempts, falling back to offline defaults. "
            "Last error: %s",
            self.max_retries,
            last_exc,
        )
        return self._offline_fallback(prompt)

    def _dispatch(self, prompt: str) -> str:
        """Route the call to the correct backend."""
        if self._backend == _BackendType.OFFLINE:
            return self._offline_fallback(prompt)
        if self._backend in (_BackendType.OPENAI, _BackendType.OPENAI_COMPAT):
            return self._call_openai(prompt)
        if self._backend == _BackendType.ANTHROPIC:
            return self._call_anthropic(prompt)
        raise ValueError(f"Unknown backend: {self._backend}")

    # ------------------------------------------------------------------ #
    #  Backend implementations                                              #
    # ------------------------------------------------------------------ #

    def _call_openai(self, prompt: str) -> str:
        """Call an OpenAI or OpenAI-compatible (Ollama, LM Studio) endpoint."""
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        kwargs: Dict[str, Any] = {
            "api_key": self._api_key,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = openai.OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": PromptBuilder.system_message()},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        """Call the Anthropic Claude API."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        message = client.messages.create(
            model=self._model_name,
            max_tokens=512,
            system=PromptBuilder.system_message(),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        block = message.content[0]
        return block.text if hasattr(block, "text") else str(block)

    def _offline_fallback(self, prompt: str) -> str:
        """Return a safe rule-based JSON response when no LLM is available."""
        if "analyze_market" in prompt or "Market snapshot" in prompt:
            return json.dumps({
                "regime": "sideways",
                "action": "flat",
                "position_scalar": 1.0,
                "rationale": "No LLM available — offline default.",
            })
        if "model_weights" in prompt or "Select the best subset" in prompt:
            return json.dumps({"model_weights": {}})
        if "override" in prompt or "exposure reduction" in prompt:
            return json.dumps({
                "override": False,
                "new_position_scalar": 1.0,
                "reason": "No LLM available — offline default.",
            })
        return json.dumps({"status": "offline"})
