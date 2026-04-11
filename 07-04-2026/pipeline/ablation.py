"""
Ablation Study for SCAF-LS

Systematically removes individual SCAF components to measure each component's
contribution, as reported in the abstract.

Six configurations tested:
  1. Full SCAF      – all components enabled
  2. No Q-learning  – static equal-weight model selection
  3. No LLM         – rule-based regime detection only
  4. No Conformal   – no prediction-set calibration
  5. Minimal        – top-3 models, no Q-learning, no LLM
  6. Single model   – LGBM only (deep-learning baseline proxy)

Usage
-----
    from pipeline.ablation import AblationStudy

    study = AblationStudy()
    results = study.run()
    print(results["comparison_table"])
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.experiment import ExperimentConfig, ExperimentPipeline

logger = logging.getLogger(__name__)


# ─────────────────────────── Ablation configurations ────────────────────── #

def _make_ablation_configs(base: ExperimentConfig) -> Dict[str, ExperimentConfig]:
    """Return the six ablation configurations derived from *base*."""

    def _clone(**overrides) -> ExperimentConfig:
        cfg = copy.deepcopy(base)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    return {
        "full_scaf": _clone(),
        "no_qlearning": _clone(use_qlearning=False),
        "no_llm": _clone(use_llm=False),
        "no_conformal": _clone(use_conformal=False),
        "minimal": _clone(
            use_qlearning=False,
            use_llm=False,
            model_names=["LGBM", "RandomForest", "BiLSTM"],
        ),
        "single_model": _clone(
            use_qlearning=False,
            use_llm=False,
            use_conformal=False,
            model_names=["LGBM"],
        ),
    }


# ─────────────────────────── AblationStudy ──────────────────────────────── #

class AblationStudy:
    """Run SCAF-LS with each ablation configuration and compare results.

    Parameters
    ----------
    base_config:
        Template ``ExperimentConfig``.  All ablations inherit from this.
    configurations:
        Subset of configuration keys to run (default: all six).
    """

    CONFIGS = [
        "full_scaf",
        "no_qlearning",
        "no_llm",
        "no_conformal",
        "minimal",
        "single_model",
    ]

    def __init__(
        self,
        base_config: Optional[ExperimentConfig] = None,
        configurations: Optional[List[str]] = None,
    ):
        self._base_cfg = base_config or ExperimentConfig()
        self._configs_to_run = configurations or self.CONFIGS

    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """Run every ablation configuration.

        Returns
        -------
        dict with keys:
            ``per_config``        – full pipeline result per configuration name
            ``comparison_table``  – side-by-side metric comparison
        """
        ablation_cfgs = _make_ablation_configs(self._base_cfg)
        per_config: Dict[str, Any] = {}

        for name in self._configs_to_run:
            if name not in ablation_cfgs:
                logger.warning("Unknown ablation config '%s' — skipping.", name)
                continue
            logger.info("=== Ablation: %s ===", name)
            pipeline = ExperimentPipeline(config=ablation_cfgs[name])
            try:
                result = pipeline.run()
                per_config[name] = result
            except Exception as exc:
                logger.error("Ablation '%s' failed: %s", name, exc)
                per_config[name] = {"error": str(exc)}

        comparison = self._build_comparison(per_config)
        return {"per_config": per_config, "comparison_table": comparison}

    # ------------------------------------------------------------------ #

    def _build_comparison(
        self, per_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build a list of per-configuration metric rows for easy comparison."""
        rows = []
        for name, result in per_config.items():
            if "error" in result:
                rows.append({"config": name, "error": result["error"]})
                continue

            s = result.get("summary", {})
            row: Dict[str, Any] = {
                "config": name,
                "sharpe_ratio": s.get("sharpe_ratio"),
                "max_drawdown": s.get("max_drawdown"),
                "cumulative_return": s.get("cumulative_return"),
                "rmse_brier": s.get("rmse_brier"),
                "conformal_coverage_rate": s.get("conformal_coverage_rate"),
                "mean_inference_latency_ms": s.get("mean_inference_latency_ms"),
            }

            # Relative change vs full_scaf (Δ%)
            rows.append(row)

        # Compute deltas vs full_scaf baseline
        full_row = next((r for r in rows if r.get("config") == "full_scaf"), None)
        if full_row:
            for row in rows:
                if row.get("config") == "full_scaf":
                    continue
                for metric in ("sharpe_ratio", "max_drawdown", "rmse_brier"):
                    base_val = full_row.get(metric)
                    row_val = row.get(metric)
                    if base_val is not None and row_val is not None and base_val != 0:
                        delta = round((row_val - base_val) / abs(base_val) * 100, 2)
                        row[f"{metric}_delta_pct"] = delta

        logger.info("=== Ablation comparison table ===")
        for row in rows:
            logger.info("  %s", row)

        return rows


# ─────────────────────────── CLI entry point ────────────────────────────── #

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    configs = sys.argv[1:] if len(sys.argv) > 1 else None
    study = AblationStudy(configurations=configs)
    results = study.run()
    print(json.dumps(results["comparison_table"], indent=2))
