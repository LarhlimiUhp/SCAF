"""
Multi-Asset Experiment for SCAF-LS

Evaluates SCAF across the four major asset classes described in the abstract:
  - S&P 500    (^GSPC)
  - EUR/USD    (EURUSD=X)
  - NASDAQ     (^IXIC)
  - Bitcoin    (BTC-USD)

spanning 2018-01-01 to 2024-12-31.

Usage
-----
    from pipeline.multi_asset_experiment import MultiAssetExperiment

    exp = MultiAssetExperiment()
    results = exp.run()
    print(results["combined_summary"])
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pipeline.experiment import ExperimentConfig, ExperimentPipeline

logger = logging.getLogger(__name__)

# ─────────────────────────── Asset definitions ──────────────────────────── #

ASSET_CONFIGS: Dict[str, Dict[str, str]] = {
    "SP500":   {"ticker": "^GSPC",    "label": "S&P 500"},
    "EURUSD":  {"ticker": "EURUSD=X", "label": "EUR/USD"},
    "NASDAQ":  {"ticker": "^IXIC",    "label": "NASDAQ"},
    "Bitcoin": {"ticker": "BTC-USD",  "label": "Bitcoin"},
}

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"


# ─────────────────────────── MultiAssetExperiment ───────────────────────── #

class MultiAssetExperiment:
    """Run SCAF-LS independently on each of the four major asset classes.

    Parameters
    ----------
    base_config:
        Optional ``ExperimentConfig`` used as template.  ``ticker``,
        ``start_date`` and ``end_date`` are overridden per asset.
    assets:
        Subset of asset keys to evaluate (default: all four).
    """

    def __init__(
        self,
        base_config: Optional[ExperimentConfig] = None,
        assets: Optional[list] = None,
    ):
        self._base_cfg = base_config or ExperimentConfig()
        self._assets = assets or list(ASSET_CONFIGS.keys())

    # ------------------------------------------------------------------ #

    def run(self, results_dir: str = "results") -> Dict[str, Any]:
        """Evaluate SCAF on every requested asset and return combined results.

        Persists per-asset summaries and the combined summary as JSON files
        under *results_dir* so that results are reproducible and inspectable.

        Returns
        -------
        dict with keys:
            ``per_asset``      – full result dict keyed by asset name
            ``combined_summary`` – cross-asset aggregate metrics
        """
        os.makedirs(results_dir, exist_ok=True)
        per_asset: Dict[str, Any] = {}

        for asset_key in self._assets:
            asset_meta = ASSET_CONFIGS[asset_key]
            logger.info(
                "=== Running SCAF on %s (%s) ===",
                asset_meta["label"],
                asset_meta["ticker"],
            )

            cfg = self._make_config(asset_meta["ticker"], results_dir=results_dir)
            pipeline = ExperimentPipeline(config=cfg)

            try:
                result = pipeline.run()
                per_asset[asset_key] = result
                summary = result.get("summary", {})
                logger.info("%s summary: %s", asset_meta["label"], summary)

                # Persist per-asset summary
                summary_path = os.path.join(
                    results_dir, f"scaf_{asset_key.lower()}_summary.json"
                )
                _save_json({"asset": asset_key, "label": asset_meta["label"],
                            "ticker": asset_meta["ticker"], "summary": summary},
                           summary_path)
                logger.info("Saved %s summary → %s", asset_meta["label"], summary_path)

            except Exception as exc:
                logger.error("Pipeline failed for %s: %s", asset_meta["label"], exc)
                per_asset[asset_key] = {"error": str(exc)}

        combined = self._combine(per_asset)

        # Persist combined summary
        combined_path = os.path.join(results_dir, "scaf_multi_asset_combined.json")
        _save_json(combined, combined_path)
        logger.info("Saved combined summary → %s", combined_path)

        return {"per_asset": per_asset, "combined_summary": combined}

    # ------------------------------------------------------------------ #

    def _make_config(self, ticker: str, results_dir: str = "results") -> ExperimentConfig:
        """Build a per-asset config from the template."""
        import copy
        cfg = copy.deepcopy(self._base_cfg)
        cfg.ticker = ticker
        cfg.start_date = START_DATE
        cfg.end_date = END_DATE
        cfg.results_dir = results_dir
        # Separate Q-table per asset to avoid cross-asset contamination
        safe_ticker = ticker.replace("^", "").replace("=", "").replace("-", "")
        cfg.ql_qtable_path = os.path.join(results_dir, f"qlearning_{safe_ticker}.json")
        return cfg

    def _combine(self, per_asset: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cross-asset aggregate statistics."""
        import numpy as np

        sharpes, max_dds, rmses, static_coverages, aci_coverages, latencies = (
            [], [], [], [], [], []
        )
        rmse_reductions, sharpe_improvements, dd_reductions = [], [], []

        for key, res in per_asset.items():
            if "error" in res:
                continue
            s = res.get("summary", {})
            if "sharpe_ratio" in s:
                sharpes.append(s["sharpe_ratio"])
            if "max_drawdown" in s:
                max_dds.append(s["max_drawdown"])
            if s.get("rmse_brier") is not None:
                rmses.append(s["rmse_brier"])
            if s.get("conformal_coverage_rate_static") is not None:
                static_coverages.append(s["conformal_coverage_rate_static"])
            if s.get("conformal_coverage_rate_aci") is not None:
                aci_coverages.append(s["conformal_coverage_rate_aci"])
            if s.get("mean_inference_latency_ms") is not None:
                latencies.append(s["mean_inference_latency_ms"])
            if s.get("rmse_reduction_vs_dl_pct") is not None:
                rmse_reductions.append(s["rmse_reduction_vs_dl_pct"])
            if s.get("sharpe_improvement_vs_dl_pct") is not None:
                sharpe_improvements.append(s["sharpe_improvement_vs_dl_pct"])
            if s.get("drawdown_reduction_vs_dl_pct") is not None:
                dd_reductions.append(s["drawdown_reduction_vs_dl_pct"])

        def _mean(lst: List) -> Optional[float]:
            return round(float(np.mean(lst)), 4) if lst else None

        return {
            "n_assets_evaluated": len([v for v in per_asset.values() if "error" not in v]),
            "mean_sharpe_ratio": _mean(sharpes),
            "mean_max_drawdown": _mean(max_dds),
            "mean_rmse_brier": _mean(rmses),
            "mean_conformal_coverage_static": _mean(static_coverages),
            "mean_conformal_coverage_aci": _mean(aci_coverages),
            "mean_inference_latency_ms": _mean(latencies),
            "mean_rmse_reduction_vs_dl_pct": _mean(rmse_reductions),
            "mean_sharpe_improvement_vs_dl_pct": _mean(sharpe_improvements),
            "mean_drawdown_reduction_vs_dl_pct": _mean(dd_reductions),
        }


# ─────────────────────────── helpers ────────────────────────────────────── #

def _save_json(data: Any, path: str) -> None:
    """Serialise *data* to *path*, converting numpy scalars and NaN to JSON-safe types."""

    def _sanitise(obj: Any) -> Any:
        """Recursively convert numpy scalars, NaN, and ndarray to JSON types."""
        import math
        import numpy as np  # noqa: PLC0415
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitise(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [_sanitise(v) for v in obj.tolist()]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_sanitise(data), fh, indent=2)
    except Exception as exc:
        logger.warning("Could not save JSON to %s: %s", path, exc)


# ─────────────────────────── CLI entry point ────────────────────────────── #

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    assets = sys.argv[1:] if len(sys.argv) > 1 else None
    exp = MultiAssetExperiment(assets=assets)
    results = exp.run()
    print(json.dumps(results["combined_summary"], indent=2))
