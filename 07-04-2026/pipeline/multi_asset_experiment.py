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

import logging
from typing import Any, Dict, Optional

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

    def run(self) -> Dict[str, Any]:
        """Evaluate SCAF on every requested asset and return combined results.

        Returns
        -------
        dict with keys:
            ``per_asset``      – full result dict keyed by asset name
            ``combined_summary`` – cross-asset aggregate metrics
        """
        per_asset: Dict[str, Any] = {}

        for asset_key in self._assets:
            asset_meta = ASSET_CONFIGS[asset_key]
            logger.info(
                "=== Running SCAF on %s (%s) ===",
                asset_meta["label"],
                asset_meta["ticker"],
            )

            cfg = self._make_config(asset_meta["ticker"])
            pipeline = ExperimentPipeline(config=cfg)

            try:
                result = pipeline.run()
                per_asset[asset_key] = result
                logger.info(
                    "%s summary: %s",
                    asset_meta["label"],
                    result.get("summary", {}),
                )
            except Exception as exc:
                logger.error("Pipeline failed for %s: %s", asset_meta["label"], exc)
                per_asset[asset_key] = {"error": str(exc)}

        combined = self._combine(per_asset)
        return {"per_asset": per_asset, "combined_summary": combined}

    # ------------------------------------------------------------------ #

    def _make_config(self, ticker: str) -> ExperimentConfig:
        """Build a per-asset config from the template."""
        import copy
        cfg = copy.deepcopy(self._base_cfg)
        cfg.ticker = ticker
        cfg.start_date = START_DATE
        cfg.end_date = END_DATE
        # Separate Q-table per asset to avoid cross-asset contamination
        safe_ticker = ticker.replace("^", "").replace("=", "").replace("-", "")
        cfg.ql_qtable_path = f"results/qlearning_{safe_ticker}.json"
        return cfg

    def _combine(self, per_asset: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cross-asset aggregate statistics."""
        import numpy as np

        sharpes, max_dds, rmses, coverages, latencies = [], [], [], [], []
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
            if s.get("conformal_coverage_rate") is not None:
                coverages.append(s["conformal_coverage_rate"])
            if s.get("mean_inference_latency_ms") is not None:
                latencies.append(s["mean_inference_latency_ms"])

        def _mean(lst):
            return round(float(np.mean(lst)), 4) if lst else None

        return {
            "n_assets_evaluated": len([v for v in per_asset.values() if "error" not in v]),
            "mean_sharpe_ratio": _mean(sharpes),
            "mean_max_drawdown": _mean(max_dds),
            "mean_rmse_brier": _mean(rmses),
            "mean_conformal_coverage": _mean(coverages),
            "mean_inference_latency_ms": _mean(latencies),
        }


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
