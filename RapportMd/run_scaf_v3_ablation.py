"""
SCAF v3 — Ablation study
========================
Answers the question raised by ``baselines_report.json``:

    > ``Donchian Only`` has Sharpe 7.46 while SCAF v3 has Sharpe 6.38.
    > Which SCAF component is actually *degrading* performance?

Mechanism
---------
The driver loads data + trained models once, then *replays* the final OOS
window (2022-2024) under seven ablation variants by surgically disabling
components of the pipeline without re-training or re-optimising anything:

  full_scaf         reference — everything enabled
  no_ml_regime      regime scalar forced to 1.0 (s_bull=s_bear=s_side=1)
  no_cs             cross-section leg disabled (w_cs=0, w_trend=1)
  trend_only        w_trend=1, w_cs=0, regime off, full risk
  no_vol_target     vol-targeting layer disabled
  no_dd_ladder      drawdown-tier scaling disabled
  no_risk           RiskManager skipped entirely
  donchian_only     bare TrendEngine on SPY, same don_win/tf_win

Outputs
-------
  results/scaf_v3_<RUN_ID>/ablation_report.json
  results/scaf_v3_<RUN_ID>/dashboard/10_ablation_study.png (if matplotlib)

The driver is lightweight: ≈1 minute per variant on the cached data, no
Optuna, no ML re-training.

Usage
-----
    cd 07-04-2026
    python run_scaf_v3_ablation.py
    # or target an existing run directory:
    python run_scaf_v3_ablation.py --run-dir results/scaf_v3_20260416_133210
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis.cost_model import resolve as _resolve_costs
from analysis.run_metadata import collect as _collect_metadata, set_global_seed

from scaf_v3.loader   import MultiAssetLoader, align_universe
from scaf_v3.features import build_feature_panel
from scaf_v3.models   import RegimeFilter, CrossSectionRanker
from scaf_v3.strategy import (TrendEngine, CrossSectionPortfolio,
                               HybridSignal, portfolio_metrics)
from scaf_v3.risk     import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scaf_ablation")

# ── Config — mirrors run_scaf_v3.py ───────────────────────────────── #
START        = "2015-01-01"
END          = "2024-12-31"
ML_TRAIN_END = "2019-12-31"
TEST_START   = "2022-01-01"
HORIZON      = 5
BENCHMARK    = "SPY"
SEED         = int(os.environ.get("SCAF_SEED", "42"))
COST_MODEL   = _resolve_costs(profile_name=os.environ.get("SCAF_COST_PROFILE"))


# ── Variant definitions ───────────────────────────────────────────── #

@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    use_regime: bool = True
    use_cs: bool = True
    use_trend: bool = True
    use_vol_target: bool = True
    use_dd_ladder: bool = True
    use_risk_manager: bool = True
    donchian_only: bool = False


VARIANTS: List[Variant] = [
    Variant("full_scaf",      "Toutes les couches actives (référence)"),
    Variant("no_ml_regime",   "Regime scalar = 1.0 partout",       use_regime=False),
    Variant("no_cs",          "Cross-section désactivé",           use_cs=False),
    Variant("trend_only",     "Trend seul, sans régime ni CS",
            use_regime=False, use_cs=False),
    Variant("no_vol_target",  "Vol-targeting désactivé",           use_vol_target=False),
    Variant("no_dd_ladder",   "DD ladder désactivé",               use_dd_ladder=False),
    Variant("no_risk",        "RiskManager totalement ignoré",     use_risk_manager=False),
    Variant("donchian_only",  "Baseline Donchian pur (pas de ML)",
            donchian_only=True, use_regime=False, use_cs=False, use_risk_manager=False),
]


# ── Replay one variant ────────────────────────────────────────────── #

def _replay(
    variant: Variant,
    *,
    closes: pd.DataFrame,
    features: Dict[str, pd.DataFrame],
    bm_features: pd.DataFrame,
    regime_filter: RegimeFilter,
    cs_ranker: CrossSectionRanker,
    best_params: Dict[str, Any],
    test_idx: np.ndarray,
    fee: float,
) -> np.ndarray:
    bm_closes = closes[BENCHMARK] if BENCHMARK in closes.columns \
                else closes.iloc[:, 0]
    engine = TrendEngine(
        don_win=int(best_params["don_win"]),
        tf_win=int(best_params["tf_win"]),
        w_don=0.5, w_tf=0.5,
    )
    trend_sig = engine.compute(bm_closes)

    if variant.donchian_only:
        # Pure directional trend on SPY, no ML, no risk layers
        trend_s   = trend_sig.iloc[test_idx].values
        log_rets  = np.log(bm_closes / bm_closes.shift(1)).fillna(0).iloc[test_idx].values
        pos       = np.clip(trend_s, -1.0, 1.0)
        prev      = np.concatenate([[0.0], pos[:-1]])
        turnover  = np.abs(pos - prev)
        return pos * log_rets - fee * turnover

    # Regime probas — disabled variants get a constant 0.5 (neutral)
    if variant.use_regime:
        bm_X     = bm_features.iloc[test_idx].fillna(0).values.astype(np.float32)
        ml_proba = regime_filter.predict_proba(bm_X)
    else:
        ml_proba = np.full(len(test_idx), 0.5, dtype=float)

    cs_pred = cs_ranker.predict(features, test_idx)

    cs_port = CrossSectionPortfolio(
        long_thr=float(best_params["ml_thr"]),
        short_thr=1.0 - float(best_params["ml_thr"]),
        max_positions=4,
    )
    cs_w = cs_port.compute(cs_pred)

    # Per-variant HybridSignal configuration. Phase C: regime scalars
    # may be absent from newer best_params JSONs — default to 1.0.
    if variant.use_regime:
        s_bull = float(best_params.get("s_bull", 1.0))
        s_bear = float(best_params.get("s_bear", 1.0))
        s_side = float(best_params.get("s_side", 1.0))
    else:
        s_bull = s_bear = s_side = 1.0

    hybrid = HybridSignal(
        w_trend = float(best_params["w_trend"]) if variant.use_trend else 0.0,
        w_cs    = float(best_params["w_cs"])    if variant.use_cs    else 0.0,
        ml_thr  = float(best_params["ml_thr"]),
        s_bull=s_bull, s_bear=s_bear, s_side=s_side,
        max_pos = float(best_params.get("max_pos", 1.0)),
        transaction_cost=fee,
    )
    pnl_raw, _ = hybrid.compute_returns(closes, trend_sig, cs_w, ml_proba, test_idx)

    if not variant.use_risk_manager:
        return pnl_raw

    target_vol = float(best_params.get("target_vol", 0.10)) \
                 if variant.use_vol_target else 1e6
    dd_scales  = (float(best_params.get("dd_s1", 0.70)),
                  float(best_params.get("dd_s2", 0.35)),
                  float(best_params.get("dd_s3", 0.10))) \
                 if variant.use_dd_ladder else (1.0, 1.0, 1.0)

    rm = RiskManager(
        target_vol=target_vol,
        vol_window=int(best_params.get("vol_window", 20)),
        dd_tiers=(float(best_params.get("dd1", 0.05)),
                  float(best_params.get("dd2", 0.10)),
                  float(best_params.get("dd3", 0.15))),
        dd_scales=dd_scales,
    )
    return rm.apply(pnl_raw, pnl_raw)


# ── Main driver ───────────────────────────────────────────────────── #

def _default_run_dir() -> Path:
    """Pick the most-recent results/scaf_v3_* directory if it exists."""
    root = Path("results")
    if not root.exists():
        return Path("results") / f"scaf_v3_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    cands = sorted([p for p in root.glob("scaf_v3_*") if p.is_dir()])
    return cands[-1] if cands else root / f"scaf_v3_ablation_{time.strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=_default_run_dir(),
                    help="Directory containing scaf_v3_report.json")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of variants to run (default: all)")
    args = ap.parse_args()

    set_global_seed(SEED)
    t0 = time.time()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "scaf_v3_report.json"
    if not report_path.exists():
        log.error("No scaf_v3_report.json at %s — run run_scaf_v3.py first",
                  report_path)
        sys.exit(2)
    scaf_report = json.loads(report_path.read_text(encoding="utf-8"))
    best_params: Dict[str, Any] = scaf_report["best_params"]
    log.info("Replaying ablations from %s", report_path)

    # ── Data + features ────────────────────────────────────────────
    log.info("[1/3] Loading data + building features (cached)...")
    loader = MultiAssetLoader(start=START, end=END, use_cache=True)
    ohlcv, sentiment = loader.load()
    ohlcv, sentiment, index = align_universe(ohlcv, sentiment)
    features, targets = build_feature_panel(ohlcv, sentiment, index, horizon=HORIZON)

    closes = pd.DataFrame(
        {t: df["Close"] for t, df in ohlcv.items()}
    ).reindex(index).ffill()

    bm_ticker   = BENCHMARK if BENCHMARK in features else list(features.keys())[0]
    bm_features = features[bm_ticker]

    ml_train_idx = np.where(index <= ML_TRAIN_END)[0]
    test_idx     = np.where(index >= TEST_START)[0]
    log.info("  ML train: %d d | OOS test: %d d", len(ml_train_idx), len(test_idx))

    # ── Train ML models — once, shared across variants ─────────────
    log.info("[2/3] Training RegimeFilter + CrossSectionRanker...")
    rf = RegimeFilter(threshold=0.50)
    n_cal = max(100, len(ml_train_idx) // 5)
    bm_X_tr = bm_features.iloc[ml_train_idx].fillna(0).values.astype(np.float32)
    bm_y_tr = targets[bm_ticker].iloc[ml_train_idx].fillna(0).values.astype(int)
    rf.fit(bm_X_tr[:-n_cal], bm_y_tr[:-n_cal],
           bm_X_tr[-n_cal:],  bm_y_tr[-n_cal:])
    cs = CrossSectionRanker(threshold=0.55)
    cs.fit(features, targets, ml_train_idx)

    # ── Replay variants ────────────────────────────────────────────
    log.info("[3/3] Replaying ablation variants...")
    selected = [v for v in VARIANTS if not args.variants or v.name in args.variants]
    fee = COST_MODEL.fee
    rows: List[Dict[str, Any]] = []
    pnls: Dict[str, List[float]] = {}

    for v in selected:
        t1 = time.time()
        pnl = _replay(
            v, closes=closes, features=features, bm_features=bm_features,
            regime_filter=rf, cs_ranker=cs, best_params=best_params,
            test_idx=test_idx, fee=fee,
        )
        m = portfolio_metrics(pnl)
        pnls[v.name] = [float(x) for x in pnl]
        row = {"variant": v.name, "description": v.description,
               "n_days": int(len(pnl)), **m,
               "elapsed_s": round(time.time() - t1, 2)}
        rows.append(row)
        log.info("  %-15s  Sharpe=%7.4f  DD=%7.4f  CumRet=%7.4f  %.1fs",
                 v.name, m.get("sharpe", 0), m.get("max_dd", 0),
                 m.get("cum_ret", 0), row["elapsed_s"])

    # ── Rank + deltas vs full_scaf ─────────────────────────────────
    full = next((r for r in rows if r["variant"] == "full_scaf"), None)
    if full is not None:
        base_sr, base_cr = full.get("sharpe", 0), full.get("cum_ret", 0)
        for r in rows:
            if r["variant"] == "full_scaf":
                continue
            r["sharpe_delta_vs_full"]  = round(
                r.get("sharpe", 0) - base_sr, 4)
            r["cum_ret_delta_vs_full"] = round(
                r.get("cum_ret", 0) - base_cr, 6)

    # ── Save JSON ──────────────────────────────────────────────────
    report = {
        "schema":         "scaf_v3.ablation_report.v1",
        "run_dir":        str(run_dir),
        "test_period":    scaf_report.get("test_results", {}).get("period"),
        "best_params":    best_params,
        "cost_model":     COST_MODEL.to_dict(),
        "variants":       rows,
        "pnl_series":     pnls,
        "run_metadata":   _collect_metadata(
            seed=SEED,
            config={"test_start": TEST_START, "cost_model": COST_MODEL.to_dict()},
            repo_root=Path(__file__).parent,
        ),
        "elapsed_s":      round(time.time() - t0, 1),
    }
    out = run_dir / "ablation_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s", out)

    # ── Plot ───────────────────────────────────────────────────────
    try:
        _plot_ablation(report, run_dir / "dashboard")
    except Exception as exc:
        log.warning("Plot failed (non-blocking): %s", exc)


def _plot_ablation(report: Dict[str, Any], dash_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dash_dir.mkdir(parents=True, exist_ok=True)
    rows = report["variants"]
    names  = [r["variant"] for r in rows]
    shrps  = [r.get("sharpe", 0) for r in rows]
    cmrets = [r.get("cum_ret", 0) for r in rows]
    mdds   = [abs(r.get("max_dd", 0)) * 100 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vals, title, ylabel in zip(
        axes, [shrps, cmrets, mdds],
        ["Sharpe Ratio", "Cum. Return", "|Max DD| (%)"],
        ["Sharpe", "Log-return cumulé", "Drawdown (%)"],
    ):
        bars = ax.bar(names, vals, color="#2563eb", edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_ha("right")

    fig.suptitle("SCAF v3 — Ablation Study (OOS 2022-2024)", fontsize=13)
    fig.tight_layout()
    fig.savefig(dash_dir / "10_ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
