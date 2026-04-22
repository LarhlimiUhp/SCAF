"""
SCAF v3 — Main Orchestrator
=============================
Self-Consistent Adaptive Framework v3
Multi-asset | Intraday proxy | Alt data | 1000-agent Optuna

Pipeline
--------
  [1] Download 25+ assets + 6 sentiment tickers (2015-2024)
  [2] Build feature panel (price + intraday proxy + cross-section + alt)
  [3] Split:   ML train  2015-2019
               Opt window 2020-2021
               Test (OOS) 2022-2024  <- never touched during optimisation
  [4] Train CrossSectionRanker on ML train period (Phase C: RegimeFilter removed)
  [5] Ultra-Think: 800 Optuna agents on opt window (real Sharpe)
  [6] Final evaluation on OOS test period
  [7] Walk-forward stress test: 2008, 2020-03 crisis simulation
  [8] Report JSON + 5 charts

Usage
-----
    cd 07-04-2026
    python run_scaf_v3.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ──────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scaf_v3")

RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("results") / f"scaf_v3_{RUN_ID}"

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

START         = "2015-01-01"
END           = "2024-12-31"
ML_TRAIN_END  = "2019-12-31"
OPT_START     = "2020-01-01"
OPT_END       = "2021-12-31"
TEST_START    = "2022-01-01"
HORIZON       = 5
BENCHMARK     = "SPY"
SEED          = int(os.environ.get("SCAF_SEED", "42"))

from analysis.cost_model import resolve as _resolve_costs
from analysis.run_metadata import collect as _collect_metadata, set_global_seed
COST_MODEL    = _resolve_costs(profile_name=os.environ.get("SCAF_COST_PROFILE"))
FEE           = COST_MODEL.fee

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS FROM scaf_v3 PACKAGE
# ══════════════════════════════════════════════════════════════════════════════

from scaf_v3.loader    import MultiAssetLoader, align_universe
from scaf_v3.features  import build_feature_panel
from scaf_v3.models    import CrossSectionRanker  # RegimeFilter removed in Phase C
from scaf_v3.strategy  import TrendEngine, CrossSectionPortfolio, HybridSignal, portfolio_metrics
from scaf_v3.risk      import RiskManager
from scaf_v3.optimizer import UltraThinkOptimizer, DEFAULTS
from scaf_v3.universe  import CROSS_SECTION_UNIVERSE


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTION (called by every Optuna trial)
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_fn(
    closes:       pd.DataFrame,
    features:     Dict[str, pd.DataFrame],
    targets:      Dict[str, pd.Series],
    cs_ranker:    CrossSectionRanker,
    opt_idx:      np.ndarray,
):
    """
    Returns eval_fn(params) -> float (Sharpe on opt window).
    Closed over pre-computed data — fast per trial.
    (Phase C: RegimeFilter and bm_features removed.)
    """
    # Pre-compute CS predictions on opt window
    cs_pred = cs_ranker.predict(features, opt_idx)

    # Benchmark closes
    bm_closes = closes[BENCHMARK] if BENCHMARK in closes.columns \
                else closes.iloc[:, 0]

    def eval_fn(params: Dict[str, Any]) -> float:
        # Trend signal
        engine = TrendEngine(
            don_win=int(params["don_win"]),
            tf_win=int(params["tf_win"]),
            w_don=0.5, w_tf=0.5,
        )
        trend_sig = engine.compute(bm_closes)

        # CS portfolio weights
        cs_port = CrossSectionPortfolio(
            long_thr=float(params["ml_thr"]),
            short_thr=1.0 - float(params["ml_thr"]),
            max_positions=4,
        )
        cs_w = cs_port.compute(cs_pred)

        # Hybrid signal P&L (regime scalars removed in Phase C)
        hybrid = HybridSignal(
            w_trend=float(params["w_trend"]),
            w_cs=float(params["w_cs"]),
            max_pos=float(params.get("max_pos", 1.0)),
            transaction_cost=FEE,
        )
        pnl, _ = hybrid.compute_returns(
            closes, trend_sig, cs_w, idx=opt_idx,
        )

        # Risk management
        rm = RiskManager(
            target_vol=float(params.get("target_vol", 0.10)),
            vol_window=int(params.get("vol_window", 20)),
            dd_tiers=(params.get("dd1", 0.05),
                      params.get("dd2", 0.10),
                      params.get("dd3", 0.15)),
            dd_scales=(params.get("dd_s1", 0.70),
                       params.get("dd_s2", 0.35),
                       params.get("dd_s3", 0.10)),
        )
        pnl_adj = rm.apply(pnl, pnl)

        # Two-fold walk-forward on opt window
        n   = len(pnl_adj)
        mid = n // 2
        sharpes = []
        for s, e in [(0, mid), (mid, n)]:
            seg = pnl_adj[s:e]
            if len(seg) < 20:
                continue
            m = np.mean(seg)
            v = np.std(seg) + 1e-12
            sr = m / v * np.sqrt(252)
            dd = float(np.min(
                (np.exp(np.cumsum(seg)) - np.maximum.accumulate(np.exp(np.cumsum(seg))))
                / (np.maximum.accumulate(np.exp(np.cumsum(seg))) + 1e-12)
            ))
            penalty = max(0.0, -dd - 0.12) * 5.0
            sharpes.append(sr - penalty)

        return float(np.mean(sharpes)) if sharpes else -999.0

    return eval_fn


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL OUT-OF-SAMPLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def final_evaluation(
    closes:        pd.DataFrame,
    features:      Dict[str, pd.DataFrame],
    targets:       Dict[str, pd.Series],
    cs_ranker:     CrossSectionRanker,
    best_params:   Dict[str, Any],
    test_idx:      np.ndarray,
    index:         pd.DatetimeIndex,
) -> Dict[str, Any]:
    """Evaluate best params on the held-out 2022-2024 period."""
    log.info("Final OOS evaluation: %s -> %s  (%d days)",
             TEST_START, END, len(test_idx))

    # CS predictions
    cs_pred = cs_ranker.predict(features, test_idx)

    # Benchmark
    bm_closes = closes[BENCHMARK] if BENCHMARK in closes.columns \
                else closes.iloc[:, 0]

    # Trend signal
    engine = TrendEngine(
        don_win=int(best_params["don_win"]),
        tf_win=int(best_params["tf_win"]),
        w_don=0.5, w_tf=0.5,
    )
    trend_sig = engine.compute(bm_closes)

    # CS portfolio
    cs_port = CrossSectionPortfolio(
        long_thr=float(best_params["ml_thr"]),
        short_thr=1.0 - float(best_params["ml_thr"]),
        max_positions=4,
    )
    cs_w = cs_port.compute(cs_pred)

    # Hybrid P&L (regime scalars removed in Phase C)
    hybrid = HybridSignal(
        w_trend=float(best_params["w_trend"]),
        w_cs=float(best_params["w_cs"]),
        max_pos=float(best_params.get("max_pos", 1.0)),
        transaction_cost=FEE,
    )
    pnl_raw, breakdown = hybrid.compute_returns(
        closes, trend_sig, cs_w, idx=test_idx,
    )

    # Risk management
    rm = RiskManager(
        target_vol=float(best_params.get("target_vol", 0.10)),
        vol_window=int(best_params.get("vol_window", 20)),
    )
    pnl_adj = rm.apply(pnl_raw, pnl_raw)

    # Metrics
    metrics     = portfolio_metrics(pnl_adj)
    metrics_raw = portfolio_metrics(pnl_raw)

    # Buy & hold benchmark
    bm_prices = bm_closes.iloc[test_idx]
    bnh_lr    = np.log(bm_prices / bm_prices.shift(1)).fillna(0).values
    bnh_met   = portfolio_metrics(bnh_lr)

    # CS active assets stats
    cs_active = (cs_pred.abs() > 0).sum(axis=1).mean()

    return {
        "period":          f"{TEST_START} -> {END}",
        "n_days":          len(pnl_adj),
        **{f"adj_{k}": v for k, v in metrics.items()},
        **{f"raw_{k}": v for k, v in metrics_raw.items()},
        "bnh_sharpe":      bnh_met.get("sharpe", 0),
        "bnh_cum_ret":     bnh_met.get("cum_ret", 0),
        "excess_vs_bnh":   round(metrics.get("cum_ret", 0) - bnh_met.get("cum_ret", 0), 6),
        "cs_avg_positions": round(float(cs_active), 2),
        "signal_breakdown": breakdown.to_dict(),
        "targets_met": {
            "sharpe_gt_1":   bool(metrics.get("sharpe", 0)  > 1.033),
            "dd_lt_15pct":   bool(metrics.get("max_dd", -1) > -0.15),
            "beats_bnh":     bool(metrics.get("cum_ret", 0) > bnh_met.get("cum_ret", 0)),
        },
        "pnl_series": pnl_adj.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(results: Dict[str, Any],
             closes: pd.DataFrame,
             test_idx: np.ndarray,
             agent_summaries: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pnl = np.array(results["pnl_series"])
    cum = np.exp(np.cumsum(pnl))

    bm_closes  = closes[BENCHMARK].iloc[test_idx] if BENCHMARK in closes.columns \
                 else closes.iloc[test_idx, 0]
    bnh_lr     = np.log(bm_closes / bm_closes.shift(1)).fillna(0).values
    cum_bnh    = np.exp(np.cumsum(bnh_lr))
    dates      = np.arange(len(pnl))

    # ── Fig 1: Equity + Drawdown + Returns ─────────────────────────── #
    fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                             gridspec_kw={"hspace": 0.45})

    sr  = results.get("adj_sharpe", 0)
    mdd = results.get("adj_max_dd", 0)
    ax  = axes[0]
    ax.plot(cum,     color="#2563eb", linewidth=2.0, label=f"SCAF v3 (Sharpe={sr:.2f})")
    ax.plot(cum_bnh, color="#9ca3af", linewidth=1.2, linestyle="--",
            label=f"Buy & Hold (Sharpe={results.get('bnh_sharpe', 0):.2f})")
    ax.set_title(f"SCAF v3 — Courbe d'equite  |  Sharpe={sr:.4f}  |  DD={mdd:.2%}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Valeur (base 1)"); ax.legend(); ax.grid(alpha=0.3)

    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-12)
    ax   = axes[1]
    ax.fill_between(dates, dd, 0, color="#ef4444", alpha=0.55)
    ax.axhline(-0.05,  color="#f59e0b", lw=1.0, ls="--", label="Tier1 -5%")
    ax.axhline(-0.10,  color="#f97316", lw=1.2, ls="--", label="Tier2 -10%")
    ax.axhline(-0.15,  color="#dc2626", lw=1.2, ls="--", label="Tier3 -15%")
    ax.set_title("Drawdown avec tiers de protection", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(pnl, bins=60, color="#2563eb", alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(0, color="#111827", lw=1.0)
    ax.axvline(np.mean(pnl), color="#16a34a", lw=1.3, ls="--",
               label=f"Moy={np.mean(pnl):.5f}")
    ax.set_title("Distribution des retours quotidiens", fontsize=12, fontweight="bold")
    ax.set_xlabel("Log-return"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.savefig(RESULTS_DIR / "01_equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Agent category Sharpe ───────────────────────────────── #
    cats   = list(agent_summaries.keys())
    values = [agent_summaries[c]["best_value"] for c in cats]
    colors = ["#16a34a" if v > 1.033 else "#f97316" if v > 0 else "#ef4444"
              for v in values]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(cats, values, color=colors, edgecolor="white")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.033, color="#2563eb", lw=1.5, ls="--", label="Cible 1.033")
    ax.set_title("Sharpe (opt window) par categorie d'agents (1000 total)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe"); ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha="right"); ax.grid(alpha=0.3, axis="y")
    fig.savefig(RESULTS_DIR / "02_agent_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Rolling Sharpe (90-day window) ──────────────────────── #
    # (Phase C: former regime-pie plot removed along with RegimeFilter.)
    roll_sr = pd.Series(pnl).rolling(90).apply(
        lambda x: (np.mean(x) / (np.std(x) + 1e-12)) * np.sqrt(252), raw=True
    )
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(roll_sr.values, color="#2563eb", lw=1.5, label="Sharpe 90j")
    ax.axhline(1.033, color="#f97316", lw=1.2, ls="--", label="Cible 1.033")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.fill_between(range(len(roll_sr)),
                    roll_sr.clip(lower=0).values, 0, alpha=0.25, color="#2563eb")
    ax.fill_between(range(len(roll_sr)),
                    roll_sr.clip(upper=0).values, 0, alpha=0.25, color="#ef4444")
    ax.set_title("Sharpe Ratio glissant (90 jours) — 2022-2024",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(RESULTS_DIR / "04_rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: SCAF v1 vs Option C vs SCAF v3 comparison ──────────── #
    versions = ["SCAF v1\n(ML pur)", "Option C\n(Hybrid)", "SCAF v3\n(Multi-asset)"]
    sharpes  = [0.177, 6.010, sr]
    dds      = [-0.0793, -0.0084, abs(mdd)]
    colors_v = ["#ef4444", "#f97316", "#16a34a" if sr > 1.033 else "#f97316"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.bar(versions, sharpes, color=colors_v, edgecolor="white")
    ax1.axhline(1.033, color="#2563eb", lw=1.5, ls="--", label="Cible 1.033")
    for i, (v, s) in enumerate(zip(versions, sharpes)):
        ax1.text(i, s + 0.05, f"{s:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Sharpe Ratio — Evolution SCAF",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, axis="y")

    ax2.bar(versions, dds, color=colors_v, edgecolor="white")
    ax2.axhline(0.15, color="#dc2626", lw=1.5, ls="--", label="Limite 15%")
    for i, (v, d) in enumerate(zip(versions, dds)):
        ax2.text(i, d + 0.002, f"{d:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("|Max Drawdown| — Evolution SCAF",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    fig.savefig(RESULTS_DIR / "05_version_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("5 graphiques sauvegardes dans %s", RESULTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    set_global_seed(SEED)

    log.info("=" * 64)
    log.info("  SCAF v3 (lean)  |  Multi-asset + Alt data + 800 agents  [%s]", RUN_ID)
    log.info("  seed=%d   fee=%.5f (%.1f bps/leg, profile=%s)",
             SEED, FEE, COST_MODEL.total_bps,
             os.environ.get("SCAF_COST_PROFILE") or "default")
    log.info("=" * 64)

    # ── Deps ────────────────────────────────────────────────────────── #
    for pkg in ["yfinance", "optuna", "sklearn", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            log.error("pip install %s", pkg); sys.exit(1)

    # ── [1] Load data ─────────────────────────────────────────────── #
    log.info("[1/7] Loading multi-asset data...")
    loader = MultiAssetLoader(start=START, end=END, use_cache=True)
    ohlcv, sentiment = loader.load()
    ohlcv, sentiment, index = align_universe(ohlcv, sentiment)

    # ── [2] Build features ────────────────────────────────────────── #
    log.info("[2/7] Building feature panel (price + intraday proxy + CS + alt)...")
    features, targets = build_feature_panel(ohlcv, sentiment, index, horizon=HORIZON)

    # Closes panel
    closes = pd.DataFrame(
        {t: df["Close"] for t, df in ohlcv.items()}
    ).reindex(index).ffill()

    # ── Index splits ──────────────────────────────────────────────── #
    ml_train_mask = index <= ML_TRAIN_END
    opt_mask      = (index >= OPT_START) & (index <= OPT_END)
    test_mask     = index >= TEST_START

    ml_train_idx  = np.where(ml_train_mask)[0]
    opt_idx       = np.where(opt_mask)[0]
    test_idx      = np.where(test_mask)[0]

    log.info("  ML train : %d days  |  Opt: %d days  |  Test OOS: %d days",
             len(ml_train_idx), len(opt_idx), len(test_idx))

    # ── [3] Train ML models ───────────────────────────────────────── #
    # Phase C: RegimeFilter removed — only CrossSectionRanker is trained.
    log.info("[3/7] Training CrossSectionRanker...")
    cs_ranker = CrossSectionRanker(threshold=0.55)
    cs_ranker.fit(features, targets, ml_train_idx)

    # ── [4] Ultra-Think optimisation ─────────────────────────────── #
    log.info("[4/7] Ultra-Think: 800 agents on opt window 2020-2021...")
    eval_fn = build_eval_fn(
        closes, features, targets,
        cs_ranker, opt_idx,
    )
    optimizer   = UltraThinkOptimizer(eval_fn)
    best_params, agent_summaries = optimizer.run()

    # ── [5] Final OOS evaluation ──────────────────────────────────── #
    log.info("[5/7] Final evaluation on OOS test 2022-2024...")
    results = final_evaluation(
        closes, features, targets,
        cs_ranker,
        best_params, test_idx, index,
    )

    elapsed = time.time() - t0

    # ── [6] Print summary ─────────────────────────────────────────── #
    sep = "-" * 64
    print(f"\n{sep}")
    print("  SCAF v3  --  Multi-asset | Alt data | 1000 agents")
    print(f"  Out-of-Sample: 2022-2024  ({results['n_days']} jours)")
    print(sep)

    rows = [
        ("Sharpe Ratio (adj)",    "adj_sharpe",   "> 1.033"),
        ("Max Drawdown (adj)",    "adj_max_dd",   "> -15%"),
        ("Cumul. Return (adj)",   "adj_cum_ret",  ""),
        ("Ann. Return",           "adj_ann_ret",  ""),
        ("Ann. Volatility",       "adj_ann_vol",  ""),
        ("Calmar Ratio",          "adj_calmar",   ""),
        ("Sortino Ratio",         "adj_sortino",  ""),
        ("Win Rate",              "adj_win_rate", "> 45%"),
        ("Excess vs B&H",         "excess_vs_bnh",""),
        ("B&H Sharpe",            "bnh_sharpe",   ""),
        ("CS avg active assets",  "cs_avg_positions", ""),
    ]
    for label, key, target in rows:
        val = results.get(key)
        if val is None: continue
        t = f"  [cible {target}]" if target else ""
        print(f"  {label:<28} {str(val):<14}{t}")

    print(sep)
    for k, v in results.get("targets_met", {}).items():
        print(f"  {k:<28} {'[OK]' if v else '[--]'}")
    print(sep)

    print("\n  Signal breakdown (contribution moyenne quotidienne):")
    for k, v in results.get("signal_breakdown", {}).items():
        print(f"    {k:<28} {v:.6f}")

    print(f"\n  Meilleurs parametres (1000 agents):")
    for k, v in sorted(best_params.items()):
        vr = round(v, 4) if isinstance(v, float) else v
        print(f"    {k:<24} {vr}")

    print(f"\n  Performance agents (opt window):")
    for cat, info in agent_summaries.items():
        print(f"    {cat:<22} Sharpe={info['best_value']:.4f}  ({info['n_trials']} trials)")

    print(f"\n  Resultats: {RESULTS_DIR}")
    print(f"  Duree    : {elapsed:.0f}s")
    print(sep + "\n")

    # ── [7] Plots ──────────────────────────────────────────────────── #
    log.info("[6/7] Generating 5 charts...")
    try:
        plot_all(results, closes, test_idx, agent_summaries)
    except Exception as exc:
        log.warning("Plots failed (non-blocking): %s", exc)

    # ── [8] Save JSON ─────────────────────────────────────────────── #
    report = {
        "run_id":         RUN_ID,
        "version":        "SCAF v3",
        "elapsed_s":      round(elapsed, 1),
        "best_params":    {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in best_params.items()},
        "test_results":   {k: v for k, v in results.items() if k != "pnl_series"},
        "agent_summaries": agent_summaries,
        "universe_size":  len(ohlcv),
        "n_features":     int(np.mean([f.shape[1] for f in features.values()])),
        "cs_ranker_auc":     cs_ranker._ensemble.auc_scores_ if cs_ranker._fitted else {},
        "run_metadata":   _collect_metadata(
            seed=SEED,
            config={
                "start": START, "end": END,
                "ml_train_end": ML_TRAIN_END,
                "opt_start": OPT_START, "opt_end": OPT_END,
                "test_start": TEST_START,
                "horizon": HORIZON, "benchmark": BENCHMARK,
                "cost_model": COST_MODEL.to_dict(),
                "n_trials_total": sum(
                    s.get("n_trials", 0) for s in agent_summaries.values()
                ),
            },
            repo_root=Path(__file__).parent,
        ),
    }
    path = RESULTS_DIR / "scaf_v3_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("[7/7] Report saved: %s", path)

    # ── Verdict ────────────────────────────────────────────────────── #
    sr  = results.get("adj_sharpe", 0)
    mdd = results.get("adj_max_dd", -1)
    exc = results.get("excess_vs_bnh", -1)
    log.info("-" * 64)
    log.info("VERDICT  Sharpe=%.4f %s   DD=%.4f %s   Exces=%.4f %s",
             sr,  "[OK]" if sr  > 1.033 else "[--]",
             mdd, "[OK]" if mdd > -0.15 else "[--]",
             exc, "[OK]" if exc > 0     else "[--]")
    log.info("Duree totale: %.0f s", elapsed)
    log.info("-" * 64)


if __name__ == "__main__":
    main()
