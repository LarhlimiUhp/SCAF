"""
SCAF v3 — Walk-Forward + Deflated Sharpe
========================================
Answers the two outstanding scientific concerns raised in the diagnostic
report:

  * selection bias  — the original pipeline picks the best of 1000 trials
                      on a single 2020-2021 window and reports that Sharpe
                      as if it were unbiased.
  * label leakage   — trials use forward-looking horizon labels, and the
                      single train/test split has no purge/embargo.

Design
------
Anchored 5-fold walk-forward with a *purge + embargo* buffer of
``HORIZON + 10`` days between windows:

    Fold  ML train (end)   Optuna window   Test (OOS)
       1  2018-12-31       2019            2020
       2  2019-12-31       2020            2021
       3  2020-12-31       2021            2022
       4  2021-12-31       2022            2023
       5  2022-12-31       2023            2024

Per fold we re-train :class:`CrossSectionRanker` on the ML window only
(``RegimeFilter`` was removed in Phase C), run :class:`UltraThinkOptimizer` with a
proportionally scaled trial budget on the Opt window, and evaluate the
best params on the Test window. Out-of-sample P&L series from every fold
are then concatenated and fed to
:func:`analysis.walk_forward.walk_forward_deflated_sharpe`, which applies
the Bailey & López de Prado (2014) Deflated Sharpe correction using the
*total* number of trials explored across all folds.

Usage
-----
    cd 07-04-2026
    python run_scaf_v3_walkforward.py                        # 300 trials/fold
    python run_scaf_v3_walkforward.py --trials-per-fold 500
    python run_scaf_v3_walkforward.py --folds 1 3 5          # subset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from analysis.cost_model   import resolve as _resolve_costs
from analysis.run_metadata import collect as _collect_metadata, set_global_seed
from analysis.walk_forward import walk_forward_deflated_sharpe
from analysis.robustness   import stationary_bootstrap_sharpe_ci, jackknife_top_k

from scaf_v3 import optimizer as _opt_mod
from scaf_v3.loader    import MultiAssetLoader, align_universe
from scaf_v3.features  import build_feature_panel
from scaf_v3.models    import CrossSectionRanker  # RegimeFilter removed in Phase C
from scaf_v3.optimizer import UltraThinkOptimizer
from scaf_v3.strategy  import portfolio_metrics

# Reuse the existing eval / final-evaluation helpers
from run_scaf_v3 import build_eval_fn, final_evaluation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scaf_walkforward")


# ── Config ────────────────────────────────────────────────────────── #
START        = "2015-01-01"
END          = "2024-12-31"
HORIZON      = 5
EMBARGO      = 10
BENCHMARK    = "SPY"
SEED         = int(os.environ.get("SCAF_SEED", "42"))
COST_MODEL   = _resolve_costs(profile_name=os.environ.get("SCAF_COST_PROFILE"))


@dataclass(frozen=True)
class Fold:
    idx:           int
    ml_train_end:  str
    opt_start:     str
    opt_end:       str
    test_start:    str
    test_end:      str


FOLDS: List[Fold] = [
    Fold(1, "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    Fold(2, "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    Fold(3, "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    Fold(4, "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    Fold(5, "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
]


def _scaled_agent_categories(trials_per_fold: int) -> Dict[str, int]:
    """Scale AGENT_CATEGORIES proportionally to hit *trials_per_fold* total."""
    base = dict(_opt_mod.AGENT_CATEGORIES)
    total_base = sum(base.values())
    scale = trials_per_fold / max(total_base, 1)
    scaled = {k: max(10, int(round(v * scale))) for k, v in base.items()}
    return scaled


def _run_one_fold(
    fold:         Fold,
    *,
    ohlcv:        Dict[str, pd.DataFrame],
    sentiment:    pd.DataFrame,
    index:        pd.DatetimeIndex,
    features:     Dict[str, pd.DataFrame],
    targets:      Dict[str, pd.Series],
    closes:       pd.DataFrame,
    trials_per_fold: int,
    seed:         int,
) -> Dict[str, Any]:
    """Train ML → optimise → evaluate on one fold. Returns a dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Purge / embargo applied by advancing ML-end and delaying opt-start
    purge_days = HORIZON + EMBARGO

    ml_train_mask = index <= fold.ml_train_end
    ml_train_idx  = np.where(ml_train_mask)[0]

    opt_idx = np.where((index >= fold.opt_start) & (index <= fold.opt_end))[0]
    if purge_days > 0 and len(opt_idx) > purge_days:
        opt_idx = opt_idx[purge_days:]  # drop first K days after ML end

    test_idx = np.where((index >= fold.test_start) & (index <= fold.test_end))[0]
    if purge_days > 0 and len(test_idx) > purge_days:
        test_idx = test_idx[purge_days:]

    log.info("Fold %d | ML=%dd  Opt=%dd  Test=%dd",
             fold.idx, len(ml_train_idx), len(opt_idx), len(test_idx))

    # ── ML models (fold-specific; RegimeFilter removed in Phase C) ──
    cs = CrossSectionRanker(threshold=0.55)
    cs.fit(features, targets, ml_train_idx)

    # ── Patch AGENT_CATEGORIES for this fold's budget ─────────────
    saved = dict(_opt_mod.AGENT_CATEGORIES)
    _opt_mod.AGENT_CATEGORIES = _scaled_agent_categories(trials_per_fold)
    try:
        eval_fn = build_eval_fn(closes, features, targets, cs, opt_idx)

        # Override Optuna seed offset so folds don't collide
        orig_run = UltraThinkOptimizer.run
        def seeded_run(self) -> Any:
            # patch TPESampler seeds deterministically per fold
            import optuna
            best_value  = -np.inf
            best_params = dict(_opt_mod.DEFAULTS)
            summaries:  Dict[str, Any] = {}
            launched = 0
            all_trial_sharpes: List[float] = []
            for category, n_trials in _opt_mod.AGENT_CATEGORIES.items():
                study = optuna.create_study(
                    direction="maximize",
                    study_name=f"scaf_v3_fold{fold.idx}_{category}",
                    sampler=optuna.samplers.TPESampler(seed=seed + fold.idx * 1000 + launched),
                )
                study.optimize(_opt_mod.make_objective(eval_fn, category),
                               n_trials=n_trials, n_jobs=1, show_progress_bar=False)
                summaries[category] = {
                    "best_value": round(float(study.best_value), 4),
                    "best_params": study.best_params,
                    "n_trials":   len(study.trials),
                }
                all_trial_sharpes.extend(
                    float(t.value) for t in study.trials
                    if t.value is not None and t.value > -900
                )
                if study.best_value > best_value:
                    best_value = study.best_value
                    best_params = {**_opt_mod.DEFAULTS, **study.best_params}
                    tw = best_params.get("w_trend", 0.6) + best_params.get("w_cs", 0.4)
                    best_params["w_trend"] /= tw
                    best_params["w_cs"]    /= tw
                launched += n_trials
            summaries["_all_trial_sharpes"] = all_trial_sharpes
            return best_params, summaries

        # Use the seeded runner directly without subclassing Optuna state
        best_params, summaries = seeded_run(UltraThinkOptimizer(eval_fn))
    finally:
        _opt_mod.AGENT_CATEGORIES = saved

    trial_sharpes = summaries.pop("_all_trial_sharpes", [])

    # ── OOS evaluation on test window ─────────────────────────────
    res = final_evaluation(closes, features, targets,
                           cs, best_params, test_idx, index)
    return {
        "fold":         fold.idx,
        "ml_train_end": fold.ml_train_end,
        "opt_period":   f"{fold.opt_start}→{fold.opt_end}",
        "test_period":  f"{fold.test_start}→{fold.test_end}",
        "ml_train_n":   int(len(ml_train_idx)),
        "opt_n":        int(len(opt_idx)),
        "test_n":       int(len(test_idx)),
        "best_params":  {k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in best_params.items()},
        "agent_summaries": summaries,
        "trial_sharpes_count": len(trial_sharpes),
        "trial_sharpes_stats": {
            "mean":   round(float(np.mean(trial_sharpes)), 4)   if trial_sharpes else None,
            "std":    round(float(np.std(trial_sharpes, ddof=1)), 4) if len(trial_sharpes) > 1 else None,
            "max":    round(float(np.max(trial_sharpes)), 4)    if trial_sharpes else None,
            "p95":    round(float(np.percentile(trial_sharpes, 95)), 4) if trial_sharpes else None,
        },
        "oos_metrics": {k: v for k, v in res.items()
                        if k not in ("pnl_series",) and not k.startswith("regime")},
        "pnl_oos":     [float(x) for x in res["pnl_series"]],
        "trial_sharpes": [round(float(x), 6) for x in trial_sharpes],
    }


# ── Aggregation ──────────────────────────────────────────────────── #

def _aggregate(fold_results: List[Dict[str, Any]],
               trials_per_fold: int) -> Dict[str, Any]:
    fold_pnls   = [np.array(f["pnl_oos"], dtype=float) for f in fold_results]
    all_trials  = [s for f in fold_results for s in f.get("trial_sharpes", [])]

    dsr_block = walk_forward_deflated_sharpe(
        fold_pnls,
        n_trials_per_fold=trials_per_fold,
        sharpe_trials=all_trials if all_trials else None,
    )

    pooled = np.concatenate(fold_pnls) if fold_pnls else np.array([])
    pooled_metrics = portfolio_metrics(pooled) if len(pooled) else {}
    block_ci = (
        stationary_bootstrap_sharpe_ci(pooled, n_boot=2000, seed=SEED)
        if len(pooled) >= 30 else {}
    )
    jk = jackknife_top_k(pooled) if len(pooled) >= 30 else {}

    return {
        "deflated_sharpe":     dsr_block,
        "pooled_oos_metrics":  pooled_metrics,
        "pooled_oos_n_days":   int(len(pooled)),
        "pooled_bootstrap_ci": block_ci,
        "pooled_jackknife":    jk,
        "n_total_trials":      len(all_trials),
        "n_folds":             len(fold_results),
    }


# ── Plot ──────────────────────────────────────────────────────────── #

def _plot(folds: List[Dict[str, Any]], aggregate: Dict[str, Any],
          out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Per-fold OOS Sharpe bars
    ax = axes[0, 0]
    names  = [f"F{f['fold']}\n{f['test_period'][:4]}" for f in folds]
    sharps = [f["oos_metrics"].get("adj_sharpe", 0) for f in folds]
    bars   = ax.bar(names, sharps, color="#2563eb", edgecolor="white")
    for b, v in zip(bars, sharps):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Sharpe OOS par fold", fontweight="bold")
    ax.set_ylabel("Sharpe annualisé")
    ax.grid(alpha=0.3, axis="y")

    # (b) Cumulative equity concatenated
    ax = axes[0, 1]
    cum = np.exp(np.cumsum(np.concatenate(
        [np.array(f["pnl_oos"]) for f in folds]
    )))
    ax.plot(cum, color="#2563eb", linewidth=1.6)
    boundaries = np.cumsum([len(f["pnl_oos"]) for f in folds])[:-1]
    for x in boundaries:
        ax.axvline(x, color="#9ca3af", ls="--", lw=0.8)
    dsr_p = aggregate["deflated_sharpe"]["deflated"].get("p_value", 1.0)
    pooled_sr = aggregate["deflated_sharpe"]["pooled_sharpe"]
    ax.set_title(f"Equity OOS concaténée — Sharpe pooled={pooled_sr:.2f}  "
                 f"Deflated p={dsr_p:.4f}", fontweight="bold")
    ax.set_ylabel("Cumul (base 1)")
    ax.grid(alpha=0.3)

    # (c) In-sample vs OOS Sharpe scatter per fold
    ax = axes[1, 0]
    is_best = [max([v["best_value"] for k, v in f["agent_summaries"].items()
                    if isinstance(v, dict) and "best_value" in v], default=0)
               for f in folds]
    oos_sr  = sharps
    ax.scatter(is_best, oos_sr, s=100, color="#2563eb", edgecolor="white", zorder=3)
    for f, xi, yi in zip(folds, is_best, oos_sr):
        ax.annotate(f"F{f['fold']}", (xi, yi),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    lim = max(max(is_best, default=1), max(oos_sr, default=1)) * 1.1
    ax.plot([0, lim], [0, lim], color="#9ca3af", ls="--", lw=1.0, label="y = x")
    ax.set_xlabel("Best in-sample Sharpe (Optuna)")
    ax.set_ylabel("OOS Sharpe")
    ax.set_title("Shrinkage IS → OOS", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (d) Summary text
    ax = axes[1, 1]; ax.axis("off")
    dsr = aggregate["deflated_sharpe"]["deflated"]
    ci  = aggregate.get("pooled_bootstrap_ci", {})
    lines = [
        "RÉSUMÉ WALK-FORWARD",
        "",
        f"  Folds              : {aggregate['n_folds']}",
        f"  Total trials       : {aggregate['n_total_trials']}",
        f"  Pooled Sharpe OOS  : {aggregate['deflated_sharpe']['pooled_sharpe']:.4f}",
        f"  Expected max (H0)  : {dsr.get('expected_max_sharpe', 0):.4f}",
        f"  Deflated z / p     : z={dsr.get('z_score', 0):.2f}  p={dsr.get('p_value', 1):.4f}",
        f"  Significant (5%)   : {dsr.get('is_significant_5pct', False)}",
        "",
        f"  Block-CI 95%       : [{ci.get('ci_lo', 'NA')}  –  {ci.get('ci_hi', 'NA')}]",
        f"  Autocorr ρ1        : {ci.get('autocorr_lag1', 'NA')}",
        f"  Block length       : {ci.get('block_length', 'NA')}",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), fontsize=11, family="monospace",
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle("SCAF v3 — Walk-Forward 5 folds + Deflated Sharpe",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "walkforward_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-per-fold", type=int, default=300,
                    help="Optuna trial budget per fold (default: 300)")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="Subset of fold indices to run (default: all)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    set_global_seed(SEED)
    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("results") / f"scaf_v3_walkforward_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(" SCAF v3 — Walk-Forward 5 folds × %d trials  [%s]",
             args.trials_per_fold, run_id)
    log.info(" seed=%d  fee=%.5f (%.1f bps/leg)",
             SEED, COST_MODEL.fee, COST_MODEL.total_bps)
    log.info("=" * 70)

    # Data + features (shared across folds)
    log.info("[data] Loading multi-asset universe...")
    loader = MultiAssetLoader(start=START, end=END, use_cache=True)
    ohlcv, sentiment = loader.load()
    ohlcv, sentiment, index = align_universe(ohlcv, sentiment)
    features, targets = build_feature_panel(ohlcv, sentiment, index, horizon=HORIZON)
    closes = pd.DataFrame(
        {t: df["Close"] for t, df in ohlcv.items()}
    ).reindex(index).ffill()
    log.info("[data]  %d assets | %d dates | %d features (avg)",
             len(ohlcv), len(index),
             int(np.mean([f.shape[1] for f in features.values()])))

    selected = [f for f in FOLDS if not args.folds or f.idx in args.folds]
    fold_results: List[Dict[str, Any]] = []
    for f in selected:
        t1 = time.time()
        res = _run_one_fold(
            f, ohlcv=ohlcv, sentiment=sentiment, index=index,
            features=features, targets=targets, closes=closes,
            trials_per_fold=args.trials_per_fold, seed=SEED,
        )
        res["elapsed_s"] = round(time.time() - t1, 1)
        fold_results.append(res)
        sr   = res["oos_metrics"].get("adj_sharpe", 0)
        mdd  = res["oos_metrics"].get("adj_max_dd", 0)
        log.info("  -> Fold %d OOS: Sharpe=%.4f  MDD=%.4f  %.1fs",
                 f.idx, sr, mdd, res["elapsed_s"])

    aggregate = _aggregate(fold_results, args.trials_per_fold)

    report = {
        "schema":          "scaf_v3.walkforward_report.v1",
        "run_id":          run_id,
        "trials_per_fold": args.trials_per_fold,
        "cost_model":      COST_MODEL.to_dict(),
        "folds":           fold_results,
        "aggregate":       aggregate,
        "elapsed_s":       round(time.time() - t0, 1),
        "run_metadata":    _collect_metadata(
            seed=SEED,
            config={
                "start": START, "end": END, "horizon": HORIZON,
                "embargo": EMBARGO, "benchmark": BENCHMARK,
                "cost_model": COST_MODEL.to_dict(),
                "trials_per_fold": args.trials_per_fold,
            },
            repo_root=Path(__file__).parent,
        ),
    }
    out_json = out_dir / "walkforward_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    log.info("Wrote %s", out_json)

    try:
        _plot(fold_results, aggregate, out_dir / "dashboard")
    except Exception as exc:
        log.warning("Plot failed (non-blocking): %s", exc)

    dsr = aggregate["deflated_sharpe"]["deflated"]
    print("=" * 70)
    print(f"  Pooled OOS Sharpe      : {aggregate['deflated_sharpe']['pooled_sharpe']:.4f}")
    print(f"  Expected max (H0)      : {dsr.get('expected_max_sharpe', 0):.4f}")
    print(f"  Deflated Sharpe z / p  : z={dsr.get('z_score', 0):.2f}  p={dsr.get('p_value', 1):.4f}")
    print(f"  Significant (5%)       : {dsr.get('is_significant_5pct', False)}")
    print(f"  Total compute          : {report['elapsed_s']:.0f}s")
    print(f"  Output                 : {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
