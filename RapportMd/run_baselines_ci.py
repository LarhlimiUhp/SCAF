"""
SCAF v3 — Baselines + Bootstrap CI + Feature Importance
=========================================================
Implements three publication-grade additions:

  [A] 3 formal baselines on the same OOS 2022-2024 period
        - Donchian Only       (isolates trend-following contribution)
        - Equal-Weight B&H    (passive multi-asset benchmark)
        - Jegadeesh-Titman    (12-1 momentum, monthly rebalance)

  [B] Bootstrap Sharpe CI (1000 resamples, BCa method)
        - Converts "Sharpe 5.16" → "Sharpe 5.16 [CI 4.02–6.31, p<0.001]"

  [C] Feature Importance (top-10, cross-sectional task)
        - RandomForest Gini importance on pooled CS training data
        - Justifies the 58-feature space & alternative data choice

Outputs
-------
  results/scaf_v3_20260416_133210/
    baselines_report.json
    dashboard/07_baselines_comparison.png
    dashboard/08_bootstrap_ci.png
    dashboard/09_feature_importance.png

Usage
-----
    cd 07-04-2026
    python run_baselines_ci.py
"""

from __future__ import annotations
import json, logging, sys, time, warnings
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("baselines_ci")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results") / "scaf_v3_20260416_133210"
DASH_DIR    = RESULTS_DIR / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = RESULTS_DIR / "scaf_v3_report.json"

# ── Config (must match run_scaf_v3.py exactly) ─────────────────────────────────
import os as _os
from analysis.cost_model import resolve as _resolve_costs
from analysis.run_metadata import collect as _collect_metadata, set_global_seed

START        = "2015-01-01"
END          = "2024-12-31"
ML_TRAIN_END = "2019-12-31"
OPT_START    = "2020-01-01"
OPT_END      = "2021-12-31"
TEST_START   = "2022-01-01"
HORIZON      = 5
BENCHMARK    = "SPY"
N_BOOTSTRAP  = 1000
SEED         = int(_os.environ.get("SCAF_SEED", "42"))
COST_MODEL   = _resolve_costs(profile_name=_os.environ.get("SCAF_COST_PROFILE"))
FEE          = COST_MODEL.fee

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"; PANEL_BG = "#161b22"; GRID_CLR = "#21262d"
TEXT_CLR = "#e6edf3"; MUTED    = "#8b949e"
ACCENT   = "#58a6ff"; GREEN    = "#3fb950"; RED      = "#f85149"
ORANGE   = "#d29922"; PURPLE   = "#bc8cff"; GOLD     = "#ffa657"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_CLR, "axes.labelcolor": TEXT_CLR,
    "axes.titlecolor": TEXT_CLR, "xtick.color": TEXT_CLR,
    "ytick.color": TEXT_CLR, "text.color": TEXT_CLR,
    "grid.color": GRID_CLR, "grid.linewidth": 0.5,
    "font.family": "monospace", "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_CLR,
})

# ── Imports from scaf_v3 ───────────────────────────────────────────────────────
from scaf_v3.loader   import MultiAssetLoader, align_universe
from scaf_v3.features import build_feature_panel
from scaf_v3.models   import RegimeFilter, CrossSectionRanker
from scaf_v3.strategy import (TrendEngine, CrossSectionPortfolio,
                               HybridSignal, portfolio_metrics)
from scaf_v3.risk     import RiskManager


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def sharpe(pnl: np.ndarray, periods: int = 252) -> float:
    s = float(np.std(pnl)) + 1e-12
    return float(np.mean(pnl)) / s * np.sqrt(periods)

def cum_ret(pnl: np.ndarray) -> float:
    return float(np.sum(pnl))

def max_dd(pnl: np.ndarray) -> float:
    eq = np.exp(np.cumsum(pnl))
    peak = np.maximum.accumulate(eq)
    return float(np.min((eq - peak) / (peak + 1e-12)))

def ann_ret(pnl: np.ndarray, periods: int = 252) -> float:
    return float(np.mean(pnl)) * periods

def metrics_dict(pnl: np.ndarray) -> Dict[str, float]:
    m = portfolio_metrics(pnl)
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE A — DONCHIAN ONLY
# ══════════════════════════════════════════════════════════════════════════════

def run_donchian_only(closes: pd.DataFrame, test_idx: np.ndarray,
                      best_params: Dict) -> np.ndarray:
    """
    TrendEngine signal on SPY, no ML scaling, no CS component.
    Isolates the pure trend-following contribution.
    """
    bm = closes[BENCHMARK] if BENCHMARK in closes.columns else closes.iloc[:, 0]
    engine = TrendEngine(
        don_win=int(best_params["don_win"]),
        tf_win=int(best_params["tf_win"]),
        w_don=0.5, w_tf=0.5,
    )
    trend_sig = engine.compute(bm)

    closes_s = closes.iloc[test_idx]
    trend_s  = trend_sig.iloc[test_idx]
    log_rets = np.log(closes_s[BENCHMARK] / closes_s[BENCHMARK].shift(1)).fillna(0).values

    pos      = np.clip(trend_s.values, -1.0, 1.0)
    pos_prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - pos_prev)
    pnl      = pos * log_rets - FEE * turnover
    return pnl


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE B — EQUAL-WEIGHT BUY-AND-HOLD
# ══════════════════════════════════════════════════════════════════════════════

def run_equal_weight_bnh(closes: pd.DataFrame, test_idx: np.ndarray) -> np.ndarray:
    """
    Equal-weight portfolio of all 18 ETFs.
    Buy at TEST_START, hold until END. No rebalancing → passive multi-asset.
    """
    closes_s = closes.iloc[test_idx]
    log_rets = np.log(closes_s / closes_s.shift(1)).fillna(0)
    # Equal weight across all columns
    pnl = log_rets.mean(axis=1).values  # simple arithmetic average of log returns
    return pnl


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE C — JEGADEESH-TITMAN MOMENTUM (12-1)
# ══════════════════════════════════════════════════════════════════════════════

def run_momentum_12_1(closes: pd.DataFrame, test_idx: np.ndarray,
                      n_long: int = 3, n_short: int = 3) -> np.ndarray:
    """
    Classic Jegadeesh-Titman cross-sectional momentum.
    Signal  = return(t-252, t-21)  [12m lookback, skip 1m]
    Rebalance monthly (approx 21 trading days).
    Long top-N, short bottom-N, equal weight, transaction cost FEE.
    """
    # Need prices before test period for lookback (12 months = ~252 days)
    # Get full closes up to end of test period
    all_dates = closes.index
    test_dates = all_dates[test_idx]

    # Rebalance dates: every ~21 trading days within the test period
    rebal_dates = test_dates[::21]

    # Build daily position DataFrame (test_idx length)
    n_test   = len(test_idx)
    n_assets = len(closes.columns)
    tickers  = list(closes.columns)
    weights  = np.zeros((n_test, n_assets))

    current_weights = np.zeros(n_assets)

    for i, d in enumerate(test_dates):
        if d in rebal_dates:
            # Compute signal: need 252 + 21 days of history
            loc = closes.index.get_loc(d)
            if loc < 273:  # not enough history
                current_weights = np.ones(n_assets) / n_assets  # fallback equal weight
            else:
                p_12m = closes.iloc[loc - 252].values.astype(float)
                p_1m  = closes.iloc[loc - 21].values.astype(float)
                signal = np.where(p_12m > 0, np.log(p_1m / p_12m), 0.0)

                # Rank: long top-N, short bottom-N
                ranked = np.argsort(signal)
                new_w  = np.zeros(n_assets)
                if n_long > 0:
                    new_w[ranked[-n_long:]]  =  1.0 / n_long
                if n_short > 0:
                    new_w[ranked[:n_short]]  = -1.0 / n_short
                current_weights = new_w

        weights[i] = current_weights

    # Compute daily PnL
    closes_s = closes.iloc[test_idx]
    log_rets = np.log(closes_s / closes_s.shift(1)).fillna(0).values

    # P&L with turnover cost
    w_prev   = np.vstack([np.zeros((1, n_assets)), weights[:-1]])
    turnover = np.abs(weights - w_prev).sum(axis=1)
    pnl      = (weights * log_rets).sum(axis=1) - FEE * turnover
    return pnl


# ══════════════════════════════════════════════════════════════════════════════
#  SCAF v3 — REPLAY (to get daily pnl_series)
# ══════════════════════════════════════════════════════════════════════════════

def replay_scaf_v3(closes, features, bm_features, regime_filter,
                   cs_ranker, best_params, test_idx) -> np.ndarray:
    """Re-runs final_evaluation logic and returns the adj pnl array."""
    bm_X     = bm_features.iloc[test_idx].fillna(0).values.astype(np.float32)
    ml_proba = regime_filter.predict_proba(bm_X)
    cs_pred  = cs_ranker.predict(features, test_idx)

    bm_closes = closes[BENCHMARK]
    engine = TrendEngine(
        don_win=int(best_params["don_win"]),
        tf_win=int(best_params["tf_win"]),
        w_don=0.5, w_tf=0.5,
    )
    trend_sig = engine.compute(bm_closes)

    cs_port = CrossSectionPortfolio(
        long_thr=float(best_params["ml_thr"]),
        short_thr=1.0 - float(best_params["ml_thr"]),
        max_positions=4,
    )
    cs_w = cs_port.compute(cs_pred)

    hybrid = HybridSignal(
        w_trend=float(best_params["w_trend"]),
        w_cs=float(best_params["w_cs"]),
        ml_thr=float(best_params["ml_thr"]),
        s_bull=float(best_params["s_bull"]),
        s_bear=float(best_params["s_bear"]),
        s_side=float(best_params["s_side"]),
        max_pos=float(best_params["max_pos"]),
        transaction_cost=FEE,
    )
    pnl_raw, _ = hybrid.compute_returns(closes, trend_sig, cs_w, ml_proba, test_idx)

    rm = RiskManager(
        target_vol=float(best_params.get("target_vol", 0.10)),
        vol_window=int(best_params.get("vol_window", 20)),
    )
    pnl_adj = rm.apply(pnl_raw, pnl_raw)
    return pnl_adj


# ══════════════════════════════════════════════════════════════════════════════
#  [B] BOOTSTRAP SHARPE CI (BCa)
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_sharpe_ci(pnl: np.ndarray, n_boot: int = 1000,
                         alpha: float = 0.05) -> Dict[str, float]:
    """
    Bootstrap confidence interval for annualised Sharpe ratio.
    Uses the BCa (bias-corrected and accelerated) method.
    Also computes one-sided p-value: H0: Sharpe <= 0.
    """
    rng  = np.random.default_rng(42)
    n    = len(pnl)
    obs_sharpe = sharpe(pnl)

    boot_sharpes = np.array([
        sharpe(rng.choice(pnl, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # BCa bias correction
    z0 = stats.norm.ppf(np.mean(boot_sharpes < obs_sharpe))

    # Acceleration (jackknife)
    jack_sharpes = np.array([sharpe(np.delete(pnl, i)) for i in range(0, n, max(1, n//200))])
    jack_mean    = np.mean(jack_sharpes)
    num          = np.sum((jack_mean - jack_sharpes) ** 3)
    den          = 6.0 * (np.sum((jack_mean - jack_sharpes) ** 2) ** 1.5)
    acc          = num / (den + 1e-12)

    def bca_bound(p_target):
        z_p   = stats.norm.ppf(p_target)
        z_adj = z0 + (z0 + z_p) / (1.0 - acc * (z0 + z_p))
        return float(np.percentile(boot_sharpes, stats.norm.cdf(z_adj) * 100))

    ci_lo = bca_bound(alpha / 2)
    ci_hi = bca_bound(1 - alpha / 2)

    # p-value: proportion of bootstrap resamples with Sharpe <= 0
    p_value = float(np.mean(boot_sharpes <= 0))
    p_value = max(p_value, 1 / n_boot)  # floor at 1/n_boot

    return {
        "observed":  round(obs_sharpe, 4),
        "boot_mean": round(float(np.mean(boot_sharpes)), 4),
        "boot_std":  round(float(np.std(boot_sharpes)), 4),
        "ci_lo":     round(ci_lo, 4),
        "ci_hi":     round(ci_hi, 4),
        "p_value":   round(p_value, 6),
        "n_boot":    n_boot,
        "alpha":     alpha,
        "method":    "BCa",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  [C] FEATURE IMPORTANCE (top-10)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(features: Dict[str, pd.DataFrame],
                                targets: Dict[str, pd.Series],
                                train_idx: np.ndarray,
                                n_estimators: int = 200) -> pd.Series:
    """
    Train a RandomForestClassifier on the pooled cross-section data.
    Returns feature importances (Gini mean decrease) as a Series.
    More interpretable than Bagging with CalibratedClassifierCV.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X_list, y_list = [], []
    feat_names = None

    for ticker in features:
        if ticker not in targets:
            continue
        feat = features[ticker].iloc[train_idx]
        tgt  = targets[ticker].iloc[train_idx]
        valid = feat.notna().all(axis=1) & tgt.notna()
        if valid.sum() < 50:
            continue
        if feat_names is None:
            feat_names = list(feat.columns)
        X_list.append(feat[valid].values)
        y_list.append(tgt[valid].values)

    if not X_list:
        return pd.Series(dtype=float)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(int)

    sc = StandardScaler()
    X  = sc.fit_transform(X)

    log.info("  RandomForest feature importance: %d samples x %d features",
             len(X), X.shape[1])

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=5,
        min_samples_leaf=20, n_jobs=-1, random_state=42
    )
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=feat_names)
    return importance.sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_baselines_comparison(all_pnls: Dict[str, np.ndarray],
                               all_metrics: Dict[str, Dict],
                               test_dates: pd.DatetimeIndex):
    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    fig.suptitle("Comparaison SCAF v3 vs Baselines — OOS 2022–2024",
                 fontsize=14, fontweight="bold", color=TEXT_CLR, y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           top=0.93, bottom=0.07, left=0.07, right=0.97)

    COLORS = {
        "SCAF v3":          ACCENT,
        "Donchian Only":    GREEN,
        "Equal-Weight B&H": ORANGE,
        "Momentum 12-1":    PURPLE,
    }

    # ── Panel 1: Cumulative returns ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Rendements Cumulés — OOS 2022–2024", fontsize=12, pad=8)
    for name, pnl in all_pnls.items():
        cum = np.exp(np.cumsum(pnl)) - 1
        n   = min(len(cum), len(test_dates))
        ax1.plot(test_dates[:n], cum[:n] * 100, color=COLORS[name],
                 linewidth=2 if name == "SCAF v3" else 1.2,
                 linestyle="-" if name == "SCAF v3" else "--",
                 label=f"{name}  Σ={cum[-1]*100:.1f}%")
    ax1.axhline(0, color=GRID_CLR, linewidth=0.8)
    ax1.set_ylabel("Rendement cumulé (%)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, axis="y")

    # ── Panel 2: Sharpe comparison bar ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Sharpe Ratio (OOS)", fontsize=12, pad=8)
    names   = list(all_metrics.keys())
    sharpes = [all_metrics[n]["sharpe"] for n in names]
    clrs    = [COLORS[n] for n in names]
    bars    = ax2.bar(names, sharpes, color=clrs, edgecolor=DARK_BG, width=0.5)
    ax2.axhline(1.0, color=GOLD, linewidth=1, linestyle="--", label="Sharpe = 1")
    for bar, v in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=TEXT_CLR)
    ax2.set_ylabel("Sharpe (annualisé)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=12, ha="right", fontsize=9)

    # ── Panel 3: Max Drawdown comparison ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("Max Drawdown (%)", fontsize=12, pad=8)
    mds = [abs(all_metrics[n]["max_dd"]) * 100 for n in names]
    bars2 = ax3.bar(names, mds, color=clrs, edgecolor=DARK_BG, width=0.5)
    ax3.axhline(15.0, color=RED, linewidth=1, linestyle="--", label="Limite 15%")
    for bar, v in zip(bars2, mds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"-{v:.1f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=TEXT_CLR)
    ax3.set_ylabel("Max DD (%)")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=12, ha="right", fontsize=9)

    fig.savefig(DASH_DIR / "07_baselines_comparison.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info("[OK] 07_baselines_comparison.png")


def plot_bootstrap_ci(ci_results: Dict[str, Any], pnl_scaf: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle("Bootstrap Sharpe CI (BCa, 1000 resamples) — SCAF v3 OOS",
                 fontsize=13, fontweight="bold", color=TEXT_CLR, y=1.02)

    # Resample distribution
    rng = np.random.default_rng(42)
    n   = len(pnl_scaf)
    boot_sharpes = np.array([
        sharpe(rng.choice(pnl_scaf, size=n, replace=True)) for _ in range(N_BOOTSTRAP)
    ])

    ax = axes[0]
    ax.hist(boot_sharpes, bins=50, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
    ax.axvline(ci_results["observed"],
               color=GREEN, linewidth=2, linestyle="-",
               label=f"Sharpe observé = {ci_results['observed']:.4f}")
    ax.axvline(ci_results["ci_lo"],
               color=GOLD, linewidth=1.5, linestyle="--",
               label=f"CI 95% lo = {ci_results['ci_lo']:.4f}")
    ax.axvline(ci_results["ci_hi"],
               color=GOLD, linewidth=1.5, linestyle="--",
               label=f"CI 95% hi = {ci_results['ci_hi']:.4f}")
    ax.axvline(0, color=RED, linewidth=1, linestyle=":", label="H0: Sharpe = 0")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500],
                     ci_results["ci_lo"], ci_results["ci_hi"],
                     alpha=0.12, color=GOLD)
    ax.set_xlabel("Sharpe bootstrap (annualisé)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution Bootstrap du Sharpe", fontsize=11, pad=6)
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y")

    # Stats summary panel
    ax2 = axes[1]
    ax2.axis("off")
    summary = [
        ("Sharpe observé",    f"{ci_results['observed']:.4f}",    GREEN),
        ("Bootstrap moyen",   f"{ci_results['boot_mean']:.4f}",   ACCENT),
        ("Bootstrap std",     f"±{ci_results['boot_std']:.4f}",   MUTED),
        ("IC 95% bas",        f"{ci_results['ci_lo']:.4f}",       GOLD),
        ("IC 95% haut",       f"{ci_results['ci_hi']:.4f}",       GOLD),
        ("p-value (H0=0)",    f"{ci_results['p_value']:.6f}",     GREEN),
        ("Méthode",           ci_results["method"],               MUTED),
        ("N resamples",       f"{ci_results['n_boot']}",          MUTED),
        ("N jours OOS",       f"{n}",                             MUTED),
    ]
    for i, (lbl, val, clr) in enumerate(summary):
        y = 0.93 - i * 0.10
        ax2.text(0.05, y, lbl, fontsize=10, color=MUTED,
                 transform=ax2.transAxes, va="top")
        ax2.text(0.95, y, val, fontsize=11, color=clr, fontweight="bold",
                 transform=ax2.transAxes, va="top", ha="right")

    # Publication string
    pub_str = (f"Sharpe = {ci_results['observed']:.2f} "
               f"[IC 95%: {ci_results['ci_lo']:.2f}–{ci_results['ci_hi']:.2f}, "
               f"p<{max(ci_results['p_value'], 0.001):.3f}]")
    ax2.text(0.5, 0.06, pub_str, fontsize=10, color=ACCENT, fontweight="bold",
             transform=ax2.transAxes, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d2137",
                       edgecolor=ACCENT, alpha=0.9))

    fig.tight_layout()
    fig.savefig(DASH_DIR / "08_bootstrap_ci.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info("[OK] 08_bootstrap_ci.png")


def plot_feature_importance(importance: pd.Series, top_n: int = 10):
    """Horizontal bar chart of top-N features with category coloring."""
    top = importance.head(top_n)

    # Category color mapping
    CAT_COLORS = {
        "vix": PURPLE, "vvix": PURPLE, "yield": PURPLE, "dxy": PURPLE,
        "credit": PURPLE, "spy_tlt": PURPLE, "tnx": PURPLE,
        "cs_rank": ORANGE, "rel_": ORANGE, "beta": ORANGE, "corr": ORANGE,
        "rank_rev": ORANGE, "cs_disp": ORANGE,
        "range": GREEN, "gap": GREEN, "close_pos": GREEN,
        "body": GREEN, "shadow": GREEN, "vol_ret": GREEN, "vol_trend": GREEN,
    }
    def get_color(feat):
        feat_l = feat.lower()
        for kw, clr in CAT_COLORS.items():
            if kw in feat_l:
                return clr
        return ACCENT  # price/momentum features

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    ax.set_title("Top-10 Features — Importance Gini (Cross-Section Ranker)",
                 fontsize=13, fontweight="bold", pad=10)

    colors = [get_color(f) for f in top.index]
    bars   = ax.barh(range(len(top)), top.values * 100,
                     color=colors, edgecolor=DARK_BG, height=0.65)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Gini (%)")
    ax.grid(True, axis="x")

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val*100:.2f}%", va="center", fontsize=9, color=TEXT_CLR)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=PURPLE, label="Alt/Sentiment"),
        Patch(facecolor=ORANGE, label="Cross-Section"),
        Patch(facecolor=GREEN,  label="Intraday Proxy"),
        Patch(facecolor=ACCENT, label="Prix/Momentum"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(DASH_DIR / "09_feature_importance.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    log.info("[OK] 09_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    set_global_seed(SEED)
    log.info("=" * 64)
    log.info("  SCAF v3 — Baselines + Bootstrap CI + Feature Importance")
    log.info("  seed=%d  fee=%.5f (%.1f bps/leg)",
             SEED, FEE, COST_MODEL.total_bps)
    log.info("=" * 64)

    best_params = json.loads(REPORT_PATH.read_text())["best_params"]

    # ── [1] Load data (from cache) ─────────────────────────────────────────
    log.info("[1/5] Loading data from cache...")
    loader = MultiAssetLoader(start=START, end=END, use_cache=True)
    ohlcv, sentiment = loader.load()
    ohlcv, sentiment, index = align_universe(ohlcv, sentiment)

    closes = pd.DataFrame(
        {t: df["Close"] for t, df in ohlcv.items()}
    ).reindex(index).ffill()

    test_mask = index >= TEST_START
    test_idx  = np.where(test_mask)[0]
    test_dates = index[test_idx]

    ml_train_mask = index <= ML_TRAIN_END
    ml_train_idx  = np.where(ml_train_mask)[0]

    log.info("  Universe: %d assets  |  OOS: %d days (%s → %s)",
             len(closes.columns), len(test_idx),
             test_dates[0].date(), test_dates[-1].date())

    # ── [2] Build features ─────────────────────────────────────────────────
    log.info("[2/5] Building feature panel...")
    features, targets = build_feature_panel(ohlcv, sentiment, index, horizon=HORIZON)
    bm_ticker  = BENCHMARK if BENCHMARK in features else list(features.keys())[0]
    bm_features = features[bm_ticker]

    # ── [3] Train models ───────────────────────────────────────────────────
    log.info("[3/5] Training RegimeFilter + CrossSectionRanker...")
    regime_filter = RegimeFilter()
    bm_X_train = bm_features.iloc[ml_train_idx].fillna(0).values.astype(np.float32)
    bm_y_train = targets[bm_ticker].iloc[ml_train_idx].fillna(0).values.astype(int)
    regime_filter.fit(bm_X_train, bm_y_train,
                      bm_X_train[-len(bm_X_train)//5:],
                      bm_y_train[-len(bm_y_train)//5:])

    cs_ranker = CrossSectionRanker()
    cs_ranker.fit(features, targets, ml_train_idx)

    # ── [4] Run everything ─────────────────────────────────────────────────
    log.info("[4/5] Running SCAF v3 + 3 baselines + bootstrap + feature importance...")

    # SCAF v3 OOS PnL
    log.info("  [A] Replaying SCAF v3 on OOS...")
    pnl_scaf = replay_scaf_v3(closes, features, bm_features,
                               regime_filter, cs_ranker, best_params, test_idx)

    # Baseline 1: Donchian Only
    log.info("  [B1] Donchian Only baseline...")
    pnl_don = run_donchian_only(closes, test_idx, best_params)

    # Baseline 2: Equal-Weight B&H
    log.info("  [B2] Equal-Weight B&H baseline...")
    pnl_ewbh = run_equal_weight_bnh(closes, test_idx)

    # Baseline 3: Jegadeesh-Titman 12-1
    log.info("  [B3] Jegadeesh-Titman 12-1 momentum...")
    pnl_mom = run_momentum_12_1(closes, test_idx)

    # Bootstrap CI — iid (legacy) + stationary block bootstrap (preferred)
    log.info("  [C] Bootstrap Sharpe CI (iid, n=%d)...", N_BOOTSTRAP)
    ci = bootstrap_sharpe_ci(pnl_scaf, n_boot=N_BOOTSTRAP)
    log.info("  Sharpe = %.4f [CI: %.4f – %.4f]  p = %.6f",
             ci["observed"], ci["ci_lo"], ci["ci_hi"], ci["p_value"])

    from analysis.robustness import (
        stationary_bootstrap_sharpe_ci, jackknife_top_k,
    )
    log.info("  [C'] Stationary-block bootstrap CI...")
    ci_block = stationary_bootstrap_sharpe_ci(
        np.asarray(pnl_scaf), n_boot=N_BOOTSTRAP, seed=SEED,
    )
    log.info("  Block-BCa Sharpe = %.4f [CI: %.4f – %.4f]  L=%d  ρ1=%.3f",
             ci_block["observed"], ci_block["ci_lo"], ci_block["ci_hi"],
             ci_block["block_length"], ci_block["autocorr_lag1"])

    log.info("  [C''] Jackknife top-|P&L| trades...")
    jk = jackknife_top_k(np.asarray(pnl_scaf))

    # Feature importance
    log.info("  [D] Feature importance (RandomForest 200 trees)...")
    feat_imp = compute_feature_importance(features, targets, ml_train_idx)
    top10 = feat_imp.head(10)
    log.info("  Top-5 features: %s", list(top10.index[:5]))

    # ── Compile metrics ────────────────────────────────────────────────────
    all_pnls = {
        "SCAF v3":          pnl_scaf,
        "Donchian Only":    pnl_don,
        "Equal-Weight B&H": pnl_ewbh,
        "Momentum 12-1":    pnl_mom,
    }
    all_metrics = {name: metrics_dict(pnl) for name, pnl in all_pnls.items()}

    log.info("\n%s", "=" * 64)
    log.info("  BASELINE COMPARISON — OOS 2022–2024")
    log.info("%s", "-" * 64)
    log.info("  %-22s  %8s  %10s  %10s  %10s",
             "Strategy", "Sharpe", "Max DD", "Ann Ret", "Cum Ret")
    log.info("%s", "-" * 64)
    for name, m in all_metrics.items():
        log.info("  %-22s  %8.4f  %10.2f%%  %10.2f%%  %10.2f%%",
                 name,
                 m.get("sharpe", 0),
                 m.get("max_dd", 0) * 100,
                 m.get("ann_ret", 0) * 100,
                 m.get("cum_ret", 0) * 100)
    log.info("%s", "=" * 64)

    # ── [5] Save + Charts ──────────────────────────────────────────────────
    log.info("[5/5] Generating charts and saving report...")

    plot_baselines_comparison(all_pnls, all_metrics, test_dates)
    plot_bootstrap_ci(ci, pnl_scaf)
    plot_feature_importance(feat_imp, top_n=10)

    # Save JSON report
    report = {
        "run_timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "bootstrap_sharpe_ci": ci,
        "stationary_block_bootstrap_ci": ci_block,
        "jackknife_top_k": jk,
        "publication_string": (
            f"Sharpe = {ci['observed']:.2f} "
            f"[IC 95%: {ci['ci_lo']:.2f}–{ci['ci_hi']:.2f}, "
            f"p<{max(ci['p_value'], 0.001):.3f}]"
        ),
        "baselines": {
            name: {k: round(v, 6) if isinstance(v, float) else v
                   for k, v in m.items()}
            for name, m in all_metrics.items()
        },
        "sharpe_improvements_vs_baselines": {
            name: round(
                (all_metrics["SCAF v3"]["sharpe"] - all_metrics[name]["sharpe"])
                / (abs(all_metrics[name]["sharpe"]) + 1e-12) * 100, 2
            )
            for name in ["Donchian Only", "Equal-Weight B&H", "Momentum 12-1"]
        },
        "feature_importance_top10": {
            feat: round(float(val), 6)
            for feat, val in top10.items()
        },
        "feature_importance_by_category": {
            "price_momentum": round(float(feat_imp[
                [f for f in feat_imp.index
                 if not any(k in f.lower() for k in
                    ["vix","vvix","yield","dxy","credit","spy_tlt","tnx",
                     "cs_rank","rel_","beta","corr","rank_rev","cs_disp",
                     "range","gap","close_pos","body","shadow","vol_ret","vol_trend"])]
            ].sum()), 4),
            "cross_section": round(float(feat_imp[
                [f for f in feat_imp.index
                 if any(k in f.lower() for k in
                    ["cs_rank","rel_","beta","corr","rank_rev","cs_disp"])]
            ].sum()), 4),
            "intraday_proxy": round(float(feat_imp[
                [f for f in feat_imp.index
                 if any(k in f.lower() for k in
                    ["range","gap","close_pos","body","shadow","vol_ret","vol_trend"])]
            ].sum()), 4),
            "alt_sentiment": round(float(feat_imp[
                [f for f in feat_imp.index
                 if any(k in f.lower() for k in
                    ["vix","vvix","yield","dxy","credit","spy_tlt","tnx"])]
            ].sum()), 4),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "run_metadata": _collect_metadata(
            seed=SEED,
            config={
                "start": START, "end": END,
                "ml_train_end": ML_TRAIN_END,
                "test_start": TEST_START,
                "horizon": HORIZON, "benchmark": BENCHMARK,
                "cost_model": COST_MODEL.to_dict(),
                "n_bootstrap": N_BOOTSTRAP,
            },
            repo_root=Path(__file__).parent,
        ),
    }

    out_path = RESULTS_DIR / "baselines_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("[OK] Report: %s", out_path)

    # Summary
    log.info("\n%s", "=" * 64)
    log.info("  PUBLICATION STATEMENT")
    log.info("  %s", report["publication_string"])
    log.info("\n  SHARPE IMPROVEMENT vs BASELINES")
    for k, v in report["sharpe_improvements_vs_baselines"].items():
        log.info("    vs %-22s  +%.1f%%", k, v)
    log.info("\n  TOP-5 FEATURES (Gini)")
    for f, v in list(report["feature_importance_top10"].items())[:5]:
        log.info("    %-30s  %.4f", f, v)
    log.info("\n  Duree: %.1f s", report["elapsed_s"])
    log.info("%s", "=" * 64)


if __name__ == "__main__":
    main()
