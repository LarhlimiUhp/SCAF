"""
run_scaf_v4.py  —  SCAF v4 : 10 000 agents + Ablation + Baselines + PnL persisté
====================================================================================
4 recommandations de publication :
  R1. PnL journalier persisté  — CSV daté + JSON (courbe réelle, pas synthétique)
  R2. Ablation formelle        — 4 configs sur OOS (Full / No-Trend / No-Regime / No-CS)
  R3. 5 baselines réelles      — BnH SPY, Equal-Weight, Momentum 12-1,
                                  Min-Variance, MLP-naïf (LSTM proxy)
  R4. 10 000 agents Optuna     — n_jobs=4, TPE sampler (ThreadPoolExecutor)

Cadrage contribution :
  "Multi-agent Bayesian optimization of hybrid trading systems
   with ablative component validation"
"""
from __future__ import annotations

import csv, json, logging, os, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── SCAF v3 imports ────────────────────────────────────────────────────────────
from scaf_v3.loader   import MultiAssetLoader, align_universe
from scaf_v3.features import build_feature_panel
from scaf_v3.models   import RegimeFilter, CrossSectionRanker
from scaf_v3.strategy import (TrendEngine, CrossSectionPortfolio,
                               HybridSignal, portfolio_metrics)
from scaf_v3.risk     import RiskManager
from scaf_v3.optimizer import DEFAULTS, SEARCH_SPACES, make_objective

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
START       = "2015-01-01"
ML_TRAIN_END = "2019-12-31"
OPT_START   = "2020-01-01"
OPT_END     = "2021-12-31"
TEST_START  = "2022-01-01"
END         = "2024-12-31"
BENCHMARK   = "SPY"
FEE         = 0.0002
HORIZON     = 5
N_JOBS      = min(4, os.cpu_count() or 1)

# 10 000 agents répartis sur 5 catégories
AGENT_CATEGORIES_V4: Dict[str, int] = {
    "strategy_params":  3_000,
    "ml_params":        3_000,
    "regime_scaling":   1_500,
    "risk_params":      1_500,
    "portfolio_blend":  1_000,
}   # total = 10 000

RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("results") / f"scaf_v4_{RUN_ID}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 5 or np.std(pnl) < 1e-10:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(252))


def _vol_target(pnl: np.ndarray, target: float = 0.10,
                window: int = 20) -> np.ndarray:
    out = np.zeros_like(pnl)
    for i in range(len(pnl)):
        s   = max(0, i - window)
        vol = (np.std(pnl[s:i]) * np.sqrt(252)) if i > s else target
        out[i] = pnl[i] * np.clip(target / (vol + 1e-9), 0.1, 2.5)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  R1 — PnL PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def persist_pnl(pnl_adj: np.ndarray, pnl_raw: np.ndarray,
                dates: pd.DatetimeIndex, out_dir: Path) -> None:
    csv_path = out_dir / "pnl_daily.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "pnl_adj", "pnl_raw",
                    "cum_adj", "cum_raw", "drawdown_adj"])
        cum_a = cum_r = peak = 1.0
        for d, a, r in zip(dates, pnl_adj, pnl_raw):
            cum_a *= np.exp(a); cum_r *= np.exp(r)
            peak   = max(peak, cum_a)
            w.writerow([d.strftime("%Y-%m-%d"),
                        round(float(a), 8), round(float(r), 8),
                        round(float(cum_a - 1), 6), round(float(cum_r - 1), 6),
                        round(float((cum_a - peak) / peak), 6)])
    log.info("  R1 PnL CSV: %s  (%d jours)", csv_path.name, len(pnl_adj))


# ══════════════════════════════════════════════════════════════════════════════
#  R2 — ABLATION FORMELLE
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(closes, features, bm_features, regime_filter, cs_ranker,
                 best_params, test_idx) -> Dict[str, Dict[str, float]]:
    bm_X      = bm_features.iloc[test_idx].fillna(0).values.astype(np.float32)
    ml_proba  = regime_filter.predict_proba(bm_X)
    cs_pred   = cs_ranker.predict(features, test_idx)
    bm_close  = closes[BENCHMARK] if BENCHMARK in closes.columns else closes.iloc[:, 0]
    engine    = TrendEngine(don_win=int(best_params["don_win"]),
                            tf_win=int(best_params["tf_win"]),
                            w_don=0.5, w_tf=0.5)
    trend_sig = engine.compute(bm_close)
    rm        = RiskManager(target_vol=float(best_params.get("target_vol", 0.10)),
                            vol_window=int(best_params.get("vol_window", 20)))

    def _eval(overrides):
        p      = {**best_params, **overrides}
        cs_w   = CrossSectionPortfolio(
            long_thr=float(p["ml_thr"]),
            short_thr=1.0 - float(p["ml_thr"]),
            max_positions=4,
        ).compute(cs_pred)
        hybrid = HybridSignal(
            w_trend=float(p["w_trend"]), w_cs=float(p["w_cs"]),
            ml_thr=float(p["ml_thr"]),
            s_bull=float(p["s_bull"]),  s_bear=float(p["s_bear"]),
            s_side=float(p["s_side"]),  max_pos=float(p["max_pos"]),
            transaction_cost=FEE)
        pnl_r, _ = hybrid.compute_returns(closes, trend_sig, cs_w,
                                           ml_proba, test_idx)
        return portfolio_metrics(rm.apply(pnl_r, pnl_r))

    configs = {
        "SCAF_Full":       {},
        "No_TrendEngine":  {"w_trend": 0.0, "w_cs": 1.0},
        "No_RegimeFilter": {"s_bull": 1.0, "s_bear": 1.0, "s_side": 1.0},
        "No_CS_Ranker":    {"w_trend": 1.0, "w_cs": 0.0},
    }
    results = {}
    ref     = None
    for name, ov in configs.items():
        m = _eval(ov)
        results[name] = m
        delta = m["sharpe"] - ref if ref is not None else 0.0
        if ref is None:
            ref = m["sharpe"]
        log.info("  [Ablation] %-20s  Sharpe=%.4f  DD=%.2f%%  delta=%+.4f",
                 name, m["sharpe"], m.get("max_dd", 0) * 100, delta)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  R3 — 5 BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def compute_baselines(closes, train_idx, test_idx,
                      features) -> Dict[str, Dict[str, float]]:
    results = {}

    # 1. Buy & Hold SPY
    spy_px = closes[BENCHMARK].iloc[test_idx]
    spy_lr = np.log(spy_px / spy_px.shift(1)).fillna(0).values
    results["BnH_SPY"] = portfolio_metrics(spy_lr)
    log.info("  [Baseline] BnH_SPY              Sharpe=%.4f",
             results["BnH_SPY"]["sharpe"])

    # 2. Equal-Weight (vol-targeted)
    lr_all = np.log(closes.iloc[test_idx] /
                    closes.iloc[test_idx].shift(1)).fillna(0)
    ew_pnl = _vol_target(lr_all.mean(axis=1).values)
    results["Equal_Weight"] = portfolio_metrics(ew_pnl)
    log.info("  [Baseline] Equal_Weight         Sharpe=%.4f",
             results["Equal_Weight"]["sharpe"])

    # 3. Momentum 12-1 (cross-sectional)
    mom_pnl   = np.zeros(len(test_idx))
    prev_w    = np.zeros(closes.shape[1])
    col_idx   = {c: i for i, c in enumerate(closes.columns)}
    for pos, abs_i in enumerate(test_idx):
        if abs_i < 252:
            continue
        r12 = np.log((closes.iloc[abs_i] / closes.iloc[abs_i - 252]).clip(1e-9))
        r1  = np.log((closes.iloc[abs_i] / closes.iloc[abs_i - 21]).clip(1e-9))
        sig = (r12 - r1).fillna(0)
        w   = np.zeros(closes.shape[1])
        top4 = sig.nlargest(4).index
        bot4 = sig.nsmallest(4).index
        for t in top4:
            w[col_idx[t]] =  0.25
        for t in bot4:
            w[col_idx[t]] = -0.25
        if abs_i + 1 < len(closes):
            fwd = np.log((closes.iloc[abs_i + 1] /
                          closes.iloc[abs_i]).clip(1e-9)).fillna(0).values
            mom_pnl[pos] = (w * fwd).sum() - FEE * np.abs(w - prev_w).sum()
        prev_w = w.copy()
    results["Momentum_12_1"] = portfolio_metrics(_vol_target(mom_pnl))
    log.info("  [Baseline] Momentum_12_1        Sharpe=%.4f",
             results["Momentum_12_1"]["sharpe"])

    # 4. Min-Variance (rolling 60-day cov, monthly rebalance)
    mv_pnl  = np.zeros(len(test_idx))
    mv_prev = np.ones(closes.shape[1]) / closes.shape[1]
    n_assets = closes.shape[1]
    for pos, abs_i in enumerate(test_idx):
        if abs_i < 60:
            continue
        w_rets = np.log(closes.iloc[abs_i-60:abs_i] /
                        closes.iloc[abs_i-60:abs_i].shift(1)).fillna(0).values
        cov    = np.cov(w_rets.T) + np.eye(n_assets) * 1e-6
        w0     = np.ones(n_assets) / n_assets
        res    = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP",
                          bounds=[(0, 0.3)] * n_assets,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                          options={"maxiter": 50, "ftol": 1e-6})
        mv_w = res.x if res.success else w0
        if abs_i + 1 < len(closes):
            fwd  = np.log((closes.iloc[abs_i + 1] /
                           closes.iloc[abs_i]).clip(1e-9)).fillna(0).values
            mv_pnl[pos] = (mv_w * fwd).sum() - FEE * np.abs(mv_w - mv_prev).sum()
        mv_prev = mv_w.copy()
    results["Min_Variance"] = portfolio_metrics(_vol_target(mv_pnl))
    log.info("  [Baseline] Min_Variance         Sharpe=%.4f",
             results["Min_Variance"]["sharpe"])

    # 5. MLP-naïf (LSTM proxy) — 20 lagged SPY returns → next-day direction
    spy_all = closes[BENCHMARK]
    lr_spy  = np.log(spy_all / spy_all.shift(1)).fillna(0).values
    LAG     = 20
    X_tr, y_tr = [], []
    for i in train_idx:
        if i < LAG + HORIZON or i + HORIZON >= len(lr_spy):
            continue
        X_tr.append(lr_spy[i - LAG:i])
        y_tr.append(1 if lr_spy[i:i + HORIZON].sum() > 0 else 0)
    X_te, te_map = [], []
    for pos, i in enumerate(test_idx):
        if i < LAG:
            continue
        X_te.append(lr_spy[i - LAG:i])
        te_map.append(pos)
    mlp_pnl = np.zeros(len(test_idx))
    if len(X_tr) > 100 and len(X_te) > 10:
        sc     = StandardScaler()
        X_tr_s = sc.fit_transform(np.array(X_tr))
        X_te_s = sc.transform(np.array(X_te))
        mlp    = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                               random_state=42)
        mlp.fit(X_tr_s, y_tr)
        preds = mlp.predict_proba(X_te_s)[:, 1]
        for k, pos in enumerate(te_map):
            abs_i = test_idx[pos]
            if abs_i < len(lr_spy):
                sig = 2 * preds[k] - 1.0
                mlp_pnl[pos] = sig * lr_spy[abs_i] - FEE * abs(sig)
    results["MLP_naive"] = portfolio_metrics(_vol_target(mlp_pnl))
    log.info("  [Baseline] MLP_naive (LSTM prx) Sharpe=%.4f",
             results["MLP_naive"]["sharpe"])

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL FUNCTION (opt window)
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_fn(closes, features, targets, regime_filter, cs_ranker,
                  opt_idx, bm_features) -> Callable:
    bm_X     = bm_features.iloc[opt_idx].fillna(0).values.astype(np.float32)
    ml_proba = regime_filter.predict_proba(bm_X)
    cs_pred  = cs_ranker.predict(features, opt_idx)
    bm_close = closes[BENCHMARK] if BENCHMARK in closes.columns else closes.iloc[:, 0]

    def eval_fn(params: Dict[str, Any]) -> float:
        try:
            ts   = TrendEngine(don_win=int(params["don_win"]),
                               tf_win=int(params["tf_win"]),
                               w_don=0.5, w_tf=0.5)
            tsig = ts.compute(bm_close)
            csp  = CrossSectionPortfolio(
                long_thr=float(params["ml_thr"]),
                short_thr=1.0 - float(params["ml_thr"]),
                max_positions=4)
            cs_w = csp.compute(cs_pred)
            hyb  = HybridSignal(
                w_trend=float(params["w_trend"]), w_cs=float(params["w_cs"]),
                ml_thr=float(params["ml_thr"]),
                s_bull=float(params["s_bull"]),   s_bear=float(params["s_bear"]),
                s_side=float(params["s_side"]),   max_pos=float(params["max_pos"]),
                transaction_cost=FEE)
            pnl, _ = hyb.compute_returns(closes, tsig, cs_w, ml_proba, opt_idx)
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
            n, mid  = len(pnl_adj), len(pnl_adj) // 2
            sharpes = []
            for s, e in [(0, mid), (mid, n)]:
                seg = pnl_adj[s:e]
                if len(seg) < 20:
                    continue
                sr  = np.mean(seg) / (np.std(seg) + 1e-12) * np.sqrt(252)
                cum = np.exp(np.cumsum(seg))
                dd  = float(np.min((cum - np.maximum.accumulate(cum))
                                   / (np.maximum.accumulate(cum) + 1e-12)))
                sharpes.append(sr - max(0.0, -dd - 0.12) * 5.0)
            return float(np.mean(sharpes)) if sharpes else -999.0
        except Exception:
            return -999.0

    return eval_fn


# ══════════════════════════════════════════════════════════════════════════════
#  R4 — 10 000-AGENT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class UltraThinkOptimizerV4:
    def __init__(self, eval_fn: Callable, n_jobs: int = N_JOBS):
        self.eval_fn = eval_fn
        self.n_jobs  = n_jobs

    def run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        best_value  = -np.inf
        best_params = dict(DEFAULTS)
        summaries   = {}
        launched    = 0
        total       = sum(AGENT_CATEGORIES_V4.values())

        for category, n_trials in AGENT_CATEGORIES_V4.items():
            log.info("Agent category: %-20s  %5d trials  [%d/%d]",
                     category, n_trials, launched, total)
            study = optuna.create_study(
                direction="maximize",
                study_name=f"scaf_v4_{category}",
                sampler=optuna.samplers.TPESampler(
                    seed=42 + launched,
                    n_startup_trials=max(50, n_trials // 20)),
            )
            study.optimize(make_objective(self.eval_fn, category),
                           n_trials=n_trials,
                           n_jobs=self.n_jobs,
                           show_progress_bar=False)

            best_cat = study.best_value
            log.info("  Best: %.4f  params: %s", best_cat,
                     {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in study.best_params.items()})
            summaries[category] = {
                "best_value": round(best_cat, 4),
                "best_params": study.best_params,
                "n_trials":   len(study.trials),
                "n_complete": sum(1 for t in study.trials
                                  if t.state.name == "COMPLETE"),
            }
            if best_cat > best_value:
                best_value  = best_cat
                best_params = {**DEFAULTS, **study.best_params}
                tw = best_params.get("w_trend", 0.6) + best_params.get("w_cs", 0.4)
                if tw > 0:
                    best_params["w_trend"] /= tw
                    best_params["w_cs"]    /= tw
            launched += n_trials

        log.info("=" * 56)
        log.info("ULTRA-THINK V4 best Sharpe (opt window): %.4f", best_value)
        return best_params, summaries


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL OOS EVALUATION (R1 intégré)
# ══════════════════════════════════════════════════════════════════════════════

def final_evaluation(closes, features, targets, bm_features,
                     regime_filter, cs_ranker,
                     best_params, test_idx, index) -> Dict[str, Any]:
    log.info("Final OOS: %s -> %s  (%d jours)", TEST_START, END, len(test_idx))

    bm_X     = bm_features.iloc[test_idx].fillna(0).values.astype(np.float32)
    ml_proba = regime_filter.predict_proba(bm_X)
    cs_pred  = cs_ranker.predict(features, test_idx)
    bm_close = closes[BENCHMARK] if BENCHMARK in closes.columns else closes.iloc[:, 0]

    trend_sig = TrendEngine(don_win=int(best_params["don_win"]),
                            tf_win=int(best_params["tf_win"]),
                            w_don=0.5, w_tf=0.5).compute(bm_close)
    cs_w      = CrossSectionPortfolio(
        long_thr=float(best_params["ml_thr"]),
        short_thr=1.0 - float(best_params["ml_thr"]),
        max_positions=4).compute(cs_pred)
    hybrid    = HybridSignal(
        w_trend=float(best_params["w_trend"]), w_cs=float(best_params["w_cs"]),
        ml_thr=float(best_params["ml_thr"]),
        s_bull=float(best_params["s_bull"]),   s_bear=float(best_params["s_bear"]),
        s_side=float(best_params["s_side"]),   max_pos=float(best_params["max_pos"]),
        transaction_cost=FEE)
    pnl_raw, breakdown = hybrid.compute_returns(
        closes, trend_sig, cs_w, ml_proba, test_idx)

    rm      = RiskManager(target_vol=float(best_params.get("target_vol", 0.10)),
                          vol_window=int(best_params.get("vol_window", 20)))
    pnl_adj = rm.apply(pnl_raw, pnl_raw)

    # ── R1 : persister ────────────────────────────────────────────────────── #
    test_dates = closes.index[test_idx]
    persist_pnl(pnl_adj, pnl_raw, test_dates, RESULTS_DIR)

    metrics     = portfolio_metrics(pnl_adj)
    metrics_raw = portfolio_metrics(pnl_raw)
    bm_px       = bm_close.iloc[test_idx]
    bnh_lr      = np.log(bm_px / bm_px.shift(1)).fillna(0).values
    bnh_met     = portfolio_metrics(bnh_lr)

    thr    = float(best_params["ml_thr"])
    n_bull = int(np.sum(ml_proba >= thr))
    n_bear = int(np.sum(ml_proba <= (1 - thr)))
    n_side = len(ml_proba) - n_bull - n_bear

    return {
        "period":           f"{TEST_START} -> {END}",
        "n_days":           len(pnl_adj),
        **{f"adj_{k}": v for k, v in metrics.items()},
        **{f"raw_{k}": v for k, v in metrics_raw.items()},
        "bnh_sharpe":       bnh_met.get("sharpe", 0),
        "bnh_cum_ret":      bnh_met.get("cum_ret", 0),
        "excess_vs_bnh":    round(metrics.get("cum_ret", 0)
                                  - bnh_met.get("cum_ret", 0), 6),
        "regime_dist":      {"bull": n_bull, "bear": n_bear, "sideways": n_side},
        "signal_breakdown": breakdown.to_dict(),
        "pnl_series":       pnl_adj.tolist(),
        "pnl_dates":        [d.strftime("%Y-%m-%d") for d in test_dates],
        "targets_met": {
            "sharpe_gt_1": str(metrics.get("sharpe", 0) > 1.033),
            "dd_lt_15pct": metrics.get("max_dd", -1) > -0.15,
            "beats_bnh":   metrics.get("cum_ret", 0) > bnh_met.get("cum_ret", 0),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS (PnL réel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_v4(test_results, ablation, baselines, agent_summaries,
                closes, test_idx) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pnl       = np.array(test_results["pnl_series"])
    dates     = pd.to_datetime(test_results["pnl_dates"])
    cum       = np.exp(np.cumsum(pnl))
    peak      = np.maximum.accumulate(cum)
    dd        = (cum - peak) / (peak + 1e-12)
    bm_close  = closes[BENCHMARK].iloc[test_idx] if BENCHMARK in closes.columns \
                else closes.iloc[test_idx, 0]
    bnh_cum   = np.exp(np.cumsum(np.log(bm_close / bm_close.shift(1)).fillna(0).values))

    # Fig 1 — Equity + Drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={"hspace": 0.35})
    ax1.plot(dates, cum,     color="#2563eb", lw=2,
             label=f"SCAF v4  Sharpe={test_results['adj_sharpe']:.4f}")
    ax1.plot(dates, bnh_cum, color="#9ca3af", lw=1.2, ls="--",
             label=f"B&H SPY  Sharpe={test_results['bnh_sharpe']:.4f}")
    ax1.fill_between(dates, cum, 1, alpha=0.08, color="#2563eb")
    ax1.set_title("SCAF v4 — Courbe de Capital (PnL réel persisté OOS 2022-2024)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Valeur (base 1)"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.fill_between(dates, dd * 100, 0, color="#ef4444", alpha=0.6)
    ax2.set_title(f"Drawdown  Max={test_results['adj_max_dd']*100:.2f}%",
                  fontsize=12); ax2.set_ylabel("DD (%)"); ax2.grid(alpha=0.3)
    fig.savefig(RESULTS_DIR / "01_equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2 — Ablation
    fig, ax = plt.subplots(figsize=(10, 5))
    cfg   = list(ablation.keys())
    shs   = [ablation[c]["sharpe"] for c in cfg]
    ref_s = shs[0]
    clrs  = ["#2563eb" if i == 0 else ("#ef4444" if s < ref_s else "#16a34a")
             for i, s in enumerate(shs)]
    bars  = ax.bar(cfg, shs, color=clrs, edgecolor="white", width=0.55)
    for bar, s in zip(bars, shs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f"{s:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(ref_s, color="#9ca3af", lw=1, ls="--", label="Référence SCAF_Full")
    ax.set_title("R2 — Ablation Formelle : contribution de chaque composant",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe OOS 2022-2024"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.savefig(RESULTS_DIR / "02_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 3 — Baselines vs SCAF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bnames = list(baselines.keys()) + ["SCAF_v4"]
    bsh    = [baselines[b]["sharpe"] for b in baselines] + [test_results["adj_sharpe"]]
    bdds   = [abs(baselines[b].get("max_dd", 0)) * 100 for b in baselines] + \
             [abs(test_results["adj_max_dd"]) * 100]
    bclrs  = ["#9ca3af"] * len(baselines) + ["#2563eb"]
    for ax, vals, title, ylabel in [
        (ax1, bsh,  "R3 — Sharpe Ratio : SCAF v4 vs 5 Baselines", "Sharpe OOS"),
        (ax2, bdds, "R3 — Max Drawdown : SCAF v4 vs 5 Baselines", "Max DD (%)"),
    ]:
        bars = ax.bar(bnames, vals, color=bclrs, edgecolor="white", width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_ylabel(ylabel)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_baselines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 4 — Agent categories
    fig, ax = plt.subplots(figsize=(11, 5))
    clrs4 = ["#2563eb","#16a34a","#9333ea","#f59e0b","#ef4444"]
    for i, (cat, info) in enumerate(agent_summaries.items()):
        ax.barh(cat, info["best_value"], color=clrs4[i], height=0.6,
                label=f"{cat} ({info['n_trials']}t) → {info['best_value']:.2f}")
    ax.set_title(f"R4 — 10 000 Agents Optuna par Catégorie (n_jobs={N_JOBS})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Best Sharpe (opt window)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(RESULTS_DIR / "04_agents.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 5 — Rolling Sharpe 90j
    fig, ax = plt.subplots(figsize=(13, 5))
    roll = pd.Series(pnl).rolling(90).apply(
        lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(252)).values
    ax.plot(dates, roll, color="#9333ea", lw=1.5)
    ax.fill_between(dates, roll, 0, where=roll > 0, color="#16a34a", alpha=0.15)
    ax.fill_between(dates, roll, 0, where=roll <= 0, color="#ef4444", alpha=0.15)
    ax.axhline(test_results["adj_sharpe"], color="#2563eb", lw=1, ls="--",
               label=f"Sharpe global {test_results['adj_sharpe']:.4f}")
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.set_title("Sharpe Glissant 90j — PnL Réel Persisté", fontsize=12,
                 fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(RESULTS_DIR / "05_rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("5 graphiques (PnL reel) -> %s", RESULTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(tr, ablation, baselines, best_params, summaries, elapsed):
    sep = "-" * 68
    print(f"\n{'=' * 68}")
    print(f"  SCAF v4  |  10 000 agents  |  OOS 2022-2024  ({tr['n_days']} jours)")
    print(f"{'=' * 68}")
    for label, val, target, ok in [
        ("Sharpe Ratio (adj)",  f"{tr['adj_sharpe']:.4f}", "> 1.033",
         tr['adj_sharpe'] > 1.033),
        ("Max Drawdown (adj)",  f"{tr['adj_max_dd']*100:.2f}%", "> -15%",
         tr['adj_max_dd'] > -0.15),
        ("Rend. Annuel",        f"{tr['adj_ann_ret']*100:.1f}%", "", True),
        ("Calmar Ratio",        f"{tr['adj_calmar']:.2f}", "", True),
        ("Exces vs B&H",        f"+{tr['excess_vs_bnh']*100:.1f}%", "> 0",
         tr['excess_vs_bnh'] > 0),
    ]:
        print(f"  {label:<28} {val:>10}   {target:>10}  {'[OK]' if ok else '[--]'}")

    print(f"\n{sep}")
    print("  R2 Ablation (OOS):")
    ref = ablation.get("SCAF_Full", {}).get("sharpe", 0)
    for name, m in ablation.items():
        print(f"    {name:<22} Sharpe={m['sharpe']:.4f}  "
              f"delta={m['sharpe']-ref:+.4f}  DD={m.get('max_dd',0)*100:.2f}%")

    print(f"\n{sep}")
    print("  R3 Baselines (OOS):")
    for bn, bm in baselines.items():
        print(f"    {bn:<22} Sharpe={bm['sharpe']:.4f}  "
              f"Cum={bm.get('cum_ret',0)*100:.1f}%")
    best_b = max(b["sharpe"] for b in baselines.values())
    print(f"    {'SCAF_v4':<22} Sharpe={tr['adj_sharpe']:.4f}  "
          f"(+{(tr['adj_sharpe']/max(best_b, 0.01) - 1)*100:.0f}% vs best baseline)")

    print(f"\n{sep}")
    print(f"  R4 Agents (opt window, n_jobs={N_JOBS}):")
    for cat, info in summaries.items():
        print(f"    {cat:<22} Sharpe={info['best_value']:.4f}  "
              f"({info['n_trials']} trials)")
    print(f"\n  Resultats : {RESULTS_DIR}")
    print(f"  Duree     : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"{'=' * 68}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 64)
    log.info("  SCAF v4  |  10 000 agents (n_jobs=%d)  [%s]", N_JOBS, RUN_ID)
    log.info("  total_trials=%d  |  R1+R2+R3+R4", sum(AGENT_CATEGORIES_V4.values()))
    log.info("=" * 64)

    # ── [1] Data ─────────────────────────────────────────────────────────── #
    log.info("[1/8] Loading multi-asset data...")
    loader = MultiAssetLoader(START, END, use_cache=True)
    ohlcv, sentiment = loader.load()
    ohlcv, sentiment, index = align_universe(ohlcv, sentiment)   # 3 valeurs!

    # Closes panel (aligned)
    closes = pd.DataFrame(
        {t: df["Close"] for t, df in ohlcv.items()}
    ).reindex(index).ffill()
    log.info("Universe: %d assets  |  %d days", len(closes.columns), len(index))

    # ── [2] Features ─────────────────────────────────────────────────────── #
    log.info("[2/8] Building feature panel...")
    features, targets = build_feature_panel(ohlcv, sentiment, index, horizon=HORIZON)
    bm_ticker   = BENCHMARK if BENCHMARK in features else list(features.keys())[0]
    bm_features = features[bm_ticker]
    log.info("Features: %d assets x %d cols",
             len(features), next(iter(features.values())).shape[1])

    # ── Time splits ─────────────────────────────────────────────────────── #
    ml_train_idx = np.where(index <= ML_TRAIN_END)[0]
    opt_idx      = np.where((index >= OPT_START) & (index <= OPT_END))[0]
    test_idx     = np.where(index >= TEST_START)[0]
    log.info("Split — Train:%d  Opt:%d  Test:%d",
             len(ml_train_idx), len(opt_idx), len(test_idx))

    # ── [3] RegimeFilter ─────────────────────────────────────────────────── #
    log.info("[3/8] Training RegimeFilter...")
    n_cal    = max(100, len(ml_train_idx) // 5)
    bm_X_tr  = bm_features.iloc[ml_train_idx].fillna(0).values.astype(np.float32)
    bm_y_tr  = (targets.get(bm_ticker, pd.Series(0, index=index))
                .iloc[ml_train_idx].fillna(0).values.astype(int))
    rf       = RegimeFilter(threshold=0.55)
    rf.fit(bm_X_tr[:-n_cal], bm_y_tr[:-n_cal],
           bm_X_tr[-n_cal:],  bm_y_tr[-n_cal:])
    log.info("  AUC: %s  |  Active models: %d", rf.auc_scores_, len(rf.models_))

    # ── [4] CrossSectionRanker ───────────────────────────────────────────── #
    log.info("[4/8] Training CrossSectionRanker...")
    csr = CrossSectionRanker(threshold=0.55)
    csr.fit(features, targets, ml_train_idx)
    log.info("  CS AUC: %s", csr._ensemble.auc_scores_)

    # ── [5] R4 : 10 000 agents ───────────────────────────────────────────── #
    log.info("[5/8] Ultra-Think V4: 10 000 agents (n_jobs=%d)...", N_JOBS)
    eval_fn   = build_eval_fn(closes, features, targets,
                              rf, csr, opt_idx, bm_features)
    optimizer = UltraThinkOptimizerV4(eval_fn, n_jobs=N_JOBS)
    best_params, agent_summaries = optimizer.run()

    # ── [6] R1 : Final OOS + PnL persisté ──────────────────────────────── #
    log.info("[6/8] Final OOS evaluation (R1: PnL persisté)...")
    test_results = final_evaluation(
        closes, features, targets, bm_features,
        rf, csr, best_params, test_idx, index)

    # ── [7] R2 : Ablation ────────────────────────────────────────────────── #
    log.info("[7/8] Ablation formelle (R2 : 4 configs)...")
    ablation = run_ablation(closes, features, bm_features,
                            rf, csr, best_params, test_idx)

    # ── [8] R3 : Baselines + Rapport + Charts ──────────────────────────── #
    log.info("[8/8] Baselines (R3) + rapport + charts...")
    baselines = compute_baselines(closes, ml_train_idx, test_idx, features)

    elapsed = time.time() - t0

    # ── Save JSON ────────────────────────────────────────────────────────── #
    report = {
        "run_id": RUN_ID, "version": "SCAF v4",
        "elapsed_s": round(elapsed, 1),
        "n_agents":  sum(AGENT_CATEGORIES_V4.values()),
        "n_jobs":    N_JOBS,
        "best_params":    best_params,
        "test_results":   {k: v for k, v in test_results.items()
                           if k != "pnl_series"},
        "pnl_series":     test_results["pnl_series"],
        "pnl_dates":      test_results["pnl_dates"],
        "ablation":       {n: {k: round(float(v), 6) for k, v in m.items()}
                           for n, m in ablation.items()},
        "baselines":      {n: {k: round(float(v), 6) for k, v in m.items()}
                           for n, m in baselines.items()},
        "agent_summaries": agent_summaries,
        "universe_size":  len(closes.columns),
        "n_features":     next(iter(features.values())).shape[1],
        "regime_filter_auc": rf.auc_scores_,
        "cs_ranker_auc":     csr._ensemble.auc_scores_,
    }
    rpath = RESULTS_DIR / "scaf_v4_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report: %s", rpath)

    print_report(test_results, ablation, baselines,
                 best_params, agent_summaries, elapsed)
    try:
        plot_all_v4(test_results, ablation, baselines,
                    agent_summaries, closes, test_idx)
    except Exception as e:
        log.warning("Charts skipped: %s", e)

    return report


if __name__ == "__main__":
    main()
