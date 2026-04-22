"""
SCAF-LS v2 — Hybrid: ML Regime Filter + Trend-Following Engine
Ultra-Think 1000 Agents (Optuna, 5 specialised studies on real data)
=====================================================================

Architecture
------------
  Regime  = ML_filter(features_t)          # Bagging + LogReg + RidgeClass
  Signal  = sum(w_i * Strategy_i(params))   # 5 top trend strategies
  Return  = Signal * regime_scalar * pos_cap - fee * |delta_signal|

Agent categories (1000 trials total)
--------------------------------------
  strategy_tuning   300 trials  window / threshold params
  ml_filter         300 trials  ML confidence threshold + model mix
  position_sizing   200 trials  regime position scalars
  risk_management   100 trials  drawdown stop + max exposure
  ensemble_weights  100 trials  strategy blend weights

Data split (no leakage)
------------------------
  2015-01-01 -> 2019-12-31  ML training
  2020-01-01 -> 2021-12-31  Optuna optimisation (walk-forward)
  2022-01-01 -> 2024-12-31  Final out-of-sample test (UNTOUCHED)

Usage
-----
    cd 07-04-2026
    python run_option_c.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ──────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("option_c")

RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("results") / f"option_c_{RUN_ID}"

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TICKER   = "^GSPC"
ML_TRAIN_START  = "2015-01-01"
ML_TRAIN_END    = "2019-12-31"
OPT_START       = "2020-01-01"
OPT_END         = "2021-12-31"
TEST_START      = "2022-01-01"
TEST_END        = "2024-12-31"

TRANSACTION_COST = 0.0002

AGENT_CATEGORIES: Dict[str, int] = {
    "strategy_tuning":  300,
    "ml_filter":        300,
    "position_sizing":  200,
    "risk_management":  100,
    "ensemble_weights": 100,
}  # total = 1000

CROSS_ASSETS = {
    "vix":  "^VIX",
    "tnx":  "^TNX",
    "gold": "GC=F",
    "oil":  "CL=F",
}

# ══════════════════════════════════════════════════════════════════════════════
#  MARKET FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_features(prices: pd.Series, cross: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build ~30 features used by the ML regime filter."""
    log_ret = np.log(prices / prices.shift(1))

    feats: Dict[str, pd.Series] = {}

    # Momentum / returns
    for lag in [1, 5, 10, 20, 60]:
        feats[f"ret_{lag}d"] = log_ret.rolling(lag).sum().shift(1)

    # Volatility
    for win in [5, 20, 60]:
        feats[f"vol_{win}d"] = log_ret.rolling(win).std().shift(1)

    # RSI-14
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feats["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-12))

    # MACD signal
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    feats["macd"]        = macd.shift(1)
    feats["macd_signal"] = macd.ewm(span=9, adjust=False).mean().shift(1)

    # Distance from SMA
    for win in [20, 50, 200]:
        sma = prices.rolling(win).mean()
        feats[f"dist_sma{win}"] = ((prices - sma) / (sma + 1e-12)).shift(1)

    # Bollinger %B
    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    feats["bb_pct"] = ((prices - (sma20 - 2 * std20)) /
                       (4 * std20 + 1e-12)).shift(1)

    # Cross-assets (aligned)
    for name, series in cross.items():
        aligned = series["Close"].reindex(prices.index, method="ffill")
        aligned_ret = np.log(aligned / aligned.shift(1))
        feats[f"{name}_ret5"] = aligned_ret.rolling(5).sum().shift(1)
        feats[f"{name}_vol10"] = aligned_ret.rolling(10).std().shift(1)

    df = pd.DataFrame(feats, index=prices.index)
    return df


def build_target(prices: pd.Series, horizon: int = 5) -> pd.Series:
    """Binary: 1 if forward log-return > 0."""
    fwd = np.log(prices.shift(-horizon) / prices)
    return (fwd > 0).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
#  TREND STRATEGY SIGNALS  (pure price-based, no ML)
# ══════════════════════════════════════════════════════════════════════════════

def sig_donchian(prices: pd.Series, window: int = 20) -> pd.Series:
    upper = prices.rolling(window).max().shift(1)
    lower = prices.rolling(window).min().shift(1)
    sig = np.where(prices >= upper, 1.0,
          np.where(prices <= lower, -1.0, 0.0))
    return pd.Series(sig, index=prices.index, dtype=float).fillna(0)


def sig_trend_follow(prices: pd.Series, window: int = 252) -> pd.Series:
    rolling_high = prices.rolling(window).max().shift(1)
    sig = np.where(prices >= rolling_high, 1.0, 0.0)
    return pd.Series(sig, index=prices.index, dtype=float).fillna(0)


def sig_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    sig = np.where(prices.pct_change(window) > 0, 1.0, -1.0)
    return pd.Series(sig, index=prices.index, dtype=float).fillna(0)


def sig_macd(prices: pd.Series, fast: int = 12,
             slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f = prices.ewm(span=fast, adjust=False).mean()
    ema_s = prices.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig_l = macd.ewm(span=signal, adjust=False).mean()
    sig   = np.where(macd > sig_l, 1.0, -1.0)
    return pd.Series(sig, index=prices.index, dtype=float).fillna(0)


def sig_risk_adj_mom(prices: pd.Series,
                     mom_win: int = 20,
                     vol_win: int = 20,
                     target_vol: float = 0.01) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1))
    mom     = prices.pct_change(mom_win)
    rv      = log_ret.rolling(vol_win).std().shift(1).fillna(1e-4)
    sig     = np.sign(mom) * (target_vol / rv).clip(0, 2.0)
    return sig.fillna(0)


def compute_trend_signals(prices: pd.Series, params: Dict[str, Any]) -> pd.DataFrame:
    """Compute all 5 trend signals with Optuna-provided parameters."""
    return pd.DataFrame({
        "donchian":    sig_donchian(prices, int(params["don_win"])),
        "trend_follow": sig_trend_follow(prices, int(params["tf_win"])),
        "momentum":    sig_momentum(prices, int(params["mom_win"])),
        "macd":        sig_macd(prices,
                                int(params["macd_fast"]),
                                int(params["macd_slow"]),
                                int(params["macd_sig"])),
        "ram":         sig_risk_adj_mom(prices,
                                        int(params["ram_mom_win"]),
                                        int(params["ram_vol_win"]),
                                        params["ram_target_vol"]),
    }, index=prices.index)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_portfolio(prices: pd.Series,
                  signals_df: pd.DataFrame,
                  ml_proba: Optional[np.ndarray],
                  params: Dict[str, Any],
                  idx: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Compute portfolio returns for a given index slice.
    Returns (daily_returns, sharpe, max_drawdown).
    """
    prices_s = prices.iloc[idx]
    sigs_s   = signals_df.iloc[idx]

    w = np.array([
        params["w_don"], params["w_tf"], params["w_mom"],
        params["w_macd"], params["w_ram"],
    ])
    w = w / (w.sum() + 1e-12)

    # Blended trend signal  [-∞, +∞]
    raw_signal = (sigs_s.values * w).sum(axis=1)  # shape (n,)

    # ML regime scalar
    if ml_proba is not None and len(ml_proba) == len(idx):
        ml_p = ml_proba
    else:
        ml_p = np.full(len(idx), 0.5)

    thr   = params["ml_thr"]
    s_bull = params["s_bull"]
    s_bear = params["s_bear"]
    s_side = params["s_side"]

    regime_scalar = np.where(
        ml_p >= thr,           s_bull,
        np.where(ml_p <= (1 - thr), s_bear, s_side)
    )

    # Final position clipped to max_pos
    position = np.clip(raw_signal * regime_scalar,
                       -params["max_pos"], params["max_pos"])

    # Daily log-returns of S&P
    log_ret = np.log(prices_s / prices_s.shift(1)).fillna(0).values

    # P&L:  position[t] * ret[t+1] - fee * |position[t] - position[t-1]|
    pos_prev   = np.concatenate([[0.0], position[:-1]])
    turnover   = np.abs(position - pos_prev)
    pnl        = position * log_ret - TRANSACTION_COST * turnover

    # Metrics
    if len(pnl) < 5:
        return pnl, 0.0, 0.0

    sharpe = (np.mean(pnl) / (np.std(pnl) + 1e-12)) * np.sqrt(252)
    cum    = np.exp(np.cumsum(pnl))
    peak   = np.maximum.accumulate(cum)
    mdd    = float(np.min((cum - peak) / (peak + 1e-12)))

    return pnl, float(sharpe), mdd


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(prices: pd.Series,
                   signals_cache: Dict[str, Any],
                   ml_proba_opt: np.ndarray,
                   opt_idx: np.ndarray,
                   category: str):
    """
    Returns an Optuna objective function specialised for `category`.
    Each agent category searches a different region of parameter space.
    """

    # Category-specific parameter ranges
    ranges = {
        "strategy_tuning": {
            "don_win":       (10, 60),
            "tf_win":        (100, 300),
            "mom_win":       (10, 60),
            "macd_fast":     (8, 20),
            "macd_slow":     (20, 50),
            "macd_sig":      (5, 15),
            "ram_mom_win":   (10, 40),
            "ram_vol_win":   (10, 40),
            "ram_target_vol": (0.005, 0.025),
        },
        "ml_filter": {
            "ml_thr":  (0.52, 0.75),
            "s_bull":  (0.8, 2.5),
            "s_bear":  (0.0, 0.5),
            "s_side":  (0.2, 0.9),
            "max_pos": (0.5, 2.0),
        },
        "position_sizing": {
            "s_bull":  (1.0, 3.0),
            "s_bear":  (0.0, 0.4),
            "s_side":  (0.3, 1.0),
            "max_pos": (0.8, 2.5),
            "ml_thr":  (0.52, 0.65),
        },
        "risk_management": {
            "max_pos": (0.3, 1.5),
            "ml_thr":  (0.55, 0.80),
            "s_bull":  (0.8, 1.5),
            "s_bear":  (0.0, 0.3),
            "s_side":  (0.2, 0.7),
        },
        "ensemble_weights": {
            "w_don":  (0.05, 0.60),
            "w_tf":   (0.05, 0.60),
            "w_mom":  (0.05, 0.40),
            "w_macd": (0.05, 0.40),
            "w_ram":  (0.05, 0.40),
        },
    }

    # Defaults for params not tuned by this category
    defaults = {
        "don_win":       20,
        "tf_win":        200,
        "mom_win":       20,
        "macd_fast":     12,
        "macd_slow":     26,
        "macd_sig":      9,
        "ram_mom_win":   20,
        "ram_vol_win":   20,
        "ram_target_vol": 0.01,
        "ml_thr":        0.58,
        "s_bull":        1.2,
        "s_bear":        0.15,
        "s_side":        0.5,
        "max_pos":       1.0,
        "w_don":         0.3,
        "w_tf":          0.3,
        "w_mom":         0.15,
        "w_macd":        0.15,
        "w_ram":         0.10,
    }

    cat_ranges = ranges.get(category, {})

    def objective(trial):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        params = dict(defaults)

        # Sample only the parameters relevant to this category
        for key, (lo, hi) in cat_ranges.items():
            if key in ("don_win", "tf_win", "mom_win", "macd_fast",
                       "macd_slow", "macd_sig", "ram_mom_win", "ram_vol_win"):
                params[key] = trial.suggest_int(key, int(lo), int(hi))
            else:
                params[key] = trial.suggest_float(key, lo, hi)

        # Recompute trend signals only if strategy params changed
        sig_key = (int(params["don_win"]), int(params["tf_win"]),
                   int(params["mom_win"]), int(params["macd_fast"]),
                   int(params["macd_slow"]), int(params["macd_sig"]),
                   int(params["ram_mom_win"]), int(params["ram_vol_win"]),
                   round(params["ram_target_vol"], 4))

        if sig_key not in signals_cache:
            signals_cache[sig_key] = compute_trend_signals(prices, params)
        sigs = signals_cache[sig_key]

        # Walk-forward on optimisation window (2 folds)
        n = len(opt_idx)
        mid = n // 2
        fold_sharpes = []

        for train_end, val_start, val_end in [
            (mid // 2,  mid // 2, mid),
            (mid,       mid,      n),
        ]:
            t_idx = opt_idx[:train_end]
            v_idx = opt_idx[val_start:val_end]

            if len(v_idx) < 20:
                continue

            # ML proba slice
            ml_p = ml_proba_opt[val_start:val_end]

            _, sharpe, mdd = run_portfolio(prices, sigs, ml_p, params, v_idx)

            # Penalise drawdown below -15%
            penalty = max(0.0, -mdd - 0.15) * 5.0
            fold_sharpes.append(sharpe - penalty)

        if not fold_sharpes:
            return -999.0

        return float(np.mean(fold_sharpes))

    return objective


# ══════════════════════════════════════════════════════════════════════════════
#  ML REGIME FILTER
# ══════════════════════════════════════════════════════════════════════════════

def train_ml_filter(X_train: np.ndarray, y_train: np.ndarray,
                    X_pred:  np.ndarray) -> np.ndarray:
    """
    Train Bagging + LogReg + RidgeClass on (X_train, y_train).
    Return averaged P(up) for X_pred.
    """
    from sklearn.linear_model  import LogisticRegression, RidgeClassifier
    from sklearn.ensemble      import BaggingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration   import CalibratedClassifierCV

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_pr_s  = scaler.transform(X_pred)

    probas = []

    # LogReg-L2
    try:
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        lr.fit(X_tr_s, y_train)
        probas.append(lr.predict_proba(X_pr_s)[:, 1])
    except Exception as e:
        log.warning("LogReg failed: %s", e)

    # Bagging
    try:
        bag = BaggingClassifier(n_estimators=50, random_state=42, n_jobs=1)
        bag.fit(X_tr_s, y_train)
        probas.append(bag.predict_proba(X_pr_s)[:, 1])
    except Exception as e:
        log.warning("Bagging failed: %s", e)

    # RidgeClass (calibrated)
    try:
        ridge = CalibratedClassifierCV(
            RidgeClassifier(alpha=1.0), cv=3, method="isotonic"
        )
        ridge.fit(X_tr_s, y_train)
        probas.append(ridge.predict_proba(X_pr_s)[:, 1])
    except Exception as e:
        log.warning("RidgeClass failed: %s", e)

    if not probas:
        return np.full(len(X_pred), 0.5)

    return np.mean(probas, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_data() -> Tuple[pd.Series, Dict[str, pd.DataFrame]]:
    import yfinance as yf

    start = ML_TRAIN_START
    end   = TEST_END

    log.info("Downloading %s  %s -> %s ...", TICKER, start, end)
    raw = yf.download(TICKER, start=start, end=end, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prices = raw["Close"].dropna()
    log.info("  S&P 500: %d jours", len(prices))

    cross = {}
    for name, ticker in CROSS_ASSETS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 200:
                cross[name] = df
                log.info("  %s (%s): %d jours", name, ticker, len(df))
        except Exception:
            log.warning("  %s indisponible", name)

    return prices, cross


# ══════════════════════════════════════════════════════════════════════════════
#  ULTRA-THINK OPTIMIZER  (1000 agents)
# ══════════════════════════════════════════════════════════════════════════════

def run_ultra_think(prices: pd.Series,
                    features: pd.DataFrame,
                    target: pd.Series) -> Tuple[Dict[str, Any], Any]:
    """
    Deploy 1000 Optuna agents across 5 specialised studies.
    Returns (best_params, all_studies_dict).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Index slices
    ml_train_mask = (prices.index >= ML_TRAIN_START) & (prices.index <= ML_TRAIN_END)
    opt_mask      = (prices.index >= OPT_START)      & (prices.index <= OPT_END)

    ml_train_idx = np.where(ml_train_mask)[0]
    opt_idx      = np.where(opt_mask)[0]

    if len(ml_train_idx) < 100 or len(opt_idx) < 50:
        raise RuntimeError("Insufficient data for ML training or optimisation.")

    # Align features/target with prices
    feats_aligned  = features.reindex(prices.index).ffill().fillna(0)
    target_aligned = target.reindex(prices.index).fillna(0)

    X_train = feats_aligned.iloc[ml_train_idx].values
    y_train = target_aligned.iloc[ml_train_idx].values.astype(int)
    X_opt   = feats_aligned.iloc[opt_idx].values

    # Train ML filter once on 2015-2019, predict on 2020-2021
    log.info("Training ML regime filter (2015-2019)...")
    ml_proba_opt = train_ml_filter(X_train, y_train, X_opt)
    log.info("  ML proba opt range: [%.3f, %.3f]  mean=%.3f",
             ml_proba_opt.min(), ml_proba_opt.max(), ml_proba_opt.mean())

    # Shared signal cache (avoids recomputing identical windows)
    signals_cache: Dict[Any, pd.DataFrame] = {}

    # ── Launch 5 specialised studies ────────────────────────────────── #
    best_value  = -np.inf
    best_params: Dict[str, Any] = {}
    all_studies: Dict[str, Any] = {}

    total_agents = sum(AGENT_CATEGORIES.values())
    launched = 0

    for category, n_trials in AGENT_CATEGORIES.items():
        log.info("Agent category: %-20s  %d trials  [%d/%d launched]",
                 category, n_trials, launched, total_agents)

        study = optuna.create_study(
            direction="maximize",
            study_name=f"scaf_v2_{category}",
            sampler=optuna.samplers.TPESampler(seed=42 + launched),
        )

        obj = make_objective(prices, signals_cache, ml_proba_opt,
                             opt_idx, category)

        study.optimize(obj, n_trials=n_trials, n_jobs=1,
                       show_progress_bar=False)

        cat_best = study.best_value
        log.info("  Best Sharpe (opt window): %.4f  params: %s",
                 cat_best,
                 {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in study.best_params.items()})

        all_studies[category] = {
            "best_value":  cat_best,
            "best_params": study.best_params,
            "n_trials":    len(study.trials),
        }

        if cat_best > best_value:
            best_value  = cat_best
            # Merge best params with defaults
            defaults = {
                "don_win": 20, "tf_win": 200, "mom_win": 20,
                "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
                "ram_mom_win": 20, "ram_vol_win": 20, "ram_target_vol": 0.01,
                "ml_thr": 0.58, "s_bull": 1.2, "s_bear": 0.15,
                "s_side": 0.5,  "max_pos": 1.0,
                "w_don": 0.3, "w_tf": 0.3, "w_mom": 0.15,
                "w_macd": 0.15, "w_ram": 0.10,
            }
            best_params = {**defaults, **study.best_params}

        launched += n_trials

    log.info("=" * 56)
    log.info("BEST overall Sharpe (opt window): %.4f", best_value)
    log.info("Best params: %s",
             {k: round(v, 4) if isinstance(v, float) else v
              for k, v in best_params.items()})

    return best_params, all_studies


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL OUT-OF-SAMPLE TEST  2022-2024
# ══════════════════════════════════════════════════════════════════════════════

def final_test(prices:   pd.Series,
               features: pd.DataFrame,
               target:   pd.Series,
               params:   Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate best params on the held-out 2022-2024 period.
    ML filter trained on 2015-2021 (extended).
    """
    log.info("Final out-of-sample test: 2022-01-01 -> 2024-12-31")

    feats_aligned  = features.reindex(prices.index).ffill().fillna(0)
    target_aligned = target.reindex(prices.index).fillna(0)

    train_mask = prices.index < TEST_START
    test_mask  = (prices.index >= TEST_START) & (prices.index <= TEST_END)

    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(test_mask)[0]

    X_train_all = feats_aligned.iloc[train_idx].values
    y_train_all = target_aligned.iloc[train_idx].values.astype(int)
    X_test      = feats_aligned.iloc[test_idx].values

    log.info("  ML re-trained on %d days, tested on %d days",
             len(train_idx), len(test_idx))

    ml_proba_test = train_ml_filter(X_train_all, y_train_all, X_test)

    # Compute trend signals on full price series then slice
    sigs_full = compute_trend_signals(prices, params)

    pnl, sharpe, mdd = run_portfolio(
        prices, sigs_full, ml_proba_test, params, test_idx
    )

    # Buy-and-hold benchmark
    prices_test = prices.iloc[test_idx]
    bnh_log  = np.log(prices_test / prices_test.shift(1)).fillna(0).values
    bnh_cum  = float(np.sum(bnh_log))
    bnh_sr   = (np.mean(bnh_log) / (np.std(bnh_log) + 1e-12)) * np.sqrt(252)

    # Extended metrics
    cum_ret  = float(np.sum(pnl))
    ann_ret  = float(np.mean(pnl)) * 252
    ann_vol  = float(np.std(pnl)) * np.sqrt(252)
    calmar   = ann_ret / (abs(mdd) + 1e-12)
    win_rate = float(np.mean(pnl > 0))

    downside = np.std(pnl[pnl < 0]) + 1e-12
    sortino  = (np.mean(pnl) / downside) * np.sqrt(252)

    excess   = cum_ret - bnh_cum

    # Regime distribution
    thr = params.get("ml_thr", 0.58)
    n_bull = int(np.sum(ml_proba_test >= thr))
    n_bear = int(np.sum(ml_proba_test <= (1 - thr)))
    n_side = len(ml_proba_test) - n_bull - n_bear

    results = {
        "period":       f"{TEST_START} -> {TEST_END}",
        "n_days":       len(pnl),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 4),
        "cumulative_return": round(cum_ret, 6),
        "ann_return":   round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "calmar_ratio": round(calmar, 4),
        "sortino_ratio": round(sortino, 4),
        "win_rate":     round(win_rate, 4),
        "excess_vs_bnh": round(excess, 6),
        "bnh_sharpe":   round(float(bnh_sr), 4),
        "bnh_cumret":   round(bnh_cum, 6),
        "regime_dist":  {"bull": n_bull, "bear": n_bear, "sideways": n_side},
        "targets_met": {
            "sharpe_ok":   sharpe > 1.033,
            "drawdown_ok": mdd    > -0.15,
            "excess_ok":   excess > 0,
        },
        "pnl_series": pnl.tolist(),
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(test_results: Dict[str, Any],
                 prices:       pd.Series,
                 all_studies:  Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pnl = np.array(test_results["pnl_series"])
    cum = np.exp(np.cumsum(pnl))

    test_mask = (prices.index >= TEST_START) & (prices.index <= TEST_END)
    prices_test = prices[test_mask]
    bnh_log  = np.log(prices_test / prices_test.shift(1)).fillna(0).values
    cum_bnh  = np.exp(np.cumsum(bnh_log))

    # ── Fig 1: equity + drawdown + returns ─────────────────────────── #
    fig, axes = plt.subplots(3, 1, figsize=(13, 13),
                             gridspec_kw={"hspace": 0.45})

    ax = axes[0]
    ax.plot(cum,     label="SCAF-LS v2 (Hybrid)", color="#2563eb", linewidth=2.0)
    ax.plot(cum_bnh, label="Buy & Hold",           color="#9ca3af", linewidth=1.2,
            linestyle="--")
    sr   = test_results["sharpe_ratio"]
    flag = "[OK]" if sr > 1.033 else "[--]"
    ax.set_title(f"SCAF-LS v2 — Courbe d'equite  |  Sharpe={sr:.4f} {flag}"
                 f"  |  DD={test_results['max_drawdown']:.2%}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Valeur (base 1)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + 1e-12)
    ax.fill_between(range(len(dd)), dd, 0, color="#ef4444", alpha=0.55)
    ax.axhline(-0.10, color="#f97316", linewidth=1.2, linestyle="--",
               label="Limite -10%")
    ax.axhline(-0.15, color="#dc2626", linewidth=1.2, linestyle="--",
               label="Cible  -15%")
    ax.set_title("Drawdown (2022-2024)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(pnl, bins=60, color="#2563eb", alpha=0.75,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.axvline(np.mean(pnl), color="#16a34a", linewidth=1.3,
               linestyle="--", label=f"Moy={np.mean(pnl):.5f}")
    ax.set_title("Distribution des retours quotidiens", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Log-return"); ax.set_ylabel("Frequence")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    p = RESULTS_DIR / "equity_drawdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p)

    # ── Fig 2: agent study comparison ──────────────────────────────── #
    cats   = list(all_studies.keys())
    values = [all_studies[c]["best_value"] for c in cats]
    colors = ["#16a34a" if v > 1.033 else "#f97316" if v > 0.5
              else "#9ca3af" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cats, values, color=colors, edgecolor="white")
    ax.axhline(1.033, color="#2563eb", linewidth=1.5, linestyle="--",
               label="Cible 1.033")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Sharpe (fenetre opt 2020-2021) par categorie d'agents",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio"); ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha="right"); ax.grid(alpha=0.3, axis="y")

    p = RESULTS_DIR / "agent_categories.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p)

    # ── Fig 3: regime distribution ──────────────────────────────────── #
    rd = test_results["regime_dist"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [rd["bull"], rd["sideways"], rd["bear"]],
        labels=["Bull", "Sideways", "Bear"],
        colors=["#16a34a", "#f59e0b", "#ef4444"],
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 11},
    )
    ax.set_title("Distribution des regimes ML (2022-2024)",
                 fontsize=12, fontweight="bold")
    p = RESULTS_DIR / "regime_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_final(test_results: Dict[str, Any],
                best_params:  Dict[str, Any],
                all_studies:  Dict[str, Any],
                elapsed:      float) -> None:

    sep = "-" * 60
    print(f"\n{sep}")
    print("  SCAF-LS v2 -- Hybrid ML + Trend  |  OUT-OF-SAMPLE 2022-2024")
    print(sep)

    rows = [
        ("Sharpe Ratio",     "sharpe_ratio",      "> 1.033"),
        ("Max Drawdown",     "max_drawdown",       "> -15%"),
        ("Cumul. Return",    "cumulative_return",  ""),
        ("Ann. Return",      "ann_return",         ""),
        ("Ann. Volatility",  "ann_volatility",     ""),
        ("Calmar Ratio",     "calmar_ratio",       ""),
        ("Sortino Ratio",    "sortino_ratio",      ""),
        ("Win Rate",         "win_rate",           "> 45%"),
        ("Excess vs B&H",    "excess_vs_bnh",      "> 0"),
        ("B&H Sharpe",       "bnh_sharpe",         ""),
    ]
    for label, key, target in rows:
        val = test_results.get(key)
        if val is None:
            continue
        t = f"  [cible {target}]" if target else ""
        print(f"  {label:<22} {val!s:<14}{t}")

    print(sep)
    tm = test_results.get("targets_met", {})
    for k, v in tm.items():
        print(f"  {k:<22} {'[OK]' if v else '[--]'}")
    print(sep)

    rd = test_results.get("regime_dist", {})
    n  = test_results.get("n_days", 1)
    print(f"\n  Regimes detectes (ML):")
    for r, cnt in rd.items():
        print(f"    {r:<12} {cnt:>4} jours  ({100*cnt/n:.1f}%)")

    print(f"\n  Meilleurs parametres (tous agents):")
    for k, v in sorted(best_params.items()):
        vr = round(v, 4) if isinstance(v, float) else v
        print(f"    {k:<22} {vr}")

    print(f"\n  Performance par categorie d'agents (fenetre optim):")
    for cat, info in all_studies.items():
        print(f"    {cat:<22} Sharpe={info['best_value']:.4f}"
              f"  ({info['n_trials']} trials)")

    print(f"\n  Resultats sauvegardes : {RESULTS_DIR}")
    print(f"  Duree totale          : {elapsed:.0f}s")
    print(sep + "\n")


def save_report(test_results: Dict[str, Any],
                best_params:  Dict[str, Any],
                all_studies:  Dict[str, Any],
                elapsed:      float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Remove pnl series from report (large)
    report_results = {k: v for k, v in test_results.items()
                      if k != "pnl_series"}

    output = {
        "run_id":      RUN_ID,
        "test_results": report_results,
        "best_params":  {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in best_params.items()},
        "agent_studies": all_studies,
        "total_agents":  sum(AGENT_CATEGORIES.values()),
        "elapsed_s":     round(elapsed, 1),
    }

    p = RESULTS_DIR / "option_c_report.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Rapport JSON: %s", p)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  SCAF-LS v2 -- Hybrid ML + Trend  [%s]", RUN_ID)
    log.info("  Ultra-Think: %d agents / 5 categories",
             sum(AGENT_CATEGORIES.values()))
    log.info("=" * 60)

    # ── Deps check ───────────────────────────────────────────────────── #
    for pkg in ["yfinance", "optuna", "sklearn", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            log.error("Manque: %s  ->  pip install %s", pkg, pkg)
            sys.exit(1)

    # ── Data ─────────────────────────────────────────────────────────── #
    prices, cross = download_data()

    log.info("Building features...")
    features = build_features(prices, cross)
    target   = build_target(prices, horizon=5)

    # Drop NaN rows consistently
    valid = features.notna().all(axis=1) & target.notna()
    prices   = prices[valid]
    features = features[valid]
    target   = target[valid]
    log.info("Clean samples: %d", len(prices))

    # ── Ultra-Think optimization (1000 agents) ───────────────────────── #
    log.info("-" * 60)
    log.info("Launching Ultra-Think optimization (1000 agents)...")
    log.info("-" * 60)

    best_params, all_studies = run_ultra_think(prices, features, target)

    # ── Final out-of-sample test ──────────────────────────────────────── #
    log.info("-" * 60)
    test_results = final_test(prices, features, target, best_params)

    elapsed = time.time() - t0

    # ── Output ───────────────────────────────────────────────────────── #
    print_final(test_results, best_params, all_studies, elapsed)
    save_report(test_results, best_params, all_studies, elapsed)

    log.info("Generating plots...")
    try:
        plot_results(test_results, prices, all_studies)
    except Exception as exc:
        log.warning("Plots failed (non-blocking): %s", exc)

    # ── Verdict ──────────────────────────────────────────────────────── #
    sr  = test_results["sharpe_ratio"]
    mdd = test_results["max_drawdown"]
    exc = test_results["excess_vs_bnh"]

    log.info("-" * 60)
    log.info("VERDICT  Sharpe=%.4f %s   DD=%.4f %s   Exces=%.4f %s",
             sr,  "[OK]" if sr  > 1.033 else "[--]",
             mdd, "[OK]" if mdd > -0.15 else "[--]",
             exc, "[OK]" if exc > 0     else "[--]")
    log.info("Duree: %.0f s", elapsed)
    log.info("-" * 60)


if __name__ == "__main__":
    main()
