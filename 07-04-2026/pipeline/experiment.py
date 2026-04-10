"""
End-to-End Experiment Pipeline for SCAF-LS

Orchestrates the full workflow:
  1. Data loading (real or synthetic)
  2. Cross-asset feature engineering
  3. Walk-forward cross-validation
  4. Model training + conformal calibration
  5. Q-learning model selection (optional)
  6. LLM risk override (optional)
  7. Performance evaluation + benchmark comparison

Usage (minimal)
---------------
    from pipeline.experiment import ExperimentPipeline

    pipeline = ExperimentPipeline()
    results = pipeline.run()
    print(results['summary'])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────── Config ─────────────────────────────────────── #

@dataclass
class ExperimentConfig:
    """Hyper-parameters and switches for the experiment pipeline."""

    # ----- Data -----
    ticker: str = "^GSPC"
    start_date: str = "2006-01-01"
    end_date: str = "2009-12-31"
    horizon: int = 5                # prediction horizon (trading days)

    # ----- Models -----
    model_names: List[str] = field(default_factory=lambda: [
        "LogReg-L2", "RandomForest", "LGBM", "KNN",
        "BiLSTM", "XGBoost", "ExtraTrees", "HistGBT", "MLP",
        "RidgeClass", "Bagging",
    ])

    # ----- Walk-forward CV -----
    n_folds: int = 3
    min_train_samples: int = 200
    embargo_days: int = 5

    # ----- Trading -----
    transaction_cost: float = 0.0003
    position_scalar: float = 1.0

    # ----- Conformal prediction -----
    use_conformal: bool = True
    conformal_alpha: float = 0.05      # target miscoverage
    calibration_fraction: float = 0.20 # fraction of train set used for calibration

    # ----- Q-learning -----
    use_qlearning: bool = True
    ql_qtable_path: str = "results/qlearning_qtable.json"

    # ----- LLM -----
    use_llm: bool = True
    llm_cache_ttl: int = 300

    # ----- Output -----
    results_dir: str = "results"


# ─────────────────────────── helpers ────────────────────────────────────── #

def _sharpe(returns: np.ndarray, periods: int = 252) -> float:
    m = float(np.mean(returns))
    s = float(np.std(returns)) + 1e-12
    return m / s * np.sqrt(periods)


def _max_drawdown(cum_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / (peak + 1e-12)
    return float(np.min(drawdown))


def _rolling_sharpe(returns: np.ndarray, window: int = 60) -> float:
    if len(returns) < window:
        return 0.0
    r = returns[-window:]
    return _sharpe(r)


def _detect_regime(vol_5d: float, vix: float = 20.0) -> str:
    """Simple rule-based regime detection used when LLM is offline."""
    if vix > 35 or vol_5d > 0.03:
        return "crisis"
    if vol_5d > 0.015:
        return "bear"
    if vol_5d < 0.007:
        return "bull"
    return "sideways"


# ─────────────────────────── walk-forward split ─────────────────────────── #

def _walk_forward_splits(
    n: int,
    n_folds: int,
    min_train: int,
    embargo: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (train_indices, val_indices) for walk-forward CV."""
    splits = []
    fold_size = (n - min_train) // n_folds
    for fold in range(n_folds):
        val_end = n - fold_size * (n_folds - fold - 1)
        val_start = val_end - fold_size
        train_end = val_start - embargo
        if train_end < min_train:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
    return splits


# ─────────────────────────── ExperimentPipeline ─────────────────────────── #

class ExperimentPipeline:
    """End-to-end SCAF-LS experiment pipeline.

    Parameters
    ----------
    config:
        ExperimentConfig instance; uses defaults when None.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.cfg = config or ExperimentConfig()

        # Lazy-initialised components
        self._ql_selector = None
        self._llm = None

    # ------------------------------------------------------------------ #
    #  Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline and return results."""
        import os
        os.makedirs(self.cfg.results_dir, exist_ok=True)

        t0 = time.time()
        logger.info("=== SCAF-LS Experiment Pipeline starting ===")

        # ── Step 1: Data ─────────────────────────────────────────────── #
        logger.info("[1/7] Loading data …")
        X_raw, y, prices = self._load_data()
        if X_raw is None:
            return {"error": "Data loading failed"}

        # ── Step 2: Convert features (per-fold scaling applied in _run_fold) ── #
        logger.info("[2/7] Converting features …")
        X = X_raw.values.astype(np.float32)
        y_arr = y.values.astype(int)

        # ── Step 3: Models ──────────────────────────────────────────── #
        logger.info("[3/7] Initialising models …")
        models = self._build_models()

        # ── Step 4: Q-Learning selector ─────────────────────────────── #
        if self.cfg.use_qlearning:
            logger.info("[4/7] Initialising Q-learning selector …")
            self._init_qlearning(list(models.keys()))

        # ── Step 5: LLM orchestrator ─────────────────────────────────── #
        if self.cfg.use_llm:
            logger.info("[5/7] Initialising LLM orchestrator …")
            self._init_llm()

        # ── Step 6: Walk-forward evaluation ─────────────────────────── #
        logger.info("[6/7] Running walk-forward CV …")
        cv_results = self._walk_forward(X, y_arr, prices, models)

        # ── Step 7: Summary ──────────────────────────────────────────── #
        logger.info("[7/7] Computing summary …")
        summary = self._summarise(cv_results, prices, time.time() - t0)

        logger.info("=== Pipeline complete in %.1f s ===", time.time() - t0)
        return {
            "config": self.cfg,
            "cv_results": cv_results,
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    #  Step implementations                                                 #
    # ------------------------------------------------------------------ #

    def _load_data(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        """Load or synthesise data and engineer features."""
        # Try real data via MultiAssetLoader
        try:
            from data.loader import MultiAssetLoader

            class _Cfg:
                USE_REAL_DATA = True
                TICKER = self.cfg.ticker
                START_DATE = self.cfg.start_date
                END_DATE = self.cfg.end_date
                CROSS_ASSET_TICKERS: Dict[str, str] = {
                    "vix": "^VIX", "tnx": "^TNX", "irx": "^IRX",
                    "gold": "GC=F", "oil": "CL=F",
                }
                def end_date(self): return self.END_DATE  # noqa: N805

            loader = MultiAssetLoader(_Cfg())
            spx, cross = loader.download()
            if spx is not None and len(spx) >= self.cfg.min_train_samples:
                from data.engineer import CrossAssetFeatureEngineer
                engineer = CrossAssetFeatureEngineer(horizon=self.cfg.horizon)
                X_df, y, prices, _ = engineer.build(spx, cross or {})
                logger.info("Real data loaded: %d samples, %d features", len(X_df), X_df.shape[1])
                return X_df, y, prices
        except Exception as exc:
            logger.warning("Real data unavailable (%s), falling back to synthetic.", exc)

        # Synthetic fallback
        return self._synthetic_data()

    def _synthetic_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Generate a synthetic dataset for offline testing."""
        logger.info("Generating synthetic data …")
        rng = np.random.default_rng(42)
        n = 1000
        dates = pd.date_range("2006-01-01", periods=n, freq="B")

        # Price (GBM)
        rets = rng.normal(0.0003, 0.012, n)
        prices = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=dates, name="Close")

        # Features
        feat_dict: Dict[str, np.ndarray] = {}
        for lag in [1, 5, 10, 20]:
            r = pd.Series(rets).rolling(lag).sum().shift(1).fillna(0).values
            feat_dict[f"ret_{lag}d"] = r
        for lag in [5, 20, 50]:
            v = pd.Series(rets).rolling(lag).std().shift(1).fillna(0).values
            feat_dict[f"vol_{lag}d"] = v
        feat_dict["rsi_14"] = rng.uniform(20, 80, n)
        feat_dict["vix"] = rng.lognormal(2.9, 0.4, n)
        feat_dict["macd"] = rng.normal(0, 0.5, n)

        X_df = pd.DataFrame(feat_dict, index=dates)

        # Target: forward 5-day return > 0
        fwd = pd.Series(rets).rolling(self.cfg.horizon).sum().shift(-self.cfg.horizon)
        y = (fwd > 0).astype(int)
        y.index = dates

        valid = y.notna() & X_df.notna().all(axis=1)
        return X_df.loc[valid], y.loc[valid], prices.loc[valid]

    def _build_models(self) -> Dict[str, Any]:
        """Instantiate models from the registry."""
        # Import extra models to trigger registration
        try:
            import models.extra_models  # noqa: F401
        except ImportError:
            pass

        from models.registry import registry

        active = {}
        for name in self.cfg.model_names:
            try:
                model = registry.create(name)
                active[name] = model
            except KeyError:
                logger.warning("Model %s not in registry — skipping.", name)
        logger.info("Loaded models: %s", list(active.keys()))
        return active

    def _init_qlearning(self, model_names: List[str]) -> None:
        from models.qlearning_selector import QLearningSelector
        import os

        if os.path.exists(self.cfg.ql_qtable_path):
            try:
                self._ql_selector = QLearningSelector.load(self.cfg.ql_qtable_path)
                logger.info("Q-table loaded from %s", self.cfg.ql_qtable_path)
                return
            except Exception as exc:
                logger.warning("Could not load Q-table: %s", exc)

        self._ql_selector = QLearningSelector(
            model_names=model_names,
            max_subset_size=min(4, len(model_names)),
        )

    def _init_llm(self) -> None:
        try:
            from llm.orchestrator import LLMOrchestrator
            self._llm = LLMOrchestrator(cache_ttl=self.cfg.llm_cache_ttl)
            logger.info("LLM orchestrator backend: %s", self._llm._backend)
        except Exception as exc:
            logger.warning("LLM orchestrator unavailable: %s", exc)

    # ------------------------------------------------------------------ #
    #  Walk-forward loop                                                    #
    # ------------------------------------------------------------------ #

    def _walk_forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prices: pd.Series,
        models: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run walk-forward cross-validation."""
        splits = _walk_forward_splits(
            len(X),
            self.cfg.n_folds,
            self.cfg.min_train_samples,
            self.cfg.embargo_days,
        )
        if not splits:
            logger.error("No valid walk-forward splits found.")
            return []

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(
                "Fold %d/%d  train=%d  val=%d",
                fold_idx + 1, len(splits), len(train_idx), len(val_idx),
            )
            result = self._run_fold(
                fold_idx, X, y, prices, models, train_idx, val_idx
            )
            fold_results.append(result)

        return fold_results

    def _run_fold(
        self,
        fold_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        prices: pd.Series,
        models: Dict[str, Any],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> Dict[str, Any]:
        """Train, calibrate and evaluate one walk-forward fold."""
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        prices_val = prices.iloc[val_idx]

        # Split train into fit / calibration
        n_cal = max(50, int(len(X_train) * self.cfg.calibration_fraction))
        X_fit, y_fit = X_train[:-n_cal], y_train[:-n_cal]
        X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]

        # Per-fold feature scaling: fit only on X_fit to avoid data leakage
        _fold_scaler = StandardScaler()
        X_fit = _fold_scaler.fit_transform(X_fit)
        X_cal = _fold_scaler.transform(X_cal)
        X_val = _fold_scaler.transform(X_val)

        # ── Train models ──────────────────────────────────────────────── #
        trained: Dict[str, Any] = {}
        model_auc: Dict[str, float] = {}
        for name, model in models.items():
            try:
                model.fit(X_fit, y_fit)
                if model.is_fitted:
                    proba_val = np.array([
                        model.predict_proba_one(X_val[i:i+1])[0]
                        for i in range(len(X_val))
                    ])
                    if len(np.unique(y_val)) >= 2:
                        auc = roc_auc_score(y_val, proba_val)
                    else:
                        auc = 0.5
                    model_auc[name] = auc
                    trained[name] = model
            except Exception as exc:
                logger.warning("Model %s failed on fold %d: %s", name, fold_idx, exc)

        if not trained:
            return {"fold": fold_idx, "error": "all models failed"}

        # ── Conformal calibration ─────────────────────────────────────── #
        if self.cfg.use_conformal:
            from models.conformal import ConformalEnsemble
            fitted_models = list(trained.values())
            ensemble = ConformalEnsemble(fitted_models, alpha=self.cfg.conformal_alpha)
            ensemble.calibrate(X_cal, y_cal)
            logger.info(
                "Fold %d conformal q_hat=%.4f", fold_idx, ensemble.q_hat or float("nan")
            )

        # ── Regime detection (used for model selection and signal scaling) ── #
        regime_label = self._infer_regime(X_val, prices_val)
        recent_sharpe = _rolling_sharpe(np.diff(np.log(prices_val.values + 1e-12)))

        # ── Q-learning model selection ────────────────────────────────── #
        active_model_names = list(trained.keys())
        if self._ql_selector is not None and active_model_names:
            ql_selection = self._ql_selector.select(regime_label, recent_sharpe)
            active_model_names = [m for m in ql_selection if m in trained]
            if not active_model_names:
                active_model_names = list(trained.keys())

        # Regime-aware base position scalar: reduce aggressiveness in adverse regimes
        if regime_label == "crisis":
            regime_pos_scalar = 0.25
        elif regime_label == "bear":
            regime_pos_scalar = 0.60
        else:
            regime_pos_scalar = self.cfg.position_scalar

        # ── Walk-forward signal generation ───────────────────────────── #
        portfolio_returns = []
        signals = []
        for i in range(len(X_val)):
            x_row = X_val[i:i+1]

            # Aggregate signals from active models
            probs = []
            for name in active_model_names:
                p, _ = trained[name].predict_proba_one(x_row)
                probs.append(p)
            prob = float(np.mean(probs)) if probs else 0.5
            # Gradual signal in [-1, 1]; position scaling applied via pos_scalar
            signal = (prob - 0.5) * 2.0
            signals.append(signal)

            # Position scalar: regime-adjusted baseline, optionally overridden by LLM
            pos_scalar = regime_pos_scalar
            if self._llm is not None and i % 20 == 0:
                try:
                    vol_5d = float(np.std(
                        [trained[n].predict_proba_one(X_val[max(0, i-5):i+1])[0]
                         for n in active_model_names[:2]]
                    ))
                    override = self._llm.assess_risk_override(
                        current_drawdown=max(0.0, -_rolling_sharpe(
                            np.array(portfolio_returns[-20:]) if portfolio_returns else np.zeros(1)
                        ) * 0.01),
                        current_vol=vol_5d,
                        position_scalar=pos_scalar,
                        vix=20.0,
                        recent_returns=portfolio_returns[-5:] if portfolio_returns else [0.0],
                    )
                    if override.get("override"):
                        pos_scalar = override["new_position_scalar"]
                except Exception:
                    pass  # LLM offline — keep default scalar

            # Return calculation
            if i + 1 < len(prices_val):
                daily_ret = float(
                    np.log(prices_val.iloc[i + 1] / (prices_val.iloc[i] + 1e-12))
                )
                net_ret = signal * pos_scalar * daily_ret - self.cfg.transaction_cost * abs(signal)
                portfolio_returns.append(net_ret)

        # ── Fold metrics ─────────────────────────────────────────────── #
        ret_arr = np.array(portfolio_returns)
        cum_ret = float(np.sum(ret_arr))
        fold_sharpe = _sharpe(ret_arr) if len(ret_arr) > 1 else 0.0
        fold_dd = _max_drawdown(np.exp(np.cumsum(ret_arr)))

        logger.info(
            "Fold %d: cum_ret=%.4f sharpe=%.3f max_dd=%.4f auc=%s",
            fold_idx, cum_ret, fold_sharpe, fold_dd,
            {k: f"{v:.3f}" for k, v in model_auc.items()},
        )

        # Update Q-learning with fold outcome
        if self._ql_selector is not None and len(ret_arr) > 0:
            n_third = max(1, len(X_val) // 3)
            next_regime_label = self._infer_regime(
                X_val[-n_third:], prices_val.iloc[-n_third:]
            )
            next_sharpe = _rolling_sharpe(
                ret_arr[-n_third:] if len(ret_arr) >= n_third else ret_arr
            )
            self._ql_selector.observe(
                state_regime=regime_label,
                state_sharpe=_rolling_sharpe(ret_arr),
                action_models=active_model_names,
                ret=float(np.mean(ret_arr)),
                vol=float(np.std(ret_arr)) + 1e-12,
                drawdown=abs(fold_dd),
                next_regime=next_regime_label,
                next_sharpe=next_sharpe,
            )

        return {
            "fold": fold_idx,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "model_auc": model_auc,
            "active_models": active_model_names,
            "cum_return": cum_ret,
            "sharpe": fold_sharpe,
            "max_drawdown": fold_dd,
            "signals": signals,
            "portfolio_returns": ret_arr.tolist(),
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _infer_regime(self, X_val: np.ndarray, prices_val: pd.Series) -> str:
        """Infer market regime from recent data features."""
        if self._llm is not None:
            try:
                n = min(5, len(X_val))
                snapshot = {
                    "vol_5d": float(np.std(X_val[-n:, 0])) if X_val.shape[1] > 0 else 0.0,
                    "recent_ret": float(
                        np.log(prices_val.iloc[-1] / (prices_val.iloc[0] + 1e-12))
                    ) if len(prices_val) > 1 else 0.0,
                }
                result = self._llm.analyze_market(snapshot)
                return result.get("regime", "sideways")
            except Exception:
                pass
        # Rule-based fallback — use only the last 20 days for current-regime detection
        recent_n = min(20, len(prices_val))
        if recent_n > 2:
            recent_prices = prices_val.values[-recent_n:]
            vol_est = float(np.std(np.diff(np.log(recent_prices + 1e-12))))
        else:
            vol_est = 0.01
        return _detect_regime(vol_est)

    def _summarise(
        self,
        cv_results: List[Dict[str, Any]],
        prices: pd.Series,
        elapsed: float,
    ) -> Dict[str, Any]:
        """Aggregate fold-level results into a summary."""
        if not cv_results:
            return {"error": "no results"}

        valid = [r for r in cv_results if "error" not in r]
        if not valid:
            return {"error": "all folds failed"}

        all_returns = np.concatenate([r["portfolio_returns"] for r in valid])
        cumulative_return = float(np.sum(all_returns))
        total_sharpe = _sharpe(all_returns) if len(all_returns) > 1 else 0.0
        max_dd = _max_drawdown(np.exp(np.cumsum(all_returns)))

        # Benchmark: buy-and-hold
        from benchmark.strategies import BenchmarkStrategies
        bnh_rets = BenchmarkStrategies.buy_and_hold(prices).dropna().values
        bnh_cum = float(np.sum(bnh_rets[-len(all_returns):]))
        excess_ret = cumulative_return - bnh_cum

        # Per-model AUC across folds
        model_aucs: Dict[str, List[float]] = {}
        for r in valid:
            for m, auc in r.get("model_auc", {}).items():
                model_aucs.setdefault(m, []).append(auc)
        mean_model_auc = {m: float(np.mean(v)) for m, v in model_aucs.items()}

        # Save Q-table if enabled
        if self._ql_selector is not None:
            try:
                self._ql_selector.save(self.cfg.ql_qtable_path)
            except Exception as exc:
                logger.warning("Could not save Q-table: %s", exc)

        summary = {
            "n_folds_completed": len(valid),
            "n_samples_total": sum(r["n_val"] for r in valid),
            "cumulative_return": round(cumulative_return, 6),
            "sharpe_ratio": round(total_sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "excess_return_vs_bnh": round(excess_ret, 6),
            "mean_model_auc": {k: round(v, 4) for k, v in mean_model_auc.items()},
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info("=== SUMMARY ===")
        for k, v in summary.items():
            logger.info("  %s: %s", k, v)

        return summary
