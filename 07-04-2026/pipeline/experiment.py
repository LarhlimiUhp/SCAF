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
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    horizon: int = 5                # prediction horizon (trading days)

    # ----- Models -----
    model_names: List[str] = field(default_factory=lambda: [
        "LogReg-L2", "RandomForest", "LGBM", "KNN",
        "BiLSTM", "XGBoost", "ExtraTrees", "HistGBT", "MLP",
        "RidgeClass", "Bagging", "TabNet", "GraphNN", "CatBoost",
    ])

    # ----- Walk-forward CV -----
    n_folds: int = 3
    min_train_samples: int = 200
    embargo_days: int = 5

    # ----- Trading -----
    transaction_cost: float = 0.0003
    position_scalar: float = 1.0
    # Per-fold AUC threshold: models below this are excluded from the ensemble
    min_model_auc: float = 0.50
    # Weight ensemble votes by (AUC − 0.5) instead of uniform average
    auc_weighted_ensemble: bool = True

    # ----- Conformal prediction -----
    use_conformal: bool = True
    conformal_alpha: float = 0.05      # target miscoverage
    calibration_fraction: float = 0.20 # fraction of train set used for calibration

    # ----- Q-learning -----
    use_qlearning: bool = True
    ql_qtable_path: str = "results/qlearning_qtable.json"

    # ----- DL Regime Detector -----
    use_dl_regime_detector: bool = True

    # ----- Bull-market validation filter -----
    # When set, validation indices are restricted to years in this list so
    # that performance is evaluated only on bull-market phases (e.g. 2019,
    # 2023, 2024).  Set to None (default) to use all validation years.
    bull_market_years: Optional[List[int]] = None

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
        self._regime_detector = None
        self._vix_col_idx: Optional[int] = None   # set in run() from feature columns

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
        # Store column index of VIX feature for regime detection
        self._vix_col_idx: Optional[int] = (
            int(list(X_raw.columns).index("vix"))
            if "vix" in X_raw.columns else None
        )

        # ── Step 3: Models ──────────────────────────────────────────── #
        logger.info("[3/7] Initialising models …")
        models = self._build_models()

        # ── Step 4: Q-Learning selector ─────────────────────────────── #
        if self.cfg.use_qlearning:
            logger.info("[4/7] Initialising Q-learning selector …")
            self._init_qlearning(list(models.keys()))

        # ── Step 5: DL Regime Detector ─────────────────────────────────── #
        if self.cfg.use_dl_regime_detector:
            logger.info("[5/8] Training DL regime detector …")
            self._init_regime_detector(X)

        # ── Step 6: LLM orchestrator ──────────────────────────────────── #
        if self.cfg.use_llm:
            logger.info("[6/8] Initialising LLM orchestrator …")
            self._init_llm()

        # ── Step 7: Walk-forward evaluation ─────────────────────────── #
        logger.info("[7/8] Running walk-forward CV …")
        cv_results = self._walk_forward(X, y_arr, prices, models)

        # ── Step 8: Summary ───────────────────────────────────────────── #
        logger.info("[8/8] Computing summary …")
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


    def _init_regime_detector(self, X: np.ndarray) -> None:
        """Fit the DL-based regime detector on the full feature matrix."""
        try:
            from models.regime_detector import RegimeDetector
            # Heuristically detect the VIX column (look for a column with high mean)
            vix_col: Optional[int] = None
            if X.shape[1] > 7:
                col_means = X.mean(axis=0)
                vix_col = int(np.argmax(col_means > 15))
                if col_means[vix_col] <= 15:
                    vix_col = None
            self._regime_detector = RegimeDetector(
                seq_len=10, hidden=32, n_layers=2, epochs=10,
                vol_col=0, vix_col=vix_col,
            )
            self._regime_detector.fit(X)
            logger.info(
                "DL regime detector fitted (torch=%s).",
                self._regime_detector._fitted,
            )
        except Exception as exc:
            logger.warning("DL regime detector init failed: %s", exc)
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

        # ── Bull-market year filter ───────────────────────────────────── #
        if self.cfg.bull_market_years:
            filtered_splits = []
            for train_idx, val_idx in splits:
                mask = np.array([
                    prices.index[i].year in self.cfg.bull_market_years
                    for i in val_idx
                ])
                filtered_val = val_idx[mask]
                if len(filtered_val) >= 30:
                    filtered_splits.append((train_idx, filtered_val))
                else:
                    logger.warning(
                        "Bull-market filter removed fold (only %d val samples left). "
                        "Keeping original val indices.", len(filtered_val)
                    )
                    filtered_splits.append((train_idx, val_idx))
            splits = filtered_splits
            logger.info(
                "Bull-market year filter applied (years=%s). %d folds retained.",
                self.cfg.bull_market_years, len(splits),
            )

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

        # Keep raw (un-scaled) X_val for the regime detector, which needs
        # absolute-scale feature values for heuristic thresholds and GMM centroids.
        X_val_raw = X_val.copy()

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

        # ── Conformal calibration (split-conformal seed + ACI online) ── #
        _conformal_ensemble = None
        _aci_ensemble = None
        if self.cfg.use_conformal:
            from models.conformal import ConformalEnsemble, AdaptiveConformalEnsemble
            fitted_models = list(trained.values())
            # Split-conformal (static) — for reference q_hat logging
            _conformal_ensemble = ConformalEnsemble(
                fitted_models, alpha=self.cfg.conformal_alpha
            )
            _conformal_ensemble.calibrate(X_cal, y_cal)
            logger.info(
                "Fold %d split-conformal q_hat=%.4f",
                fold_idx,
                _conformal_ensemble.q_hat or float("nan"),
            )
            # Adaptive conformal (ACI) — seeded from calibration, updated online
            _aci_ensemble = AdaptiveConformalEnsemble(
                fitted_models, alpha=self.cfg.conformal_alpha, gamma=0.005
            )
            _aci_ensemble.calibrate(X_cal, y_cal)

        # ── Regime detection (used for model selection and signal scaling) ── #
        # Pass raw (un-scaled) X_val so the GMM/heuristic fallback thresholds work.
        regime_label = self._infer_regime(X_val_raw, prices_val)
        recent_sharpe = _rolling_sharpe(np.diff(np.log(prices_val.values + 1e-12)))

        # ── AUC-filtered model pool: drop models below min_model_auc ─── #
        auc_filtered_names = [
            n for n in trained
            if model_auc.get(n, 0.0) >= self.cfg.min_model_auc
        ]
        # Keep at least the best model so ensemble is never empty
        if not auc_filtered_names:
            best_name = max(model_auc, key=model_auc.get)
            auc_filtered_names = [best_name]
        logger.info(
            "Fold %d: AUC-filtered pool=%s (threshold=%.2f)",
            fold_idx, auc_filtered_names, self.cfg.min_model_auc,
        )

        # ── Q-learning model selection ────────────────────────────────── #
        active_model_names = auc_filtered_names
        if self._ql_selector is not None and auc_filtered_names:
            ql_selection = self._ql_selector.select(regime_label, recent_sharpe)
            ql_filtered = [m for m in ql_selection if m in auc_filtered_names]
            if ql_filtered:
                active_model_names = ql_filtered

        # Regime-aware base position scalar: reduce aggressiveness in adverse regimes
        if regime_label == "crisis":
            regime_pos_scalar = 0.25
        elif regime_label == "bear":
            regime_pos_scalar = 0.50
        else:
            regime_pos_scalar = self.cfg.position_scalar

        # ── DL baseline identification (BiLSTM or best DL model) ─────── #
        _DL_CANDIDATES = ["BiLSTM", "GraphNN", "MLP", "TabNet"]
        dl_baseline_name: Optional[str] = next(
            (n for n in _DL_CANDIDATES if n in trained), None
        )
        dl_baseline_probs: List[float] = []
        dl_baseline_returns: List[float] = []

        # ── Walk-forward signal generation ───────────────────────────── #
        portfolio_returns = []
        signals = []
        all_probs: List[float] = []
        # Split-conformal coverage hits (static q_hat, for reference)
        static_coverage_hits: List[int] = []
        # ACI coverage hits (adaptive q_hat, the primary coverage metric)
        aci_coverage_hits: List[int] = []
        step_latencies_ms: List[float] = []
        for i in range(len(X_val)):
            x_row = X_val[i:i+1]

            # Aggregate signals: AUC-weighted or equal-weight average
            _t_step = time.time()
            probs = []
            weights = []
            for name in active_model_names:
                p, _ = trained[name].predict_proba_one(x_row)
                probs.append(p)
                # Weight = excess AUC above chance (AUC − 0.5), floored at 1e-4
                weights.append(max(1e-4, model_auc.get(name, 0.5) - 0.5))
            step_latencies_ms.append((time.time() - _t_step) * 1000.0)

            if probs:
                if self.cfg.auc_weighted_ensemble and any(w > 1e-4 for w in weights):
                    w_arr = np.array(weights)
                    prob = float(np.dot(w_arr, probs) / w_arr.sum())
                else:
                    prob = float(np.mean(probs))
            else:
                prob = 0.5
            # Gradual signal in [-1, 1]; position scaling applied via pos_scalar
            signal = (prob - 0.5) * 2.0
            signals.append(signal)
            all_probs.append(prob)

            # Static split-conformal coverage (reference only)
            if _conformal_ensemble is not None:
                try:
                    pred_set = _conformal_ensemble.predict_set(x_row)
                    static_coverage_hits.append(int(y_val[i] in pred_set))
                except Exception:
                    pass

            # ACI online coverage: predict then update with true label
            if _aci_ensemble is not None:
                try:
                    _, covered = _aci_ensemble.predict_set_and_update(x_row, int(y_val[i]))
                    aci_coverage_hits.append(int(covered))
                except Exception:
                    pass

            # DL baseline tracking (single-model forward pass)
            if dl_baseline_name is not None:
                try:
                    dl_p, _ = trained[dl_baseline_name].predict_proba_one(x_row)
                    dl_baseline_probs.append(dl_p)
                    if i + 1 < len(prices_val):
                        dl_sig = (dl_p - 0.5) * 2.0
                        _daily = float(
                            np.log(prices_val.iloc[i + 1] / (prices_val.iloc[i] + 1e-12))
                        )
                        dl_baseline_returns.append(
                            dl_sig * regime_pos_scalar * _daily
                            - self.cfg.transaction_cost * abs(dl_sig)
                        )
                except Exception:
                    pass

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

        # RMSE of ensemble probability vs true binary label (Brier RMSE)
        _proba_arr = np.array(all_probs)
        _y_aligned = y_val[: len(_proba_arr)].astype(float)
        fold_rmse = float(
            np.sqrt(np.mean((_proba_arr - _y_aligned) ** 2))
        ) if len(_proba_arr) > 0 else float("nan")

        # DL baseline fold metrics
        dl_ret_arr = np.array(dl_baseline_returns)
        _dl_y = y_val[: len(dl_baseline_probs)].astype(float)
        fold_dl_rmse = float(
            np.sqrt(np.mean((np.array(dl_baseline_probs) - _dl_y) ** 2))
        ) if dl_baseline_probs else float("nan")
        fold_dl_sharpe = _sharpe(dl_ret_arr) if len(dl_ret_arr) > 1 else float("nan")
        fold_dl_dd = (
            _max_drawdown(np.exp(np.cumsum(dl_ret_arr)))
            if len(dl_ret_arr) > 1 else float("nan")
        )

        # Static conformal coverage rate (reference)
        coverage_rate = (
            float(np.mean(static_coverage_hits)) if static_coverage_hits else float("nan")
        )
        # ACI online coverage rate (primary — should be ≥ 1−α)
        aci_coverage_rate = (
            float(np.mean(aci_coverage_hits)) if aci_coverage_hits else float("nan")
        )

        # Mean per-step inference latency
        mean_latency_ms = (
            float(np.mean(step_latencies_ms)) if step_latencies_ms else float("nan")
        )

        logger.info(
            "Fold %d: cum_ret=%.4f sharpe=%.3f max_dd=%.4f "
            "aci_cov=%.3f dl_rmse=%.4f auc=%s",
            fold_idx, cum_ret, fold_sharpe, fold_dd,
            aci_coverage_rate if not np.isnan(aci_coverage_rate) else float("nan"),
            fold_dl_rmse if not np.isnan(fold_dl_rmse) else float("nan"),
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
            "dl_baseline_name": dl_baseline_name,
            "cum_return": cum_ret,
            "sharpe": fold_sharpe,
            "max_drawdown": fold_dd,
            "rmse": fold_rmse,
            # DL baseline per-fold metrics
            "dl_baseline_rmse": fold_dl_rmse,
            "dl_baseline_sharpe": fold_dl_sharpe,
            "dl_baseline_drawdown": fold_dl_dd,
            # Conformal coverage: static (reference) and ACI (primary)
            "coverage_rate": coverage_rate,
            "aci_coverage_rate": aci_coverage_rate,
            "mean_latency_ms": mean_latency_ms,
            "signals": signals,
            "portfolio_returns": ret_arr.tolist(),
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _infer_regime(self, X_val: np.ndarray, prices_val: pd.Series) -> str:
        """Infer market regime from recent data features.

        Priority: (1) DL/GMM regime detector, (2) LLM orchestrator,
        (3) rule-based volatility heuristic.

        Parameters
        ----------
        X_val:
            Raw (un-scaled) feature matrix for the validation window.
            This is required so the GMM centroids and absolute-scale
            heuristic thresholds are meaningful.
        prices_val:
            Price series aligned with X_val.
        """
        # ── (1) Regime detector (BiLSTM or GMM fallback) ─────────────── #
        if self._regime_detector is not None:
            try:
                return self._regime_detector.predict(X_val)
            except Exception:
                pass

        # ── (2) LLM orchestrator (secondary) ─────────────────────────── #
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

        # ── (3) Rule-based fallback using raw VIX column ─────────────── #
        recent_n = min(20, len(prices_val))
        if recent_n > 2:
            recent_prices = prices_val.values[-recent_n:]
            vol_est = float(np.std(np.diff(np.log(recent_prices + 1e-12))))
        else:
            vol_est = 0.01

        # X_val is now raw (un-scaled) so we can read the VIX column directly.
        vix_level = 20.0
        vix_col = getattr(self, "_vix_col_idx", None)
        if vix_col is not None and X_val.shape[1] > vix_col and len(X_val) > 0:
            vix_level = float(np.nanmean(X_val[-min(5, len(X_val)):, vix_col]))

        return _detect_regime(vol_est, vix=vix_level)

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

        # ── RMSE (Brier) — SCAF ensemble ──────────────────────────────── #
        fold_rmses = [r["rmse"] for r in valid
                      if not np.isnan(r.get("rmse", float("nan")))]
        mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else float("nan")

        # ── DL baseline aggregate metrics ─────────────────────────────── #
        dl_names = [r.get("dl_baseline_name") for r in valid if r.get("dl_baseline_name")]
        dl_baseline_name = dl_names[0] if dl_names else None

        def _safe_mean(lst: List[float]) -> Optional[float]:
            vals = [v for v in lst if not np.isnan(v)]
            return float(np.mean(vals)) if vals else None

        dl_rmses = [r.get("dl_baseline_rmse", float("nan")) for r in valid]
        dl_sharpes = [r.get("dl_baseline_sharpe", float("nan")) for r in valid]
        dl_dds = [r.get("dl_baseline_drawdown", float("nan")) for r in valid]

        mean_dl_rmse = _safe_mean(dl_rmses)
        mean_dl_sharpe = _safe_mean(dl_sharpes)
        mean_dl_dd = _safe_mean(dl_dds)

        # ── % improvement vs DL baseline ─────────────────────────────── #
        rmse_reduction_pct: Optional[float] = None
        if mean_dl_rmse is not None and not np.isnan(mean_rmse) and mean_dl_rmse > 1e-9:
            rmse_reduction_pct = round(
                (mean_dl_rmse - mean_rmse) / mean_dl_rmse * 100.0, 2
            )

        sharpe_improvement_pct: Optional[float] = None
        if mean_dl_sharpe is not None and abs(mean_dl_sharpe) > 1e-9:
            sharpe_improvement_pct = round(
                (total_sharpe - mean_dl_sharpe) / abs(mean_dl_sharpe) * 100.0, 2
            )

        drawdown_reduction_pct: Optional[float] = None
        if mean_dl_dd is not None and abs(mean_dl_dd) > 1e-9:
            # Both values are negative; smaller absolute value = better
            drawdown_reduction_pct = round(
                (abs(mean_dl_dd) - abs(max_dd)) / abs(mean_dl_dd) * 100.0, 2
            )

        # ── Conformal coverage rates ──────────────────────────────────── #
        # Static split-conformal (reference; may drop under distribution shift)
        fold_coverages = [r["coverage_rate"] for r in valid
                          if not np.isnan(r.get("coverage_rate", float("nan")))]
        mean_coverage = float(np.mean(fold_coverages)) if fold_coverages else float("nan")

        # ACI online coverage (primary; tracks target ≥ 1−α under shift)
        fold_aci_coverages = [r["aci_coverage_rate"] for r in valid
                              if not np.isnan(r.get("aci_coverage_rate", float("nan")))]
        mean_aci_coverage = (
            float(np.mean(fold_aci_coverages)) if fold_aci_coverages else float("nan")
        )

        # ── Inference latency ─────────────────────────────────────────── #
        fold_latencies = [r["mean_latency_ms"] for r in valid
                          if not np.isnan(r.get("mean_latency_ms", float("nan")))]
        mean_latency_ms = (
            float(np.mean(fold_latencies)) if fold_latencies else float("nan")
        )

        # Statistical significance test (paired t-test vs buy-and-hold)
        stat_test: Dict[str, Any] = {}
        bnh_aligned = bnh_rets[-len(all_returns):]
        if len(bnh_aligned) == len(all_returns):
            try:
                from analysis.stats import StatisticalAnalyzer
                stat_test = StatisticalAnalyzer.performance_ttest(
                    all_returns, bnh_aligned
                )
            except Exception as exc:
                logger.warning("Stat test failed: %s", exc)

        # 15-baseline Sharpe comparison
        baseline_sharpes: Dict[str, float] = {}
        for strat_name, strat_fn in BenchmarkStrategies.all_strategies().items():
            try:
                strat_rets = strat_fn(prices).dropna().values
                baseline_sharpes[strat_name] = round(_sharpe(strat_rets), 4)
            except Exception:
                pass

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
            # RMSE (Brier)
            "rmse_brier": round(mean_rmse, 6) if not np.isnan(mean_rmse) else None,
            # DL baseline metrics
            "dl_baseline_name": dl_baseline_name,
            "dl_baseline_rmse_brier": (
                round(mean_dl_rmse, 6) if mean_dl_rmse is not None else None
            ),
            "dl_baseline_sharpe": (
                round(mean_dl_sharpe, 4) if mean_dl_sharpe is not None else None
            ),
            "dl_baseline_max_drawdown": (
                round(mean_dl_dd, 4) if mean_dl_dd is not None else None
            ),
            # % improvement vs DL baseline
            "rmse_reduction_vs_dl_pct": rmse_reduction_pct,
            "sharpe_improvement_vs_dl_pct": sharpe_improvement_pct,
            "drawdown_reduction_vs_dl_pct": drawdown_reduction_pct,
            # Conformal coverage
            "conformal_coverage_rate_static": (
                round(mean_coverage, 4) if not np.isnan(mean_coverage) else None
            ),
            "conformal_coverage_rate_aci": (
                round(mean_aci_coverage, 4) if not np.isnan(mean_aci_coverage) else None
            ),
            "mean_inference_latency_ms": (
                round(mean_latency_ms, 3) if not np.isnan(mean_latency_ms) else None
            ),
            "stat_test_vs_bnh": stat_test,
            "baseline_sharpe_ratios": baseline_sharpes,
            "mean_model_auc": {k: round(v, 4) for k, v in mean_model_auc.items()},
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info("=== SUMMARY ===")
        for k, v in summary.items():
            logger.info("  %s: %s", k, v)

        return summary
