"""
SCAF End-to-End Experiment Pipeline
=====================================
Orchestrates a complete walk-forward experiment on real market data:

  1. Download data via yfinance (S&P 500 + cross assets)
  2. Engineer features (CrossAssetFeatureEngineer)
  3. Walk-forward training / validation with purged splits
  4. Conformal prediction calibration on each fold's calibration window
  5. Q-Learning model selection updated every fold
  6. LLM cognitive orchestration at each prediction step
  7. Risk management via AgentOrchestrator (300 sub-agents)
  8. Benchmark comparison (buy-and-hold, momentum, 60/40, risk-parity)
  9. Full metrics report (Sharpe, CAGR, max drawdown, accuracy, AUC, CP coverage)
 10. Saves results to JSON + CSV

Quick start
-----------
    from pipeline.experiment import ExperimentPipeline, ExperimentConfig
    cfg = ExperimentConfig()
    pipeline = ExperimentPipeline(cfg)
    results = pipeline.run()
    results.print_summary()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # ---- Data ----
    ticker: str = "^GSPC"
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    forecast_horizon: int = 5          # days ahead to predict
    cross_asset_tickers: Dict[str, str] = field(default_factory=lambda: {
        "vix":  "^VIX",
        "tnx":  "^TNX",
        "irx":  "^IRX",
        "gold": "GC=F",
        "oil":  "CL=F",
        "dxy":  "DX-Y.NYB",
        "hyg":  "HYG",
        "spy":  "SPY",
        "qqq":  "QQQ",
        "xlk":  "XLK",
        "xlv":  "XLV",
        "eem":  "EEM",
    })

    # ---- Walk-forward ----
    train_window: int = 500            # trading days in the training window
    val_window: int = 63               # 1 quarter for validation / CP calibration
    step_size: int = 21                # roll forward 1 month at a time
    purge_gap: int = 5                 # days between train end and val start

    # ---- Model registry ----
    # All registered model names to include. Leave empty to auto-detect.
    models: List[str] = field(default_factory=list)

    # ---- Features ----
    n_top_features: int = 40

    # ---- Conformal Prediction ----
    cp_alpha: float = 0.05             # 95 % coverage target
    use_adaptive_cp: bool = True

    # ---- Q-Learning ----
    ql_alpha: float = 0.1
    ql_gamma: float = 0.9
    ql_epsilon: float = 1.0
    ql_max_subset_size: int = 3
    ql_checkpoint: str = "qtable.pkl"

    # ---- LLM ----
    use_llm: bool = True               # set False to disable LLM calls
    llm_backend: str = "auto"
    llm_model: str = "gpt-4o-mini"

    # ---- Risk agents ----
    use_risk_agents: bool = True

    # ---- Output ----
    output_dir: str = "results"
    experiment_name: str = "scaf_experiment"


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    config: ExperimentConfig
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_returns: Optional[pd.Series] = None
    benchmark_returns: Optional[Dict[str, pd.Series]] = None
    conformal_reports: List[Dict[str, Any]] = field(default_factory=list)
    qlearning_summary: Optional[Dict[str, Any]] = None
    llm_call_stats: Optional[Dict[str, int]] = None
    runtime_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict[str, float]:
        if not self.fold_metrics:
            return {}
        keys = [k for k in self.fold_metrics[0] if isinstance(self.fold_metrics[0][k], (int, float))]
        return {k: float(np.mean([f[k] for f in self.fold_metrics if k in f])) for k in keys}

    def sharpe(self, ann_factor: float = 252) -> float:
        if self.portfolio_returns is None or len(self.portfolio_returns) < 2:
            return float("nan")
        r = self.portfolio_returns
        return float(r.mean() / (r.std() + 1e-9) * np.sqrt(ann_factor))

    def cagr(self) -> float:
        if self.portfolio_returns is None or len(self.portfolio_returns) < 2:
            return float("nan")
        equity = (1 + self.portfolio_returns).cumprod()
        years = len(equity) / 252
        return float(equity.iloc[-1] ** (1 / max(years, 1e-3)) - 1)

    def max_drawdown(self) -> float:
        if self.portfolio_returns is None:
            return float("nan")
        equity = (1 + self.portfolio_returns).cumprod()
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        return float(dd.min())

    def cp_coverage(self) -> float:
        if not self.conformal_reports:
            return float("nan")
        vals = [r.get("coverage", float("nan")) for r in self.conformal_reports
                if "coverage" in r]
        return float(np.mean(vals)) if vals else float("nan")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_summary(self):
        agg = self.aggregate()
        sep = "=" * 60
        print(sep)
        print(f"  SCAF Experiment: {self.config.experiment_name}")
        print(sep)
        print(f"  Runtime          : {self.runtime_seconds:.1f}s")
        print(f"  Walk-forward folds: {len(self.fold_metrics)}")
        print()
        print(f"  Portfolio Sharpe : {self.sharpe():.3f}")
        print(f"  CAGR             : {self.cagr()*100:.2f}%")
        print(f"  Max Drawdown     : {self.max_drawdown()*100:.2f}%")
        print(f"  CP Coverage      : {self.cp_coverage()*100:.2f}%  (target {(1-self.config.cp_alpha)*100:.0f}%)")
        print()
        if self.benchmark_returns:
            print("  Benchmarks:")
            for name, ret in self.benchmark_returns.items():
                sr = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252))
                print(f"    {name:<18}: Sharpe={sr:.3f}")
        print()
        if agg:
            print("  Mean fold metrics:")
            for k, v in sorted(agg.items()):
                print(f"    {k:<22}: {v:.4f}")
        print(sep)

    def save(self, output_dir: str, name: str):
        os.makedirs(output_dir, exist_ok=True)
        # Summary JSON
        summary = {
            "experiment": name,
            "runtime_s": self.runtime_seconds,
            "n_folds": len(self.fold_metrics),
            "sharpe": self.sharpe(),
            "cagr": self.cagr(),
            "max_drawdown": self.max_drawdown(),
            "cp_coverage": self.cp_coverage(),
            "aggregate_fold_metrics": self.aggregate(),
            "qlearning": self.qlearning_summary,
            "llm_stats": self.llm_call_stats,
        }
        with open(f"{output_dir}/{name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Portfolio returns CSV
        if self.portfolio_returns is not None:
            self.portfolio_returns.to_csv(f"{output_dir}/{name}_returns.csv", header=["return"])

        # Benchmark returns
        if self.benchmark_returns:
            bench_df = pd.DataFrame(self.benchmark_returns)
            bench_df.to_csv(f"{output_dir}/{name}_benchmarks.csv")

        logger.info("Results saved to %s/", output_dir)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ExperimentPipeline:
    """
    Full SCAF walk-forward experiment pipeline.

    Parameters
    ----------
    cfg : ExperimentConfig
    """

    def __init__(self, cfg: Optional[ExperimentConfig] = None):
        self.cfg = cfg or ExperimentConfig()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResults:
        t0 = time.time()
        results = ExperimentResults(config=self.cfg)

        # ---- 1. Load data ----
        logger.info("Loading market data (%s → %s)…", self.cfg.start_date, self.cfg.end_date)
        spx_df, cross = self._load_data()

        # ---- 2. Feature engineering ----
        logger.info("Engineering features…")
        X, y, prices, log_returns = self._engineer_features(spx_df, cross)

        # ---- 3. Instantiate models ----
        logger.info("Instantiating model registry…")
        models = self._build_models()
        model_names = [m.name for m in models]
        logger.info("  Models: %s", model_names)

        # ---- 4. Q-Learning selector ----
        from models.qlearning_selector import QLearningModelSelector  # noqa
        ql = QLearningModelSelector(
            model_names=model_names,
            alpha=self.cfg.ql_alpha,
            gamma=self.cfg.ql_gamma,
            epsilon=self.cfg.ql_epsilon,
            max_subset_size=self.cfg.ql_max_subset_size,
        )
        ql_checkpoint = os.path.join(self.cfg.output_dir, self.cfg.ql_checkpoint)
        ql.load(ql_checkpoint)

        # ---- 5. LLM orchestrator ----
        llm = None
        if self.cfg.use_llm:
            from llm.orchestrator import LLMOrchestrator, LLMConfig  # noqa
            llm = LLMOrchestrator(LLMConfig(
                backend=self.cfg.llm_backend,
                model=self.cfg.llm_model,
                allow_offline_fallback=True,
            ))
            logger.info("  LLM online: %s", llm.is_online())

        # ---- 6. Risk agents ----
        orchestrator = None
        if self.cfg.use_risk_agents:
            from agents.framework import AgentOrchestrator  # noqa
            class _Cfg:
                pass
            orchestrator = AgentOrchestrator(_Cfg())

        # ---- 7. Walk-forward loop ----
        logger.info("Starting walk-forward validation…")
        all_preds: List[float] = []
        all_true: List[int] = []
        all_portfolio_rets: List[float] = []
        ensemble_sharpe = 0.0

        n = len(X)
        fold_idx = 0
        t_start = self.cfg.train_window + self.cfg.purge_gap

        while t_start + self.cfg.val_window <= n:
            train_end = t_start - self.cfg.purge_gap
            train_start = max(0, train_end - self.cfg.train_window)
            val_start = t_start
            val_end = min(n, val_start + self.cfg.val_window)

            X_train = X.iloc[train_start:train_end].values
            y_train = y.iloc[train_start:train_end].values
            X_val = X.iloc[val_start:val_end].values
            y_val = y.iloc[val_start:val_end].values
            ret_val = log_returns.iloc[val_start:val_end].values
            dates_val = X.index[val_start:val_end]

            # Skip folds without enough data
            if len(X_train) < 100 or len(np.unique(y_train)) < 2:
                t_start += self.cfg.step_size
                continue

            # ---- 7a. Determine Q-Learning regime ----
            regime = self._detect_regime(X_train, log_returns.iloc[train_start:train_end].values)
            selected_names = ql.select_models(regime, ensemble_sharpe, explore=True)
            active_models = [m for m in models if m.name in selected_names] or models

            # ---- 7b. Fit active models ----
            for m in active_models:
                try:
                    m.fit(X_train, y_train)
                except Exception as exc:
                    logger.debug("Model %s fit error: %s", m.name, exc)

            # ---- 7c. Conformal calibration on cal window ----
            cal_start = max(train_start, train_end - self.cfg.val_window)
            X_cal = X.iloc[cal_start:train_end].values
            y_cal = y.iloc[cal_start:train_end].values

            from models.conformal import ConformalEnsemble  # noqa
            cp_ensemble = ConformalEnsemble(
                active_models,
                alpha=self.cfg.cp_alpha,
                use_adaptive=self.cfg.use_adaptive_cp,
            )
            if len(X_cal) >= 30 and len(np.unique(y_cal)) >= 2:
                cp_ensemble.calibrate(X_cal, y_cal)

            # ---- 7d. Inference on validation window ----
            fold_preds: List[float] = []
            fold_true: List[int] = []
            fold_rets: List[float] = []

            for i in range(len(X_val)):
                x_row = X_val[i : i + 1]
                true_label = int(y_val[i])
                actual_ret = float(ret_val[i])

                # Conformal prediction
                point_pred, pred_set, is_certain, proba = cp_ensemble.predict(
                    x_row, y_true=true_label
                )

                # LLM regime override
                position_scalar = 1.0
                if llm is not None and i % 5 == 0:  # query LLM every 5 steps (cost control)
                    try:
                        market_stats = self._extract_market_stats(X_val, i)
                        model_sigs = {m.name: float(m.predict_proba_one(x_row)[0])
                                      for m in active_models if m.is_fitted}
                        risk_sigs = {}
                        if orchestrator is not None:
                            md = {"returns": list(ret_val[:i+1]), "current_vol": float(np.std(ret_val[:max(1,i)])*np.sqrt(252))}
                            agg_sig, _ = orchestrator.get_aggregate_signal(md)
                            risk_sigs = {"aggregate_risk": agg_sig}
                        decision = llm.analyze_market(market_stats, model_sigs, risk_sigs)
                        position_scalar = float(decision.get("position_scalar", 1.0))
                    except Exception:
                        pass

                # Risk agent override
                if orchestrator is not None:
                    try:
                        md = {
                            "returns": list(ret_val[:i+1]),
                            "current_vol": float(np.std(ret_val[:max(1,i)]) * np.sqrt(252)),
                            "current_equity": float(np.exp(np.sum(fold_rets))),
                            "current_drawdown": self._current_drawdown(fold_rets),
                        }
                        agg_sig, _ = orchestrator.get_aggregate_signal(md)
                        position_scalar *= max(0.0, 1.0 + float(agg_sig))
                    except Exception:
                        pass

                # Portfolio return: signal × actual return
                signal = (proba - 0.5) * 2  # scale to [-1, 1]
                port_ret = float(np.clip(signal * position_scalar, -1.0, 1.0) * actual_ret)

                fold_preds.append(proba)
                fold_true.append(true_label)
                fold_rets.append(port_ret)

            # ---- 7e. Q-Learning update ----
            if fold_rets:
                fold_pnl = sum(fold_rets)
                fold_sharpe = self._compute_sharpe(fold_rets)
                fold_dd = self._compute_max_drawdown(fold_rets)

                next_regime = self._detect_regime(
                    X.iloc[val_start:val_end].values, ret_val
                )
                td_err = ql.update(next_regime, fold_sharpe, fold_pnl, fold_dd)
                ql.replay(batch_size=min(32, max(1, len(ql._replay))))
                ensemble_sharpe = fold_sharpe

            ql.end_episode()

            # ---- 7f. Fold metrics ----
            fold_metrics = self._compute_fold_metrics(
                fold_preds, fold_true, fold_rets, fold_idx, dates_val
            )

            # Conformal coverage
            if cp_ensemble._is_calibrated:
                cp_report = cp_ensemble.coverage_report(X_val, y_val)
                fold_metrics["cp_coverage"] = cp_report.get("coverage", float("nan"))
                fold_metrics["cp_avg_set_size"] = cp_report.get("avg_set_size", float("nan"))
                results.conformal_reports.append(cp_report)

            results.fold_metrics.append(fold_metrics)
            all_preds.extend(fold_preds)
            all_true.extend(fold_true)
            all_portfolio_rets.extend(fold_rets)

            fold_idx += 1
            t_start += self.cfg.step_size

            if fold_idx % 5 == 0:
                logger.info(
                    "  Fold %d | regime=%s | models=%s | sharpe=%.3f | cp_cov=%.3f",
                    fold_idx, regime, selected_names,
                    fold_metrics.get("sharpe", 0.0),
                    fold_metrics.get("cp_coverage", float("nan")),
                )

        # ---- 8. Benchmarks ----
        results.portfolio_returns = pd.Series(
            all_portfolio_rets,
            index=X.index[self.cfg.train_window + self.cfg.purge_gap : 
                          self.cfg.train_window + self.cfg.purge_gap + len(all_portfolio_rets)],
            name="scaf",
        )
        results.benchmark_returns = self._run_benchmarks(prices, log_returns, results.portfolio_returns.index)

        # ---- 9. Save Q-Learning state ----
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        ql.save(ql_checkpoint)
        results.qlearning_summary = ql.summary()
        if llm is not None:
            results.llm_call_stats = llm.stats

        results.runtime_seconds = time.time() - t0

        # ---- 10. Save results ----
        results.save(self.cfg.output_dir, self.cfg.experiment_name)
        results.print_summary()

        return results

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for real data. Install with: pip install yfinance"
            )

        def _dl(ticker: str) -> Optional[pd.DataFrame]:
            try:
                df = yf.download(ticker, start=self.cfg.start_date,
                                  end=self.cfg.end_date, progress=False)
                if df is None or df.empty:
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if 'Adj Close' in df.columns and 'Close' not in df.columns:
                    df = df.rename(columns={'Adj Close': 'Close'})
                return df[['Close', 'Volume'] if 'Volume' in df.columns else ['Close']].dropna(subset=['Close'])
            except Exception as e:
                logger.warning("Download failed for %s: %s", ticker, e)
                return None

        spx_df = _dl(self.cfg.ticker)
        if spx_df is None or len(spx_df) < 500:
            raise RuntimeError(
                f"Failed to download sufficient data for {self.cfg.ticker}. "
                "Check your internet connection and ticker symbol."
            )

        cross: Dict[str, pd.DataFrame] = {}
        for name, ticker in self.cfg.cross_asset_tickers.items():
            df = _dl(ticker)
            if df is not None and len(df) > 100:
                cross[name] = df
            else:
                logger.warning("  Skipping cross asset %s (%s)", name, ticker)

        logger.info(
            "Loaded %d rows for %s; %d cross-asset series.",
            len(spx_df), self.cfg.ticker, len(cross),
        )
        return spx_df, cross

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(
        self, spx_df: pd.DataFrame, cross: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        # Inline lightweight config object
        class _Cfg:
            N_TOP_FEATURES = self.cfg.n_top_features

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.engineer import CrossAssetFeatureEngineer  # noqa

        eng = CrossAssetFeatureEngineer(horizon=self.cfg.forecast_horizon)
        X, y, prices, log_returns = eng.build(spx_df, cross, cfg=_Cfg())
        return X, y, prices, log_returns

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def _build_models(self) -> List[Any]:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models import sklearn_models, torch_models, extra_models  # noqa – trigger registration
        from models.registry import registry  # noqa

        names = self.cfg.models or registry.names()
        models = []
        for name in names:
            try:
                m = registry.create(name)
                models.append(m)
            except Exception as exc:
                logger.warning("Could not instantiate model %s: %s", name, exc)

        if not models:
            raise RuntimeError("No models could be instantiated from the registry.")
        return models

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(self, X: np.ndarray, returns: np.ndarray) -> str:
        if len(returns) < 20:
            return "sideways"
        recent = returns[-20:]
        vol = float(np.std(recent) * np.sqrt(252))
        trend = float(np.mean(recent))

        if vol > 0.35:
            return "crisis"
        if trend > 0.001 and vol < 0.20:
            return "bull"
        if trend < -0.001:
            return "bear"
        return "sideways"

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(returns: List[float], ann: float = 252) -> float:
        r = np.array(returns)
        return float(r.mean() / (r.std() + 1e-9) * np.sqrt(ann)) if len(r) > 1 else 0.0

    @staticmethod
    def _compute_max_drawdown(returns: List[float]) -> float:
        equity = np.cumprod(1 + np.array(returns))
        roll_max = np.maximum.accumulate(equity)
        dd = (equity - roll_max) / roll_max
        return float(abs(dd.min())) if len(dd) > 0 else 0.0

    @staticmethod
    def _current_drawdown(returns: List[float]) -> float:
        if not returns:
            return 0.0
        equity = np.cumprod(1 + np.array(returns))
        peak = equity.max()
        return float((peak - equity[-1]) / peak)

    @staticmethod
    def _extract_market_stats(X_val: np.ndarray, i: int) -> Dict[str, float]:
        """Extract simplified market statistics from the feature matrix."""
        # Feature indices are approximate – use safe defaults if out of range
        def safe_get(idx: int) -> float:
            try:
                return float(X_val[i, idx])
            except (IndexError, ValueError):
                return 0.0

        return {
            "ret_5d": safe_get(1),
            "vol_20d_ann": abs(safe_get(5)) * np.sqrt(252),
            "vix": safe_get(12) if X_val.shape[1] > 12 else 20.0,
            "yield_spread": safe_get(15) if X_val.shape[1] > 15 else 0.0,
            "zscore_20d": safe_get(7) if X_val.shape[1] > 7 else 0.0,
        }

    def _compute_fold_metrics(
        self,
        preds: List[float],
        true: List[int],
        rets: List[float],
        fold_idx: int,
        dates: pd.DatetimeIndex,
    ) -> Dict[str, Any]:
        from sklearn.metrics import roc_auc_score, accuracy_score  # noqa

        metrics: Dict[str, Any] = {
            "fold": fold_idx,
            "start_date": str(dates[0]) if len(dates) > 0 else "",
            "end_date": str(dates[-1]) if len(dates) > 0 else "",
            "n_samples": len(preds),
        }

        if len(preds) > 0 and len(np.unique(true)) > 1:
            try:
                metrics["auc"] = float(roc_auc_score(true, preds))
            except Exception:
                metrics["auc"] = 0.5
            try:
                hard = [int(p >= 0.5) for p in preds]
                metrics["accuracy"] = float(accuracy_score(true, hard))
            except Exception:
                metrics["accuracy"] = 0.5

        if rets:
            metrics["sharpe"] = self._compute_sharpe(rets)
            metrics["max_drawdown"] = self._compute_max_drawdown(rets)
            metrics["total_return"] = float(np.sum(rets))
            metrics["win_rate"] = float(np.mean(np.array(rets) > 0))

        return metrics

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def _run_benchmarks(
        self,
        prices: pd.Series,
        log_returns: pd.Series,
        eval_index: pd.DatetimeIndex,
    ) -> Dict[str, pd.Series]:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from benchmark.strategies import BenchmarkStrategies  # noqa

        bench: Dict[str, pd.Series] = {}

        # Align to eval period
        pr = prices.reindex(eval_index).ffill()
        lr = log_returns.reindex(eval_index).fillna(0)

        # 1. Buy-and-Hold
        bench["buy_and_hold"] = lr

        # 2. Momentum
        bench["momentum"] = BenchmarkStrategies.momentum(pr)

        # 3. 60/40 – requires bond series; skip gracefully if unavailable
        # (would need cross['tnx'] prices, which are yield not price, so skip)

        # 4. Risk Parity (single-asset degenerate case = B&H)
        try:
            asset_df = pd.DataFrame({"spx": pr})
            bench["risk_parity"] = BenchmarkStrategies.risk_parity(asset_df)
        except Exception:
            pass

        # Clip all to eval index
        bench = {k: v.reindex(eval_index).fillna(0) for k, v in bench.items()}
        return bench


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SCAF End-to-End Experiment Pipeline")
    parser.add_argument("--ticker",       default="^GSPC")
    parser.add_argument("--start",        default="2015-01-01")
    parser.add_argument("--end",          default="2024-12-31")
    parser.add_argument("--horizon",      type=int, default=5)
    parser.add_argument("--train-window", type=int, default=500)
    parser.add_argument("--val-window",   type=int, default=63)
    parser.add_argument("--step",         type=int, default=21)
    parser.add_argument("--no-llm",       action="store_true")
    parser.add_argument("--no-risk",      action="store_true")
    parser.add_argument("--output-dir",   default="results")
    parser.add_argument("--name",         default="scaf_experiment")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        forecast_horizon=args.horizon,
        train_window=args.train_window,
        val_window=args.val_window,
        step_size=args.step,
        use_llm=not args.no_llm,
        use_risk_agents=not args.no_risk,
        output_dir=args.output_dir,
        experiment_name=args.name,
    )

    pipeline = ExperimentPipeline(cfg)
    results = pipeline.run()
    return results


if __name__ == "__main__":
    main()
