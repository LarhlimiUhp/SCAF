"""
SCAF-LS — Backtest historique complet
======================================
Lance le pipeline walk-forward sur données réelles (yfinance).
Produit un rapport JSON + graphiques dans results/backtest_<date>/

Usage
-----
    cd 07-04-2026
    python run_backtest.py

Dépendances
-----------
    pip install yfinance lightgbm scikit-learn torch optuna pandas numpy
    pip install scipy statsmodels matplotlib seaborn
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # pas de fenêtre — sauvegarde fichier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ─── Ajoute le dossier courant au path pour les imports SCAF ─────────────── #
sys.path.insert(0, str(Path(__file__).parent))

# ─── Logging ─────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("backtest")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — aligner avec les paramètres optimaux ULTRA-THINK
# ══════════════════════════════════════════════════════════════════════════════

RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("results") / f"backtest_{RUN_ID}"

# Période historique : 10 ans hors-sample
START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
TICKER      = "^GSPC"

# Paramètres alignés sur les optimums ULTRA-THINK
TRANSACTION_COST = 0.0002      # TRADING_FEE_PCT
HORIZON          = 5           # RETURN_HORIZON
EMBARGO_DAYS     = 10          # PURGE_GAP_DAYS
N_FOLDS          = 5           # robustesse accrue vs 3

# Modèles actifs (désactiver TabNet/GraphNN/CatBoost si non installés)
MODELS = [
    "LogReg-L2", "RandomForest", "LGBM", "KNN",
    "BiLSTM", "XGBoost", "ExtraTrees", "HistGBT", "MLP",
    "RidgeClass", "Bagging",
]

# Désactiver LLM pour un run reproductible et rapide
USE_LLM            = False
USE_QLEARNING      = True
USE_CONFORMAL      = True
USE_DL_REGIME      = True


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(returns: np.ndarray, periods: int = 252) -> float:
    m = float(np.mean(returns))
    s = float(np.std(returns)) + 1e-12
    return m / s * np.sqrt(periods)


def _max_drawdown(returns: np.ndarray) -> float:
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / (peak + 1e-12)
    return float(np.min(dd))


def _calmar(returns: np.ndarray, periods: int = 252) -> float:
    ann_ret = float(np.mean(returns)) * periods
    mdd = abs(_max_drawdown(returns)) + 1e-12
    return ann_ret / mdd


def _sortino(returns: np.ndarray, periods: int = 252, target: float = 0.0) -> float:
    excess = returns - target / periods
    downside = np.std(excess[excess < 0]) + 1e-12
    return float(np.mean(excess)) / downside * np.sqrt(periods)


def _win_rate(returns: np.ndarray) -> float:
    return float(np.mean(returns > 0))


def _print_summary(summary: dict) -> None:
    """Affiche le résumé dans la console."""
    sep = "-" * 56
    print(f"\n{sep}")
    print("  SCAF-LS  --  Resultats Backtest Historique")
    print(sep)

    metrics = [
        ("Sharpe Ratio",          "sharpe_ratio",              ">1.033"),
        ("Max Drawdown",          "max_drawdown",              ">-0.15"),
        ("Cumulative Return",     "cumulative_return",         ""),
        ("Excess vs B&H",         "excess_return_vs_bnh",      ">0"),
        ("Conformal Coverage",    "conformal_coverage_rate",   "~0.95"),
        ("Brier RMSE",            "rmse_brier",                "<0.50"),
        ("Latency moy. (ms)",     "mean_inference_latency_ms", ""),
        ("Folds complétés",       "n_folds_completed",         ""),
        ("Échantillons totaux",   "n_samples_total",           ""),
    ]

    for label, key, target in metrics:
        val = summary.get(key)
        if val is None:
            continue
        target_str = f"  [cible {target}]" if target else ""
        print(f"  {label:<30} {val!s:<12}{target_str}")

    print(sep)

    # Test statistique
    st = summary.get("stat_test_vs_bnh", {})
    if st:
        outperform = st.get("outperform", False)
        pval       = st.get("p_value", 1.0)
        print(f"\n  Test t vs B&H  ->  p={pval:.4f}  "
              f"{'[OK] Surperformance significative (p<0.001)' if outperform else '[--] Non significatif'}")

    print(sep)

    # Benchmarks
    baselines = summary.get("baseline_sharpe_ratios", {})
    if baselines:
        print("\n  Sharpe des 15 benchmarks :")
        for name, sr in sorted(baselines.items(), key=lambda x: -x[1]):
            flag = " ←" if sr > summary.get("sharpe_ratio", 0) else ""
            print(f"    {name:<30} {sr:.4f}{flag}")

    # AUC par modèle
    aucs = summary.get("mean_model_auc", {})
    if aucs:
        print("\n  AUC moyenne par modèle :")
        for name, auc in sorted(aucs.items(), key=lambda x: -x[1]):
            flag = "  [sous-random]" if auc < 0.50 else ""
            print(f"    {name:<20} {auc:.4f}{flag}")

    print(f"\n  Résultats sauvegardés dans : {RESULTS_DIR}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(summary: dict, cv_results: list, prices: pd.Series) -> None:
    """Génère les graphiques du backtest et les sauvegarde."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Courbe de portefeuille vs B&H ─────────────────────────────── #
    all_returns = np.concatenate([
        r["portfolio_returns"] for r in cv_results if "error" not in r
    ])
    cum_strat = np.exp(np.cumsum(all_returns))

    bnh_log = np.log(prices / prices.shift(1)).dropna().values
    bnh_aligned = bnh_log[-len(all_returns):]
    cum_bnh = np.exp(np.cumsum(bnh_aligned))

    fig, axes = plt.subplots(3, 1, figsize=(13, 13),
                             gridspec_kw={"hspace": 0.40})

    # Courbe d'équité
    ax = axes[0]
    ax.plot(cum_strat, label="SCAF-LS", color="#2563eb", linewidth=1.8)
    ax.plot(cum_bnh,   label="Buy & Hold", color="#9ca3af",
            linewidth=1.2, linestyle="--")
    ax.set_title("Courbe d'équité — SCAF-LS vs Buy & Hold",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Valeur du portefeuille (base 1)")
    ax.legend(); ax.grid(alpha=0.3)

    # Drawdown
    ax = axes[1]
    peak = np.maximum.accumulate(cum_strat)
    drawdown = (cum_strat - peak) / (peak + 1e-12)
    ax.fill_between(range(len(drawdown)), drawdown, 0,
                    color="#ef4444", alpha=0.55)
    ax.axhline(-0.10, color="#f97316", linewidth=1.2,
               linestyle="--", label="Limite 10%")
    ax.axhline(-0.15, color="#dc2626", linewidth=1.2,
               linestyle="--", label="Cible 15%")
    ax.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Distribution des retours quotidiens
    ax = axes[2]
    ax.hist(all_returns, bins=60, color="#2563eb", alpha=0.75,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#111827", linewidth=1.0)
    mean_r = np.mean(all_returns)
    ax.axvline(mean_r, color="#16a34a", linewidth=1.2, linestyle="--",
               label=f"Moyenne {mean_r:.5f}")
    ax.set_title("Distribution des retours quotidiens", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Retour log"); ax.set_ylabel("Fréquence")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    path1 = RESULTS_DIR / "equity_drawdown.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Graphique sauvegardé : %s", path1)

    # ── 2. Sharpe par fold ────────────────────────────────────────────── #
    valid = [r for r in cv_results if "error" not in r]
    folds  = [r["fold"] + 1 for r in valid]
    sharpes = [r["sharpe"]   for r in valid]
    dds     = [r["max_drawdown"] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#16a34a" if s > 1.033 else "#f97316" if s > 0 else "#ef4444"
              for s in sharpes]
    ax1.bar(folds, sharpes, color=colors, edgecolor="white")
    ax1.axhline(1.033, color="#2563eb", linewidth=1.5, linestyle="--",
                label="Cible 1.033")
    ax1.axhline(0.0,   color="#111827", linewidth=0.8)
    ax1.set_title("Sharpe Ratio par fold", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fold"); ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, axis="y")

    dd_colors = ["#16a34a" if d > -0.10 else "#f97316" if d > -0.15 else "#ef4444"
                 for d in dds]
    ax2.bar(folds, dds, color=dd_colors, edgecolor="white")
    ax2.axhline(-0.10, color="#f97316", linewidth=1.5, linestyle="--",
                label="Limite 10%")
    ax2.axhline(-0.15, color="#dc2626", linewidth=1.5, linestyle="--",
                label="Cible 15%")
    ax2.set_title("Max Drawdown par fold", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fold"); ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    path2 = RESULTS_DIR / "fold_metrics.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Graphique sauvegardé : %s", path2)

    # ── 3. Comparaison benchmarks ─────────────────────────────────────── #
    baselines = summary.get("baseline_sharpe_ratios", {})
    if baselines:
        scaf_sr = summary.get("sharpe_ratio", 0)
        names   = list(baselines.keys()) + ["SCAF-LS"]
        values  = list(baselines.values()) + [scaf_sr]
        colors_b = ["#2563eb" if n == "SCAF-LS" else
                    "#16a34a" if v > scaf_sr else "#9ca3af"
                    for n, v in zip(names, values)]

        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = range(len(names))
        ax.barh(y_pos, values, color=colors_b, edgecolor="white")
        ax.set_yticks(list(y_pos)); ax.set_yticklabels(names, fontsize=9)
        ax.axvline(1.033, color="#f97316", linewidth=1.5,
                   linestyle="--", label="Cible 1.033")
        ax.axvline(0, color="#111827", linewidth=0.8)
        ax.set_title("Sharpe Ratio — SCAF-LS vs 15 Benchmarks",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Sharpe Ratio (annualisé)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")

        path3 = RESULTS_DIR / "benchmark_comparison.png"
        fig.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Graphique sauvegardé : %s", path3)

    # ── 4. AUC par modèle ─────────────────────────────────────────────── #
    aucs = summary.get("mean_model_auc", {})
    if aucs:
        sorted_aucs = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
        m_names = [x[0] for x in sorted_aucs]
        m_vals  = [x[1] for x in sorted_aucs]
        colors_m = ["#16a34a" if v >= 0.52 else
                    "#f97316" if v >= 0.50 else "#ef4444"
                    for v in m_vals]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(m_names, m_vals, color=colors_m, edgecolor="white")
        ax.axhline(0.50, color="#dc2626", linewidth=1.5,
                   linestyle="--", label="AUC plancher (0.50)")
        ax.axhline(0.52, color="#f97316", linewidth=1.2,
                   linestyle="--", label="Seuil acceptable (0.52)")
        ax.set_title("AUC moyenne par modèle (walk-forward)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("AUC ROC"); ax.set_xlabel("Modèle")
        plt.xticks(rotation=35, ha="right", fontsize=9)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

        path4 = RESULTS_DIR / "model_auc.png"
        fig.savefig(path4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Graphique sauvegardé : %s", path4)


# ══════════════════════════════════════════════════════════════════════════════
#  RAPPORT JSON
# ══════════════════════════════════════════════════════════════════════════════

def _save_report(summary: dict, cv_results: list, elapsed: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Métriques supplémentaires calculées ici
    all_returns = np.concatenate([
        r["portfolio_returns"] for r in cv_results if "error" not in r
    ])

    extra = {
        "run_id":            RUN_ID,
        "period":            f"{START_DATE} → {END_DATE}",
        "ticker":            TICKER,
        "n_folds":           N_FOLDS,
        "horizon_days":      HORIZON,
        "transaction_cost":  TRANSACTION_COST,
        "calmar_ratio":      round(_calmar(all_returns), 4),
        "sortino_ratio":     round(_sortino(all_returns), 4),
        "win_rate":          round(_win_rate(all_returns), 4),
        "ann_return":        round(float(np.mean(all_returns)) * 252, 4),
        "ann_volatility":    round(float(np.std(all_returns)) * np.sqrt(252), 4),
        "elapsed_seconds":   round(elapsed, 1),
        "targets": {
            "sharpe_ok":   summary.get("sharpe_ratio", 0) > 1.033,
            "drawdown_ok": summary.get("max_drawdown", -1) > -0.15,
            "excess_ok":   summary.get("excess_return_vs_bnh", -1) > 0,
        },
    }

    report = {**summary, **extra}

    # Fold details (sans les séries de signaux pour alléger)
    fold_summary = []
    for r in cv_results:
        if "error" in r:
            fold_summary.append(r)
            continue
        fold_summary.append({
            "fold":           r["fold"],
            "n_train":        r["n_train"],
            "n_val":          r["n_val"],
            "sharpe":         round(r["sharpe"], 4),
            "max_drawdown":   round(r["max_drawdown"], 4),
            "cum_return":     round(r["cum_return"], 6),
            "rmse":           round(r.get("rmse", float("nan")), 6),
            "coverage_rate":  r.get("coverage_rate"),
            "active_models":  r.get("active_models", []),
            "model_auc":      {k: round(v, 4) for k, v in r.get("model_auc", {}).items()},
        })

    output = {"report": report, "folds": fold_summary}

    path = RESULTS_DIR / "backtest_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Rapport JSON sauvegardé : %s", path)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 56)
    log.info("  SCAF-LS — Backtest historique  [%s]", RUN_ID)
    log.info("  Période : %s → %s", START_DATE, END_DATE)
    log.info("  Ticker  : %s   Folds : %d   Horizon : %d j", TICKER, N_FOLDS, HORIZON)
    log.info("=" * 56)

    # ── Vérification dépendances ──────────────────────────────────────── #
    missing = []
    for pkg in ["yfinance", "lightgbm", "sklearn", "scipy", "statsmodels"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Dépendances manquantes : %s", missing)
        log.error("Installer : pip install %s", " ".join(missing))
        sys.exit(1)

    # ── Configuration pipeline ────────────────────────────────────────── #
    from pipeline.experiment import ExperimentConfig, ExperimentPipeline

    config = ExperimentConfig(
        ticker             = TICKER,
        start_date         = START_DATE,
        end_date           = END_DATE,
        horizon            = HORIZON,
        model_names        = MODELS,
        n_folds            = N_FOLDS,
        min_train_samples  = 300,
        embargo_days       = EMBARGO_DAYS,
        transaction_cost   = TRANSACTION_COST,
        position_scalar    = 1.0,
        use_conformal      = USE_CONFORMAL,
        conformal_alpha    = 0.05,
        calibration_fraction = 0.20,
        use_qlearning      = USE_QLEARNING,
        use_dl_regime_detector = USE_DL_REGIME,
        use_llm            = USE_LLM,
        results_dir        = str(RESULTS_DIR),
    )

    # ── Lancement ────────────────────────────────────────────────────────
    pipeline = ExperimentPipeline(config)

    log.info("Lancement du pipeline...")
    results = pipeline.run()

    if "error" in results:
        log.error("Pipeline échoué : %s", results["error"])
        sys.exit(1)

    summary    = results["summary"]
    cv_results = results["cv_results"]

    elapsed = time.time() - t_start

    # ── Métriques additionnelles ─────────────────────────────────────── #
    all_returns = np.concatenate([
        r["portfolio_returns"] for r in cv_results if "error" not in r
    ])
    summary["calmar_ratio"]   = round(_calmar(all_returns), 4)
    summary["sortino_ratio"]  = round(_sortino(all_returns), 4)
    summary["win_rate"]       = round(_win_rate(all_returns), 4)
    summary["ann_return"]     = round(float(np.mean(all_returns)) * 252, 4)
    summary["ann_volatility"] = round(float(np.std(all_returns)) * np.sqrt(252), 4)

    # ── Affichage console ─────────────────────────────────────────────── #
    _print_summary(summary)

    # ── Récupère les prix pour les graphiques ─────────────────────────── #
    try:
        import yfinance as yf
        raw = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        prices = raw["Close"].dropna()
    except Exception:
        # Fallback : prix synthétiques si yfinance KO ici
        n = len(all_returns)
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))))

    # ── Graphiques ───────────────────────────────────────────────────── #
    log.info("Génération des graphiques...")
    try:
        _plot_results(summary, cv_results, prices)
    except Exception as exc:
        log.warning("Graphiques échoués (non bloquant) : %s", exc)

    # ── Rapport JSON ─────────────────────────────────────────────────── #
    _save_report(summary, cv_results, elapsed)

    # ── Verdict final ─────────────────────────────────────────────────── #
    sr  = summary.get("sharpe_ratio", 0)
    mdd = summary.get("max_drawdown", -1)
    exc = summary.get("excess_return_vs_bnh", -1)

    log.info("-" * 56)
    log.info("VERDICT  Sharpe=%s %s   DD=%s %s   Exces=%s %s",
             f"{sr:.4f}", "[OK]" if sr > 1.033 else "[--]",
             f"{mdd:.4f}", "[OK]" if mdd > -0.15 else "[--]",
             f"{exc:.4f}", "[OK]" if exc > 0 else "[--]",
    )
    log.info("Duree totale : %.1f s", elapsed)
    log.info("-" * 56)


if __name__ == "__main__":
    main()
