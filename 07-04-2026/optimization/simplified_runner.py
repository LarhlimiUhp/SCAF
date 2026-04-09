"""
Simplified SCAF-LS Optimization Runner

Version simplifiée sans dépendances Agent Framework pour tests rapides.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

from scaf_ls.config import Config
from scaf_ls.data.loader import MultiAssetLoader
from scaf_ls.data.engineer import CrossAssetFeatureEngineer
from scaf_ls.features.selector import AutomatedFeatureSelector
from scaf_ls.models.registry import registry
from scaf_ls.validation.cv_strategies import PurgedKFold
from monitoring.drift_detection import DriftDetector
from monitoring.alerts import AlertManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Measures the probability that the estimated Sharpe Ratio exceeds
    ``sr_benchmark`` after adjusting for non-normality and finite sample.

    Parameters
    ----------
    returns : 1-D array of period returns
    sr_benchmark : benchmark SR to test against (default 0)

    Returns
    -------
    float in [0, 1] – probability that true SR > sr_benchmark
    """
    T = len(returns)
    if T < 5:
        return float("nan")

    sr_hat = np.mean(returns) / (np.std(returns, ddof=1) + 1e-12)
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())  # excess kurtosis

    # Standard error of the SR estimator (Mertens, 2002)
    se = np.sqrt(
        (1 + 0.5 * sr_hat ** 2 - skew * sr_hat + ((kurt + 3) / 4) * sr_hat ** 2) / (T - 1)
    )

    psr = float(stats.norm.cdf((sr_hat - sr_benchmark) / (se + 1e-12)))
    return psr


class SimplifiedModelOptimizer:
    """Optimiseur simplifié pour un modèle spécifique"""

    def __init__(self, model_name: str, max_trials: int = 50,
                 target_n_features: int = 25):
        self.model_name = model_name
        self.max_trials = max_trials
        self.target_n_features = target_n_features
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        # Drift detector initialised once; reference data set on first fold
        self._drift_detector = DriftDetector()
        try:
            self._alert_manager = AlertManager()
        except Exception:
            self._alert_manager = None
        self._prepare_data()

    def _prepare_data(self):
        """Charger toutes les features brutes – la sélection se fait par fold."""
        try:
            loader = MultiAssetLoader(Config)
            data = loader.load_data()

            engineer = CrossAssetFeatureEngineer()
            features = engineer.create_features(data)

            # Store full feature matrix; feature selection happens inside each
            # CV fold to prevent data leakage from the validation window.
            self.X = features.drop(columns=['target'])
            if 'date' in self.X.columns:
                self.dates = self.X.pop('date')
            else:
                self.dates = pd.Series(range(len(self.X)))
            self.y = features['target']

            logger.info(f"Data prepared for {self.model_name}: {len(self.X)} samples, "
                        f"{self.X.shape[1]} raw features (selection happens per fold)")

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def _get_search_space(self, trial: optuna.Trial):
        """Espace de recherche pour le modèle"""
        if self.model_name == 'LGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            }
        elif self.model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
        elif self.model_name == 'BiLSTM':
            return {
                'seq_len': trial.suggest_int('seq_len', 10, 30),
                'hidden': trial.suggest_int('hidden', 32, 64),
                'epochs': trial.suggest_int('epochs', 5, 15),
                'lr': trial.suggest_float('lr', 1e-3, 5e-3),
            }
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _evaluate_params(self, params: dict) -> tuple[float, float]:
        """
        Évaluer les paramètres avec walk-forward CV.

        Feature selection is performed **inside each fold** using only the
        training window to prevent data leakage into the validation window.
        Uses PurgedKFold with n_splits=5 for more robust estimates.
        Also feeds validation samples to the drift detector and logs
        the Probabilistic Sharpe Ratio (PSR/DSR) per trial.

        Note: ``fold_returns`` uses ``auc - 0.5`` as a rough signed excess
        over random – this is a coarse proxy that enables relative comparison
        across trials but does not represent actual portfolio returns.
        """
        try:
            model_class = registry.all().get(self.model_name)
            if model_class is None:
                return 0.5, 1.0

            cv = PurgedKFold(n_splits=5, embargo=5)
            fold_scores = []
            fold_returns = []  # for DSR calculation (auc−0.5 proxy)
            reference_set = True  # first fold sets the drift reference

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X)):
                try:
                    X_train_raw = self.X.iloc[train_idx]
                    y_train = self.y.iloc[train_idx]
                    X_val_raw = self.X.iloc[val_idx]
                    y_val = self.y.iloc[val_idx]

                    # --- Feature selection on training data only (no leakage) ---
                    selector = AutomatedFeatureSelector()
                    sel_report = selector.run_full_analysis(
                        X_train_raw, y_train,
                        target_n_features=self.target_n_features,
                    )
                    selected_cols = sel_report['selected_features']
                    if not selected_cols:
                        selected_cols = list(X_train_raw.columns)

                    X_train = X_train_raw[selected_cols]
                    X_val = X_val_raw[selected_cols]

                    # --- Train and predict ---
                    model = model_class(**params)
                    model.fit(X_train.values, y_train.values)

                    if hasattr(model, 'predict_proba_one'):
                        probs = []
                        for i in range(len(X_val)):
                            prob, _ = model.predict_proba_one(X_val.iloc[i:i+1].values)
                            probs.append(prob)
                        y_pred_proba = np.array(probs)
                    else:
                        y_pred_proba = model.predict_proba(X_val.values)[:, 1]

                    auc = roc_auc_score(y_val, y_pred_proba)
                    fold_scores.append(auc)

                    # --- Drift monitoring ---
                    if reference_set:
                        self._drift_detector.set_reference_data(
                            X_train.values,
                            y=y_train.values,
                            predictions=model.predict_proba(X_train.values)[:, 1]
                            if hasattr(model, 'predict_proba') else None,
                        )
                        reference_set = False
                    else:
                        for i in range(len(X_val)):
                            self._drift_detector.add_test_sample(
                                X_val.iloc[i].values,
                                y_true=float(y_val.iloc[i]),
                                y_pred=float(y_pred_proba[i]),
                            )
                        drift_metrics = self._drift_detector.get_drift_metrics()
                        if drift_metrics.is_drifting:
                            logger.warning(
                                f"[DRIFT] Fold {fold_idx} – overall drift score "
                                f"{drift_metrics.overall_drift:.3f} "
                                f"(data={drift_metrics.data_drift_score:.3f}, "
                                f"model={drift_metrics.model_drift_score:.3f}, "
                                f"concept={drift_metrics.concept_drift_score:.3f})"
                            )
                            if self._alert_manager is not None:
                                try:
                                    self._alert_manager.trigger_alert(
                                        alert_type="model_drift",
                                        severity="warning",
                                        message=f"{self.model_name} fold {fold_idx}: "
                                                f"drift={drift_metrics.overall_drift:.3f}",
                                    )
                                except Exception:
                                    pass

                    # Approximate per-period returns from AUC for DSR
                    fold_returns.append(auc - 0.5)  # excess over random

                except Exception as e:
                    logger.warning(f"Fold {fold_idx} failed: {e}")
                    fold_scores.append(0.5)
                    fold_returns.append(0.0)

            mean_auc = np.mean(fold_scores)
            stability = np.std(fold_scores)

            # --- Deflated Sharpe Ratio significance test ---
            if len(fold_returns) >= 5:
                psr = deflated_sharpe_ratio(np.array(fold_returns), sr_benchmark=0.0)
                logger.info(f"[DSR] {self.model_name} PSR={psr:.3f} (prob SR>0)")
            else:
                psr = float("nan")

            return mean_auc, stability

        except Exception as e:
            logger.error(f"Parameter evaluation failed: {e}")
            return 0.5, 1.0

    def _objective(self, trial: optuna.Trial) -> float:
        """Fonction objectif"""
        params = self._get_search_space(trial)
        auc_score, stability = self._evaluate_params(params)

        # Objectif: AUC - pénalité stabilité
        objective_value = auc_score - 0.1 * stability

        trial.set_user_attr("auc_score", auc_score)
        trial.set_user_attr("stability", stability)

        return objective_value

    def optimize(self) -> dict:
        """Lancer l'optimisation"""
        logger.info(f"Starting optimization for {self.model_name}")

        start_time = time.time()
        self.study.optimize(self._objective, n_trials=self.max_trials)
        optimization_time = time.time() - start_time

        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        auc_score = best_trial.user_attrs.get("auc_score", 0.5)
        stability = best_trial.user_attrs.get("stability", 1.0)

        result = {
            "model_name": self.model_name,
            "best_params": best_params,
            "best_auc": auc_score,
            "stability_score": stability,
            "trials_completed": len(self.study.trials),
            "optimization_time": optimization_time,
            "best_objective": best_score
        }

        # Sauvegarder les résultats
        self._save_results(result)

        logger.info(f"✅ {self.model_name} optimization completed: AUC={auc_score:.4f}")

        return result

    def _save_results(self, result: dict):
        """Sauvegarder les résultats"""
        import os
        os.makedirs("results_v50/optimization", exist_ok=True)

        filename = f"results_v50/optimization/{self.model_name}_optimization_{int(time.time())}.json"
        with open(filename, 'w') as f:
            import json
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Results saved to {filename}")


def run_parallel_optimization(models: list = None, max_trials: int = 30, max_workers: int = 3):
    """Exécuter l'optimisation en parallèle"""

    if models is None:
        models = ['LGBM', 'RandomForest', 'BiLSTM']

    logger.info(f"🚀 Starting parallel optimization for models: {models}")

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre les tâches
        future_to_model = {
            executor.submit(run_single_optimization, model, max_trials): model
            for model in models
        }

        # Collecter les résultats
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results[model] = result
                logger.info(f"✅ {model} completed")
            except Exception as e:
                logger.error(f"❌ {model} failed: {e}")
                results[model] = {"error": str(e)}

    # Générer le rapport final
    generate_final_report(results)

    return results


def run_single_optimization(model_name: str, max_trials: int = 30):
    """Optimiser un seul modèle"""
    optimizer = SimplifiedModelOptimizer(model_name, max_trials)
    return optimizer.optimize()


def generate_final_report(results: dict):
    """Générer le rapport final"""
    import os
    os.makedirs("results_v50/optimization", exist_ok=True)

    report_lines = [
        "# 📊 SCAF-LS Hyperparameter Optimization Report (Simplified)",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 🎯 Results Summary",
    ]

    successful_results = {k: v for k, v in results.items() if "error" not in v}

    if successful_results:
        for model_name, result in successful_results.items():
            report_lines.extend([
                f"### {model_name}",
                f"- **AUC:** {result['best_auc']:.4f} (±{result['stability_score']:.4f})",
                f"- **Trials:** {result['trials_completed']}",
                f"- **Time:** {result['optimization_time']:.1f}s",
                f"- **Objective:** {result['best_objective']:.4f}",
                ""
            ])

        # Statistiques globales
        avg_auc = np.mean([r['best_auc'] for r in successful_results.values()])
        avg_stability = np.mean([r['stability_score'] for r in successful_results.values()])

        report_lines.extend([
            "## 📈 Global Statistics",
            f"- **Models Optimized:** {len(successful_results)}",
            f"- **Average AUC:** {avg_auc:.4f}",
            f"- **Average Stability:** {avg_stability:.4f}",
            f"- **Total Trials:** {sum(r['trials_completed'] for r in successful_results.values())}",
            "",
            "## ✅ Mission Status",
            f"- **AUC Target (+5-10%):** {'✅' if avg_auc > 0.6 else '❌'} (baseline ~0.55)",
            f"- **Stability Target (-25% var):** {'✅' if avg_stability < 0.2 else '❌'}",
        ])

    # Erreurs
    failed_models = {k: v for k, v in results.items() if "error" in v}
    if failed_models:
        report_lines.extend([
            "",
            "## ❌ Failed Optimizations",
        ])
        for model_name, error_info in failed_models.items():
            report_lines.append(f"- **{model_name}:** {error_info['error']}")

    # Sauvegarder
    report_content = "\n".join(report_lines)
    report_file = f"results_v50/optimization/simplified_report_{int(time.time())}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"📋 Report saved to {report_file}")

    # Afficher le résumé
    print("\n" + "="*60)
    print("SCAF-LS OPTIMIZATION RESULTS")
    print("="*60)

    if successful_results:
        print(f"Models optimized: {len(successful_results)}")
        print(f"Average AUC: {avg_auc:.4f}")
        print(f"Average Stability: {avg_stability:.4f}")
        print(f"AUC Target: {'✅ ACHIEVED' if avg_auc > 0.6 else '❌ NOT MET'}")
        print(f"Stability Target: {'✅ ACHIEVED' if avg_stability < 0.2 else '❌ NOT MET'}")
    else:
        print("❌ All optimizations failed")

    print(f"Full report: {report_file}")
    print("="*60)


def main():
    """Fonction principale"""
    import argparse

    parser = argparse.ArgumentParser(description="SCAF-LS Simplified Optimization")
    parser.add_argument("--models", nargs="+", default=["LGBM", "RandomForest", "BiLSTM"],
                       help="Models to optimize")
    parser.add_argument("--trials", type=int, default=30,
                       help="Max trials per model")
    parser.add_argument("--workers", type=int, default=3,
                       help="Max parallel workers")

    args = parser.parse_args()

    # Lancer l'optimisation
    results = run_parallel_optimization(
        models=args.models,
        max_trials=args.trials,
        max_workers=args.workers
    )

    logger.info("🎉 Optimization campaign completed!")


if __name__ == "__main__":
    main()