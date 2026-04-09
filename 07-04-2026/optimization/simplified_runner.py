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
from sklearn.metrics import roc_auc_score

from scaf_ls.config import Config
from scaf_ls.data.loader import MultiAssetLoader
from scaf_ls.data.engineer import CrossAssetFeatureEngineer
from scaf_ls.features.selector import AutomatedFeatureSelector
from scaf_ls.models.registry import registry
from scaf_ls.validation.cv_strategies import PurgedKFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedModelOptimizer:
    """Optimiseur simplifié pour un modèle spécifique"""

    def __init__(self, model_name: str, max_trials: int = 50):
        self.model_name = model_name
        self.max_trials = max_trials
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        self._prepare_data()

    def _prepare_data(self):
        """Préparer les données"""
        try:
            loader = MultiAssetLoader()
            data = loader.load_data()

            engineer = CrossAssetFeatureEngineer()
            features = engineer.create_features(data)

            selector = AutomatedFeatureSelector()
            selected_features = selector.run_full_analysis(features.drop(columns=['target']), features['target'])['selected_features']
            # Re-add target column
            selected_features.append('target')
            selected_features = features[selected_features]

            self.X = selected_features.drop(columns=['target', 'date'])
            self.y = selected_features['target']
            self.dates = selected_features['date']

            logger.info(f"Data prepared for {self.model_name}: {len(self.X)} samples")

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
        """Évaluer les paramètres avec CV"""
        try:
            model_class = get_model(self.model_name)
            if model_class is None:
                return 0.5, 1.0

            cv = PurgedKFold(n_splits=3, embargo=2)
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X)):
                try:
                    X_train = self.X.iloc[train_idx]
                    y_train = self.y.iloc[train_idx]
                    X_val = self.X.iloc[val_idx]
                    y_val = self.y.iloc[val_idx]

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

                except Exception as e:
                    logger.warning(f"Fold {fold_idx} failed: {e}")
                    fold_scores.append(0.5)

            mean_auc = np.mean(fold_scores)
            stability = np.std(fold_scores)

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
        print(".4f")
        print(".4f")
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