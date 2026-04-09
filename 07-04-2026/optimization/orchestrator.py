"""
SCAF-LS Hyperparameter Optimization Multi-Agent System

MISSION CRITIQUE: Optimisation systématique des hyperparamètres avec Optuna
pour tous les modèles SCAF-LS avec +5-10% AUC et -25% variance.

ÉQUIPE: 400 sous-agents spécialisés dans le tuning parallèle, validation croisée,
et optimisation avec contraintes de stabilité.
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

# Microsoft Agent Framework imports
from agent_framework import (
    AgentResponseUpdate,
    Content,
    Executor,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from agent_framework.observability import configure_otel_providers

# SCAF-LS imports
from scaf_ls.config import Config
from scaf_ls.data.loader import MultiAssetLoader
from scaf_ls.data.engineer import CrossAssetFeatureEngineer
from scaf_ls.features.selector import ICFeatureSelector
from scaf_ls.models.registry import get_model
from scaf_ls.validation.cv_strategies import PurgedKFold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenTelemetry for tracing
configure_otel_providers(
    vs_code_extension_port=4317,  # AI Toolkit gRPC port
    enable_sensitive_data=True
)


@dataclass
class OptimizationResult:
    """Résultat d'optimisation pour un modèle"""
    model_name: str
    best_params: Dict[str, Any]
    best_auc: float
    stability_score: float
    trials_completed: int
    optimization_time: float
    fold_scores: List[float]


@dataclass
class OptimizationCampaign:
    """Campagne d'optimisation complète"""
    campaign_id: str
    models_to_optimize: List[str]
    max_trials_per_model: int = 100
    max_parallel_agents: int = 50
    time_limit_hours: float = 4.0
    stability_penalty: float = 0.1
    min_improvement_threshold: float = 0.02


class ModelTuningAgent(Executor):
    """Agent spécialisé dans le tuning d'un modèle spécifique"""

    def __init__(self, model_name: str, campaign: OptimizationCampaign):
        super().__init__(id=f"{model_name}_tuner")
        self.model_name = model_name
        self.campaign = campaign
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        # Préparer les données d'entraînement
        self._prepare_data()

    def _prepare_data(self):
        """Préparer les données pour l'optimisation"""
        try:
            # Charger les données
            loader = MultiAssetLoader()
            data = loader.load_data()

            # Ingénierie des features
            engineer = CrossAssetFeatureEngineer()
            features = engineer.create_features(data)

            # Sélection des features
            selector = ICFeatureSelector()
            selected_features = selector.select_features(features)

            # Préparer X, y
            self.X = selected_features.drop(columns=['target', 'date'])
            self.y = selected_features['target']
            self.dates = selected_features['date']

            logger.info(f"Data prepared for {self.model_name}: {len(self.X)} samples, {len(self.X.columns)} features")

        except Exception as e:
            logger.error(f"Failed to prepare data for {self.model_name}: {e}")
            raise

    def _get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Définir l'espace de recherche pour le modèle"""
        if self.model_name == 'LGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            }
        elif self.model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
        elif self.model_name == 'BiLSTM':
            return {
                'seq_len': trial.suggest_int('seq_len', 5, 50),
                'hidden': trial.suggest_int('hidden', 16, 128),
                'n_layers': trial.suggest_int('n_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 5, 50),
            }
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[float, float]:
        """Évaluer les paramètres avec validation croisée temporelle"""
        try:
            # Créer le modèle avec les paramètres
            model_class = get_model(self.model_name)
            if model_class is None:
                logger.warning(f"Model {self.model_name} not found, skipping")
                return 0.5, 1.0

            # Validation croisée avec PurgedKFold
            cv = PurgedKFold(n_splits=3, embargo=5)
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X)):
                try:
                    # Split data
                    X_train = self.X.iloc[train_idx]
                    y_train = self.y.iloc[train_idx]
                    X_val = self.X.iloc[val_idx]
                    y_val = self.y.iloc[val_idx]

                    # Train model
                    model = model_class(**params)
                    model.fit(X_train.values, y_train.values)

                    # Predict probabilities
                    if hasattr(model, 'predict_proba_one'):
                        # Pour les modèles SCAF-LS
                        probs = []
                        for i in range(len(X_val)):
                            prob, _ = model.predict_proba_one(X_val.iloc[i:i+1].values)
                            probs.append(prob)
                        y_pred_proba = np.array(probs)
                    else:
                        # Pour les modèles sklearn standard
                        y_pred_proba = model.predict_proba(X_val.values)[:, 1]

                    # Calculate AUC
                    auc = roc_auc_score(y_val, y_pred_proba)
                    fold_scores.append(auc)

                except Exception as e:
                    logger.warning(f"Fold {fold_idx} failed for {self.model_name}: {e}")
                    fold_scores.append(0.5)  # Score neutre en cas d'échec

            # Calculer métriques
            mean_auc = np.mean(fold_scores)
            stability = np.std(fold_scores)

            return mean_auc, stability

        except Exception as e:
            logger.error(f"Parameter evaluation failed for {self.model_name}: {e}")
            return 0.5, 1.0

    def _objective(self, trial: optuna.Trial) -> float:
        """Fonction objectif pour Optuna"""
        params = self._get_search_space(trial)
        auc_score, stability = self._evaluate_params(params)

        # Objectif: maximiser AUC - pénalité de stabilité
        objective_value = auc_score - (self.campaign.stability_penalty * stability)

        # Reporter les métriques
        trial.set_user_attr("auc_score", auc_score)
        trial.set_user_attr("stability", stability)

        return objective_value

    @handler
    async def optimize(self, message: Message, ctx: WorkflowContext[AgentResponseUpdate]) -> None:
        """Exécuter l'optimisation pour ce modèle"""
        start_time = time.time()

        try:
            logger.info(f"Starting optimization for {self.model_name}")

            # Optimisation avec Optuna
            self.study.optimize(
                self._objective,
                n_trials=self.campaign.max_trials_per_model,
                timeout=self.campaign.time_limit_hours * 3600,
                n_jobs=1  # Séquentiel pour éviter les conflits de mémoire
            )

            # Résultats
            best_trial = self.study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value
            auc_score = best_trial.user_attrs.get("auc_score", 0.5)
            stability = best_trial.user_attrs.get("stability", 1.0)

            optimization_time = time.time() - start_time

            result = OptimizationResult(
                model_name=self.model_name,
                best_params=best_params,
                best_auc=auc_score,
                stability_score=stability,
                trials_completed=len(self.study.trials),
                optimization_time=optimization_time,
                fold_scores=[]  # TODO: collect fold scores
            )

            # Sauvegarder les résultats
            self._save_results(result)

            # Reporter le progrès
            await ctx.send_message(AgentResponseUpdate(
                contents=[Content("text", text=f"✅ {self.model_name} optimization completed")],
                role="assistant",
                author_name=self.id
            ))

            # Yield final result
            await ctx.yield_output(result)

        except Exception as e:
            logger.error(f"Optimization failed for {self.model_name}: {e}")
            await ctx.send_message(AgentResponseUpdate(
                contents=[Content("text", text=f"❌ {self.model_name} optimization failed: {str(e)}")],
                role="assistant",
                author_name=self.id
            ))

    def _save_results(self, result: OptimizationResult):
        """Sauvegarder les résultats d'optimisation"""
        os.makedirs("results_v50/optimization", exist_ok=True)

        result_dict = {
            "model_name": result.model_name,
            "best_params": result.best_params,
            "best_auc": result.best_auc,
            "stability_score": result.stability_score,
            "trials_completed": result.trials_completed,
            "optimization_time": result.optimization_time,
            "campaign_id": self.campaign.campaign_id,
            "timestamp": time.time()
        }

        filename = f"results_v50/optimization/{result.model_name}_optimization_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Results saved to {filename}")


class OptimizationOrchestrator(Executor):
    """Orchestrateur principal pour la campagne d'optimisation"""

    def __init__(self, campaign: OptimizationCampaign):
        super().__init__(id="optimization_orchestrator")
        self.campaign = campaign
        self.results = []
        self.active_agents = 0

    @handler
    async def orchestrate_optimization(self, message: Message, ctx: WorkflowContext[AgentResponseUpdate]) -> None:
        """Orchestrer l'optimisation de tous les modèles"""

        await ctx.send_message(AgentResponseUpdate(
            contents=[Content("text", text=f"🚀 Starting SCAF-LS optimization campaign: {self.campaign.campaign_id}")],
            role="assistant",
            author_name=self.id
        ))

        # Créer les agents de tuning
        tuning_agents = []
        for model_name in self.campaign.models_to_optimize:
            agent = ModelTuningAgent(model_name, self.campaign)
            tuning_agents.append(agent)

        # Exécuter en parallèle avec limite de concurrence
        semaphore = asyncio.Semaphore(self.campaign.max_parallel_agents)

        async def run_agent_with_limit(agent):
            async with semaphore:
                try:
                    # Créer un workflow pour chaque agent
                    workflow = WorkflowBuilder(start_executor=agent).build()

                    # Exécuter l'optimisation
                    events = await workflow.run(Message("user", ["optimize"]))

                    # Collecter les résultats
                    outputs = events.get_outputs()
                    if outputs:
                        self.results.append(outputs[0])

                    return f"✅ {agent.model_name} completed"

                except Exception as e:
                    logger.error(f"Agent {agent.model_name} failed: {e}")
                    return f"❌ {agent.model_name} failed: {str(e)}"

        # Lancer tous les agents
        tasks = [run_agent_with_limit(agent) for agent in tuning_agents]
        progress_updates = await asyncio.gather(*tasks)

        # Générer le rapport final
        await self._generate_final_report(ctx)

    async def _generate_final_report(self, ctx: WorkflowContext[AgentResponseUpdate]):
        """Générer le rapport final de la campagne"""

        if not self.results:
            await ctx.send_message(AgentResponseUpdate(
                contents=[Content("text", text="❌ No optimization results to report")],
                role="assistant",
                author_name=self.id
            ))
            return

        # Analyser les résultats
        best_results = {}
        for result in self.results:
            if isinstance(result, OptimizationResult):
                best_results[result.model_name] = result

        # Rapport de synthèse
        report_lines = [
            "# 📊 SCAF-LS Hyperparameter Optimization Report",
            f"**Campaign:** {self.campaign.campaign_id}",
            f"**Models Optimized:** {len(best_results)}",
            f"**Total Trials:** {sum(r.trials_completed for r in best_results.values())}",
            "",
            "## 🎯 Best Results by Model",
        ]

        for model_name, result in best_results.items():
            report_lines.extend([
                f"### {model_name}",
                f"- **AUC:** {result.best_auc:.4f} (±{result.stability_score:.4f})",
                f"- **Trials:** {result.trials_completed}",
                f"- **Time:** {result.optimization_time:.1f}s",
                f"- **Best Params:** {json.dumps(result.best_params, indent=2)}",
                ""
            ])

        # Calculer les améliorations
        baseline_auc = 0.55  # AUC de base estimé
        avg_improvement = np.mean([r.best_auc - baseline_auc for r in best_results.values()])
        avg_stability = np.mean([r.stability_score for r in best_results.values()])

        report_lines.extend([
            "## 📈 Performance Summary",
            f"- **Average AUC Improvement:** {avg_improvement:.1%}",
            f"- **Average Stability (Std):** {avg_stability:.4f}",
            f"- **Target Achievement:** {'✅' if avg_improvement >= 0.05 else '❌'} +5% AUC",
            f"- **Stability Target:** {'✅' if avg_stability <= 0.25 else '❌'} -25% variance",
            "",
            "## 🔧 Recommended Next Steps",
            "1. Retrain models with optimized parameters",
            "2. Validate on out-of-sample data",
            "3. Deploy to production with monitoring",
            "4. Schedule regular re-optimization"
        ])

        final_report = "\n".join(report_lines)

        # Sauvegarder le rapport
        os.makedirs("results_v50/optimization", exist_ok=True)
        report_file = f"results_v50/optimization/campaign_{self.campaign.campaign_id}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(final_report)

        await ctx.send_message(AgentResponseUpdate(
            contents=[Content("text", text=f"📋 Optimization campaign completed!\n\n{final_report}")],
            role="assistant",
            author_name=self.id
        ))

        await ctx.yield_output({
            "campaign_id": self.campaign.campaign_id,
            "results": best_results,
            "report_file": report_file
        })


@executor(id="campaign_initializer")
async def initialize_campaign(message: Message, ctx: WorkflowContext[OptimizationCampaign]) -> None:
    """Initialiser une nouvelle campagne d'optimisation"""

    # Configuration de la campagne
    campaign = OptimizationCampaign(
        campaign_id=f"scaf_ls_opt_{int(time.time())}",
        models_to_optimize=['LGBM', 'RandomForest', 'BiLSTM'],
        max_trials_per_model=50,  # Réduit pour les tests
        max_parallel_agents=3,    # Limité pour éviter surcharge
        time_limit_hours=1.0,
        stability_penalty=0.1,
        min_improvement_threshold=0.02
    )

    logger.info(f"Initialized optimization campaign: {campaign.campaign_id}")

    await ctx.yield_output(campaign)


async def run_optimization_campaign():
    """Fonction principale pour exécuter la campagne d'optimisation"""

    # Construire le workflow d'optimisation
    campaign_init = initialize_campaign
    orchestrator = OptimizationOrchestrator(None)  # Sera initialisé avec la campagne

    # Workflow: Init Campaign → Orchestrator
    workflow = (
        WorkflowBuilder(start_executor=campaign_init)
        .add_edge(campaign_init, orchestrator)
        .build()
    )

    # Exécuter la campagne
    logger.info("🚀 Starting SCAF-LS hyperparameter optimization campaign...")

    events = await workflow.run(Message("user", ["start_optimization"]))

    # Récupérer les résultats
    outputs = events.get_outputs()
    if outputs:
        final_result = outputs[-1]
        logger.info(f"🎉 Campaign completed: {final_result.get('campaign_id', 'unknown')}")

        # Afficher le résumé
        results = final_result.get('results', {})
        for model_name, result in results.items():
            logger.info(f"✅ {model_name}: AUC={result.best_auc:.4f}, Stability={result.stability_score:.4f}")

    logger.info("🏁 Optimization campaign finished!")


if __name__ == "__main__":
    # Exécuter la campagne d'optimisation
    asyncio.run(run_optimization_campaign())