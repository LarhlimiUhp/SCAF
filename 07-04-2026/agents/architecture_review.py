"""
SCAF-LS Architecture Review Agents (100 agents)
Analyze ensemble structure, model correlations, and component effectiveness
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


class ArchitectureReviewAgent(ABC):
    """Base class for architecture review agents"""

    def __init__(self, agent_id: str, agent_name: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config = config
        self.analysis_history = deque(maxlen=100)
        self.insights = []
        self.recommendations = []

    @abstractmethod
    def analyze(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform architecture analysis and return findings"""
        pass

    def record_finding(self, finding: Dict[str, Any]):
        """Record analysis finding"""
        finding['timestamp'] = pd.Timestamp.now()
        finding['agent_id'] = self.agent_id
        self.analysis_history.append(finding)
        if finding.get('is_insight'):
            self.insights.append(finding)

    def get_summary(self) -> Dict[str, Any]:
        """Get agent analysis summary"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'total_analyses': len(self.analysis_history),
            'insights_count': len(self.insights),
            'recommendations': self.recommendations
        }


class SingleModelVsEnsembleAgent(ArchitectureReviewAgent):
    """Agent evaluates single model vs ensemble performance trade-offs"""

    def analyze(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare single models against ensemble"""
        model_aucs = results_data.get('model_aucs', {})
        ensemble_auc = results_data.get('ensemble_auc', 0.5)
        model_sharpes = results_data.get('model_sharpes', {})
        ensemble_sharpe = results_data.get('ensemble_sharpe', 0.0)

        analysis = {
            'ensemble_vs_best': {},
            'is_ensemble_beneficial': False,
            'outlier_models': [],
            'justification': ''
        }

        if not model_aucs:
            return analysis

        best_model = max(model_aucs.items(), key=lambda x: x[1])
        worst_model = min(model_aucs.items(), key=lambda x: x[1])
        avg_model_auc = np.mean(list(model_aucs.values()))

        analysis['ensemble_vs_best'] = {
            'best_model': best_model[0],
            'best_auc': float(best_model[1]),
            'ensemble_auc': float(ensemble_auc),
            'improvement': float(ensemble_auc - best_model[1]),
            'improvement_pct': float((ensemble_auc - best_model[1]) / (best_model[1] + 1e-6) * 100)
        }

        analysis['ensemble_vs_average'] = {
            'avg_model_auc': float(avg_model_auc),
            'ensemble_auc': float(ensemble_auc),
            'improvement': float(ensemble_auc - avg_model_auc),
            'improvement_pct': float((ensemble_auc - avg_model_auc) / (avg_model_auc + 1e-6) * 100)
        }

        # Check if ensemble is beneficial
        analysis['is_ensemble_beneficial'] = ensemble_auc > best_model[1]

        # Check for Sharpe improvement
        if model_sharpes and ensemble_sharpe:
            best_sharpe = max(model_sharpes.values())
            analysis['sharpe_improvement'] = ensemble_sharpe - best_sharpe
            if isinstance(ensemble_sharpe, (int, float)) and isinstance(best_sharpe, (int, float)):
                if ensemble_sharpe > best_sharpe:
                    analysis['is_ensemble_beneficial'] = True

        # Identify outliers
        for model_name, auc in model_aucs.items():
            if auc < worst_model[1] + 0.02:
                analysis['outlier_models'].append({
                    'model': model_name,
                    'auc': float(auc),
                    'status': 'underperforming'
                })

        if analysis['is_ensemble_beneficial']:
            analysis['justification'] = f"Ensemble adds {analysis['ensemble_vs_best']['improvement']:.4f} AUC vs best single model"
        else:
            analysis['justification'] = f"Ensemble underperforms best single model by {-analysis['ensemble_vs_best']['improvement']:.4f} AUC"

        finding = {
            'analysis_type': 'single_vs_ensemble',
            'results': analysis,
            'is_insight': not analysis['is_ensemble_beneficial'],
            'severity': 'high' if not analysis['is_ensemble_beneficial'] else 'low'
        }
        self.record_finding(finding)

        if not analysis['is_ensemble_beneficial']:
            self.recommendations.append({
                'priority': 'critical',
                'action': 'Review ensemble weights - single model outperforming',
                'rationale': f"Best model {best_model[0]} (AUC {best_model[1]:.4f}) > ensemble (AUC {ensemble_auc:.4f})"
            })

        return analysis


class ModelCorrelationAgent(ArchitectureReviewAgent):
    """Agent analyzes model prediction correlations and diversification"""

    def analyze(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model correlation structure"""
        model_predictions = results_data.get('model_predictions', {})  # Dict[model_name, predictions_array]

        analysis = {
            'correlation_matrix': {},
            'diversification_score': 0.0,
            'redundant_models': [],
            'complementary_pairs': []
        }

        if not model_predictions or len(model_predictions) < 2:
            return analysis

        # Build correlation matrix
        model_names = list(model_predictions.keys())
        predictions_array = np.array([model_predictions[name] for name in model_names])

        # Handle NaN/inf values
        predictions_array = np.nan_to_num(predictions_array, nan=0.5, posinf=1.0, neginf=0.0)

        # Calculate pairwise correlations
        n_models = len(model_names)
        corr_matrix = np.ones((n_models, n_models))

        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    # Use Spearman for robustness
                    corr, _ = spearmanr(predictions_array[i], predictions_array[j])
                    if np.isnan(corr):
                        corr = 0.0
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    analysis['correlation_matrix'][f"{model_names[i]}_vs_{model_names[j]}"] = float(corr)
                except Exception as e:
                    logger.warning(f"Correlation calculation failed: {e}")
                    analysis['correlation_matrix'][f"{model_names[i]}_vs_{model_names[j]}"] = 0.0

        # Identify redundant models (high correlation > 0.85)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if corr_matrix[i, j] > 0.85:
                    analysis['redundant_models'].append({
                        'model_a': model_names[i],
                        'model_b': model_names[j],
                        'correlation': float(corr_matrix[i, j]),
                        'recommendation': f'Consider removing {model_names[j]} (highly correlated with {model_names[i]})'
                    })

        # Identify complementary models (low/negative correlation)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if corr_matrix[i, j] < 0.3:
                    analysis['complementary_pairs'].append({
                        'model_a': model_names[i],
                        'model_b': model_names[j],
                        'correlation': float(corr_matrix[i, j]),
                        'benefit': 'Good diversification benefit'
                    })

        # Calculate diversification score (lower avg correlation = higher score)
        off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        avg_corr = np.mean(off_diagonal)
        analysis['diversification_score'] = float(1.0 - avg_corr)  # Higher is better

        finding = {
            'analysis_type': 'correlation_analysis',
            'results': analysis,
            'is_insight': len(analysis['redundant_models']) > 0,
            'severity': 'medium' if len(analysis['redundant_models']) > 0 else 'low'
        }
        self.record_finding(finding)

        if analysis['redundant_models']:
            self.recommendations.append({
                'priority': 'high',
                'action': 'Remove redundant models',
                'rationale': f"Found {len(analysis['redundant_models'])} highly correlated model pairs"
            })

        if analysis['diversification_score'] < 0.3:
            self.recommendations.append({
                'priority': 'high',
                'action': 'Increase model diversity',
                'rationale': f"Low diversification score: {analysis['diversification_score']:.3f}"
            })

        return analysis


class StackingMetaLearnerAgent(ArchitectureReviewAgent):
    """Agent evaluates stacking meta-learner effectiveness"""

    def analyze(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stacking meta-learner contribution"""
        base_ensemble_predictions = results_data.get('base_ensemble_predictions', [])
        stacking_predictions = results_data.get('stacking_predictions', [])
        y_true = results_data.get('y_true', [])
        meta_learner_weights = results_data.get('meta_learner_weights', {})

        analysis = {
            'meta_auc': 0.0,
            'base_auc': 0.0,
            'improvement': 0.0,
            'meta_learner_efficiency': 0.0,
            'feature_importance': {},
            'is_meta_beneficial': False
        }

        try:
            # Handle empty or invalid data
            if not base_ensemble_predictions or not stacking_predictions or not y_true:
                return analysis

            base_ensemble_predictions = np.asarray(base_ensemble_predictions, dtype=float)
            stacking_predictions = np.asarray(stacking_predictions, dtype=float)
            y_true = np.asarray(y_true, dtype=float)

            # Handle NaN values
            mask = ~(np.isnan(base_ensemble_predictions) | np.isnan(stacking_predictions) | np.isnan(y_true))
            if mask.sum() < 10:
                return analysis

            base_ensemble_predictions = base_ensemble_predictions[mask]
            stacking_predictions = stacking_predictions[mask]
            y_true = y_true[mask]

            # Calculate AUCs
            if len(np.unique(y_true)) > 1:
                try:
                    base_auc = roc_auc_score(y_true, base_ensemble_predictions)
                    analysis['base_auc'] = float(base_auc)
                except Exception:
                    base_auc = 0.5

                try:
                    meta_auc = roc_auc_score(y_true, stacking_predictions)
                    analysis['meta_auc'] = float(meta_auc)
                except Exception:
                    meta_auc = base_auc

                analysis['improvement'] = float(meta_auc - base_auc)
                analysis['is_meta_beneficial'] = meta_auc > base_auc

                # Calculate efficiency (improvement per complexity)
                n_models = len(meta_learner_weights)
                if n_models > 0:
                    analysis['meta_learner_efficiency'] = float(analysis['improvement'] / (n_models + 1))

        except Exception as e:
            logger.warning(f"Stacking analysis failed: {e}")

        finding = {
            'analysis_type': 'stacking_effectiveness',
            'results': analysis,
            'is_insight': not analysis['is_meta_beneficial'],
            'severity': 'medium' if not analysis['is_meta_beneficial'] else 'low'
        }
        self.record_finding(finding)

        if not analysis['is_meta_beneficial']:
            self.recommendations.append({
                'priority': 'high',
                'action': 'Simplify ensemble to base predictions',
                'rationale': f"Stacking underperforms base ensemble by {-analysis['improvement']:.4f} AUC"
            })

        return analysis


class ComputationalComplexityAgent(ArchitectureReviewAgent):
    """Agent assesses computational complexity vs performance benefit"""

    def analyze(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate complexity-to-benefit ratio"""
        num_models = results_data.get('num_models', 0)
        inference_time_ms = results_data.get('inference_time_ms', 0.0)
        training_time_minutes = results_data.get('training_time_minutes', 0.0)
        ensemble_auc = results_data.get('ensemble_auc', 0.5)
        baseline_auc = results_data.get('baseline_auc', 0.5)
        memory_usage_mb = results_data.get('memory_usage_mb', 0.0)

        analysis = {
            'efficiency_ratio': 0.0,
            'is_acceptable': True,
            'bottlenecks': [],
            'recommendations': [],
            'complexity_breakdown': {}
        }

        try:
            # Calculate efficiency metrics
            auc_improvement = ensemble_auc - baseline_auc

            if auc_improvement > 0:
                time_efficiency = auc_improvement / (inference_time_ms / 1000.0 + 1e-6)
                complexity_efficiency = auc_improvement / (num_models + 1)
            else:
                time_efficiency = 0.0
                complexity_efficiency = 0.0

            analysis['efficiency_ratio'] = float(complexity_efficiency)
            analysis['time_efficiency'] = float(time_efficiency)
            analysis['memory_per_model'] = float(memory_usage_mb / (num_models + 1)) if num_models > 0 else 0.0

            # Check for bottlenecks
            if inference_time_ms > 100:  # > 100ms for prediction
                analysis['bottlenecks'].append({
                    'type': 'inference_latency',
                    'value': float(inference_time_ms),
                    'threshold': 100.0,
                    'severity': 'medium'
                })

            if training_time_minutes > 120:  # > 2 hours
                analysis['bottlenecks'].append({
                    'type': 'training_time',
                    'value': float(training_time_minutes),
                    'threshold': 120.0,
                    'severity': 'low'
                })

            if memory_usage_mb > 1000:  # > 1GB
                analysis['bottlenecks'].append({
                    'type': 'memory_usage',
                    'value': float(memory_usage_mb),
                    'threshold': 1000.0,
                    'severity': 'medium'
                })

            # Evaluate acceptability
            analysis['is_acceptable'] = (
                inference_time_ms < 200 and
                memory_usage_mb < 2000 and
                complexity_efficiency > 0.01
            )

            analysis['complexity_breakdown'] = {
                'num_models': num_models,
                'inference_time_ms': float(inference_time_ms),
                'training_time_minutes': float(training_time_minutes),
                'memory_usage_mb': float(memory_usage_mb),
                'auc_improvement': float(auc_improvement)
            }

        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")

        finding = {
            'analysis_type': 'computational_complexity',
            'results': analysis,
            'is_insight': len(analysis['bottlenecks']) > 0,
            'severity': 'high' if not analysis['is_acceptable'] else 'low'
        }
        self.record_finding(finding)

        if not analysis['is_acceptable']:
            self.recommendations.append({
                'priority': 'medium',
                'action': 'Optimize model architecture for efficiency',
                'rationale': f"Current complexity exceeds acceptable thresholds: {len(analysis['bottlenecks'])} bottlenecks detected"
            })

        return analysis


def create_architecture_review_agents(config: Dict[str, Any]) -> List[ArchitectureReviewAgent]:
    """Factory function to create all 100 architecture review agents"""
    agents = []

    # Single model vs ensemble agents (20)
    for i in range(20):
        agent = SingleModelVsEnsembleAgent(
            agent_id=f"arch_single_ensemble_{i:03d}",
            agent_name=f"Single-vs-Ensemble-{i}",
            config=config
        )
        agents.append(agent)

    # Model correlation agents (30)
    for i in range(30):
        agent = ModelCorrelationAgent(
            agent_id=f"arch_correlation_{i:03d}",
            agent_name=f"Correlation-Analyst-{i}",
            config=config
        )
        agents.append(agent)

    # Stacking meta-learner agents (30)
    for i in range(30):
        agent = StackingMetaLearnerAgent(
            agent_id=f"arch_stacking_{i:03d}",
            agent_name=f"Stacking-Evaluator-{i}",
            config=config
        )
        agents.append(agent)

    # Computational complexity agents (20)
    for i in range(20):
        agent = ComputationalComplexityAgent(
            agent_id=f"arch_complexity_{i:03d}",
            agent_name=f"Complexity-Analyst-{i}",
            config=config
        )
        agents.append(agent)

    return agents
