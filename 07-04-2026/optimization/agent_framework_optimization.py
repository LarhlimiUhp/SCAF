"""
SCAF-LS 500 Sub-Agent LightGBM Optimization System
Using Microsoft Agent Framework for comprehensive model optimization

MISSION: Deploy 500 specialized agents to optimize LightGBM AUC from 0.51 to 0.60+
TARGET: Sharpe >1.0, Drawdown <15%, stable performance across temporal folds
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Microsoft Agent Framework imports
from agent_framework import Agent, AgentRuntime, Message, Topic
from agent_framework.azure_ai import AzureAIClient
from agent_framework.core import AgentContext, AgentResponse

# SCAF-LS imports
from scaf_ls.config import Config
from scaf_ls.data.loader import MultiAssetLoader
from scaf_ls.data.engineer import CrossAssetFeatureEngineer
from scaf_ls.features.selector import ICFeatureSelector
from scaf_ls.validation.cv_strategies import PurgedKFold

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent optimization"""
    agent_id: str
    agent_type: str
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    improvement: float
    stability_score: float
    execution_time: float
    trials_completed: int
    fold_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationCampaign:
    """Complete optimization campaign configuration"""
    campaign_id: str
    target_auc: float = 0.60
    target_sharpe: float = 1.0
    target_max_drawdown: float = 0.15
    max_trials_per_agent: int = 50
    max_parallel_agents: int = 20
    time_limit_hours: float = 2.0
    stability_penalty: float = 0.1
    min_improvement_threshold: float = 0.01

    # Agent allocation
    n_hyperparam_agents: int = 200
    n_feature_agents: int = 150
    n_architecture_agents: int = 100
    n_pipeline_agents: int = 50


class BaseOptimizationAgent(Agent):
    """Base class for all LightGBM optimization agents"""

    def __init__(self, agent_id: str, campaign: OptimizationCampaign):
        super().__init__(agent_id)
        self.campaign = campaign
        self.agent_type = "base"
        self.start_time = None
        self.baseline_auc = None

        # Prepare data once for all agents
        self._prepare_shared_data()

    def _prepare_shared_data(self):
        """Prepare training and validation data with proper temporal splits"""
        loader = MultiAssetLoader(Config)
        spx, cross = loader.download()
        engineer = CrossAssetFeatureEngineer(horizon=Config.RETURN_HORIZON)
        X, y, _, _ = engineer.build(spx, cross, Config)

        # Feature selection
        selector = ICFeatureSelector(n_features=Config.N_TOP_FEATURES)
        X_selected = selector.fit_transform(X, y)

        # Temporal split (80/20)
        split_idx = int(len(X_selected) * 0.8)
        self.X_train = X_selected[:split_idx].values.astype(np.float32)
        self.y_train = y[:split_idx].values.astype(np.float32)
        self.X_val = X_selected[split_idx:].values.astype(np.float32)
        self.y_val = y[split_idx:].values.astype(np.float32)

        # Calculate baseline
        self.baseline_auc = self._get_baseline_auc()

    def _get_baseline_auc(self) -> float:
        """Get baseline AUC with current parameters"""
        baseline_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'class_weight': 'balanced',
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1
        }

        model = lgb.LGBMClassifier(**baseline_params)
        model.fit(self.X_train, self.y_train)
        preds = model.predict_proba(self.X_val)[:, 1]
        return roc_auc_score(self.y_val, preds)

    async def optimize(self) -> AgentResult:
        """Main optimization method to be implemented by subclasses"""
        raise NotImplementedError

    async def receive_message(self, message: Message, context: AgentContext) -> AgentResponse:
        """Handle incoming messages"""
        if message.topic == "start_optimization":
            logger.info(f"Agent {self.id} starting optimization")
            self.start_time = time.time()
            result = await self.optimize()
            execution_time = time.time() - self.start_time

            return AgentResponse(
                content=json.dumps({
                    "agent_id": result.agent_id,
                    "agent_type": result.agent_type,
                    "best_score": result.best_score,
                    "improvement": result.improvement,
                    "execution_time": execution_time,
                    "best_params": result.best_params
                }),
                topic="optimization_complete"
            )
        return AgentResponse(content="Unknown message", topic="error")


class LightGBMHyperparameterAgent(BaseOptimizationAgent):
    """Advanced hyperparameter tuning agent for LightGBM core parameters"""

    def __init__(self, agent_id: str, campaign: OptimizationCampaign, focus_area: str = "core"):
        super().__init__(agent_id, campaign)
        self.agent_type = f"hyperparameter_{focus_area}"
        self.focus_area = focus_area

        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )

    async def optimize(self) -> AgentResult:
        """Run hyperparameter optimization"""
        start_time = time.time()

        # Define parameter search space based on focus area
        if self.focus_area == "core":
            param_space = self._core_params_space
        elif self.focus_area == "regularization":
            param_space = self._regularization_params_space
        elif self.focus_area == "sampling":
            param_space = self._sampling_params_space
        else:
            param_space = self._advanced_params_space

        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, param_space),
            n_trials=self.campaign.max_trials_per_agent,
            timeout=self.campaign.time_limit_hours * 3600
        )

        execution_time = time.time() - start_time

        # Evaluate best parameters
        best_params = self.study.best_params.copy()
        best_params.update({
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })

        model = lgb.LGBMClassifier(**best_params)
        model.fit(self.X_train, self.y_train)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        improvement = val_auc - self.baseline_auc

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=best_params,
            best_score=val_auc,
            improvement=improvement,
            stability_score=1.0 - np.std([t.value for t in self.study.trials if t.value is not None]),
            execution_time=execution_time,
            trials_completed=len(self.study.trials),
            fold_scores=[t.value for t in self.study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'focus_area': self.focus_area,
                'parameter_importance': optuna.importance.get_param_importances(self.study)
            }
        )

    def _core_params_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Core parameters: num_leaves, max_depth, learning_rate, n_estimators"""
        return {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        }

    def _regularization_params_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Regularization parameters"""
        return {
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-8, 1.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

    def _sampling_params_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sampling parameters"""
        return {
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        }

    def _advanced_params_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Advanced parameters"""
        return {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        }

    def _objective(self, trial: optuna.Trial, param_space_func) -> float:
        """Optuna objective function"""
        params = param_space_func(trial)
        params.update({
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })

        # Cross-validation with temporal splits
        cv = PurgedKFold(n_splits=5, embargo=10)
        fold_scores = []

        for train_idx, val_idx in cv.split(self.X_train):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, preds)
            fold_scores.append(auc)

        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)
        stability_penalty = self.campaign.stability_penalty * std_auc

        return mean_auc - stability_penalty


class LightGBMFeatureEngineeringAgent(BaseOptimizationAgent):
    """Feature engineering agent specialized for LightGBM preprocessing"""

    def __init__(self, agent_id: str, campaign: OptimizationCampaign, strategy: str):
        super().__init__(agent_id, campaign)
        self.agent_type = f"feature_{strategy}"
        self.strategy = strategy  # 'categorical', 'monotonic', 'interactions', 'scaling', 'outliers'

        # Prepare raw data for feature engineering
        self._prepare_raw_data()

    def _prepare_raw_data(self):
        """Prepare raw data before feature engineering"""
        loader = MultiAssetLoader(Config)
        spx, cross = loader.download()
        engineer = CrossAssetFeatureEngineer(horizon=Config.RETURN_HORIZON)
        X, y, _, _ = engineer.build(spx, cross, Config)

        # Split temporally
        split_idx = int(len(X) * 0.8)
        self.X_train_raw = X[:split_idx].copy()
        self.y_train = y[:split_idx].values
        self.X_val_raw = X[split_idx:].copy()
        self.y_val = y[split_idx:].values

    async def optimize(self) -> AgentResult:
        """Run feature engineering optimization"""
        start_time = time.time()

        # Apply feature engineering strategy
        X_train_processed = self._apply_feature_engineering(self.X_train_raw.copy())
        X_val_processed = self._apply_feature_engineering(self.X_val_raw.copy())

        # Convert to numpy arrays
        X_train_array = X_train_processed.values.astype(np.float32)
        X_val_array = X_val_processed.values.astype(np.float32)

        # Train and evaluate LightGBM with processed features
        best_params = self._get_optimized_params(X_train_array, self.y_train)

        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train_array, self.y_train)
        val_preds = model.predict_proba(X_val_array)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        improvement = val_auc - self.baseline_auc
        execution_time = time.time() - start_time

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=best_params,
            best_score=val_auc,
            improvement=improvement,
            stability_score=self._calculate_stability_score(X_train_array, self.y_train),
            execution_time=execution_time,
            trials_completed=1,  # Feature engineering is not iterative like hyperparameter tuning
            fold_scores=[val_auc],
            metadata={
                'baseline_auc': self.baseline_auc,
                'strategy': self.strategy,
                'n_features_before': self.X_train_raw.shape[1],
                'n_features_after': X_train_processed.shape[1]
            }
        )

    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply specific feature engineering strategy"""
        X_processed = X.copy()

        if self.strategy == 'categorical':
            # Categorical encoding for ordinal financial features
            ordinal_features = ['vix_quartile', 'vol_regime', 'yield_curve_slope', 'credit_regime']
            for feat in ordinal_features:
                if feat in X_processed.columns:
                    X_processed[feat] = X_processed[feat].astype('category').cat.codes

        elif self.strategy == 'monotonic':
            # Apply monotonic constraints hints (LightGBM can use these)
            # For now, just identify potentially monotonic features
            monotonic_candidates = ['vix_zscore', 'volatility_20d', 'credit_spread_zscore']
            for feat in monotonic_candidates:
                if feat in X_processed.columns:
                    # Could add monotonic constraint hints here
                    pass

        elif self.strategy == 'interactions':
            # Create financial domain-specific interactions
            if 'spx_rsi' in X_processed.columns and 'vix_zscore' in X_processed.columns:
                X_processed['rsi_vix_interaction'] = X_processed['spx_rsi'] * X_processed['vix_zscore']

            if 'yield_curve_10y3m' in X_processed.columns and 'credit_spread' in X_processed.columns:
                X_processed['yield_credit_interaction'] = X_processed['yield_curve_10y3m'] * X_processed['credit_spread']

            if 'spx_momentum' in X_processed.columns and 'vol_regime' in X_processed.columns:
                X_processed['momentum_vol_interaction'] = X_processed['spx_momentum'] * X_processed['vol_regime']

        elif self.strategy == 'scaling':
            # Optimize scaling method for LightGBM
            from sklearn.preprocessing import StandardScaler, RobustScaler

            numeric_features = X_processed.select_dtypes(include=[np.number]).columns
            # Use RobustScaler for financial data (handles outliers better)
            scaler = RobustScaler()
            X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])

        elif self.strategy == 'outliers':
            # Handle outliers using robust statistics
            numeric_features = X_processed.select_dtypes(include=[np.number]).columns
            for feat in numeric_features:
                # Winsorize extreme outliers
                q1, q99 = X_processed[feat].quantile([0.01, 0.99])
                X_processed[feat] = X_processed[feat].clip(q1, q99)

        return X_processed

    def _get_optimized_params(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Get optimized LightGBM parameters for the processed features"""
        # Use a quick optimization for feature engineering evaluation
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(X_train):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=20)
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })
        return best_params

    def _calculate_stability_score(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate stability score across different CV folds"""
        cv = PurgedKFold(n_splits=5, embargo=10)
        scores = []

        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        for train_idx, val_idx in cv.split(X_train):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds))

        return 1.0 - np.std(scores)  # Higher stability = lower std


class LightGBMArchitectureAgent(BaseOptimizationAgent):
    """Architecture enhancement agent for advanced LightGBM training strategies"""

    def __init__(self, agent_id: str, campaign: OptimizationCampaign, architecture_type: str):
        super().__init__(agent_id, campaign)
        self.agent_type = f"architecture_{architecture_type}"
        self.architecture_type = architecture_type  # 'custom_loss', 'dart', 'goss', 'early_stopping'

    async def optimize(self) -> AgentResult:
        """Run architecture optimization"""
        start_time = time.time()

        if self.architecture_type == 'custom_loss':
            result = await self._optimize_custom_loss()
        elif self.architecture_type == 'dart':
            result = await self._optimize_dart()
        elif self.architecture_type == 'goss':
            result = await self._optimize_goss()
        elif self.architecture_type == 'early_stopping':
            result = await self._optimize_early_stopping()
        else:
            result = await self._optimize_default()

        execution_time = time.time() - start_time
        result.execution_time = execution_time

        return result

    async def _optimize_custom_loss(self) -> AgentResult:
        """Optimize with custom financial loss functions"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            # Custom loss parameters
            loss_type = trial.suggest_categorical('loss_type', ['focal', 'weighted', 'robust'])
            alpha = trial.suggest_float('alpha', 0.1, 0.9)  # For focal loss
            gamma = trial.suggest_float('gamma', 1.0, 3.0)  # For focal loss
            beta = trial.suggest_float('beta', 0.5, 2.0)    # For weighted loss

            # Base parameters
            params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 32,
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            # Apply custom loss
            if loss_type == 'focal':
                # Implement focal loss (simplified)
                params['scale_pos_weight'] = alpha
                # Could implement custom objective here
            elif loss_type == 'weighted':
                params['scale_pos_weight'] = beta
            elif loss_type == 'robust':
                # Robust loss for outliers
                params['lambda_l1'] = 0.1
                params['lambda_l2'] = 0.1

            # Evaluate
            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=30)

        # Evaluate best
        best_params = study.best_params.copy()
        best_params.update({
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })

        model = lgb.LGBMClassifier(**best_params)
        model.fit(self.X_train, self.y_train)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=best_params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std([t.value for t in study.trials if t.value is not None]),
            execution_time=0,  # Will be set by caller
            trials_completed=len(study.trials),
            fold_scores=[t.value for t in study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'architecture_type': self.architecture_type,
                'loss_type': best_params.get('loss_type', 'default')
            }
        )

    async def _optimize_dart(self) -> AgentResult:
        """Optimize DART (Dropouts meet Multiple Additive Regression Trees)"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            params = {
                'boosting_type': 'dart',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2),
                'skip_drop': trial.suggest_float('skip_drop', 0.4, 0.6),
                'max_drop': trial.suggest_int('max_drop', 40, 60),
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=25)

        best_params = study.best_params.copy()
        best_params.update({
            'boosting_type': 'dart',
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })

        model = lgb.LGBMClassifier(**best_params)
        model.fit(self.X_train, self.y_train)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=best_params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std([t.value for t in study.trials if t.value is not None]),
            execution_time=0,
            trials_completed=len(study.trials),
            fold_scores=[t.value for t in study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'architecture_type': 'dart',
                'drop_rate': best_params.get('drop_rate', 0.1)
            }
        )

    async def _optimize_goss(self) -> AgentResult:
        """Optimize GOSS (Gradient-based One-Side Sampling)"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            params = {
                'boosting_type': 'goss',
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'top_rate': trial.suggest_float('top_rate', 0.1, 0.3),
                'other_rate': trial.suggest_float('other_rate', 0.1, 0.2),
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=25)

        best_params = study.best_params.copy()
        best_params.update({
            'boosting_type': 'goss',
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })

        model = lgb.LGBMClassifier(**best_params)
        model.fit(self.X_train, self.y_train)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=best_params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std([t.value for t in study.trials if t.value is not None]),
            execution_time=0,
            trials_completed=len(study.trials),
            fold_scores=[t.value for t in study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'architecture_type': 'goss',
                'top_rate': best_params.get('top_rate', 0.2)
            }
        )

    async def _optimize_early_stopping(self) -> AgentResult:
        """Optimize early stopping parameters"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            early_stopping_rounds = trial.suggest_int('early_stopping_rounds', 10, 100)
            params = {
                'n_estimators': 1000,  # Large number, let early stopping handle it
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=20)

        best_params = study.best_params.copy()
        best_params.update({
            'n_estimators': 1000,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        })

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(best_params['early_stopping_rounds'], verbose=False)]
        )
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=best_params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std([t.value for t in study.trials if t.value is not None]),
            execution_time=0,
            trials_completed=len(study.trials),
            fold_scores=[t.value for t in study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'architecture_type': 'early_stopping',
                'early_stopping_rounds': best_params.get('early_stopping_rounds', 50)
            }
        )

    async def _optimize_default(self) -> AgentResult:
        """Default optimization"""
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=0.5,
            execution_time=0,
            trials_completed=1,
            fold_scores=[val_auc],
            metadata={'baseline_auc': self.baseline_auc, 'architecture_type': 'default'}
        )


class LightGBMPipelineAgent(BaseOptimizationAgent):
    """Data pipeline optimization agent for temporal CV and preprocessing"""

    def __init__(self, agent_id: str, campaign: OptimizationCampaign, pipeline_type: str):
        super().__init__(agent_id, campaign)
        self.agent_type = f"pipeline_{pipeline_type}"
        self.pipeline_type = pipeline_type  # 'temporal_cv', 'sample_weighting', 'robust_stats'

    async def optimize(self) -> AgentResult:
        """Run pipeline optimization"""
        start_time = time.time()

        if self.pipeline_type == 'temporal_cv':
            result = await self._optimize_temporal_cv()
        elif self.pipeline_type == 'sample_weighting':
            result = await self._optimize_sample_weighting()
        elif self.pipeline_type == 'robust_stats':
            result = await self._optimize_robust_stats()
        else:
            result = await self._optimize_default_pipeline()

        execution_time = time.time() - start_time
        result.execution_time = execution_time

        return result

    async def _optimize_temporal_cv(self) -> AgentResult:
        """Optimize temporal cross-validation parameters"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            n_splits = trial.suggest_int('n_splits', 3, 10)
            embargo = trial.suggest_int('embargo', 1, 20)
            gap = trial.suggest_int('gap', 0, 10)

            cv = PurgedKFold(n_splits=n_splits, embargo=embargo, gap=gap)

            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 32,
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            scores = []
            for train_idx, val_idx in cv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=20)

        # Use best CV parameters for final evaluation
        best_cv_params = study.best_params
        cv = PurgedKFold(**best_cv_params)

        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        # Evaluate with optimized CV
        scores = []
        for train_idx, val_idx in cv.split(self.X_train):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds))

        val_auc = np.mean(scores)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std(scores),
            execution_time=0,
            trials_completed=len(study.trials),
            fold_scores=scores,
            metadata={
                'baseline_auc': self.baseline_auc,
                'pipeline_type': 'temporal_cv',
                'cv_params': best_cv_params
            }
        )

    async def _optimize_sample_weighting(self) -> AgentResult:
        """Optimize sample weighting strategies"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            # Different weighting strategies
            strategy = trial.suggest_categorical('strategy', ['uniform', 'class_balance', 'temporal_decay', 'volatility_based'])

            if strategy == 'uniform':
                sample_weight = None
            elif strategy == 'class_balance':
                # Balance classes
                class_counts = np.bincount(self.y_train.astype(int))
                sample_weight = np.where(self.y_train == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
                sample_weight = sample_weight * len(self.y_train) / np.sum(sample_weight)
            elif strategy == 'temporal_decay':
                # More recent samples get higher weight
                decay_factor = trial.suggest_float('decay_factor', 0.95, 0.999)
                weights = np.power(decay_factor, np.arange(len(self.y_train))[::-1])
                sample_weight = weights
            elif strategy == 'volatility_based':
                # Higher weight for high volatility periods (proxy for importance)
                # Use a simple volatility proxy from features
                vol_proxy = np.random.rand(len(self.y_train))  # Placeholder
                sample_weight = 1.0 + vol_proxy

            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 32,
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(self.X_train):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]

                fold_weights = sample_weight[train_idx] if sample_weight is not None else None

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=15)

        # Evaluate best strategy
        best_strategy = study.best_params['strategy']
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        # Apply best weighting
        if best_strategy == 'class_balance':
            class_counts = np.bincount(self.y_train.astype(int))
            sample_weight = np.where(self.y_train == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
            sample_weight = sample_weight * len(self.y_train) / np.sum(sample_weight)
        elif best_strategy == 'temporal_decay':
            decay_factor = study.best_params.get('decay_factor', 0.99)
            sample_weight = np.power(decay_factor, np.arange(len(self.y_train))[::-1])
        else:
            sample_weight = None

        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train, sample_weight=sample_weight)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std([t.value for t in study.trials if t.value is not None]),
            execution_time=0,
            trials_completed=len(study.trials),
            fold_scores=[t.value for t in study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'pipeline_type': 'sample_weighting',
                'weighting_strategy': best_strategy
            }
        )

    async def _optimize_robust_stats(self) -> AgentResult:
        """Optimize robust statistical methods for preprocessing"""
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            # Robust preprocessing options
            outlier_method = trial.suggest_categorical('outlier_method', ['none', 'iqr', 'isolation_forest', 'robust_scaler'])
            imputation_method = trial.suggest_categorical('imputation_method', ['none', 'median', 'knn'])

            # Apply preprocessing
            X_processed = self.X_train.copy()

            # Outlier handling
            if outlier_method == 'iqr':
                for col in range(X_processed.shape[1]):
                    q1, q3 = np.percentile(X_processed[:, col], [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    X_processed[:, col] = np.clip(X_processed[:, col], lower_bound, upper_bound)
            elif outlier_method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso.fit_predict(X_processed)
                X_processed = X_processed[outliers == 1]  # Keep only inliers
                y_processed = self.y_train[outliers == 1]
            elif outlier_method == 'robust_scaler':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                X_processed = scaler.fit_transform(X_processed)

            # Imputation
            if imputation_method == 'median':
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_processed = imputer.fit_transform(X_processed)
            elif imputation_method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                X_processed = imputer.fit_transform(X_processed)

            if outlier_method == 'isolation_forest':
                y_train_processed = y_processed
            else:
                y_train_processed = self.y_train

            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 32,
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42
            }

            cv = PurgedKFold(n_splits=3, embargo=5)
            scores = []
            for train_idx, val_idx in cv.split(X_processed):
                X_fold_train = X_processed[train_idx]
                y_fold_train = y_train_processed[train_idx]
                X_fold_val = X_processed[val_idx]
                y_fold_val = y_train_processed[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict_proba(X_fold_val)[:, 1]
                scores.append(roc_auc_score(y_fold_val, preds))

            return np.mean(scores)

        study.optimize(objective, n_trials=15)

        # Apply best preprocessing and evaluate
        best_params_preprocessing = study.best_params
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        # Apply preprocessing to full dataset
        X_train_processed = self._apply_robust_preprocessing(self.X_train, best_params_preprocessing)
        X_val_processed = self._apply_robust_preprocessing(self.X_val, best_params_preprocessing)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_processed, self.y_train)
        val_preds = model.predict_proba(X_val_processed)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=1.0 - np.std([t.value for t in study.trials if t.value is not None]),
            execution_time=0,
            trials_completed=len(study.trials),
            fold_scores=[t.value for t in study.trials if t.value is not None],
            metadata={
                'baseline_auc': self.baseline_auc,
                'pipeline_type': 'robust_stats',
                'preprocessing_params': best_params_preprocessing
            }
        )

    def _apply_robust_preprocessing(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply robust preprocessing"""
        X_processed = X.copy()

        outlier_method = params.get('outlier_method', 'none')
        imputation_method = params.get('imputation_method', 'none')

        # Outlier handling
        if outlier_method == 'iqr':
            for col in range(X_processed.shape[1]):
                q1, q3 = np.percentile(X_processed[:, col], [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                X_processed[:, col] = np.clip(X_processed[:, col], lower_bound, upper_bound)
        elif outlier_method == 'robust_scaler':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_processed = scaler.fit_transform(X_processed)

        # Imputation
        if imputation_method == 'median':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_processed = imputer.fit_transform(X_processed)
        elif imputation_method == 'knn':
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            X_processed = imputer.fit_transform(X_processed)

        return X_processed

    async def _optimize_default_pipeline(self) -> AgentResult:
        """Default pipeline optimization"""
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 32,
            'objective': 'binary',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        val_preds = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_preds)

        return AgentResult(
            agent_id=self.id,
            agent_type=self.agent_type,
            model_name="LightGBM",
            best_params=params,
            best_score=val_auc,
            improvement=val_auc - self.baseline_auc,
            stability_score=0.5,
            execution_time=0,
            trials_completed=1,
            fold_scores=[val_auc],
            metadata={'baseline_auc': self.baseline_auc, 'pipeline_type': 'default'}
        )


class LightGBMMasterOrchestrator(Agent):
    """Master orchestrator for the 500 sub-agent LightGBM optimization system"""

    def __init__(self, campaign: OptimizationCampaign):
        super().__init__("master_orchestrator")
        self.campaign = campaign
        self.all_results: List[AgentResult] = []
        self.agent_registry: Dict[str, Agent] = {}
        self.runtime = None

        # Progress tracking
        self.completed_agents = 0
        self.total_agents = (
            campaign.n_hyperparam_agents +
            campaign.n_feature_agents +
            campaign.n_architecture_agents +
            campaign.n_pipeline_agents
        )

    async def initialize_agents(self):
        """Initialize all 500 sub-agents"""
        logger.info(f"Initializing {self.total_agents} sub-agents...")

        # Hyperparameter agents (200)
        for i in range(self.campaign.n_hyperparam_agents):
            if i < 100:
                agent = LightGBMHyperparameterAgent(f"hyperparam_bayes_{i}", self.campaign, "core")
            elif i < 150:
                agent = LightGBMHyperparameterAgent(f"hyperparam_reg_{i-100}", self.campaign, "regularization")
            elif i < 175:
                agent = LightGBMHyperparameterAgent(f"hyperparam_sample_{i-150}", self.campaign, "sampling")
            else:
                agent = LightGBMHyperparameterAgent(f"hyperparam_adv_{i-175}", self.campaign, "advanced")

            self.agent_registry[agent.id] = agent

        # Feature engineering agents (150)
        strategies = ['categorical', 'monotonic', 'interactions', 'scaling', 'outliers']
        agents_per_strategy = self.campaign.n_feature_agents // len(strategies)

        idx = 0
        for strategy in strategies:
            for j in range(agents_per_strategy):
                agent = LightGBMFeatureEngineeringAgent(f"feature_{strategy}_{j}", self.campaign, strategy)
                self.agent_registry[agent.id] = agent
                idx += 1

        # Architecture agents (100)
        arch_types = ['custom_loss', 'dart', 'goss', 'early_stopping']
        agents_per_arch = self.campaign.n_architecture_agents // len(arch_types)

        idx = 0
        for arch_type in arch_types:
            for j in range(agents_per_arch):
                agent = LightGBMArchitectureAgent(f"arch_{arch_type}_{j}", self.campaign, arch_type)
                self.agent_registry[agent.id] = agent
                idx += 1

        # Pipeline agents (50)
        pipeline_types = ['temporal_cv', 'sample_weighting', 'robust_stats']
        agents_per_pipeline = self.campaign.n_pipeline_agents // len(pipeline_types)

        idx = 0
        for pipeline_type in pipeline_types:
            for j in range(agents_per_pipeline):
                agent = LightGBMPipelineAgent(f"pipeline_{pipeline_type}_{j}", self.campaign, pipeline_type)
                self.agent_registry[agent.id] = agent
                idx += 1

        logger.info(f"Successfully initialized {len(self.agent_registry)} agents")

    async def run_optimization_campaign(self) -> Dict[str, Any]:
        """Run the complete 500-agent optimization campaign"""
        logger.info("🚀 Starting LightGBM 500-Agent Optimization Campaign")
        logger.info(f"Target: AUC > {self.campaign.target_auc}, Sharpe > {self.campaign.target_sharpe}")
        logger.info(f"Time limit: {self.campaign.time_limit_hours} hours")

        start_time = time.time()

        # Initialize runtime and agents
        self.runtime = AgentRuntime()
        await self.initialize_agents()

        # Register all agents
        for agent in self.agent_registry.values():
            await self.runtime.register_agent(agent)

        # Start optimization in parallel batches
        semaphore = asyncio.Semaphore(self.campaign.max_parallel_agents)
        tasks = []

        async def run_agent_with_semaphore(agent_id: str):
            async with semaphore:
                agent = self.agent_registry[agent_id]
                try:
                    # Send start message
                    message = Message(
                        from_agent=self.id,
                        to_agent=agent_id,
                        topic="start_optimization",
                        content=json.dumps({"campaign_id": self.campaign.campaign_id})
                    )

                    response = await self.runtime.send_message(message)
                    result_data = json.loads(response.content)

                    # Create AgentResult
                    result = AgentResult(
                        agent_id=result_data["agent_id"],
                        agent_type=result_data["agent_type"],
                        model_name="LightGBM",
                        best_params={},  # Will be populated from detailed results
                        best_score=result_data["best_score"],
                        improvement=result_data.get("improvement", 0.0),
                        stability_score=0.5,
                        execution_time=result_data["execution_time"],
                        trials_completed=1,
                        fold_scores=[result_data["best_score"]],
                        metadata={"raw_response": result_data}
                    )

                    self.all_results.append(result)
                    self.completed_agents += 1

                    logger.info(f"✅ Agent {agent_id} completed: AUC={result.best_score:.4f}, "
                              f"Improvement={result.improvement:.4f}")

                except Exception as e:
                    logger.error(f"❌ Agent {agent_id} failed: {e}")

        # Create tasks for all agents
        for agent_id in self.agent_registry.keys():
            task = asyncio.create_task(run_agent_with_semaphore(agent_id))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        campaign_results = self._aggregate_results()
        execution_time = time.time() - start_time

        logger.info(f"🎯 Campaign completed in {execution_time:.1f}s")
        logger.info(f"Best AUC achieved: {campaign_results['best_auc']:.4f}")
        logger.info(f"Agents completed: {len(self.all_results)}/{self.total_agents}")

        return {
            "campaign_id": self.campaign.campaign_id,
            "execution_time": execution_time,
            "total_agents": self.total_agents,
            "completed_agents": len(self.all_results),
            "best_result": campaign_results["best_result"],
            "all_results": [result.__dict__ for result in self.all_results],
            "summary": campaign_results
        }

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all agents"""
        if not self.all_results:
            return {"best_auc": 0.0, "best_result": None}

        # Find best result
        best_result = max(self.all_results, key=lambda x: x.best_score)

        # Calculate statistics
        auc_scores = [r.best_score for r in self.all_results]
        improvements = [r.improvement for r in self.all_results]

        summary = {
            "best_auc": best_result.best_score,
            "best_result": best_result.__dict__,
            "mean_auc": np.mean(auc_scores),
            "std_auc": np.std(auc_scores),
            "mean_improvement": np.mean(improvements),
            "max_improvement": max(improvements),
            "agents_above_target": sum(1 for r in self.all_results if r.best_score >= self.campaign.target_auc),
            "target_auc": self.campaign.target_auc
        }

        return summary

    async def receive_message(self, message: Message, context: AgentContext) -> AgentResponse:
        """Handle incoming messages"""
        if message.topic == "campaign_status":
            status = {
                "completed": self.completed_agents,
                "total": self.total_agents,
                "progress": self.completed_agents / self.total_agents if self.total_agents > 0 else 0,
                "best_auc": max([r.best_score for r in self.all_results], default=0.0)
            }
            return AgentResponse(content=json.dumps(status), topic="status_response")

        return AgentResponse(content="Unknown message", topic="error")


async def run_500_agent_optimization() -> Dict[str, Any]:
    """Main function to run the 500-agent LightGBM optimization"""

    # Configure campaign
    campaign = OptimizationCampaign(
        campaign_id=f"lightgbm_opt_{int(time.time())}",
        target_auc=0.60,
        target_sharpe=1.0,
        target_max_drawdown=0.15,
        max_trials_per_agent=50,
        max_parallel_agents=20,  # Limit parallel execution
        time_limit_hours=2.0,
        stability_penalty=0.1,
        min_improvement_threshold=0.01,
        n_hyperparam_agents=200,
        n_feature_agents=150,
        n_architecture_agents=100,
        n_pipeline_agents=50
    )

    # Create and run orchestrator
    orchestrator = LightGBMMasterOrchestrator(campaign)
    results = await orchestrator.run_optimization_campaign()

    # Save results
    output_file = f"lightgbm_500_agent_results_{campaign.campaign_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📄 Results saved to {output_file}")

    return results


if __name__ == "__main__":
    # Run the 500-agent optimization
    asyncio.run(run_500_agent_optimization())