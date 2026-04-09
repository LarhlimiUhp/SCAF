"""
Feature Importance Analysis for SCAF-LS
SHAP and permutation importance calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """Advanced feature importance analysis using multiple methods"""

    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.shap_explainer = None
        self.permutation_results = None

    def calculate_shap_importance(self, model: BaseEstimator, X: pd.DataFrame,
                                 y: pd.Series, max_evals: int = 1000) -> Dict:
        """
        Calculate SHAP feature importance with comprehensive analysis
        """
        print("🔮 Calculating SHAP values...")

        # Create explainer
        try:
            self.shap_explainer = shap.Explainer(model)
            shap_values = self.shap_explainer(X)
        except Exception as e:
            print(f"SHAP TreeExplainer failed: {e}, using LinearExplainer")
            # Fallback for non-tree models
            self.shap_explainer = shap.LinearExplainer(model, X)
            shap_values = self.shap_explainer(X)

        # Calculate feature importance
        feature_importance = np.abs(shap_values.values).mean(axis=0)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_mean_abs': feature_importance,
            'shap_std': np.abs(shap_values.values).std(axis=0),
            'shap_mean': shap_values.values.mean(axis=0)
        }).sort_values('shap_mean_abs', ascending=False)

        # Calculate feature interactions (top 10 features)
        top_features = importance_df.head(10)['feature'].tolist()
        interactions = self._calculate_shap_interactions(shap_values, top_features)

        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'top_10_features': top_features,
            'interactions': interactions,
            'explainer': self.shap_explainer
        }

    def calculate_permutation_importance(self, model: BaseEstimator, X: pd.DataFrame,
                                       y: pd.Series, n_repeats: int = 10) -> Dict:
        """
        Calculate permutation feature importance with statistical significance
        """
        print("🔄 Calculating permutation importance...")

        from sklearn.inspection import permutation_importance

        # Calculate permutation importance
        perm_results = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring='neg_mean_squared_error'
        )

        # Create results DataFrame
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_results.importances_mean,
            'importance_std': perm_results.importances_std,
            'importance_p_value': self._calculate_p_values(perm_results)
        }).sort_values('importance_mean', ascending=False)

        self.permutation_results = perm_results

        return {
            'permutation_results': perm_results,
            'feature_importance': perm_df,
            'significant_features': perm_df[perm_df['importance_p_value'] < 0.05]['feature'].tolist()
        }

    def calculate_feature_stability(self, model: BaseEstimator, X: pd.DataFrame,
                                  y: pd.Series, n_bootstraps: int = 50) -> Dict:
        """
        Calculate feature importance stability across bootstraps
        """
        print("🎲 Calculating feature stability...")

        importance_bootstraps = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            # Fit model and get importance
            model_boot = model.__class__(**model.get_params())
            model_boot.fit(X_boot, y_boot)

            # Get feature importance (using coefficients for linear models, feature_importances_ for trees)
            if hasattr(model_boot, 'feature_importances_'):
                importance = model_boot.feature_importances_
            elif hasattr(model_boot, 'coef_'):
                importance = np.abs(model_boot.coef_.flatten())
            else:
                # Fallback: use permutation importance
                from sklearn.inspection import permutation_importance
                perm = permutation_importance(model_boot, X_boot, y_boot, n_repeats=5, random_state=i)
                importance = perm.importances_mean

            importance_bootstraps.append(importance)

        # Calculate stability metrics
        importance_matrix = np.array(importance_bootstraps)
        stability_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': importance_matrix.mean(axis=0),
            'importance_std': importance_matrix.std(axis=0),
            'stability_score': 1 / (1 + importance_matrix.std(axis=0) / (importance_matrix.mean(axis=0) + 1e-6)),
            'importance_cv': importance_matrix.std(axis=0) / (importance_matrix.mean(axis=0) + 1e-6)
        }).sort_values('stability_score', ascending=False)

        return {
            'stability_scores': stability_df,
            'importance_bootstraps': importance_matrix,
            'stable_features': stability_df[stability_df['stability_score'] > 0.7]['feature'].tolist()
        }

    def compare_feature_sets(self, model_class, X_full: pd.DataFrame, y: pd.Series,
                           feature_sets: Dict[str, List[str]], cv_folds: int = 5) -> Dict:
        """
        Compare performance of different feature sets
        """
        print("⚖️  Comparing feature sets...")

        results = {}

        for name, features in feature_sets.items():
            print(f"Testing {name}: {len(features)} features")

            X_subset = X_full[features]

            # Cross-validation scores
            model = model_class(random_state=self.random_state)
            cv_scores = cross_val_score(
                model, X_subset, y,
                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring='neg_mean_squared_error'
            )

            # Fit model and get predictions for additional metrics
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)

            results[name] = {
                'n_features': len(features),
                'cv_mse_mean': -cv_scores.mean(),
                'cv_mse_std': cv_scores.std(),
                'train_r2': r2_score(y, y_pred),
                'train_mse': mean_squared_error(y, y_pred),
                'features': features
            }

        # Create comparison DataFrame
        comparison_df = pd.DataFrame.from_dict(results, orient='index')
        comparison_df = comparison_df.sort_values('cv_mse_mean')

        return {
            'comparison_results': comparison_df,
            'best_feature_set': comparison_df.index[0],
            'performance_gain': (comparison_df.iloc[0]['cv_mse_mean'] - comparison_df.iloc[-1]['cv_mse_mean']) / comparison_df.iloc[-1]['cv_mse_mean']
        }

    def _calculate_shap_interactions(self, shap_values, top_features: List[str]) -> Dict:
        """Calculate SHAP feature interactions for top features"""
        try:
            # Calculate interaction values for top features
            interactions = {}
            feature_indices = [shap_values.feature_names.index(f) for f in top_features]

            for i, feat1 in enumerate(top_features):
                for j, feat2 in enumerate(top_features[i+1:], i+1):
                    idx1 = shap_values.feature_names.index(feat1)
                    idx2 = shap_values.feature_names.index(feat2)

                    # Simple interaction measure: correlation of SHAP values
                    shap_corr = np.corrcoef(
                        shap_values.values[:, idx1],
                        shap_values.values[:, idx2]
                    )[0, 1]

                    interactions[f"{feat1}_{feat2}"] = {
                        'correlation': shap_corr,
                        'strength': abs(shap_corr)
                    }

            # Sort by interaction strength
            sorted_interactions = sorted(
                interactions.items(),
                key=lambda x: x[1]['strength'],
                reverse=True
            )

            return dict(sorted_interactions[:10])  # Top 10 interactions

        except Exception as e:
            print(f"Could not calculate SHAP interactions: {e}")
            return {}

    def _calculate_p_values(self, perm_results) -> np.ndarray:
        """Calculate p-values for permutation importance"""
        # Simple approximation: features with importance > 0 are significant
        # More sophisticated: compare to null distribution
        p_values = np.zeros(len(perm_results.importances_mean))

        for i in range(len(perm_results.importances_mean)):
            # Two-tailed test: importance significantly different from zero
            null_dist = perm_results.importances[i, :]  # Permuted importances
            observed = perm_results.importances_mean[i]

            # P-value: proportion of null distribution more extreme than observed
            p_value = np.mean(np.abs(null_dist) >= abs(observed))
            p_values[i] = p_value

        return p_values

class FeatureSelectionValidator:
    """Validate feature selection quality"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def validate_selection(self, X_full: pd.DataFrame, y: pd.Series,
                          selected_features: List[str], model_class=None) -> Dict:
        """
        Validate that selected features maintain predictive power
        """
        if model_class is None:
            from sklearn.ensemble import RandomForestRegressor
            model_class = RandomForestRegressor

        # Compare full vs selected features
        feature_sets = {
            'full_features': X_full.columns.tolist(),
            'selected_features': selected_features,
            'random_subset': np.random.choice(X_full.columns.tolist(), len(selected_features), replace=False).tolist()
        }

        analyzer = FeatureImportanceAnalyzer(random_state=self.random_state)
        comparison = analyzer.compare_feature_sets(model_class, X_full, y, feature_sets)

        # Additional validation: redundancy check
        X_selected = X_full[selected_features]
        corr_matrix = X_selected.corr().abs()
        max_corr = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool)).max().max()

        return {
            'comparison': comparison,
            'selected_max_correlation': max_corr,
            'validation_passed': (
                comparison['comparison_results'].loc['selected_features', 'cv_mse_mean'] <=
                comparison['comparison_results'].loc['full_features', 'cv_mse_mean'] * 1.1  # Allow 10% degradation
            ),
            'efficiency_gain': len(selected_features) / len(X_full.columns)
        }