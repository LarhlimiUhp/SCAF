"""
LightGBM-specific feature engineering for SCAF-LS

Specialized preprocessing to optimize LightGBM performance:
- Categorical encoding for ordinal features
- Monotonic constraints detection
- Feature interactions
- Missing value optimization
- Quantile binning for continuous features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif


class LightGBMFeatureEngineer:
    """LightGBM-specific feature engineering"""

    def __init__(self, monotonic_features: Optional[Dict[str, int]] = None):
        """
        Args:
            monotonic_features: Dict mapping feature names to monotonic constraints
                               1 for positive monotonic, -1 for negative monotonic
        """
        self.monotonic_features = monotonic_features or {}
        self.feature_interactions = []
        self.categorical_features = []
        self.binned_features = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features for LightGBM"""
        X_processed = X.copy()

        # 1. Categorical encoding for ordinal features
        X_processed = self._encode_categorical_features(X_processed)

        # 2. Add feature interactions
        X_processed = self._add_feature_interactions(X_processed)

        # 3. Quantile binning for continuous features
        X_processed = self._add_quantile_binning(X_processed)

        # 4. Handle missing values optimally for LGBM
        X_processed = self._handle_missing_values(X_processed)

        # 5. Add temporal features
        X_processed = self._add_temporal_features(X_processed)

        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters"""
        X_processed = X.copy()

        # Apply same transformations
        X_processed = self._encode_categorical_features(X_processed)
        X_processed = self._add_feature_interactions(X_processed)
        X_processed = self._add_quantile_binning(X_processed)
        X_processed = self._handle_missing_values(X_processed)
        X_processed = self._add_temporal_features(X_processed)

        return X_processed

    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical/ordinal features for LightGBM"""
        # Financial ordinal features
        ordinal_mappings = {
            'vix_quartile': {'low': 0, 'medium': 1, 'high': 2, 'extreme': 3},
            'vol_regime': {'low': 0, 'normal': 1, 'high': 2, 'crisis': 3},
            'yield_trend': {'falling': -1, 'stable': 0, 'rising': 1},
            'credit_trend': {'improving': -1, 'stable': 0, 'deteriorating': 1}
        }

        for feature, mapping in ordinal_mappings.items():
            if feature in X.columns:
                X[feature] = X[feature].map(mapping).fillna(0).astype(int)
                self.categorical_features.append(feature)

        return X

    def _add_feature_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add meaningful feature interactions for LightGBM"""

        # Risk-adjusted momentum
        if 'spx_momentum' in X.columns and 'vix_zscore' in X.columns:
            X['momentum_risk_adjusted'] = X['spx_momentum'] / (X['vix_zscore'] + 1e-8)
            self.feature_interactions.append('momentum_risk_adjusted')

        # Yield curve steepness
        if 'yield_10y' in X.columns and 'yield_2y' in X.columns:
            X['yield_curve_steepness'] = X['yield_10y'] - X['yield_2y']
            self.feature_interactions.append('yield_curve_steepness')

        # Credit spreads relative to yields
        if 'credit_spread' in X.columns and 'yield_10y' in X.columns:
            X['credit_risk_premium'] = X['credit_spread'] / (X['yield_10y'] + 1e-8)
            self.feature_interactions.append('credit_risk_premium')

        # RSI and volatility interaction
        if 'spx_rsi' in X.columns and 'realized_vol_20' in X.columns:
            X['rsi_vol_interaction'] = X['spx_rsi'] * X['realized_vol_20']
            self.feature_interactions.append('rsi_vol_interaction')

        # Commodity momentum relative to SPX
        if 'gold_momentum' in X.columns and 'spx_momentum' in X.columns:
            X['gold_relative_momentum'] = X['gold_momentum'] - X['spx_momentum']
            self.feature_interactions.append('gold_relative_momentum')

        return X

    def _add_quantile_binning(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add quantile binning for continuous features"""
        continuous_features = [
            'spx_rsi', 'vix_zscore', 'yield_10y', 'credit_spread',
            'spx_momentum', 'realized_vol_20'
        ]

        for feature in continuous_features:
            if feature in X.columns:
                # Create quantile bins
                binned_feature = f'{feature}_quantile'
                try:
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                    X[binned_feature] = discretizer.fit_transform(X[[feature]]).astype(int)
                    self.binned_features.append(binned_feature)
                except:
                    # Fallback for features with insufficient variation
                    X[binned_feature] = pd.qcut(X[feature], q=5, labels=False, duplicates='drop').fillna(0)

        return X

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values optimally for LightGBM"""
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['float64', 'int64']:
                    # For numerical features, use median (LightGBM handles NaN well, but this ensures consistency)
                    if X[col].isnull().sum() / len(X) < 0.1:  # Less than 10% missing
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        # For high missing rate, create missing indicator
                        X[f'{col}_missing'] = X[col].isnull().astype(int)
                        X[col] = X[col].fillna(0)
                else:
                    # For categorical features
                    X[col] = X[col].fillna('missing')
                    if col not in self.categorical_features:
                        self.categorical_features.append(col)

        return X

    def _add_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for time series patterns"""
        if isinstance(X.index, pd.DatetimeIndex):
            # Cyclical encoding for temporal features
            X['day_of_year_sin'] = np.sin(2 * np.pi * X.index.dayofyear / 365.25)
            X['day_of_year_cos'] = np.cos(2 * np.pi * X.index.dayofyear / 365.25)

            X['month_sin'] = np.sin(2 * np.pi * X.index.month / 12)
            X['month_cos'] = np.cos(2 * np.pi * X.index.month / 12)

            # Quarter and business cycle features
            X['quarter'] = X.index.quarter
            X['is_quarter_end'] = X.index.is_quarter_end.astype(int)
            X['is_month_end'] = X.index.is_month_end.astype(int)

        return X

    def get_monotonic_constraints(self) -> Dict[str, int]:
        """Get monotonic constraints for LightGBM"""
        constraints = self.monotonic_features.copy()

        # Add domain knowledge constraints
        domain_constraints = {
            'spx_rsi': 1,  # Higher RSI should increase positive return probability
            'vix_zscore': -1,  # Higher VIX should decrease positive return probability
            'yield_10y': -1,  # Higher yields might indicate weaker economy
            'credit_spread': -1,  # Wider spreads indicate higher risk
            'yield_curve_steepness': 1,  # Steeper curve is positive for stocks
        }

        constraints.update(domain_constraints)
        return constraints

    def get_feature_info(self) -> Dict[str, List[str]]:
        """Get information about engineered features"""
        return {
            'categorical_features': self.categorical_features,
            'feature_interactions': self.feature_interactions,
            'binned_features': self.binned_features,
            'monotonic_constraints': list(self.get_monotonic_constraints().keys())
        }


class LightGBMFeatureSelector:
    """Feature selection optimized for LightGBM"""

    def __init__(self, n_features: int = 22, use_gain_importance: bool = True):
        self.n_features = n_features
        self.use_gain_importance = use_gain_importance
        self.selected_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series, lgbm_params: Optional[Dict] = None) -> 'LightGBMFeatureSelector':
        """Fit feature selector using LightGBM importance"""
        try:
            import lightgbm as lgb

            # Default parameters for feature selection
            default_params = {
                'objective': 'binary',
                'verbosity': -1,
                'n_jobs': -1,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31
            }

            params = lgbm_params or default_params
            model = lgb.LGBMClassifier(**params)
            model.fit(X.values, y.values)

            # Get feature importance
            if self.use_gain_importance:
                importance = model.feature_importances_
            else:
                # Use split importance
                importance = model.booster_.feature_importance(importance_type='split')

            # Select top features
            feature_indices = np.argsort(importance)[-self.n_features:]
            self.selected_features = X.columns[feature_indices].tolist()

        except ImportError:
            # Fallback to mutual information
            mi_scores = mutual_info_classif(X.values, y.values)
            feature_indices = np.argsort(mi_scores)[-self.n_features:]
            self.selected_features = X.columns[feature_indices].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features"""
        available_features = [f for f in self.selected_features if f in X.columns]
        return X[available_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform"""
        return self.fit(X, y).transform(X)