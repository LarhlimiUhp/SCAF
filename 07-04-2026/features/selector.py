"""
SCAF-LS Feature Selection System
Advanced feature analysis and selection using SHAP + permutation importance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import shap
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """Advanced feature analyzer for SCAF-LS"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.correlation_matrix = None
        self.shap_values = None
        self.permutation_importance = None
        self.feature_scores = {}

    def analyze_correlations(self, X: pd.DataFrame, threshold: float = 0.95) -> Dict:
        """Analyze feature correlations and identify highly correlated pairs"""
        corr_matrix = X.corr().abs()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        self.correlation_matrix = corr_matrix

        return {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True),
            'n_high_corr_pairs': len(high_corr_pairs),
            'max_correlation': corr_matrix.max().max() if not corr_matrix.empty else 0
        }

    def calculate_shap_importance(self, X: pd.DataFrame, y: pd.Series,
                                 model=None, max_evals: int = 1000) -> Dict:
        """Calculate SHAP feature importance"""
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)

        # Fit model
        model.fit(X, y)

        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Aggregate SHAP importance
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)

        self.shap_values = shap_values

        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'top_features': importance_df.head(25)['feature'].tolist()
        }

    def calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                                       model=None, cv: int = 5) -> Dict:
        """Calculate permutation feature importance"""
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)

        # Fit model
        model.fit(X, y)

        # Calculate permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=self.random_state)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        self.permutation_importance = perm_importance

        return {
            'permutation_importance': perm_importance,
            'feature_importance': importance_df,
            'top_features': importance_df.head(25)['feature'].tolist()
        }

    def calculate_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Calculate mutual information between features and target"""
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)

        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        return {
            'mutual_info_scores': mi_df,
            'top_features': mi_df.head(25)['feature'].tolist()
        }

    def detect_redundant_features(self, X: pd.DataFrame, corr_threshold: float = 0.95,
                                importance_threshold: float = 0.01) -> Dict:
        """Detect redundant and low-importance features"""
        # Get correlations
        corr_analysis = self.analyze_correlations(X, corr_threshold)

        # Get importance scores (use permutation if available, else mutual info)
        if self.permutation_importance is not None:
            importance_scores = pd.Series(
                self.permutation_importance.importances_mean,
                index=X.columns
            )
        else:
            mi_analysis = self.calculate_mutual_info(X, X.iloc[:, 0])  # dummy target
            importance_scores = mi_analysis['mutual_info_scores'].set_index('feature')['mutual_info']

        # Identify redundant features (highly correlated)
        redundant_features = set()
        for pair in corr_analysis['high_corr_pairs']:
            feat1, feat2 = pair['feature1'], pair['feature2']
            # Keep the more important one
            if importance_scores[feat1] >= importance_scores[feat2]:
                redundant_features.add(feat2)
            else:
                redundant_features.add(feat1)

        # Identify low-importance features
        low_importance = importance_scores[importance_scores < importance_threshold].index.tolist()

        # Identify noisy features (high variance but low importance)
        feature_std = X.std()
        noise_candidates = []
        for feature in X.columns:
            if feature_std[feature] > feature_std.quantile(0.75):  # High variance
                if importance_scores[feature] < importance_scores.quantile(0.25):  # Low importance
                    noise_candidates.append(feature)

        return {
            'redundant_features': list(redundant_features),
            'low_importance_features': low_importance,
            'noise_candidates': noise_candidates,
            'features_to_remove': list(set(redundant_features) | set(low_importance) | set(noise_candidates))
        }

    def select_optimal_features(self, X: pd.DataFrame, y: pd.Series,
                              target_n_features: int = 25) -> Dict:
        """Select optimal feature subset using multiple criteria"""

        # Calculate all importance metrics
        shap_results = self.calculate_shap_importance(X, y)
        perm_results = self.calculate_permutation_importance(X, y)
        mi_results = self.calculate_mutual_info(X, y)

        # Combine scores with weights
        feature_scores = {}
        for feature in X.columns:
            shap_score = shap_results['feature_importance'].set_index('feature').loc[feature, 'shap_importance']
            perm_score = perm_results['feature_importance'].set_index('feature').loc[feature, 'importance_mean']
            mi_score = mi_results['mutual_info_scores'].set_index('feature').loc[feature, 'mutual_info']

            # Weighted combination
            combined_score = 0.4 * shap_score + 0.4 * perm_score + 0.2 * mi_score
            feature_scores[feature] = combined_score

        # Sort by combined score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top features
        selected_features = [f[0] for f in sorted_features[:target_n_features]]

        # Check for correlations in selected set
        selected_X = X[selected_features]
        selected_corr = self.analyze_correlations(selected_X, threshold=0.9)

        # Remove highly correlated from selection if needed
        final_features = selected_features.copy()
        for pair in selected_corr['high_corr_pairs']:
            if pair['correlation'] > 0.95:
                # Remove the less important one
                feat1, feat2 = pair['feature1'], pair['feature2']
                if feature_scores[feat1] >= feature_scores[feat2]:
                    if feat2 in final_features:
                        final_features.remove(feat2)
                else:
                    if feat1 in final_features:
                        final_features.remove(feat1)

        return {
            'selected_features': final_features,
            'feature_scores': dict(sorted_features),
            'correlation_analysis': selected_corr,
            'n_selected': len(final_features),
            'importance_methods': {
                'shap': shap_results,
                'permutation': perm_results,
                'mutual_info': mi_results
            }
        }

class AutomatedFeatureSelector:
    """Automated feature selection pipeline for SCAF-LS"""

    def __init__(self, analyzer: FeatureAnalyzer = None):
        self.analyzer = analyzer or FeatureAnalyzer()
        self.selected_features = None
        self.selection_report = None

    def run_full_analysis(self, X: pd.DataFrame, y: pd.Series,
                         target_n_features: int = 25) -> Dict:
        """Run complete feature analysis and selection"""

        print(f"[ANALYZE] Starting feature analysis on {X.shape[1]} features...")

        # Step 1: Correlation analysis
        print("[CORRELATIONS] Analyzing correlations...")
        corr_results = self.analyzer.analyze_correlations(X)
        print(f"Found {corr_results['n_high_corr_pairs']} highly correlated pairs (>0.95)")

        # Step 2: Feature importance analysis
        print("[IMPORTANCE] Calculating feature importance...")
        importance_results = self.analyzer.select_optimal_features(X, y, target_n_features)

        # Step 3: Redundancy detection
        print("[REDUNDANCY] Detecting redundant features...")
        redundancy_results = self.analyzer.detect_redundant_features(X)

        # Step 4: Final selection
        selected_features = importance_results['selected_features']

        # Ensure we don't have too many correlated features in final selection
        final_X = X[selected_features]
        final_corr = self.analyzer.analyze_correlations(final_X, threshold=0.9)

        if final_corr['n_high_corr_pairs'] > 0:
            print(f"[WARNING] Found {final_corr['n_high_corr_pairs']} correlated pairs in selection, refining...")
            # Remove most correlated features
            to_remove = set()
            for pair in final_corr['high_corr_pairs']:
                if pair['correlation'] > 0.95:
                    feat1_score = importance_results['feature_scores'].get(pair['feature1'], 0)
                    feat2_score = importance_results['feature_scores'].get(pair['feature2'], 0)
                    if feat1_score <= feat2_score:
                        to_remove.add(pair['feature1'])
                    else:
                        to_remove.add(pair['feature2'])

            selected_features = [f for f in selected_features if f not in to_remove]

        self.selected_features = selected_features

        # Generate comprehensive report
        report = {
            'original_n_features': X.shape[1],
            'selected_n_features': len(selected_features),
            'selected_features': selected_features,
            'correlation_analysis': corr_results,
            'importance_analysis': importance_results,
            'redundancy_analysis': redundancy_results,
            'final_correlation_check': final_corr,
            'reduction_ratio': len(selected_features) / X.shape[1],
            'expected_performance_improvement': self._estimate_performance_gain(
                corr_results, redundancy_results
            )
        }

        self.selection_report = report

        print(f"[OK] Feature selection complete!")
        print(f"Selected {len(selected_features)}/{X.shape[1]} features ({report['reduction_ratio']:.1%})")
        print(f"Expected performance improvement: {report['expected_performance_improvement']:.1%}")

        return report

    def _estimate_performance_gain(self, corr_results: Dict, redundancy_results: Dict) -> float:
        """
        Estimate expected performance improvement using empirical AUC train/test gap.

        A smaller train-test AUC gap on the selected feature subset (compared to
        the full set) indicates reduced overfitting.  We return the reduction in
        that gap, capped at 50 %.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        if self.selection_report is None:
            return 0.0

        # We need the raw X/y to compute the gap; they are stored temporarily
        # by run_full_analysis via importance_results
        try:
            importance_results = self.selection_report.get('importance_analysis', {})
            shap_info = importance_results.get('importance_methods', {}).get('shap', {})
            shap_obj = shap_info.get('shap_values')
            if shap_obj is None:
                return 0.0

            # Recover X from SHAP object (TreeExplainer stores .data)
            if hasattr(shap_obj, 'data') and shap_obj.data is not None:
                X_full = pd.DataFrame(shap_obj.data) if not isinstance(shap_obj.data, pd.DataFrame) else shap_obj.data
                y_proxy = X_full.iloc[:, 0].rank().gt(X_full.iloc[:, 0].median()).astype(int)
            else:
                return 0.0

            X_sel = X_full[self.selected_features] if self.selected_features else X_full
            X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y_proxy, test_size=0.3, shuffle=False)

            clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
            clf.fit(X_tr.values, y_tr.values)

            auc_train = roc_auc_score(y_tr, clf.predict_proba(X_tr.values)[:, 1])
            auc_val = roc_auc_score(y_val, clf.predict_proba(X_val.values)[:, 1])
            gap = max(0.0, auc_train - auc_val)

            # Return estimated improvement capped at 50 %
            return min(gap, 0.5)

        except Exception:
            return 0.0

    def get_selected_features(self) -> List[str]:
        """Get the selected feature names"""
        return self.selected_features or []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features only"""
        if self.selected_features is None:
            raise ValueError("No features selected. Run run_full_analysis first.")
        return X[self.selected_features]

    def save_report(self, filepath: str):
        """Save selection report to file"""
        if self.selection_report is None:
            raise ValueError("No report available. Run analysis first.")

        import json
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, bool):
                return obj
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, '__class__') and 'shap' in str(obj.__class__).lower():
                # Skip SHAP objects for serialization
                return f"<SHAP {obj.__class__.__name__} object - not serializable>"
            elif hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__).lower():
                # Skip sklearn objects
                return f"<{obj.__class__.__name__} object - not serializable>"
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_report = convert_to_serializable(self.selection_report)

        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)

        print(f"📄 Report saved to {filepath}")