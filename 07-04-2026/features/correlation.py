"""
Feature Correlation Analysis for SCAF-LS
Detect and handle multicollinearity in feature sets
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx

class CorrelationAnalyzer:
    """Advanced correlation analysis for feature selection"""

    def __init__(self, method: str = 'spearman'):
        self.method = method  # 'pearson', 'spearman', or 'kendall'
        self.correlation_matrix = None
        self.distance_matrix = None

    def analyze_correlations(self, X: pd.DataFrame, plot: bool = False) -> Dict:
        """
        Comprehensive correlation analysis
        """
        print(f"🔗 Analyzing {self.method} correlations...")

        # Calculate correlation matrix
        if self.method == 'spearman':
            corr_matrix = X.corr(method='spearman')
        elif self.method == 'kendall':
            corr_matrix = X.corr(method='kendall')
        else:
            corr_matrix = X.corr(method='pearson')

        self.correlation_matrix = corr_matrix

        # Calculate distance matrix (1 - |corr|)
        self.distance_matrix = 1 - np.abs(corr_matrix)

        # Find highly correlated pairs
        high_corr_pairs = self._find_high_corr_pairs(corr_matrix, threshold=0.95)

        # Identify correlation clusters
        clusters = self._identify_correlation_clusters(corr_matrix, threshold=0.8)

        # Calculate correlation statistics
        stats = self._calculate_correlation_stats(corr_matrix)

        results = {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'correlation_clusters': clusters,
            'correlation_stats': stats,
            'n_high_corr_pairs': len(high_corr_pairs),
            'n_clusters': len(clusters)
        }

        if plot:
            self._plot_correlation_analysis(results)

        return results

    def _find_high_corr_pairs(self, corr_matrix: pd.DataFrame,
                            threshold: float = 0.95) -> List[Dict]:
        """Find pairs of highly correlated features"""
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'method': self.method
                    })

        return sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)

    def _identify_correlation_clusters(self, corr_matrix: pd.DataFrame,
                                     threshold: float = 0.8) -> List[Dict]:
        """Identify clusters of correlated features using hierarchical clustering"""
        # Convert correlation to distance (ensure non-negative)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Ensure no negative distances (shouldn't happen with abs, but just in case)
        distance_matrix = np.maximum(distance_matrix, 0)
        
        # For small matrices or when linkage fails, use a simpler approach
        try:
            linkage_matrix = linkage(squareform(distance_matrix), method='complete')
            
            # Form clusters
            cluster_labels = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
        except Exception as e:
            print(f"Hierarchical clustering failed: {e}, using simple correlation-based clustering")
            # Fallback: simple clustering based on correlation threshold
            cluster_labels = self._simple_correlation_clustering(corr_matrix, threshold)

        # Group features by cluster
        clusters = {}
        for feature, cluster_id in zip(corr_matrix.columns, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(feature)

        # Convert to list of dicts
        cluster_list = []
        for cluster_id, features in clusters.items():
            if len(features) > 1:  # Only include clusters with multiple features
                cluster_corr = corr_matrix.loc[features, features]
                cluster_list.append({
                    'cluster_id': cluster_id,
                    'features': features,
                    'size': len(features),
                    'avg_correlation': cluster_corr.mean().mean(),
                    'max_correlation': cluster_corr.max().max()
                })

        return sorted(cluster_list, key=lambda x: x['size'], reverse=True)

    def _simple_correlation_clustering(self, corr_matrix: pd.DataFrame, threshold: float) -> np.ndarray:
        """Simple correlation-based clustering as fallback"""
        n_features = len(corr_matrix.columns)
        cluster_labels = np.arange(n_features)  # Start with each feature in its own cluster
        
        # Iteratively merge highly correlated clusters
        merged = True
        while merged:
            merged = False
            for i in range(n_features):
                for j in range(i+1, n_features):
                    if cluster_labels[i] != cluster_labels[j]:
                        # Check if any features from these clusters are highly correlated
                        cluster_i_features = corr_matrix.columns[cluster_labels == cluster_labels[i]]
                        cluster_j_features = corr_matrix.columns[cluster_labels == cluster_labels[j]]
                        
                        max_corr = 0
                        for feat_i in cluster_i_features:
                            for feat_j in cluster_j_features:
                                corr_val = abs(corr_matrix.loc[feat_i, feat_j])
                                max_corr = max(max_corr, corr_val)
                        
                        if max_corr > threshold:
                            # Merge clusters
                            old_label = cluster_labels[j]
                            cluster_labels[cluster_labels == old_label] = cluster_labels[i]
                            merged = True
                            break
                if merged:
                    break
        
        return cluster_labels

    def _calculate_correlation_stats(self, corr_matrix: pd.DataFrame) -> Dict:
        """Calculate correlation statistics"""
        abs_corr = np.abs(corr_matrix)

        # Remove diagonal
        abs_corr.values[np.diag_indices_from(abs_corr)] = np.nan

        return {
            'mean_abs_correlation': np.nanmean(abs_corr.values),
            'median_abs_correlation': np.nanmedian(abs_corr.values),
            'max_correlation': np.nanmax(abs_corr.values),
            'min_correlation': np.nanmin(abs_corr.values),
            'correlation_std': np.nanstd(abs_corr.values),
            'n_correlations_above_09': np.sum(abs_corr > 0.9),
            'n_correlations_above_08': np.sum(abs_corr > 0.8),
            'n_correlations_above_07': np.sum(abs_corr > 0.7)
        }

    def select_uncorrelated_features(self, X: pd.DataFrame, importance_scores: Dict = None,
                                   max_correlation: float = 0.8) -> List[str]:
        """
        Select maximally uncorrelated features, preferring important ones
        """
        corr_matrix = self.correlation_matrix
        if corr_matrix is None:
            self.analyze_correlations(X)

        selected_features = []
        remaining_features = list(X.columns)

        # Sort by importance if provided
        if importance_scores:
            remaining_features.sort(key=lambda x: importance_scores.get(x, 0), reverse=True)

        while remaining_features:
            # Select the most important remaining feature
            current_feature = remaining_features[0]
            selected_features.append(current_feature)
            remaining_features.remove(current_feature)

            # Remove highly correlated features
            to_remove = []
            for feature in remaining_features:
                if abs(corr_matrix.loc[current_feature, feature]) > max_correlation:
                    to_remove.append(feature)

            for feature in to_remove:
                remaining_features.remove(feature)

        return selected_features

    def build_correlation_network(self, corr_matrix: pd.DataFrame,
                                threshold: float = 0.8) -> nx.Graph:
        """Build network graph of feature correlations"""
        G = nx.Graph()

        # Add nodes
        for feature in corr_matrix.columns:
            G.add_node(feature)

        # Add edges for highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    G.add_edge(
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        weight=corr_val
                    )

        return G

    def find_redundant_groups(self, corr_matrix: pd.DataFrame,
                            importance_scores: Dict = None) -> List[Dict]:
        """
        Find groups of redundant features that can be consolidated
        """
        # Build correlation network
        G = self.build_correlation_network(corr_matrix, threshold=0.9)

        # Find connected components (redundant groups)
        redundant_groups = []
        for component in nx.connected_components(G):
            if len(component) > 1:
                features = list(component)

                # Calculate group statistics
                group_corr = corr_matrix.loc[features, features]
                avg_corr = group_corr.mean().mean()

                # Select representative feature (most important or first)
                if importance_scores:
                    representative = max(features, key=lambda x: importance_scores.get(x, 0))
                else:
                    representative = features[0]

                redundant_groups.append({
                    'features': features,
                    'representative': representative,
                    'size': len(features),
                    'avg_correlation': avg_corr,
                    'redundant_features': [f for f in features if f != representative]
                })

        return sorted(redundant_groups, key=lambda x: x['size'], reverse=True)

    def _plot_correlation_analysis(self, results: Dict):
        """Create correlation analysis plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Correlation heatmap
            sns.heatmap(results['correlation_matrix'], ax=axes[0,0],
                       cmap='coolwarm', center=0, vmin=-1, vmax=1)
            axes[0,0].set_title('Feature Correlation Matrix')

            # Correlation distribution
            abs_corr = np.abs(results['correlation_matrix'].values)
            abs_corr = abs_corr[np.triu_indices_from(abs_corr, k=1)]
            axes[0,1].hist(abs_corr, bins=50, alpha=0.7)
            axes[0,1].axvline(x=0.95, color='red', linestyle='--', label='High corr threshold')
            axes[0,1].axvline(x=0.8, color='orange', linestyle='--', label='Medium corr threshold')
            axes[0,1].set_xlabel('Absolute Correlation')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Correlation Distribution')
            axes[0,1].legend()

            # Cluster sizes
            cluster_sizes = [c['size'] for c in results['correlation_clusters']]
            if cluster_sizes:
                axes[1,0].bar(range(len(cluster_sizes)), cluster_sizes)
                axes[1,0].set_xlabel('Cluster ID')
                axes[1,0].set_ylabel('Number of Features')
                axes[1,0].set_title('Correlation Cluster Sizes')

            # High correlation pairs
            if results['high_corr_pairs']:
                pairs = results['high_corr_pairs'][:10]  # Top 10
                features = [f"{p['feature1']}-{p['feature2']}" for p in pairs]
                corrs = [p['correlation'] for p in pairs]
                axes[1,1].barh(features, corrs)
                axes[1,1].set_xlabel('Correlation')
                axes[1,1].set_title('Top 10 Highly Correlated Pairs')

            plt.tight_layout()
            plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("📊 Correlation analysis plot saved as 'correlation_analysis.png'")

        except Exception as e:
            print(f"Could not create correlation plots: {e}")

class FeatureRedundancyResolver:
    """Resolve feature redundancy by selecting optimal representatives"""

    def __init__(self, correlation_analyzer: CorrelationAnalyzer):
        self.corr_analyzer = correlation_analyzer

    def resolve_redundancy(self, X: pd.DataFrame, importance_scores: Dict = None,
                          stability_scores: Dict = None) -> Dict:
        """
        Resolve feature redundancy using multiple criteria
        """
        # Find redundant groups
        redundant_groups = self.corr_analyzer.find_redundant_groups(
            self.corr_analyzer.correlation_matrix, importance_scores
        )

        resolved_features = []
        removed_features = []

        for group in redundant_groups:
            features = group['features']

            # Score each feature in the group
            feature_scores = {}
            for feature in features:
                score = 0

                # Importance score (40%)
                if importance_scores:
                    score += 0.4 * importance_scores.get(feature, 0)

                # Stability score (30%)
                if stability_scores:
                    score += 0.3 * stability_scores.get(feature, 0)

                # Inverse correlation with others (20%) - prefer less correlated
                other_features = [f for f in features if f != feature]
                avg_corr_with_others = np.mean([
                    abs(self.corr_analyzer.correlation_matrix.loc[feature, other])
                    for other in other_features
                ])
                score += 0.2 * (1 - avg_corr_with_others)  # Lower correlation is better

                # Variance score (10%) - prefer higher variance (more information)
                variance = X[feature].var()
                normalized_variance = variance / X[feature].max() if X[feature].max() > 0 else 0
                score += 0.1 * normalized_variance

                feature_scores[feature] = score

            # Select the best feature
            best_feature = max(feature_scores.items(), key=lambda x: x[1])[0]
            resolved_features.append(best_feature)

            # Mark others as removed
            for feature in features:
                if feature != best_feature:
                    removed_features.append({
                        'feature': feature,
                        'group_representative': best_feature,
                        'reason': 'redundant',
                        'correlation_with_representative': self.corr_analyzer.correlation_matrix.loc[feature, best_feature]
                    })

        return {
            'selected_features': resolved_features,
            'removed_features': removed_features,
            'redundancy_groups': redundant_groups,
            'n_resolved_groups': len(redundant_groups),
            'total_features_removed': len(removed_features)
        }