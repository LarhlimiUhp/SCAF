"""
Analyseur de résultats de validation robuste
Génère rapports détaillés et recommandations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ValidationReportGenerator:
    """
    Génère des rapports détaillés à partir des résultats de validation
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.thresholds = self._get_thresholds()

    def _get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Seuils de validation"""
        return {
            'performance': {
                'auc_min': 0.55,
                'sharpe_min': 0.5,
                'max_drawdown_max': 0.25,
                'stability_min': 0.7
            },
            'robustness': {
                'bull_bear_ratio_min': 0.8,
                'crisis_recovery_max_days': 90,
                'feature_drift_max': 0.3,
                'concept_drift_max': 0.4
            },
            'sensitivity': {
                'hyperparam_variance_max': 0.2,
                'data_robustness_min': 0.85
            },
            'realism': {
                'slippage_tolerance_max': 0.05,
                'latency_tolerance_max': 0.03,
                'fees_impact_max': 0.1
            }
        }

    def generate_comprehensive_report(self) -> str:
        """Génère un rapport complet"""
        report = []
        report.append("# 📊 RAPPORT DE VALIDATION ROBUSTE SCAF-LS")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Score global
        summary = self.results.get('summary', {})
        overall_score = summary.get('overall_score', 0)
        report.append(f"## 🎯 Score Global: {overall_score:.3f}")
        report.append(self._interpret_overall_score(overall_score))
        report.append("")

        # 1. Walk-forward validation
        report.append("## 📅 1. Walk-forward Validation Étendue (2005-2024)")
        report.extend(self._report_walk_forward())
        report.append("")

        # 2. Tests de robustesse
        report.append("## 📈 2. Tests de Robustesse")
        report.extend(self._report_robustness())
        report.append("")

        # 3. Validation multi-horizons
        report.append("## ⏰ 3. Validation Multi-Horizons")
        report.extend(self._report_multi_horizon())
        report.append("")

        # 4. Tests de stabilité
        report.append("## 🔄 4. Tests de Stabilité")
        report.extend(self._report_stability())
        report.append("")

        # 5. Validation out-of-sample
        report.append("## 🎯 5. Validation Out-of-Sample Étendue")
        report.extend(self._report_out_of_sample())
        report.append("")

        # 6. Tests de sensibilité
        report.append("## ⚙️ 6. Tests de Sensibilité")
        report.extend(self._report_sensitivity())
        report.append("")

        # 7. Validation statistique
        report.append("## 📊 7. Validation Statistique Rigoureuse")
        report.extend(self._report_statistical())
        report.append("")

        # 8. Backtesting réaliste
        report.append("## 💰 8. Backtesting avec Réalisme")
        report.extend(self._report_realistic_backtest())
        report.append("")

        # Recommandations
        report.append("## 💡 Recommandations")
        report.extend(self._generate_recommendations())

        return "\n".join(report)

    def _interpret_overall_score(self, score: float) -> str:
        """Interprète le score global"""
        if score >= 0.9:
            return "🟢 **EXCELLENT** - Modèle prêt pour production"
        elif score >= 0.8:
            return "🟡 **BON** - Modèle viable avec quelques améliorations"
        elif score >= 0.7:
            return "🟠 **MOYEN** - Améliorations nécessaires avant production"
        elif score >= 0.6:
            return "🔴 **FAIBLE** - Retravail significatif requis"
        else:
            return "❌ **CRITIQUE** - Modèle non viable"

    def _report_walk_forward(self) -> List[str]:
        """Rapport walk-forward"""
        wf = self.results.get('walk_forward', {})
        windows = wf.get('windows', [])
        summary = wf.get('summary', {})

        report = []
        report.append(f"- **Fenêtres testées:** {summary.get('total_windows', 0)}")

        if windows:
            # Statistiques des métriques
            aucs = [w['metrics'].auc for w in windows if hasattr(w['metrics'], 'auc')]
            sharpes = [w['metrics'].sharpe_ratio for w in windows if hasattr(w['metrics'], 'sharpe_ratio')]

            if aucs:
                report.append(f"- **AUC moyen:** {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
            if sharpes:
                report.append(f"- **Sharpe moyen:** {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")

            # Stabilité temporelle
            auc_threshold = self.thresholds['performance']['auc_min']
            stable_windows = sum(1 for auc in aucs if auc >= auc_threshold)
            stability_pct = stable_windows / len(aucs) if aucs else 0
            report.append(f"- **Stabilité temporelle:** {stability_pct:.1%} fenêtres ≥ {auc_threshold}")

        return report

    def _report_robustness(self) -> List[str]:
        """Rapport robustesse"""
        robustness = self.results.get('robustness', {})
        report = []

        regimes = ['bull', 'bear', 'crisis', 'high_vol', 'low_vol']
        for regime in regimes:
            if regime in robustness:
                metrics = robustness[regime]
                if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
                    sharpe = metrics['sharpe_ratio']
                    status = "🟢" if sharpe > 0 else "🔴"
                    report.append(f"- **{regime.title()}:** {status} Sharpe = {sharpe:.3f}")

        # Ratio bull/bear
        if 'bull' in robustness and 'bear' in robustness:
            bull_sharpe = robustness['bull'].get('sharpe_ratio', 0)
            bear_sharpe = robustness['bear'].get('sharpe_ratio', 0)
            if bear_sharpe != 0:
                ratio = bull_sharpe / abs(bear_sharpe)
                threshold = self.thresholds['robustness']['bull_bear_ratio_min']
                status = "🟢" if ratio >= threshold else "🔴"
                report.append(f"- **Ratio Bull/Bear:** {status} {ratio:.2f} (min: {threshold})")

        return report

    def _report_multi_horizon(self) -> List[str]:
        """Rapport multi-horizon"""
        multi_horizon = self.results.get('multi_horizon', {})
        report = []

        horizons = ['1d', '5d', '10d']
        for horizon in horizons:
            if horizon in multi_horizon:
                metrics = multi_horizon[horizon]
                if isinstance(metrics, dict) and 'auc' in metrics:
                    auc = metrics['auc']
                    status = "🟢" if auc >= 0.55 else "🔴"
                    report.append(f"- **{horizon}:** {status} AUC = {auc:.3f}")

        return report

    def _report_stability(self) -> List[str]:
        """Rapport stabilité"""
        stability = self.results.get('stability', {})
        report = []

        # Feature drift
        feature_drift = stability.get('feature_drift', {})
        drift_mean = feature_drift.get('mean', 0)
        drift_max = feature_drift.get('max', 0)
        threshold = self.thresholds['robustness']['feature_drift_max']

        status = "🟢" if drift_max <= threshold else "🔴"
        report.append(f"- **Feature Drift:** {status} Moy = {drift_mean:.3f}, Max = {drift_max:.3f} (max: {threshold})")

        # Concept drift
        concept_drift = stability.get('concept_drift', {})
        concept_mean = concept_drift.get('mean', 0)
        concept_max = concept_drift.get('max', 0)
        threshold = self.thresholds['robustness']['concept_drift_max']

        status = "🟢" if concept_max <= threshold else "🔴"
        report.append(f"- **Concept Drift:** {status} Moy = {concept_mean:.3f}, Max = {concept_max:.3f} (max: {threshold})")

        return report

    def _report_out_of_sample(self) -> List[str]:
        """Rapport out-of-sample"""
        oos = self.results.get('out_of_sample', {})
        report = []

        scenarios = ['recent_crisis', 'dot_com_bubble', 'financial_crisis', 'taper_tantrum', 'recent_years']
        for scenario in scenarios:
            if scenario in oos:
                metrics = oos[scenario]
                if isinstance(metrics, dict) and 'auc' in metrics:
                    auc = metrics['auc']
                    status = "🟢" if auc >= 0.55 else "🔴"
                    scenario_name = scenario.replace('_', ' ').title()
                    report.append(f"- **{scenario_name}:** {status} AUC = {auc:.3f}")

        return report

    def _report_sensitivity(self) -> List[str]:
        """Rapport sensibilité"""
        sensitivity = self.results.get('sensitivity', {})
        report = []

        # Hyperparamètres
        hyper = sensitivity.get('hyperparameters', {})
        variance = hyper.get('variance', 0)
        threshold = self.thresholds['sensitivity']['hyperparam_variance_max']
        status = "🟢" if variance <= threshold else "🔴"
        report.append(f"- **Variance Hyperparam:** {status} {variance:.3f} (max: {threshold})")

        # Données
        data = sensitivity.get('data', {})
        robustness = data.get('robustness', 0)
        threshold = self.thresholds['sensitivity']['data_robustness_min']
        status = "🟢" if robustness >= threshold else "🔴"
        report.append(f"- **Robustesse Données:** {status} {robustness:.3f} (min: {threshold})")

        return report

    def _report_statistical(self) -> List[str]:
        """Rapport statistique"""
        statistical = self.results.get('statistical', {})
        report = []

        tests = ['normality', 'autocorrelation', 'stationarity', 'independence']
        for test in tests:
            if test in statistical:
                test_results = statistical[test]
                if isinstance(test_results, dict):
                    # Évaluation simplifiée
                    p_value = list(test_results.values())[0] if test_results else 1.0
                    status = "🟢" if p_value > 0.05 else "🟡"  # p > 0.05 = OK
                    report.append(f"- **{test.title()}:** {status} p-value = {p_value:.3f}")

        return report

    def _report_realistic_backtest(self) -> List[str]:
        """Rapport backtesting réaliste"""
        realistic = self.results.get('realistic_backtest', {})
        report = []

        if realistic:
            # Impact des frais moyen
            impacts = []
            for config, results in realistic.items():
                if isinstance(results, dict) and 'impact' in results:
                    impacts.append(results['impact'])

            if impacts:
                avg_impact = np.mean(impacts)
                threshold = self.thresholds['realism']['fees_impact_max']
                status = "🟢" if avg_impact <= threshold else "🔴"
                report.append(f"- **Impact Frais Moyen:** {status} {avg_impact:.1%} (max: {threshold:.1%})")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []

        # Analyser chaque section pour recommandations
        summary = self.results.get('summary', {})
        overall_score = summary.get('overall_score', 0)

        if overall_score < 0.7:
            recommendations.append("🔴 **CRITIQUE:** Retravail fondamental du modèle requis")
            recommendations.append("   - Réviser la stratégie de feature engineering")
            recommendations.append("   - Tester d'autres architectures de modèle")
            recommendations.append("   - Améliorer la gestion du risque")

        elif overall_score < 0.8:
            recommendations.append("🟡 **AMÉLIORATIONS:** Optimisations recommandées")
            recommendations.append("   - Régularisation accrue pour réduire l'overfitting")
            recommendations.append("   - Amélioration de la stabilité temporelle")
            recommendations.append("   - Réduction de la sensibilité aux hyperparamètres")

        else:
            recommendations.append("🟢 **VALIDATION RÉUSSIE:** Modèle prêt pour production")
            recommendations.append("   - Monitoring continu des drifts")
            recommendations.append("   - Mise en place de re-training automatique")
            recommendations.append("   - Tests de stress réguliers")

        # Recommandations spécifiques par section
        stability = self.results.get('stability', {})
        feature_drift = stability.get('feature_drift', {}).get('max', 0)
        if feature_drift > self.thresholds['robustness']['feature_drift_max']:
            recommendations.append("   - ⚠️ Feature drift détecté: recalibrage fréquent nécessaire")

        concept_drift = stability.get('concept_drift', {}).get('max', 0)
        if concept_drift > self.thresholds['robustness']['concept_drift_max']:
            recommendations.append("   - ⚠️ Concept drift détecté: adaptation du modèle requise")

        return recommendations

    def save_report(self, filename: str = "validation_report.md"):
        """Sauvegarde le rapport"""
        report = self.generate_comprehensive_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Rapport sauvegardé: {filename}")

    def generate_plots(self, output_dir: str = "validation_plots"):
        """Génère des graphiques de validation"""
        Path(output_dir).mkdir(exist_ok=True)

        # Plot walk-forward performance
        self._plot_walk_forward(f"{output_dir}/walk_forward.png")

        # Plot robustness across regimes
        self._plot_robustness(f"{output_dir}/robustness.png")

        # Plot stability metrics
        self._plot_stability(f"{output_dir}/stability.png")

        print(f"📊 Graphiques sauvegardés dans {output_dir}/")

    def _plot_walk_forward(self, filename: str):
        """Plot performance walk-forward"""
        wf = self.results.get('walk_forward', {})
        windows = wf.get('windows', [])

        if not windows:
            return

        dates = [w['window_start'] for w in windows]
        aucs = [w['metrics'].auc for w in windows]
        sharpes = [w['metrics'].sharpe_ratio for w in windows]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(dates, aucs, 'b-o', label='AUC')
        ax1.axhline(y=self.thresholds['performance']['auc_min'], color='r', linestyle='--', label='Seuil')
        ax1.set_title('Walk-forward AUC')
        ax1.legend()

        ax2.plot(dates, sharpes, 'g-o', label='Sharpe Ratio')
        ax2.axhline(y=self.thresholds['performance']['sharpe_min'], color='r', linestyle='--', label='Seuil')
        ax2.set_title('Walk-forward Sharpe Ratio')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_robustness(self, filename: str):
        """Plot robustness across regimes"""
        robustness = self.results.get('robustness', {})

        regimes = []
        sharpes = []

        for regime in ['bull', 'bear', 'crisis', 'high_vol', 'low_vol']:
            if regime in robustness:
                metrics = robustness[regime]
                if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
                    regimes.append(regime.title())
                    sharpes.append(metrics['sharpe_ratio'])

        if not regimes:
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(regimes, sharpes, color=['green' if s > 0 else 'red' for s in sharpes])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Performance par Régime de Marché')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)

        for bar, sharpe in zip(bars, sharpes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    '.2f', ha='center', va='bottom' if sharpe > 0 else 'top')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_stability(self, filename: str):
        """Plot stability metrics"""
        stability = self.results.get('stability', {})

        feature_drift = stability.get('feature_drift', {}).get('scores', [])
        concept_drift = stability.get('concept_drift', {}).get('scores', [])

        if not feature_drift and not concept_drift:
            return

        plt.figure(figsize=(12, 6))

        if feature_drift:
            plt.subplot(1, 2, 1)
            plt.plot(feature_drift, 'b-o', label='Feature Drift')
            plt.axhline(y=self.thresholds['robustness']['feature_drift_max'],
                       color='r', linestyle='--', label='Seuil')
            plt.title('Évolution Feature Drift')
            plt.xlabel('Période')
            plt.ylabel('Score de Drift')
            plt.legend()

        if concept_drift:
            plt.subplot(1, 2, 2)
            plt.plot(concept_drift, 'r-o', label='Concept Drift')
            plt.axhline(y=self.thresholds['robustness']['concept_drift_max'],
                       color='r', linestyle='--', label='Seuil')
            plt.title('Évolution Concept Drift')
            plt.xlabel('Période')
            plt.ylabel('Score de Drift')
            plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def generate_validation_report(results: Dict[str, Any], output_dir: str = ".") -> str:
    """
    Fonction principale pour générer le rapport de validation
    """
    generator = ValidationReportGenerator(results)

    # Rapport texte
    report_file = f"{output_dir}/validation_report.md"
    generator.save_report(report_file)

    # Graphiques
    plots_dir = f"{output_dir}/validation_plots"
    generator.generate_plots(plots_dir)

    return report_file