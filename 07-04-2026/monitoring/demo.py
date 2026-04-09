"""
Démarrage rapide du Monitoring & Maintenance SCAF-LS
"""

import sys
from pathlib import Path

# Ajouter le project root au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scaf_ls.monitoring import (
    start_monitoring,
    stop_monitoring,
    get_monitoring_service,
)
from scaf_ls.monitoring.logger import get_logger
from scaf_ls.monitoring.app_metrics import track_latency
from scaf_ls.config import Config
import json


def print_header(title):
    """Afficher un header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_system_monitoring():
    """Démonstration du monitoring système"""
    print_header("1. MONITORING SYSTÈME")
    
    monitoring = get_monitoring_service()
    
    # Collecter les métriques système
    from scaf_ls.monitoring.system_metrics import SystemMetricsCollector
    collector = SystemMetricsCollector()
    metrics = collector.collect()
    
    print(f"📊 Métriques Système:")
    print(f"  CPU:    {metrics.cpu_percent:.1f}%")
    print(f"  Mémoire: {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.0f} MB)")
    print(f"  Disque:  {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f} GB utilisé)")
    
    if metrics.gpu_available:
        print(f"  GPU:    {metrics.gpu_percent:.1f}% ({metrics.gpu_memory_used_mb:.0f} MB)")
    else:
        print(f"  GPU:    Non disponible")
    
    print(f"\n✅ Système en bon état")


def demo_latency_monitoring():
    """Démonstration du monitoring de latence"""
    print_header("2. MONITORING APPLICATIF - LATENCE")
    
    import time
    from scaf_ls.monitoring.app_metrics import get_collector
    
    # Simuler des opérations avec latence
    @track_latency("data_loading")
    def slow_operation():
        time.sleep(0.05)
    
    print("Simulation de 10 opérations...")
    for i in range(10):
        slow_operation()
    
    collector = get_collector()
    latency = collector.get_latency_metric("data_loading")
    
    if latency:
        print(f"\n📊 Métriques de Latence (data_loading):")
        print(f"  Appels:      {latency.count}")
        print(f"  Moyenne:     {latency.avg:.2f} ms")
        print(f"  P50:         {latency.percentile_p50:.2f} ms")
        print(f"  P95:         {latency.percentile_p95:.2f} ms")
        print(f"  P99:         {latency.percentile_p99:.2f} ms")
        print(f"  Min/Max:     {latency.min:.2f} / {latency.max:.2f} ms")
    
    print(f"\n✅ Latence acceptable")


def demo_business_monitoring():
    """Démonstration du monitoring métier"""
    print_header("3. MONITORING MÉTIER - P&L & TRADING")
    
    monitoring = get_monitoring_service()
    business = monitoring.business_metrics
    
    # Simuler des trades
    print("Simulation de 10 trades...")
    
    equity = 100000
    trades = [
        (100.0, 102.0, 100, 'long'),   # Win +200
        (100.0, 99.5, 100, 'long'),    # Loss -50
        (100.0, 101.5, 50, 'long'),    # Win +75
        (100.0, 98.0, 100, 'long'),    # Loss -200
        (100.0, 101.0, 100, 'long'),   # Win +100
        (100.0, 99.8, 100, 'long'),    # Loss -20
        (100.0, 102.5, 100, 'long'),   # Win +250
        (100.0, 100.2, 100, 'long'),   # Win +20
        (100.0, 99.0, 100, 'long'),    # Loss -100
        (100.0, 101.8, 100, 'long'),   # Win +180
    ]
    
    for entry, exit, qty, side in trades:
        pnl = (exit - entry) * qty if side == 'long' else (entry - exit) * qty
        equity += pnl
        business.record_trade(entry, exit, qty, side)
    
    business.update_equity(equity)
    
    # Afficher les métriques
    portfolio = business.get_portfolio_metrics()
    trades_metrics = business.get_trade_metrics()
    
    print(f"\n📊 Métriques de Trading:")
    print(f"  Initial:       ${business.initial_equity:,.2f}")
    print(f"  Équité finale: ${portfolio.current_equity:,.2f}")
    print(f"  Return:        {portfolio.total_return:.2f}%")
    print(f"\n  Total trades:  {trades_metrics.total_trades}")
    print(f"  Winning:       {trades_metrics.winning_trades}")
    print(f"  Losing:        {trades_metrics.losing_trades}")
    print(f"  Win rate:      {trades_metrics.win_rate:.1f}%")
    print(f"  Profit factor: {trades_metrics.profit_factor:.2f}")
    print(f"\n  Sharpe ratio:  {portfolio.sharpe_ratio:.2f}")
    print(f"  Max drawdown:  {portfolio.max_drawdown:.2f}%")
    
    print(f"\n✅ Trading en bon état")


def demo_alerts():
    """Démonstration du système d'alertes"""
    print_header("4. SYSTÈME D'ALERTES & ANOMALIES")
    
    monitoring = get_monitoring_service()
    alerts = monitoring.alerts
    
    # Générer quelques alertes
    print("Génération d'alertes de test...")
    
    alerts.check_cpu_usage(88)  # Warning
    alerts.check_latency("test_op", 1500)  # High latency
    alerts.check_drawdown(-22)  # High drawdown
    
    # Afficher les alertes
    active = alerts.get_active_alerts()
    critical = alerts.get_critical_alerts()
    
    print(f"\n📊 Alertes:")
    print(f"  Total alertes:    {len(active)}")
    print(f"  Alertes critiques: {len(critical)}")
    
    for alert in active[:5]:
        icon = "🔴" if alert.severity.name == "CRITICAL" else "🟡"
        print(f"  {icon} {alert.alert_type.value}: {alert.message}")
    
    # Afficher le résumé
    summary = alerts.get_alert_summary()
    print(f"\n  Résumé: {summary['total_unacknowledged']} alertes non traitées")
    print(f"  Par type: {summary['alert_types']}")


def demo_health_checks():
    """Démonstration des health checks"""
    print_header("5. HEALTH CHECKS")
    
    from scaf_ls.monitoring import HealthCheckSystem
    
    health = HealthCheckSystem()
    checks = health.run_all_checks()
    overall = health.get_overall_health(checks)
    
    print(f"📊 État du Système:")
    print(f"  Score global: {overall['overall_health_score']:.1f}/100")
    print(f"  Status: {overall['overall_status']}")
    
    print(f"\n  Checks détaillés:")
    for check_name, result in checks.items():
        icon = "✅" if result.status == 'healthy' else "⚠️" if result.status == 'degraded' else "❌"
        print(f"  {icon} {check_name}: {result.status}")


def demo_drift_detection():
    """Démonstration de la détection de drift"""
    print_header("6. DRIFT DETECTION")
    
    import numpy as np
    from scaf_ls.monitoring import DriftDetector
    
    detector = DriftDetector()
    
    # Créer les données d'entraînement de base
    X_train = np.random.normal(0, 1, (1000, 5))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.normal(0, 0.1, 1000)
    
    detector.set_reference_data(X_train, y_train)
    
    # Ajouter des données de test avec un drift
    print("Ajout de données de test (sans drift)...")
    for _ in range(50):
        x = np.random.normal(0, 1, 5)
        y = x[0] + 0.5 * x[1] + np.random.normal(0, 0.1)
        detector.add_test_sample(x.reshape(1, -1), y, y)
    
    metrics = detector.get_drift_metrics()
    
    print(f"\n📊 Drift Detection:")
    print(f"  Data drift:     {metrics.data_drift_score:.3f}")
    print(f"  Model drift:    {metrics.model_drift_score:.3f}")
    print(f"  Concept drift:  {metrics.concept_drift_score:.3f}")
    print(f"  Overall drift:  {metrics.overall_drift:.3f}")
    print(f"  Is drifting:    {metrics.is_drifting}")
    print(f"  Retrain needed: {detector.should_retrain()}")


def demo_maintenance():
    """Démonstration de la maintenance prédictive"""
    print_header("7. MAINTENANCE PRÉDICTIVE")
    
    from scaf_ls.monitoring import PredictiveMaintenanceEngine
    
    maintenance = PredictiveMaintenanceEngine()
    
    # Simuler quelques conditions
    print("Évaluation des conditions de maintenance...")
    
    maintenance.check_retraining_needed(
        drift_score=0.65,
        time_since_training_days=8,
        model_performance_degradation=0.12
    )
    
    maintenance.check_cleanup_needed(
        log_size_mb=1200,
        cache_size_mb=450,
        total_disk_free_gb=15
    )
    
    # Afficher le calendrier
    schedule = maintenance.get_maintenance_schedule()
    
    print(f"\n📊 Maintenance Schedule:")
    print(f"  Tâches en attente: {schedule['pending_tasks']}")
    print(f"  Haute priorité:    {schedule['high_priority_tasks']}")
    print(f"  Durée totale:      {schedule['total_duration_hours']:.1f} heures")
    
    if schedule['recommended_next_task']:
        task = schedule['recommended_next_task']
        print(f"\n  Prochaine tâche recommandée:")
        print(f"    Type: {task['type']}")
        print(f"    Description: {task['description']}")
        print(f"    Priorité: {task['priority']:.0f}/100")
    
    # Afficher les recommandations
    recommendations = maintenance.get_health_recommendations()
    print(f"\n  Recommandations:")
    for rec in recommendations:
        print(f"    • {rec}")


def demo_dashboard():
    """Démonstration du dashboard"""
    print_header("8. DASHBOARD COMPLET")
    
    monitoring = get_monitoring_service()
    dashboard = monitoring.get_dashboard_data()
    
    print("📊 Dashboard Data (JSON):\n")
    print(json.dumps(dashboard, indent=2, default=str)[:1000] + "\n...")


def main():
    """Exécuter la démo complète"""
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║   SCAF-LS Monitoring & Maintenance - Démo Rapide    ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    logger = get_logger("demo")
    logger.info("Starting SCAF-LS Monitoring demo")
    
    try:
        # Démarrer le monitoring
        monitoring = start_monitoring({
            'initial_equity': 100000,
        })
        
        # Exécuter les démos
        demo_system_monitoring()
        demo_latency_monitoring()
        demo_business_monitoring()
        demo_alerts()
        demo_health_checks()
        demo_drift_detection()
        demo_maintenance()
        demo_dashboard()
        
        # Résumé final
        print_header("RÉSUMÉ")
        print("""
        ✅ Monitoring système: Active
        ✅ Tracking applicatif: Active
        ✅ Tracking métier: Active
        ✅ Alertes intelligentes: Active
        ✅ Health checks: Active
        ✅ Drift detection: Active
        ✅ Maintenance prédictive: Active
        
        📊 Pour voir le monitoring en continu:
           - Utiliser monitoring.get_dashboard_data()
           - Configurer Prometheus + Grafana
           - Voir les logs dans logs/scaf-ls_*.json
        
        🚀 Integration avec SCAF-LS:
           Voir scaf_ls/monitoring/integration_guide.py
        """)
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error("Demo failed", error=str(e), exc_info=True)
        raise
    
    finally:
        stop_monitoring()
        print("\n✅ Demo terminée\n")


if __name__ == "__main__":
    main()
