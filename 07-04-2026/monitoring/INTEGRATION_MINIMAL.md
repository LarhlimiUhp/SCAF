"""
Guide d'intégration minimale - Comment ajouter le monitoring à SCAF-LS existant
Modifications minimales requises pour activer la surveillance complète
"""

# ============================================================================
# OPTION 1: Intégration minimale dans main.py
# ============================================================================

# AVANT: scaf_ls/main.py

"""
def run_backtest(cfg=None):
    cfg = cfg or Config
    loader = MultiAssetLoader(cfg)
    spx, cross = loader.download()
    # ... reste du code
"""

# APRÈS: scaf_ls/main.py

"""
def run_backtest(cfg=None):
    from scaf_ls.monitoring import start_monitoring, stop_monitoring, get_monitoring_service
    from scaf_ls.monitoring.app_metrics import track_latency
    from scaf_ls.monitoring.logger import get_logger
    
    cfg = cfg or Config
    logger = get_logger("scaf-ls-backtest")
    
    # Démarrer le monitoring
    monitoring = start_monitoring({
        'initial_equity': cfg.INITIAL_CAPITAL,
        'thresholds': {
            'drawdown_percent': cfg.MAX_DRAWDOWN_LIMIT,
            'cpu_percent': 85,
            'memory_percent': 90,
        }
    })
    
    try:
        logger.info("Starting backtest", config=cfg.TICKER)
        
        loader = MultiAssetLoader(cfg)
        spx, cross = loader.download()
        # ... reste du code normal ...
        
        # Mettre à jour les métriques métier
        engine = BacktestEngine(cfg, base_models, aggregator=stacking)
        backtest_df = engine.run_folded_backtest(X_array, y_array, returns, prices)
        
        # Simuler les updates d'équité
        for idx, row in backtest_df.iterrows():
            if idx % 100 == 0:
                monitoring.update_equity(float(row['equity']))
        
        # Log final
        summary = monitoring.business_metrics.get_summary_dict()
        logger.info("Backtest completed",
                   final_equity=summary['portfolio']['current_equity'],
                   sharpe=summary['portfolio']['sharpe_ratio'],
                   win_rate=summary['trades']['win_rate'])
        
        return backtest_df
        
    except Exception as e:
        logger.error("Backtest failed", error=str(e), exc_info=True)
        raise
    
    finally:
        stop_monitoring()
"""


# ============================================================================
# OPTION 2: Décorateurs pour instrumenter les fonctions existantes
# ============================================================================

"""
# Dans n'importe quel module de SCAF-LS

from scaf_ls.monitoring.app_metrics import track_latency

# Ajouter simplement un décorateur
@track_latency("data_engineering")
def build_features(spx, cross):
    # Code existant inchangé
    pass

@track_latency("model_training")
def fit_final_models(cfg, X_array, y_array):
    # Code existant inchangé
    pass

@track_latency("cross_validation")
def cross_validate_models(cfg, X_array, y_array, n_splits):
    # Code existant inchangé
    pass
"""


# ============================================================================
# OPTION 3: Health Checks avant le backtest
# ============================================================================

"""
from scaf_ls.monitoring import HealthCheckSystem

def validate_before_backtest(cfg, X, y, models):
    \"\"\"Vérifier la santé du système avant le backtest\"\"\"
    
    health = HealthCheckSystem()
    checks = health.run_all_checks({
        'data_loader': None,  # Optional
        'models': models,
        'features': X,
    })
    
    overall = health.get_overall_health(checks)
    
    if overall['overall_health_score'] < 50:
        print(f"⚠️ WARNING: System health is degraded")
        print(f"Overall health: {overall['overall_health_score']:.0f}/100")
        print(overall)
    
    return overall['overall_status'] != 'unhealthy'
"""


# ============================================================================
# OPTION 4: Export Prometheus pour monitoring en continu
# ============================================================================

"""
# Créer un simple serveur Flask pour exposer les métriques

from flask import Flask, Response
from scaf_ls.monitoring import get_monitoring_service

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    \"\"\"Endpoint Prometheus\"\"\"
    service = get_monitoring_service()
    prometheus_text = service.export_prometheus_metrics()
    return Response(prometheus_text, mimetype='text/plain')

@app.route('/health')
def health():
    \"\"\"Health check endpoint\"\"\"
    service = get_monitoring_service()
    checks = service.health_checks.run_all_checks()
    overall = service.health_checks.get_overall_health(checks)
    return overall

# Démarrer après le backtest:
# python app.py
# Puis accéder à http://localhost:5000/metrics
"""


# ============================================================================
# OPTION 5: Logging structuré dans les calculs kritiques
# ============================================================================

"""
from scaf_ls.monitoring.logger import get_logger

logger = get_logger("scaf-ls-backtest")

# Utiliser dans les code existants:

logger.info("Feature engineering started", asset_count=len(CROSS_ASSET_TICKERS))

logger.info("Model training", 
           model_name=model_name,
           X_shape=X_array.shape,
           y_shape=y_array.shape)

logger.warning("Low win rate detected", win_rate=45.2)

logger.error("Model prediction failed", error=str(e), model=model_name)
"""


# ============================================================================
# OPTION 6: Configuration personnalisée des alertes
# ============================================================================

"""
from scaf_ls.monitoring import get_monitoring_service
from scaf_ls.monitoring.alerts import AlertType

monitoring = get_monitoring_service()

# Enregistrer des handlers personnalisés

def on_high_drawdown(alert):
    print(f"🚨 CRITICAL: {alert.message}")
    # Envoyer email/Slack/webhook
    import requests
    requests.post("https://hooks.slack.com/...", json={
        "text": f"⚠️ High Drawdown: {alert.metric_value}%"
    })

def on_data_drift(alert):
    print(f"⚠️ Data drift detected: {alert.metric_value:.3f}")
    # Planifier un réentraînement

monitoring.alerts.register_handler(AlertType.HIGH_DRAWDOWN, on_high_drawdown)
monitoring.alerts.register_handler(AlertType.DATA_DRIFT, on_data_drift)
"""


# ============================================================================
# OPTION 7: Intégration avec Backtest Engine personnalisé
# ============================================================================

"""
# Wrapper minimal autour de BacktestEngine

from scaf_ls.monitoring.app_metrics import track_latency

class MonitoredBacktestEngine:
    def __init__(self, base_engine):
        self.engine = base_engine
    
    @track_latency("backtest_step")
    def step(self, signals, prices):
        return self.engine.step(signals, prices)
    
    def run_folded_backtest(self, *args, **kwargs):
        # Code existant de base_engine.run_folded_backtest
        # mais avec updates du monitoring à intervalles réguliers
        pass
"""


# ============================================================================
# QUICK START - Copier/coller dans run_scaf_ls.py
# ============================================================================

QUICK_START_CODE = """
# Ajouter au top de run_scaf_ls.py:

from scaf_ls.monitoring import start_monitoring, stop_monitoring
from scaf_ls.monitoring.logger import get_logger

logger = get_logger("scaf-ls-run")

# Ajouter avant run_backtest():
monitoring = start_monitoring({
    'initial_equity': Config.INITIAL_CAPITAL
})

try:
    results = run_backtest(Config)
    
    # Évaluer les résultats avec monitoring
    health_score = monitoring.business_metrics.get_health_score()
    logger.info("Backtest completed", health=health_score)
    
    # Afficher le dashboard
    dashboard = monitoring.get_dashboard_data()
    print(f"\\n📊 Final Score: {dashboard['business']['portfolio']['total_return']:.2f}%")
    
finally:
    stop_monitoring()
"""


# ============================================================================
# FILES À MODIFIER MINIMALEMENT
# ============================================================================

MINIMAL_MODIFICATIONS = """
Fichiers existants à modifier (minimal):

1. scaf_ls/main.py
   - Ajouter 3-4 lignes d'import
   - Wrapper le backtest avec monitoring
   - Envoyer les updates d'équité

2. run_scaf_ls.py (optionnel)
   - Démarrer/arrêter le monitoring
   - Logger les résultats

3. scaf_ls/backtest.py (optionnel)
   - Ajouter @track_latency décorateurs
   - Envoyer les equity updates

Aucune autre modification requise!
Le monitoring fonctionne en parallèle (thread daemon).
"""


# ============================================================================
# VERIFICATION DE L'INTEGRATION
# ============================================================================

def verify_monitoring_integration():
    """Vérifier que le monitoring est bien intégré"""
    
    from scaf_ls.monitoring import get_monitoring_service
    
    service = get_monitoring_service()
    
    # Éléments clés à vérifier
    checks = {
        'logger': hasattr(service, 'logger'),
        'system_metrics': hasattr(service, 'system_metrics'),
        'app_metrics': hasattr(service, 'app_metrics'),
        'business_metrics': hasattr(service, 'business_metrics'),
        'alerts': hasattr(service, 'alerts'),
        'health_checks': hasattr(service, 'health_checks'),
        'profiler': hasattr(service, 'profiler'),
        'drift_detector': hasattr(service, 'drift_detector'),
        'maintenance_engine': hasattr(service, 'maintenance_engine'),
    }
    
    all_ok = all(checks.values())
    
    print("✅ Monitoring Integration Status:")
    for component, ok in checks.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {component}")
    
    if all_ok:
        print("\n🎉 Tous les composants de monitoring sont actifs!")
    
    return all_ok


# ============================================================================
# EXEMPLE COMPLET D'INTEGRATION
# ============================================================================

COMPLETE_INTEGRATION_EXAMPLE = '''
# File: run_scaf_ls_monitored.py

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scaf_ls.main import run_backtest
from scaf_ls.config import Config
from scaf_ls.monitoring import start_monitoring, stop_monitoring
from scaf_ls.monitoring.logger import get_logger
from scaf_ls.monitoring.app_metrics import track_latency

logger = get_logger("scaf-ls")

def main():
    """Main entry point with monitoring"""
    
    # Démarrer le monitoring
    logger.info("Starting SCAF-LS with monitoring")
    monitoring = start_monitoring({
        'initial_equity': Config.INITIAL_CAPITAL,
    })
    
    try:
        # Exécuter le backtest
        logger.info("Running backtest...", config=Config.TICKER)
        results = run_backtest(Config)
        
        if results and 'backtest' in results:
            backtest_df = results['backtest']
            
            # Mettre à jour les métriques
            for idx, row in backtest_df.iterrows():
                if idx % 100 == 0:
                    monitoring.update_equity(float(row['equity']))
            
            # Afficher le résumé
            summary = monitoring.business_metrics.get_summary_dict()
            
            print(f"""
            ╔════════════════════════════════════╗
            ║     SCAF-LS Backtest Results       ║
            ╚════════════════════════════════════╝
            
            📊 Performance Metrics:
               Initial Capital:  ${Config.INITIAL_CAPITAL:,}
               Final Equity:     ${summary['portfolio']['current_equity']:,.2f}
               Total Return:     {summary['portfolio']['total_return']:.2f}%
               
               Sharpe Ratio:     {summary['portfolio']['sharpe_ratio']:.2f}
               Max Drawdown:     {summary['portfolio']['max_drawdown']:.2f}%
               
               Total Trades:     {summary['trades']['total_trades']}
               Win Rate:         {summary['trades']['win_rate']:.1f}%
               Profit Factor:    {summary['trades']['profit_factor']:.2f}
            
            ⚠️  Health Status:
               Overall Health:   {summary['health']['overall_health']:.0f}/100
            """)
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error("Backtest failed", error=str(e), exc_info=True)
        raise
    
    finally:
        stop_monitoring()
        logger.info("Monitoring stopped")

if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    print("""
    ╔═════════════════════════════════════════════════════════╗
    ║  SCAF-LS Monitoring & Maintenance - Integration Guide  ║
    ╚═════════════════════════════════════════════════════════╝
    
    Ce fichier contient les instructions d'intégration minimale.
    
    Kopier les snippets et adapter à votre code.
    
    Étapes rapides (< 5 minutes):
    
    1. Ajouter les imports:
       from scaf_ls.monitoring import start_monitoring, stop_monitoring
    
    2. Démarrer avant run_backtest():
       monitoring = start_monitoring({'initial_equity': INITIAL_CAPITAL})
    
    3. Arrêter après:
       stop_monitoring()
    
    C'est tout! Le monitoring démarre automatiquement.
    """)
    
    verify_monitoring_integration()
