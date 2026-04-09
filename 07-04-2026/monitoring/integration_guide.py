"""
Guide d'intégration du Monitoring & Maintenance dans SCAF-LS
Exemple d'utilisation et best practices
"""

from scaf_ls.monitoring import (
    MonitoringService,
    get_monitoring_service,
    start_monitoring,
    stop_monitoring,
)
from scaf_ls.monitoring.app_metrics import track_latency, get_collector
from scaf_ls.monitoring.logger import get_logger

# ============================================================================
# 1. INITIALISATION DU MONITORING
# ============================================================================

def initialize_monitoring(config):
    """Initialiser le monitoring avec configuration SCAF-LS"""
    
    monitoring_config = {
        'initial_equity': config.INITIAL_CAPITAL,
        'thresholds': {
            'cpu_percent': 85,
            'memory_percent': 90,
            'disk_percent': 85,
            'latency_ms': 500,  # P99 latency
            'error_rate_per_min': 10,
            'drawdown_percent': config.MAX_DRAWDOWN_LIMIT,  # -20%
            'win_rate_percent': 45,
            'data_drift': 0.7,
            'model_drift': 0.6,
        },
        'data_loader': None,  # Sera fourni plus tard
        'models': {},  # Sera populé après l'entraînement
    }
    
    service = start_monitoring(monitoring_config)
    logger = get_logger("scaf-ls")
    
    logger.info("Monitoring initialized", 
               initial_capital=config.INITIAL_CAPITAL)
    
    return service


# ============================================================================
# 2. INSTRUMENTATION DES FONCTIONS PRINCIPALES
# ============================================================================

class InstrumentedBacktestEngine:
    """Wrapper autour du BacktestEngine avec monitoring"""
    
    def __init__(self, base_engine, monitoring_service):
        self.engine = base_engine
        self.monitoring = monitoring_service
        self.logger = get_logger("backtest")
    
    @track_latency("backtest_iteration")
    def run_iteration(self, *args, **kwargs):
        """Exécuter une itération de backtest avec tracking"""
        try:
            result = self.engine.run_iteration(*args, **kwargs)
            return result
        except Exception as e:
            collector = get_collector()
            collector.record_error(type(e).__name__, str(e))
            raise
    
    def run_folded_backtest_with_monitoring(self, X_array, y_array, returns, prices):
        """Backtest avec monitoring continu"""
        
        initial_equity = self.monitoring.business_metrics.current_equity
        self.logger.info("Starting backtesting", 
                        initial_equity=initial_equity,
                        samples=len(X_array))
        
        try:
            result_df = self.engine.run_folded_backtest(X_array, y_array, returns, prices)
            
            # Mettre à jour le monitoring avec les résultats
            if len(result_df) > 0:
                final_equity = result_df['equity'].iloc[-1]
                self.monitoring.update_equity(final_equity)
                
                # Log du résumé
                summary = self.monitoring.business_metrics.get_summary_dict()
                self.logger.info("Backtest completed",
                               final_equity=final_equity,
                               total_return_percent=summary['portfolio']['total_return'],
                               sharpe_ratio=summary['portfolio']['sharpe_ratio'],
                               win_rate=summary['trades']['win_rate'])
            
            return result_df
            
        except Exception as e:
            self.logger.error("Backtest failed", error=str(e), exc_info=True)
            raise


# ============================================================================
# 3. MONITORING CONTINU PENDANT LE TRADING EN DIRECT
# ============================================================================

class MonitoredTradingSystem:
    """Système de trading avec monitoring en continu"""
    
    def __init__(self, trading_engine, monitoring_service):
        self.engine = trading_engine
        self.monitoring = monitoring_service
        self.logger = get_logger("trading")
    
    @track_latency("order_execution")
    def execute_order(self, signal, price, quantity):
        """Exécuter un ordre avec monitoring"""
        try:
            order = self.engine.execute_order(signal, price, quantity)
            
            # Enregistrer le trade
            if signal == 'long':
                self.monitoring.record_trade(
                    entry=price,
                    exit=price,  # Exit sera updaté à la clôture
                    qty=quantity,
                    side='long'
                )
            elif signal == 'short':
                self.monitoring.record_trade(
                    entry=price,
                    exit=price,
                    qty=quantity,
                    side='short'
                )
            
            self.logger.info("Order executed",
                           signal=signal, price=price, qty=quantity)
            
            return order
            
        except Exception as e:
            collector = get_collector()
            collector.record_error("OrderExecutionError", str(e))
            self.logger.error("Order execution failed", error=str(e))
            raise
    
    def update_daily_pnl(self, daily_pnl, equity):
        """Mettre à jour le P&L journalier"""
        self.monitoring.update_equity(equity)
        
        # Vérifier les alertes critiques
        metrics = self.monitoring.business_metrics.get_portfolio_metrics()
        health = self.monitoring.business_metrics.get_health_score()
        
        self.logger.info("Daily update",
                        equity=equity,
                        daily_pnl=daily_pnl,
                        overall_health=health['overall_health'],
                        sharpe_ratio=metrics.sharpe_ratio)


# ============================================================================
# 4. DASHBOARD & REPORTING
# ============================================================================

def generate_monitoring_report(monitoring_service):
    """Générer un rapport de monitoring complet"""
    
    dashboard_data = monitoring_service.get_dashboard_data()
    
    report = {
        'timestamp': dashboard_data['timestamp'],
        'system': dashboard_data['system'],
        'application': dashboard_data['application'],
        'business': dashboard_data['business'],
        'alerts': dashboard_data['alerts'],
        'health': dashboard_data['health'],
        'maintenance': dashboard_data['maintenance'],
        'recommendations': monitoring_service.maintenance_engine.get_health_recommendations(),
    }
    
    return report


def export_prometheus_metrics(monitoring_service):
    """Exporter les métriques pour Prometheus/Grafana"""
    prometheus_text = monitoring_service.export_prometheus_metrics()
    
    # Sauvegarder dans un fichier
    with open('metrics.prom', 'w') as f:
        f.write(prometheus_text)
    
    return prometheus_text


# ============================================================================
# 5. EXEMPLE D'UTILISATION COMPLET
# ============================================================================

def example_integration():
    """Exemple d'intégration complète"""
    
    from scaf_ls.config import Config
    from scaf_ls.main import run_backtest
    
    # Étape 1: Initialiser le monitoring
    monitoring = initialize_monitoring(Config)
    
    logger = get_logger("scaf-ls")
    
    try:
        # Étape 2: Exécuter le backtest normal
        logger.info("Starting SCAF-LS backtest with monitoring...")
        results = run_backtest(Config)
        
        # Étape 3: Enregistrer les résultats
        if results and 'backtest' in results:
            backtest_df = results['backtest']
            
            # Simuler l'update de l'equity à chaque pas
            for idx, row in backtest_df.iterrows():
                if idx % 100 == 0:  # Update tous les 100 pas
                    monitoring.update_equity(float(row['equity']))
        
        # Étape 4: Générer le rapport
        report = generate_monitoring_report(monitoring)
        
        logger.info("Backtest completed with monitoring",
                   report=report)
        
        # Étape 5: Vérifier les alertes critiques
        critical_alerts = monitoring.alerts.get_critical_alerts()
        if critical_alerts:
            logger.warning("Critical alerts detected",
                          count=len(critical_alerts),
                          alerts=[a.message for a in critical_alerts[:5]])
        
        # Étape 6: Obtenir les recommandations
        recommendations = monitoring.maintenance_engine.get_health_recommendations()
        logger.info("Maintenance recommendations",
                   recommendations=recommendations)
        
        return results, report
        
    except Exception as e:
        logger.error("Backtest failed", error=str(e), exc_info=True)
        raise
        
    finally:
        # Étape 7: Arrêter le monitoring
        stop_monitoring()
        logger.info("Monitoring stopped")


# ============================================================================
# 6. ALERTES PERSONNALISÉES (Optional)
# ============================================================================

def register_custom_alert_handlers(monitoring_service):
    """Enregistrer des handlers personnalisés pour les alertes"""
    
    def on_high_drawdown(alert):
        """Handler pour drawdown élevé"""
        logger = get_logger("alerts")
        logger.critical("CRITICAL: High drawdown alert",
                       message=alert.message,
                       value=alert.metric_value)
        # Possibilité d'envoyer une notification (email, Slack, etc.)
    
    def on_model_drift(alert):
        """Handler pour model drift"""
        logger = get_logger("alerts")
        logger.warning("Model drift detected",
                      message=alert.message,
                      value=alert.metric_value)
        # Planifier un réentraînement
    
    # Enregistrer les handlers
    from scaf_ls.monitoring.alerts import AlertType
    
    monitoring_service.alerts.register_handler(
        AlertType.HIGH_DRAWDOWN,
        on_high_drawdown
    )
    
    monitoring_service.alerts.register_handler(
        AlertType.MODEL_DRIFT,
        on_model_drift
    )


if __name__ == "__main__":
    # Exemple simple
    example_integration()
