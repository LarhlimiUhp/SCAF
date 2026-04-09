"""
Service de Monitoring Principal - Orchestration centralisée
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from pathlib import Path

from .system_metrics import SystemMetricsCollector
from .app_metrics import ApplicationMetricsCollector, get_collector as get_app_collector
from .business_metrics import BusinessMetricsCollector
from .alerts import AlertSystem, AlertType, AlertSeverity
from .logger import StructuredLogger, get_logger
from .health_checks import HealthCheckSystem
from .profiler import PerformanceProfiler
from .drift_detection import DriftDetector
from .maintenance import PredictiveMaintenanceEngine


class MonitoringService:
    """Service de monitoring principal"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("scaf-ls-monitoring")
        
        # Collecteurs
        self.system_metrics = SystemMetricsCollector()
        self.app_metrics = get_app_collector()
        self.business_metrics = BusinessMetricsCollector(
            initial_equity=self.config.get('initial_equity', 100000.0)
        )
        
        # Systèmes
        self.alerts = AlertSystem()
        self.health_checks = HealthCheckSystem()
        self.profiler = PerformanceProfiler()
        self.drift_detector = DriftDetector()
        self.maintenance_engine = PredictiveMaintenanceEngine()
        
        # État
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_full_check = None
        self.metrics_history: Dict[str, list] = {}
        
        # Thresholds configurables
        self.thresholds = self.config.get('thresholds', self._get_default_thresholds())
        
        self.logger.info("Monitoring service initialized", 
                        initial_equity=self.config.get('initial_equity', 100000.0))
    
    def _get_default_thresholds(self) -> Dict[str, Any]:
        """Thresholds par défaut"""
        return {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 80,
            'latency_ms': 1000,
            'error_rate_per_min': 5,
            'drawdown_percent': -20,
            'win_rate_percent': 30,
            'data_drift': 0.7,
            'model_drift': 0.6,
        }
    
    def start(self):
        """Démarrer le service de monitoring"""
        if self.is_running:
            self.logger.warning("Monitoring service already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Monitoring service started")
    
    def stop(self):
        """Arrêter le service de monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Monitoring service stopped")
    
    def _monitoring_loop(self):
        """Boucle principale de monitoring"""
        full_check_interval = 60  # secondes
        last_full_check = time.time()
        
        while self.is_running:
            try:
                # Collecte rapide (toutes les 5 secondes)
                self._collect_quick_metrics()
                
                # Full check (toutes les 60 secondes)
                if time.time() - last_full_check > full_check_interval:
                    self._run_full_check()
                    last_full_check = time.time()
                
                time.sleep(5)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e), exc_info=True)
                self.alerts.create_alert(
                    AlertType.SERVICE_DOWN,
                    AlertSeverity.CRITICAL,
                    f"Monitoring service error: {str(e)}"
                )
    
    def _collect_quick_metrics(self):
        """Collecter les métriques rapides"""
        try:
            # Métriques système
            sys_metrics = self.system_metrics.collect()
            self._store_metric('system', asdict(sys_metrics))
            
            # Vérifier les seuils système
            self.alerts.check_cpu_usage(
                sys_metrics.cpu_percent,
                threshold=self.thresholds['cpu_percent']
            )
            self.alerts.check_memory_usage(
                sys_metrics.memory_percent,
                threshold=self.thresholds['memory_percent']
            )
            self.alerts.check_disk_usage(
                sys_metrics.disk_percent,
                threshold=self.thresholds['disk_percent']
            )
            
            # Métriques applicatives
            error_metric = self.app_metrics.get_error_metric()
            self.alerts.check_error_rate(
                error_metric.errors_per_minute,
                threshold=self.thresholds['error_rate_per_min']
            )
            
            # Métriques métier
            portfolio_metrics = self.business_metrics.get_portfolio_metrics()
            self.alerts.check_drawdown(
                portfolio_metrics.current_drawdown,
                max_threshold=self.thresholds['drawdown_percent']
            )
            
            trade_metrics = self.business_metrics.get_trade_metrics()
            if trade_metrics.total_trades > 0:
                self.alerts.check_win_rate(
                    trade_metrics.win_rate,
                    min_threshold=self.thresholds['win_rate_percent']
                )
            
            self.alerts.check_pnl(
                trade_metrics.total_pnl,
                portfolio_metrics.daily_return
            )
            
        except Exception as e:
            self.logger.error("Error collecting quick metrics", error=str(e))
    
    def _run_full_check(self):
        """Exécuter les vérifications complètes"""
        try:
            self.logger.info("Running full monitoring check")
            self.last_full_check = datetime.utcnow()
            
            # Health checks
            checks = self.health_checks.run_all_checks({
                'data_loader': self.config.get('data_loader'),
                'models': self.config.get('models'),
                'features': self.config.get('features'),
                'backtest_results': self.config.get('backtest_results'),
            })
            
            overall_health = self.health_checks.get_overall_health(checks)
            self._store_metric('health', overall_health)
            
            # Drift detection
            if self.config.get('features') is not None:
                self.drift_detector.add_test_sample(
                    self.config.get('features'),
                    self.config.get('target'),
                    self.config.get('prediction')
                )
                
                drift_metrics = self.drift_detector.get_drift_metrics()
                
                self.alerts.check_data_drift(
                    drift_metrics.data_drift_score,
                    threshold=self.thresholds['data_drift']
                )
                
                self.alerts.check_model_drift(
                    drift_metrics.model_drift_score,
                    threshold=self.thresholds['model_drift']
                )
                
                if self.drift_detector.should_retrain():
                    self.maintenance_engine.check_retraining_needed(
                        drift_metrics.overall_drift,
                        0,  # Placeholder pour jours depuis dernier training
                        0   # Placeholder pour dégradation performance
                    )
            
            # Vérifications maintenance
            self.maintenance_engine.check_cleanup_needed(
                log_size_mb=100,  # Placeholder
                cache_size_mb=50,
                total_disk_free_gb=self.system_metrics.collect().disk_free_gb
            )
            
            # Résumé des alertes
            alert_summary = self.alerts.get_alert_summary()
            self._store_metric('alerts', alert_summary)
            
            # Résumé de la maintenance
            maintenance_summary = self.maintenance_engine.get_maintenance_schedule()
            self._store_metric('maintenance', maintenance_summary)
            
            self.logger.info("Full check completed",
                            health_score=overall_health.get('overall_health_score'),
                            critical_alerts=alert_summary.get('critical_count', 0))
            
        except Exception as e:
            self.logger.error("Error in full check", error=str(e), exc_info=True)
    
    def _store_metric(self, category: str, data: Dict[str, Any]):
        """Stocker une métrique dans l'historique"""
        if category not in self.metrics_history:
            self.metrics_history[category] = []
        
        self.metrics_history[category].append({
            'timestamp': datetime.utcnow().isoformat(),
            'data': data,
        })
        
        # Garder seulement les 100 dernières entrées par catégorie
        if len(self.metrics_history[category]) > 100:
            self.metrics_history[category] = self.metrics_history[category][-100:]
    
    def update_equity(self, equity: float):
        """Mettre à jour le montant du portefeuille"""
        self.business_metrics.update_equity(equity)
    
    def record_trade(self, entry: float, exit: float, qty: int, side: str = "long"):
        """Enregistrer un trade"""
        self.business_metrics.record_trade(entry, exit, qty, side)
        self.logger.info("Trade recorded",
                        side=side, entry=entry, exit=exit, qty=qty)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtenir toutes les données pour le dashboard"""
        sys_metrics = self.system_metrics.collect()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu_percent': sys_metrics.cpu_percent,
                'memory_percent': sys_metrics.memory_percent,
                'disk_percent': sys_metrics.disk_percent,
                'gpu_available': sys_metrics.gpu_available,
                'gpu_percent': sys_metrics.gpu_percent,
            },
            'application': {
                'latency_metrics': {
                    name: {
                        'p50_ms': m.percentile_p50,
                        'p95_ms': m.percentile_p95,
                        'p99_ms': m.percentile_p99,
                    }
                    for name, m in self.app_metrics.get_all_latency_metrics().items()
                },
                'error_rate': self.app_metrics.get_error_metric().errors_per_minute,
                'health': self.app_metrics.get_health_score(),
            },
            'business': self.business_metrics.get_summary_dict(),
            'alerts': self.alerts.get_alert_summary(),
            'health': self._get_last_health_check(),
            'maintenance': self.maintenance_engine.get_maintenance_schedule(),
        }
    
    def _get_last_health_check(self) -> Dict[str, Any]:
        """Obtenir le dernier health check"""
        if 'health' not in self.metrics_history or not self.metrics_history['health']:
            return {'status': 'unknown'}
        
        return self.metrics_history['health'][-1]['data']
    
    def export_prometheus_metrics(self) -> str:
        """Exporter au format Prometheus"""
        lines = []
        sys_metrics = self.system_metrics.collect()
        
        # Métriques système
        lines.append(f"scaf_cpu_percent {sys_metrics.cpu_percent}")
        lines.append(f"scaf_memory_percent {sys_metrics.memory_percent}")
        lines.append(f"scaf_disk_percent {sys_metrics.disk_percent}")
        
        # Métriques applicatives
        error_metric = self.app_metrics.get_error_metric()
        lines.append(f"scaf_errors_total {error_metric.total_errors}")
        lines.append(f"scaf_errors_per_minute {error_metric.errors_per_minute}")
        
        # Métriques métier
        portfolio = self.business_metrics.get_portfolio_metrics()
        lines.append(f"scaf_equity {portfolio.current_equity}")
        lines.append(f"scaf_total_return_percent {portfolio.total_return}")
        lines.append(f"scaf_max_drawdown_percent {portfolio.max_drawdown}")
        lines.append(f"scaf_sharpe_ratio {portfolio.sharpe_ratio}")
        
        trades = self.business_metrics.get_trade_metrics()
        lines.append(f"scaf_total_trades {trades.total_trades}")
        lines.append(f"scaf_win_rate_percent {trades.win_rate}")
        lines.append(f"scaf_profit_factor {trades.profit_factor}")
        
        # Alertes
        alert_summary = self.alerts.get_alert_summary()
        lines.append(f"scaf_alerts_critical {alert_summary.get('critical_count', 0)}")
        
        return "\n".join(lines)


# Décorateur pour instrumenter les fonctions automatiquement
def with_monitoring(operation_name: str):
    """Décorateur pour monitorer une fonction"""
    def decorator(func: Callable):
        from .app_metrics import track_latency
        return track_latency(operation_name)(func)
    return decorator


# Instance globale
_monitoring_service: Optional[MonitoringService] = None

def get_monitoring_service(config: Optional[Dict[str, Any]] = None) -> MonitoringService:
    """Récupérer l'instance globale du service de monitoring"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService(config)
    return _monitoring_service


def start_monitoring(config: Optional[Dict[str, Any]] = None):
    """Démarrer le monitoring"""
    service = get_monitoring_service(config)
    service.start()
    return service


def stop_monitoring():
    """Arrêter le monitoring"""
    global _monitoring_service
    if _monitoring_service:
        _monitoring_service.stop()
        _monitoring_service = None
