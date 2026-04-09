"""
Système d'alertes intelligentes avec seuils dynamiques et anomalies ML
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque
import warnings

warnings.filterwarnings('ignore')


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2


class AlertType(Enum):
    """Types d'alertes"""
    # Système
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    DISK_FULL = "disk_full"
    GPU_FAILURE = "gpu_failure"
    
    # Application
    HIGH_LATENCY = "high_latency"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_THROUGHPUT = "low_throughput"
    EXCEPTION_SPIKE = "exception_spike"
    
    # Trading
    HIGH_DRAWDOWN = "high_drawdown"
    LOW_WIN_RATE = "low_win_rate"
    NEGATIVE_PNL = "negative_pnl"
    SHARP_DECLINE = "sharp_decline"
    NO_TRADES = "no_trades"
    
    # Data
    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    CONCEPT_DRIFT = "concept_drift"
    
    # Infrastructure
    SERVICE_DOWN = "service_down"
    CONNECTION_TIMEOUT = "connection_timeout"
    

@dataclass
class Alert:
    """Représentation d'une alerte"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    context: Dict[str, Any] = None
    acknowledged: bool = False
    

class DynamicThreshold:
    """Seuil dynamique basé sur statistiques glissantes"""
    
    def __init__(self, base_value: float, lookback_window: int = 100):
        self.base_value = base_value
        self.lookback_window = lookback_window
        self.historical_values: deque = deque(maxlen=lookback_window)
        
    def add_value(self, value: float):
        """Ajouter une valeur à l'historique"""
        self.historical_values.append(value)
    
    def get_threshold(self, multiplier: float = 1.5) -> float:
        """Obtenir le seuil dynamique"""
        if len(self.historical_values) < 10:
            return self.base_value
        
        # Seuil = mean + multiplier * std_dev
        values = np.array(list(self.historical_values))
        mean = np.mean(values)
        std = np.std(values)
        
        dynamic_thresh = mean + multiplier * std
        return max(self.base_value, dynamic_thresh)


class AnomalyDetector:
    """Détecteur d'anomalies basé sur écart-type"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.values: deque = deque(maxlen=window_size)
        
    def add_value(self, value: float) -> bool:
        """Ajouter une valeur et vérifier si c'est une anomalie"""
        self.values.append(value)
        
        if len(self.values) < 10:
            return False
        
        values = np.array(list(self.values))
        mean = np.mean(values)
        std = np.std(values)
        
        # Anomalie si > sensitivity * std away from mean
        z_score = abs((value - mean) / (std + 1e-6))
        
        return z_score > self.sensitivity
    
    def get_anomaly_score(self, value: float) -> float:
        """Obtenir un score d'anomalie (0-1)"""
        if len(self.values) < 10:
            return 0.0
        
        values = np.array(list(self.values))
        mean = np.mean(values)
        std = np.std(values)
        
        z_score = abs((value - mean) / (std + 1e-6))
        
        # Normaliser entre 0 et 1
        return min(1.0, z_score / self.sensitivity)


class AlertSystem:
    """Système d'alertes centralisant tous les types d'alertes"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_counter = 0
        self.handlers: Dict[AlertType, List[Callable]] = {}
        self.thresholds: Dict[str, DynamicThreshold] = {}
        self.detectors: Dict[str, AnomalyDetector] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Historique pour détection d'anomalies
        self.cpu_detector = AnomalyDetector(window_size=100, sensitivity=2.5)
        self.memory_detector = AnomalyDetector(window_size=100, sensitivity=2.5)
        self.latency_detector = AnomalyDetector(window_size=100, sensitivity=2.0)
        self.error_detector = AnomalyDetector(window_size=100, sensitivity=3.0)
        self.pnl_detector = AnomalyDetector(window_size=50, sensitivity=1.5)
        
    def register_handler(self, alert_type: AlertType, handler: Callable):
        """Enregistrer un handler pour un type d'alerte"""
        if alert_type not in self.handlers:
            self.handlers[alert_type] = []
        self.handlers[alert_type].append(handler)
    
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     message: str, **kwargs) -> Alert:
        """Créer une nouvelle alerte"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{datetime.utcnow():%H%M%S}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            timestamp=datetime.utcnow().isoformat(),
            message=message,
            **kwargs
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Déclencher les handlers
        if alert_type in self.handlers:
            for handler in self.handlers[alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Handler error for {alert_type}: {e}")
        
        return alert
    
    # --- Alertes Système ---
    def check_cpu_usage(self, cpu_percent: float, threshold: float = 80):
        """Vérifier l'utilisation CPU"""
        self.cpu_detector.add_value(cpu_percent)
        
        if cpu_percent > threshold:
            severity = AlertSeverity.CRITICAL if cpu_percent > 95 else AlertSeverity.WARNING
            self.create_alert(
                AlertType.HIGH_CPU,
                severity,
                f"CPU usage high: {cpu_percent:.1f}%",
                metric_name="cpu_percent",
                metric_value=cpu_percent,
                threshold=threshold,
            )
    
    def check_memory_usage(self, memory_percent: float, threshold: float = 85):
        """Vérifier l'utilisation mémoire"""
        self.memory_detector.add_value(memory_percent)
        
        if memory_percent > threshold:
            severity = AlertSeverity.CRITICAL if memory_percent > 95 else AlertSeverity.WARNING
            self.create_alert(
                AlertType.HIGH_MEMORY,
                severity,
                f"Memory usage high: {memory_percent:.1f}%",
                metric_name="memory_percent",
                metric_value=memory_percent,
                threshold=threshold,
            )
    
    def check_disk_usage(self, disk_percent: float, threshold: float = 80):
        """Vérifier l'utilisation disque"""
        if disk_percent > threshold:
            severity = AlertSeverity.CRITICAL if disk_percent > 90 else AlertSeverity.WARNING
            self.create_alert(
                AlertType.DISK_FULL,
                severity,
                f"Disk usage high: {disk_percent:.1f}%",
                metric_name="disk_percent",
                metric_value=disk_percent,
                threshold=threshold,
            )
    
    # --- Alertes Application ---
    def check_latency(self, operation: str, latency_ms: float, threshold_ms: float = 1000):
        """Vérifier la latence d'une opération"""
        self.latency_detector.add_value(latency_ms)
        
        if latency_ms > threshold_ms:
            severity = AlertSeverity.CRITICAL if latency_ms > threshold_ms * 2 else AlertSeverity.WARNING
            self.create_alert(
                AlertType.HIGH_LATENCY,
                severity,
                f"{operation} latency high: {latency_ms:.0f}ms",
                metric_name=f"latency_{operation}",
                metric_value=latency_ms,
                threshold=threshold_ms,
            )
    
    def check_error_rate(self, errors_per_minute: float, threshold: float = 5):
        """Vérifier le taux d'erreurs"""
        self.error_detector.add_value(errors_per_minute)
        
        if errors_per_minute > threshold:
            severity = AlertSeverity.CRITICAL if errors_per_minute > threshold * 2 else AlertSeverity.WARNING
            self.create_alert(
                AlertType.HIGH_ERROR_RATE,
                severity,
                f"Error rate high: {errors_per_minute:.1f} errors/min",
                metric_name="errors_per_minute",
                metric_value=errors_per_minute,
                threshold=threshold,
            )
    
    # --- Alertes Trading ---
    def check_drawdown(self, current_drawdown: float, max_threshold: float = -20):
        """Vérifier le drawdown"""
        if current_drawdown < max_threshold:
            severity = AlertSeverity.CRITICAL if current_drawdown < max_threshold * 1.5 else AlertSeverity.WARNING
            self.create_alert(
                AlertType.HIGH_DRAWDOWN,
                severity,
                f"Drawdown critical: {current_drawdown:.2f}%",
                metric_name="current_drawdown",
                metric_value=current_drawdown,
                threshold=max_threshold,
            )
    
    def check_win_rate(self, win_rate: float, min_threshold: float = 30):
        """Vérifier le win rate minimum"""
        if win_rate < min_threshold and win_rate >= 0:  # Ignorer les -1 (no data)
            severity = AlertSeverity.WARNING if win_rate > min_threshold * 0.8 else AlertSeverity.CRITICAL
            self.create_alert(
                AlertType.LOW_WIN_RATE,
                severity,
                f"Win rate low: {win_rate:.1f}%",
                metric_name="win_rate",
                metric_value=win_rate,
                threshold=min_threshold,
            )
    
    def check_pnl(self, pnl: float, daily_pnl: float):
        """Vérifier le P&L"""
        self.pnl_detector.add_value(pnl)
        
        if pnl < 0:
            severity = AlertSeverity.WARNING if pnl > -1000 else AlertSeverity.CRITICAL
            self.create_alert(
                AlertType.NEGATIVE_PNL,
                severity,
                f"Negative P&L: {pnl:.2f}",
                metric_name="total_pnl",
                metric_value=pnl,
            )
        
        # Détection de déclin brutal
        if daily_pnl < -5000:  # Perte > 5k en un jour
            self.create_alert(
                AlertType.SHARP_DECLINE,
                AlertSeverity.CRITICAL,
                f"Sharp decline detected: {daily_pnl:.2f} today",
                metric_name="daily_pnl",
                metric_value=daily_pnl,
            )
    
    # --- Alertes Data ---
    def check_data_drift(self, drift_score: float, threshold: float = 0.7):
        """Vérifier la dérive des données"""
        if drift_score > threshold:
            severity = AlertSeverity.WARNING if drift_score < 0.9 else AlertSeverity.CRITICAL
            self.create_alert(
                AlertType.DATA_DRIFT,
                severity,
                f"Data drift detected: {drift_score:.3f}",
                metric_name="data_drift_score",
                metric_value=drift_score,
                threshold=threshold,
            )
    
    def check_model_drift(self, drift_score: float, threshold: float = 0.6):
        """Vérifier la dérive du modèle"""
        if drift_score > threshold:
            severity = AlertSeverity.WARNING if drift_score < 0.8 else AlertSeverity.CRITICAL
            self.create_alert(
                AlertType.MODEL_DRIFT,
                severity,
                f"Model drift detected: {drift_score:.3f}",
                metric_name="model_drift_score",
                metric_value=drift_score,
                threshold=threshold,
            )
    
    # --- Gestion des alertes ---
    def acknowledge_alert(self, alert_id: str):
        """Reconnaître une alerte"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtenir toutes les alertes non reconnues"""
        return [a for a in self.alerts.values() if not a.acknowledged]
    
    def get_critical_alerts(self) -> List[Alert]:
        """Obtenir toutes les alertes critiques"""
        return [a for a in self.alerts.values() 
                if a.severity == AlertSeverity.CRITICAL and not a.acknowledged]
    
    def clear_old_alerts(self, hours: int = 24):
        """Effacer les anciennes alertes"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        to_delete = []
        
        for alert_id, alert in self.alerts.items():
            alert_time = datetime.fromisoformat(alert.timestamp)
            if alert_time < cutoff:
                to_delete.append(alert_id)
        
        for alert_id in to_delete:
            del self.alerts[alert_id]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des alertes"""
        critical_alerts = self.get_critical_alerts()
        all_unack = self.get_active_alerts()
        
        alert_counts = {}
        for alert in all_unack:
            alert_counts[alert.alert_type.value] = alert_counts.get(alert.alert_type.value, 0) + 1
        
        return {
            'total_unacknowledged': len(all_unack),
            'critical_count': len(critical_alerts),
            'alert_types': alert_counts,
            'critical_alerts': [
                {
                    'id': a.alert_id,
                    'type': a.alert_type.value,
                    'message': a.message,
                    'timestamp': a.timestamp,
                }
                for a in critical_alerts[:10]
            ],
        }
