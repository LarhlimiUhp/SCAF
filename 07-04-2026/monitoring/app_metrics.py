"""
Collecteur de métriques applicatives : Latence, Throughput, Erreurs
"""

import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque
import functools


@dataclass
class LatencyMetric:
    """Métrique de latence"""
    timestamp: str
    operation: str
    duration_ms: float
    percentile_p50: float
    percentile_p95: float
    percentile_p99: float
    avg: float
    min: float
    max: float
    count: int


@dataclass
class ErrorMetric:
    """Métrique d'erreurs"""
    timestamp: str
    total_errors: int
    errors_per_minute: float
    error_types: Dict[str, int]
    last_error_msg: Optional[str] = None
    last_error_time: Optional[str] = None


@dataclass
class ThroughputMetric:
    """Métrique de débit"""
    timestamp: str
    operation: str
    items_per_second: float
    items_processed: int
    bytes_processed_mb: float
    duration_seconds: float


class ApplicationMetricsCollector:
    """Collecteur de métriques applicatives"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.latencies: Dict[str, deque] = {}
        self.throughputs: Dict[str, dict] = {}
        self.errors: List[Dict] = []
        self.operation_counts: Dict[str, int] = {}
        self.start_time = time.time()
        
    def record_latency(self, operation: str, duration: float):
        """Enregistrer une métrique de latence (en secondes)"""
        duration_ms = duration * 1000
        
        if operation not in self.latencies:
            self.latencies[operation] = deque(maxlen=self.max_history)
        
        self.latencies[operation].append(duration_ms)
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
    
    def record_error(self, error_type: str, message: str = ""):
        """Enregistrer une erreur"""
        self.errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': message,
        })
        
        # Garder que les 1000 dernières erreurs
        if len(self.errors) > 1000:
            self.errors = self.errors[-1000:]
    
    def record_throughput(self, operation: str, items: int, bytes_processed: int = 0):
        """Enregistrer un débit"""
        now = time.time()
        
        if operation not in self.throughputs:
            self.throughputs[operation] = {
                'items': 0,
                'bytes': 0,
                'start_time': now,
                'last_update': now
            }
        
        self.throughputs[operation]['items'] += items
        self.throughputs[operation]['bytes'] += bytes_processed
        self.throughputs[operation]['last_update'] = now
    
    def get_latency_metric(self, operation: str) -> Optional[LatencyMetric]:
        """Obtenir les métriques de latence pour une opération"""
        if operation not in self.latencies or not self.latencies[operation]:
            return None
        
        data = np.array(list(self.latencies[operation]))
        
        return LatencyMetric(
            timestamp=datetime.utcnow().isoformat(),
            operation=operation,
            duration_ms=float(data[-1]),  # Last measurement
            percentile_p50=float(np.percentile(data, 50)),
            percentile_p95=float(np.percentile(data, 95)),
            percentile_p99=float(np.percentile(data, 99)),
            avg=float(np.mean(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            count=len(data),
        )
    
    def get_error_metric(self) -> ErrorMetric:
        """Obtenir les métriques d'erreurs"""
        error_types = {}
        last_error_msg = None
        last_error_time = None
        
        # Compter les erreurs des 60 dernières secondes
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=60)
        errors_per_minute = 0
        
        for error in self.errors:
            error_time = datetime.fromisoformat(error['timestamp'])
            error_types[error['type']] = error_types.get(error['type'], 0) + 1
            
            if error_time > cutoff:
                errors_per_minute += 1
            
            last_error_msg = error.get('message')
            last_error_time = error.get('timestamp')
        
        return ErrorMetric(
            timestamp=datetime.utcnow().isoformat(),
            total_errors=len(self.errors),
            errors_per_minute=errors_per_minute,
            error_types=error_types,
            last_error_msg=last_error_msg,
            last_error_time=last_error_time,
        )
    
    def get_throughput_metric(self, operation: str) -> Optional[ThroughputMetric]:
        """Obtenir les métriques de débit"""
        if operation not in self.throughputs:
            return None
        
        tp = self.throughputs[operation]
        elapsed = tp['last_update'] - tp['start_time']
        
        if elapsed <= 0:
            return None
        
        return ThroughputMetric(
            timestamp=datetime.utcnow().isoformat(),
            operation=operation,
            items_per_second=tp['items'] / elapsed if elapsed > 0 else 0,
            items_processed=tp['items'],
            bytes_processed_mb=tp['bytes'] / 1024 / 1024,
            duration_seconds=elapsed,
        )
    
    def get_all_latency_metrics(self) -> Dict[str, LatencyMetric]:
        """Obtenir toutes les métriques de latence"""
        result = {}
        for op in self.latencies:
            metric = self.get_latency_metric(op)
            if metric:
                result[op] = metric
        return result
    
    def reset_operation(self, operation: str):
        """Réinitialiser les métriques d'une opération"""
        if operation in self.latencies:
            self.latencies[operation].clear()
        if operation in self.throughputs:
            self.throughputs[operation] = {
                'items': 0,
                'bytes': 0,
                'start_time': time.time(),
                'last_update': time.time()
            }
    
    def get_health_score(self) -> Dict[str, Any]:
        """Score de santé applicatif (0-100)"""
        scores = []
        
        # Score basé sur les erreurs
        error_metric = self.get_error_metric()
        error_score = max(0, 100 - error_metric.errors_per_minute * 10)
        scores.append(error_score)
        
        # Score basé sur la latence
        if self.latencies:
            for op in self.latencies:
                metric = self.get_latency_metric(op)
                if metric and metric.percentile_p99 > 1000:  # > 1 seconde
                    latency_score = max(0, 100 - (metric.percentile_p99 - 1000) / 10)
                    scores.append(latency_score)
        
        return {
            'overall_health': float(np.mean(scores)) if scores else 100.0,
            'error_score': float(error_score),
            'error_count': error_metric.total_errors,
            'errors_per_minute': error_metric.errors_per_minute,
        }


# Décorateur pour instrumenter les fonctions
def track_latency(operation_name: str):
    """Décorateur pour tracer la latence d'une fonction"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = _get_collector()
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                collector.record_error(type(e).__name__, str(e))
                raise
            finally:
                elapsed = time.time() - start
                collector.record_latency(operation_name, elapsed)
        return wrapper
    return decorator


# Instance globale
_collector_instance: Optional[ApplicationMetricsCollector] = None

def get_collector() -> ApplicationMetricsCollector:
    """Récupérer l'instance globale du collecteur"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = ApplicationMetricsCollector()
    return _collector_instance

def _get_collector() -> ApplicationMetricsCollector:
    """Alias interne"""
    return get_collector()
