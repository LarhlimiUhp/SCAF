"""
Maintenance Prédictive - Prévention de pannes et optimisation continue
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque


@dataclass
class MaintenanceTask:
    """Tâche de maintenance"""
    task_id: str
    task_type: str  # "retraining", "cleanup", "optimization", "restart"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommended_time: str
    estimated_duration_minutes: int
    priority_score: float  # 0-100
    status: str = "pending"  # "pending", "running", "completed", "failed"


class PredictiveMaintenanceEngine:
    """Moteur de maintenance prédictive"""
    
    def __init__(self):
        self.tasks: Dict[str, MaintenanceTask] = {}
        self.task_counter = 0
        self.maintenance_history: deque = deque(maxlen=1000)
        self.last_retraining: Optional[datetime] = None
        self.last_cleanup: Optional[datetime] = None
        self.last_optimization: Optional[datetime] = None
        self.start_time = datetime.utcnow()
        
    def create_maintenance_task(self, task_type: str, severity: str,
                               description: str, duration_minutes: int = 15) -> MaintenanceTask:
        """Créer une tâche de maintenance"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        severity_scores = {
            'low': 10,
            'medium': 40,
            'high': 70,
            'critical': 95,
        }
        
        task = MaintenanceTask(
            task_id=task_id,
            task_type=task_type,
            severity=severity,
            description=description,
            recommended_time=datetime.utcnow().isoformat(),
            estimated_duration_minutes=duration_minutes,
            priority_score=float(severity_scores.get(severity, 50)),
        )
        
        self.tasks[task_id] = task
        self.maintenance_history.append(task)
        
        return task
    
    def check_retraining_needed(self, drift_score: float, time_since_training_days: float,
                                model_performance_degradation: float) -> bool:
        """Vérifier si réentraînement est nécessaire"""
        # Condition 1: Drift élevé
        if drift_score > 0.6:
            self._schedule_retraining("Model drift detected", severity="high")
            return True
        
        # Condition 2: Performance dégradée
        if model_performance_degradation > 0.15:  # 15% dégradation
            self._schedule_retraining("Performance degradation detected", severity="high")
            return True
        
        # Condition 3: Réentraînement prévu régulièrement (ex. chaque 10 jours)
        if time_since_training_days > 10:
            self._schedule_retraining("Scheduled maintenance retraining", severity="medium")
            return True
        
        return False
    
    def check_cleanup_needed(self, log_size_mb: float, cache_size_mb: float,
                            total_disk_free_gb: float) -> bool:
        """Vérifier si nettoyage est nécessaire"""
        cleanup_needed = False
        
        # Condition 1: Logs trop volumineux
        if log_size_mb > 1000:  # > 1GB
            self._schedule_cleanup("Large log files", severity="medium")
            cleanup_needed = True
        
        # Condition 2: Cache trop volumineux
        if cache_size_mb > 500:  # > 500MB
            self._schedule_cleanup("Large cache", severity="low")
            cleanup_needed = True
        
        # Condition 3: Disque faible
        if total_disk_free_gb < 10:
            self._schedule_cleanup("Disk space low", severity="critical")
            cleanup_needed = True
        
        return cleanup_needed
    
    def check_optimization_needed(self, system_memory_percent: float,
                                 cpu_efficiency: float,
                                 error_rate: float) -> bool:
        """Vérifier si optimisation est nécessaire"""
        optimization_needed = False
        
        # Utilisation mémoire élevée
        if system_memory_percent > 85:
            self._schedule_optimization("Memory pressure", severity="high")
            optimization_needed = True
        
        # Efficacité CPU faible (variance haute)
        if cpu_efficiency < 0.5:
            self._schedule_optimization("CPU efficiency low", severity="medium")
            optimization_needed = True
        
        # Taux d'erreurs élevé
        if error_rate > 0.05:  # > 5%
            self._schedule_optimization("Error rate high", severity="high")
            optimization_needed = True
        
        return optimization_needed
    
    def check_restart_needed(self, uptime_days: float, memory_leak_detected: bool,
                            critical_errors_count: int) -> bool:
        """Vérifier si restart est nécessaire"""
        restart_needed = False
        
        # Uptime élevé (prévention de fuites mémoire)
        if uptime_days > 30:
            self._schedule_restart("Long uptime", severity="low")
            restart_needed = True
        
        # Fuite mémoire détectée
        if memory_leak_detected:
            self._schedule_restart("Memory leak detected", severity="critical")
            restart_needed = True
        
        # Erreurs critiques accumulées
        if critical_errors_count > 10:
            self._schedule_restart("Multiple critical errors", severity="high")
            restart_needed = True
        
        return restart_needed
    
    def _schedule_retraining(self, reason: str, severity: str = "medium"):
        """Planifier un réentraînement"""
        self.create_maintenance_task(
            task_type="retraining",
            severity=severity,
            description=f"Model retraining: {reason}",
            duration_minutes=60,
        )
        self.last_retraining = datetime.utcnow()
    
    def _schedule_cleanup(self, reason: str, severity: str = "low"):
        """Planifier un nettoyage"""
        self.create_maintenance_task(
            task_type="cleanup",
            severity=severity,
            description=f"System cleanup: {reason}",
            duration_minutes=15,
        )
        self.last_cleanup = datetime.utcnow()
    
    def _schedule_optimization(self, reason: str, severity: str = "medium"):
        """Planifier une optimisation"""
        self.create_maintenance_task(
            task_type="optimization",
            severity=severity,
            description=f"System optimization: {reason}",
            duration_minutes=30,
        )
        self.last_optimization = datetime.utcnow()
    
    def _schedule_restart(self, reason: str, severity: str = "medium"):
        """Planifier un restart"""
        self.create_maintenance_task(
            task_type="restart",
            severity=severity,
            description=f"System restart: {reason}",
            duration_minutes=10,
        )
    
    def get_pending_tasks(self) -> List[MaintenanceTask]:
        """Obtenir les tâches en attente"""
        return [t for t in self.tasks.values() if t.status == "pending"]
    
    def get_high_priority_tasks(self) -> List[MaintenanceTask]:
        """Obtenir les tâches haute priorité"""
        pending = self.get_pending_tasks()
        return sorted(pending, key=lambda t: t.priority_score, reverse=True)[:5]
    
    def mark_task_completed(self, task_id: str, success: bool = True):
        """Marquer une tâche comme complétée"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed" if success else "failed"
    
    def get_maintenance_schedule(self) -> Dict[str, Any]:
        """Obtenir le calendrier de maintenance"""
        pending = self.get_pending_tasks()
        high_priority = self.get_high_priority_tasks()
        
        # Estimer l'uptime jusqu'au prochain événement
        uptime = datetime.utcnow() - self.start_time
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'pending_tasks': len(pending),
            'high_priority_tasks': len(high_priority),
            'total_duration_hours': sum(t.estimated_duration_minutes for t in pending) / 60,
            'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
            'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'system_uptime_days': uptime.days,
            'recommended_next_task': {
                'task_id': high_priority[0].task_id,
                'type': high_priority[0].task_type,
                'priority': high_priority[0].priority_score,
                'description': high_priority[0].description,
            } if high_priority else None,
        }
    
    def get_health_recommendations(self) -> List[str]:
        """Obtenir des recommandations d'amélioration"""
        recommendations = []
        
        high_priority = self.get_high_priority_tasks()
        
        if any(t.task_type == "retraining" for t in high_priority):
            recommendations.append("Perform model retraining ASAP - drift or performance issues detected")
        
        if any(t.task_type == "restart" for t in high_priority):
            recommendations.append("Schedule system restart - memory leaks or critical errors")
        
        if any(t.task_type == "cleanup" for t in high_priority):
            recommendations.append("Run cleanup operations - disk/cache issues")
        
        if any(t.task_type == "optimization" for t in high_priority):
            recommendations.append("Optimize system - resource pressure detected")
        
        if not recommendations:
            recommendations.append("System in good health - continue monitoring")
        
        return recommendations
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporter les données de maintenance"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'schedule': self.get_maintenance_schedule(),
            'recommendations': self.get_health_recommendations(),
            'tasks': {
                task_id: {
                    'type': task.task_type,
                    'severity': task.severity,
                    'status': task.status,
                    'priority': task.priority_score,
                    'description': task.description,
                }
                for task_id, task in self.tasks.items()
            },
        }
