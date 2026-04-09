"""
Health Checks automatiques - Vérification système, modèle, données
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import psutil


@dataclass
class HealthCheckResult:
    """Résultat d'un health check"""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    details: Dict[str, Any]
    error_message: Optional[str] = None


class HealthCheckSystem:
    """Système de health checks automatiques"""
    
    def __init__(self):
        self.checks: Dict[str, Any] = {}
        self.results: List[HealthCheckResult] = []
        
    def check_system_resources(self) -> HealthCheckResult:
        """Vérifier les ressources système"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_count = psutil.cpu_count()
            
            details = {
                'cpu_count': cpu_count,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_available_gb': disk.free / 1024 / 1024 / 1024,
            }
            
            # Déterminer le statut
            if memory.percent > 95 or disk.percent > 90:
                status = "unhealthy"
            elif memory.percent > 85 or disk.percent > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                details={},
                error_message=str(e),
            )
    
    def check_data_availability(self, data_loader: Optional[Any] = None) -> HealthCheckResult:
        """Vérifier la disponibilité des données"""
        try:
            details = {
                'data_loader_available': data_loader is not None,
            }
            
            if data_loader:
                try:
                    # Vérifier si on peut charger les données
                    test_data = data_loader.download()
                    details['data_loadable'] = test_data is not None
                    details['last_load_time'] = datetime.utcnow().isoformat()
                    status = "healthy" if test_data else "degraded"
                except Exception as e:
                    details['data_loadable'] = False
                    details['error'] = str(e)
                    status = "unhealthy"
            else:
                status = "degraded"
            
            return HealthCheckResult(
                name="data_availability",
                status=status,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
            )
        except Exception as e:
            return HealthCheckResult(
                name="data_availability",
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                details={},
                error_message=str(e),
            )
    
    def check_model_status(self, models: Optional[Dict[str, Any]] = None) -> HealthCheckResult:
        """Vérifier l'état des modèles"""
        try:
            details = {
                'models_count': len(models) if models else 0,
                'models_loaded': [],
            }
            
            if models:
                for name, model in models.items():
                    try:
                        # Vérifier que le modèle est accessible
                        details['models_loaded'].append({
                            'name': name,
                            'type': type(model).__name__,
                            'status': 'loaded'
                        })
                    except Exception as e:
                        details['models_loaded'].append({
                            'name': name,
                            'status': 'error',
                            'error': str(e)
                        })
                
                loaded_count = sum(1 for m in details['models_loaded'] if m['status'] == 'loaded')
                
                if loaded_count == len(models):
                    status = "healthy"
                elif loaded_count > 0:
                    status = "degraded"
                else:
                    status = "unhealthy"
            else:
                status = "degraded"
            
            return HealthCheckResult(
                name="model_status",
                status=status,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
            )
        except Exception as e:
            return HealthCheckResult(
                name="model_status",
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                details={},
                error_message=str(e),
            )
    
    def check_feature_validity(self, X: Optional[Any] = None) -> HealthCheckResult:
        """Vérifier la validité des features"""
        try:
            details = {}
            
            if X is not None:
                import numpy as np
                
                X_array = X.values if hasattr(X, 'values') else X
                
                details['total_features'] = X_array.shape[1] if len(X_array.shape) > 1 else 1
                details['total_samples'] = X_array.shape[0]
                details['nan_count'] = int(np.isnan(X_array).sum())
                details['inf_count'] = int(np.isinf(X_array).sum())
                details['finite_count'] = int(np.isfinite(X_array).sum())
                
                nan_ratio = details['nan_count'] / X_array.size if X_array.size > 0 else 0
                inf_ratio = details['inf_count'] / X_array.size if X_array.size > 0 else 0
                
                if nan_ratio > 0.1 or inf_ratio > 0.01:
                    status = "unhealthy"
                elif nan_ratio > 0.01 or inf_ratio > 0.001:
                    status = "degraded"
                else:
                    status = "healthy"
            else:
                status = "degraded"
            
            return HealthCheckResult(
                name="feature_validity",
                status=status,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
            )
        except Exception as e:
            return HealthCheckResult(
                name="feature_validity",
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                details={},
                error_message=str(e),
            )
    
    def check_backtest_validity(self, backtest_results: Optional[Dict[str, Any]] = None) -> HealthCheckResult:
        """Vérifier la validité des résultats de backtest"""
        try:
            details = {}
            
            if backtest_results and 'backtest' in backtest_results:
                backtest_df = backtest_results['backtest']
                details['rows'] = len(backtest_df)
                
                if 'equity' in backtest_df.columns:
                    details['final_equity'] = float(backtest_df['equity'].iloc[-1])
                    details['max_equity'] = float(backtest_df['equity'].max())
                    details['min_equity'] = float(backtest_df['equity'].min())
                
                if 'strategy_return' in backtest_df.columns:
                    import numpy as np
                    returns = backtest_df['strategy_return'].dropna()
                    details['avg_return'] = float(returns.mean())
                    details['std_return'] = float(returns.std())
                
                # Vérifier la cohérence
                if details.get('final_equity', 0) > 0 and details.get('rows', 0) > 0:
                    status = "healthy"
                else:
                    status = "degraded"
            else:
                status = "degraded"
            
            return HealthCheckResult(
                name="backtest_validity",
                status=status,
                timestamp=datetime.utcnow().isoformat(),
                details=details,
            )
        except Exception as e:
            return HealthCheckResult(
                name="backtest_validity",
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                details={},
                error_message=str(e),
            )
    
    def run_all_checks(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, HealthCheckResult]:
        """Exécuter tous les health checks"""
        context = context or {}
        
        results = {
            'system_resources': self.check_system_resources(),
            'data_availability': self.check_data_availability(context.get('data_loader')),
            'model_status': self.check_model_status(context.get('models')),
            'feature_validity': self.check_feature_validity(context.get('features')),
            'backtest_validity': self.check_backtest_validity(context.get('backtest_results')),
        }
        
        self.results.extend(results.values())
        
        return results
    
    def get_overall_health(self, checks: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Obtenir la santé globale du système"""
        statuses = [check.status for check in checks.values()]
        
        # Scorer: healthy=100, degraded=50, unhealthy=0
        scores = {
            'healthy': 100,
            'degraded': 50,
            'unhealthy': 0,
        }
        
        overall_score = sum(scores.get(s, 0) for s in statuses) / len(statuses) if statuses else 50
        
        # Déterminer le statut global
        if 'unhealthy' in statuses:
            overall_status = 'unhealthy'
        elif 'degraded' in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_health_score': overall_score,
            'overall_status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'detailed_checks': {
                name: {
                    'status': check.status,
                    'details': check.details,
                    'error': check.error_message,
                }
                for name, check in checks.items()
            },
        }
