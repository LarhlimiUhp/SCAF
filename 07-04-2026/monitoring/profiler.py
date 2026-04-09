"""
Performance Profiler - Détection de bottlenecks et profiling continu
"""

import cProfile
import pstats
import io
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from collections import OrderedDict
import time


@dataclass
class ProfileResult:
    """Résultat d'un profiling"""
    function_name: str
    total_time: float
    calls: int
    time_per_call: float
    cumulative_time: float


class PerformanceProfiler:
    """Profiler de performance avec bottleneck detection"""
    
    def __init__(self):
        self.profiles: Dict[str, Any] = {}
        self.bottlenecks: Dict[str, ProfileResult] = {}
        self.function_times: Dict[str, list] = {}
        
    def profile_function(self, func_name: str) -> Callable:
        """Décorateur pour profiler une fonction"""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                profiler.enable()
                
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                profiler.disable()
                
                # Enregistrer le résultat
                if func_name not in self.function_times:
                    self.function_times[func_name] = []
                self.function_times[func_name].append(elapsed)
                
                # Enregistrer le profil
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(10)  # Top 10 fonctions
                
                self.profiles[func_name] = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'total_time': elapsed,
                    'profile_output': s.getvalue(),
                }
                
                return result
            
            return wrapper
        return decorator
    
    def detect_bottlenecks(self, threshold_percentile: float = 90) -> Dict[str, ProfileResult]:
        """Détecter les bottlenecks"""
        import numpy as np
        
        bottlenecks = {}
        
        for func_name, times in self.function_times.items():
            if not times:
                continue
            
            times_array = np.array(times)
            threshold = np.percentile(times_array, threshold_percentile)
            avg_time = np.mean(times_array)
            
            if avg_time > 0.1:  # Plus que 100ms
                bottlenecks[func_name] = ProfileResult(
                    function_name=func_name,
                    total_time=float(np.sum(times_array)),
                    calls=len(times_array),
                    time_per_call=float(avg_time),
                    cumulative_time=float(np.sum(times_array)),
                )
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def get_slowest_functions(self, n: int = 10) -> Dict[str, ProfileResult]:
        """Obtenir les fonctions les plus lentes"""
        if not self.function_times:
            return {}
        
        import numpy as np
        
        avg_times = {}
        for func_name, times in self.function_times.items():
            avg_times[func_name] = np.mean(times)
        
        sorted_funcs = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
        
        result = {}
        for func_name, avg_time in sorted_funcs[:n]:
            times = self.function_times[func_name]
            result[func_name] = ProfileResult(
                function_name=func_name,
                total_time=float(sum(times)),
                calls=len(times),
                time_per_call=float(avg_time),
                cumulative_time=float(sum(times)),
            )
        
        return result
    
    def get_memory_profile(self, func: Callable) -> Dict[str, Any]:
        """Profiler l'utilisation mémoire d'une fonction"""
        try:
            import tracemalloc
            
            tracemalloc.start()
            
            start_time = time.time()
            func()
            elapsed = time.time() - start_time
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {
                'elapsed_time': elapsed,
                'current_memory_mb': current / 1024 / 1024,
                'peak_memory_mb': peak / 1024 / 1024,
                'memory_increase_mb': (peak - current) / 1024 / 1024,
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé de la performance"""
        import numpy as np
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_functions_profiled': len(self.function_times),
            'bottlenecks': len(self.bottlenecks),
        }
        
        if self.function_times:
            all_times = []
            for times in self.function_times.values():
                all_times.extend(times)
            
            summary['total_calls'] = len(all_times)
            summary['avg_execution_time_ms'] = float(np.mean(all_times) * 1000)
            summary['max_execution_time_ms'] = float(np.max(all_times) * 1000)
            summary['p95_execution_time_ms'] = float(np.percentile(all_times, 95) * 1000)
        
        if self.bottlenecks:
            summary['slowest_function'] = min(
                self.bottlenecks.items(),
                key=lambda x: x[1].time_per_call
            )[0]
        
        return summary
    
    def get_profile_report(self, func_name: str) -> Optional[str]:
        """Obtenir le rapport de profiling pour une fonction"""
        if func_name in self.profiles:
            return self.profiles[func_name].get('profile_output')
        return None
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporter les résultats pour Prometheus/Grafana"""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'functions': {},
            'bottlenecks': {},
        }
        
        for func_name, times in self.function_times.items():
            import numpy as np
            data['functions'][func_name] = {
                'calls': len(times),
                'total_time': float(sum(times)),
                'avg_time': float(np.mean(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'p95_time': float(np.percentile(times, 95)),
            }
        
        for func_name, result in self.bottlenecks.items():
            data['bottlenecks'][func_name] = {
                'total_time': result.total_time,
                'calls': result.calls,
                'avg_time': result.time_per_call,
            }
        
        return data
