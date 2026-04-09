"""
Collecteur de métriques système : CPU, RAM, GPU, Disque, Réseau
"""

import psutil
import platform
import time
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class SystemMetrics:
    """Métriques système instantanées"""
    timestamp: str
    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None
    process_cpu_percent: Optional[float] = None
    process_memory_mb: Optional[float] = None
    

class SystemMetricsCollector:
    """Collecteur de métriques système temps réel"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.hostname = platform.node()
        self.platform_name = platform.system()
        self.gpu_devices = self._detect_gpu_devices()
        self.last_net_io = psutil.net_io_counters()
        self.last_net_time = time.time()
        
    def _detect_gpu_devices(self) -> list:
        """Détecte les GPU disponibles"""
        devices = []
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        devices.append({
                            'backend': 'torch',
                            'index': i,
                            'name': torch.cuda.get_device_name(i)
                        })
            except Exception:
                pass
        
        if GPU_AVAILABLE and not devices:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    devices.append({
                        'backend': 'gputil',
                        'index': gpu.id,
                        'name': gpu.name
                    })
            except Exception:
                pass
        
        return devices
    
    def collect(self) -> SystemMetrics:
        """Collecter les métriques système"""
        now = datetime.utcnow().isoformat()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024
        memory_available_mb = memory.available / 1024 / 1024
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / 1024 / 1024 / 1024
        disk_free_gb = disk.free / 1024 / 1024 / 1024
        
        # Process-specific
        try:
            process_cpu = self.process.cpu_percent(interval=0.01)
            process_memory_mb = self.process.memory_info().rss / 1024 / 1024
        except:
            process_cpu = None
            process_memory_mb = None
        
        # GPU
        gpu_metrics = self._get_gpu_metrics()
        
        # Network
        net_sent_mb, net_recv_mb = self._get_network_throughput()
        
        return SystemMetrics(
            timestamp=now,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            gpu_available=len(self.gpu_devices) > 0,
            gpu_percent=gpu_metrics.get('gpu_percent'),
            gpu_memory_used_mb=gpu_metrics.get('gpu_memory_used_mb'),
            gpu_memory_percent=gpu_metrics.get('gpu_memory_percent'),
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory_mb,
        )
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Obtient les métriques GPU"""
        metrics = {}
        
        if not self.gpu_devices:
            return metrics
        
        try:
            if self.gpu_devices[0]['backend'] == 'torch' and TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    # GPU 0
                    gpu_mem = torch.cuda.memory_stats(0)
                    allocated = gpu_mem.get('allocated_bytes.all.current', 0) / 1024 / 1024
                    reserved = gpu_mem.get('reserved_bytes.all.current', 0) / 1024 / 1024
                    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    
                    metrics['gpu_percent'] = (allocated / total * 100) if total > 0 else 0
                    metrics['gpu_memory_used_mb'] = allocated
                    metrics['gpu_memory_percent'] = (allocated / total * 100) if total > 0 else 0
                    
        except Exception:
            pass
        
        if not metrics and GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_percent'] = gpu.load * 100
                    metrics['gpu_memory_used_mb'] = gpu.memoryUsed
                    metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
            except Exception:
                pass
        
        return metrics
    
    def _get_network_throughput(self) -> tuple:
        """Calcule le throughput réseau (MB/s)"""
        try:
            current_net_io = psutil.net_io_counters()
            current_time = time.time()
            
            time_delta = current_time - self.last_net_time
            if time_delta > 0:
                sent_delta = current_net_io.bytes_sent - self.last_net_io.bytes_sent
                recv_delta = current_net_io.bytes_recv - self.last_net_io.bytes_recv
                
                sent_mb = (sent_delta / 1024 / 1024) / time_delta
                recv_mb = (recv_delta / 1024 / 1024) / time_delta
            else:
                sent_mb = recv_mb = 0.0
            
            self.last_net_io = current_net_io
            self.last_net_time = current_time
            
            return sent_mb, recv_mb
        except Exception:
            return None, None
    
    def get_alerts_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Seuils d'alerte recommandés"""
        return {
            'cpu_percent': {'warning': 80, 'critical': 95},
            'memory_percent': {'warning': 85, 'critical': 95},
            'disk_percent': {'warning': 80, 'critical': 90},
            'gpu_percent': {'warning': 90, 'critical': 98},
        }
    
    def to_dict(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return asdict(metrics)
