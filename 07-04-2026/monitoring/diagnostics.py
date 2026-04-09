"""
Vérification et diagnostic du système de monitoring SCAF-LS
"""

import sys
from pathlib import Path
import importlib.util

def check_module_installed(module_name: str) -> bool:
    """Vérifier si un module est installé"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def run_diagnostics():
    """Exécuter les diagnostiques complets"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     SCAF-LS Monitoring System - Diagnostic Report        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 1. Vérifier les modules requis
    print("\n📋 1. Checking Required Modules...")
    required_modules = {
        'psutil': 'System metrics collection',
        'numpy': 'Numerical computations',
        'pandas': 'Data manipulation',
        'scipy': 'Statistical functions',
    }
    
    missing_modules = []
    for module, description in required_modules.items():
        if check_module_installed(module):
            print(f"  ✅ {module:15} - {description}")
        else:
            print(f"  ❌ {module:15} - {description} (MISSING)")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n  ⚠️  Install missing modules:")
        print(f"     pip install {' '.join(missing_modules)}")
    
    # 2. Vérifier les fichiers du monitoring
    print("\n📦 2. Checking Monitoring Files...")
    
    monitoring_dir = Path(__file__).parent
    required_files = {
        '__init__.py': 'Package init',
        'logger.py': 'Structured logging',
        'system_metrics.py': 'System metrics collection',
        'app_metrics.py': 'Application metrics',
        'business_metrics.py': 'Business/trading metrics',
        'alerts.py': 'Alert system',
        'health_checks.py': 'Health checks',
        'profiler.py': 'Performance profiling',
        'drift_detection.py': 'Drift detection',
        'maintenance.py': 'Predictive maintenance',
        'monitoring_service.py': 'Main service',
        'integration_guide.py': 'Integration examples',
        'demo.py': 'Demo script',
        'README.md': 'Documentation',
    }
    
    all_files_ok = True
    for filename, description in required_files.items():
        filepath = monitoring_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✅ {filename:25} - {description} ({size:,} bytes)")
        else:
            print(f"  ❌ {filename:25} - {description} (MISSING)")
            all_files_ok = False
    
    # 3. Vérifier le contenu principal
    print("\n🔍 3. Checking Core Functionality...")
    
    checks = []
    
    try:
        from scaf_ls.monitoring.logger import get_logger
        checks.append(('Structured Logger', True))
    except Exception as e:
        checks.append(('Structured Logger', False, str(e)))
    
    try:
        from scaf_ls.monitoring.system_metrics import SystemMetricsCollector
        checks.append(('System Metrics', True))
    except Exception as e:
        checks.append(('System Metrics', False, str(e)))
    
    try:
        from scaf_ls.monitoring.app_metrics import ApplicationMetricsCollector
        checks.append(('App Metrics', True))
    except Exception as e:
        checks.append(('App Metrics', False, str(e)))
    
    try:
        from scaf_ls.monitoring.business_metrics import BusinessMetricsCollector
        checks.append(('Business Metrics', True))
    except Exception as e:
        checks.append(('Business Metrics', False, str(e)))
    
    try:
        from scaf_ls.monitoring.alerts import AlertSystem
        checks.append(('Alert System', True))
    except Exception as e:
        checks.append(('Alert System', False, str(e)))
    
    try:
        from scaf_ls.monitoring.health_checks import HealthCheckSystem
        checks.append(('Health Checks', True))
    except Exception as e:
        checks.append(('Health Checks', False, str(e)))
    
    try:
        from scaf_ls.monitoring.drift_detection import DriftDetector
        checks.append(('Drift Detector', True))
    except Exception as e:
        checks.append(('Drift Detector', False, str(e)))
    
    try:
        from scaf_ls.monitoring.maintenance import PredictiveMaintenanceEngine
        checks.append(('Maintenance Engine', True))
    except Exception as e:
        checks.append(('Maintenance Engine', False, str(e)))
    
    try:
        from scaf_ls.monitoring.monitoring_service import MonitoringService
        checks.append(('Monitoring Service', True))
    except Exception as e:
        checks.append(('Monitoring Service', False, str(e)))
    
    all_ok = True
    for check in checks:
        if len(check) == 2:
            name, ok = check
            icon = "✅" if ok else "❌"
            print(f"  {icon} {name}")
        else:
            name, ok, error = check
            print(f"  ❌ {name} - {error}")
            all_ok = False
    
    # 4. Vérifier les capacités système
    print("\n💻 4. Checking System Capabilities...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        print(f"  ✅ CPU:          {cpu_count} cores")
        print(f"  ✅ Memory:       {memory.total / 1024**3:.1f} GB")
        print(f"  ✅ Available:    {memory.available / 1024**3:.1f} GB")
    except Exception as e:
        print(f"  ❌ System info: {e}")
    
    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU:          {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⓘ GPU:          Not available (CPU mode)")
    except:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                print(f"  ✅ GPU:          {gpus[0].name}")
            else:
                print(f"  ⓘ GPU:          Not detected")
        except:
            print(f"  ⓘ GPU:          Detection disabled")
    
    # 5. Test rapide du monitoring
    print("\n🧪 5. Quick Functionality Test...")
    
    try:
        from scaf_ls.monitoring import get_monitoring_service, start_monitoring
        
        monitoring = start_monitoring()
        
        # Test système
        sys_metrics = monitoring.system_metrics.collect()
        print(f"  ✅ System metrics: CPU {sys_metrics.cpu_percent:.1f}%")
        
        # Test app metrics
        app_metrics = monitoring.app_metrics
        print(f"  ✅ App metrics: ready")
        
        # Test business metrics
        monitoring.business_metrics.update_equity(100000)
        print(f"  ✅ Business metrics: ready")
        
        # Test alerts
        monitoring.alerts.check_cpu_usage(50)
        print(f"  ✅ Alerts: ready")
        
        # Test health checks
        checks = monitoring.health_checks.run_all_checks()
        print(f"  ✅ Health checks: {len(checks)} checks performed")
        
        from scaf_ls.monitoring import stop_monitoring
        stop_monitoring()
        
        print(f"\n  ✅ All systems operational!")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
    
    # 6. Recommendations
    print("\n📚 6. Recommendations...")
    
    if missing_modules:
        print(f"  • Install missing modules: pip install {' '.join(missing_modules)}")
    
    if not all_files_ok:
        print(f"  • Check that all monitoring files are present")
    
    if not all_ok:
        print(f"  • Some core modules failed to import - check errors above")
    
    if check_module_installed('torch'):
        print(f"  • PyTorch detected - GPU monitoring available")
    
    if check_module_installed('prometheus_client'):
        print(f"  • Prometheus client installed - metrics export ready")
    else:
        print(f"  • Optional: pip install prometheus-client for Prometheus export")
    
    print(f"""
    ✅ Getting Started:
    
    1. Quick demo:
       python -m scaf_ls.monitoring.demo
    
    2. Integration with SCAF-LS:
       See scaf_ls/monitoring/INTEGRATION_MINIMAL.md
    
    3. Full documentation:
       See scaf_ls/monitoring/README.md
    4. Examples:
       See scaf_ls/monitoring/integration_guide.py
    """)


def print_file_structure():
    """Afficher la structure des fichiers du monitoring"""
    
    print("""
    📁 Monitoring System File Structure
    =====================================
    
    scaf_ls/monitoring/
    ├── __init__.py                    # Package exports
    ├── logger.py                      # Structured logging (JSON + console)
    ├── system_metrics.py              # CPU, RAM, GPU, Disk, Network metrics
    ├── app_metrics.py                 # Latency, Throughput, Errors tracking
    ├── business_metrics.py            # P&L, Drawdown, Sharpe, Win Rate tracking
    ├── alerts.py                      # Intelligent alert system
    ├── health_checks.py               # System, Data, Model health verification
    ├── profiler.py                    # Performance profiling & bottleneck detection
    ├── drift_detection.py             # Data, Model, Concept drift detection
    ├── maintenance.py                 # Predictive maintenance scheduling
    ├── monitoring_service.py          # Main orchestrating service
    ├── integration_guide.py           # Integration examples & best practices
    ├── demo.py                        # Demonstration script
    ├── prometheus_grafana_setup.py    # Grafana/Prometheus configuration
    ├── requirements_monitoring.txt    # Dependencies
    ├── README.md                      # Full documentation
    └── INTEGRATION_MINIMAL.md         # Quick integration guide
    
    Key Features:
    =============
    
    🎯 10 Monitoring Modules
      • System metrics (CPU, RAM, GPU, Disk)
      • Application latency & throughput
      • Business metrics (P&L, Sharpe, Drawdown)
      • Intelligent alerts with ML anomaly detection
      • Structured JSON logging
      • Health checks (system, data, model)
      • Performance profiling
      • Data/Model/Concept drift detection
      • Predictive maintenance
      • Prometheus/Grafana integration
    
    ⚡ Quick Integration
      • Add 3 lines of code to enable full monitoring
      • Automatic data collection
      • Non-blocking (thread-based)
      • Zero changes to existing business logic
    
    📊 Real-time Dashboards
      • System metrics dashboard
      • Trading performance metrics
      • Alert dashboard
      • Health status monitoring
    
    🔔 Intelligent Alerting
      • Dynamic thresholds (statistical)
      • Anomaly detection (Z-score based)
      • Multi-level severity (INFO, WARNING, CRITICAL)
      • Custom handlers (email, Slack, webhook)
    
    🔧 Advanced Features
      • GPU monitoring (CUDA, GPUtil)
      • Memory profiling
      • Bottleneck detection
      • Drift detection with multiple methods
      • Predictive maintenance scheduling
      • Performance trend analysis
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--structure":
        print_file_structure()
    else:
        run_diagnostics()
        print("\nRun with --structure to see file organization\n")
