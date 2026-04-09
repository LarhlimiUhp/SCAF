#!/usr/bin/env python3
"""
SCAF-LS Monitoring System - Quick Verification Script
Vérification que tous les composants fonctionnent
"""

def verify_installation():
    """Vérifier l'installation rapide"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║    SCAF-LS Monitoring System - Installation Verify        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    checks_passed = 0
    checks_total = 0
    
    # Test 1: Import modules
    print("\n✓ Testing Core Modules...")
    modules_to_test = [
        ('scaf_ls.monitoring', 'Monitoring Package'),
        ('scaf_ls.monitoring.logger', 'Structured Logger'),
        ('scaf_ls.monitoring.system_metrics', 'System Metrics'),
        ('scaf_ls.monitoring.app_metrics', 'App Metrics'),
        ('scaf_ls.monitoring.business_metrics', 'Business Metrics'),
        ('scaf_ls.monitoring.alerts', 'Alert System'),
        ('scaf_ls.monitoring.health_checks', 'Health Checks'),
        ('scaf_ls.monitoring.profiler', 'Performance Profiler'),
        ('scaf_ls.monitoring.drift_detection', 'Drift Detector'),
        ('scaf_ls.monitoring.maintenance', 'Maintenance Engine'),
        ('scaf_ls.monitoring.monitoring_service', 'Monitoring Service'),
    ]
    
    for module_name, description in modules_to_test:
        checks_total += 1
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
            checks_passed += 1
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
    
    # Test 2: Start monitoring
    print("\n✓ Testing Monitoring Start...")
    checks_total += 1
    try:
        from scaf_ls.monitoring import start_monitoring, stop_monitoring
        monitoring = start_monitoring()
        stop_monitoring()
        print(f"  ✅ Monitoring start/stop")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Monitoring start/stop: {e}")
    
    # Test 3: Collect metrics
    print("\n✓ Testing Metric Collection...")
    checks_total += 1
    try:
        from scaf_ls.monitoring import start_monitoring, stop_monitoring
        monitoring = start_monitoring()
        
        # Test system metrics
        sys_metrics = monitoring.system_metrics.collect()
        assert sys_metrics.cpu_percent >= 0, "CPU percent invalid"
        print(f"  ✅ System metrics (CPU: {sys_metrics.cpu_percent:.1f}%)")
        
        stop_monitoring()
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Metric collection: {e}")
    
    # Test 4: Logger
    print("\n✓ Testing Logger...")
    checks_total += 1
    try:
        from scaf_ls.monitoring.logger import get_logger
        logger = get_logger("test")
        logger.info("Test message", test_value=123)
        print(f"  ✅ Structured logging")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Logger: {e}")
    
    # Results
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║ Results: {checks_passed}/{checks_total} checks passed
    """)
    
    if checks_passed == checks_total:
        print("""
    ✅ All checks passed! System is ready.
    
    Next steps:
    1. Run the demo: python -m scaf_ls.monitoring.demo
    2. Integrate with SCAF-LS: see integration_guide.py
    3. Setup Grafana: python prometheus_grafana_setup.py
    
    ╚════════════════════════════════════════════════════════════╝
        """)
        return True
    else:
        print(f"""
    ⚠️  {checks_total - checks_passed} checks failed
    
    Fix issues above and re-run verification.
    
    ╚════════════════════════════════════════════════════════════╝
        """)
        return False


if __name__ == "__main__":
    import sys
    success = verify_installation()
    sys.exit(0 if success else 1)
