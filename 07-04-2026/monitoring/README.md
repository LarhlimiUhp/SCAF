# 📊 SCAF-LS Monitoring & Maintenance System

Système de monitoring complet et maintenance prédictive pour les performances optimales de SCAF-LS.

## 🎯 Objectifs

- **Détection précoce** des problèmes système, applicatifs et métier
- **Maintenance proactive** avec prédiction des pannes
- **Optimisation continue** des performances
- **Réduction du downtime** à quasi-zéro (targeté: 99.9%+)
- **Amélioration continue** basée sur les données

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           MonitoringService (Central)            │
│  - Orchestration de tous les collecteurs        │
│  - Gestion des threads de monitoring            │
│  - API unifiée                                  │
└─────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                 │
   Collecteurs       Systèmes d'Analyse   Export
   
┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐
│ System Metrics  │  │ Alert System     │  │ Prometheus  │
│ - CPU/RAM/GPU   │  │ - Seuils dynamiq │  │ - Grafana   │
│ - Disk/Network  │  │ - ML anomalies   │  │ - JSON Logs │
└─────────────────┘  └──────────────────┘  └─────────────┘

┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐
│ App Metrics     │  │ Health Checks    │  │ Maintenance │
│ - Latency       │  │ - System health  │  │ - Retraining│
│ - Throughput    │  │ - Model validity │  │ - Cleanup   │
│ - Errors        │  │ - Data integrity │  │ - Restart   │
└─────────────────┘  └──────────────────┘  └─────────────┘

┌─────────────────┐  ┌──────────────────┐
│ Business Metrics│  │ Drift Detection  │
│ - P&L realtime  │  │ - Data Drift     │
│ - Drawdown      │  │ - Model Drift    │
│ - Sharpe Ratio  │  │ - Concept Drift  │
│ - Win Rate      │  │ - Retraining rec │
└─────────────────┘  └──────────────────┘
```

## 📦 Modules

### 1. **logger.py** - Logging Structuré
```python
from scaf_ls.monitoring import StructuredLogger, get_logger

logger = get_logger("scaf-ls")
logger.info("Event occurred", key1=value1, key2=value2)
# Exporte en JSON pour ELK stack
```

**Fonctionnalités:**
- ✅ Logs JSON structurés
- ✅ Contexte persistent
- ✅ Multi-handlers (fichier + console)
- ✅ Traçage d'exceptions

### 2. **system_metrics.py** - Métriques Système
```python
from scaf_ls.monitoring import SystemMetricsCollector

collector = SystemMetricsCollector()
metrics = collector.collect()

print(metrics.cpu_percent)      # CPU usage
print(metrics.memory_percent)   # RAM usage
print(metrics.gpu_percent)      # GPU (si disponible)
print(metrics.disk_percent)     # Disk usage
```

**Métriques collectées:**
- CPU (global + process-specific)
- Mémoire (globale + process-specific)
- GPU (PyTorch + GPUtil)
- Disque
- Réseau (throughput)

### 3. **app_metrics.py** - Métriques Applicatives
```python
from scaf_ls.monitoring import ApplicationMetricsCollector, track_latency

collector = ApplicationMetricsCollector()

# Tracer la latence d'une fonction
@track_latency("operation_name")
def my_operation():
    pass

# Enregistrer manuellement
collector.record_latency("operation", duration_seconds=0.5)
collector.record_error("ErrorType", "Error message")
collector.record_throughput("operation", items=1000, bytes_processed=500000)

# Obtenir les métriques
latency = collector.get_latency_metric("operation")
print(latency.percentile_p99)  # P99 latency

errors = collector.get_error_metric()
print(errors.errors_per_minute)  # Erreurs par minute
```

**Fonctionnalités:**
- ✅ P50/P95/P99 latency
- ✅ Erreur total + taux
- ✅ Throughput (items/sec)
- ✅ Décorateur automatique

### 4. **business_metrics.py** - Métriques Métier
```python
from scaf_ls.monitoring import BusinessMetricsCollector

metrics = BusinessMetricsCollector(initial_equity=100000)

# Updater l'équité
metrics.update_equity(105000)

# Enregistrer un trade
metrics.record_trade(
    entry_price=100.0,
    exit_price=105.0,
    quantity=100,
    side="long",
    commission=10.0
)

# Obtenir les métriques
portfolio = metrics.get_portfolio_metrics()
print(portfolio.sharpe_ratio)     # Sharpe ratio
print(portfolio.max_drawdown)     # Max drawdown (%)
print(portfolio.calmar_ratio)     # Calmar ratio

trades = metrics.get_trade_metrics()
print(trades.win_rate)            # Win rate (%)
print(trades.profit_factor)       # Profit factor
```

**Métriques disponibles:**
- P&L total et journalier
- Sharpe, Sortino, Calmar ratios
- Max drawdown + current drawdown
- Win rate, Profit factor
- Recovery factor

### 5. **alerts.py** - Système d'Alertes Intelligentes
```python
from scaf_ls.monitoring import AlertSystem, AlertType, AlertSeverity

alerts = AlertSystem()

# Vérifier les seuils
alerts.check_cpu_usage(cpu_percent=92, threshold=80)
# → Crée une alerte WARNING

alerts.check_drawdown(current_drawdown=-22.5, max_threshold=-20)
# → Crée une alerte CRITICAL

alerts.check_data_drift(drift_score=0.75, threshold=0.7)
# → Crée une alerte WARNING

# Enregistrer un handler personnalisé
def on_critical_alert(alert):
    print(f"🚨 CRITICAL: {alert.message}")
    # Envoyer notification (email, Slack, etc.)

alerts.register_handler(AlertType.HIGH_DRAWDOWN, on_critical_alert)

# Obtenir les alertes
critical = alerts.get_critical_alerts()
summary = alerts.get_alert_summary()
```

**Types d'alertes:**
- Système: CPU, Mémoire, Disque, GPU
- Application: Latence, Erreurs, Throughput
- Trading: Drawdown, Win rate, P&L
- Data: Drift, Model Drift, Concept Drift

**Intelligence:**
- ✅ Seuils dynamiques (basés sur écart-type)
- ✅ Détection d'anomalies (Z-score)
- ✅ Tendances (tendance croissante d'erreurs)

### 6. **health_checks.py** - Vérifications d'État
```python
from scaf_ls.monitoring import HealthCheckSystem

health = HealthCheckSystem()

# Exécuter tous les checks
checks = health.run_all_checks({
    'data_loader': data_loader,
    'models': models,
    'features': X,
})

# Obtenir la santé globale
overall = health.get_overall_health(checks)
print(overall['overall_health_score'])  # 0-100
# "healthy" / "degraded" / "unhealthy"
```

**Checks disponibles:**
- Ressources système (CPU/RAM/Disk)
- Disponibilité des données
- État des modèles
- Validité des features
- Validité des résultats

### 7. **profiler.py** - Performance Profiling
```python
from scaf_ls.monitoring import PerformanceProfiler

profiler = PerformanceProfiler()

# Décorateur pour profiler une fonction
@profiler.profile_function("my_function")
def expensive_function():
    # code à profiler
    pass

# Déterminer les bottlenecks
bottlenecks = profiler.detect_bottlenecks(threshold_percentile=90)

# Obtenir les fonctions les plus lentes
slowest = profiler.get_slowest_functions(n=10)

# Profiler la mémoire
result = profiler.get_memory_profile(expensive_function)
print(result['peak_memory_mb'])
```

**Fonctionnalités:**
- ✅ Profilage CPU
- ✅ Détection bottlenecks
- ✅ Profilage mémoire
- ✅ Export pour Grafana

### 8. **drift_detection.py** - Détection de Drift
```python
from scaf_ls.monitoring import DriftDetector

detector = DriftDetector()

# Définir les données de référence (entraînement)
detector.set_reference_data(X_train, y_train, predictions_train)

# Ajouter des données de test
detector.add_test_sample(X_test_sample, y_true, y_pred)

# Obtenir les métriques de drift
drift_metrics = detector.get_drift_metrics()
print(drift_metrics.data_drift_score)      # 0-1
print(drift_metrics.model_drift_score)     # 0-1
print(drift_metrics.concept_drift_score)   # 0-1
print(drift_metrics.is_drifting)           # bool

# Déterminer si réentraînement est nécessaire
if detector.should_retrain(overall_drift_threshold=0.6):
    print("Retraining recommended")

# Obtenir la tendance
trends = detector.get_drift_trend(window_size=30)
print(trends['overall_drift_trend'])  # Taux de changement
```

**Méthodes:**
- Kolmogorov-Smirnov test
- Distance Manhattan
- Distance Wasserstein
- Corrélation input-output

### 9. **maintenance.py** - Maintenance Prédictive
```python
from scaf_ls.monitoring import PredictiveMaintenanceEngine

maintenance = PredictiveMaintenanceEngine()

# Vérifier si réentraînement est nécessaire
maintenance.check_retraining_needed(
    drift_score=0.65,
    time_since_training_days=8.0,
    model_performance_degradation=0.12
)

# Vérifier si cleanup est nécessaire
maintenance.check_cleanup_needed(
    log_size_mb=1500,
    cache_size_mb=600,
    total_disk_free_gb=5.0
)

# Vérifier si optimisation est nécessaire
maintenance.check_optimization_needed(
    system_memory_percent=88,
    cpu_efficiency=0.4,
    error_rate=0.06
)

# Obtenir le calendrier
schedule = maintenance.get_maintenance_schedule()

# Obtenir les recommandations
recommendations = maintenance.get_health_recommendations()
```

**Tâches de maintenance:**
- Retraining: Déclenché par drift ou dégradation
- Cleanup: Logs/cache/disque
- Optimization: Ressources ou performance
- Restart: Uptime élevé ou fuites mémoire

### 10. **monitoring_service.py** - Service Principal
```python
from scaf_ls.monitoring import (
    start_monitoring,
    get_monitoring_service,
    stop_monitoring,
)

# Démarrer le monitoring
monitoring = start_monitoring({
    'initial_equity': 100000,
    'thresholds': {
        'cpu_percent': 85,
        'memory_percent': 90,
        'drawdown_percent': -20,
    }
})

# Utiliser le monitoring
monitoring.update_equity(105000)
monitoring.record_trade(entry=100, exit=105, qty=100, side="long")

# Obtenir le dashboard
dashboard = monitoring.get_dashboard_data()
print(dashboard['business']['total_return'])

# Exporter en Prometheus
prometheus_text = monitoring.export_prometheus_metrics()

# Arrêter
stop_monitoring()
```

**Boucle principale:**
- Quick metrics: Toutes les 5 secondes
- Full check: Toutes les 60 secondes
- Thread daemon (non-bloquant)

## 🚀 Guide d'Intégration

### Étape 1: Initialisation

```python
from scaf_ls.config import Config
from scaf_ls.monitoring import start_monitoring

# Démarrer le monitoring
monitoring = start_monitoring({
    'initial_equity': Config.INITIAL_CAPITAL,
    'thresholds': {
        'drawdown_percent': Config.MAX_DRAWDOWN_LIMIT,
    }
})
```

### Étape 2: Instrumenter les Fonctions

```python
from scaf_ls.monitoring.app_metrics import track_latency

@track_latency("data_loading")
def load_data():
    # code
    pass

@track_latency("model_training")
def train_model():
    # code
    pass

@track_latency("backtest_loop")
def run_backtest():
    # code
    pass
```

### Étape 3: Mettre à Jour les Métriques

```python
# Dans la boucle de backtest
for idx, row in backtest_df.iterrows():
    monitoring.update_equity(float(row['equity']))
    
    if idx % 100 == 0:
        if row['return'] > 0:
            monitoring.record_trade(entry=prev_price, exit=row['price'], qty=qty)

# Après le backtest
summary = monitoring.business_metrics.get_summary_dict()
print(summary)
```

### Étape 4: Utiliser les Alertes

```python
# Enregistrer un handler personnalisé
def on_high_drawdown(alert):
    print(f"🚨 Drawdown critique: {alert.message}")
    # Envoyer email/Slack/etc.

monitoring.alerts.register_handler(AlertType.HIGH_DRAWDOWN, on_high_drawdown)
```

### Étape 5: Générer les Rapports

```python
# Dashboard complet
dashboard = monitoring.get_dashboard_data()

# Rapports spécifiques
health_status = monitoring.health_checks.run_all_checks()
drift_metrics = monitoring.drift_detector.get_drift_metrics()
maintenance_plan = monitoring.maintenance_engine.get_maintenance_schedule()
```

## 📊 Dashboard Grafana

Connexion à Prometheus:

1. Ajouter source: `http://localhost:9090`
2. Importer le dashboard ou créer custom

Métriques disponibles:
- `scaf_cpu_percent`
- `scaf_memory_percent`
- `scaf_disk_percent`
- `scaf_equity`
- `scaf_total_return_percent`
- `scaf_max_drawdown_percent`
- `scaf_sharpe_ratio`
- `scaf_win_rate_percent`
- `scaf_errors_total`
- `scaf_alerts_critical`

## 📈 Exemple Complet

Voir [integration_guide.py](./integration_guide.py) pour un exemple d'intégration complet.

## 🔧 Installation des Dépendances

```bash
pip install -r requirements_monitoring.txt
```

## 📝 Logging

Les logs sont sauvegardés dans:
- `logs/scaf-ls_*.json` - Format JSON (pour ELK)
- Console - Format texte

## 🎛️ Configuration Avancée

### Seuils Personnalisés

```python
monitoring_config = {
    'thresholds': {
        'cpu_percent': 85,
        'memory_percent': 90,
        'disk_percent': 85,
        'latency_ms': 500,
        'error_rate_per_min': 10,
        'drawdown_percent': -20,
        'win_rate_percent': 45,
        'data_drift': 0.7,
        'model_drift': 0.6,
    }
}

monitoring = start_monitoring(monitoring_config)
```

### Handlers Personnalisés

```python
async def send_slack_notification(alert):
    from slack_sdk import WebClient
    client = WebClient(token=os.environ['SLACK_TOKEN'])
    client.chat_postMessage(
        channel="#alerts",
        text=f"🚨 {alert.severity}: {alert.message}"
    )

monitoring.alerts.register_handler(AlertType.CRITICAL, send_slack_notification)
```

## 🎯 KPIs Suivis

| KPI | Cible | Alert |
|-----|-------|-------|
| CPU Usage | < 80% | > 80% warning, > 95% critical |
| Memory | < 85% | > 85% warning, > 95% critical |
| Latency P99 | < 500ms | > 1000ms warning |
| Error Rate | < 1/min | > 5/min warning |
| Sharpe Ratio | > 1.0 | < 0.5 warning |
| Max Drawdown | > -20% | < -20% warning |
| Data Drift | < 0.5 | > 0.7 warning |
| Model Drift | < 0.3 | > 0.6 warning |

## 🔍 Troubleshooting

### Monitoring Service ne démarre pas
- Vérifier les permissions PSUtil
- Sur Linux: `pip install psutil --upgrade`

### GPU non détecté
- Vérifier CUDA installation `nvidia-smi`
- Installer GPUtil: `pip install GPUtil`

### Logs JSON pas créés
- Vérifier le répertoire `logs/` existe
- Permissions d'écriture

## 📚 Références

- [Prometheus Metrics](https://prometheus.io/docs/practices/instrumentation/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ELK Stack](https://www.elastic.co/what-is/elk-stack)
- [Performance Profiling](https://docs.python.org/3/library/profile.html)

## 🤝 Support

Pour les questions ou rapports de bugs:
1. Vérifier les logs dans `logs/`
2. Examiner les alertes dans `monitoring.alerts.get_alert_summary()`
3. Vérifier la santé du système: `monitoring.health_checks.run_all_checks()`
