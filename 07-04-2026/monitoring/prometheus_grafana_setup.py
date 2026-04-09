"""
Configuration et setup pour Prometheus + Grafana monitoring
"""

import json
from pathlib import Path

# Configuration Prometheus
PROMETHEUS_CONFIG = """
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'scaf-ls'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    scrape_timeout: 5s
"""

# Dashboard Grafana JSON
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "SCAF-LS Monitoring Dashboard",
        "tags": ["scaf-ls", "trading", "monitoring"],
        "refresh": "5s",
        "time": {"from": "now-6h", "to": "now"},
        "panels": [
            {
                "id": 1,
                "title": "CPU Usage",
                "targets": [{"expr": "scaf_cpu_percent"}],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Memory Usage",
                "targets": [{"expr": "scaf_memory_percent"}],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "id": 3,
                "title": "Portfolio Equity",
                "targets": [{"expr": "scaf_equity"}],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 4,
                "title": "Total Return %",
                "targets": [{"expr": "scaf_total_return_percent"}],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            },
            {
                "id": 5,
                "title": "Max Drawdown %",
                "targets": [{"expr": "scaf_max_drawdown_percent"}],
                "type": "gauge",
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 16}
            },
            {
                "id": 6,
                "title": "Sharpe Ratio",
                "targets": [{"expr": "scaf_sharpe_ratio"}],
                "type": "gauge",
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 16}
            },
            {
                "id": 7,
                "title": "Win Rate %",
                "targets": [{"expr": "scaf_win_rate_percent"}],
                "type": "gauge",
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 16}
            },
            {
                "id": 8,
                "title": "Total Trades",
                "targets": [{"expr": "scaf_total_trades"}],
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 16}
            },
            {
                "id": 9,
                "title": "Errors per Minute",
                "targets": [{"expr": "rate(scaf_errors_total[1m])"}],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24}
            },
            {
                "id": 10,
                "title": "Critical Alerts",
                "targets": [{"expr": "scaf_alerts_critical"}],
                "type": "stat",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24}
            },
            {
                "id": 11,
                "title": "Disk Usage %",
                "targets": [{"expr": "scaf_disk_percent"}],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 32}
            },
        ]
    }
}

# Docker Compose for Prometheus + Grafana
DOCKER_COMPOSE = """
version: '3'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
"""

# Setup script
SETUP_SCRIPT = """
#!/bin/bash

# Setup script for Prometheus + Grafana monitoring

echo "Setting up SCAF-LS Monitoring Stack..."

# Create config files
echo "$PROMETHEUS_CONFIG" > prometheus.yml
echo "$DOCKER_COMPOSE" > docker-compose.yml

# Create directories
mkdir -p monitoring

# Start Docker Compose
docker-compose up -d

echo "✅ Prometheus running on http://localhost:9090"
echo "✅ Grafana running on http://localhost:3000"
echo "   - Username: admin"
echo "   - Password: admin"
echo ""
echo "Import dashboard from: grafana_dashboard.json"
"""


def setup_monitoring_stack():
    """Setup complet de la stack de monitoring"""
    
    # Créer prometheus.yml
    with open('prometheus.yml', 'w') as f:
        f.write(PROMETHEUS_CONFIG)
    print("✅ Créé prometheus.yml")
    
    # Créer docker-compose.yml
    with open('docker-compose.yml', 'w') as f:
        f.write(DOCKER_COMPOSE)
    print("✅ Créé docker-compose.yml")
    
    # Créer le dashboard Grafana
    with open('grafana_dashboard.json', 'w') as f:
        json.dump(GRAFANA_DASHBOARD, f, indent=2)
    print("✅ Créé grafana_dashboard.json")
    
    # Créer setup script
    with open('setup_monitoring.sh', 'w') as f:
        f.write(SETUP_SCRIPT)
    print("✅ Créé setup_monitoring.sh")
    
    print("""
╔════════════════════════════════════════════════════════╗
║     SCAF-LS Monitoring Stack Setup Complete            ║
╚════════════════════════════════════════════════════════╝

Prochaine étape:
  1. docker-compose up -d
  2. Aller sur http://localhost:3000
  3. Importer grafana_dashboard.json
  4. Ajouter source de données Prometheus (http://prometheus:9090)
    """)


def get_prometheus_metrics_endpoint(port: int = 8000) -> str:
    """Obtenir l'endpoint pour exposer les métriques Prometheus"""
    
    from flask import Flask, jsonify
    from scaf_ls.monitoring import get_monitoring_service
    
    app = Flask(__name__)
    
    @app.route('/metrics')
    def metrics():
        """Endpoint Prometheus"""
        service = get_monitoring_service()
        return service.export_prometheus_metrics(), 200, {'Content-Type': 'text/plain'}
    
    @app.route('/dashboard')
    def dashboard():
        """Endpoint dashboard JSON"""
        service = get_monitoring_service()
        return jsonify(service.get_dashboard_data())
    
    @app.route('/health')
    def health():
        """Health check"""
        return jsonify({'status': 'healthy'})
    
    return app


if __name__ == "__main__":
    setup_monitoring_stack()
