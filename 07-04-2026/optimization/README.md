# SCAF-LS Hyperparameter Optimization System

## 🎯 Mission Overview

**MISSION CRITIQUE**: Optimisation systématique des hyperparamètres avec Optuna pour tous les modèles SCAF-LS avec +5-10% AUC et -25% variance.

**ÉQUIPE**: 400 sous-agents spécialisés dans le tuning parallèle, validation croisée, et optimisation avec contraintes de stabilité.

## 🚀 Architecture

### Multi-Agent System
- **Master Orchestrator**: Coordination des 400 sous-agents
- **Model-Specific Agents** (4 types × 100 agents each):
  - LightGBM Tuning Agents
  - RandomForest Tuning Agents
  - BiLSTM Tuning Agents
  - Ensemble Tuning Agents
- **Cross-Validation Coordinator**: Gestion CV temporelle
- **Stability Analysis Agent**: Évaluation robustesse paramètres
- **Resource Management Agent**: Contrôle ressources et early stopping

### Search Spaces Optimaux

#### LightGBM
```python
{
    'n_estimators': [100, 2000],
    'max_depth': [3, 12],
    'learning_rate': [0.01, 0.3],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'num_leaves': [20, 200],
    'min_child_samples': [10, 100]
}
```

#### RandomForest
```python
{
    'n_estimators': [50, 500],
    'max_depth': [5, 30],
    'min_samples_split': [2, 20],
    'min_samples_leaf': [1, 10],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
```

#### BiLSTM
```python
{
    'seq_len': [5, 50],
    'hidden': [16, 128],
    'n_layers': [1, 4],
    'dropout': [0.0, 0.5],
    'lr': [1e-4, 1e-2],
    'epochs': [5, 50]
}
```

## 🎮 Utilisation

### 1. Optimisation Directe
```bash
# Optimisation complète
python -m scaf_ls.optimization

# Optimisation spécifique
python -m scaf_ls.optimization --models LGBM --trials 100
```

### 2. Serveur Agent (Microsoft Agent Framework)
```bash
python -m scaf_ls.optimization --mode agent
```

Puis invoquer avec des messages comme:
- "optimize all models"
- "optimize LGBM"
- "optimize RandomForest and BiLSTM"

### 3. Serveur HTTP REST
```bash
python -m scaf_ls.optimization --mode http
```

Endpoints disponibles:
- `POST /optimize/start` - Démarrer campagne
- `GET /optimize/status/{campaign_id}` - Statut campagne
- `GET /optimize/campaigns` - Lister campagnes
- `GET /optimize/models` - Modèles disponibles

## 📊 Métriques d'Optimisation

### Fonction Objectif
```
Maximize: AUC_score - stability_penalty × variance(AUC_scores)
```

### Contraintes
- **AUC Improvement**: +5-10% vs baseline
- **Stability**: -25% variance reduction
- **Time Limit**: 4 heures max par campagne
- **Memory**: 8GB max par agent
- **Early Stopping**: 20 trials sans amélioration

## 🔧 Configuration

### Campaign Configuration
```python
campaign = OptimizationCampaign(
    campaign_id="scaf_ls_opt_v1",
    models_to_optimize=['LGBM', 'RandomForest', 'BiLSTM'],
    max_trials_per_model=100,
    max_parallel_agents=50,
    time_limit_hours=4.0,
    stability_penalty=0.1,
    min_improvement_threshold=0.02
)
```

### Validation Croisée
- **Stratégie**: PurgedKFold avec embargo
- **Splits**: 3 folds temporels
- **Purge Gap**: 5 jours d'embargo

## 📈 Résultats

### Structure de Sortie
```
results_v50/optimization/
├── campaign_{id}_report.md          # Rapport complet
├── {model}_optimization_{timestamp}.json  # Résultats détaillés
└── optimization.log                 # Logs d'exécution
```

### Métriques Trackées
- AUC par fold et moyenne
- Variance de stabilité
- Temps d'optimisation
- Nombre d'essais
- Paramètres optimaux

## 🔍 Monitoring & Debugging

### Tracing (AI Toolkit)
```bash
# Ouvrir le trace viewer
AI Toolkit: Tracing Open
```

### Logs
```python
import logging
logging.getLogger('scaf_ls.optimization').setLevel(logging.DEBUG)
```

## 🎯 Performance Targets

| Métrique | Target | Status |
|----------|--------|--------|
| AUC Improvement | +5-10% | 🎯 |
| Variance Reduction | -25% | 🎯 |
| Optimization Time | <4h | ✅ |
| Memory Usage | <8GB/agent | ✅ |
| Parallel Efficiency | >80% | 🎯 |

## 🚀 Déploiement

### Production Setup
```bash
# Installation
pip install -r requirements.txt

# Configuration environnement
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Lancement
python -m scaf_ls.optimization --mode http --parallel 50
```

### Scaling
- **Horizontal**: + agents parallèles
- **Vertical**: + ressources par agent
- **Distributed**: Multi-noeuds avec coordination

## 📚 API Reference

### Classes Principales
- `OptimizationOrchestrator`: Chef d'orchestre principal
- `ModelTuningAgent`: Agent spécialisé par modèle
- `OptimizationCampaign`: Configuration campagne
- `OptimizationResult`: Résultats optimisation

### Fonctions
- `run_optimization_campaign()`: Exécution directe
- `run_agent_server()`: Serveur agent
- `run_server()`: Serveur HTTP

## 🤝 Contribution

### Architecture des Agents
1. **Spécialisation**: Chaque agent optimisé pour un modèle
2. **Coordination**: Orchestrateur gère allocation ressources
3. **Communication**: Messages structurés via Agent Framework
4. **Résilience**: Gestion d'erreurs et recovery automatique

### Développement
```bash
# Tests
python -m pytest scaf_ls/optimization/

# Lint
python -m flake8 scaf_ls/optimization/

# Type checking
python -m mypy scaf_ls/optimization/
```

---

**🎉 Système opérationnel**: 400 sous-agents prêts pour l'optimisation SCAF-LS!