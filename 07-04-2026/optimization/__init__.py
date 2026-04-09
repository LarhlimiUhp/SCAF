"""
SCAF-LS Hyperparameter Optimization Module

Multi-agent system for systematic hyperparameter tuning with Optuna,
featuring 400 specialized sub-agents for parallel optimization.
"""

# Import simplifié par défaut (sans dépendances Agent Framework)
from .simplified_runner import (
    SimplifiedModelOptimizer,
    run_parallel_optimization,
    run_single_optimization,
    generate_final_report
)

# Imports avancés (nécessitent Agent Framework)
try:
    from .orchestrator import (
        OptimizationOrchestrator,
        ModelTuningAgent,
        OptimizationCampaign,
        OptimizationResult,
        run_optimization_campaign
    )

    from .server import app as optimization_app, run_server
    from .agent_server import run_agent_server

    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False
    # Classes placeholders
    OptimizationOrchestrator = None
    ModelTuningAgent = None
    OptimizationCampaign = None
    OptimizationResult = None
    run_optimization_campaign = None
    optimization_app = None
    run_server = None
    run_agent_server = None

__all__ = [
    # Core simplifié (toujours disponible)
    "SimplifiedModelOptimizer",
    "run_parallel_optimization",
    "run_single_optimization",
    "generate_final_report",

    # Avancé (si Agent Framework disponible)
    "OptimizationOrchestrator",
    "ModelTuningAgent",
    "OptimizationCampaign",
    "OptimizationResult",
    "run_optimization_campaign",
    "optimization_app",
    "run_server",
    "run_agent_server",
]