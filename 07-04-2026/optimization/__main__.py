"""
Script principal pour lancer l'optimisation SCAF-LS

Usage:
    python -m scaf_ls.optimization.run_optimization
    python -m scaf_ls.optimization.run_agent_server
    python -m scaf_ls.optimization.run_http_server
"""

import argparse
import asyncio
import sys

def main():
    parser = argparse.ArgumentParser(description="SCAF-LS Hyperparameter Optimization")
    parser.add_argument(
        "--mode",
        choices=["direct", "agent", "http"],
        default="direct",
        help="Mode d'exécution: direct (optimisation directe), agent (serveur agent), http (serveur HTTP)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LGBM", "RandomForest", "BiLSTM"],
        help="Modèles à optimiser"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Nombre maximum d'essais par modèle"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Nombre maximum d'agents en parallèle"
    )

    args = parser.parse_args()

    if args.mode == "direct":
        # Import et exécution directe
        from .orchestrator import run_optimization_campaign
        asyncio.run(run_optimization_campaign())

    elif args.mode == "agent":
        # Serveur agent
        from .agent_server import run_agent_server
        asyncio.run(run_agent_server())

    elif args.mode == "http":
        # Serveur HTTP
        from .server import run_server
        run_server()

if __name__ == "__main__":
    main()