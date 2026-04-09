"""
SCAF-LS Optimization Server

Serveur HTTP pour l'exécution des campagnes d'optimisation SCAF-LS
avec interface RESTful pour les 400 sous-agents.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Microsoft Agent Framework Server
from azure.ai.agentserver.agentframework import from_agent_framework

# SCAF-LS Optimization
from .orchestrator import OptimizationOrchestrator, OptimizationCampaign, initialize_campaign
from agent_framework import WorkflowBuilder, Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SCAF-LS Optimization Server",
    description="Multi-agent hyperparameter optimization system for SCAF-LS models",
    version="1.0.0"
)

# Modèles de requête/réponse
class OptimizationRequest(BaseModel):
    campaign_id: str = None
    models: List[str] = ["LGBM", "RandomForest", "BiLSTM"]
    max_trials: int = 100
    max_parallel: int = 50
    time_limit_hours: float = 4.0
    stability_penalty: float = 0.1

class OptimizationStatus(BaseModel):
    campaign_id: str
    status: str  # "running", "completed", "failed"
    progress: float
    results: Dict[str, Any] = None
    error: str = None

# Stockage des campagnes en cours
active_campaigns: Dict[str, Dict[str, Any]] = {}

@app.post("/optimize/start", response_model=OptimizationStatus)
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Démarrer une nouvelle campagne d'optimisation"""

    try:
        # Créer la campagne
        campaign = OptimizationCampaign(
            campaign_id=request.campaign_id or f"campaign_{int(asyncio.get_event_loop().time())}",
            models_to_optimize=request.models,
            max_trials_per_model=request.max_trials,
            max_parallel_agents=request.max_parallel,
            time_limit_hours=request.time_limit_hours,
            stability_penalty=request.stability_penalty
        )

        # Initialiser le statut
        active_campaigns[campaign.campaign_id] = {
            "status": "running",
            "progress": 0.0,
            "results": None,
            "error": None,
            "start_time": asyncio.get_event_loop().time()
        }

        # Lancer l'optimisation en arrière-plan
        background_tasks.add_task(run_campaign_async, campaign)

        return OptimizationStatus(
            campaign_id=campaign.campaign_id,
            status="running",
            progress=0.0
        )

    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/status/{campaign_id}", response_model=OptimizationStatus)
async def get_optimization_status(campaign_id: str):
    """Obtenir le statut d'une campagne d'optimisation"""

    if campaign_id not in active_campaigns:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign_data = active_campaigns[campaign_id]

    return OptimizationStatus(
        campaign_id=campaign_id,
        status=campaign_data["status"],
        progress=campaign_data["progress"],
        results=campaign_data["results"],
        error=campaign_data["error"]
    )

@app.get("/optimize/campaigns")
async def list_campaigns():
    """Lister toutes les campagnes"""

    return {
        "campaigns": [
            {
                "id": cid,
                "status": data["status"],
                "progress": data["progress"],
                "start_time": data["start_time"]
            }
            for cid, data in active_campaigns.items()
        ]
    }

@app.delete("/optimize/campaign/{campaign_id}")
async def cancel_campaign(campaign_id: str):
    """Annuler une campagne en cours"""

    if campaign_id not in active_campaigns:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Marquer comme annulée
    active_campaigns[campaign_id]["status"] = "cancelled"
    active_campaigns[campaign_id]["error"] = "Campaign cancelled by user"

    return {"message": f"Campaign {campaign_id} cancelled"}

@app.get("/optimize/models")
async def get_available_models():
    """Obtenir la liste des modèles disponibles pour l'optimisation"""

    return {
        "models": [
            {
                "name": "LGBM",
                "description": "LightGBM classifier with optimized search space",
                "parameters": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "num_leaves", "min_child_samples"]
            },
            {
                "name": "RandomForest",
                "description": "Random Forest classifier with comprehensive tuning",
                "parameters": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap"]
            },
            {
                "name": "BiLSTM",
                "description": "Bidirectional LSTM for sequential data",
                "parameters": ["seq_len", "hidden", "n_layers", "dropout", "lr", "epochs"]
            }
        ]
    }

@app.get("/optimize/results/{campaign_id}")
async def get_campaign_results(campaign_id: str):
    """Obtenir les résultats détaillés d'une campagne"""

    if campaign_id not in active_campaigns:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign_data = active_campaigns[campaign_id]

    if campaign_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Campaign not completed yet")

    # Charger le rapport depuis le fichier
    report_file = f"results_v50/optimization/campaign_{campaign_id}_report.md"
    report_content = ""
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()

    return {
        "campaign_id": campaign_id,
        "results": campaign_data["results"],
        "report": report_content
    }

async def run_campaign_async(campaign: OptimizationCampaign):
    """Exécuter une campagne d'optimisation de manière asynchrone"""

    try:
        logger.info(f"Starting campaign {campaign.campaign_id}")

        # Créer l'orchestrateur
        orchestrator = OptimizationOrchestrator(campaign)

        # Workflow de campagne
        campaign_init = initialize_campaign
        workflow = (
            WorkflowBuilder(start_executor=campaign_init)
            .add_edge(campaign_init, orchestrator)
            .build()
        )

        # Exécuter
        events = await workflow.run(Message("user", ["start_optimization"]))

        # Récupérer les résultats
        outputs = events.get_outputs()
        if outputs:
            final_result = outputs[-1]
            active_campaigns[campaign.campaign_id]["results"] = final_result
            active_campaigns[campaign.campaign_id]["status"] = "completed"
            active_campaigns[campaign.campaign_id]["progress"] = 100.0

            logger.info(f"Campaign {campaign.campaign_id} completed successfully")
        else:
            raise Exception("No results from optimization workflow")

    except Exception as e:
        logger.error(f"Campaign {campaign.campaign_id} failed: {e}")
        active_campaigns[campaign.campaign_id]["status"] = "failed"
        active_campaigns[campaign.campaign_id]["error"] = str(e)

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage du serveur"""
    logger.info("🚀 SCAF-LS Optimization Server starting...")
    logger.info("📊 Multi-agent hyperparameter optimization system ready")
    logger.info("🔗 Available endpoints:")
    logger.info("   POST /optimize/start - Start optimization campaign")
    logger.info("   GET  /optimize/status/{campaign_id} - Get campaign status")
    logger.info("   GET  /optimize/campaigns - List all campaigns")
    logger.info("   GET  /optimize/models - Get available models")
    logger.info("   GET  /optimize/results/{campaign_id} - Get campaign results")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt du serveur"""
    logger.info("🛑 SCAF-LS Optimization Server shutting down...")

    # Annuler toutes les campagnes actives
    for campaign_id in active_campaigns:
        if active_campaigns[campaign_id]["status"] == "running":
            active_campaigns[campaign_id]["status"] = "cancelled"
            active_campaigns[campaign_id]["error"] = "Server shutdown"

def run_server():
    """Fonction principale pour démarrer le serveur"""
    uvicorn.run(
        "scaf_ls.optimization.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()