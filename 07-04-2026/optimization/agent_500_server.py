"""
SCAF-LS LightGBM Agent Optimization Server
FastAPI server for running 500-agent optimization campaigns
"""

import asyncio
import json
import logging
import uvicorn
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from scaf_ls.optimization.lightgbm_500_agent_system import (
    run_500_agent_optimization,
    OptimizationCampaign,
    LightGBMMasterOrchestrator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SCAF-LS LightGBM Agent Optimization Server",
    description="500-Agent System for LightGBM Optimization in Trading Framework",
    version="1.0.0"
)

# Global state
active_campaigns: Dict[str, Dict] = {}
campaign_results: Dict[str, Dict] = {}


class CampaignConfig(BaseModel):
    """Configuration for optimization campaign"""
    target_auc: float = 0.60
    target_sharpe: float = 1.0
    target_max_drawdown: float = 0.15
    max_trials_per_agent: int = 50
    max_parallel_agents: int = 20
    time_limit_hours: float = 2.0
    stability_penalty: float = 0.1
    min_improvement_threshold: float = 0.01
    n_hyperparam_agents: int = 200
    n_feature_agents: int = 150
    n_architecture_agents: int = 100
    n_pipeline_agents: int = 50


class CampaignStatus(BaseModel):
    """Status of an optimization campaign"""
    campaign_id: str
    status: str  # "running", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    best_auc: float = 0.0
    completed_agents: int = 0
    total_agents: int = 0


@app.post("/campaigns/start", response_model=Dict[str, str])
async def start_campaign(config: CampaignConfig, background_tasks: BackgroundTasks):
    """Start a new optimization campaign"""
    campaign_id = f"campaign_{int(datetime.now().timestamp())}"

    # Create campaign configuration
    campaign = OptimizationCampaign(
        campaign_id=campaign_id,
        target_auc=config.target_auc,
        target_sharpe=config.target_sharpe,
        target_max_drawdown=config.target_max_drawdown,
        max_trials_per_agent=config.max_trials_per_agent,
        max_parallel_agents=config.max_parallel_agents,
        time_limit_hours=config.time_limit_hours,
        stability_penalty=config.stability_penalty,
        min_improvement_threshold=config.min_improvement_threshold,
        n_hyperparam_agents=config.n_hyperparam_agents,
        n_feature_agents=config.n_feature_agents,
        n_architecture_agents=config.n_architecture_agents,
        n_pipeline_agents=config.n_pipeline_agents
    )

    # Initialize campaign status
    active_campaigns[campaign_id] = {
        "status": "running",
        "start_time": datetime.now(),
        "config": config.dict(),
        "orchestrator": None
    }

    # Start campaign in background
    background_tasks.add_task(run_campaign_async, campaign_id, campaign)

    logger.info(f"🚀 Started campaign {campaign_id} with {config.n_hyperparam_agents + config.n_feature_agents + config.n_architecture_agents + config.n_pipeline_agents} agents")

    return {"campaign_id": campaign_id, "message": "Campaign started successfully"}


async def run_campaign_async(campaign_id: str, campaign: OptimizationCampaign):
    """Run campaign asynchronously"""
    try:
        orchestrator = LightGBMMasterOrchestrator(campaign)
        active_campaigns[campaign_id]["orchestrator"] = orchestrator

        results = await orchestrator.run_optimization_campaign()

        # Store results
        campaign_results[campaign_id] = {
            "status": "completed",
            "end_time": datetime.now(),
            "results": results
        }

        active_campaigns[campaign_id]["status"] = "completed"
        active_campaigns[campaign_id]["end_time"] = datetime.now()

        logger.info(f"✅ Campaign {campaign_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Campaign {campaign_id} failed: {e}")
        active_campaigns[campaign_id]["status"] = "failed"
        active_campaigns[campaign_id]["error"] = str(e)


@app.get("/campaigns/{campaign_id}/status", response_model=CampaignStatus)
async def get_campaign_status(campaign_id: str):
    """Get status of a campaign"""
    if campaign_id not in active_campaigns and campaign_id not in campaign_results:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if campaign_id in campaign_results:
        # Campaign completed
        result = campaign_results[campaign_id]
        return CampaignStatus(
            campaign_id=campaign_id,
            status=result["status"],
            start_time=active_campaigns[campaign_id]["start_time"],
            end_time=result["end_time"],
            progress=1.0,
            best_auc=result["results"]["summary"]["best_auc"],
            completed_agents=result["results"]["completed_agents"],
            total_agents=result["results"]["total_agents"]
        )
    else:
        # Campaign running
        campaign_data = active_campaigns[campaign_id]
        orchestrator = campaign_data.get("orchestrator")

        if orchestrator:
            progress = orchestrator.completed_agents / orchestrator.total_agents if orchestrator.total_agents > 0 else 0
            best_auc = max([r.best_score for r in orchestrator.all_results], default=0.0)
        else:
            progress = 0.0
            best_auc = 0.0

        return CampaignStatus(
            campaign_id=campaign_id,
            status=campaign_data["status"],
            start_time=campaign_data["start_time"],
            progress=progress,
            best_auc=best_auc,
            completed_agents=getattr(orchestrator, 'completed_agents', 0) if orchestrator else 0,
            total_agents=getattr(orchestrator, 'total_agents', 0) if orchestrator else 0
        )


@app.get("/campaigns/{campaign_id}/results")
async def get_campaign_results(campaign_id: str):
    """Get detailed results of a completed campaign"""
    if campaign_id not in campaign_results:
        raise HTTPException(status_code=404, detail="Campaign results not found")

    return JSONResponse(content=campaign_results[campaign_id])


@app.get("/campaigns")
async def list_campaigns():
    """List all campaigns"""
    campaigns = []

    for campaign_id in active_campaigns.keys():
        try:
            status = await get_campaign_status(campaign_id)
            campaigns.append(status.dict())
        except:
            continue

    for campaign_id in campaign_results.keys():
        if campaign_id not in active_campaigns:
            try:
                status = await get_campaign_status(campaign_id)
                campaigns.append(status.dict())
            except:
                continue

    return {"campaigns": campaigns}


@app.post("/campaigns/{campaign_id}/stop")
async def stop_campaign(campaign_id: str):
    """Stop a running campaign"""
    if campaign_id not in active_campaigns:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if active_campaigns[campaign_id]["status"] != "running":
        raise HTTPException(status_code=400, detail="Campaign is not running")

    # Note: In a real implementation, you'd need to implement proper cancellation
    # For now, just mark as stopped
    active_campaigns[campaign_id]["status"] = "stopped"

    return {"message": f"Campaign {campaign_id} stop requested"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SCAF-LS LightGBM Agent Optimization Server"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SCAF-LS LightGBM 500-Agent Optimization Server",
        "version": "1.0.0",
        "endpoints": [
            "POST /campaigns/start - Start optimization campaign",
            "GET /campaigns/{id}/status - Get campaign status",
            "GET /campaigns/{id}/results - Get campaign results",
            "GET /campaigns - List all campaigns",
            "POST /campaigns/{id}/stop - Stop campaign",
            "GET /health - Health check"
        ]
    }


if __name__ == "__main__":
    print("🚀 Starting SCAF-LS LightGBM Agent Optimization Server")
    print("📊 Server will be available at http://localhost:8000")
    print("📖 API documentation at http://localhost:8000/docs")

    uvicorn.run(
        "scaf_ls.optimization.agent_500_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )