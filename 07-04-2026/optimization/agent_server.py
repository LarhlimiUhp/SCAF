"""
SCAF-LS Optimization Agent Server

Wrapper pour exposer le système d'optimisation comme agent Microsoft Agent Framework
avec serveur HTTP intégré.
"""

import asyncio
import logging
from typing import List

from agent_framework import (
    AgentResponseUpdate,
    Content,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
)
from azure.ai.agentserver.agentframework import from_agent_framework

from .orchestrator import OptimizationOrchestrator, OptimizationCampaign, initialize_campaign

logger = logging.getLogger(__name__)


@executor(id="scaf_ls_optimizer")
async def scaf_ls_optimizer(messages: List[Message], ctx: WorkflowContext[AgentResponseUpdate]) -> None:
    """Agent principal pour l'optimisation SCAF-LS"""

    try:
        # Analyser le message utilisateur
        user_message = messages[0].content if messages else "optimize all models"

        # Extraire les paramètres de la requête
        models_to_optimize = ['LGBM', 'RandomForest', 'BiLSTM']

        if "lgbm" in user_message.lower():
            models_to_optimize = ['LGBM']
        elif "randomforest" in user_message.lower() or "rf" in user_message.lower():
            models_to_optimize = ['RandomForest']
        elif "bilstm" in user_message.lower() or "lstm" in user_message.lower():
            models_to_optimize = ['BiLSTM']

        # Créer la campagne
        campaign = OptimizationCampaign(
            campaign_id=f"agent_campaign_{int(asyncio.get_event_loop().time())}",
            models_to_optimize=models_to_optimize,
            max_trials_per_model=30,  # Réduit pour les tests via agent
            max_parallel_agents=2,
            time_limit_hours=0.5,
            stability_penalty=0.1
        )

        await ctx.yield_output(
            AgentResponseUpdate(
                contents=[Content("text", text=f"🚀 Starting SCAF-LS optimization for models: {', '.join(models_to_optimize)}")],
                role="assistant",
                author_name="scaf_ls_optimizer"
            )
        )

        # Initialiser l'orchestrateur
        orchestrator = OptimizationOrchestrator(campaign)

        # Construire le workflow interne
        campaign_init = initialize_campaign
        internal_workflow = (
            WorkflowBuilder(start_executor=campaign_init)
            .add_edge(campaign_init, orchestrator)
            .build()
        )

        # Exécuter l'optimisation
        events = await internal_workflow.run(Message("user", ["start_optimization"]))

        # Traiter les résultats
        outputs = events.get_outputs()
        if outputs:
            final_result = outputs[-1]
            results = final_result.get('results', {})

            # Formater la réponse
            response_lines = ["## 🎯 Optimization Results\n"]

            for model_name, result in results.items():
                response_lines.extend([
                    f"### {model_name}",
                    f"- **AUC:** {result.best_auc:.4f} (±{result.stability_score:.4f})",
                    f"- **Trials:** {result.trials_completed}",
                    f"- **Best Params:** {result.best_params}",
                    ""
                ])

            # Calculer les métriques globales
            if results:
                avg_auc = sum(r.best_auc for r in results.values()) / len(results)
                avg_stability = sum(r.stability_score for r in results.values()) / len(results)

                response_lines.extend([
                    "## 📊 Summary",
                    f"- **Average AUC:** {avg_auc:.4f}",
                    f"- **Average Stability:** {avg_stability:.4f}",
                    f"- **Models Optimized:** {len(results)}",
                    "",
                    "✅ Optimization completed successfully!"
                ])

            final_response = "\n".join(response_lines)
        else:
            final_response = "❌ Optimization failed - no results generated"

        await ctx.yield_output(
            AgentResponseUpdate(
                contents=[Content("text", text=final_response)],
                role="assistant",
                author_name="scaf_ls_optimizer"
            )
        )

    except Exception as e:
        logger.error(f"Agent optimization failed: {e}")
        await ctx.yield_output(
            AgentResponseUpdate(
                contents=[Content("text", text=f"❌ Optimization failed: {str(e)}")],
                role="assistant",
                author_name="scaf_ls_optimizer"
            )
        )


async def run_agent_server():
    """Démarrer le serveur d'agent pour l'optimisation SCAF-LS"""

    # Créer l'agent d'optimisation
    optimizer_agent = scaf_ls_optimizer

    # Wrapper pour le serveur
    agent_server = from_agent_framework(optimizer_agent)

    logger.info("🚀 Starting SCAF-LS Optimization Agent Server...")
    logger.info("📊 Multi-agent hyperparameter optimization system ready")
    logger.info("🎯 Agent can be invoked with messages like:")
    logger.info("   - 'optimize all models'")
    logger.info("   - 'optimize LGBM'")
    logger.info("   - 'optimize RandomForest'")

    # Démarrer le serveur
    await agent_server.run_async()


if __name__ == "__main__":
    asyncio.run(run_agent_server())