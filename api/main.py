"""
FastAPI application for the EV Tariff Optimization System.

Architecture
────────────
Streamlit Frontend  →  FastAPI Backend  →  PipelineService
                              │
                    ┌─────────┴──────────┐
                    │                    │
              DemandPredictionAgent   TariffOptimizationAgent
                    │                    │
              MonitoringAgent         SimulationEngine
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make src importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routers import health, predict, optimize, simulate, metrics, pipeline as pipeline_router
from api.services.pipeline_service import pipeline as pipeline_svc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: auto-run pipeline with demo data on first boot
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("Starting up – running ML pipeline with demo data …")
    try:
        pipeline_svc.run(demo=True)
        logger.info("Pipeline ready. Best model: %s", pipeline_svc.best_model_name)
    except Exception as exc:
        logger.error("Pipeline failed at startup: %s", exc)
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EV Tariff Optimization API",
    description=(
        "Production-grade REST API for EV charging demand forecasting and "
        "agentic dynamic tariff optimization.\n\n"
        "**Agents**\n"
        "- **Demand Prediction Agent** (`/predict`) — forecasts station-hour utilization\n"
        "- **Tariff Optimization Agent** (`/optimize`) — numerically optimises tariffs\n"
        "- **Simulation Engine** (`/simulate`) — fixed vs dynamic pricing scenarios\n"
        "- **Monitoring & Learning Agent** (`/metrics`) — revenue, congestion, efficiency KPIs\n\n"
        "Run `POST /pipeline/run` to (re)train on your own data."
    ),
    version="1.0.0",
    contact={
        "name": "EV Tariff System",
        "url": "https://github.com/your-repo/ev-tariff-optimization",
    },
    lifespan=lifespan,
)

# CORS — allow Streamlit (localhost:8501) and any origin in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Register routers
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(optimize.router)
app.include_router(simulate.router)
app.include_router(metrics.router)
app.include_router(pipeline_router.router)
