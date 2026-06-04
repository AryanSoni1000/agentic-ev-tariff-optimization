"""POST /pipeline/run — triggers pipeline execution."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from api.services.pipeline_service import pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class RunRequest(BaseModel):
    demo: bool = True


class RunResponse(BaseModel):
    status: str
    message: str


@router.post(
    "/run",
    response_model=RunResponse,
    summary="Trigger full pipeline execution",
    description=(
        "Runs data loading → preprocessing → feature engineering → modeling → "
        "tariff optimization → simulation → monitoring in one call. "
        "Set `demo=true` (default) to use synthetic data when raw files are absent."
    ),
)
def run_pipeline(body: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    if pipeline.ready:
        return RunResponse(
            status="already_ready",
            message=f"Pipeline already initialised. Best model: {pipeline.best_model_name}",
        )

    def _run() -> None:
        try:
            pipeline.run(demo=body.demo)
        except Exception as exc:
            logger.exception("Pipeline execution failed: %s", exc)

    background_tasks.add_task(_run)
    return RunResponse(
        status="started",
        message="Pipeline started in the background. Poll GET / to check readiness.",
    )


@router.post(
    "/reset",
    response_model=RunResponse,
    summary="Reset pipeline state (forces re-run on next /pipeline/run)",
)
def reset_pipeline() -> RunResponse:
    pipeline._ready = False
    pipeline._best_model = None
    pipeline._best_model_name = None
    return RunResponse(status="reset", message="Pipeline state cleared.")
