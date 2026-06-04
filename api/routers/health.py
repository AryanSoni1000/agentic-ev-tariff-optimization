"""GET / — health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas.models import HealthResponse
from api.services.pipeline_service import pipeline

router = APIRouter(tags=["health"])


@router.get("/", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """
    Returns the operational status of the API and whether the
    ML pipeline has been initialised.
    """
    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline.ready,
        model_loaded=pipeline._best_model is not None,
        best_model=pipeline.best_model_name,
        data_rows=pipeline.data_rows,
        version="1.0.0",
    )
