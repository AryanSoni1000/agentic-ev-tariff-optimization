"""GET /metrics — monitoring & model metrics endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.models import MetricsResponse
from api.services.pipeline_service import pipeline

router = APIRouter(prefix="/metrics", tags=["monitoring"])


@router.get(
    "",
    response_model=MetricsResponse,
    summary="Monitoring & Learning Agent",
    description=(
        "Returns three metric tables: (1) model comparison metrics (MAE, RMSE, R²), "
        "(2) monitoring metrics (revenue gain, congestion reduction, pricing efficiency), "
        "(3) EDA summary statistics."
    ),
)
def metrics() -> MetricsResponse:
    if not pipeline.ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Call POST /pipeline/run first.",
        )

    return MetricsResponse(
        model_metrics=pipeline.model_metrics.to_dict(orient="records"),
        monitoring_metrics=pipeline.monitoring_metrics.to_dict(orient="records"),
        eda_summary=pipeline.eda_summary.to_dict(orient="records"),
    )
