"""POST /predict — demand prediction endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.models import PredictRequest, PredictResponse
from api.services.pipeline_service import pipeline

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post(
    "",
    response_model=PredictResponse,
    summary="Demand Prediction Agent",
    description=(
        "Returns station-hour utilization predictions from the best trained model. "
        "Optionally filter by `station_id` or `timestamp_hour` prefix."
    ),
)
def predict(body: PredictRequest = PredictRequest()) -> PredictResponse:
    if not pipeline.ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Call POST /pipeline/run first.",
        )

    df = pipeline.predict_on_demand(
        station_id=body.station_id,
        timestamp_hour=body.timestamp_hour,
        limit=body.limit,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="No predictions match the supplied filters.")

    metrics_row = pipeline.model_metrics.iloc[0]

    # Serialise timestamps as strings for JSON
    df = df.copy()
    df["timestamp_hour"] = df["timestamp_hour"].astype(str)

    return PredictResponse(
        model=str(metrics_row["model"]),
        mae=round(float(metrics_row["MAE"]), 4),
        rmse=round(float(metrics_row["RMSE"]), 4),
        r2=round(float(metrics_row["R2"]), 4),
        rows_returned=len(df),
        predictions=df.to_dict(orient="records"),
    )
