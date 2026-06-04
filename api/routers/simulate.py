"""POST /simulate — fixed-vs-dynamic pricing simulation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.models import SimulateRequest, SimulateResponse
from api.services.pipeline_service import pipeline

router = APIRouter(prefix="/simulate", tags=["simulation"])


@router.post(
    "",
    response_model=SimulateResponse,
    summary="Pricing Simulation Agent",
    description=(
        "Simulates fixed-tariff baseline vs dynamic-tariff outcomes using an explicit "
        "price-elasticity assumption. Accepts a custom `price_elasticity` value so callers "
        "can run what-if scenarios without re-training the model."
    ),
)
def simulate(body: SimulateRequest = SimulateRequest()) -> SimulateResponse:
    if not pipeline.ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Call POST /pipeline/run first.",
        )

    df = pipeline.simulate_on_demand(
        station_id=body.station_id,
        start_hour=body.start_hour,
        end_hour=body.end_hour,
        price_elasticity=body.price_elasticity,
        limit=body.limit,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="No simulation rows match the supplied filters.")

    df = df.copy()
    df["timestamp_hour"] = df["timestamp_hour"].astype(str)

    return SimulateResponse(
        rows_returned=len(df),
        price_elasticity_used=body.price_elasticity,
        simulation=df.to_dict(orient="records"),
    )
