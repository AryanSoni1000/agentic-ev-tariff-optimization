"""POST /optimize — tariff optimization endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.models import OptimizeRequest, OptimizeResponse
from api.services.pipeline_service import pipeline

router = APIRouter(prefix="/optimize", tags=["optimization"])


@router.post(
    "",
    response_model=OptimizeResponse,
    summary="Tariff Optimization Agent",
    description=(
        "Returns numerically optimised tariff recommendations for each station-hour. "
        "The objective function balances revenue maximisation with congestion penalties "
        "and off-peak utilisation uplift. Guardrails enforce min/max tariff bounds."
    ),
)
def optimize(body: OptimizeRequest = OptimizeRequest()) -> OptimizeResponse:
    if not pipeline.ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Call POST /pipeline/run first.",
        )

    df = pipeline.optimize_on_demand(
        station_id=body.station_id,
        start_hour=body.start_hour,
        end_hour=body.end_hour,
        limit=body.limit,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="No recommendations match the supplied filters.")

    df = df.copy()
    df["timestamp_hour"] = df["timestamp_hour"].astype(str)

    # Revenue summary over the filtered window
    sim_slice = pipeline.simulation.copy()
    if body.station_id:
        sim_slice = sim_slice[sim_slice["station_id"].astype(str) == body.station_id]

    baseline_rev = float(sim_slice["baseline_revenue"].sum()) if not sim_slice.empty else 0.0
    dynamic_rev = float(sim_slice["dynamic_revenue"].sum()) if not sim_slice.empty else 0.0
    gain_pct = (
        round(100 * (dynamic_rev - baseline_rev) / baseline_rev, 2)
        if baseline_rev > 0
        else 0.0
    )

    return OptimizeResponse(
        rows_returned=len(df),
        baseline_revenue=round(baseline_rev, 2),
        dynamic_revenue=round(dynamic_rev, 2),
        revenue_gain_pct=gain_pct,
        recommendations=df.to_dict(orient="records"),
    )
