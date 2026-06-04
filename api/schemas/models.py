"""Pydantic schemas for request validation and response serialization."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Body for POST /predict.

    station_id and timestamp_hour are optional filters.
    When omitted the full test-set predictions are returned.
    """
    station_id: Optional[str] = Field(None, description="Filter to a single station ID")
    timestamp_hour: Optional[str] = Field(
        None,
        description="ISO-8601 datetime string, e.g. '2024-03-15T08:00:00'",
    )
    limit: int = Field(500, ge=1, le=5000, description="Max rows returned")


class PredictionRow(BaseModel):
    station_id: str
    timestamp_hour: str
    utilization_rate: float
    predicted_utilization: float
    best_model: str


class PredictResponse(BaseModel):
    model: str
    mae: float
    rmse: float
    r2: float
    rows_returned: int
    predictions: list[dict[str, Any]]


# ─────────────────────────────────────────────
# /optimize
# ─────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    """
    Body for POST /optimize.

    station_id / start_hour / end_hour narrow the window.
    """
    station_id: Optional[str] = Field(None)
    start_hour: Optional[str] = Field(None, description="ISO-8601 start of window")
    end_hour: Optional[str] = Field(None, description="ISO-8601 end of window")
    limit: int = Field(500, ge=1, le=5000)


class OptimizeResponse(BaseModel):
    rows_returned: int
    baseline_revenue: float
    dynamic_revenue: float
    revenue_gain_pct: float
    recommendations: list[dict[str, Any]]


# ─────────────────────────────────────────────
# /simulate
# ─────────────────────────────────────────────

class SimulateRequest(BaseModel):
    station_id: Optional[str] = None
    start_hour: Optional[str] = None
    end_hour: Optional[str] = None
    price_elasticity: float = Field(-0.15, ge=-1.0, le=0.0)
    limit: int = Field(500, ge=1, le=5000)


class SimulateResponse(BaseModel):
    rows_returned: int
    price_elasticity_used: float
    simulation: list[dict[str, Any]]


# ─────────────────────────────────────────────
# /metrics
# ─────────────────────────────────────────────

class MetricsResponse(BaseModel):
    model_metrics: list[dict[str, Any]]
    monitoring_metrics: list[dict[str, Any]]
    eda_summary: list[dict[str, Any]]


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    model_loaded: bool
    best_model: Optional[str]
    data_rows: int
    version: str
