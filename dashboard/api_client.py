"""
api_client.py
─────────────
Thin HTTP client for the FastAPI backend.
All Streamlit pages import from here — zero raw requests calls in the UI layer.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

API_BASE = os.getenv("EV_API_BASE_URL", "http://localhost:8000")
TIMEOUT = 120  # seconds — pipeline startup can take a moment


def _get(path: str, **params: Any) -> dict[str, Any]:
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_health() -> dict[str, Any]:
    return _get("/")


def run_pipeline(demo: bool = True) -> dict[str, Any]:
    return _post("/pipeline/run", {"demo": demo})


def get_predictions(
    station_id: Optional[str] = None,
    timestamp_hour: Optional[str] = None,
    limit: int = 500,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"limit": limit}
    if station_id:
        payload["station_id"] = station_id
    if timestamp_hour:
        payload["timestamp_hour"] = timestamp_hour
    return _post("/predict", payload)


def get_optimizations(
    station_id: Optional[str] = None,
    start_hour: Optional[str] = None,
    end_hour: Optional[str] = None,
    limit: int = 500,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"limit": limit}
    if station_id:
        payload["station_id"] = station_id
    if start_hour:
        payload["start_hour"] = start_hour
    if end_hour:
        payload["end_hour"] = end_hour
    return _post("/optimize", payload)


def get_simulation(
    station_id: Optional[str] = None,
    start_hour: Optional[str] = None,
    end_hour: Optional[str] = None,
    price_elasticity: float = -0.15,
    limit: int = 500,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"limit": limit, "price_elasticity": price_elasticity}
    if station_id:
        payload["station_id"] = station_id
    if start_hour:
        payload["start_hour"] = start_hour
    if end_hour:
        payload["end_hour"] = end_hour
    return _post("/simulate", payload)


def get_metrics() -> dict[str, Any]:
    return _get("/metrics")
