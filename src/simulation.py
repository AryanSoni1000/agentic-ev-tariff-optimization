"""
simulation.py
─────────────
Simulates fixed-tariff baseline vs dynamic-tariff outcomes.

Key fix vs original
───────────────────
The original used a single global price_elasticity (−0.15) for ALL hours.
This caused off-peak discounts to generate a demand_adjustment_factor > 1
(correct) but the revenue calculation still lost money because the tariff
dropped by −43%.  The fix is to clip demand_adjustment_factor correctly
and compute net revenue as:

    dynamic_revenue = adjusted_energy_kwh × recommended_tariff

which is unchanged.  The real fix is in tariff_engine.py (the tariff is
now always >= baseline when revenue would otherwise fall).  The simulation
layer is clean; we add an explicit revenue_floor_enforced flag for audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASE_TARIFF_PER_KWH, PRICE_ELASTICITY


def simulate_dynamic_pricing(
    station_hourly: pd.DataFrame,
    recommendations: pd.DataFrame,
    price_elasticity: float = PRICE_ELASTICITY,
) -> pd.DataFrame:
    """
    Merge station features with tariff recommendations and compute:
        - baseline_revenue       (fixed tariff × energy)
        - adjusted_energy_kwh    (demand response to price change)
        - dynamic_revenue        (recommended tariff × adjusted energy)
        - revenue_gain           (dynamic − baseline)
        - adjusted_utilization_rate
        - peak_congestion_reduction_proxy
        - waiting_time_reduction_proxy

    Parameters
    ----------
    station_hourly   : DataFrame with station-hour features (energy, utilization, queue)
    recommendations  : output of apply_tariff_recommendations()
    price_elasticity : demand sensitivity to price (default −0.15, negative)
    """
    merge_cols = ["station_id", "timestamp_hour"]

    base = station_hourly.merge(
        recommendations[
            merge_cols + [
                "predicted_utilization",
                "recommended_tariff_per_kwh",
                "baseline_tariff_per_kwh",
                "tariff_change_pct",
                "tariff_reason",
            ]
        ],
        on=merge_cols,
        how="inner",
    )

    # ── Revenue baseline ─────────────────────────────────────────────────
    base["baseline_revenue"] = (
        base["energy_kwh_total"] * base["baseline_tariff_per_kwh"]
    )

    # ── Demand adjustment ────────────────────────────────────────────────
    # price_elasticity is negative: price rise → demand fall, price cut → demand rise
    # Clipped to [0.75, 1.30] — realistic EV charging demand response
    base["demand_adjustment_factor"] = (
        1.0 + price_elasticity * base["tariff_change_pct"]
    ).clip(0.75, 1.30)

    # ── Dynamic revenue ──────────────────────────────────────────────────
    base["adjusted_energy_kwh"] = (
        base["energy_kwh_total"] * base["demand_adjustment_factor"]
    )
    base["dynamic_revenue"] = (
        base["adjusted_energy_kwh"] * base["recommended_tariff_per_kwh"]
    )
    base["revenue_gain"] = base["dynamic_revenue"] - base["baseline_revenue"]

    # ── Audit flag: did any row violate the revenue floor? ───────────────
    base["revenue_floor_enforced"] = (
        base["dynamic_revenue"] < base["baseline_revenue"]
    ).astype(int)

    # ── Utilization & congestion ─────────────────────────────────────────
    base["adjusted_utilization_rate"] = (
        base["utilization_rate"] * base["demand_adjustment_factor"]
    ).clip(0, 1.5)

    base["peak_congestion_reduction_proxy"] = np.maximum(
        0.0,
        base["utilization_rate"] - base["adjusted_utilization_rate"],
    )

    # ── Waiting time proxy ───────────────────────────────────────────────
    base["baseline_waiting_time"] = base["queue_length_proxy"] * 5.0
    base["adjusted_waiting_time"] = (
        base["baseline_waiting_time"] * base["demand_adjustment_factor"]
    )
    base["waiting_time_reduction_proxy"] = np.maximum(
        0.0,
        base["baseline_waiting_time"] - base["adjusted_waiting_time"],
    )

    return base
