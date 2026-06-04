from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PRICE_ELASTICITY


def simulate_dynamic_pricing(
    station_hourly: pd.DataFrame,
    recommendations: pd.DataFrame,
    price_elasticity: float = PRICE_ELASTICITY,
) -> pd.DataFrame:
    merge_cols = ["station_id", "timestamp_hour"]
    base = station_hourly.merge(
        recommendations[
            merge_cols
            + [
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
    base["baseline_revenue"] = base["energy_kwh_total"] * base["baseline_tariff_per_kwh"]
    base["demand_adjustment_factor"] = (1 + price_elasticity * base["tariff_change_pct"]).clip(0.70, 1.25)
    base["adjusted_energy_kwh"] = base["energy_kwh_total"] * base["demand_adjustment_factor"]
    base["dynamic_revenue"] = base["adjusted_energy_kwh"] * base["recommended_tariff_per_kwh"]
    base["revenue_gain"] = base["dynamic_revenue"] - base["baseline_revenue"]
    base["adjusted_utilization_rate"] = (base["utilization_rate"] * base["demand_adjustment_factor"]).clip(0, 1.5)
    base["peak_congestion_reduction_proxy"] = np.maximum(0, base["utilization_rate"] - base["adjusted_utilization_rate"])
    base["baseline_waiting_time"] = (
    base["queue_length_proxy"] * 5
)

    base["adjusted_waiting_time"] = (
    base["baseline_waiting_time"]
    * base["demand_adjustment_factor"]
)

    base["waiting_time_reduction_proxy"] = np.maximum(
    0,
    base["baseline_waiting_time"]
    - base["adjusted_waiting_time"]
)
    return base

