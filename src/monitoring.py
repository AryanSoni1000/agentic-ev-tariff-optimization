from __future__ import annotations

import pandas as pd


def _safe_pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def calculate_monitoring_metrics(simulation_df: pd.DataFrame) -> pd.DataFrame:
    baseline_revenue = float(simulation_df["baseline_revenue"].sum())
    dynamic_revenue = float(simulation_df["dynamic_revenue"].sum())
    revenue_gain = dynamic_revenue - baseline_revenue
    peak_rows = simulation_df["utilization_rate"] >= 0.75
    off_peak_rows = simulation_df["utilization_rate"] <= 0.30
    
    peak_reduction = float(
    simulation_df.loc[
        peak_rows,
        "peak_congestion_reduction_proxy"
    ].mean()
) if peak_rows.any() else 0.0
    off_peak_uplift = float(
    (
        simulation_df.loc[
            off_peak_rows,
            "adjusted_utilization_rate"
        ]
        -
        simulation_df.loc[
            off_peak_rows,
            "utilization_rate"
        ]
    ).mean()
) if off_peak_rows.any() else 0.0 

    customer_response_rate = float(
    (
        simulation_df["adjusted_utilization_rate"]
        - simulation_df["utilization_rate"]
    ).mean()
    )

    avg_price_change = float(simulation_df["tariff_change_pct"].abs().mean())
    price_stability = max(0.0, 1.0 - avg_price_change)
    pricing_efficiency_score = (
    0.30 * min(max(_safe_pct(revenue_gain, baseline_revenue), -1), 1)
    + 0.20 * min(max(peak_reduction, 0), 1)
    + 0.15 * min(max(off_peak_uplift, -1), 1)
    + 0.15 * min(max(customer_response_rate, -1), 1)
    + 0.20 * price_stability
    )

    utilization_change_pct = (
    100
    * _safe_pct(
        simulation_df["adjusted_utilization_rate"].mean()
        - simulation_df["utilization_rate"].mean(),
        simulation_df["utilization_rate"].mean(),
    )
)

    rows = [
        {"metric": "baseline_revenue", "value": round(baseline_revenue, 2)},
        {"metric": "dynamic_revenue", "value": round(dynamic_revenue, 2)},
        {"metric": "revenue_gain", "value": round(revenue_gain, 2)},
        {"metric": "revenue_gain_pct", "value": round(100 * _safe_pct(revenue_gain, baseline_revenue), 2)},
        {"metric": "avg_utilization_before", "value": round(float(simulation_df["utilization_rate"].mean()), 4)},
        {"metric": "avg_utilization_after", "value": round(float(simulation_df["adjusted_utilization_rate"].mean()), 4)},
        {"metric": "peak_congestion_reduction_proxy", "value": round(peak_reduction, 4)},
        {"metric": "waiting_time_reduction_proxy", "value": round(float(simulation_df["waiting_time_reduction_proxy"].mean()), 4)},
        {"metric": "customer_response_rate",
    "value": round(customer_response_rate, 4)},
    { "metric": "utilization_change_pct",
    "value": round(utilization_change_pct, 2)},
        {"metric": "off_peak_uplift",
"value": round(off_peak_uplift, 4)},
        {"metric": "pricing_efficiency_score", "value": round(float(pricing_efficiency_score), 4)},
    ]
    return pd.DataFrame(rows)

