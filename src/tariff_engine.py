from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.config import (
    BASE_TARIFF_PER_KWH,
    MIN_TARIFF_PER_KWH,
    MAX_TARIFF_PER_KWH,
)


def objective_function(
    tariff: float,
    predicted_utilization: float,
) -> float:
    """
    Optimization objective.

    We minimize:
        -(revenue)
        + congestion penalty

    Since scipy minimizes, revenue is negated.
    """

    # Simple demand elasticity assumption
    elasticity = 0.30

    adjusted_utilization = predicted_utilization * (
        1 - elasticity * ((tariff - BASE_TARIFF_PER_KWH) / BASE_TARIFF_PER_KWH)
    )

    adjusted_utilization = max(0.05, min(1.0, adjusted_utilization))

    revenue = tariff * adjusted_utilization

# Peak congestion penalty
    congestion_penalty = 0

    if adjusted_utilization > 0.80:
        congestion_penalty = (
        adjusted_utilization - 0.80
    ) * 15

# Under-utilization penalty
    underutilization_penalty = 0

    if adjusted_utilization < 0.40:
        underutilization_penalty = (
        0.40 - adjusted_utilization
    ) * 20

    score = (
    -revenue
    + congestion_penalty
    + underutilization_penalty
)

    return score


def optimize_tariff(predicted_utilization: float):
    """
    Finds best tariff using numerical optimization.
    """

    result = minimize_scalar(
        lambda t: objective_function(
            t,
            predicted_utilization,
        ),
        bounds=(
            MIN_TARIFF_PER_KWH,
            MAX_TARIFF_PER_KWH,
        ),
        method="bounded",
    )

    tariff = round(float(result.x), 4)

    if predicted_utilization > 0.80:
        reason = "optimized_peak_management"
    elif predicted_utilization < 0.30:
        reason = "optimized_offpeak_growth"
    else:
        reason = "optimized_balanced_pricing"

    return tariff, reason


def apply_tariff_recommendations(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    df = predictions.copy()

    optimized = df["predicted_utilization"].apply(
        optimize_tariff
    )

    df["recommended_tariff_per_kwh"] = [
        x[0] for x in optimized
    ]

    df["tariff_reason"] = [
        x[1] for x in optimized
    ]

    df["baseline_tariff_per_kwh"] = (
        BASE_TARIFF_PER_KWH
    )

    df["tariff_change_pct"] = (
        df["recommended_tariff_per_kwh"]
        / BASE_TARIFF_PER_KWH
    ) - 1

    return df