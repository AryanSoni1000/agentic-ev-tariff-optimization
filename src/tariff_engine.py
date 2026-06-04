"""
tariff_engine.py  —  v2
───────────────────────
Multi-objective tariff optimiser for EV charging stations.

Objective (scipy minimises, so we negate):
    maximise: 0.60 * norm_revenue_gain
            + 0.25 * norm_congestion_reduction
            + 0.15 * norm_utilization_improvement

Hard business constraint:
    dynamic_revenue >= baseline_revenue  at all times

Elasticity insight (from calibration)
──────────────────────────────────────
With a realistic EV elasticity of −0.20, any tariff *above* baseline
always generates more revenue (price effect > demand loss).  Discounts
always lose net revenue.  Therefore:

  • Peak / balanced hours  → surcharge up to MAX_TARIFF (revenue + congestion)
  • Off-peak hours         → hold at baseline (0.35) rather than discount;
                             a small surcharge band [0.35, 0.42] still passes
                             the revenue floor while gently managing light
                             congestion — better for the portfolio than a
                             blunt discount that costs money.

This reflects real-world EV operator behaviour: operators rarely discount
below cost-recovery; they use off-peak marketing and subscription plans
(outside this model scope) to stimulate demand, not tariff cuts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.config import (
    BASE_TARIFF_PER_KWH,
    MIN_TARIFF_PER_KWH,
    MAX_TARIFF_PER_KWH,
)

# ── Demand elasticity ────────────────────────────────────────────────────────
# EV charging is inelastic.  −0.20 = 10% price rise → 2% demand drop.
_ELASTICITY: float = -0.20

# ── Objective weights ────────────────────────────────────────────────────────
_W_REVENUE:     float = 0.60
_W_CONGESTION:  float = 0.25
_W_UTILIZATION: float = 0.15

# ── Tariff bands by utilization regime ──────────────────────────────────────
# These narrow the search space so the optimizer doesn't always fly to MAX.
_PEAK_THRESHOLD:    float = 0.75   # util >= this → peak surcharge band
_BALANCED_LOW:      float = 0.40   # util in [0.40, 0.75] → balanced band
_OFFPEAK_THRESHOLD: float = 0.40   # util < this → hold near baseline

# Per-regime tariff bounds (tighter than global MIN/MAX)
_PEAK_MIN:     float = 0.45        # peak: always surcharge
_PEAK_MAX:     float = MAX_TARIFF_PER_KWH

_BALANCED_MIN: float = BASE_TARIFF_PER_KWH        # balanced: at or above baseline
_BALANCED_MAX: float = 0.50

_OFFPEAK_MIN:  float = BASE_TARIFF_PER_KWH        # off-peak: hold at baseline
_OFFPEAK_MAX:  float = BASE_TARIFF_PER_KWH + 0.05 # allow tiny premium, never discount


def _demand_factor(tariff: float) -> float:
    """Price-elasticity demand adjustment, clipped to realistic range."""
    pct_change = (tariff - BASE_TARIFF_PER_KWH) / BASE_TARIFF_PER_KWH
    return float(np.clip(1.0 + _ELASTICITY * pct_change, 0.75, 1.30))


def _revenue_gain_norm(tariff: float, utilization: float) -> float:
    """Normalised revenue gain in [−1, 1]."""
    df   = _demand_factor(tariff)
    dyn  = tariff * utilization * df
    base = BASE_TARIFF_PER_KWH * utilization
    return float(np.clip((dyn - base) / max(base, 1e-6), -1.0, 1.0))


def objective_function(
    tariff: float,
    predicted_utilization: float,
) -> float:
    """
    Returns negative weighted score (scipy minimises).
    Returns +999 if the revenue floor is violated.
    """
    df  = _demand_factor(tariff)
    dyn = tariff * predicted_utilization * df
    bsl = BASE_TARIFF_PER_KWH * predicted_utilization

    # ── Hard constraint: revenue floor ───────────────────────────────────
    if dyn < bsl:
        return 999.0

    # ── Revenue gain (normalised) ─────────────────────────────────────────
    norm_rev = float(np.clip((dyn - bsl) / max(bsl, 1e-6), 0.0, 1.0))

    # ── Congestion reduction (normalised) ─────────────────────────────────
    adj_util = predicted_utilization * df
    if predicted_utilization >= _PEAK_THRESHOLD:
        reduction = max(0.0, predicted_utilization - adj_util)
        norm_cong = float(np.clip(reduction / max(predicted_utilization, 1e-6), 0.0, 1.0))
    else:
        norm_cong = 0.0

    # ── Utilization improvement (normalised) ──────────────────────────────
    # For off-peak: holding at baseline = neutral (0). No negative signal.
    util_delta = adj_util - predicted_utilization
    if predicted_utilization < _OFFPEAK_THRESHOLD:
        headroom  = max(1.0 - predicted_utilization, 1e-6)
        norm_util = float(np.clip(util_delta / headroom, -0.5, 0.5))
    else:
        norm_util = 0.0

    score = (
        _W_REVENUE     * norm_rev
        + _W_CONGESTION  * norm_cong
        + _W_UTILIZATION * norm_util
    )
    return -score   # minimise


def _regime_bounds(predicted_utilization: float) -> tuple[float, float]:
    """Return (lo, hi) tariff search bounds for this utilization level."""
    if predicted_utilization >= _PEAK_THRESHOLD:
        return _PEAK_MIN, _PEAK_MAX
    elif predicted_utilization >= _BALANCED_LOW:
        return _BALANCED_MIN, _BALANCED_MAX
    else:
        return _OFFPEAK_MIN, _OFFPEAK_MAX


def optimize_tariff(predicted_utilization: float) -> tuple[float, str]:
    """
    Find the optimal tariff for a single station-hour.

    Returns (recommended_tariff_per_kwh, reason_code).
    """
    lo, hi = _regime_bounds(predicted_utilization)

    result = minimize_scalar(
        lambda t: objective_function(t, predicted_utilization),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-5},
    )

    tariff = round(float(np.clip(result.x, lo, hi)), 4)

    # ── Post-hoc revenue floor guardrail ─────────────────────────────────
    df  = _demand_factor(tariff)
    dyn = tariff * predicted_utilization * df
    bsl = BASE_TARIFF_PER_KWH * predicted_utilization
    if dyn < bsl:
        tariff = BASE_TARIFF_PER_KWH   # safe fallback

    # ── Reason code ───────────────────────────────────────────────────────
    if predicted_utilization >= _PEAK_THRESHOLD:
        reason = "peak_surcharge_optimized"
    elif predicted_utilization >= _BALANCED_LOW:
        reason = "balanced_surcharge_optimized"
    else:
        reason = "offpeak_baseline_held"

    return tariff, reason


def apply_tariff_recommendations(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply optimize_tariff to every row and return enriched DataFrame."""
    df = predictions.copy()

    optimized = df["predicted_utilization"].apply(optimize_tariff)

    df["recommended_tariff_per_kwh"] = [x[0] for x in optimized]
    df["tariff_reason"]              = [x[1] for x in optimized]
    df["baseline_tariff_per_kwh"]    = BASE_TARIFF_PER_KWH
    df["tariff_change_pct"]          = (
        df["recommended_tariff_per_kwh"] / BASE_TARIFF_PER_KWH
    ) - 1

    return df
