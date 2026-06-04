from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEFAULT_CHARGERS_PER_STATION, DEFAULT_GRID_COST_PER_KWH


def add_time_features(df: pd.DataFrame, timestamp_col: str = "start_time") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_name"] = ts.dt.day_name()
    df["month"] = ts.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19, 20, 21]).astype(int)
    df["time_block"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 10, 16, 21, 23],
        labels=["overnight", "morning", "midday", "evening", "late"],
    ).astype(str)
    return df


def create_station_hourly_features(
    sessions: pd.DataFrame,
    chargers_per_station: int = DEFAULT_CHARGERS_PER_STATION,
    grid_cost_per_kwh: float = DEFAULT_GRID_COST_PER_KWH,
) -> pd.DataFrame:
    sessions = add_time_features(sessions)
    sessions["timestamp_hour"] = pd.to_datetime(sessions["start_time"]).dt.floor("h")
    sessions["occupied_minutes"] = sessions["connection_duration_hours"].clip(0, 1) * 60
    sessions["charging_minutes"] = sessions["charging_duration_hours"].clip(0, 1) * 60
    grouped = (
        sessions.groupby(["station_id", "timestamp_hour"], as_index=False)
        .agg(
            sessions_count=("session_id", "count"),
            energy_kwh_total=("energy_kwh", "sum"),
            avg_energy_kwh=("energy_kwh", "mean"),
            avg_connection_duration_hours=("connection_duration_hours", "mean"),
            occupied_minutes=("occupied_minutes", "sum"),
            charging_minutes=("charging_minutes", "sum"),
            revenue_total=("revenue", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
    )
    grouped["available_charger_minutes"] = chargers_per_station * 60
    grouped["utilization_rate"] = (grouped["occupied_minutes"] / grouped["available_charger_minutes"]).clip(0, 1.5)
    grouped["occupancy_density"] = grouped["occupied_minutes"] / grouped["available_charger_minutes"]
    grouped["queue_length_proxy"] = np.maximum(0, (grouped["utilization_rate"] - 0.70) * 10)
    grouped["revenue_per_session"] = grouped["revenue_total"] / grouped["sessions_count"].replace(0, np.nan)
    grouped["energy_cost_per_kwh"] = grid_cost_per_kwh
    grouped["estimated_energy_cost"] = grouped["energy_kwh_total"] * grid_cost_per_kwh
    grouped["gross_margin"] = grouped["revenue_total"] - grouped["estimated_energy_cost"]
    grouped = add_time_features(grouped, "timestamp_hour")
    grouped["station_hour_index"] = grouped["station_id"].astype(str) + "_" + grouped["timestamp_hour"].astype(str)
    return grouped.sort_values(["timestamp_hour", "station_id"]).reset_index(drop=True)


def create_modeling_dataset(station_hourly: pd.DataFrame) -> pd.DataFrame:
    df = station_hourly.copy().sort_values(["station_id", "timestamp_hour"])
    if "avg_energy_kwh" not in df.columns:
        df["avg_energy_kwh"] = df["energy_kwh_total"] / df["sessions_count"].replace(0, np.nan)
    if "avg_connection_duration_hours" not in df.columns:
        df["avg_connection_duration_hours"] = df.get("charging_duration_hours_total", df["energy_kwh_total"] * 0)
    for lag in [1, 2, 24, 168]:
        df[f"utilization_lag_{lag}h"] = df.groupby("station_id")["utilization_rate"].shift(lag)
        df[f"energy_lag_{lag}h"] = df.groupby("station_id")["energy_kwh_total"].shift(lag)
    df["utilization_rolling_24h"] = (
        df.groupby("station_id")["utilization_rate"].shift(1).rolling(24, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    df["sessions_rolling_24h"] = (
        df.groupby("station_id")["sessions_count"].shift(1).rolling(24, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    df = df.dropna(subset=["utilization_lag_1h", "utilization_rolling_24h"]).reset_index(drop=True)
    return df
