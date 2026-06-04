from __future__ import annotations

import pandas as pd

from src.config import BASE_TARIFF_PER_KWH


REQUIRED_COLUMNS = [
    "session_id",
    "station_id",
    "start_time",
    "end_time",
    "energy_kwh",
    "tariff_per_kwh",
    "dataset_source",
]


def standardize_session_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    if "session_id" in df.columns:
        missing_ids = df["session_id"].isna()
        df.loc[missing_ids, "session_id"] = [f"SESSION-{idx:08d}" for idx in df.index[missing_ids]]
    df["station_id"] = df["station_id"].fillna("UNKNOWN_STATION").astype(str)
    df["dataset_source"] = df["dataset_source"].fillna("unknown").astype(str)
    df["tariff_per_kwh"] = pd.to_numeric(df["tariff_per_kwh"], errors="coerce").fillna(BASE_TARIFF_PER_KWH)
    return df


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_session_schema(df)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True).dt.tz_convert(None)
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce", utc=True).dt.tz_convert(None)
    if "done_charging_time" in df.columns:
        df["done_charging_time"] = pd.to_datetime(df["done_charging_time"], errors="coerce", utc=True).dt.tz_convert(None)

    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df["missing_start_time"] = df["start_time"].isna()
    df["missing_end_time"] = df["end_time"].isna()
    df["missing_energy_kwh"] = df["energy_kwh"].isna()

    df = df.dropna(subset=["start_time", "end_time"]).copy()
    df["connection_duration_hours"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600
    df = df[df["connection_duration_hours"].between(0.05, 24)].copy()
    df["energy_kwh"] = df["energy_kwh"].fillna(df["energy_kwh"].median())
    df = df[df["energy_kwh"].between(0.1, 150)].copy()

    if "done_charging_time" in df.columns:
        done_duration = (df["done_charging_time"] - df["start_time"]).dt.total_seconds() / 3600
        df["charging_duration_hours"] = done_duration.clip(lower=0.05, upper=df["connection_duration_hours"])
    else:
        df["charging_duration_hours"] = df["connection_duration_hours"]

    df["revenue"] = df["energy_kwh"] * df["tariff_per_kwh"]
    df["data_quality_flag"] = "clean"
    df.loc[df["station_id"].eq("UNKNOWN_STATION"), "data_quality_flag"] = "missing_station_id"
    return df.reset_index(drop=True)


def build_data_quality_report(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "raw_sessions", "value": len(raw)},
        {"metric": "clean_sessions", "value": len(clean)},
        {"metric": "removed_sessions", "value": len(raw) - len(clean)},
        {"metric": "unique_stations", "value": clean["station_id"].nunique() if not clean.empty else 0},
        {"metric": "total_energy_kwh", "value": round(float(clean["energy_kwh"].sum()), 2) if not clean.empty else 0},
        {"metric": "total_revenue_baseline", "value": round(float(clean["revenue"].sum()), 2) if not clean.empty else 0},
    ]
    return pd.DataFrame(rows)

