from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_GRID_COST_PER_KWH, OFFICIAL_ST_EVCDP_DIR


def _read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name} must contain a timestamp column.")
    return df


def _melt_matrix(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return df.melt(id_vars="timestamp", var_name="station_id", value_name=value_name)


def load_st_evcdp_station_hourly(data_dir: Path = OFFICIAL_ST_EVCDP_DIR) -> pd.DataFrame:
    """Convert official ST-EVCDP 5-minute matrices into station-hour analytical features."""
    required = ["time.csv", "occupancy.csv", "duration.csv", "volume.csv", "price.csv", "information.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing official ST-EVCDP files: {missing}")

    time = pd.read_csv(data_dir / "time.csv")
    time["timestamp_index"] = range(1, len(time) + 1)
    time["timestamp_hour"] = pd.to_datetime(
        dict(
            year=time["year"],
            month=time["month"],
            day=time["day"],
            hour=time["hour"],
            minute=time["minute"],
            second=time["second"],
        )
    ).dt.floor("h")

    occupancy = _melt_matrix(_read_matrix(data_dir / "occupancy.csv"), "occupancy_count")
    duration = _melt_matrix(_read_matrix(data_dir / "duration.csv"), "duration_hours_5min")
    volume = _melt_matrix(_read_matrix(data_dir / "volume.csv"), "energy_kwh_5min")
    price = _melt_matrix(_read_matrix(data_dir / "price.csv"), "tariff_per_kwh")

    long = (
        occupancy.merge(duration, on=["timestamp", "station_id"], how="left")
        .merge(volume, on=["timestamp", "station_id"], how="left")
        .merge(price, on=["timestamp", "station_id"], how="left")
    )
    long["timestamp_index"] = pd.to_numeric(long["timestamp"], errors="coerce").astype("Int64")
    long = long.merge(time[["timestamp_index", "timestamp_hour"]], on="timestamp_index", how="left")

    info = pd.read_csv(data_dir / "information.csv")
    info["station_id"] = info["grid"].astype(str)
    station_meta = info.rename(
        columns={
            "count": "charger_count",
            "lon": "longitude",
            "la": "latitude",
            "dynamic_pricing": "source_dynamic_pricing_flag",
            "CBD": "is_cbd",
        }
    )[
        [
            "station_id",
            "charger_count",
            "fast_count",
            "slow_count",
            "area",
            "longitude",
            "latitude",
            "is_cbd",
            "source_dynamic_pricing_flag",
        ]
    ]

    hourly = (
        long.groupby(["station_id", "timestamp_hour"], as_index=False)
        .agg(
            sessions_count=("occupancy_count", "mean"),
            occupancy_count=("occupancy_count", "mean"),
            energy_kwh_total=("energy_kwh_5min", "sum"),
            charging_duration_hours_total=("duration_hours_5min", "sum"),
            tariff_per_kwh=("tariff_per_kwh", "mean"),
        )
        .merge(station_meta, on="station_id", how="left")
    )
    hourly["charger_count"] = hourly["charger_count"].fillna(hourly["charger_count"].median()).clip(lower=1)
    hourly["utilization_rate"] = (hourly["occupancy_count"] / hourly["charger_count"]).clip(0, 1.5)
    hourly["occupied_minutes"] = hourly["occupancy_count"] * 60
    hourly["charging_minutes"] = hourly["charging_duration_hours_total"] * 60
    hourly["available_charger_minutes"] = hourly["charger_count"] * 60
    hourly["occupancy_density"] = hourly["occupancy_count"]
    hourly["queue_length_proxy"] = (hourly["occupancy_count"] - hourly["charger_count"]).clip(lower=0)
    hourly["revenue_total"] = hourly["energy_kwh_total"] * hourly["tariff_per_kwh"]
    hourly["revenue_per_session"] = hourly["revenue_total"] / hourly["sessions_count"].replace(0, pd.NA)
    hourly["energy_cost_per_kwh"] = DEFAULT_GRID_COST_PER_KWH
    hourly["estimated_energy_cost"] = hourly["energy_kwh_total"] * DEFAULT_GRID_COST_PER_KWH
    hourly["gross_margin"] = hourly["revenue_total"] - hourly["estimated_energy_cost"]
    hourly["dataset_source"] = "official_st_evcdp"

    ts = hourly["timestamp_hour"]
    hourly["date"] = ts.dt.date
    hourly["hour"] = ts.dt.hour
    hourly["day_of_week"] = ts.dt.dayofweek
    hourly["day_name"] = ts.dt.day_name()
    hourly["month"] = ts.dt.month
    hourly["is_weekend"] = hourly["day_of_week"].isin([5, 6]).astype(int)
    hourly["is_peak_hour"] = hourly["hour"].isin([7, 8, 9, 17, 18, 19, 20, 21]).astype(int)
    hourly["station_hour_index"] = hourly["station_id"].astype(str) + "_" + hourly["timestamp_hour"].astype(str)
    return hourly.sort_values(["timestamp_hour", "station_id"]).reset_index(drop=True)


def official_st_evcdp_available(data_dir: Path = OFFICIAL_ST_EVCDP_DIR) -> bool:
    return all((data_dir / name).exists() for name in ["time.csv", "occupancy.csv", "duration.csv", "volume.csv", "price.csv", "information.csv"])

