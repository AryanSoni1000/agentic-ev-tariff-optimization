from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_TARIFF_PER_KWH, RAW_DIR, RANDOM_STATE


COLUMN_ALIASES = {
    "session_id": ["session_id", "id", "_id", "connectionid", "transaction_id"],
    "station_id": ["station_id", "evse_id", "evseid", "station", "spaceid", "space_id", "charger_id"],
    "user_id": ["user_id", "userid", "user", "driver_id"],
    "start_time": ["start_time", "connectiontime", "connect_time", "started_at", "start_datetime"],
    "end_time": ["end_time", "disconnecttime", "disconnect_time", "ended_at", "end_datetime"],
    "done_charging_time": ["done_charging_time", "donechargingtime", "charge_done_time"],
    "energy_kwh": ["energy_kwh", "kwhdelivered", "kwh_delivered", "energy", "energy_delivered"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "site_id": ["site_id", "site", "location_id", "location"],
    "tariff_per_kwh": ["tariff_per_kwh", "price_per_kwh", "tariff", "price"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    rename_map: dict[str, str] = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = alias.lower().replace(" ", "_")
            if alias_norm in df.columns:
                rename_map[alias_norm] = standard
                break
    return df.rename(columns=rename_map)


def _flatten_acn_records(payload: object) -> pd.DataFrame:
    if isinstance(payload, dict):
        if "_items" in payload:
            records = payload["_items"]
        elif "sessions" in payload:
            records = payload["sessions"]
        else:
            records = list(payload.values()) if all(isinstance(v, dict) for v in payload.values()) else [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    return pd.json_normalize(records)


def load_files(folder: Path, source_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".parquet"}:
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            df = pd.json_normalize(rows)
        else:
            with path.open("r", encoding="utf-8") as handle:
                df = _flatten_acn_records(json.load(handle))
        df = _normalize_columns(df)
        df["dataset_source"] = source_name
        df["raw_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_acn_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return load_files(raw_dir / "acn_data", "acn")


def load_urbanev_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return load_files(raw_dir / "urbanev", "urbanev")


def load_all_sessions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = [load_acn_data(raw_dir), load_urbanev_data(raw_dir)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def generate_demo_sessions(n_sessions: int = 18000, n_stations: int = 18) -> pd.DataFrame:
    """Create realistic demo data so the full pipeline is runnable before raw data is added."""
    rng = np.random.default_rng(RANDOM_STATE)
    start = pd.Timestamp("2024-01-01")
    timestamps = start + pd.to_timedelta(rng.integers(0, 120 * 24 * 60, n_sessions), unit="m")
    station_ids = rng.choice([f"ST-{idx:03d}" for idx in range(1, n_stations + 1)], n_sessions)
    hour = timestamps.hour
    commute_peak = ((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 21))
    duration = rng.gamma(shape=2.4, scale=1.2, size=n_sessions) + commute_peak * rng.uniform(0.8, 1.8, n_sessions)
    duration = np.clip(duration, 0.25, 10)
    energy = np.clip(duration * rng.normal(6.0, 1.6, n_sessions), 1.0, 85.0)
    end_times = timestamps + pd.to_timedelta(duration, unit="h")
    sites = rng.choice(["Caltech", "JPL", "Urban-Core", "Urban-West"], n_sessions)
    tariff = BASE_TARIFF_PER_KWH + rng.normal(0, 0.015, n_sessions)
    return pd.DataFrame(
        {
            "session_id": [f"DEMO-{idx:06d}" for idx in range(n_sessions)],
            "station_id": station_ids,
            "user_id": rng.choice([f"U-{idx:05d}" for idx in range(2500)], n_sessions),
            "start_time": timestamps,
            "end_time": end_times,
            "energy_kwh": energy,
            "site_id": sites,
            "latitude": rng.normal(34.14, 0.08, n_sessions),
            "longitude": rng.normal(-118.13, 0.08, n_sessions),
            "tariff_per_kwh": tariff,
            "dataset_source": "demo",
            "raw_file": "synthetic_pipeline_demo",
        }
    )
