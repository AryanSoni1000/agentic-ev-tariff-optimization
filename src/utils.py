import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_frame(df: pd.DataFrame, path: Path) -> Path:
    """Write parquet when available, otherwise fall back to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(path, index=False)
    return path


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path.with_suffix(".csv"))
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

