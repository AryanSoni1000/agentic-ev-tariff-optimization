from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR, RANDOM_STATE


FEATURE_COLUMNS = [
    "sessions_count",
    "energy_kwh_total",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "utilization_lag_1h",
    "utilization_lag_2h",
    "utilization_lag_24h",
    "energy_lag_1h",
    "energy_lag_24h",
    "utilization_rolling_24h",
    "sessions_rolling_24h",
]
TARGET_COLUMN = "utilization_rate"


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: dict[str, float]


class HistoricalAverageRegressor:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalAverageRegressor":
        self.global_mean_ = float(y.mean())
        train = X.copy()
        train["_target"] = y.values
        self.lookup_ = train.groupby(["hour", "day_of_week"])["_target"].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.lookup_.get((row.hour, row.day_of_week), self.global_mean_) for row in X.itertuples()])


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp_hour").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def get_candidate_models() -> dict[str, object]:
    return {
        "historical_average": HistoricalAverageRegressor(),
        "ridge_regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=140, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
        ),
    }


def train_and_evaluate_models(modeling_df: pd.DataFrame) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    available_features = [col for col in FEATURE_COLUMNS if col in modeling_df.columns]
    train_df, test_df = time_based_split(modeling_df)
    X_train, y_train = train_df[available_features], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[available_features], test_df[TARGET_COLUMN]

    results: list[ModelResult] = []
    prediction_frame = test_df[["station_id", "timestamp_hour", TARGET_COLUMN]].copy()
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, 1.5)
        prediction_frame[f"pred_{name}"] = preds
        results.append(ModelResult(name=name, model=model, metrics=evaluate_predictions(y_test, preds)))

    metrics_df = pd.DataFrame([{"model": result.name, **result.metrics} for result in results]).sort_values("RMSE")
    best_name = metrics_df.iloc[0]["model"]
    best_model = next(result.model for result in results if result.name == best_name)
    prediction_frame["predicted_utilization"] = prediction_frame[f"pred_{best_name}"]
    prediction_frame["best_model"] = best_name
    return best_model, metrics_df.reset_index(drop=True), prediction_frame


def save_model(model: object, metrics_df: pd.DataFrame, model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "demand_utilization_model.pkl")
    metadata = {
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "best_model": metrics_df.iloc[0]["model"],
        "metrics": metrics_df.to_dict(orient="records"),
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
