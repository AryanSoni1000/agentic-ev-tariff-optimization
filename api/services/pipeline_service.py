"""
PipelineService
───────────────
Runs the full ML pipeline **once** at startup (or on demand) and keeps
all artefacts in memory so every API request is sub-millisecond.

This is intentionally stateless at the HTTP layer: all mutable state
lives here, injected into FastAPI via Depends().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import: src package lives one level above api/
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    METRIC_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    SIMULATION_DIR,
    TABLE_DIR,
    ensure_directories,
    PRICE_ELASTICITY,
)
from src.data_loading import generate_demo_sessions, load_all_sessions
from src.feature_engineering import create_modeling_dataset, create_station_hourly_features
from src.modeling import (
    FEATURE_COLUMNS,
    save_model,
    train_and_evaluate_models,
)
from src.monitoring import calculate_monitoring_metrics
from src.official_data import load_st_evcdp_station_hourly, official_st_evcdp_available
from src.preprocessing import build_data_quality_report, clean_sessions
from src.simulation import simulate_dynamic_pricing
from src.tariff_engine import apply_tariff_recommendations
from src.utils import write_frame
from src.agentic_loop import PricingDecisionAgent


class PipelineService:
    """
    Singleton-style service that owns every pipeline artefact.

    Usage
    -----
    svc = PipelineService()
    svc.run(demo=True)          # executes pipeline
    svc.predictions             # pd.DataFrame
    svc.monitoring_metrics      # pd.DataFrame
    """

    def __init__(self) -> None:
        self._ready: bool = False
        self._best_model: Optional[object] = None
        self._best_model_name: Optional[str] = None

        self.station_hourly: pd.DataFrame = pd.DataFrame()
        self.modeling_df: pd.DataFrame = pd.DataFrame()
        self.predictions: pd.DataFrame = pd.DataFrame()
        self.recommendations: pd.DataFrame = pd.DataFrame()
        self.simulation: pd.DataFrame = pd.DataFrame()
        self.model_metrics: pd.DataFrame = pd.DataFrame()
        self.monitoring_metrics: pd.DataFrame = pd.DataFrame()
        self.eda_summary: pd.DataFrame = pd.DataFrame()
        self.quality_report: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def best_model_name(self) -> Optional[str]:
        return self._best_model_name

    @property
    def data_rows(self) -> int:
        return len(self.station_hourly)

    def run(self, demo: bool = False) -> None:
        """Execute the full pipeline and cache every artefact."""
        logger.info("Pipeline starting (demo=%s)", demo)
        ensure_directories()

        raw_sessions = load_all_sessions()

        if official_st_evcdp_available():
            logger.info("Loading official ST-EVCDP data")
            self.station_hourly = load_st_evcdp_station_hourly()
            self.quality_report = self.station_hourly.groupby(
                "dataset_source", as_index=False
            ).agg(
                station_hours=("station_hour_index", "count"),
                stations=("station_id", "nunique"),
                total_energy_kwh=("energy_kwh_total", "sum"),
                total_revenue=("revenue_total", "sum"),
            )
        elif raw_sessions.empty:
            if not demo:
                raise RuntimeError(
                    "No raw data found. Add files under data/raw/ or set demo=True."
                )
            logger.info("No raw data – generating demo sessions")
            raw_sessions = generate_demo_sessions()
            clean = clean_sessions(raw_sessions)
            self.quality_report = build_data_quality_report(raw_sessions, clean)
            self.station_hourly = create_station_hourly_features(clean)
        else:
            clean = clean_sessions(raw_sessions)
            self.quality_report = build_data_quality_report(raw_sessions, clean)
            self.station_hourly = create_station_hourly_features(clean)

        self.modeling_df = create_modeling_dataset(self.station_hourly)

        # ── Modeling ──────────────────────────────────────────────────
        self._best_model, self.model_metrics, self.predictions = train_and_evaluate_models(
            self.modeling_df
        )
        self._best_model_name = str(self.model_metrics.iloc[0]["model"])

        # ── Agentic loop ──────────────────────────────────────────────
        # IMPORTANT: The model was trained on modeling_df which contains
        # lag/rolling features (utilization_lag_1h, energy_lag_24h, etc.).
        # station_hourly does NOT have these columns, so we must run the
        # agent on modeling_df and join simulation back to station_hourly.
        agent = PricingDecisionAgent(
            demand_model=self._best_model,
            feature_columns=FEATURE_COLUMNS,
            price_elasticity=PRICE_ELASTICITY,
        )

        # Step 1: use already-computed test-set predictions for recommendations
        # predictions has: station_id, timestamp_hour, utilization_rate, predicted_utilization
        self.recommendations = agent.recommend(self.predictions)

        # Step 2: simulate against modeling_df rows that overlap with predictions.
        # Use a merge (O(n log n)) instead of row-wise lambda (O(n²)).
        pred_keys = self.predictions[["station_id", "timestamp_hour"]].copy()
        pred_keys["station_id"] = pred_keys["station_id"].astype(str)
        pred_keys["timestamp_hour"] = pd.to_datetime(pred_keys["timestamp_hour"])

        modeling_keyed = self.modeling_df.copy()
        modeling_keyed["station_id"] = modeling_keyed["station_id"].astype(str)
        modeling_keyed["timestamp_hour"] = pd.to_datetime(modeling_keyed["timestamp_hour"])

        station_hourly_subset = modeling_keyed.merge(
            pred_keys, on=["station_id", "timestamp_hour"], how="inner"
        )

        self.simulation = agent.simulate(station_hourly_subset, self.recommendations)
        self.monitoring_metrics = agent.monitor(self.simulation)

        # ── EDA summary ───────────────────────────────────────────────
        from src.eda import create_eda_summary
        self.eda_summary = create_eda_summary(self.station_hourly)

        # ── Persist artefacts ─────────────────────────────────────────
        write_frame(self.station_hourly, PROCESSED_DIR / "station_hourly_features.parquet")
        write_frame(self.modeling_df, PROCESSED_DIR / "modeling_dataset.parquet")
        write_frame(self.predictions, PROCESSED_DIR / "test_predictions.csv")
        write_frame(self.model_metrics, METRIC_DIR / "model_metrics.csv")
        write_frame(self.monitoring_metrics, METRIC_DIR / "monitoring_metrics.csv")
        write_frame(self.recommendations, SIMULATION_DIR / "tariff_recommendations.csv")
        write_frame(self.simulation, SIMULATION_DIR / "fixed_vs_dynamic_simulation.csv")
        write_frame(self.eda_summary, TABLE_DIR / "eda_summary.csv")
        write_frame(self.quality_report, METRIC_DIR / "data_quality_report.csv")
        save_model(self._best_model, self.model_metrics, MODEL_DIR)

        self._ready = True
        logger.info(
            "Pipeline complete. Best model: %s | R²=%.4f | RMSE=%.4f",
            self._best_model_name,
            float(self.model_metrics.iloc[0]["R2"]),
            float(self.model_metrics.iloc[0]["RMSE"]),
        )

    def predict_on_demand(
        self,
        station_id: Optional[str] = None,
        timestamp_hour: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Return cached predictions with optional filters."""
        df = self.predictions.copy()
        if station_id:
            df = df[df["station_id"].astype(str) == station_id]
        if timestamp_hour:
            df = df[df["timestamp_hour"].astype(str).str.startswith(timestamp_hour[:16])]
        return df.head(limit)

    def optimize_on_demand(
        self,
        station_id: Optional[str] = None,
        start_hour: Optional[str] = None,
        end_hour: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Return cached recommendations with optional filters."""
        df = self.recommendations.copy()
        if station_id:
            df = df[df["station_id"].astype(str) == station_id]
        if start_hour:
            df = df[pd.to_datetime(df["timestamp_hour"]) >= pd.to_datetime(start_hour)]
        if end_hour:
            df = df[pd.to_datetime(df["timestamp_hour"]) <= pd.to_datetime(end_hour)]
        return df.head(limit)

    def simulate_on_demand(
        self,
        station_id: Optional[str] = None,
        start_hour: Optional[str] = None,
        end_hour: Optional[str] = None,
        price_elasticity: float = PRICE_ELASTICITY,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Re-run simulation with a custom elasticity value,
        or return cached results when elasticity matches default.
        """
        if abs(price_elasticity - PRICE_ELASTICITY) < 1e-6:
            df = self.simulation.copy()
        else:
            df = simulate_dynamic_pricing(
                self.station_hourly, self.recommendations, price_elasticity
            )

        if station_id:
            df = df[df["station_id"].astype(str) == station_id]
        if start_hour:
            df = df[pd.to_datetime(df["timestamp_hour"]) >= pd.to_datetime(start_hour)]
        if end_hour:
            df = df[pd.to_datetime(df["timestamp_hour"]) <= pd.to_datetime(end_hour)]
        return df.head(limit)


# Module-level singleton – imported by FastAPI app and all routers
pipeline = PipelineService()
