from __future__ import annotations

import argparse

from src.config import (
    METRIC_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    SIMULATION_DIR,
    TABLE_DIR,
    ensure_directories,
)
from src.data_loading import generate_demo_sessions, load_all_sessions
from src.eda import (
    create_eda_summary,
    save_hourly_demand_plot,
    save_station_congestion_plot,
    save_weekday_heatmap,
)
from src.feature_engineering import create_modeling_dataset, create_station_hourly_features
from src.modeling import save_model, train_and_evaluate_models
from src.monitoring import calculate_monitoring_metrics
from src.official_data import load_st_evcdp_station_hourly, official_st_evcdp_available
from src.preprocessing import build_data_quality_report, clean_sessions
from src.simulation import simulate_dynamic_pricing
from src.tariff_engine import apply_tariff_recommendations
from src.utils import write_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EV charging demand forecasting and dynamic tariff simulation pipeline.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use generated demo sessions when raw ACN/UrbanEV files are not available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    raw_sessions = load_all_sessions()
    if official_st_evcdp_available():
        station_hourly = load_st_evcdp_station_hourly()
        clean = raw_sessions if not raw_sessions.empty else station_hourly.head(0)
        quality_report = station_hourly.groupby("dataset_source", as_index=False).agg(
            station_hours=("station_hour_index", "count"),
            stations=("station_id", "nunique"),
            total_energy_kwh=("energy_kwh_total", "sum"),
            total_revenue=("revenue_total", "sum"),
        )
    elif raw_sessions.empty:
        if not args.demo:
            raise FileNotFoundError(
                "No raw data found. Add files under data/raw/acn_data or data/raw/urbanev, "
                "or run `python main.py --demo` to execute the full demo pipeline."
            )
        raw_sessions = generate_demo_sessions()
        clean = clean_sessions(raw_sessions)
        quality_report = build_data_quality_report(raw_sessions, clean)
        station_hourly = create_station_hourly_features(clean)
    else:
        clean = clean_sessions(raw_sessions)
        quality_report = build_data_quality_report(raw_sessions, clean)
        station_hourly = create_station_hourly_features(clean)
    modeling_df = create_modeling_dataset(station_hourly)

    write_frame(clean, PROCESSED_DIR / "unified_sessions.parquet")
    write_frame(station_hourly, PROCESSED_DIR / "station_hourly_features.parquet")
    write_frame(modeling_df, PROCESSED_DIR / "modeling_dataset.parquet")
    write_frame(quality_report, METRIC_DIR / "data_quality_report.csv")

    eda_summary = create_eda_summary(station_hourly)
    write_frame(eda_summary, TABLE_DIR / "eda_summary.csv")
    save_hourly_demand_plot(station_hourly)
    save_weekday_heatmap(station_hourly)
    save_station_congestion_plot(station_hourly)

    best_model, metrics_df, predictions = train_and_evaluate_models(modeling_df)
    save_model(best_model, metrics_df, MODEL_DIR)
    write_frame(metrics_df, METRIC_DIR / "model_metrics.csv")
    write_frame(predictions, PROCESSED_DIR / "test_predictions.csv")

    recommendations = apply_tariff_recommendations(predictions)
    simulation = simulate_dynamic_pricing(station_hourly, recommendations)
    monitoring_metrics = calculate_monitoring_metrics(simulation)
    write_frame(recommendations, SIMULATION_DIR / "tariff_recommendations.csv")
    write_frame(simulation, SIMULATION_DIR / "fixed_vs_dynamic_simulation.csv")
    write_frame(monitoring_metrics, METRIC_DIR / "monitoring_metrics.csv")

    print("Pipeline completed successfully.")
    print(f"Best model: {metrics_df.iloc[0]['model']}")
    print(metrics_df.to_string(index=False))
    print(monitoring_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
