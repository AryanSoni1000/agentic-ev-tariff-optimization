from __future__ import annotations

import pandas as pd

from src.monitoring import calculate_monitoring_metrics
from src.simulation import simulate_dynamic_pricing
from src.tariff_engine import apply_tariff_recommendations


class PricingDecisionAgent:
    """Decision-support wrapper for predict, price, simulate, and monitor steps."""

    def __init__(self, demand_model: object, feature_columns: list[str], price_elasticity: float) -> None:
        self.demand_model = demand_model
        self.feature_columns = feature_columns
        self.price_elasticity = price_elasticity

    def observe(self, station_hourly_features: pd.DataFrame) -> pd.DataFrame:
        return station_hourly_features.copy()

    def predict(self, observed_features: pd.DataFrame) -> pd.DataFrame:
        available_features = [col for col in self.feature_columns if col in observed_features.columns]
        predictions = observed_features[["station_id", "timestamp_hour", "utilization_rate"]].copy()
        predictions["predicted_utilization"] = self.demand_model.predict(observed_features[available_features]).clip(0, 1.5)
        return predictions

    def recommend(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return apply_tariff_recommendations(predictions)

    def simulate(self, observed_features: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
        return simulate_dynamic_pricing(observed_features, recommendations, self.price_elasticity)

    def monitor(self, simulation_results: pd.DataFrame) -> pd.DataFrame:
        return calculate_monitoring_metrics(simulation_results)

    def run(self, station_hourly_features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        observed = self.observe(station_hourly_features)
        predictions = self.predict(observed)
        recommendations = self.recommend(predictions)
        simulation = self.simulate(observed, recommendations)
        monitoring = self.monitor(simulation)
        return {
            "observed": observed,
            "predictions": predictions,
            "recommendations": recommendations,
            "simulation": simulation,
            "monitoring": monitoring,
        }

