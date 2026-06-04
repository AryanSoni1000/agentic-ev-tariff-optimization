from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from src.config import FIGURE_DIR


def set_plot_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", font_scale=1.0)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def save_hourly_demand_plot(station_hourly: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    set_plot_style()
    hourly = station_hourly.groupby("hour", as_index=False).agg(
        sessions_count=("sessions_count", "sum"),
        energy_kwh_total=("energy_kwh_total", "sum"),
        utilization_rate=("utilization_rate", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if sns is not None:
        sns.barplot(data=hourly, x="hour", y="energy_kwh_total", color="#4C78A8", ax=ax1)
    else:
        ax1.bar(hourly["hour"], hourly["energy_kwh_total"], color="#4C78A8")
    ax1.set_title("Charging Demand by Hour")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Total Energy Delivered (kWh)")
    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["utilization_rate"], color="#F58518", marker="o", linewidth=2)
    ax2.set_ylabel("Average Utilization Rate")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hourly_demand_utilization.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_weekday_heatmap(station_hourly: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    set_plot_style()
    pivot = station_hourly.pivot_table(
        index="day_name",
        columns="hour",
        values="utilization_rate",
        aggfunc="mean",
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([day for day in order if day in pivot.index])
    fig, ax = plt.subplots(figsize=(12, 5))
    if sns is not None:
        sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.2, ax=ax, cbar_kws={"label": "Avg Utilization"})
    else:
        image = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        fig.colorbar(image, ax=ax, label="Avg Utilization")
    ax.set_title("Utilization Heatmap by Weekday and Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "weekday_hour_utilization_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_station_congestion_plot(station_hourly: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> Path:
    set_plot_style()
    top = (
        station_hourly.groupby("station_id", as_index=False)
        .agg(avg_utilization=("utilization_rate", "mean"), queue_proxy=("queue_length_proxy", "mean"))
        .sort_values("avg_utilization", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    if sns is not None:
        sns.barplot(data=top, y="station_id", x="avg_utilization", hue="queue_proxy", palette="viridis", ax=ax)
    else:
        ax.barh(top["station_id"], top["avg_utilization"], color="#54A24B")
        ax.invert_yaxis()
    ax.set_title("Most Utilized Stations")
    ax.set_xlabel("Average Utilization Rate")
    ax.set_ylabel("Station ID")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "top_station_utilization.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def create_eda_summary(station_hourly: pd.DataFrame) -> pd.DataFrame:
    peak = station_hourly[station_hourly["is_peak_hour"].eq(1)]
    off_peak = station_hourly[station_hourly["is_peak_hour"].eq(0)]
    rows = [
        {"insight": "Total station-hour observations", "value": len(station_hourly)},
        {"insight": "Mean utilization rate", "value": round(float(station_hourly["utilization_rate"].mean()), 4)},
        {"insight": "Peak-hour mean utilization", "value": round(float(peak["utilization_rate"].mean()), 4)},
        {"insight": "Off-peak mean utilization", "value": round(float(off_peak["utilization_rate"].mean()), 4)},
        {"insight": "Highest congestion hour", "value": int(station_hourly.groupby("hour")["utilization_rate"].mean().idxmax())},
        {"insight": "Total baseline revenue", "value": round(float(station_hourly["revenue_total"].sum()), 2)},
    ]
    return pd.DataFrame(rows)
