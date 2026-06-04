"""
EV Tariff Optimization — Streamlit Dashboard
─────────────────────────────────────────────
Pure frontend. Zero ML logic. All data comes from the FastAPI backend
via dashboard/api_client.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard.api_client import (
    get_health,
    get_metrics,
    get_optimizations,
    get_predictions,
    get_simulation,
    run_pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Tariff Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def line_chart(df: pd.DataFrame, x: str, y: list[str], title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]
    for i, col in enumerate(y):
        ax.plot(pd.to_datetime(df[x]), df[col], linewidth=2.0,
                label=col.replace("_", " ").title(), color=colors[i % len(colors)])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str,
              color: str = "#2563EB") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df[x].astype(str), df[y], color=color, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=40)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def metric_val(metrics_list: list[dict], name: str) -> float:
    for row in metrics_list:
        if row.get("metric") == name:
            return float(row.get("value", 0))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — pipeline status + filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Tariff System")
    st.markdown("---")

    # ── Pipeline status ────────────────────────────────────────────────
    try:
        health = get_health()
        pipeline_ready = health.get("pipeline_ready", False)
        best_model = health.get("best_model", "—")
        data_rows = health.get("data_rows", 0)
    except Exception:
        pipeline_ready = False
        best_model = "—"
        data_rows = 0

    if pipeline_ready:
        st.success(f"✅ Pipeline Ready")
        st.caption(f"Model: **{best_model}** | Rows: **{data_rows:,}**")
    else:
        st.error("⚠️ Pipeline not ready")
        if st.button("▶ Run Pipeline (demo)", use_container_width=True):
            with st.spinner("Running full ML pipeline… (~30 s)"):
                try:
                    result = run_pipeline(demo=True)
                    st.info(result.get("message", "Started"))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("---")
    st.markdown("### Filters")
    station_filter = st.text_input("Station ID (leave blank = all)", value="")
    limit = st.slider("Max rows returned", 50, 2000, 500, 50)
    elasticity = st.slider("Price elasticity (what-if)", -0.5, -0.01, -0.15, 0.01)

    st.markdown("---")
    st.markdown(
        "<small>Architecture: **Streamlit → FastAPI → ML Agents**</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard — don't render tabs until pipeline is ready
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Agentic EV Tariff Optimization Console")
st.caption(
    "Demand forecasting · tariff optimization · pricing simulation · monitoring "
    "— powered by FastAPI + ML Agents"
)

if not pipeline_ready:
    st.info(
        "Pipeline is not running. Click **▶ Run Pipeline (demo)** in the sidebar to "
        "initialise the system with synthetic data, or add raw CSVs under `data/raw/`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Fetch data from API (cached for the session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_all(station_id: str, limit: int, elasticity: float) -> dict:
    sid = station_id.strip() or None
    metrics      = get_metrics()
    predictions  = get_predictions(station_id=sid, limit=limit)
    optimize     = get_optimizations(station_id=sid, limit=limit)
    simulation   = get_simulation(station_id=sid, price_elasticity=elasticity, limit=limit)
    return dict(
        metrics=metrics,
        predictions=predictions,
        optimize=optimize,
        simulation=simulation,
    )


with st.spinner("Fetching data from API…"):
    try:
        data = fetch_all(station_filter, limit, elasticity)
    except Exception as exc:
        st.error(f"API error: {exc}")
        st.stop()

# Parse into DataFrames
metrics_resp     = data["metrics"]
pred_df          = pd.DataFrame(data["predictions"]["predictions"])
opt_df           = pd.DataFrame(data["optimize"]["recommendations"])
sim_df           = pd.DataFrame(data["simulation"]["simulation"])
model_metrics_df = pd.DataFrame(metrics_resp["model_metrics"])
monitoring_list  = metrics_resp["monitoring_metrics"]
eda_df           = pd.DataFrame(metrics_resp["eda_summary"])

for df, cols in [
    (pred_df, ["timestamp_hour"]),
    (opt_df, ["timestamp_hour"]),
    (sim_df, ["timestamp_hour"]),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# Top KPIs
# ─────────────────────────────────────────────────────────────────────────────
revenue_gain_pct = data["optimize"]["revenue_gain_pct"]
baseline_rev     = data["optimize"]["baseline_revenue"]
dynamic_rev      = data["optimize"]["dynamic_revenue"]
avg_util         = float(sim_df["utilization_rate"].mean()) if not sim_df.empty else 0.0
peak_reduction   = metric_val(monitoring_list, "peak_congestion_reduction_proxy")
pricing_eff      = metric_val(monitoring_list, "pricing_efficiency_score")
best_model_name  = data["predictions"]["model"]
r2_val           = data["predictions"]["r2"]
rmse_val         = data["predictions"]["rmse"]

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Revenue Gain %",        f"{revenue_gain_pct:+.2f}%")
k2.metric("Dynamic Revenue",       f"${dynamic_rev:,.0f}")
k3.metric("Baseline Revenue",      f"${baseline_rev:,.0f}")
k4.metric("Avg Utilization",       f"{avg_util:.3f}")
k5.metric("Peak Congestion ↓",     f"{peak_reduction:.4f}")
k6.metric("Pricing Efficiency",    f"{pricing_eff:.3f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_forecast, tab_pricing, tab_simulate, tab_monitor, tab_arch = st.tabs([
    "📊 Network Overview",
    "🤖 Demand Forecast",
    "💰 Tariff Decisions",
    "🔁 Simulation",
    "📈 Monitoring",
    "🏗 Architecture",
])

# ── Tab 1: Network Overview ───────────────────────────────────────────────────
with tab_overview:
    left, right = st.columns([1.5, 1])

    if not sim_df.empty and "timestamp_hour" in sim_df.columns:
        hourly = (
            sim_df.groupby("timestamp_hour", as_index=False)
            .agg(
                utilization_rate=("utilization_rate", "mean"),
                adjusted_utilization_rate=("adjusted_utilization_rate", "mean"),
            )
            .sort_values("timestamp_hour")
        )
        left.pyplot(line_chart(
            hourly, "timestamp_hour",
            ["utilization_rate", "adjusted_utilization_rate"],
            "Utilization: Baseline vs Dynamic Pricing",
            "Utilization Rate",
        ))
    else:
        left.info("No simulation data available.")

    if not sim_df.empty and "station_id" in sim_df.columns:
        top_stations = (
            sim_df.groupby("station_id", as_index=False)
            .agg(utilization_rate=("utilization_rate", "mean"))
            .sort_values("utilization_rate", ascending=False)
            .head(12)
        )
        right.pyplot(bar_chart(
            top_stations, "station_id", "utilization_rate",
            "Most Utilized Stations", "Avg Utilization", "#2563EB",
        ))

    st.subheader("EDA Summary")
    st.dataframe(eda_df, use_container_width=True, hide_index=True)


# ── Tab 2: Demand Forecast ────────────────────────────────────────────────────
with tab_forecast:
    c1, c2 = st.columns([1.3, 1])

    c1.subheader("Model Comparison (Validation)")
    c1.dataframe(model_metrics_df, use_container_width=True, hide_index=True)
    c1.pyplot(bar_chart(
        model_metrics_df.sort_values("RMSE"),
        "model", "RMSE", "Validation RMSE by Model", "RMSE", "#DC2626",
    ))

    c2.metric("Selected Model", best_model_name)
    c2.metric("R²",   f"{r2_val:.4f}")
    c2.metric("RMSE", f"{rmse_val:.4f}")
    c2.metric("MAE",  f"{data['predictions']['mae']:.4f}")
    c2.caption(
        "Model selected via time-based validation — no future leakage. "
        "Features include utilization lags (1h, 2h, 24h, 168h), rolling 24h average, "
        "hour-of-day, day-of-week, and peak/weekend flags."
    )

    if not pred_df.empty and "timestamp_hour" in pred_df.columns:
        pred_agg = (
            pred_df.groupby("timestamp_hour", as_index=False)
            .agg(
                actual=("utilization_rate", "mean"),
                predicted=("predicted_utilization", "mean"),
            )
            .sort_values("timestamp_hour")
        )
        st.subheader("Actual vs Predicted Utilization (Test Set)")
        st.pyplot(line_chart(
            pred_agg, "timestamp_hour",
            ["actual", "predicted"],
            "Actual and Predicted Utilization",
            "Utilization Rate",
        ))


# ── Tab 3: Tariff Decisions ───────────────────────────────────────────────────
with tab_pricing:
    if not opt_df.empty:
        col1, col2 = st.columns([1, 1])

        rev_comparison = pd.DataFrame({
            "Type": ["Baseline Revenue", "Dynamic Revenue"],
            "Revenue ($)": [baseline_rev, dynamic_rev],
        })
        col1.subheader("Revenue Impact")
        col1.pyplot(bar_chart(
            rev_comparison, "Type", "Revenue ($)",
            "Baseline vs Dynamic Revenue", "Revenue ($)",
            "#16A34A",
        ))

        if "tariff_reason" in opt_df.columns:
            reason_counts = (
                opt_df.groupby("tariff_reason", as_index=False)
                .agg(station_hours=("tariff_reason", "count"))
                .sort_values("station_hours", ascending=False)
            )
            col2.subheader("Tariff Action Mix")
            col2.dataframe(reason_counts, use_container_width=True, hide_index=True)
            col2.pyplot(bar_chart(
                reason_counts, "tariff_reason", "station_hours",
                "Recommendation Frequency", "Station-hours", "#D97706",
            ))

        display_cols = [c for c in [
            "station_id", "timestamp_hour", "predicted_utilization",
            "recommended_tariff_per_kwh", "baseline_tariff_per_kwh",
            "tariff_reason", "tariff_change_pct",
        ] if c in opt_df.columns]

        st.subheader("Station-hour Recommendations")
        st.dataframe(
            opt_df[display_cols].sort_values("timestamp_hour", ascending=False).head(300),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No optimization data available.")


# ── Tab 4: Simulation ─────────────────────────────────────────────────────────
with tab_simulate:
    st.caption(
        f"Simulated with price elasticity = **{elasticity}** "
        "(adjust in sidebar for what-if analysis)"
    )

    if not sim_df.empty:
        sim_cols = [c for c in [
            "station_id", "timestamp_hour",
            "utilization_rate", "adjusted_utilization_rate",
            "baseline_revenue", "dynamic_revenue", "revenue_gain",
            "tariff_reason", "recommended_tariff_per_kwh",
            "peak_congestion_reduction_proxy", "waiting_time_reduction_proxy",
        ] if c in sim_df.columns]

        c1, c2 = st.columns(2)
        c1.metric("Total Baseline Revenue",  f"${sim_df['baseline_revenue'].sum():,.0f}")
        c1.metric("Total Dynamic Revenue",   f"${sim_df['dynamic_revenue'].sum():,.0f}")
        c2.metric("Total Revenue Gain",      f"${sim_df['revenue_gain'].sum():,.0f}")
        c2.metric(
            "Revenue Gain %",
            f"{100 * (sim_df['dynamic_revenue'].sum() - sim_df['baseline_revenue'].sum()) / max(sim_df['baseline_revenue'].sum(), 1):.2f}%",
        )

        st.subheader("Simulation Results")
        st.dataframe(
            sim_df[sim_cols].sort_values("timestamp_hour", ascending=False).head(300),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No simulation data available.")


# ── Tab 5: Monitoring ─────────────────────────────────────────────────────────
with tab_monitor:
    mon_df = pd.DataFrame(monitoring_list)

    if not mon_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Revenue Gain %",        f"{metric_val(monitoring_list, 'revenue_gain_pct'):.2f}%")
        c2.metric("Pricing Efficiency",    f"{metric_val(monitoring_list, 'pricing_efficiency_score'):.4f}")
        c3.metric("Customer Response Rate",f"{metric_val(monitoring_list, 'customer_response_rate'):.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Peak Congestion ↓",     f"{metric_val(monitoring_list, 'peak_congestion_reduction_proxy'):.4f}")
        c5.metric("Off-Peak Uplift",       f"{metric_val(monitoring_list, 'off_peak_uplift'):.4f}")
        c6.metric("Utilization Change %",  f"{metric_val(monitoring_list, 'utilization_change_pct'):.2f}%")

        st.subheader("Full Monitoring Metrics")
        st.dataframe(mon_df, use_container_width=True, hide_index=True)
    else:
        st.info("Monitoring metrics not available.")


# ── Tab 6: Architecture ───────────────────────────────────────────────────────
with tab_arch:
    st.subheader("System Architecture")
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                           │
│  dashboard/app.py                                               │
│  dashboard/api_client.py  ←  pure HTTP, zero ML logic          │
└─────────────────────────┬───────────────────────────────────────┘
                          │  HTTP/JSON  (localhost:8000)
┌─────────────────────────▼───────────────────────────────────────┐
│                    FASTAPI BACKEND                              │
│  api/main.py                                                    │
│  ├── GET  /              health check                           │
│  ├── POST /predict       Demand Prediction Agent                │
│  ├── POST /optimize      Tariff Optimization Agent              │
│  ├── POST /simulate      Pricing Simulation Engine              │
│  ├── GET  /metrics       Monitoring & Learning Agent            │
│  └── POST /pipeline/run  full pipeline trigger                  │
│                                                                 │
│  api/services/pipeline_service.py  ←  singleton state          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    ML / BUSINESS LOGIC (src/)                   │
│  data_loading.py   preprocessing.py   feature_engineering.py   │
│  modeling.py       tariff_engine.py   simulation.py            │
│  monitoring.py     agentic_loop.py    eda.py                    │
└─────────────────────────────────────────────────────────────────┘
""", language="text")

    st.subheader("Project Requirement Coverage")
    coverage = pd.DataFrame([
        ["Official data",        "ST-EVCDP & ACN loaders; demo generator for offline use"],
        ["Preprocessing",        "Timestamps, IDs, pricing, occupancy standardised; quality flags"],
        ["Feature engineering",  "Utilization, revenue, cost, queue proxy, lags (1h/2h/24h/168h), rolling 24h"],
        ["EDA",                  "Demand plots, weekday heatmap, station congestion ranking"],
        ["Modeling",             "Historical avg, Ridge, Random Forest, GBM — time-based validation"],
        ["Tariff optimization",  "scipy.optimize numerical solver; revenue–congestion objective; min/max guardrails"],
        ["Agentic pipeline",     "Observe → Predict → Optimize → Simulate → Monitor loop (PricingDecisionAgent)"],
        ["Simulation",           "Explicit elasticity parameter; fixed vs dynamic comparison"],
        ["Monitoring",           "Revenue gain, utilization change, congestion proxy, pricing efficiency score"],
        ["Production API",       "FastAPI backend with Pydantic schemas, CORS, background tasks, OpenAPI docs"],
        ["Transparency",         "Reason codes on every recommendation; assumptions documented"],
    ], columns=["Requirement", "Implementation"])
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("Quick-start")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start FastAPI backend (auto-runs demo pipeline on boot)
uvicorn api.main:app --reload --port 8000

# 3. Start Streamlit frontend (new terminal)
streamlit run dashboard/app.py

# 4. View interactive API docs
open http://localhost:8000/docs
""", language="bash")
