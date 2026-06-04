# ⚡ Agentic EV Tariff Optimization System

> Production-grade demand forecasting and dynamic pricing for EV charging networks —
> built with FastAPI, scikit-learn, scipy, and Streamlit.

---

## Results

| Metric | Value |
|---|---|
| **Revenue Gain (dynamic vs fixed)** | **+18.67%** (+$4,759 per simulation window) |
| **Revenue Floor Violations** | **0** |
| Best Forecasting Model | Gradient Boosting |
| R² (time-based test split) | **0.9724** |
| MAE | 0.0063 |
| RMSE | 0.0189 |
| Peak Congestion Reduction | 0.0542 |
| Pricing Efficiency Score | 0.226 |

---

## Business Problem

Static EV charging tariffs cause three simultaneous inefficiencies:

- **Peak congestion** — stations hit capacity, drivers queue and leave
- **Off-peak underutilization** — revenue left on the table during low-demand hours
- **No demand signal** — flat pricing gives drivers no incentive to shift behavior

This system builds an **agentic pricing loop** that observes real charging data,
forecasts station utilization, recommends revenue-safe dynamic tariffs, simulates
outcomes against a fixed-tariff baseline, and monitors KPIs continuously.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  STREAMLIT FRONTEND  (dashboard/app.py)                      │
│  dashboard/api_client.py  ←  pure HTTP, zero ML logic        │
└──────────────────────────┬───────────────────────────────────┘
                           │  HTTP / JSON  :8000
┌──────────────────────────▼───────────────────────────────────┐
│  FASTAPI BACKEND  (api/)                                     │
│  GET  /              health + pipeline status                 │
│  POST /predict       Demand Prediction Agent                  │
│  POST /optimize      Tariff Optimization Agent                │
│  POST /simulate      Pricing Simulation Engine                │
│  GET  /metrics       Monitoring & Learning Agent              │
│  POST /pipeline/run  full pipeline trigger                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│  ML / BUSINESS LOGIC  (src/)                                 │
│  data_loading · preprocessing · feature_engineering          │
│  modeling · tariff_engine · simulation · monitoring           │
│  agentic_loop · eda · utils                                  │
└──────────────────────────────────────────────────────────────┘
```

The Streamlit frontend is a **pure HTTP client** — zero ML imports.
All business logic lives in the API layer, independently testable and deployable.

---

## Agentic Decision Loop

```
Observe  →  station-hour operating state (utilization, energy, queue)
Predict  →  Gradient Boosting forecasts next-hour utilization
Optimize →  scipy.minimize_scalar finds revenue-maximising tariff
             subject to: dynamic_revenue ≥ baseline_revenue (hard floor)
Simulate →  fixed-tariff baseline vs dynamic-tariff outcome
Monitor  →  revenue gain %, congestion reduction, pricing efficiency score
```

Every recommendation includes an interpretable reason code and is bounded
by configurable min/max tariff guardrails.

---

## Project Structure

```
ev_tariff_system/
├── api/                          FastAPI backend
│   ├── main.py                   app factory, CORS, startup lifespan
│   ├── routers/
│   │   ├── health.py             GET /
│   │   ├── predict.py            POST /predict
│   │   ├── optimize.py           POST /optimize
│   │   ├── simulate.py           POST /simulate
│   │   ├── metrics.py            GET /metrics
│   │   └── pipeline.py           POST /pipeline/run
│   ├── schemas/models.py         Pydantic request/response models
│   └── services/
│       └── pipeline_service.py   ML singleton, owns all state
│
├── src/                          ML and business logic
│   ├── config.py                 paths and constants
│   ├── data_loading.py           ACN + ST-EVCDP loaders + demo generator
│   ├── preprocessing.py          cleaning, schema standardisation
│   ├── feature_engineering.py    station-hour feature construction
│   ├── modeling.py               train/evaluate 4 models, time split
│   ├── tariff_engine.py          scipy multi-objective optimizer
│   ├── simulation.py             fixed vs dynamic pricing simulation
│   ├── monitoring.py             KPI calculation
│   ├── agentic_loop.py           PricingDecisionAgent
│   ├── eda.py                    EDA plots and summary table
│   └── utils.py                  IO helpers
│
├── dashboard/
│   ├── app.py                    Streamlit frontend (API calls only)
│   └── api_client.py             centralised HTTP client
│
├── data/raw/                     drop ACN / ST-EVCDP files here
├── models/                       saved model + metadata (auto-generated)
├── outputs/                      figures, metrics, simulation (auto-generated)
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

---

## Datasets

| Dataset | Description |
|---|---|
| **ACN-Data** | Adaptive Charging Network sessions from Caltech / JPL |
| **ST-EVCDP** | Large-scale urban EV charging with temporal and spatial patterns |
| **Demo** | Synthetic 18,000-session dataset for offline pipeline testing |

Raw files go in:
```
data/raw/acn_data/      ← CSV, JSON, JSONL, Parquet
data/raw/urbanev/       ← ST-EVCDP matrix files
```

---

## Feature Engineering

| Feature | Description |
|---|---|
| `utilization_rate` | occupied_minutes / available_charger_minutes |
| `queue_length_proxy` | (utilization − 0.70) × 10, floored at 0 |
| `revenue_per_session` | total_revenue / sessions_count |
| `gross_margin` | revenue − estimated grid cost |
| `is_peak_hour` | 1 if hour ∈ {7,8,9,17,18,19,20,21} |
| `utilization_lag_1h` | lag-1 autoregressive demand signal |
| `utilization_lag_24h` | same hour yesterday |
| `utilization_rolling_24h` | 24h rolling mean (shift-1 to prevent leakage) |
| `sessions_rolling_24h` | 24h rolling session count |

---

## Models Compared

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Gradient Boosting** ✓ | **0.0063** | **0.0189** | **0.9724** |
| Random Forest | 0.0061 | 0.0191 | 0.9717 |
| Ridge Regression | 0.0152 | 0.0279 | 0.9394 |
| Historical Average | 0.0763 | 0.1141 | −0.011 |

Evaluation uses **time-based train/test split (80/20)** — no future data leakage.

---

## Dynamic Tariff Logic

The optimizer uses `scipy.minimize_scalar` with a multi-objective function:

```
Maximise:  0.60 × revenue_gain
         + 0.25 × congestion_reduction
         + 0.15 × utilization_improvement

Subject to: dynamic_revenue ≥ baseline_revenue  (hard constraint, always enforced)
            tariff ∈ [MIN_TARIFF, MAX_TARIFF]
            demand elasticity = −0.20
```

**Tariff bands by utilization regime:**

| Utilization | Band | Strategy |
|---|---|---|
| ≥ 0.75 (peak) | $0.45 – $0.60 | Surcharge: revenue + congestion relief |
| 0.40 – 0.75 (balanced) | $0.35 – $0.50 | Moderate surcharge: revenue priority |
| < 0.40 (off-peak) | $0.35 – $0.40 | Hold near baseline: no revenue-negative discounts |

---

## Simulation

Demand response is modelled with an explicit price-elasticity assumption:

```
adjusted_demand = baseline_demand × (1 + elasticity × price_change_pct)
dynamic_revenue = adjusted_demand × recommended_tariff
```

This is **scenario analysis**, not a causal claim. Real demand response
requires observed A/B pricing data.

---

## Monitoring KPIs

| KPI | Description |
|---|---|
| `revenue_gain_pct` | % lift in revenue vs fixed-tariff baseline |
| `peak_congestion_reduction_proxy` | avg utilization drop during peak hours |
| `waiting_time_reduction_proxy` | estimated queue time saved |
| `customer_response_rate` | avg utilization change across all hours |
| `pricing_efficiency_score` | weighted composite of all KPIs |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start FastAPI backend (auto-runs demo pipeline on boot)
uvicorn api.main:app --reload --port 8000

# 3. Start Streamlit frontend (new terminal)
streamlit run dashboard/app.py

# 4. View interactive API docs
open http://localhost:8000/docs

# 5. Run original CLI pipeline
python main.py --demo
```

---

## Docker

```bash
docker-compose up --build

# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + pipeline status |
| `POST` | `/predict` | Demand Prediction Agent |
| `POST` | `/optimize` | Tariff Optimization Agent |
| `POST` | `/simulate` | Fixed vs dynamic pricing simulation |
| `GET` | `/metrics` | Monitoring KPIs + model metrics |
| `POST` | `/pipeline/run` | Trigger full pipeline (re)training |

Interactive Swagger UI at `http://localhost:8000/docs`.

---

## Deployment

**Render.com** (simplest for portfolio):
Push to GitHub → connect Render → `render.yaml` handles both services.

**Google Cloud Run:**
```bash
docker build -t ev-tariff-api .
gcloud run deploy ev-tariff-api --image ev-tariff-api --platform managed
```

---

## Assumptions & Limitations

- **Elasticity is assumed, not estimated**: `−0.20` is literature-backed for EV charging but not derived from this dataset. Real estimation requires A/B pricing experiments or instrumental variables.
- **Demo data is synthetic**: R² of 0.97 reflects clean synthetic patterns. Real-world data will have additional noise, missing values, and sensor errors.
- **Greedy per-station optimization**: A joint network optimizer (LP/MILP) would account for demand substitution between nearby stations.
- **No external signals**: Weather, local events, grid spot prices, and fleet composition are not modelled.
- **Simulation uses assumed elasticity**: Results are scenario analysis, not observed outcomes.

---

## Resume Bullets

- Architected a production-grade EV charging analytics system with a FastAPI backend, Streamlit frontend, and a 5-step agentic pricing loop (Observe → Predict → Optimize → Simulate → Monitor)
- Trained and compared 4 demand forecasting models using time-based validation; Gradient Boosting achieved R²=0.9724, MAE=0.0063 on station-hour utilization prediction
- Designed a revenue-safe tariff optimization engine using scipy numerical optimization with a multi-objective function (60% revenue, 25% congestion, 15% utilization) and hard revenue floor — achieving +18.67% revenue gain with zero floor violations across 3,038 station-hours
- Exposed all ML functionality through a typed REST API with Pydantic schemas, CORS middleware, background task execution, and auto-generated OpenAPI documentation
- Implemented fixed vs dynamic pricing simulation with configurable price elasticity for what-if scenario analysis

---

## Optional Enhancements

- [ ] SHAP feature importance for model explainability
- [ ] MLflow experiment tracking
- [ ] Joint network-level optimization (LP/MILP)
- [ ] Weather and grid price feature integration
- [ ] Real elasticity estimation via causal inference
- [ ] Reinforcement learning pricing agent
- [ ] Real-time streaming pipeline (Kafka)
