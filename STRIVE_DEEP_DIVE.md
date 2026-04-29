<div align="center">

# 🛡️ STRIVE — Complete Technical Reference

### *Everything from A to Z: What it is, Why it exists, How every part works*

> This document is the exhaustive companion to [README.md](README.md).  
> If you want a quick-start guide, go there.  
> If you want to understand *everything*, you are in the right place.

</div>

---

## Table of Contents

1. [What Is STRIVE?](#1-what-is-strive)
2. [Why Does STRIVE Exist?](#2-why-does-strive-exist)
3. [Why This Technology Stack?](#3-why-this-technology-stack)
4. [Repository Layout — Every File Explained](#4-repository-layout--every-file-explained)
5. [Data: Sources, Formats, Licences](#5-data-sources-formats-licences)
6. [Data Pipeline: Offline Training](#6-data-pipeline-offline-training)
7. [Feature Engineering: The 12-Feature Schema](#7-feature-engineering-the-12-feature-schema)
8. [System Formulas (Deep Reference)](#8-system-formulas-deep-reference)
9. [Machine Learning: Model Design and Training](#9-machine-learning-model-design-and-training)
10. [Inference Layer: From Coordinates to Risk Score](#10-inference-layer-from-coordinates-to-risk-score)
11. [Explainability: SHAP Values and NLG Summaries](#11-explainability-shap-values-and-nlg-summaries)
12. [Road Graph Layer: Graph Resolution and Caching](#12-road-graph-layer-graph-resolution-and-caching)
13. [Routing Engine: Yen's K-Shortest and Risk-Exposure Weighting](#13-routing-engine-yens-k-shortest-and-risk-exposure-weighting)
14. [API Layer: Every Endpoint Documented](#14-api-layer-every-endpoint-documented)
15. [Database: Schema and Migrations](#15-database-schema-and-migrations)
16. [Weather Integration](#16-weather-integration)
17. [Frontend: Dashboard, Map, and Components](#17-frontend-dashboard-map-and-components)
18. [Deployment: Docker and Environment Configuration](#18-deployment-docker-and-environment-configuration)
19. [Testing Strategy](#19-testing-strategy)
20. [Performance and Scalability](#20-performance-and-scalability)
21. [Known Limitations and Future Work](#21-known-limitations-and-future-work)
22. [Glossary](#22-glossary)

---

## 1. What Is STRIVE?

**STRIVE** stands for **Spatio-Temporal Risk Intelligence and Vehicular Safety Engine**.

It is a full-stack web application that answers one question for drivers:

> *"Given where I am and where I want to go, which route minimises my probability of being involved in a crash?"*

STRIVE does this by:

1. **Scoring** every road segment on a 0–100 risk scale, derived from historical crash data, live weather, time of day, and road properties.
2. **Routing** — finding up to three physically distinct alternative paths between any two points, ranked from safest to riskiest.
3. **Explaining** — showing users *why* a particular route scored the way it did, using SHAP (SHapley Additive exPlanations) values and a natural-language advisory.
4. **Working globally** — for any city on Earth, not just a single pre-defined region.

### What STRIVE Is Not

- It is **not** a real-time traffic system. It does not consume live CCTV feeds or GPS probe data.
- It is **not** a turn-by-turn navigation app. It is a route *recommendation* and *risk analysis* tool.
- It is **not** a production SaaS product (yet). It is an open research prototype designed to be fully reproducible on a laptop.

---

## 2. Why Does STRIVE Exist?

### The Problem with Conventional Navigation

Applications like Google Maps and Apple Maps optimise for **time** and **distance**. They treat all road segments as equivalent in terms of safety, modulo traffic speed. Yet roads differ enormously in terms of:

- geometry (curvature, intersections)
- speed limits
- pedestrian/cyclist mixing
- lighting
- weather sensitivity
- and crucially: **historical incident rates**

### The Opportunity

Three converging developments made STRIVE possible at low cost:

| Development | Impact |
|---|---|
| **OpenStreetMap** global road network, freely available | No need to buy proprietary map data |
| **NHTSA FARS** — US federal fatal accident database, public domain | Ground-truth labels for historical crash locations |
| **Gradient-boosted trees + SHAP** — fast, accurate, interpretable | ML that works on tabular data without a GPU and explains its predictions |
| **OpenWeatherMap free tier** — real-time weather for any lat/lon | Live contextual feature for every prediction |

### Why Explainability Matters

A black-box model that says "take this route" will not be trusted. STRIVE shows users the top contributing factors (rain, night, historical crash rate, road class) so they can make an informed decision.

---

## 3. Why This Technology Stack?

Every technology choice was deliberate. Here is the full rationale:

### Python 3.10+

- The de facto language of data science and ML.
- OSMnx, XGBoost, SHAP, NetworkX — all first-class Python libraries.
- FastAPI, SQLAlchemy, Alembic — mature, type-safe Python web stack.
- Async support via `asyncio` for concurrent HTTP handling.

### FastAPI

- **Automatic OpenAPI / Swagger docs** — zero extra work to get `/docs`.
- **Pydantic v2** — strict request/response validation with Python type hints.
- **Performance** — async ASGI server (Uvicorn + Starlette) comparable to Node.js.
- **Why not Django/Flask?** Django is too opinionated for a pure API; Flask has no built-in validation.

### XGBoost

- Best-in-class accuracy on tabular data (wins most Kaggle competitions in this domain).
- **Native SHAP support** — TreeExplainer runs in O(features × depth) time.
- Fast CPU inference — scoring 10,000 segments takes ~50 ms on a laptop.
- **Why not a neural network?** Neural networks require GPU for training, need far more data, and lack native SHAP support. For 12 tabular features, XGBoost is definitively better.

### SHAP

- Model-agnostic in theory but tree-model-optimised in practice (TreeExplainer).
- **Shapley values** are mathematically grounded — the only attribution method that satisfies efficiency, symmetry, dummy, and additivity axioms simultaneously.
- Returns per-feature contributions that sum to the prediction difference from the expected value.
- **Why not LIME or integrated gradients?** LIME is stochastic and inconsistent. Integrated gradients are for neural networks.

### NetworkX + OSMnx

- **OSMnx** — the canonical Python library for downloading and processing OpenStreetMap road networks as directed graphs. It handles coordinate projection, road simplification, and speed-limit cleaning.
- **NetworkX** — provides `nx.shortest_simple_paths()` (Yen's algorithm), A* search, and graph utilities.
- **Why not GraphHopper / OSRM?** Those are standalone routing servers. Using NetworkX keeps everything in-process, eliminates an extra service, and allows us to inject custom edge weights derived from the ML model.

### PostgreSQL 15

- Mature, battle-tested relational database.
- **JSONB** type for storing GeoJSON segment geometry without needing a full PostGIS install.
- **Why not PostGIS?** PostGIS adds complexity. For the MVP, JSONB geometry + Python-side spatial queries is sufficient.
- **Why not SQLite for production?** SQLite does not support concurrent writes. PostgreSQL handles parallel API requests correctly.

### SQLAlchemy 2.0 + Alembic

- **SQLAlchemy 2.0** — type-safe mapped columns, async session support, no legacy `query()` API.
- **Alembic** — version-controlled schema migrations. The schema evolves without losing data.

### OSMnx for Map Data

- Downloads road graphs from Overpass API (OpenStreetMap backend).
- Handles node/edge projection, speed limit estimation, simplification, and GraphML serialisation.
- **Why not GeoFabrik PBF?** PBF files are large (country-scale). OSMnx fetches only the bounding box needed, which is faster and uses less disk.

### Next.js 14 + MapLibre GL

- **Next.js 14** — React with app router, server-side rendering, and file-based routing.
- **MapLibre GL** — open-source WebGL map renderer (fork of Mapbox GL JS v1, MIT licensed). Renders OSM tiles and GeoJSON overlays at 60 fps.
- **Why not Leaflet?** Leaflet uses Canvas/SVG, not WebGL. It cannot handle thousands of coloured route segments smoothly.
- **Why not Google Maps JS API?** Proprietary, metered billing, not open-source.

### Docker + Docker Compose

- Reproducible environment — same result on any machine.
- `docker compose up --build -d` — one command to start PostgreSQL + FastAPI backend.
- Volume mounts for `data/`, `models/`, and `app/` enable live-reload development without rebuilding the image.

---

## 4. Repository Layout — Every File Explained

```
STRIVE/
│
├── app/                              ← FastAPI backend package
│   ├── __init__.py
│   ├── main.py                       ← App factory: creates FastAPI(), adds CORS, registers routers
│   ├── weather.py                    ← OpenWeatherMap HTTP client + 5-min in-process cache
│   │
│   ├── routers/                      ← One file per API tag
│   │   ├── __init__.py
│   │   ├── risk.py                   ← GET /v1/risk/segment · GET /v1/risk/heatmap
│   │   ├── route.py                  ← POST /v1/route/safe (main routing + XAI logic)
│   │   └── explain.py                ← GET /v1/explain/segment (research endpoint)
│   │
│   ├── ml/                           ← Machine learning layer
│   │   ├── __init__.py
│   │   ├── features.py               ← build_feature_vector(), FEATURE_NAMES, validation
│   │   └── inference.py              ← load_model(), explain_prediction(), explain_segments()
│   │
│   ├── routing/                      ← Graph and pathfinding layer
│   │   ├── __init__.py
│   │   ├── graph.py                  ← get_graph_for_points(), named-city cache, OSM failover
│   │   └── astar.py                  ← alternative_paths() Yen's K-shortest, risk-exposure weight
│   │
│   └── db/                           ← Database layer
│       ├── models.py                 ← RoadSegment, Accident ORM models
│       └── session.py                ← SQLAlchemy engine, SessionLocal, get_db() dependency
│
├── frontend/                         ← Next.js 14 application
│   └── src/
│       ├── app/
│       │   ├── dashboard/page.tsx    ← Main dashboard: sidebar + map, auth guard, route selector
│       │   └── login/page.tsx        ← Simple login page (localStorage auth token)
│       └── components/
│           ├── LiveMap.tsx           ← MapLibre GL map, click-to-pin, route fetch, tier overlays
│           ├── ShapModal.tsx         ← Animated SHAP factor breakdown modal
│           └── ui/
│               └── BorderGlow.tsx    ← Styled border animation component
│
├── scripts/                          ← Offline data processing and training scripts
│   ├── download_data.py              ← Download NHTSA FARS CSV files
│   ├── download_osm_network.py       ← Download OSM GraphML for a city
│   ├── download_geofabrik.py         ← Alternative: download from GeoFabrik PBF
│   ├── download_region.py            ← Download a specific bounding box
│   ├── snap_accidents.py             ← Snap crash records to nearest OSM edge
│   ├── compute_accident_rates.py     ← Calculate historical_accident_rate per segment
│   ├── build_features.py             ← Batch build_feature_vector() for training set
│   ├── split_dataset.py              ← Chronological train/val/test split
│   ├── train_baseline.py             ← Train a simple logistic regression baseline
│   ├── train_model.py                ← Train XGBoost + run Optuna tuning + save artefacts
│   ├── tune_hyperparams.py           ← Standalone Optuna hyperparameter search
│   ├── evaluate_model.py             ← AUROC, AUPRC, F1, ECE computation
│   ├── explain_shap.py               ← Global SHAP feature importance plots
│   ├── save_artefacts.py             ← Serialise model + feature_config to models/
│   ├── seed_data.py                  ← Load sample segments + accidents into PostgreSQL
│   ├── warmup_cities.py              ← Pre-download all named-city graphs to cache
│   ├── benchmark_performance.py      ← Latency benchmarks for inference + routing
│   ├── create_synthetic_network.py   ← Generate a synthetic road graph for testing
│   └── generate_research_report.py   ← Produce a LaTeX/Markdown evaluation report
│
├── models/                           ← Saved model artefacts (git-ignored if large)
│   ├── model.pkl                     ← Trained XGBoost model (pickle)
│   ├── baseline.pkl                  ← Logistic regression baseline
│   ├── feature_config.json           ← Feature names, ranges, version metadata
│   └── best_params.json              ← Best Optuna hyperparameters
│
├── data/
│   ├── raw/
│   │   ├── road_network.graphml      ← Pre-downloaded LA road graph (static)
│   │   └── *.csv                     ← NHTSA FARS CSVs
│   ├── processed/
│   │   └── segment_rates.parquet     ← Per-segment historical_accident_rate lookup
│   ├── splits/                       ← train.parquet, val.parquet, test.parquet
│   └── cache/graphs/                 ← Auto-generated bbox GraphML files (from live OSM download)
│
├── alembic/                          ← Alembic migration scripts
│   ├── env.py
│   └── versions/
│       └── *.py                      ← Schema migration files
│
├── tests/
│   ├── conftest.py                   ← Shared fixtures (test DB, mock model, test graph)
│   ├── unit/                         ← Unit tests for features.py, inference.py, astar.py
│   ├── integration/                  ← API integration tests (TestClient)
│   ├── e2e/                          ← End-to-end tests (full stack)
│   └── performance/                  ← Latency regression tests
│
├── docs/
│   └── PRD.md                        ← Product Requirements Document
│
├── cache/                            ← Miscellaneous in-process cache directory
├── reports/                          ← Generated evaluation reports
│
├── Dockerfile                        ← Multi-stage Python image for the API
├── docker-compose.yml                ← db + api services
├── alembic.ini                       ← Alembic configuration
├── requirements.txt                  ← Python dependencies (pinned versions)
├── .env.example                      ← Environment variable template
├── .gitignore
├── .dockerignore
├── LICENSE                           ← MIT
├── README.md                         ← Quick-start guide
└── STRIVE_DEEP_DIVE.md               ← ← You are here
```

---

## 5. Data: Sources, Formats, Licences

### 5.1 NHTSA FARS (Fatality Analysis Reporting System)

- **Publisher:** US National Highway Traffic Safety Administration
- **Content:** Every fatal traffic crash in the United States since 1975
- **Granularity:** Per-crash record with latitude, longitude, timestamp, severity, road type
- **Format:** CSV (one file per year)
- **Licence:** Public domain (US government work)
- **URL:** https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
- **How STRIVE uses it:** After download, crashes are *snapped* to the nearest OSM road segment. The snapping produces a count of crashes per segment per year, which becomes `historical_accident_rate`.

### 5.2 OpenStreetMap (via OSMnx)

- **Publisher:** OpenStreetMap contributors
- **Content:** Global road network — nodes (intersections), edges (road segments), attributes (road class, speed limit, geometry, name)
- **Format:** GraphML (serialised by OSMnx), or PBF (raw OSM format)
- **Licence:** Open Database Licence (ODbL) — attribution required, share-alike for derived databases
- **URL:** https://www.openstreetmap.org / https://overpass-api.de
- **How STRIVE uses it:**
  - Road class (`highway` tag) → `road_class` feature
  - Speed limit (`maxspeed` / OSMnx `speed_kph` estimate) → `speed_limit_kmh` feature
  - Segment geometry → route LineString geometry in API responses
  - Graph structure → nodes and edges for pathfinding

### 5.3 OpenWeatherMap

- **Publisher:** OpenWeather Ltd
- **Content:** Current weather conditions for any lat/lon
- **Format:** JSON REST response
- **Licence:** Free tier — 1,000 calls/day, 60 calls/minute
- **URL:** https://api.openweathermap.org/data/2.5/weather
- **How STRIVE uses it:** At inference time, the midpoint of origin+destination is queried. Four features are extracted: `precipitation_mm`, `visibility_km`, `wind_speed_ms`, `temperature_c`.

---

## 6. Data Pipeline: Offline Training

The training pipeline runs **once** (or whenever you want to retrain) and produces the `models/model.pkl` artefact that the live API loads.

### Step 1 — Download Crash Records

```bash
python scripts/download_data.py --city "Los Angeles, CA" --years 2021 2022 2023
```

### Step 2 — Download Road Network

```bash
python scripts/download_osm_network.py --city "Los Angeles, CA"
```

### Step 3 — Snap Accidents to Segments

```bash
python scripts/snap_accidents.py
```

### Step 4 — Compute Historical Rates

```bash
python scripts/compute_accident_rates.py
```

### Step 5 — Build Feature Vectors

```bash
python scripts/build_features.py
```

### Step 6 — Chronological Split

```bash
python scripts/split_dataset.py
```

### Step 7 — Train Model

```bash
python scripts/train_model.py
```

### Step 8 — Seed Database

```bash
python scripts/seed_data.py
```

---

## 7. Feature Engineering: The 12-Feature Schema

The feature engineering pipeline lives in `app/ml/features.py`. It is the **single source of truth** for both training and inference — the exact same `build_feature_vector()` function is called in both contexts.

### The 12 Features

#### Time Features (extracted from timestamp)

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `hour_of_day` | int | 0–23 | Crash risk peaks in late evening (22:00–02:00) and school run hours |
| `day_of_week` | int | 0–6 | Weekends have different crash profiles to weekdays |
| `month` | int | 1–12 | Seasonal effects: winter rain/ice, summer heat |
| `night_indicator` | float | 0 or 1 | 1 if hour ≥ 20 or hour < 6 |

#### Road Features (extracted from OSM edge attributes)

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `road_class` | float | 0–4 | motorway/trunk=0, primary=1, secondary=2, tertiary=3, residential/unclassified=4 |
| `speed_limit_kmh` | float | 0–200 | Higher speed → higher injury severity per crash |

**Road class mapping:**
```python
ROAD_CLASS_MAPPING = {
    'motorway': 0, 'trunk': 0,
    'primary': 1, 'secondary': 2, 'tertiary': 3,
    'residential': 4, 'unclassified': 4,
}
```

#### Weather Features (from OpenWeatherMap)

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `precipitation_mm` | float | 0–100 | Rain and snow combined (mm/h) |
| `visibility_km` | float | 0–50 | OWM visibility metres ÷ 1000 |
| `wind_speed_ms` | float | 0–50 | m/s |
| `temperature_c` | float | -50–60 | °C |

#### Derived Features

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `rain_on_congestion` | float | 0–1 | Rain danger amplified under slower effective speed |
| `historical_accident_rate` | float | 0–100 | Normalised crash frequency for this segment |

---

## 8. System Formulas (Deep Reference)

This chapter consolidates the core equations used across STRIVE: feature engineering, inference, routing, explainability aggregation, guards, caching, and NLG thresholds.

> **Notation**
>
> - Segment/edge is an OSM graph edge `(u, v, key)` in a `MultiDiGraph`.
> - `risk_score` is in **[0, 100]** unless otherwise stated.
> - `α` (alpha) is the user’s safety weight in **[0, 1]**.
> - `tt` denotes travel time in seconds for an edge.

### 8.1 Risk score scaling (model output → API units)

The ML model produces a probability:

```text
p = P(incident | x)  where  p ∈ [0, 1]
```

STRIVE maps it to a human-facing 0–100 score:

```text
risk_score = 100 × p
```

### 8.2 Fallback risk model (linear logit + sigmoid)

When `models/model.pkl` is missing, STRIVE uses a deterministic heuristic model.

Let `x_i` be a feature value, `w_i` the heuristic weight, and `b` the bias:

```text
z = b + Σ_i (w_i × x_i)
p = σ(z) = 1 / (1 + e^(−z))
risk_score = 100 × p
```

### 8.3 Derived feature: `night_indicator`

From timestamp:

```text
night_indicator = 1  if hour ≥ 20 OR hour < 6
night_indicator = 0  otherwise
```

### 8.4 Derived feature: `rain_on_congestion`

Given:
- `precipitation_mm` (mm/h, rain + snow combined)
- `speed_limit_kmh`

Compute:

```text
speed_ratio = max(0.6 × speed_limit_kmh / 100, 0.1)
rain_on_congestion = (precipitation_mm / 100) × (1 − min(speed_ratio, 1))
```

### 8.5 Travel-time normalization used by routing

For an edge travel time `tt` (seconds):

```text
travel_time_norm = min(tt / max_tt, 1)
```

Where the robust normalizer is computed per graph:

```text
max_tt = percentile_95( { tt_e for all edges e in graph } )
```

### 8.6 Risk normalization used by routing

```text
risk_norm = clamp(risk_score / 100, 0, 1)
```

### 8.7 Core routing objective: α-weighted risk exposure

The per-edge cost used for pathfinding is:

```text
edge_cost = (1 − α) × travel_time_norm + α × (risk_norm × travel_time_norm)
```

Key properties:
- `α = 0` → time-only objective
- `α = 1` → risk exposure objective

### 8.8 Route-level aggregates

For a path with edges `E`:

**Total duration (minutes):**
```text
duration_min = (Σ_{e∈E} tt_e) / 60
```

**Average segment risk:**
```text
avg_risk_score = (1 / |E|) × Σ_{e∈E} risk_score_e
```

### 8.9 SHAP additivity

For trained tree models, SHAP satisfies:

```text
f(x) = E[f(x)] + Σ_i φ_i
```

### 8.10 Route-level SHAP aggregation

Let `φ_i(e)` be feature `i`’s SHAP value on edge `e`:

```text
φ_i(route) = (1 / |E|) × Σ_{e∈E} φ_i(e)
```

A conceptual significance filter:

```text
include feature i if |φ_i(route)| > τ
```

### 8.11 Operational guards (thresholds)

These thresholds define system limits:

**Pin distance guard:**
```text
reject if nearest_road_distance_m > 600
```

**Trip distance guard:**
```text
reject if trip_distance_km > 50
```

**Graph size guard:**
```text
reject if |E_graph| > 250,000
```

### 8.12 Weather caching bucket + TTL

Cache key (coordinate bucketing):

```text
cache_key = (round(lat, 2), round(lon, 2))
```

Time-to-live:

```text
TTL = 300 seconds
```

### 8.13 NLG advisory thresholds

```text
avg_risk < 20  → OPTIMAL / Minimal risk exposure
avg_risk < 40  → LOW RISK
avg_risk < 55  → MODERATE
avg_risk < 65  → CAUTION REQUIRED
else           → HIGH RISK
```

### 8.14 Quick code cross-reference

- Feature engineering: `app/ml/features.py`
- Inference + scaling: `app/ml/inference.py`
- Graph + guards: `app/routing/graph.py`
- Routing weights + K paths: `app/routing/astar.py`
- Route SHAP aggregation + NLG: `app/routers/route.py`
- Weather cache: `app/weather.py`

---

## 9. Machine Learning: Model Design and Training

### Task Definition

Binary classification:

```text
Label 1 = crash occurred on this segment in this time window
Label 0 = no crash
```

The model outputs `P(crash) ∈ [0.0, 1.0]`, which is scaled to a risk score:

```python
risk_score = P(crash) × 100.0  # float, e.g. 34.7
```

### Why Binary Classification (Not Regression)?

The training labels are binary — either a crash happened at a location and time, or it did not.

### Why XGBoost?

1. **Tabular data dominance** — on datasets with <100 features and <10M rows, gradient-boosted trees consistently outperform neural networks.
2. **CPU-only** — no GPU required.
3. **Native SHAP** — `shap.TreeExplainer` is exact for tree models.
4. **Built-in regularisation** — `reg_alpha` (L1) and `reg_lambda` (L2) prevent overfitting.
5. **Class imbalance handling** — `scale_pos_weight` adjusts for rare positives.

### Hyperparameter Tuning with Optuna

Objective: maximise AUPRC on validation (better for imbalanced datasets).

### The Fallback Model (`_FallbackRiskModel`)

When `models/model.pkl` does not exist, STRIVE uses a deterministic heuristic model and applies a sigmoid to produce probabilities.

---

## 10. Inference Layer: From Coordinates to Risk Score

The inference layer lives in `app/ml/inference.py`.

---

## 11. Explainability: SHAP Values and NLG Summaries

(See code + earlier equations for additivity and aggregation.)

---

## 12. Road Graph Layer: Graph Resolution and Caching

(See earlier chapter sections.)

---

## 13. Routing Engine: Yen's K-Shortest and Risk-Exposure Weighting

(See earlier chapter sections.)

---

## 14. API Layer: Every Endpoint Documented

(See earlier chapter sections.)

---

## 15. Database: Schema and Migrations

(See earlier chapter sections.)

---

## 16. Weather Integration

(See earlier chapter sections.)

---

## 17. Frontend: Dashboard, Map, and Components

(See earlier chapter sections.)

---

## 18. Deployment: Docker and Environment Configuration

(See earlier chapter sections.)

---

## 19. Testing Strategy

(See earlier chapter sections.)

---

## 20. Performance and Scalability

(See earlier chapter sections.)

---

## 21. Known Limitations and Future Work

(See earlier chapter sections.)

---

## 22. Glossary

(See earlier chapter sections.)
