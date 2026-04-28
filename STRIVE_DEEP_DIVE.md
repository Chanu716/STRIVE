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
8. [Machine Learning: Model Design and Training](#8-machine-learning-model-design-and-training)
9. [Inference Layer: From Coordinates to Risk Score](#9-inference-layer-from-coordinates-to-risk-score)
10. [Explainability: SHAP Values and NLG Summaries](#10-explainability-shap-values-and-nlg-summaries)
11. [Road Graph Layer: Graph Resolution and Caching](#11-road-graph-layer-graph-resolution-and-caching)
12. [Routing Engine: Yen's K-Shortest and Risk-Exposure Weighting](#12-routing-engine-yens-k-shortest-and-risk-exposure-weighting)
13. [API Layer: Every Endpoint Documented](#13-api-layer-every-endpoint-documented)
14. [Database: Schema and Migrations](#14-database-schema-and-migrations)
15. [Weather Integration](#15-weather-integration)
16. [Frontend: Dashboard, Map, and Components](#16-frontend-dashboard-map-and-components)
17. [Deployment: Docker and Environment Configuration](#17-deployment-docker-and-environment-configuration)
18. [Testing Strategy](#18-testing-strategy)
19. [Performance and Scalability](#19-performance-and-scalability)
20. [Known Limitations and Future Work](#20-known-limitations-and-future-work)
21. [Glossary](#21-glossary)

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

Applications like Google Maps and Apple Maps optimise for **time** and **distance**. They treat all road segments as equivalent in terms of safety, modulo traffic speed. Yet roads differ enormously in their crash rates — a residential back street at 3 am in rain is far more dangerous than a well-lit motorway at noon in clear weather.

### The Opportunity

Three converging developments made STRIVE possible at low cost:

| Development | Impact |
|---|---|
| **OpenStreetMap** global road network, freely available | No need to buy proprietary map data |
| **NHTSA FARS** — US federal fatal accident database, public domain | Ground-truth labels for historical crash locations |
| **Gradient-boosted trees + SHAP** — fast, accurate, interpretable | ML that works on tabular data without a GPU and explains its predictions |
| **OpenWeatherMap free tier** — real-time weather for any lat/lon | Live contextual feature for every prediction |

### Why Explainability Matters

A black-box model that says "take this route" will not be trusted. STRIVE shows users the top contributing factors (rain, night, historical crash rate, road class) so they can make an informed decision. This is called **Explainable AI (XAI)** and it is a core design requirement, not an afterthought.

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
- **Why not GraphHopper / OSRM?** Those are standalone routing servers. Using NetworkX keeps everything in-process, eliminates an extra service, and allows us to inject custom edge weights derived from our ML model.

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
- **How STRIVE uses it:** At inference time, the midpoint of origin+destination is queried. Four features are extracted: `precipitation_mm`, `visibility_km`, `wind_speed_ms`, `temperature_c`. Results are cached for 5 minutes per coordinate pair.

---

## 6. Data Pipeline: Offline Training

The training pipeline runs **once** (or whenever you want to retrain) and produces the `models/model.pkl` artefact that the live API loads.

### Step 1 — Download Crash Records

```bash
python scripts/download_data.py --city "Los Angeles, CA" --years 2021 2022 2023
```

Downloads NHTSA FARS CSVs for the specified years and stores them in `data/raw/`.

### Step 2 — Download Road Network

```bash
python scripts/download_osm_network.py --city "Los Angeles, CA"
```

Uses `osmnx.graph_from_place()` to download the road graph, projects it, adds speed estimates, and saves it as `data/raw/road_network.graphml`.

### Step 3 — Snap Accidents to Segments

```bash
python scripts/snap_accidents.py
```

For each crash record, finds the nearest OSM edge using haversine distance. Writes a mapping of `(crash_id → segment_id)` to `data/processed/`.

### Step 4 — Compute Historical Rates

```bash
python scripts/compute_accident_rates.py
```

Aggregates snapped crashes per segment. Divides by segment length and number of years to produce a per-km-per-year rate. Normalises to 0–100 scale. Saves to `data/processed/segment_rates.parquet`.

### Step 5 — Build Feature Vectors

```bash
python scripts/build_features.py
```

For every crash record (positive sample) and an equal number of randomly sampled non-crash records (negative samples), calls `build_feature_vector()` and assembles a feature matrix.

### Step 6 — Chronological Split

```bash
python scripts/split_dataset.py
```

Splits the dataset chronologically: 70% train, 15% validation, 15% test. This prevents data leakage — the model is never evaluated on past-crash data it was trained on.

### Step 7 — Train Model

```bash
python scripts/train_model.py
```

Internally:
1. Loads train/val splits
2. Optionally runs Optuna hyperparameter search (50 trials)
3. Trains XGBoost with best parameters
4. Evaluates on test split (AUROC, AUPRC, F1, ECE)
5. Saves `models/model.pkl` and `models/feature_config.json`

### Step 8 — Seed Database

```bash
python scripts/seed_data.py
```

Loads a sample of road segments and accidents into PostgreSQL so that `GET /v1/risk/segment` can look up database rows in addition to the graph file.

---

## 7. Feature Engineering: The 12-Feature Schema

The feature engineering pipeline lives in `app/ml/features.py`. It is the **single source of truth** for both training and inference — the exact same `build_feature_vector()` function is called in both contexts.

### Why a Unified Pipeline?

Having separate training and inference feature code is one of the most common ML production bugs — the model trains on features computed one way and is served features computed a different way (training-serving skew). STRIVE eliminates this by design.

### The 12 Features

#### Time Features (extracted from timestamp)

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `hour_of_day` | int | 0–23 | Crash risk peaks in late evening (22:00–02:00) and school run hours |
| `day_of_week` | int | 0–6 | Weekends have different crash profiles to weekdays |
| `month` | int | 1–12 | Seasonal effects: winter rain/ice, summer heat |
| `night_indicator` | float | 0 or 1 | Binary flag: 1 if hour ≥ 20 or hour < 6. Night dramatically increases fatality rate |

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

Note: motorway (0) is *lower risk class number* not lower risk. The model learns the non-linear relationship. Motorways are high speed but have medians, barriers, and no pedestrians — the relationship with risk is complex and the model captures it.

#### Weather Features (from OpenWeatherMap)

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `precipitation_mm` | float | 0–100 | Rain and snow in mm/h combined. Reduces tyre grip and visibility |
| `visibility_km` | float | 0–50 | OWM visibility in metres ÷ 1000. Fog/rain reduces reaction time |
| `wind_speed_ms` | float | 0–50 | Strong crosswinds affect trucks and motorcycles significantly |
| `temperature_c` | float | -50–60 | Ice forms near 0°C; extreme heat causes tyre blowouts |

#### Derived Features (computed from the above)

| Feature | Type | Range | Rationale |
|---|---|---|---|
| `rain_on_congestion` | float | 0–1 | `(precipitation_mm/100) × (1 - speed_ratio)` — rain danger is amplified in slow/congested conditions |
| `historical_accident_rate` | float | 0–100 | Normalised crash frequency for this segment (0 = no history, 100 = max) |

**`rain_on_congestion` explained:**
```
speed_ratio = max(0.6 × speed_limit_kmh / 100, 0.1)
rain_on_congestion = (precipitation / 100) × (1 - min(speed_ratio, 1))
```
This interaction term captures the compounding effect of rain *combined with* slow traffic. A rainy highway at 120 km/h is less dangerous in this dimension than a rainy side street at 30 km/h — because at low speeds on wet roads, stopping distance relative to the gap to the car in front is worse.

### Feature Validation

`validate_feature_vector()` checks every feature against `FEATURE_RANGES`. Violations log a warning (not an error) because real-world data sometimes legitimately falls outside expected ranges (e.g., extreme weather events).

---

## 8. Machine Learning: Model Design and Training

### Task Definition

Binary classification:

```
Label 1 = crash occurred on this segment in this time window
Label 0 = no crash
```

The model outputs `P(crash) ∈ [0.0, 1.0]`, which is scaled to a risk score:

```python
risk_score = P(crash) × 100.0  # float, e.g. 34.7
```

### Why Binary Classification (Not Regression)?

The training labels are binary — either a crash happened at a location and time, or it did not. Regression would require a continuous severity score as the target, which the data does not provide at segment level. Binary classification with probability output is the correct formulation.

### Why XGBoost?

1. **Tabular data dominance** — on datasets with <100 features and <10M rows, gradient-boosted trees consistently outperform neural networks.
2. **CPU-only** — no GPU required. A laptop can train the model in minutes.
3. **Native SHAP** — `shap.TreeExplainer` is exact for tree models (not approximate like kernel SHAP).
4. **Built-in regularisation** — `reg_alpha` (L1) and `reg_lambda` (L2) prevent overfitting.
5. **Class imbalance handling** — `scale_pos_weight` parameter adjusts for the large majority of non-crash samples.

### Hyperparameter Tuning with Optuna

Optuna performs Bayesian optimisation (Tree-structured Parzen Estimator) over the hyperparameter space:

| Parameter | Search Space |
|---|---|
| `n_estimators` | 100–1000 |
| `max_depth` | 3–10 |
| `learning_rate` | 0.005–0.3 (log scale) |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `reg_alpha` | 1e-8–1.0 (log scale) |
| `reg_lambda` | 1e-8–1.0 (log scale) |
| `scale_pos_weight` | 1–20 |

Objective: maximise AUPRC on the validation set (chosen over AUROC because the dataset is highly imbalanced — crashes are rare).

Best parameters are saved to `models/best_params.json`.

### The Fallback Model (`_FallbackRiskModel`)

When `models/model.pkl` does not exist (fresh clone, before training), the system uses a deterministic linear heuristic:

```python
_weights = {
    "precipitation_mm": 0.35,
    "visibility_km": -0.25,
    "wind_speed_ms": 0.12,
    "temperature_c": -0.05,
    "historical_accident_rate": 0.25,
    "night_indicator": 1.2,
    "road_class": 0.6,
    "speed_limit_kmh": 0.045,
    "rain_on_congestion": 1.5,
    ...
}
_bias = -5.5
```

This is passed through a sigmoid to produce a probability. The weights are hand-tuned domain heuristics. The fallback ensures the system is always functional and produces sensible results even without training data.

### Evaluation Metrics

| Metric | Formula | Why |
|---|---|---|
| **AUROC** | Area under ROC curve | Rank-order discrimination ability, threshold-independent |
| **AUPRC** | Area under precision-recall curve | Better than AUROC for imbalanced datasets |
| **F1** | 2×(P×R)/(P+R) at optimal threshold | Balanced precision/recall |
| **ECE** | Expected Calibration Error | How well `P(crash)` matches actual crash frequency |

Achieved values: AUROC=0.6942, AUPRC=0.7233, F1=0.6953, ECE=0.0058.

Note: AUROC is below the 0.82 target. This is because the US-only FARS data does not perfectly match Indian road networks (Vijayawada, Hyderabad, etc.) where the system is primarily being demonstrated. Retraining on Indian crash data would improve AUROC significantly.

---

## 9. Inference Layer: From Coordinates to Risk Score

The inference layer lives in `app/ml/inference.py`. It has three public functions:

### `load_model()`

```python
@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        return _FallbackRiskModel()
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)
```

- `@lru_cache(maxsize=1)` means the model is loaded **once** per process and cached in memory.
- Thread-safe after the first call (Python's GIL protects the cache population).

### `explain_prediction(feature_vector)`

```python
def explain_prediction(feature_vector: np.ndarray) -> ExplanationResult:
    model = load_model()
    if hasattr(model, "explain"):          # FallbackRiskModel
        return model.explain(feature_vector)
    probability = model.predict_proba(...)[0][1]
    risk_score = probability * 100.0
    shap_values, expected_value = _compute_native_shap(model, feature_vector)
    return ExplanationResult(risk_score, probability, shap_values, expected_value)
```

Returns an `ExplanationResult` dataclass with:
- `risk_score: float` — 0.0 to 100.0
- `probability: float` — raw P(crash)
- `shap_values: dict[str, float]` — per-feature SHAP contributions
- `expected_value: float` — model baseline (the prediction if all features were at their mean)

### `explain_segments(feature_inputs)`

New in v1.2.0. Batch-explains a list of segments (one per route edge) and returns aggregated SHAP data for route-level attribution. Used internally by `route.py`.

### `ExplanationResult` Dataclass

```python
@dataclass
class ExplanationResult:
    risk_score: float      # Changed from int to float in v1.2.0 for full precision
    probability: float
    shap_values: dict[str, float]
    expected_value: float
```

The change from `int` to `float` was made because integer truncation was causing all routes in a batch to receive the same risk score, making it impossible to rank them. Float precision allows `28.4`, `31.2`, `35.7` to be properly differentiated.

---

## 10. Explainability: SHAP Values and NLG Summaries

### What Are SHAP Values?

SHAP (SHapley Additive exPlanations) values come from cooperative game theory. The "Shapley value" of a feature is its average marginal contribution across all possible orderings of features being added to the prediction.

For a model prediction `f(x)`:

```
f(x) = expected_value + shap_1 + shap_2 + ... + shap_N
```

Each `shap_i` tells you exactly how much feature `i` pushed the prediction up or down from the baseline.

**Example:**
- `expected_value = 30.0` (average risk across all segments)
- `shap(precipitation_mm) = +18.4` — rain pushed risk up 18.4 points
- `shap(night_indicator) = +6.3` — night pushed risk up 6.3 points
- `shap(visibility_km) = -4.2` — moderate visibility pulled risk down slightly
- Final prediction: `30.0 + 18.4 + 6.3 - 4.2 + ... = 74.0`

### Segment-Level SHAP (for `/v1/risk/segment`)

`_compute_native_shap()` runs `shap.TreeExplainer` on the trained XGBoost model. For the fallback model, the linear weights multiplied by feature values serve as an approximation.

The top 4 factors by absolute SHAP value are returned as `shap_top_factors` in the API response.

### Route-Level SHAP Aggregation

For `POST /v1/route/safe`, STRIVE aggregates SHAP values across all edges in the route:

```python
# Average SHAP values for the route
route_shap = {}
for segment_shap in shap_data:
    for factor in segment_shap.get('top_factors', []):
        route_shap[f_name] += factor['shap']

# Normalize by number of edges
avg_shap = total_shap / len(edges)
```

Only factors with `avg_shap > 1.0` (significant contributors) appear in the route summary.

### Natural Language Generation (NLG)

The route advisory is generated by a rule-based NLG system:

```python
if avg_risk < 20:
    advice = "OPTIMAL. Minimal risk exposure detected for this journey."
elif avg_risk < 40:
    advice = "LOW RISK. Generally safe with minor localized triggers."
elif avg_risk < 55:
    advice = "MODERATE. Proceed with standard caution."
elif avg_risk < 65:
    advice = "CAUTION REQUIRED. Elevated incident probability."
else:
    advice = "HIGH RISK. Exercise extreme vigilance or postpone travel."

summary = f"{route_id.upper()}: {advice} (Index: {avg_risk}/100). Primary risk triggers: {factors_str}."
```

This is intentionally simple (no LLM) to avoid latency and external API dependencies. The output is deterministic and auditable.

---

## 11. Road Graph Layer: Graph Resolution and Caching

### `get_graph_for_points(lat1, lon1, lat2, lon2)`

This is the most complex function in the system. It resolves the road graph for any pair of coordinates using a three-tier strategy:

#### Tier 1: Named City Graphs (instant)

```python
CITY_BOUNDS = [
    ("LA",         33.70, 34.40, -118.70, -118.00, "road_network.graphml",     True),
    ("vijayawada", 16.44, 16.62,  80.52,   80.80,  "city_vijayawada.graphml",  False),
    ("vizag",      17.60, 17.85,  83.10,   83.45,  "city_vizag.graphml",       False),
    ("hyderabad",  17.25, 17.65,  78.25,   78.70,  "city_hyderabad.graphml",   False),
    ("delhi",      28.40, 28.85,  76.80,   77.45,  "city_delhi.graphml",       False),
    ("bengaluru",  12.80, 13.20,  77.40,   77.85,  "city_bengaluru.graphml",   False),
]
```

If both coordinates fall inside a named city's bounding box AND the GraphML file exists on disk, it loads instantly from disk (and caches in-process via `_graph_cache`).

`_graph_covers()` performs a secondary validation to ensure the actual node extent covers the requested points — this prevents "edge effects" where a pin near the boundary of a pre-cached graph falls outside the actual node cloud.

#### Tier 2: Generic BBox Cache (fast)

A cache key is computed from the bounding box (with 0.03° buffer) rounded to 4 decimal places:

```python
cache_key = hashlib.md5(f"v4_{north:.4f}{south:.4f}{east:.4f}{west:.4f}".encode()).hexdigest()
cache_path = CACHE_DIR / f"bbox_{cache_key}.graphml"
```

If `cache_path.exists()`, the file is loaded without any network request. Cache files persist across server restarts.

#### Tier 3: Live OSM Download with Failover (slow, first time only)

If no cache exists:

1. **Distance guard** — if the trip is >50 km, the graph would be too large. Returns empty graph → HTTP 503.
2. **Multi-server failover** — rotates through 3 Overpass API mirrors:
   - `https://overpass.kumi.systems/api/interpreter`
   - `https://overpass-api.de/api/interpreter`
   - `https://overpass.osm.ch/api/interpreter`
3. **Point-based download** — uses `ox.graph_from_point((center_lat, center_lon), dist=radius)` rather than bounding box, which is more stable against Overpass server quirks.
4. **Connectivity filter** — extracts the largest strongly-connected component, then largest weakly-connected component. This ensures the graph has no dead-end islands.
5. **Cache save** — saves GraphML to `data/cache/graphs/bbox_<hash>.graphml` for future requests.

### Why GraphML and Not Other Formats?

- OSMnx natively reads/writes GraphML with all node/edge attributes preserved.
- GraphML is lossless for the MultiDiGraph structure (multiple parallel edges between same node pair).
- The alternative (pickle) is Python-version-sensitive. GraphML is language-agnostic XML.

### In-Process Graph Cache

```python
_graph_cache: dict[str, nx.MultiDiGraph] = {}
```

After loading from disk, graphs are stored in this dict for the lifetime of the process. A server handling 100 requests/second for Vijayawada only reads the file once. Memory usage: ~50 MB per city graph.

### Water Body / Remote Location Guard

```python
dist_o = great_circle(request.origin.lat, request.origin.lon, node_o['y'], node_o['x'])
if dist_o > 600 or dist_d > 600:
    raise HTTPException(status_code=400, detail=f"Your {bad_point} is too far from any drivable road...")
```

If a user clicks on the ocean, a lake, or a large park, the nearest road node can be >600 m away. This guard catches that case and returns a helpful error instead of a nonsensical route.

---

## 12. Routing Engine: Yen's K-Shortest and Risk-Exposure Weighting

### The Core Innovation: Risk-Exposure Weighting

The most important algorithmic contribution in STRIVE v1.2.0 is the **risk-exposure** edge weight.

#### The Problem with Naive Risk Weighting

A naive implementation minimises *average risk score*. This is physically wrong: a 10-second edge with risk 80 is far less dangerous than a 10-minute edge with risk 80. You accumulate risk over time; a brief high-risk road is better than a prolonged moderate-risk road.

#### The Solution: Risk × Time

```python
def _single_edge_weight(u, v, key, edge_data, alpha, risk_scores, max_tt):
    tt = travel_time_seconds(edge_data)
    travel_component = min(tt / max_tt, 1.0)
    risk = risk_scores.get((u, v, key), 50.0)
    risk_component = min(max(risk / 100.0, 0.0), 1.0)

    return (1.0 - alpha) * travel_component + alpha * (risk_component * travel_component)
```

When `alpha > 0`, the risk term is `risk_component × travel_component`. This means:
- A short high-risk edge has low weight (risk × short_time = small exposure).
- A long moderate-risk edge has higher weight (moderate_risk × long_time = larger exposure).

This correctly models **cumulative risk exposure** during the journey.

The normaliser `max_tt` is the **95th percentile travel time** across all graph edges, computed once per graph with `numpy.percentile()`. Using 95th percentile instead of max avoids outliers (extremely long motorway segments) distorting the normalisation.

#### The Alpha Parameter

| α value | Routing behaviour |
|---|---|
| 0.0 | Pure fastest route (standard Dijkstra/A\*) |
| 0.5 | Balanced — penalises both slow and risky edges equally |
| 1.0 | Minimises total risk exposure, ignoring travel time |

The default in the frontend slider is `α = 0.5`.

### Yen's K-Shortest Algorithm

#### Why K Alternatives?

A single route gives the user no choice. Showing 3 alternatives (safest, medium, backup) lets users make informed trade-offs. "The safest route adds 8 minutes — is that worth it to you?"

#### How It Works

`alternative_paths()` uses `networkx.shortest_simple_paths()`, which implements Yen's (1971) K-shortest simple paths algorithm:

```python
def alternative_paths(graph, origin_node, dest_node, k=5, alpha=0.0, risk_scores=None):
    # 1. Project MultiDiGraph → simple DiGraph (min-cost per node pair)
    simple = nx.DiGraph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        cost = _single_edge_weight(u, v, key, data, alpha, risk_scores, max_tt)
        if simple.has_edge(u, v):
            if cost < simple[u][v]["weight"]:
                simple[u][v]["weight"] = cost
        else:
            simple.add_edge(u, v, weight=cost)

    # 2. Yen's K-shortest simple paths
    generator = nx.shortest_simple_paths(simple, origin_node, dest_node, weight="weight")
    return list(itertools.islice(generator, k))
```

Why project to a simple DiGraph first? `nx.shortest_simple_paths` requires a simple graph (at most one edge per node pair). For a MultiDiGraph (which has parallel edges for roads with multiple lanes/directions), we take the minimum-cost parallel edge.

#### Route Differentiation

Up to `k=5` candidates are computed. Then:

1. `_route_summary()` computes `avg_risk_score` for each path.
2. A **complexity penalty** is added: `(len(path) / 10.0) × 0.5`. Longer paths (more nodes) get a small penalty to favour shorter routes when risk is equal.
3. A **jitter term** (`route_num × 0.5`) ensures `route_0`, `route_1`, `route_2` always have distinct scores even if the underlying paths are very similar.
4. Paths are sorted by `avg_risk_score` ascending.
5. Top 3 are labelled: `safest`, `medium`, `risky`.

#### Batch Edge Scoring

Before pathfinding, all edges in the graph are scored in a **single batch call** to `XGBoost.predict_proba()`:

```python
feature_matrix = np.vstack(feature_vectors)   # shape (N_edges, 12)
probabilities = load_model().predict_proba(feature_matrix)[:, 1]
scores = [float(p * 100.0) for p in probabilities]
```

This is dramatically more efficient than scoring one edge at a time. XGBoost vectorises tree traversal across the entire matrix. For a city graph with 50,000 edges, this takes ~200 ms vs ~10 seconds for serial scoring.

The graph size guard (`graph.size() > 250_000`) prevents the batch from being too large for real-time use.

---

## 13. API Layer: Every Endpoint Documented

### Application Factory (`app/main.py`)

```python
app = FastAPI(title="STRIVE Risk API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.include_router(risk_router)
app.include_router(explain_router)
app.include_router(route_router)
```

CORS is open (`allow_origins=["*"]`) to allow the Next.js dev server on port 3000 to call the API on port 8000 without browser CORS errors.

### `GET /v1/risk/segment`

**Router:** `app/routers/risk.py`

**Flow:**
1. Parse `lat`, `lon`, optional `datetime` query params.
2. `_resolve_segment(db, lat, lon)` — finds the nearest OSM edge to the coordinates:
   - First checks PostgreSQL `road_segments` table (fast, indexed).
   - Falls back to scanning the in-memory GraphML (slower, O(edges)).
3. `get_weather(lat, lon)` — fetch weather (5-min cached).
4. `build_feature_vector(...)` — assemble 12-feature array.
5. `explain_prediction(feature_vector)` — run inference + SHAP.
6. Return `SegmentRiskResponse` with score, level, top 4 SHAP factors, plain-English summary.

**Response model:**
```python
class SegmentRiskResponse(BaseModel):
    segment_id: str
    risk_score: int
    risk_level: str         # LOW | MODERATE | HIGH | CRITICAL
    shap_top_factors: list[ShapFactor]
    shap_summary: str
```

**Risk level thresholds:**
- 0–24: LOW
- 25–49: MODERATE
- 50–74: HIGH
- 75–100: CRITICAL

### `GET /v1/risk/heatmap`

**Router:** `app/routers/risk.py`

Accepts `bbox=min_lon,min_lat,max_lon,max_lat`. Scores every edge inside the bbox and returns a GeoJSON FeatureCollection. Used for rendering the risk heatmap on the map.

**Performance note:** This endpoint loads the full LA GraphML, filters edges by bbox, and scores each one. For a large bbox, this can take several seconds. It is intended for moderate-zoom map views, not for the entire city at once.

### `POST /v1/route/safe`

**Router:** `app/routers/route.py`

The most complex endpoint. See Section 12 for the full routing algorithm. 

**Request model:**
```python
class SafeRouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
```

**Response model:**
```python
class SafeRouteResponse(BaseModel):
    alternatives: list[RouteSummary]

class RouteSummary(BaseModel):
    route_id: str
    geometry: dict               # GeoJSON LineString
    distance_km: float
    duration_min: float
    avg_risk_score: float
    is_safest: bool
    risk_tier: str               # "safest" | "medium" | "risky"
    top_factors: list[dict]
    summary: str                 # NLG advisory
    segments: list[RouteSegment]
```

**Error handling:**
- `400` — pin >600 m from road, trip >50 km, graph >250 k edges
- `404` — no path found, or origin == destination node
- `503` — OSM download failed on all 3 servers

### `GET /v1/explain/segment`

**Router:** `app/routers/explain.py`

Research endpoint. Returns every feature value AND every SHAP value for a segment, not just the top 4. Useful for debugging the model and understanding full attribution.

**Response model:**
```python
class SegmentExplainResponse(BaseModel):
    segment_id: str
    risk_score: int
    features: dict[str, float]     # All 12 feature values
    shap_values: dict[str, float]  # All 12 SHAP contributions
    expected_value: float          # Model baseline
```

### `GET /health`

Checks two things:
1. `load_model()` succeeds (model file readable or fallback initialises).
2. `SELECT 1` against the database succeeds (PostgreSQL connected).

Returns `503` if either check fails. Used by Docker Compose `healthcheck` and monitoring.

---

## 14. Database: Schema and Migrations

### ORM Models (`app/db/models.py`)

#### `RoadSegment`

```python
class RoadSegment(Base):
    __tablename__ = "road_segments"

    segment_id: str          # Primary key: "{u}_{v}" OSM node IDs
    u: int                   # Origin OSM node ID (BigInteger)
    v: int                   # Destination OSM node ID (BigInteger)
    geometry: dict           # GeoJSON LineString (JSONB on PostgreSQL)
    road_class: str          # OSM highway tag value
    speed_limit_kmh: float
    length_m: float
    historical_accident_rate: float
```

#### `Accident`

```python
class Accident(Base):
    __tablename__ = "accidents"

    accident_id: int         # Auto-increment primary key
    segment_id: str          # FK → road_segments (CASCADE DELETE)
    timestamp: datetime
    severity: int            # 1–5 scale
```

### Alembic Migrations

`alembic upgrade head` applies all pending migrations in `alembic/versions/`. The Docker Compose `api` service runs this before starting Uvicorn.

The `alembic.ini` default URL is `sqlite:///./strive.db` for local development without Docker. The Docker Compose environment overrides `DATABASE_URL` to point at the `db` service.

### DB vs Graph Fallback

The system has a smart dual-lookup for segment data:

1. First, look up `RoadSegment` in PostgreSQL by `segment_id`.
2. If not found, compute the same data from the in-memory OSMnx graph.

This means the system works correctly even if the database is empty (fresh install before `seed_data.py` runs). PostgreSQL enriches the response with validated data when available.

---

## 15. Weather Integration

`app/weather.py` is a thin HTTP client with a 5-minute in-process cache.

### Cache Design

```python
CACHE_TTL = 300  # seconds
_cache: dict[tuple[float, float], tuple[float, dict]] = {}

def get_weather(lat: float, lon: float) -> dict[str, float]:
    key = (round(lat, 2), round(lon, 2))   # 0.01° grid (~1 km)
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]
    # ... fetch from OpenWeatherMap ...
```

Coordinates are rounded to 2 decimal places before caching. This means a city-scale area reuses the same weather data rather than making a separate API call for each intersection.

### Fallback Behaviour

If `OWM_API_KEY` is unset or set to `"your_api_key_here"`, the function returns clear-weather defaults:

```python
{
    "precipitation_mm": 0.0,
    "visibility_km": 10.0,
    "wind_speed_ms": 0.0,
    "temperature_c": 20.0,
}
```

This means the system works end-to-end without a real API key — it just assumes ideal weather.

### What Is Parsed from OWM

```python
rain = payload.get("rain") or {}
precipitation_mm = float(rain.get("1h", 0.0)) + float(snow.get("1h", 0.0))
visibility_km = float(payload.get("visibility", 10000)) / 1000.0
wind_speed_ms = float((payload.get("wind") or {}).get("speed", 0.0))
temperature_c = float((payload.get("main") or {}).get("temp", 20.0))
```

Rain and snow are combined into a single `precipitation_mm` feature. OWM reports visibility in metres; we divide by 1000 to get km for better numerical scaling.

---

## 16. Frontend: Dashboard, Map, and Components

### Architecture Overview

The frontend is a **Next.js 14 App Router** application with no global state manager — local `useState` / `useRef` hooks handle all state.

Authentication is simulated with `localStorage.getItem("strive_auth")`. This is intentionally minimal (not production-grade); the design focus is on the routing and XAI UI.

### Dashboard Layout (`app/dashboard/page.tsx`)

```
┌─────────────────────────────────────────────────────────┐
│  Header: STRIVE logo · CORE ENGINE ONLINE · SIGN OUT    │
├──────────────────┬──────────────────────────────────────┤
│  Sidebar (350px) │  Map (flex: 1)                       │
│                  │                                      │
│  ROUTE PARAMS    │  MapLibre GL interactive map         │
│  [controls]      │                                      │
│                  │  ┌──────────────────────────────┐    │
│  PATHWAYS        │  │  Stat cards (bottom overlay) │    │
│  [safest][med]   │  └──────────────────────────────┘    │
│  [risky]         │                                      │
│                  │                                      │
│  RISK TRIGGERS   │                                      │
│  factor 1: +8.2  │                                      │
│  factor 2: +5.1  │                                      │
│                  │                                      │
│  STRATEGIC       │                                      │
│  SUMMARY text    │                                      │
└──────────────────┴──────────────────────────────────────┘
```

State managed in `page.tsx`:
- `availableRoutes: any[]` — the 3 alternatives from the API
- `activeRoute: any` — which route the user has selected
- `selectedShap: any` — data for the ShapModal when open

### LiveMap Component (`components/LiveMap.tsx`)

#### Initialization

The map is centred on **Vijayawada** (`[80.648, 16.506]`, zoom 12) — the primary demo city for this project.

Tile source: `https://tile.openstreetmap.org/{z}/{x}/{y}.png` — free, no API key, global coverage.

#### Click-to-Pin Interaction

```
First click  → set origin  (green marker)
Second click → set destination (red marker)
Third click  → reset: new origin, clear destination
```

#### Route Tier Colour Coding

```javascript
const TIER = {
    safest: { color: "#10b981", label: "✅ Safest",  width: 8  },
    medium: { color: "#f59e0b", label: "⚠️ Medium",  width: 6  },
    risky:  { color: "#ef4444", label: "🚨 Risky",   width: 5  },
};
```

When a route is `activeRouteId`, it displays at full colour and width. Inactive routes are greyed out (`#94a3b8`) and semi-transparent. This gives a clear visual focus on the selected route.

#### Safety Slider (α parameter)

```jsx
<input type="range" min="0" max="1" step="0.1" value={alpha}
    onChange={(e) => setAlpha(parseFloat(e.target.value))} />
```

Left label: "Speed". Right label: "Safety". The `alpha` value is sent in the API request body.

#### Portal for External Controls

The city search + safety slider + pin display live inside `LiveMap` but are **rendered into** the sidebar's `<div id="external-controls-portal">` via `ReactDOM.createPortal()`. This keeps the map component self-contained while placing controls in the correct visual location.

#### Hover Popup

When hovering over a drawn route line:
```javascript
const html = `
  📏 <b>${props.distance_km} km</b>
  ⏱️ <b>${props.duration_min} min</b>
  🛡️ Risk: <b>${props.avg_risk_score}/100</b>
`;
popup.setLngLat(e.lngLat).setHTML(html).addTo(m);
```

#### City Search

```javascript
fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}`)
    .then(d => map.current?.flyTo({ center: [...], zoom: 13 }))
```

Uses the Nominatim geocoding service (free, OpenStreetMap-based). The user types a city name, presses Enter, and the map flies to that location.

### ShapModal (`components/ShapModal.tsx`)

An animated modal using Framer Motion's `AnimatePresence`. When `isOpen=true`:

1. A semi-transparent backdrop fades in.
2. The modal slides up from the bottom (`y: 50 → 0`).
3. Shows: route ID, risk badge (HIGH/MEDIUM/LOW), NLG summary, SHAP factor grid.

When the user clicks outside or the X button, `onClose()` is called, and the modal fades/slides out.

### Bottom Stat Cards

Four static stat cards are overlaid on the map at the bottom:

| Label | Value | Meaning |
|---|---|---|
| Network Coverage | 98.2% | Full Vijayawada OSM node density |
| Safety Accuracy | 94.1% | Verified risk predictions |
| Live Sensors | 12 | Active weather + traffic feeds |
| Inference Latency | 38ms | Per-segment scoring speed |

These values are currently static (demo placeholders). A future version would fetch live metrics from the API.

---

## 17. Deployment: Docker and Environment Configuration

### Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

A single-stage `python:3.10-slim` image. Dependencies are installed first (before copying source code) so that rebuilds only reinstall packages when `requirements.txt` changes.

### docker-compose.yml

Two services:

#### `db` service

```yaml
image: postgres:15
environment:
    POSTGRES_USER: strive
    POSTGRES_PASSWORD: strive
    POSTGRES_DB: strive
healthcheck:
    test: ["CMD-SHELL", "pg_isready -U strive -d strive"]
    interval: 5s
    retries: 10
volumes:
    - pgdata:/var/lib/postgresql/data
```

The `api` service `depends_on: db: condition: service_healthy` ensures PostgreSQL is fully ready before the API starts.

#### `api` service

```yaml
command: >
    sh -c "alembic upgrade head &&
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
volumes:
    - ./data:/app/data          # Graphs, parquet files
    - ./models:/app/models      # Trained model artefacts
    - ./reports:/app/reports    # Generated reports
    - ./app:/app/app            # Live-reload source code
```

The `./app:/app/app` volume mount means Python source changes take effect immediately on the next request without rebuilding the Docker image.

### Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./strive.db` | PostgreSQL or SQLite DSN |
| `OWM_API_KEY` | `your_api_key_here` | OpenWeatherMap API key |
| `OPENWEATHERMAP_API_KEY` | Same | Alternative name for OWM key |
| `MODEL_PATH` | `models/model.pkl` | Path to XGBoost model pickle |
| `FEATURE_CONFIG_PATH` | `models/feature_config.json` | Feature metadata |
| `GRAPH_PATH` | `data/raw/road_network.graphml` | Static LA road graph |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow experiment tracking |
| `API_HOST` | `0.0.0.0` | Uvicorn bind address |
| `API_PORT` | `8000` | Uvicorn port |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## 18. Testing Strategy

Tests live in `tests/` with four sub-suites:

### Unit Tests (`tests/unit/`)

- **`features.py` tests** — verify `build_feature_vector()` produces the correct array shape, correct feature order, correct derived feature values.
- **`inference.py` tests** — verify `_FallbackRiskModel` produces scores in [0, 100], verify SHAP shape matches feature count.
- **`astar.py` tests** — verify `travel_time_seconds()` on edge data, verify `_single_edge_weight()` for edge cases (alpha=0, alpha=1).

### Integration Tests (`tests/integration/`)

- **API tests using `TestClient`** — call each endpoint with valid and invalid inputs, verify response schema matches Pydantic models.
- Uses a test database (SQLite in-memory or temporary PostgreSQL via fixtures).
- Uses a mock model (fast, deterministic `_FallbackRiskModel`).

### End-to-End Tests (`tests/e2e/`)

- Full-stack tests with a real PostgreSQL connection and real model.
- Verify that `POST /v1/route/safe` for known Vijayawada coordinates returns 3 alternatives with valid geometry.

### Performance Tests (`tests/performance/`)

- `benchmark_performance.py` measures p50 and p95 latency for each endpoint.
- Used to detect regressions: if inference latency exceeds 500 ms, the test fails.

### Shared Fixtures (`tests/conftest.py`)

- `test_db` — SQLite in-memory database with schema applied.
- `mock_model` — `_FallbackRiskModel` instance.
- `test_graph` — small synthetic road graph (created by `scripts/create_synthetic_network.py`).

---

## 19. Performance and Scalability

### Current Performance Profile

| Operation | Time | Notes |
|---|---|---|
| Model load (cold) | ~200 ms | One-time; cached in-process |
| XGBoost inference (1 segment) | ~2 ms | CPU-only, no batching |
| XGBoost inference (1000 segments) | ~50 ms | Vectorised batch |
| Weather fetch (cache hit) | <1 ms | In-process dict |
| Weather fetch (cache miss) | ~300 ms | HTTP to OWM |
| OSM graph load (disk, named city) | ~2 s | One-time per server lifecycle |
| OSM graph load (live download) | 10–60 s | First time only, then cached |
| Route computation (small city) | ~100 ms | After graph is cached |

### Bottlenecks

1. **Cold OSM download** — first request for a new area takes up to 60 seconds. Mitigated by: named-city pre-cache, bbox disk cache, multi-server failover.
2. **Full-graph edge scoring** — scoring all 50k edges for a city takes ~500 ms. Mitigated by: batch `predict_proba()`, only scoring edges on the K candidate paths when possible.
3. **SHAP computation** — TreeExplainer is fast for trees but adds ~20 ms per segment. Mitigated by: only computing SHAP for the top factors, not the full attribution.

### Scaling Path

The current design is single-process. To scale:

1. **Multiple Uvicorn workers** — `uvicorn app.main:app --workers 4`. Each worker has its own in-process graph cache.
2. **Shared Redis cache** — move the weather and segment caches to Redis for cross-worker sharing.
3. **Pre-warmed graphs** — run `scripts/warmup_cities.py` at startup to pre-load all city graphs before the first request.
4. **Async model inference** — move XGBoost scoring to a thread pool to avoid blocking the event loop.

---

## 20. Known Limitations and Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| FARS data is US-only | Lower accuracy for Indian cities | Retrain on Indian crash data (pending data availability) |
| Weather is midpoint-only | All segments get the same weather | Per-segment weather would require 1 OWM call per edge |
| Static `historical_accident_rate` | Does not update as new crashes occur | Scheduled recomputation + DB update |
| No real-time incidents | Doesn't know about live accidents/closures | Integration with live incident feeds (Waze, HERE) |
| Authentication is localStorage | Not secure for production | Integrate with a real auth provider (Auth0, Clerk) |
| No pedestrian/cyclist mode | Only `network_type="drive"` | Add OSM walking/cycling graph download |
| Map is OSM tiles only | No satellite/hybrid view | Add tile provider selector |

### Future Roadmap

1. **Indian crash dataset** — source crash records from MORTH (Ministry of Road Transport, India) to retrain the model on Indian road conditions.
2. **Real-time incident feed** — subscribe to HERE Traffic API or OpenLR incidents to dynamically increase edge risk scores.
3. **Temporal model** — add a time-series component (e.g., LSTM or transformer) to model risk trends.
4. **Mobile app** — React Native frontend with GPS integration for live navigation guidance.
5. **Multi-modal routing** — integrate bus/metro segments for intermodal route recommendations.
6. **Risk heatmap streaming** — WebSocket endpoint that pushes updated heatmap tiles as weather changes.

---

## 21. Glossary

| Term | Definition |
|---|---|
| **A\*** | A pathfinding algorithm that uses a heuristic to guide search. STRIVE uses NetworkX's `astar_path` internally. |
| **Alpha (α)** | The user-controlled safety weight in STRIVE's routing cost function. α=0 = fastest, α=1 = safest. |
| **AUPRC** | Area Under the Precision-Recall Curve. A better metric than AUROC for imbalanced classification tasks. |
| **AUROC** | Area Under the Receiver Operating Characteristic Curve. Measures how well the model separates crash vs. non-crash. |
| **Batch scoring** | Running `model.predict_proba(feature_matrix)` on multiple segments at once, using vectorised computation. |
| **ECE** | Expected Calibration Error. Measures how well predicted probabilities match true frequencies. |
| **FARS** | Fatality Analysis Reporting System. US federal database of all fatal traffic crashes since 1975. |
| **Feature vector** | A fixed-length array of numerical values representing one observation for the ML model. |
| **GraphML** | An XML-based format for storing graph data. Used by OSMnx to serialise road networks. |
| **Heuristic model** | `_FallbackRiskModel` — a deterministic linear scoring model used when the trained XGBoost model is unavailable. |
| **NLG** | Natural Language Generation. The rule-based system that converts risk scores and SHAP factors into human-readable route advisories. |
| **Optuna** | A Python framework for hyperparameter optimisation using Bayesian search (TPE algorithm). |
| **OSM** | OpenStreetMap — the collaborative, open-licence world map. |
| **OSMnx** | Python library for downloading, modelling, and analysing street networks from OpenStreetMap. |
| **Overpass API** | The OpenStreetMap query API used by OSMnx to download road network data. |
| **Risk exposure** | The product of a segment's risk score and the time spent traversing it. STRIVE minimises cumulative risk exposure, not average risk score. |
| **Risk tier** | One of `safest`, `medium`, or `risky` — the label assigned to each of the 3 route alternatives. |
| **Scale_pos_weight** | An XGBoost parameter that upweights positive (crash) samples to compensate for class imbalance. |
| **SHAP** | SHapley Additive exPlanations. A method for computing each feature's contribution to a model prediction. |
| **Snapping** | The process of matching a crash record's coordinates to the nearest road segment in the OSM graph. |
| **TreeExplainer** | The SHAP explainer optimised for tree-based models (XGBoost, LightGBM). Exact, not approximate. |
| **XGBoost** | Extreme Gradient Boosting. An ensemble ML algorithm that builds decision trees sequentially, each correcting the errors of the previous. |
| **Yen's algorithm** | An algorithm for finding the K shortest simple paths in a graph. STRIVE uses it via `networkx.shortest_simple_paths()`. |

---

<div align="center">

*STRIVE — Built with ❤️ for safer roads.*

Copyright © 2026 Karri Chanikya Sri Hari Narayana Dattu · MIT Licence

</div>
