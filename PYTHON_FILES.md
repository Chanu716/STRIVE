# STRIVE — Python Files Reference

A complete catalogue of every Python source file in the project, describing what each file does and showing representative inputs and outputs.

---

## Table of Contents

1. [Application Core](#application-core)
2. [ML Layer](#ml-layer)
3. [Routing Layer](#routing-layer)
4. [API Routers](#api-routers)
5. [Database Layer](#database-layer)
6. [Top-Level Pipeline Scripts](#top-level-pipeline-scripts)
7. [scripts/ — Data & Training Pipeline](#scripts----data--training-pipeline)
8. [alembic/ — Database Migrations](#alembic----database-migrations)
9. [tests/ — Test Suite](#tests----test-suite)

---

## Application Core

### `app/main.py`

**What it does:** FastAPI application entry point. Creates the FastAPI app, registers CORS middleware, mounts the three API routers (`/v1/risk`, `/v1/explain`, `/v1/route`), and exposes a `/health` endpoint that checks whether the ML model loaded and the database is reachable.

**Input:** None at import time. The `/health` endpoint takes no parameters.

**Output example (`GET /health`):**
```json
{ "status": "ok", "model_loaded": true, "db_connected": true }
```
If either check fails, it returns HTTP 503:
```json
{
  "detail": {
    "status": "unavailable",
    "model_loaded": false,
    "db_connected": true
  }
}
```

---

### `app/weather.py`

**What it does:** OpenWeatherMap client with a 5-minute in-process cache. Calls the OWM current-weather API for a lat/lon pair and normalises the response into four model features. Falls back to safe defaults (`0 mm rain, 10 km visibility, 0 m/s wind, 20 °C`) when no API key is configured.

**Input:**
```python
get_weather(lat=34.05, lon=-118.24)
```

**Output:**
```python
{
    "precipitation_mm": 0.0,
    "visibility_km": 10.0,
    "wind_speed_ms": 0.0,
    "temperature_c": 20.0
}
```

---

## ML Layer

### `app/ml/features.py`

**What it does:** Core feature engineering module shared by training (batch) and inference (real-time). Transforms raw inputs — timestamp, road attributes, weather, historical accident rate — into a fixed 12-element NumPy feature vector consumed by XGBoost.

Key exports: `FEATURE_NAMES` (list of 12 feature names), `build_feature_vector()`, `build_feature_dataframe()`, `create_training_dataset()`.

**Input:**
```python
raw = {
    "timestamp": datetime(2023, 6, 15, 14, 30),
    "highway": "secondary",
    "speed_limit_kmh": 50.0,
    "precipitation_mm": 5.0,
    "visibility_km": 8.0,
    "wind_speed_ms": 3.0,
    "temperature_c": 25.0,
    "historical_accident_rate": 0.5,
}
build_feature_vector(raw)
```

**Output:**
```
Feature vector (shape (12,)):
  hour_of_day              = 14.0000
  day_of_week              =  3.0000   # Thursday
  month                    =  6.0000
  night_indicator          =  0.0000
  road_class               =  2.0000   # secondary
  speed_limit_kmh          = 50.0000
  precipitation_mm         =  5.0000
  visibility_km            =  8.0000
  wind_speed_ms            =  3.0000
  temperature_c            = 25.0000
  rain_on_congestion       =  0.0280
  historical_accident_rate =  5.0000
```

---

### `app/ml/inference.py`

**What it does:** Model loading, inference, and SHAP-based explainability. Loads a trained XGBoost model from `models/model.pkl` (cached with `lru_cache`). Falls back to `_FallbackRiskModel` — a hand-tuned logistic heuristic — when no trained model exists. Exposes `run_inference()`, `explain_prediction()`, and `explain_segments()`.

**Input:**
```python
feature_vector = np.array([14, 3, 6, 0, 2, 50, 5, 8, 3, 25, 0.028, 5])
result = explain_prediction(feature_vector)
```

**Output (`ExplanationResult`):**
```python
ExplanationResult(
    risk_score=42.7,          # 0-100 scale
    probability=0.427,
    shap_values={
        "precipitation_mm": 1.75,
        "historical_accident_rate": 1.25,
        "night_indicator": 0.0,
        ...
    },
    expected_value=-5.5
)
```

---

## Routing Layer

### `app/routing/astar.py`

**What it does:** Safety-aware A* routing utilities. Provides `safe_route()` which runs `networkx.astar_path` with a composite edge weight combining normalised travel time and risk score, controlled by an `alpha` parameter (0 = fastest, 1 = safest). Also provides `alternative_paths()` which uses Yen's K-shortest-paths algorithm to return up to K physically distinct routes.

**Input:**
```python
safe_route(
    graph=G,              # nx.MultiDiGraph
    origin_node=123,
    dest_node=456,
    alpha=0.6,            # 60% weight on risk
    risk_scores={(u, v, key): score, ...}
)
```

**Output:**
```python
[123, 789, 234, 456]   # ordered list of OSM node IDs
```

---

### `app/routing/graph.py`

**What it does:** Road graph loading and lookup. Maintains a list of pre-downloaded named city graphs (LA, Vijayawada, Vizag, Hyderabad, Delhi, Bengaluru). For a given pair of coordinates it first checks named city bounds, then a file-based bbox cache, then downloads from OSM via Overpass with multi-server failover. Exposes `get_graph_for_points()` and `nearest_node()`.

**Input:**
```python
G = get_graph_for_points(lat1=16.50, lon1=80.62, lat2=16.52, lon2=80.65)
node = nearest_node(G, lat=16.51, lon=80.63)
```

**Output:**
```
Using named city graph: vijayawada
G  →  nx.MultiDiGraph with ~12,000 nodes
node  →  1234567890  (OSM node ID)
```

---

## API Routers

### `app/routers/risk.py`

**What it does:** Two risk-scoring REST endpoints:
- `GET /v1/risk/segment` — resolves the nearest road segment for a lat/lon, fetches weather, builds a feature vector, runs inference, and returns the risk score with top-4 SHAP factors.
- `GET /v1/risk/heatmap` — scores all segments inside a bounding box and returns a GeoJSON `FeatureCollection`.

**Input (`/v1/risk/segment`):**
```
GET /v1/risk/segment?lat=34.05&lon=-118.24&datetime=2024-06-15T22:00:00
```

**Output:**
```json
{
  "segment_id": "110123_110456",
  "risk_score": 68,
  "risk_level": "HIGH",
  "shap_top_factors": [
    { "feature": "night_indicator", "value": 1.0, "shap": 1.2 },
    { "feature": "historical_accident_rate", "value": 5.0, "shap": 1.25 },
    { "feature": "speed_limit_kmh", "value": 80.0, "shap": 0.9 },
    { "feature": "precipitation_mm", "value": 0.0, "shap": 0.0 }
  ],
  "shap_summary": "Risk is mainly driven by night-time conditions and high historical crash rate."
}
```

---

### `app/routers/route.py`

**What it does:** `POST /v1/route/safe` — safety-aware routing endpoint. Loads the road graph for the origin/destination area, scores every edge with the ML model, runs Yen's K-shortest-paths algorithm with the safety-time `alpha` weight, and returns up to 3 alternative routes ranked by average risk score.

**Input:**
```json
POST /v1/route/safe
{
  "origin":      { "lat": 16.50, "lon": 80.62 },
  "destination": { "lat": 16.52, "lon": 80.65 },
  "alpha": 0.6
}
```

**Output:**
```json
{
  "alternatives": [
    {
      "route_id": "route_0",
      "geometry": { "type": "LineString", "coordinates": [[80.62, 16.50], ...] },
      "distance_km": 3.4,
      "duration_min": 6.2,
      "avg_risk_score": 31.5,
      "is_safest": true,
      "risk_tier": "safest",
      "top_factors": [{ "feature": "speed_limit_kmh", "shap": 2.1, "label": "Speed Limit Kmh" }],
      "summary": "ROUTE 0: LOW RISK. Generally safe with minor localized triggers. (Index: 31.5/100).",
      "segments": [
        { "segment_id": "110123_110456", "risk_score": 28.0, "risk_level": "LOW" }
      ]
    }
  ]
}
```

---

### `app/routers/explain.py`

**What it does:** `GET /v1/explain/segment` — returns a full per-feature explanation payload for research inspection. In addition to the risk score it returns every feature value and its SHAP contribution, plus the model's expected value baseline.

**Input:**
```
GET /v1/explain/segment?lat=34.05&lon=-118.24
```

**Output:**
```json
{
  "segment_id": "110123_110456",
  "risk_score": 42,
  "features": {
    "hour_of_day": 14.0,
    "night_indicator": 0.0,
    "precipitation_mm": 0.0,
    "historical_accident_rate": 5.0,
    "..."
  },
  "shap_values": {
    "precipitation_mm": 0.0,
    "night_indicator": 0.0,
    "historical_accident_rate": 1.25,
    "..."
  },
  "expected_value": -5.5
}
```

---

## Database Layer

### `app/db/models.py`

**What it does:** SQLAlchemy ORM definitions for the two backend tables:
- `RoadSegment` — road network edges enriched with geometry, road class, speed limit, length, and historical accident rate.
- `Accident` — historical crash records snapped to a segment, with timestamp and severity.

**Input/Output:** Used internally by SQLAlchemy; no direct user-facing I/O.

---

### `app/db/session.py`

**What it does:** Creates the SQLAlchemy engine and `SessionLocal` factory from the `DATABASE_URL` environment variable (defaults to `sqlite:///./strive.db`). Exposes `get_db()` as a FastAPI dependency that yields a session and closes it automatically.

**Input/Output:** Provides `engine`, `SessionLocal`, `init_db()`, and `get_db()` for use by routers and seed scripts.

---

## Top-Level Pipeline Scripts

### `data_prepare.py`

**What it does:** M1 data-preparation orchestrator. Runs the full pipeline — FARS download → OSM download → accident snapping → rate computation → feature building — by invoking each `scripts/*.py` as a subprocess in sequence.

**Input (CLI):**
```bash
python data_prepare.py --city "Los Angeles, CA" --years 2021 2022 2023
# Optional flags: --skip-fars, --skip-osm
```

**Output (console + files):**
```
✓ ALL TASKS COMPLETED SUCCESSFULLY
Generated files:
  data/raw/fars_*.csv
  data/raw/road_network.graphml
  data/processed/accidents_snapped.parquet
  data/processed/segment_rates.parquet
  data/processed/features.parquet
```

---

### `check_outputs.py`

**What it does:** Quick sanity-check script. Loads all M1/M2 parquet files and model artefacts and prints summary statistics plus a deliverable checklist.

**Input:** Expects the processed data and model files to already exist on disk.

**Output (console):**
```
=== M1 Outputs ===
  accidents_snapped : (5823, 14)  cols: ['accident_id', 'osmid', ...]
  segment_rates     : (42108, 2)  cols: ['osmid', 'historical_accident_rate']
  features          : (5823, 13)  cols: ['hour_of_day', 'day_of_week', ...]
  snap_distance stats (m): mean=18.3  max=49.9
  rate stats: mean=0.0043  max=12.1

=== M2 Splits ===
  train :  8,148  (70.0%)  incident rate=50.00%
  val   :  1,746   (15.0%)  incident rate=50.00%
  test  :  1,746   (15.0%)  incident rate=50.00%

=== Missing deliverables check ===
  [OK] T-01 data/raw/fars_2021.csv
  ...
```

---

### `check2.py`

**What it does:** Minimal debug helper. Prints column names of the snapped accidents parquet file and lists a sample of OSM edge attributes to verify that `WEATHER`, `LGT_COND`, `FUNC_SYS`, `speed_kph`, `maxspeed`, and `highway` fields are present.

**Input:** Requires `data/processed/accidents_snapped.parquet` and `data/raw/road_network.graphml`.

**Output (console):**
```
Snapped cols: ['accident_id', 'osmid', 'latitude', ...]
  WEATHER in snapped? False
  speed_kph in snapped? True
Edge keys: ['osmid', 'name', 'highway', 'speed_kph', ...]
  speed_kph: 50.0
  maxspeed: NOT FOUND
  highway: primary
```

---

### `convert_osm_pbf_to_graphml.py`

**What it does:** Converts a large `.osm.pbf` file (e.g. a Geofabrik state extract) to a `road_network.graphml` file using `osmium`. Filters to valid highway types only and builds a `networkx.MultiDiGraph`.

**Input:** `data/raw/california-260418.osm.pbf`

**Output:**
```
✓ Ways processed: 1,823,456
✓ Edges created: 4,102,789
✓ Graph nodes: 2,198,034
✓ Graph edges: 4,102,789
✓ File size: 1812.3 MB
SUCCESS! Ready for pipeline
```
Output file: `data/raw/road_network.graphml`

---

### `migrate_to_real_data.py`

**What it does:** Migration helper that transitions the project from synthetic sample data to real NHTSA FARS and OpenStreetMap datasets. Backs up existing data, copies real FARS CSVs, optionally downloads a fresh OSM network, and re-triggers the data pipeline.

**Input (CLI):**
```bash
python migrate_to_real_data.py --fars-dir /path/to/real/fars --download-osm
```

**Output:**
```
✓ Migration complete!
Real data is now in use:
  data/raw/fars_*.csv (real NHTSA data)
  data/raw/road_network.graphml (real OSM)
  data/processed/*.parquet (reprocessed)
Sample data backed up to: data/backup_sample_data/
```

---

### `run_pipeline_wait_for_osm.py`

**What it does:** Automated pipeline runner that polls for the OSM road-network file to appear (useful when an async download is running in a separate process), validates it, then sequentially runs accident snapping and rate computation.

**Input:** No CLI arguments. Polls `data/raw/road_network.graphml`.

**Output:**
```
Step 1: Waiting for Road Network Download...
  Progress: 450.2 MB (elapsed: 8 min)
Road network ready! (912.4 MB, 14 min)
Step 2: Validating Road Network...
  Nodes: 2,198,034
  Edges: 4,102,789
Step 3: Running Accident Snapping...
Step 4: Computing Historical Rates...
SUCCESS! Pipeline Complete
```

---

## `scripts/` — Data & Training Pipeline

### `scripts/download_data.py`

**What it does:** Thin wrapper that calls `download_real_fars_data()` and `download_road_network()` from the other download scripts in one command.

**Input (CLI):**
```bash
python scripts/download_data.py --city "Los Angeles" --state CA --years 2021 2022 2023
```

**Output:**
```
Download summary:
- data/raw/fars_2021.csv: 245.3 KB
- data/raw/fars_2022.csv: 249.1 KB
- data/raw/fars_2023.csv: 251.7 KB
- data/raw/road_network.graphml: 189,432.8 KB
```

---

### `scripts/download_fars_data.py`

**What it does:** T-01 — Downloads or generates NHTSA FARS (Fatality Analysis Reporting System) accident data. In this research prototype the script generates realistic synthetic CSV data. A production implementation would download real CSVs from the NHTSA portal.

**Input (CLI):**
```bash
python scripts/download_fars_data.py --city "Los Angeles" --state CA --years 2021 2022 2023 --output data/raw
```

**Output files:** `data/raw/fars_2021.csv`, `data/raw/fars_2022.csv`, `data/raw/fars_2023.csv`

Each CSV has columns:
```
FARS_ID, YEAR, MONTH, DAY, HOUR, MINUTE, STATE, CITY,
LATITUDE, LONGITUDE, SEVERITY, FATALITIES, INJURIES,
NUM_VEHICLES, WEATHER, ROAD_SURFACE
```

Console:
```
✓ Saved data/raw/fars_2021.csv (2000 records)
✓ data/raw/fars_2021.csv: 2000 records, schema valid
```

---

### `scripts/download_osm_network.py`

**What it does:** T-02 — Downloads a driveable road network for a given place via OSMnx, parses real `maxspeed` strings (handling mph, km/h, and special tokens like `"motorway"`), adds `speed_kph` and `travel_time_sec` attributes to every edge, and saves to GraphML.

**Input (CLI):**
```bash
python scripts/download_osm_network.py --place "Los Angeles, California, USA" --network-type drive --validate
```

**Output:**
```
✓ Downloaded graph: 98,432 nodes, 241,876 edges
  Speed sources: 186,432 parsed from maxspeed, 55,444 defaulted to 50 km/h
✓ Saved graph to data/raw/road_network.graphml
Network Summary:
  Nodes: 98,432    Edges: 241,876    Avg degree: 4.91
```

---

### `scripts/download_geofabrik.py`

**What it does:** Downloads the Vijayawada metro area road network from OpenStreetMap via a targeted Overpass bounding-box query and saves it to `data/cache/graphs/city_vijayawada.graphml`.

**Input:** No CLI arguments needed; bbox is hard-coded (`north=16.65, south=16.40, east=80.75, west=80.40`).

**Output:**
```
Fetching Vijayawada road network via targeted Overpass query ...
Downloaded 14,231 nodes, 33,874 edges
Saved to data/cache/graphs/city_vijayawada.graphml
```

---

### `scripts/download_region.py`

**What it does:** Pre-downloads road networks for five Indian cities (Vijayawada, Vizag, Hyderabad, New Delhi, Bengaluru) at an 8 km radius and saves each as a named GraphML file in `data/cache/graphs/`.

**Input:** No CLI arguments.

**Output:**
```
[DOWNLOADING] Vijayawada, Andhra Pradesh, India (radius=8000m) ...
[OK] Saved → data/cache/graphs/city_vijayawada.graphml  (14231 nodes)
[DOWNLOADING] Visakhapatnam, Andhra Pradesh, India (radius=8000m) ...
...
All done! Restart the Docker container.
```

---

### `scripts/snap_accidents.py`

**What it does:** T-03 — Matches each FARS accident record to the nearest OSM road edge using `osmnx.nearest_edges`. Discards matches further than `threshold_m` metres (default 50 m). Extracts real FARS weather code (`WEATHER`), light condition (`LGT_COND`), and road type (`FUNC_SYS`) from each row.

**Input (CLI):**
```bash
python scripts/snap_accidents.py \
  --fars-dir data/raw --network data/raw/road_network.graphml \
  --threshold 50.0 --output data/processed/accidents_snapped.parquet
```

**Output file:** `data/processed/accidents_snapped.parquet`

Columns:
```
accident_id, osmid, latitude, longitude, snap_distance_m,
year, month, day, hour, minute,
weather_code, lgt_cond, func_sys, fatalities, drunk_drivers
```

Console:
```
Input:  6000 accidents
Output: 5823 snapped accidents
Match rate: 97.1%
Mean snap distance: 18.3 m
```

---

### `scripts/compute_accident_rates.py`

**What it does:** T-04 — Computes a per-segment historical accident rate using the formula `rate = count / length_km / years`. Merges snapped accident counts with OSM edge lengths and outputs a parquet file with one rate per segment.

**Input (CLI):**
```bash
python scripts/compute_accident_rates.py \
  --snapped data/processed/accidents_snapped.parquet \
  --network data/raw/road_network.graphml \
  --years 3 --output data/processed/segment_rates.parquet
```

**Output file:** `data/processed/segment_rates.parquet` — columns: `osmid`, `historical_accident_rate`

Console:
```
✓ Computed rates for 241,876 segments
  Mean rate: 0.0043 incidents/km/year
  Max rate: 12.1
  Segments with accidents: 4,812
```

---

### `scripts/build_features.py`

**What it does:** T-05 — Builds `data/processed/features.parquet` from real FARS data. Merges snapped accidents with segment rates and OSM edge attributes, then constructs all 12 model features using real FARS weather codes, light conditions, and functional system codes. Outputs one row per accident (all positives; `incident=1`).

**Input (CLI):**
```bash
python scripts/build_features.py \
  --snapped data/processed/accidents_snapped.parquet \
  --rates   data/processed/segment_rates.parquet \
  --network data/raw/road_network.graphml \
  --output  data/processed/features.parquet
```

**Output file:** `data/processed/features.parquet` — 12 feature columns + `incident` label

Console:
```
✓ Saved 5823 records to data/processed/features.parquet
=== Feature Quality Report ===
  hour_of_day        : 24 unique values  mean=13.241  std=5.821
  precipitation_mm   :  8 unique values  mean=1.123   std=3.201
  ...
```

---

### `scripts/split_dataset.py`

**What it does:** T-06 — Creates chronological train/val/test splits (70/15/15). Generates real negative samples (incident=0) by selecting zero-accident OSM segments and assigning them shuffled real weather values from the FARS positive pool — no random weather generation.

**Input:** Reads `data/processed/features.parquet`, `accidents_snapped.parquet`, `segment_rates.parquet`.

**Output files:** `data/splits/train.parquet`, `data/splits/val.parquet`, `data/splits/test.parquet`

Console:
```
  Zero-accident OSM segments available for negatives: 237,064
  Negatives built: 5823
  Combined shape : (11646, 13)
  Positives:  5,823  (50.0%)
  Negatives:  5,823  (50.0%)

Split sizes:
  train :   8,152  (70.0%)  incident=50.00%
  val   :   1,747   (15.0%)  incident=50.00%
  test  :   1,747   (15.0%)  incident=50.00%
[OK] Real-data splits saved to data/splits/
```

---

### `scripts/train_baseline.py`

**What it does:** T-07 — Trains an XGBoost baseline classifier with default-ish hyperparameters and early stopping on the validation set. Logs results to MLflow and saves the model to `models/baseline.pkl`.

**Input:** `data/splits/train.parquet`, `data/splits/val.parquet`

**Output file:** `models/baseline.pkl`

Console:
```
  Best iteration : 187
  Val AUROC      : 0.8341
[OK] Baseline model saved to models/baseline.pkl
```

---

### `scripts/tune_hyperparams.py`

**What it does:** T-08 — Runs Optuna hyperparameter search (50 trials by default, configurable via `STRIVE_OPTUNA_TRIALS`) with 3-fold time-series cross-validation on a 40 000-row chronological subsample of the training data. Logs every trial to MLflow.

**Input:** `data/splits/train.parquet`

**Output file:** `models/best_params.json`

Console:
```
Running Optuna search (50 trials, 3-fold time-series CV)...
  Best CV AUROC : 0.8612
  Best params   : { "max_depth": 7, "learning_rate": 0.048, ... }
[OK] Best params saved to models/best_params.json
```

---

### `scripts/evaluate_model.py`

**What it does:** T-09 — Trains the final XGBoost model using the best hyperparameters on train+val, evaluates on the held-out test set, and generates evaluation artefacts: ROC curve, PR curve, calibration plot, confusion matrix, and a Markdown evaluation report.

**Input:** `data/splits/*.parquet`, `models/best_params.json`

**Output files:**
- `models/model.pkl`
- `reports/evaluation.md`
- `reports/roc_curve.png`
- `reports/pr_curve.png`
- `reports/calibration.png`
- `reports/confusion_matrix.png`

Console:
```
  AUROC     : 0.8612  (target >= 0.82)  ✅
  AUPRC     : 0.4123  (target >= 0.35)  ✅
  F1        : 0.5891  (target >= 0.55)  ✅
  ECE       : 0.0512  (target <= 0.08)  ✅
[OK] reports/evaluation.md written
[OK] models/model.pkl saved
```

---

### `scripts/explain_shap.py`

**What it does:** T-10 — Computes SHAP values on up to 2 000 test samples using `shap.TreeExplainer`, generates a summary bar chart and beeswarm plot, runs domain validation checks (precipitation in top-5 for wet-night cases, night indicator positive for night-time), and writes `reports/shap_analysis.md`.

**Input:** `models/model.pkl`, `data/splits/test.parquet`

**Output files:** `reports/shap_summary.png`, `reports/shap_beeswarm.png`, `reports/shap_analysis.md`

Console:
```
Global feature importance (mean |SHAP|):
  historical_accident_rate           0.2341
  speed_limit_kmh                    0.1892
  night_indicator                    0.1541
  ...
Validation results:
  precipitation_mm in top-5 (wet-night): PASS
  historical_accident_rate in top-5    : PASS
  night_indicator positive for night   : PASS
```

---

### `scripts/save_artefacts.py`

**What it does:** T-11 — Derives the optimal classification threshold (maximising F1 on the test set) and writes `models/feature_config.json` — the full inference config consumed by the API at runtime.

**Input:** `models/model.pkl`, `data/splits/test.parquet`

**Output file:** `models/feature_config.json`

```json
{
  "feature_names": ["hour_of_day", "day_of_week", ...],
  "n_features": 12,
  "threshold": 0.483,
  "xgboost_version": "2.0.3",
  "risk_levels": {
    "LOW": [0, 25], "MODERATE": [26, 50], "HIGH": [51, 75], "CRITICAL": [76, 100]
  },
  "model_path": "models/model.pkl",
  "notes": "risk_score = round(100 * P(incident))."
}
```

---

### `scripts/train_model.py`

**What it does:** End-to-end training pipeline stitcher. Runs split → baseline → tune → evaluate → SHAP → save artefacts → research report in sequence by invoking each script as a subprocess.

**Input (CLI):**
```bash
python scripts/train_model.py --trials 50
# Optional flags: --skip-tuning, --rebuild-splits
```

**Output:** All model and report artefacts from T-06 through T-11 plus `reports/research_report.pdf`.

Console:
```
==> Splitting dataset
==> Training baseline model
==> Hyperparameter tuning
==> Final training and evaluation
==> SHAP analysis
==> Exporting artefact config
==> Generating research report
==> Pipeline complete
    models/baseline.pkl: present
    models/model.pkl: present
    reports/evaluation.md: present
    ...
```

---

### `scripts/generate_research_report.py`

**What it does:** Generates a compact PDF research report by parsing `reports/evaluation.md` and `reports/shap_analysis.md`, extracting metrics, and rendering three A4 pages (overview, feature engineering & SHAP, routing & limitations) with Matplotlib's `PdfPages`.

**Input:** `reports/evaluation.md`, `reports/shap_analysis.md`

**Output file:** `reports/research_report.pdf` (3-page PDF)

Console:
```
Created reports/research_report.pdf
```

---

### `scripts/benchmark_performance.py`

**What it does:** Measures the p50/p95 latency of `GET /v1/risk/segment` and `POST /v1/route/safe` by running 10 in-process requests via `httpx.AsyncClient` against the live FastAPI app (with all external dependencies patched to deterministic fixtures). Writes results to `reports/performance.md`.

**Input:** No CLI arguments.

**Output file:** `reports/performance.md`

```markdown
# STRIVE Performance Report

| Endpoint | p50 (ms) | p95 (ms) | Target |
|---|---:|---:|---|
| GET /v1/risk/segment | 12.4 | 18.7 | <= 500 ms |
| POST /v1/route/safe | 23.1 | 31.5 | <= 2000 ms |
```

---

### `scripts/seed_data.py`

**What it does:** Seeds the PostgreSQL/SQLite database from processed parquet files. Loads road-network edges and historical accident records, maps them to ORM models, truncates existing tables, and bulk-inserts fresh data.

**Input (CLI):**
```bash
python scripts/seed_data.py \
  --snapped data/processed/accidents_snapped.parquet \
  --rates   data/processed/segment_rates.parquet \
  --network data/raw/road_network.graphml
```

**Output:**
```
Seeded 241,876 road segments and 5823 accidents.
```

---

### `scripts/warmup_cities.py`

**What it does:** Pre-downloads road networks for four cities (Vijayawada, Visakhapatnam, Delhi, Agra) at a 10 km radius using `osmnx.graph_from_address` and saves them to `data/raw/`.

**Input:** No CLI arguments.

**Output:**
```
Pre-downloading Vijayawada, India...
Successfully saved Vijayawada, India to data/raw/vijayawada.graphml
...
```

---

### `scripts/create_synthetic_network.py`

**What it does:** Creates a synthetic 8×8 grid road network in an LA-area bounding box for development and testing. Road class varies by grid position (outer edges = motorway, inner = residential). Saves to `data/raw/road_network.graphml`.

**Input:** No arguments (run as script).

**Output:**
```
[OK] Created synthetic network: 64 nodes, 196 edges
[OK] Saved to data/raw/road_network.graphml
[OK] Verified: 64 nodes, 196 edges
Network created successfully!
```

---

## `alembic/` — Database Migrations

### `alembic/env.py`

**What it does:** Standard Alembic environment config. Reads `DATABASE_URL` from the environment, wires it to the Alembic config, sets `target_metadata` to the ORM models, and implements both offline and online migration modes.

**Input/Output:** Called by Alembic CLI commands (`alembic upgrade head`, etc.).

---

### `alembic/versions/20260420_000001_init_risk_api_schema.py`

**What it does:** Initial database migration. Creates the `road_segments` and `accidents` tables with all columns, indexes (`u`, `v`, `segment_id`), and the foreign-key constraint from `accidents.segment_id` to `road_segments.segment_id`.

**Input/Output:**
```bash
alembic upgrade head   # runs upgrade()
alembic downgrade -1   # runs downgrade() — drops both tables
```

---

## `tests/` — Test Suite

### `tests/conftest.py`
Shared pytest fixtures (test app client, dummy graph, mock DB session, patched weather/model).

### `tests/unit/test_features.py`
Unit tests for `build_feature_vector()`, `validate_feature_vector()`, and individual extractor functions.

### `tests/unit/test_inference.py`
Unit tests for `load_model()`, `run_inference()`, `explain_prediction()`, and `_FallbackRiskModel`.

### `tests/unit/test_astar.py`
Unit tests for `safe_route()`, `alternative_paths()`, `travel_time_seconds()`, and `_weight_function()`.

### `tests/unit/test_routing.py`
Unit tests for `get_graph_for_points()`, `nearest_node()`, and graph-cache logic.

### `tests/integration/test_risk_api.py`
Integration tests for `GET /v1/risk/segment` — checks HTTP status, response schema, risk level bucketing.

### `tests/integration/test_risk_heatmap.py`
Integration tests for `GET /v1/risk/heatmap` — bbox parsing, GeoJSON structure, empty-bbox edge case.

### `tests/integration/test_explain_api.py`
Integration tests for `GET /v1/explain/segment` — checks all feature and SHAP keys are present.

### `tests/integration/test_explain_segment.py`
Integration tests for explain segment response values and expected_value field.

### `tests/integration/test_risk_segment.py`
Integration tests verifying risk score is in [0, 100] and risk_level is a known label.

### `tests/integration/test_route_api.py`
Integration tests for `POST /v1/route/safe` — valid request/response, alpha extremes, bad coordinates.

### `tests/integration/test_route.py`
Integration tests for routing internals — edge scoring, geometry building, route summary computation.

### `tests/integration/test_data_pipeline.py`
Integration tests for the data pipeline steps — snapping, rate computation, feature building.

### `tests/integration/test_health.py`
Integration tests for `GET /health` — 200 OK when model and DB are reachable.

### `tests/e2e/test_full_pipeline.py`
End-to-end test that exercises the complete flow: data preparation → model training → API inference.

### `tests/e2e/test_demo.py`
End-to-end demo test that runs a representative routing request and prints a human-readable summary.

### `tests/performance/test_latency.py`
Performance tests asserting that p95 latency for key endpoints stays within the targets defined in `reports/performance.md`.
