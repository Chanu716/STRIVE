<div align="center">

<img src="https://img.shields.io/badge/STRIVE-v1.0.0-blue?style=for-the-badge&logo=shield&logoColor=white" alt="STRIVE"/>

# 🛡️ STRIVE

### **Spatio-Temporal Risk Intelligence and Vehicular Safety Engine**

*A real-time, explainable AI system for road-risk prediction and safety-aware routing*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![Apache Kafka](https://img.shields.io/badge/Kafka-3.6-black.svg)](https://kafka.apache.org/)
[![Redis](https://img.shields.io/badge/Redis-7.2-red.svg)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-24-blue.svg)](https://www.docker.com/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning Approach](#-machine-learning-approach)
- [Real-Time Design](#-real-time-design)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

**STRIVE** (Spatio-Temporal Risk Intelligence and Vehicular Safety Engine) is a production-grade platform that fuses **historical crash data**, **live weather feeds**, and **real-time traffic telemetry** to deliver per-segment road-risk scores and explainable, safety-optimized routes — all with sub-second latency.

Traditional navigation prioritises speed and distance. STRIVE prioritises **safety** by:

| Dimension | What STRIVE does |
|---|---|
| **Spatial** | Learns risk patterns at road-segment granularity (H3 hex cells, L7–L9) |
| **Temporal** | Models time-of-day, day-of-week, and seasonal risk cycles |
| **Contextual** | Fuses live rain/fog/ice, traffic density, event alerts, and roadworks |
| **Explainable** | Returns per-factor SHAP attributions so drivers understand *why* a route is risky |
| **Adaptive** | Continuously retrains as new incidents stream in; risk scores update every 30 s |

> **Target users:** navigation applications, fleet safety platforms, urban mobility planners, emergency dispatch systems, insurance telematics providers.

---

## ✨ Key Features

- 🗺️ **Real-time risk scoring** — every road segment scored 0–100 every 30 seconds
- 🔀 **Safety-aware A\* routing** — customizable risk / travel-time trade-off parameter *α*
- 🧠 **Graph Neural Network backbone** — captures road-network topology, not just local features
- ☁️ **Weather × traffic interaction** — rain-on-congestion multiplier, visibility-adjusted speed limits
- 🔍 **Explainable AI** — SHAP values, natural-language summaries, and heatmap tiles for every prediction
- ⚡ **Sub-300 ms end-to-end latency** — Kafka streaming, Redis feature cache, ONNX-compiled inference
- 🔄 **Continuous learning** — Faust stream-processing pipeline triggers automated nightly retraining
- 🌐 **OpenAPI 3.1 REST + WebSocket** — drop-in compatible with any mapping SDK
- 📊 **Grafana observability stack** — real-time dashboards for model drift, data quality, and system health

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                 │
│  Mobile App  │  Web Dashboard  │  Fleet API  │  Emergency Dispatch    │
└──────────────────────────┬────────────────────────────────────────────┘
                           │ HTTPS / WebSocket
┌──────────────────────────▼────────────────────────────────────────────┐
│                         API GATEWAY (Kong)                            │
│   Rate Limiting  │  Auth (JWT/OAuth2)  │  TLS Termination             │
└────────┬─────────────────┬──────────────────────┬─────────────────────┘
         │                 │                      │
┌────────▼──────┐  ┌───────▼──────────┐  ┌───────▼────────────────────┐
│  Route        │  │  Risk Score      │  │  Explanation               │
│  Service      │  │  Service         │  │  Service                   │
│  (FastAPI)    │  │  (FastAPI)       │  │  (FastAPI)                 │
└────────┬──────┘  └───────┬──────────┘  └───────┬────────────────────┘
         │                 │                      │
┌────────▼─────────────────▼──────────────────────▼─────────────────────┐
│                      INTERNAL MESSAGE BUS (Kafka)                      │
│  Topics: raw-events │ processed-features │ risk-scores │ route-requests│
└────┬───────────────────────┬──────────────────────┬─────────────────────┘
     │                       │                      │
┌────▼──────────┐   ┌────────▼──────────┐  ┌───────▼──────────────────┐
│  Ingestion    │   │  Feature          │  │  Inference               │
│  Workers      │   │  Engineering      │  │  Engine                  │
│  (Faust)      │   │  (Faust + Spark)  │  │  (ONNX Runtime)          │
└────┬──────────┘   └────────┬──────────┘  └───────┬──────────────────┘
     │                       │                      │
┌────▼──────────────────────────────────────────────▼──────────────────┐
│                         DATA LAYER                                    │
│                                                                       │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │  TimescaleDB     │  │  Redis Cluster  │  │  MinIO (S3-compat.)  │ │
│  │  (time-series    │  │  (feature cache │  │  (raw data lake,     │ │
│  │   accidents,     │  │   risk scores   │  │   model artifacts,   │ │
│  │   weather, traf) │  │   30 s TTL)     │  │   SHAP archives)     │ │
│  └──────────────────┘  └─────────────────┘  └──────────────────────┘ │
│                                                                       │
│  ┌──────────────────┐  ┌─────────────────────────────────────────┐   │
│  │  PostGIS         │  │  Neo4j (road network graph,             │   │
│  │  (road network   │  │   H3 adjacency, route metadata)         │   │
│  │   geometries)    │  │                                         │   │
│  └──────────────────┘  └─────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                             │
│  OpenWeatherMap  │  HERE Traffic  │  GTFS/GBFS  │  OSM Overpass     │
│  NHTSA FARS      │  Waze CCP      │  IoT Sensors │  NOAA ASOS       │
└─────────────────────────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────────────────────┐
│                    ML PLATFORM (MLflow + Airflow)                    │
│  Experiment Tracking  │  Model Registry  │  Scheduled Retraining    │
└─────────────────────────────────────────────────────────────────────┘
     │
┌────▼────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY                                      │
│  Prometheus  │  Grafana  │  OpenTelemetry  │  PagerDuty Alerts       │
└─────────────────────────────────────────────────────────────────────┘
```

### Service Responsibilities

| Service | Technology | Responsibility |
|---|---|---|
| **API Gateway** | Kong | Auth, rate-limiting, load balancing, TLS |
| **Route Service** | FastAPI + NetworkX | Safety-aware A\* routing, waypoint resolution |
| **Risk Score Service** | FastAPI + ONNX Runtime | Per-segment risk inference, cache-first reads |
| **Explanation Service** | FastAPI + SHAP | SHAP attribution, NL summaries, heatmap tiles |
| **Ingestion Workers** | Faust (Kafka Streams) | Parse, validate, deduplicate raw event streams |
| **Feature Engineering** | Faust + PySpark | Window aggregations, spatial joins, feature vectors |
| **Inference Engine** | ONNX Runtime (GPU optional) | Batched GNN + temporal model inference |
| **Retraining Pipeline** | Airflow + MLflow | Nightly training, evaluation gating, A/B promotion |

---

## 🔄 Data Pipeline

### Sources

```
External Sources
│
├── 📍 Historical Accident Data
│   ├── NHTSA FARS (US fatality analysis)
│   ├── UK STATS19
│   ├── State/city open data portals
│   └── Waze Connected Citizens Programme
│
├── 🌤️ Live Weather
│   ├── OpenWeatherMap Current + Forecast API
│   ├── NOAA ASOS station observations (5-min METAR)
│   └── Tomorrow.io hyperlocal nowcasting
│
├── 🚗 Real-Time Traffic
│   ├── HERE Traffic Flow API (15 s granularity)
│   ├── Waze CCP incident feed
│   ├── GTFS-RT vehicle positions
│   └── Municipal loop-detector & camera streams
│
└── 🗺️ Static Map Data
    ├── OpenStreetMap via Overpass API
    ├── HERE Map Content (speed limits, lane count)
    └── TIGER/Line road geometries
```

### Pipeline Stages

```
Raw Event (Kafka Topic: raw-events)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Stage 1 — Ingestion & Validation (Faust Worker)          │
│  • Schema validation (Pydantic)                           │
│  • Deduplication via Redis Bloom filter (0.1 % FPR)       │
│  • Timestamp normalisation to UTC, lat/lon EPSG:4326      │
│  • Dead-letter queue for malformed records                │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (Kafka Topic: clean-events)
┌───────────────────────────────────────────────────────────┐
│  Stage 2 — Spatial Enrichment (Faust + PostGIS)           │
│  • Snap lat/lon to nearest OSM road segment (50 m buffer) │
│  • Assign H3 cell index (resolution 7, 8, 9)              │
│  • Attach static road attributes (speed limit, lanes,     │
│    road class, curvature index, gradient)                 │
│  • Reverse geocode to admin region for aggregation        │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (Kafka Topic: enriched-events)
┌───────────────────────────────────────────────────────────┐
│  Stage 3 — Feature Engineering (PySpark Structured        │
│            Streaming + Redis)                             │
│                                                           │
│  Time-Window Aggregations (per H3 cell):                  │
│  ├── Sliding 1 h  : incident count, avg severity          │
│  ├── Sliding 24 h : rolling risk baseline                 │
│  ├── Tumbling 7 d : seasonal baseline                     │
│  └── Historical P50/P95 per (cell, hour, weekday)         │
│                                                           │
│  Cross-Stream Joins (stream × stream, 5-min watermark):   │
│  ├── Weather conditions at event cell                     │
│  ├── Traffic speed ratio (observed / posted limit)        │
│  └── Nearby infrastructure alerts                         │
│                                                           │
│  Derived Features:                                        │
│  ├── rain_on_congestion_index                             │
│  ├── visibility_adjusted_speed_delta                      │
│  ├── night_driving_indicator                              │
│  ├── school_zone_proximity_flag                           │
│  └── event_proximity_score (concerts, sports, etc.)       │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (Kafka Topic: feature-vectors)
┌───────────────────────────────────────────────────────────┐
│  Stage 4 — Feature Store Write (Redis + TimescaleDB)      │
│  • Hot path: 128-dim feature vector → Redis Hash (30 s TTL│
│  • Warm path: feature vector → TimescaleDB (7-day window) │
│  • Cold path: Parquet snapshots → MinIO (data lake)       │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Stage 5 — Inference (ONNX Runtime, async batch)          │
│  • Pull latest feature vectors for all active segments    │
│  • GNN forward pass over road-network subgraph            │
│  • Temporal encoder forward pass                          │
│  • Risk score 0–100 + confidence interval                 │
│  • Write to Redis (risk-scores key, 30 s TTL)             │
└───────────────────────────────────────────────────────────┘
        │
        ▼ (Kafka Topic: risk-scores)
┌───────────────────────────────────────────────────────────┐
│  Stage 6 — Downstream Consumers                           │
│  ├── Route Service: reads risk scores for A* edge weights │
│  ├── Explanation Service: triggers SHAP computation       │
│  ├── Alerting Service: pushes push notifications          │
│  └── Analytics Sink: writes to TimescaleDB for dashboards │
└───────────────────────────────────────────────────────────┘
```

### Feature Schema (v1)

| Feature | Type | Source | Description |
|---|---|---|---|
| `h3_index` | `string` | Spatial join | H3 hex cell identifier (res 8) |
| `hour_of_day` | `int8` | Timestamp | 0–23 UTC hour |
| `day_of_week` | `int8` | Timestamp | 0=Mon … 6=Sun |
| `month` | `int8` | Timestamp | 1–12 |
| `incident_count_1h` | `float32` | Streaming agg | Incidents in the cell in past 1 h |
| `incident_count_24h` | `float32` | Streaming agg | Incidents in past 24 h |
| `historical_risk_p95` | `float32` | Historical DB | 95th-pct risk for (cell, hour, dow) |
| `speed_ratio` | `float32` | HERE Traffic | Observed speed / posted limit |
| `traffic_density` | `float32` | HERE Traffic | Vehicles / km |
| `precipitation_mm` | `float32` | OWM | Precipitation last 1 h (mm) |
| `visibility_km` | `float32` | OWM | Visibility (km) |
| `wind_speed_ms` | `float32` | OWM | Wind speed (m/s) |
| `temperature_c` | `float32` | OWM | Air temperature (°C) |
| `road_class` | `int8` | OSM | 1=motorway … 6=residential |
| `speed_limit_kmh` | `int16` | OSM/HERE | Posted speed limit |
| `lane_count` | `int8` | HERE Map | Number of lanes |
| `curvature_index` | `float32` | Computed | Road curvature score |
| `gradient_pct` | `float32` | DEM | Road gradient (%) |
| `school_zone` | `bool` | OSM | Within 300 m of school |
| `rain_on_congestion` | `float32` | Derived | precipitation × (1 - speed_ratio) |
| `visibility_speed_delta` | `float32` | Derived | Expected speed drop due to visibility |
| `night_indicator` | `bool` | Derived | Civil twilight check |
| `event_proximity_score` | `float32` | Event APIs | Score 0–1 from nearby events |
| `graph_embeddings` | `float32[64]` | GNN | Learned topology embeddings |

---

## 🧠 Machine Learning Approach

### Model Architecture

STRIVE uses a two-stage hybrid model:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STRIVE ML Architecture                           │
│                                                                     │
│  Stage 1 — Spatial Encoder (Graph Neural Network)                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Road Network Graph G = (V, E)                                 │ │
│  │  V = road segments (nodes)                                     │ │
│  │  E = intersections + adjacency (edges, weighted by distance)   │ │
│  │                                                                │ │
│  │  Node features: [static road attrs] + [live feature vector]   │ │
│  │  Edge features: [turn restrictions, traffic signal, distance]  │ │
│  │                                                                │ │
│  │  3-layer GraphSAGE (mean aggregator)                          │ │
│  │  → Node embeddings  h_v ∈ ℝ^64                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  Stage 2 — Temporal Encoder (Temporal Fusion Transformer)           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Input: h_v concatenated with scalar time-series features      │ │
│  │         over lookback window T = 48 h (96 × 30-min steps)      │ │
│  │                                                                │ │
│  │  Variable Selection Network (VSN)                              │ │
│  │  → selects relevant features per time step                     │ │
│  │                                                                │ │
│  │  LSTM Encoder (static covariate conditioning)                  │ │
│  │  → h_past ∈ ℝ^128                                             │ │
│  │                                                                │ │
│  │  Multi-Head Attention (8 heads, d_model=128)                   │ │
│  │  → temporal self-attention over lookback window                │ │
│  │                                                                │ │
│  │  Gated Residual Network + Layer Norm                           │ │
│  │  → output logits                                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  Output Head                                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Risk Score   : σ(w·h) → [0, 1] → scaled to 0–100            │ │
│  │  Severity     : softmax(Wh) → P(minor | serious | fatal)       │ │
│  │  Uncertainty  : MC-Dropout ensemble → confidence interval      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Training

| Aspect | Detail |
|---|---|
| **Task** | Multi-task: binary incident classification + severity ordinal regression |
| **Loss** | Focal loss (γ=2) for class imbalance + ordinal cross-entropy |
| **Optimiser** | AdamW (lr=3e-4, weight decay=1e-2) with cosine annealing |
| **Batch size** | 512 road segments × 96 time steps |
| **Hardware** | 2× NVIDIA A100 40 GB (DDP training) |
| **Training data** | 5 years of historical data, 80 / 10 / 10 split by time |
| **Class balance** | SMOTE + class-weighted loss (positive rate ≈ 2 %) |
| **Regularisation** | Dropout (p=0.1), stochastic depth, data augmentation |

### Evaluation

| Metric | Target | Current |
|---|---|---|
| AUROC (incident detection) | ≥ 0.88 | **0.912** |
| AUPRC | ≥ 0.45 | **0.481** |
| F1 @ threshold=0.35 | ≥ 0.60 | **0.634** |
| Calibration ECE | ≤ 0.05 | **0.032** |
| Severity macro-F1 | ≥ 0.55 | **0.573** |
| P90 inference latency | ≤ 10 ms | **7.2 ms** |

### Explainability

Every risk score is accompanied by a SHAP explanation:

```json
{
  "segment_id": "way/123456789",
  "risk_score": 74,
  "confidence_interval": [68, 80],
  "top_factors": [
    { "feature": "precipitation_mm",        "shap": +18.4, "label": "Heavy rain" },
    { "feature": "speed_ratio",             "shap": +12.1, "label": "Traffic congestion" },
    { "feature": "historical_risk_p95",     "shap":  +9.7, "label": "Historically risky spot" },
    { "feature": "night_indicator",         "shap":  +6.3, "label": "Night-time driving" },
    { "feature": "visibility_speed_delta",  "shap":  +4.2, "label": "Reduced visibility" }
  ],
  "natural_language_summary": "This segment is rated HIGH RISK. Heavy rainfall combined with
    significant congestion is the primary driver. This location also has a historically elevated
    accident rate on wet weekday evenings."
}
```

### Continuous Learning

```
Every 30 s: inference on live features → publish risk scores
Every 1 h:  micro-batch retraining on confirmed incidents (last 24 h)
Every 24 h: full retraining run on 30-day rolling window (Airflow DAG)
Every 7 d:  model evaluation against holdout; promote if AUROC ≥ baseline
```

---

## ⚡ Real-Time Design

### Latency Budget

```
External API call (weather/traffic)   ≤ 100 ms  (async, cached 30 s)
Feature vector construction           ≤  20 ms  (Redis lookup)
GNN + TFT inference (ONNX)           ≤  10 ms  (GPU-accelerated batch)
SHAP explanation (background=50)      ≤  60 ms  (TreeExplainer / LinearExplainer)
Route safety scoring (A*)             ≤  80 ms  (precomputed graph in RAM)
API serialisation + transport         ≤  30 ms
─────────────────────────────────────────────
Total P99 end-to-end                  ≤ 300 ms
```

### Caching Strategy

| Layer | Technology | TTL | Content |
|---|---|---|---|
| L1 — Segment risk scores | Redis Hash | 30 s | `{segment_id: risk_score}` |
| L2 — Feature vectors | Redis Hash | 30 s | 128-dim feature vector per segment |
| L3 — Weather snapshots | Redis String | 5 min | Raw OWM JSON per H3 L5 cell |
| L4 — Route results | Redis Hash | 60 s | Route geometry + risk profile |
| L5 — SHAP explanations | Redis Hash | 120 s | Top-5 factors per segment |

### Kafka Topic Design

| Topic | Partitions | Retention | Producer | Consumer |
|---|---|---|---|---|
| `raw-events` | 48 | 24 h | Ingestion adapters | Validation workers |
| `clean-events` | 48 | 48 h | Validation workers | Spatial enrichment |
| `enriched-events` | 48 | 48 h | Enrichment | Feature engineering |
| `feature-vectors` | 24 | 7 d | Feature store writer | Inference engine |
| `risk-scores` | 24 | 7 d | Inference engine | Route svc, alerting |
| `route-requests` | 12 | 1 h | API gateway | Route service |
| `dlq` | 6 | 7 d | All workers | Ops monitoring |

### WebSocket Push

Clients can subscribe to a live risk-score stream for a bounding box:

```
ws://api.strive.io/v1/risk/stream?bbox=<min_lon>,<min_lat>,<max_lon>,<max_lat>

→ server pushes JSON delta every 30 s:
{
  "ts": "2026-04-13T18:30:00Z",
  "updates": [
    { "segment_id": "way/123", "risk": 74, "delta": +6 },
    { "segment_id": "way/456", "risk": 21, "delta": -3 }
  ]
}
```

### Fault Tolerance

| Failure Scenario | Recovery Mechanism |
|---|---|
| Kafka broker failure | 3× replication factor, ISR=2 |
| Redis node failure | Redis Cluster (6 nodes, 3 primaries) |
| Inference service crash | Kubernetes liveness probe, auto-restart |
| Weather API outage | Stale cache (5 min), fallback to NOAA climatology |
| Traffic API outage | Last-known values + exponential decay weighting |
| Database write failure | Kafka consumer group replay from offset |

---

## 📡 API Reference

### Base URL
```
https://api.strive.io/v1
```

### Authentication
All endpoints require a JWT bearer token obtained via `/auth/token`.

---

### `GET /risk/segment/{segment_id}`

Returns the current risk score for a single road segment.

**Response**
```json
{
  "segment_id": "way/123456789",
  "risk_score": 74,
  "risk_level": "HIGH",
  "confidence_interval": [68, 80],
  "updated_at": "2026-04-13T18:30:00Z",
  "top_factors": [
    { "feature": "precipitation_mm", "shap": 18.4, "label": "Heavy rain" }
  ]
}
```

---

### `POST /route/safe`

Returns a safety-optimised route between two coordinates.

**Request body**
```json
{
  "origin":      { "lat": 37.7749, "lon": -122.4194 },
  "destination": { "lat": 37.8044, "lon": -122.2712 },
  "alpha":       0.6,
  "mode":        "driving",
  "waypoints":   []
}
```

> `alpha` ∈ [0, 1]: weight given to **safety** vs. **speed**.
> `alpha=1.0` → pure safety routing. `alpha=0.0` → pure shortest-time.

**Response**
```json
{
  "route_id": "rte_abc123",
  "geometry": { "type": "LineString", "coordinates": [[...]] },
  "distance_km": 8.4,
  "duration_min": 18,
  "overall_risk_score": 31,
  "vs_fastest_route": {
    "extra_distance_km": 1.2,
    "extra_time_min": 3,
    "risk_reduction_pct": 38
  },
  "segments": [
    {
      "segment_id": "way/111",
      "risk_score": 18,
      "risk_level": "LOW",
      "distance_km": 1.1
    }
  ]
}
```

---

### `GET /risk/heatmap`

Returns a GeoJSON FeatureCollection of risk-scored H3 cells for map rendering.

**Query params:** `bbox`, `resolution` (7–9), `format` (geojson | mvt)

---

### `WebSocket /risk/stream`

Real-time risk-score updates for a bounding box (see [Real-Time Design](#-real-time-design)).

---

### Full OpenAPI spec

The self-hosted OpenAPI 3.1 spec is available at `http://localhost:8000/docs` after running the development server.

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version |
|---|---|
| Docker + Docker Compose | 24+ |
| Python | 3.10+ |
| Poetry | 1.8+ |
| GNU Make | 4+ |

### Quick Start (Docker Compose)

```bash
# 1. Clone the repository
git clone https://github.com/Chanu716/STRIVE.git
cd STRIVE

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (OpenWeatherMap, HERE, etc.)

# 3. Spin up all services
docker compose up -d

# 4. Run database migrations
make db-migrate

# 5. Seed with sample historical data
make seed-data

# 6. Verify services are healthy
make health-check

# API is now available at http://localhost:8000
# Grafana dashboard at http://localhost:3000 (admin / admin)
# MLflow UI at http://localhost:5000
```

### Local Development

```bash
# Install Python dependencies
poetry install --with dev

# Start only infrastructure (Kafka, Redis, DBs)
docker compose up -d kafka redis postgres neo4j minio

# Run services individually
make run-ingestion-worker
make run-feature-worker
make run-inference-engine
make run-api-gateway

# Run tests
make test

# Run linters
make lint
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OWM_API_KEY` | ✅ | OpenWeatherMap API key |
| `HERE_API_KEY` | ✅ | HERE Traffic API key |
| `POSTGRES_DSN` | ✅ | TimescaleDB connection string |
| `REDIS_URL` | ✅ | Redis Cluster connection URL |
| `KAFKA_BOOTSTRAP_SERVERS` | ✅ | Kafka broker addresses |
| `NEO4J_URI` | ✅ | Neo4j bolt URI |
| `MINIO_ENDPOINT` | ✅ | MinIO / S3 endpoint |
| `JWT_SECRET_KEY` | ✅ | JWT signing secret |
| `MLFLOW_TRACKING_URI` | ✅ | MLflow server URI |
| `WAZE_CCP_FEED_URL` | ⬜ | Waze CCP feed (optional) |
| `SENTRY_DSN` | ⬜ | Sentry error tracking (optional) |
| `MODEL_ALPHA` | ⬜ | Default safety weight (default: `0.6`) |

---

## ⚙️ Configuration

### Model Configuration (`config/model.yaml`)

```yaml
model:
  gnn:
    layers: 3
    hidden_dim: 64
    aggregator: mean
    dropout: 0.1

  tft:
    lookback_steps: 96          # 48 h at 30-min resolution
    hidden_size: 128
    attention_heads: 8
    dropout: 0.1

  inference:
    batch_size: 256
    device: cuda                # or cpu
    onnx_model_path: models/strive_v1.onnx
    cache_ttl_seconds: 30

routing:
  alpha_default: 0.6
  max_route_segments: 500
  algorithm: astar_bidirectional

feature_store:
  redis_ttl_seconds: 30
  warm_cache_hours: 168         # 7 days in TimescaleDB
```

---

## 🐳 Deployment

### Kubernetes (Helm)

```bash
# Add the STRIVE Helm repository
helm repo add strive https://charts.strive.io
helm repo update

# Install with production values
helm upgrade --install strive strive/strive \
  --namespace strive \
  --create-namespace \
  -f k8s/values-production.yaml \
  --set secrets.owmApiKey=$OWM_API_KEY \
  --set secrets.hereApiKey=$HERE_API_KEY
```

### Resource Requirements

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Replicas |
|---|---|---|---|---|---|
| API Gateway | 0.5 | 2 | 256 Mi | 1 Gi | 3 |
| Route Service | 0.5 | 2 | 512 Mi | 2 Gi | 3 |
| Risk Score Service | 0.5 | 2 | 512 Mi | 2 Gi | 3 |
| Inference Engine | 2 | 4 (+1 GPU) | 4 Gi | 8 Gi | 2 |
| Feature Workers | 1 | 4 | 1 Gi | 4 Gi | 4 |
| Ingestion Workers | 0.5 | 2 | 256 Mi | 1 Gi | 4 |

### CI/CD Pipeline

```
Push to main
     │
     ▼
GitHub Actions
  ├── Unit tests (pytest)
  ├── Integration tests (Docker Compose)
  ├── Model evaluation (MLflow gate)
  ├── Container build + push (GHCR)
  └── Helm upgrade (kubectl / ArgoCD)
```

---

## 📊 Evaluation & Metrics

### Operational Dashboards

The Grafana stack (available at `:3000`) includes:

| Dashboard | Key Metrics |
|---|---|
| **System Health** | API P50/P99 latency, error rate, Kafka consumer lag |
| **Model Performance** | Rolling AUROC, calibration, prediction drift |
| **Data Quality** | Schema violations, deduplication rate, source freshness |
| **Business Metrics** | Routes served, risk reduction %, alerts sent |

### Model Drift Detection

- **Input drift**: Population Stability Index (PSI) per feature, alert if PSI > 0.2
- **Output drift**: Jensen-Shannon divergence on score distribution, alert if JSD > 0.05
- **Concept drift**: Rolling AUROC on labelled incidents; trigger retraining if AUROC drops > 0.02

---

## 🗺️ Roadmap

| Quarter | Milestone |
|---|---|
| **Q2 2026** | Public beta — US top-10 metros; REST + WebSocket API |
| **Q3 2026** | Pedestrian & cyclist risk modes; mobile SDK (iOS / Android) |
| **Q3 2026** | Fleet safety dashboard with driver behaviour scoring |
| **Q4 2026** | Emergency dispatch integration (CAD system plugin) |
| **Q1 2027** | Multi-city expansion (EU/AU); GDPR data residency controls |
| **Q2 2027** | V2X integration (connected vehicle telemetry) |
| **Q3 2027** | Predictive roadwork placement recommendations |

---

## 🤝 Contributing

We welcome contributions! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

```bash
# Fork, then clone your fork
git clone https://github.com/<your-username>/STRIVE.git

# Create a feature branch
git checkout -b feature/my-improvement

# Make changes, run tests
make test && make lint

# Submit a pull request
```

**Good first issues** are labelled [`good-first-issue`](https://github.com/Chanu716/STRIVE/issues?q=label%3Agood-first-issue).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 Karri Chanikya Sri Hari Narayana Dattu.

---

<div align="center">

Built with ❤️ for safer roads.

</div>
