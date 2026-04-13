# STRIVE — Task Sheet: M4 Backend Engineer (Routing)

**Member role:** Backend Engineer — Routing + DevOps  
**Depends on:** M3 (`app/` project structure, risk endpoints)  
**Produces for:** M5 (routing endpoint for integration test documentation)

---

## Responsibility

Build the safety-aware A\* routing engine, expose the routing endpoint, and own the Docker setup that packages the entire project. M4's work integrates everything into a single `docker compose up` workflow.

---

## Task List

| ID | Phase | Description | Output |
|---|---|---|---|
| **T-17** | 2 | Implement `app/routing/graph.py` — OSMnx road graph loading and in-memory caching | `app/routing/graph.py` |
| **T-18** | 2 | Implement `app/routing/astar.py` — A\* with risk-weighted edges | `app/routing/astar.py` |
| **T-19** | 2 | Implement `POST /v1/route/safe` endpoint | `app/routers/route.py` |
| **T-36** | 4 | Write `Dockerfile` and `docker-compose.yml` (API + PostgreSQL) | `Dockerfile`, `docker-compose.yml` |
| **T-37** | 4 | Write `.env.example` with all required environment variables documented | `.env.example` |

**Total: 5 tasks**

---

## Detailed Task Notes

### T-17 — Road Graph Loading (`app/routing/graph.py`)

The graph must be loaded once at API startup and cached in memory for the lifetime of the process.

```python
import osmnx as ox
import networkx as nx
from functools import lru_cache

_graph: nx.MultiDiGraph | None = None

def get_graph() -> nx.MultiDiGraph:
    global _graph
    if _graph is None:
        _graph = ox.load_graphml("data/raw/road_network.graphml")
    return _graph

def nearest_node(lat: float, lon: float) -> int:
    G = get_graph()
    return ox.nearest_nodes(G, X=lon, Y=lat)
```

- Expose `get_graph()` and `nearest_node(lat, lon)` as the public API for `astar.py` and `route.py`.
- The graphml path should be read from `GRAPH_PATH` env var with a sensible default.

### T-18 — A\* Routing with Risk-Weighted Edges (`app/routing/astar.py`)

**Edge cost function:**

```
cost(e) = α × risk_score(e) / 100  +  (1 − α) × travel_time(e) / max_travel_time
```

Where:
- `α ∈ [0, 1]` is the safety weight (user-provided; default 0.5)
- `risk_score(e)` is fetched from `GET /v1/risk/segment` or from a pre-scored cache
- `travel_time(e) = length_m / (speed_limit_kmh / 3.6)`
- Both terms are normalised to [0, 1] before combining

Implementation:

```python
import networkx as nx

def safe_route(
    G: nx.MultiDiGraph,
    origin_node: int,
    dest_node: int,
    alpha: float,
    risk_scores: dict[tuple, float],  # {(u, v, key): risk_score}
) -> list[int]:
    def weight(u, v, data):
        tt = data.get("length", 50) / max(data.get("speed_kph", 50) / 3.6, 1)
        risk = risk_scores.get((u, v, 0), 50) / 100
        return alpha * risk + (1 - alpha) * tt / MAX_TT

    path = nx.astar_path(G, origin_node, dest_node, weight=weight)
    return path
```

- `MAX_TT` should be computed as the 95th percentile travel time across all edges (pre-computed at startup).
- Return the ordered list of OSM node IDs on the route.

### T-19 — `POST /v1/route/safe`

**Request body:**
```json
{
  "origin":      { "lat": 34.052, "lon": -118.243 },
  "destination": { "lat": 34.073, "lon": -118.200 },
  "alpha":       0.6
}
```

**Processing:**
1. Snap origin and destination to nearest OSM nodes using `nearest_node()`.
2. Pre-score all candidate edges using cached weather + `run_inference()` (or load from DB cache).
3. Run `safe_route(G, origin_node, dest_node, alpha, risk_scores)`.
4. Also compute the **fastest route** (A\* with `alpha=0`, i.e., travel-time only).
5. Compute `vs_fastest` comparison.

**Response:**
```json
{
  "route": {
    "geometry": { "type": "LineString", "coordinates": [...] },
    "distance_km": 8.3,
    "duration_min": 14,
    "avg_risk_score": 38
  },
  "fastest": {
    "distance_km": 7.9,
    "duration_min": 12,
    "avg_risk_score": 61
  },
  "risk_reduction_pct": 37.7,
  "segments": [
    { "segment_id": "...", "risk_score": 28, "risk_level": "MODERATE" }
  ]
}
```

### T-36 — `Dockerfile` and `docker-compose.yml`

**`Dockerfile`:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`:**
```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: strive
      POSTGRES_PASSWORD: strive
      POSTGRES_DB: strive
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U strive"]
      interval: 5s
      retries: 5
```

Verify with `docker compose up --build` and `docker compose ps` — both services must be healthy.

### T-37 — `.env.example`

Document every required environment variable with a description and example value:

```dotenv
# PostgreSQL connection string
DATABASE_URL=postgresql://strive:strive@db:5432/strive

# OpenWeatherMap API key (free tier, register at openweathermap.org)
OPENWEATHERMAP_API_KEY=your_api_key_here

# Path to pre-downloaded OSM road graph
GRAPH_PATH=data/raw/road_network.graphml

# Path to trained XGBoost model
MODEL_PATH=models/model.pkl

# Optional: MLflow tracking server URI (default: local file store)
MLFLOW_TRACKING_URI=mlruns
```

Copy this file to `.env` and fill in real values before running.

---

## Deliverables Checklist

- [ ] `app/routing/graph.py` — graph loader with in-memory cache and `nearest_node()`
- [ ] `app/routing/astar.py` — `safe_route()` with composite edge cost
- [ ] `app/routers/route.py` — `POST /v1/route/safe` with vs-fastest comparison
- [ ] `Dockerfile` — builds and starts the API cleanly
- [ ] `docker-compose.yml` — starts API + PostgreSQL with health check
- [ ] `.env.example` — all variables documented

---

## Dependencies from Others

| Needs | Provided by |
|---|---|
| `app/main.py` (to register router) | M3 (T-12) |
| `app/ml/inference.py` (for edge scoring) | M3 (T-12) |
| `data/raw/road_network.graphml` | M1 (T-02) |

## What Others Depend On from M4

| Deliverable | Used by |
|---|---|
| `docker-compose.yml` | M5 (end-to-end test and performance check) |
| `.env.example` | Every team member (local setup) |
| `POST /v1/route/safe` endpoint | M5 (integration test documentation) |
