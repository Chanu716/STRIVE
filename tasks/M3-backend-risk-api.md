# STRIVE — Task Sheet: M3 Backend Engineer (Risk API)

**Member role:** Backend Engineer — Risk API  
**Depends on:** M1 (`app/ml/features.py`), M2 (`models/model.pkl`, `models/feature_config.json`)  
**Produces for:** M4 (API endpoints for integration tests), M5 (API endpoints for testing docs)

---

## Responsibility

Set up the FastAPI project, define the PostgreSQL schema, implement the weather client, and build all risk-scoring and explainability endpoints. The API is the integration layer that ties together M1's features, M2's model, and the downstream routing work.

---

## Task List

| ID | Phase | Description | Output |
|---|---|---|---|
| **T-12** | 2 | Initialise FastAPI project structure and `requirements.txt` | `app/`, `requirements.txt` |
| **T-13** | 2 | Define PostgreSQL schema with Alembic migrations | `app/db/models.py`, `alembic/` |
| **T-14** | 2 | Implement `app/weather.py` — OpenWeatherMap client with 5-min in-process cache | `app/weather.py` |
| **T-15** | 2 | Implement `GET /v1/risk/segment` endpoint | `app/routers/risk.py` |
| **T-16** | 2 | Implement `GET /v1/risk/heatmap` endpoint (GeoJSON output) | `app/routers/risk.py` |
| **T-20** | 2 | Implement `GET /v1/explain/segment` endpoint (full SHAP output) | `app/routers/explain.py` |
| **T-21** | 2 | Add `GET /health` endpoint | `app/main.py` |
| **T-24** | 2 | Validate Swagger UI at `/docs` — confirm all endpoints are documented with examples | manual verification |

**Total: 8 tasks**

---

## Detailed Task Notes

### T-12 — FastAPI Project Structure

Create the following layout:

```
app/
├── main.py               # FastAPI app, CORS, router registration
├── db/
│   ├── __init__.py
│   ├── models.py         # SQLAlchemy ORM models
│   └── session.py        # get_db() dependency, engine setup
├── ml/
│   ├── __init__.py
│   ├── features.py       # provided by M1
│   └── inference.py      # load model.pkl; run_inference(feature_vec)
├── routers/
│   ├── __init__.py
│   ├── risk.py           # /v1/risk/segment, /v1/risk/heatmap
│   ├── explain.py        # /v1/explain/segment
│   └── route.py          # /v1/route/safe (implemented by M4)
└── weather.py            # OpenWeatherMap client
```

`requirements.txt` must pin exact versions of:
`fastapi`, `uvicorn[standard]`, `sqlalchemy`, `alembic`, `psycopg2-binary`, `xgboost`, `shap`, `osmnx`, `networkx`, `httpx`, `python-dotenv`, `optuna`

### T-13 — PostgreSQL Schema (Alembic)

**`road_segments` table:**

| Column | Type | Notes |
|---|---|---|
| `segment_id` | VARCHAR PK | OSM edge ID |
| `u` | BIGINT | OSM node u |
| `v` | BIGINT | OSM node v |
| `geometry` | JSONB | GeoJSON LineString |
| `road_class` | VARCHAR | highway tag value |
| `speed_limit_kmh` | FLOAT | extracted from OSM or defaults |
| `length_m` | FLOAT | edge length in metres |
| `historical_accident_rate` | FLOAT | from M1 T-04 |

**`accidents` table:**

| Column | Type | Notes |
|---|---|---|
| `accident_id` | SERIAL PK | |
| `segment_id` | VARCHAR FK → road_segments | |
| `timestamp` | TIMESTAMP | UTC |
| `severity` | SMALLINT | 1=fatal, 2=injury, 3=PDO |

Run migrations with:
```bash
alembic upgrade head
```

### T-14 — Weather Client (`app/weather.py`)

```python
import time, httpx
from functools import lru_cache

CACHE_TTL = 300  # seconds

_cache: dict = {}  # {(lat_round, lon_round): (timestamp, data)}

def get_weather(lat: float, lon: float) -> dict:
    key = (round(lat, 2), round(lon, 2))
    now = time.time()
    if key in _cache and now - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]
    resp = httpx.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": OPENWEATHERMAP_API_KEY, "units": "metric"},
        timeout=5,
    )
    resp.raise_for_status()
    data = _parse_weather(resp.json())
    _cache[key] = (now, data)
    return data
```

`_parse_weather()` must extract: `precipitation_mm`, `visibility_km`, `wind_speed_ms`, `temperature_c`.

### T-15 — `GET /v1/risk/segment`

**Request:** `?lat=34.05&lon=-118.24&datetime=2024-06-15T22:00:00`

**Processing pipeline:**
1. Snap lat/lon to nearest `road_segments` row in DB (use OSMnx `nearest_edges`).
2. Call `get_weather(lat, lon)`.
3. Build feature vector via `build_feature_vector()` from `app/ml/features.py`.
4. Load model from `models/model.pkl`; call `run_inference(feature_vec)` → `risk_score` [0–100].
5. Compute SHAP values; extract top 4 features by |SHAP|.

**Response schema:**
```json
{
  "segment_id": "123456789_987654321",
  "risk_score": 72,
  "risk_level": "HIGH",
  "shap_top_factors": [
    {"feature": "precipitation_mm", "value": 12.0, "shap": 0.31},
    {"feature": "night_indicator",  "value": 1,    "shap": 0.18}
  ],
  "shap_summary": "Risk is mainly driven by heavy rain and night-time conditions."
}
```

`risk_level` mapping: 0–24 LOW, 25–49 MODERATE, 50–74 HIGH, 75–100 CRITICAL.

### T-16 — `GET /v1/risk/heatmap`

**Request:** `?bbox=34.0,-118.3,34.1,-118.2`

Returns a GeoJSON FeatureCollection. Each feature is one road segment with properties `risk_score` and `risk_level`. Score all segments within the bounding box using cached weather for the bbox centroid.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "LineString", "coordinates": [...] },
      "properties": { "segment_id": "...", "risk_score": 55, "risk_level": "HIGH" }
    }
  ]
}
```

### T-20 — `GET /v1/explain/segment`

**Request:** `?lat=34.05&lon=-118.24&datetime=2024-06-15T22:00:00`

Returns all SHAP values (not just top 4) plus full feature values. Intended for research inspection.

```json
{
  "segment_id": "...",
  "risk_score": 72,
  "features": { "precipitation_mm": 12.0, "night_indicator": 1, ... },
  "shap_values": { "precipitation_mm": 0.31, "night_indicator": 0.18, ... },
  "expected_value": 0.25
}
```

### T-21 — `GET /health`

```json
{ "status": "ok", "model_loaded": true, "db_connected": true }
```

Returns HTTP 200 when healthy; HTTP 503 if model file or DB is unavailable.

### T-24 — Swagger UI Validation

- Open `/docs` and confirm every endpoint has:
  - A summary line and description.
  - At least one request example.
  - Documented response schema with field descriptions.
- Add `description=` and `summary=` to every route decorator; use Pydantic `Field(description=...)` on all request/response models.

---

## Deliverables Checklist

- [ ] `app/main.py` — FastAPI app initialised with CORS and all routers registered
- [ ] `app/db/models.py` — SQLAlchemy models for `road_segments` and `accidents`
- [ ] `app/db/session.py` — `get_db()` dependency and engine from `DATABASE_URL`
- [ ] `app/ml/inference.py` — model loading and inference wrapper
- [ ] `app/weather.py` — weather client with TTL cache
- [ ] `app/routers/risk.py` — `/v1/risk/segment` and `/v1/risk/heatmap`
- [ ] `app/routers/explain.py` — `/v1/explain/segment`
- [ ] `alembic/` — migration scripts; `alembic.ini`
- [ ] `requirements.txt` — pinned dependencies
- [ ] Swagger UI validated (screenshot or checklist in PR)

---

## Dependencies from Others

| Needs | Provided by |
|---|---|
| `app/ml/features.py` | M1 (T-05) |
| `models/model.pkl`, `models/feature_config.json` | M2 (T-11) |

## What Others Depend On from M3

| Deliverable | Used by |
|---|---|
| All API endpoints running | M4 (integration tests T-23) |
| `app/db/models.py` + migrations | M1 (seed script T-35) |
| Swagger UI docs | M5 (end-to-end test documentation) |
