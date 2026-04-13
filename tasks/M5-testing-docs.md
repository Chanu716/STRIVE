# STRIVE — Task Sheet: M5 Testing, QA & Documentation

**Member role:** QA Engineer + Technical Writer  
**Depends on:** M3 (all API endpoints), M4 (Docker setup, routing endpoint), M2 (evaluation results)  
**Produces for:** Project team (test suite, validated performance, final README)

---

## Responsibility

Ensure the entire system works correctly and meets performance targets by writing the unit and integration test suites, performing end-to-end validation, running performance checks, and producing the final README that documents setup and results for a new user.

---

## Task List

| ID | Phase | Description | Output |
|---|---|---|---|
| **T-22** | 2 | Write unit tests for feature engineering, model inference, and routing (≥ 70 % coverage) | `tests/unit/` |
| **T-23** | 2 | Write integration tests for all API endpoints using pytest + httpx | `tests/integration/` |
| **T-38** | 4 | End-to-end demo validation: full pipeline from API call to risk score, SHAP output, and routed response | `tests/e2e/` |
| **T-39** | 4 | Performance check: confirm risk scoring ≤ 500 ms, routing ≤ 2 s under normal load | `reports/performance.md` |
| **T-40** | 4 | Update `README.md` with final setup instructions and model evaluation results | `README.md` |

**Total: 5 tasks**

---

## Detailed Task Notes

### T-22 — Unit Tests (`tests/unit/`)

Write pytest tests covering the following modules. Target **≥ 70 % line coverage** measured by `pytest --cov`.

#### Feature engineering (`tests/unit/test_features.py`)

```python
from app.ml.features import build_feature_vector, FEATURE_NAMES

def test_feature_vector_length():
    raw = { "hour_of_day": 22, "precipitation_mm": 5.0, ... }
    vec = build_feature_vector(raw)
    assert len(vec) == len(FEATURE_NAMES)

def test_night_indicator_set_at_midnight():
    raw = { "hour_of_day": 0, ... }
    vec = build_feature_vector(raw)
    assert vec[FEATURE_NAMES.index("night_indicator")] == 1

def test_rain_on_congestion_zero_when_dry():
    raw = { "precipitation_mm": 0.0, ... }
    vec = build_feature_vector(raw)
    assert vec[FEATURE_NAMES.index("rain_on_congestion")] == 0.0
```

#### Model inference (`tests/unit/test_inference.py`)

```python
from app.ml.inference import run_inference
import numpy as np

def test_inference_returns_score_in_range():
    dummy_vec = np.zeros(len(FEATURE_NAMES))
    score = run_inference(dummy_vec)
    assert 0 <= score <= 100

def test_high_rain_night_yields_elevated_score():
    # Set precipitation and night_indicator high
    raw = { "precipitation_mm": 20, "night_indicator": 1, ... }
    score = run_inference(build_feature_vector(raw))
    assert score >= 50
```

#### Routing (`tests/unit/test_astar.py`)

```python
from app.routing.astar import safe_route
import networkx as nx

def test_route_connects_origin_to_destination():
    G = small_test_graph()  # fixture: 10-node grid graph
    path = safe_route(G, origin_node=0, dest_node=9, alpha=0.5, risk_scores={})
    assert path[0] == 0 and path[-1] == 9

def test_high_alpha_avoids_high_risk_edges():
    G, risky_edge = build_graph_with_detour()
    path_safe = safe_route(G, ..., alpha=0.9, risk_scores={risky_edge: 95})
    path_fast = safe_route(G, ..., alpha=0.0, risk_scores={risky_edge: 95})
    assert risky_edge not in zip(path_safe, path_safe[1:])
    assert risky_edge in zip(path_fast, path_fast[1:])
```

Run with:
```bash
pytest tests/unit/ --cov=app --cov-report=term-missing
```

---

### T-23 — Integration Tests (`tests/integration/`)

Use `pytest` + `httpx.AsyncClient` against a running test instance. Use an in-memory SQLite database and a mock weather client for isolation.

**Fixtures (`tests/conftest.py`):**
```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c
```

**Tests to cover:**

| File | Endpoint | Scenarios |
|---|---|---|
| `test_health.py` | `GET /health` | Returns 200 with `status: ok` |
| `test_risk_segment.py` | `GET /v1/risk/segment` | Valid coords return score 0–100; missing params return 422 |
| `test_risk_heatmap.py` | `GET /v1/risk/heatmap` | Valid bbox returns GeoJSON FeatureCollection; empty bbox returns empty features list |
| `test_explain_segment.py` | `GET /v1/explain/segment` | Returns all SHAP values; `shap_values` sums approximately to `risk_score − expected_value` |
| `test_route.py` | `POST /v1/route/safe` | Valid origin/destination returns route geometry; `alpha=0` matches fastest route; `alpha=1` reduces avg risk vs fastest |

Run with:
```bash
pytest tests/integration/ -v
```

---

### T-38 — End-to-End Demo Validation (`tests/e2e/`)

With the full Docker stack running (`docker compose up`), execute a scripted demo that exercises the complete pipeline:

```
Step 1: GET /health                         → status: ok
Step 2: GET /v1/risk/segment?lat=34.05&lon=-118.24
                                            → risk_score in 0–100; shap_top_factors non-empty
Step 3: GET /v1/explain/segment?lat=34.05&lon=-118.24
                                            → shap_values for all features returned
Step 4: POST /v1/route/safe {alpha: 0.8}   → safe route geometry returned
                                            → risk_reduction_pct > 0
Step 5: GET /v1/risk/heatmap?bbox=...      → GeoJSON FeatureCollection with > 0 features
```

Write this as a pytest file (`tests/e2e/test_full_pipeline.py`) that uses `httpx` against `http://localhost:8000`. The test must pass with a fresh `docker compose up && scripts/seed_data.py` execution.

---

### T-39 — Performance Validation (`reports/performance.md`)

Measure latency for two critical paths under normal load (no concurrent users — single request at a time is sufficient for a research demo):

| Path | Target | How to measure |
|---|---|---|
| `GET /v1/risk/segment` | ≤ 500 ms end-to-end | `time curl http://localhost:8000/v1/risk/segment?lat=...` × 10 runs; report p50 and p95 |
| `POST /v1/route/safe` | ≤ 2 000 ms end-to-end | Same approach with a representative city-centre-to-suburb route |

Document results in `reports/performance.md`:
```markdown
## Performance Results

| Endpoint | p50 (ms) | p95 (ms) | Target met? |
|---|---|---|---|
| GET /v1/risk/segment | 187 | 312 | ✅ |
| POST /v1/route/safe  | 843 | 1210 | ✅ |
```

If either target is missed, open a GitHub Issue tagged `performance` describing the slow path and a proposed fix.

---

### T-40 — Final `README.md` Update

Update the existing `README.md` to reflect the final state of the project. Sections to add or update:

1. **Quick Start** — exact commands from `git clone` to a working demo:
   ```bash
   cp .env.example .env   # fill in OPENWEATHERMAP_API_KEY
   python scripts/download_data.py --city "Los Angeles, CA" --years 2021 2022
   python scripts/train_model.py --skip-tuning
   docker compose up --build -d
   python scripts/seed_data.py
   # API available at http://localhost:8000/docs
   ```
2. **Evaluation Results** — copy final AUROC, AUPRC, F1, ECE values from `reports/evaluation.md` (provided by M2).
3. **API Examples** — `curl` one-liners for each endpoint.
4. **Performance** — p50 / p95 latency table from `reports/performance.md`.
5. **Frontend** — note that a Leaflet.js interactive map frontend is available at `/map` after startup (maintained separately; see the `frontend/` directory).

---

## Deliverables Checklist

- [ ] `tests/unit/test_features.py` — ≥ 3 test cases for `build_feature_vector`
- [ ] `tests/unit/test_inference.py` — ≥ 2 test cases for `run_inference`
- [ ] `tests/unit/test_astar.py` — ≥ 2 test cases for `safe_route`
- [ ] `tests/integration/test_health.py`
- [ ] `tests/integration/test_risk_segment.py`
- [ ] `tests/integration/test_risk_heatmap.py`
- [ ] `tests/integration/test_explain_segment.py`
- [ ] `tests/integration/test_route.py`
- [ ] `tests/e2e/test_full_pipeline.py` — passes against live Docker stack
- [ ] `reports/performance.md` — p50/p95 latency for both endpoints
- [ ] `README.md` — updated with Quick Start, evaluation results, and API examples
- [ ] Coverage report: `pytest --cov=app` shows ≥ 70 % line coverage

---

## Dependencies from Others

| Needs | Provided by |
|---|---|
| `app/ml/features.py` | M1 (T-05) — for unit tests |
| `app/ml/inference.py` | M3 (T-12) — for unit tests |
| `app/routing/astar.py` | M4 (T-18) — for unit tests |
| All API endpoints running | M3 + M4 — for integration + e2e tests |
| `docker-compose.yml` | M4 (T-36) — for e2e tests |
| `reports/evaluation.md` | M2 (T-09) — for README update |
| `reports/performance.md` (draft) | Self (T-39) |

## What Others Depend On from M5

Nothing — M5 is the final quality gate.
