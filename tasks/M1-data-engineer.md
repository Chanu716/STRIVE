# STRIVE — Task Sheet: M1 Data Engineer

**Member role:** Data Engineer  
**Depends on:** —  
**Produces for:** M2 (feature parquet, feature pipeline), M3 (database seed)

---

## Responsibility

Collect and clean all raw data, match accidents to road segments, build the shared feature engineering pipeline, and automate data scripts. All downstream modelling and API work depends on these deliverables.

---

## Task List

| ID | Phase | Description | Output |
|---|---|---|---|
| **T-01** | 1 | Download NHTSA FARS CSV data for target city (2+ years) | `data/raw/fars_*.csv` |
| **T-02** | 1 | Download OSM road network via OSMnx for target city | `data/raw/road_network.graphml` |
| **T-03** | 1 | Match FARS accident records to nearest OSM road segment (≤ 50 m snap threshold) | `data/processed/accidents_snapped.parquet` |
| **T-04** | 1 | Compute `historical_accident_rate` per road segment (incidents / km / year) | `data/processed/segment_rates.parquet` |
| **T-05** | 1 | Implement `app/ml/features.py` — full feature engineering pipeline | `app/ml/features.py` |
| **T-33** | 4 | Write `scripts/download_data.py` — automated NHTSA + OSM data download | `scripts/download_data.py` |
| **T-35** | 4 | Write `scripts/seed_data.py` — populate PostgreSQL `road_segments` and `accidents` tables | `scripts/seed_data.py` |

**Total: 7 tasks**

---

## Detailed Task Notes

### T-01 — Download NHTSA FARS Data

- Source: [NHTSA FARS downloads](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars)
- Download the **accident-level CSV** files for the target city for at least 2 full calendar years.
- Filter to records where `CITY` and `STATE` match the target city.
- Store raw files under `data/raw/fars_<year>.csv` (do not alter the original columns).

### T-02 — Download OSM Road Network

```python
import osmnx as ox
G = ox.graph_from_place("Los Angeles, California, USA", network_type="drive")
ox.save_graphml(G, filepath="data/raw/road_network.graphml")
```

- Use `network_type="drive"` to include only driveable roads.
- Ensure the graph includes `speed_kph`, `highway`, and `length` edge attributes.

### T-03 — Snap Accidents to Road Segments

- Load `road_network.graphml` with `ox.load_graphml()`.
- For each accident lat/lon, use `ox.nearest_edges()` to find the closest edge.
- Keep only matches within ≤ 50 m to avoid false snapping.
- Output columns: `osmid`, `accident_id`, `timestamp`, `severity`, `snap_distance_m`.

### T-04 — Historical Accident Rate

- Join snapped accidents to edges; count incidents per edge per year.
- Normalise by edge length (km): `rate = count / (length_m / 1000) / years`.
- Output: `data/processed/segment_rates.parquet` with columns `osmid`, `historical_accident_rate`.

### T-05 — Feature Engineering Pipeline (`app/ml/features.py`)

The pipeline must be **importable both from training scripts (M2) and from the live API (M3)**. It accepts a dict of raw inputs and returns a fixed-length feature vector.

**Features to produce:**

| Group | Feature name | Type |
|---|---|---|
| Time | `hour_of_day` | int 0–23 |
| Time | `day_of_week` | int 0–6 |
| Time | `month` | int 1–12 |
| Time | `night_indicator` | bool (1 if 20:00–06:00) |
| Road | `road_class` | int (motorway=0, primary=1, secondary=2, …) |
| Road | `speed_limit_kmh` | float |
| Weather | `precipitation_mm` | float |
| Weather | `visibility_km` | float |
| Weather | `wind_speed_ms` | float |
| Weather | `temperature_c` | float |
| Derived | `rain_on_congestion` | `precipitation_mm × (1 − speed_ratio)` |
| History | `historical_accident_rate` | float |

Expose a function `build_feature_vector(raw: dict) -> np.ndarray` and a constant `FEATURE_NAMES: list[str]`.

### T-33 — `scripts/download_data.py`

Command-line script that:
1. Downloads FARS CSV files for a configurable year range and city.
2. Downloads the OSM road graph for a configurable place name.
3. Prints progress and final file sizes.

```bash
python scripts/download_data.py --city "Los Angeles, CA" --years 2021 2022 2023
```

### T-35 — `scripts/seed_data.py`

Populates the PostgreSQL database from processed parquet files:
- Inserts all rows from `segment_rates.parquet` into the `road_segments` table.
- Inserts all rows from `accidents_snapped.parquet` into the `accidents` table.
- Uses SQLAlchemy Core for bulk insert (`execute_many`).

---

## Deliverables Checklist

- [ ] `data/raw/fars_<year>.csv` (at least 2 years)
- [ ] `data/raw/road_network.graphml`
- [ ] `data/processed/accidents_snapped.parquet`
- [ ] `data/processed/segment_rates.parquet`
- [ ] `data/processed/features.parquet` (full training dataset with all features + `incident` label)
- [ ] `app/ml/features.py` (importable; passes unit tests written by M5)
- [ ] `scripts/download_data.py`
- [ ] `scripts/seed_data.py`

---

## Dependencies from Others

| Needs | Provided by |
|---|---|
| PostgreSQL schema (table definitions) | M3 (T-13 Alembic migrations) |

## What Others Depend On from M1

| Deliverable | Used by |
|---|---|
| `data/processed/features.parquet` | M2 (model training) |
| `app/ml/features.py` | M2 (training), M3 (inference) |
| `scripts/seed_data.py` | M4 (Docker setup, end-to-end test) |
