# M1 Data Engineer — Implementation Summary

**Status:** T-01 ✓, T-02 (in progress), T-03-T-05 ready to run

---

## What We've Built

### T-01: Download NHTSA FARS Data ✓
- **Script:** `scripts/download_fars_data.py`
- **Output:** `data/raw/fars_2021.csv`, `data/raw/fars_2022.csv`, `data/raw/fars_2023.csv`
- **Records Generated:** 6,000 sample accident records across 3 years
- **Schema:** FARS_ID, YEAR, MONTH, DAY, HOUR, LATITUDE, LONGITUDE, SEVERITY, FATALITIES, INJURIES, NUM_VEHICLES, WEATHER, ROAD_SURFACE

**Note:** For production, download real FARS data from NHTSA portal (https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars)

### T-02: Download OSM Road Network
- **Script:** `scripts/download_osm_network.py`
- **Output:** `data/raw/road_network.graphml`
- **Format:** NetworkX GraphML (preserves all edge attributes)
- **Status:** Running — downloads ~50K+ road segments for Los Angeles

**Command:**
```bash
python scripts/download_osm_network.py \
  --place "Los Angeles, California, USA" \
  --network-type drive \
  --output data/raw \
  --validate
```

### T-03: Snap Accidents to Road Segments (Ready)
- **Script:** `scripts/snap_accidents.py`
- **Algorithm:** OSMnx nearest_edges() with 50m distance threshold
- **Output:** `data/processed/accidents_snapped.parquet`
- **Expected:** ~5,500 accidents within threshold

**Command:**
```bash
python scripts/snap_accidents.py \
  --fars-dir data/raw \
  --network data/raw/road_network.graphml \
  --threshold 50.0 \
  --output data/processed/accidents_snapped.parquet
```

### T-04: Compute Historical Accident Rates (Ready)
- **Script:** `scripts/compute_accident_rates.py`
- **Formula:** `rate = count / (length_km) / years`
- **Output:** `data/processed/segment_rates.parquet`
- **Columns:** osmid, historical_accident_rate

**Command:**
```bash
python scripts/compute_accident_rates.py \
  --snapped data/processed/accidents_snapped.parquet \
  --network data/raw/road_network.graphml \
  --years 3 \
  --output data/processed/segment_rates.parquet
```

### T-05: Feature Engineering Pipeline ✓
- **Module:** `app/ml/features.py`
- **Features:** 12 engineered features for XGBoost
- **Importable:** `from app.ml.features import build_feature_vector, FEATURE_NAMES`
- **Works for:** Both training (batch) and inference (real-time API)

**Feature List:**
1. `hour_of_day` (0-23)
2. `day_of_week` (0-6)
3. `month` (1-12)
4. `night_indicator` (0-1, True if 20:00-06:00)
5. `road_class` (0-5, motorway to residential)
6. `speed_limit_kmh` (0-200)
7. `precipitation_mm` (0-100)
8. `visibility_km` (0-50)
9. `wind_speed_ms` (0-50)
10. `temperature_c` (-50 to 60)
11. `rain_on_congestion` (derived interaction)
12. `historical_accident_rate` (0-100, normalized)

**Usage Example:**
```python
from datetime import datetime
from app.ml.features import build_feature_vector, FEATURE_NAMES

raw = {
    'timestamp': datetime(2023, 6, 15, 14, 30),
    'latitude': 34.05,
    'longitude': -118.24,
    'highway': 'secondary',
    'speed_limit_kmh': 50.0,
    'precipitation_mm': 5.0,
    'visibility_km': 8.0,
    'wind_speed_ms': 3.0,
    'temperature_c': 25.0,
    'historical_accident_rate': 0.5,
}

vector = build_feature_vector(raw)  # Returns np.ndarray of length 12
print(f"Features: {dict(zip(FEATURE_NAMES, vector))}")
```

---

## Running the Complete Pipeline

All 5 tasks can be executed in sequence:

```bash
# Option 1: Run individual scripts (for debugging)
python scripts/download_fars_data.py --city "Los Angeles" --state "CA" --years 2021 2022 2023
python scripts/download_osm_network.py --place "Los Angeles, California, USA" --validate
python scripts/snap_accidents.py --fars-dir data/raw --network data/raw/road_network.graphml
python scripts/compute_accident_rates.py --snapped data/processed/accidents_snapped.parquet --network data/raw/road_network.graphml

# Option 2: Run all tasks with orchestration script
python data_prepare.py --city "Los Angeles, CA" --years 2021 2022 2023
```

---

## Data Flow Diagram

```
FARS CSV files ──┐
                 ├──> Snap Accidents ──> accidents_snapped.parquet ──┐
Road Network ────┘                                                   ├──> Training Data
                                                                      │    (M2 ML Engineer)
                 historical_accident_rate ────────────────────────┘

Feature Engineering Pipeline (app/ml/features.py) ───────> API Inference (M3)
```

---

## Deliverables Checklist

- [x] `data/raw/fars_*.csv` (3 years of sample FARS data)
- [ ] `data/raw/road_network.graphml` (OSM network — in progress)
- [ ] `data/processed/accidents_snapped.parquet` (Ready to run)
- [ ] `data/processed/segment_rates.parquet` (Ready to run)
- [x] `app/ml/features.py` (Complete and tested)
- [ ] `scripts/download_data.py` (T-33, Phase 4)
- [ ] `scripts/seed_data.py` (T-35, Phase 4)

---

## Dependencies

All scripts require the following Python packages (see `requirements.txt`):
- pandas, numpy, geopandas, shapely (data processing)
- osmnx (road network download)
- scikit-learn (future ML work)

Install: `pip install -r requirements.txt`

---

## Testing

Unit tests for feature engineering:
```bash
pytest tests/unit/test_features.py -v
```

Integration tests for full pipeline:
```bash
pytest tests/integration/test_data_pipeline.py -v
```

---

## Next: M2 ML Engineer

Once M1 completes, M2 will:
1. Load `data/processed/accidents_snapped.parquet`
2. Use `app/ml/features.py` to build feature vectors
3. Create chronological train/val/test splits
4. Train XGBoost model (T-06 to T-11)
5. Save `models/model.pkl`

---

## Notes

- **FARS Data:** Using sample data for MVP. Swap with real NHTSA data for production.
- **Geographic Scope:** Los Angeles, California. Can be changed by modifying `--place` argument.
- **Time Range:** 2021-2023. Can be extended by downloading more years.
- **Snapping Threshold:** 50m is conservative. Lower values are more strict, higher values may snap to wrong segments.

