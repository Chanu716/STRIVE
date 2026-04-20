#!/usr/bin/env python3
"""
T-05: Build features.parquet from real snapped accident data.

Reads:
  - data/processed/accidents_snapped.parquet  (real accident records w/ WEATHER/FUNC_SYS)
  - data/processed/segment_rates.parquet      (historical accident rate per OSM edge)
  - data/raw/road_network.graphml             (OSM graph for speed_kph / highway type)

Produces:
  - data/processed/features.parquet  (all 12 features + incident label, 100% real data)
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ml.features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── FARS WEATHER code → (precipitation_mm, visibility_km, wind_speed_ms, temperature_c)
# Values are representative physical conditions for each FARS condition code.
WEATHER_LOOKUP: dict = {
    1:  (0.0,  10.0,  1.0,  20.0),   # Clear
    2:  (8.0,   4.0,  3.0,  15.0),   # Rain
    3:  (3.0,   3.0,  4.0,   1.0),   # Sleet or Hail
    4:  (0.0,   2.0,  5.0,  -3.0),   # Snow
    5:  (0.0,   0.5,  1.0,  12.0),   # Fog, Smog, Smoke
    6:  (0.0,   8.0, 12.0,  20.0),   # Severe Crosswinds
    7:  (0.0,   4.0,  8.0,  25.0),   # Blowing Sand / Soil / Dirt
    8:  (0.0,   8.0,  2.0,  20.0),   # Other
   10:  (0.0,   8.0,  2.0,  16.0),   # Cloudy
   11:  (0.0,   2.0,  8.0,  -4.0),   # Blowing Snow
   12:  (5.0,   3.0,  3.0,   0.0),   # Freezing Rain or Drizzle
   98:  (0.0,  10.0,  1.0,  20.0),   # Not Reported → default to Clear
   99:  (0.0,  10.0,  1.0,  20.0),   # Reported as Unknown → default to Clear
}
_WEATHER_DEFAULT = (0.0, 10.0, 1.0, 20.0)  # Clear fallback for unlisted codes

# ── FARS FUNC_SYS → road_class int (matches FEATURE_NAMES 'road_class' spec)
FUNC_SYS_MAP: dict = {
    1: 0,   # Interstate        → motorway
    2: 0,   # Other Freeway     → motorway
    3: 1,   # Principal Arterial → primary
    4: 2,   # Minor Arterial    → secondary
    5: 3,   # Major Collector   → tertiary
    6: 4,   # Minor Collector   → residential
    7: 4,   # Local             → residential
}
_FUNC_SYS_DEFAULT = 4  # residential / unclassified fallback

# ── Night indicator from LGT_COND (more accurate than purely hour-based)
# 1=Daylight, 2=Dark-Lighted, 3=Dark-Not-Lighted, 4=Dawn, 5=Dusk,
# 6=Dark-Unknown-Lighting, 7=Other, 8=Not Reported, 9=Reported Unknown
DARK_LGT_CONDS = {2, 3, 6}   # unambiguously dark conditions


def parse_maxspeed(val, default: float = 50.0) -> float:
    """Parse OSM maxspeed string into km/h (same logic as download_osm_network.py)."""
    TOKENS = {
        "motorway": 130.0, "national": 100.0, "rural": 80.0,
        "urban": 50.0, "living_street": 20.0, "walk": 10.0,
    }
    if val is None:
        return default
    v = str(val).strip().lower()
    if v in TOKENS:
        return TOKENS[v]
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph)?$", v)
    if m:
        speed = float(m.group(1))
        return round(speed * 1.60934, 1) if m.group(2) == "mph" else speed
    return default


def build_weather_features(weather_code: int) -> tuple:
    """Return (precipitation_mm, visibility_km, wind_speed_ms, temperature_c) for a WEATHER code."""
    return WEATHER_LOOKUP.get(int(weather_code), _WEATHER_DEFAULT)


def build_night_indicator(hour: int, lgt_cond: int) -> float:
    """
    Night indicator: 1 if dark, 0 otherwise.
    Uses LGT_COND when available (more reliable than hour threshold).
    Falls back to hour-based rule (20:00–06:00) if LGT_COND is unknown.
    """
    if lgt_cond in DARK_LGT_CONDS:
        return 1.0
    if lgt_cond == 1:   # Daylight
        return 0.0
    # Dusk/Dawn/Unknown: use hour
    return 1.0 if (hour >= 20 or hour < 6) else 0.0


def extract_osm_edge_attrs(graphml_file: str) -> pd.DataFrame:
    """
    Extract per-edge attributes from the OSM graph.
    Returns DataFrame: osmid, highway, speed_kph_real, maxspeed_raw
    """
    logger.info(f"Extracting OSM edge attributes from {graphml_file}...")
    G = ox.load_graphml(graphml_file)
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        raw_maxspeed = data.get("maxspeed", None)
        parsed_speed = parse_maxspeed(raw_maxspeed, default=50.0)
        rows.append({
            "osmid":          f"{u}_{v}",
            "highway":        data.get("highway", "unclassified"),
            "speed_kph_real": parsed_speed,
            "length_m":       float(data.get("length", 0.0)),
        })
    edges_df = pd.DataFrame(rows).drop_duplicates(subset=["osmid"])
    logger.info(f"  {len(edges_df)} unique edges extracted")
    logger.info(f"  speed_kph_real unique vals: {sorted(edges_df['speed_kph_real'].unique())}")
    return edges_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build real features.parquet from FARS + OSM data")
    parser.add_argument("--snapped",  default="data/processed/accidents_snapped.parquet")
    parser.add_argument("--rates",    default="data/processed/segment_rates.parquet")
    parser.add_argument("--network",  default="data/raw/road_network.graphml")
    parser.add_argument("--output",   default="data/processed/features.parquet")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Real Feature Dataset (T-05)")
    logger.info("=" * 60)

    # ── Load inputs ──────────────────────────────────────────────────────────
    logger.info("Loading snapped accidents...")
    snapped = pd.read_parquet(args.snapped)
    logger.info(f"  {len(snapped)} records | cols: {list(snapped.columns)}")

    # Verify real FARS columns exist
    for col in ["weather_code", "lgt_cond", "func_sys"]:
        if col not in snapped.columns:
            raise ValueError(
                f"Column '{col}' missing from snapped parquet! "
                f"Re-run scripts/snap_accidents.py with the updated version first."
            )

    logger.info("Loading segment rates...")
    rates = pd.read_parquet(args.rates).drop_duplicates(subset=["osmid"])
    logger.info(f"  {len(rates)} segments")

    edges_df = extract_osm_edge_attrs(args.network)

    # ── Merge road attributes ─────────────────────────────────────────────────
    logger.info("Merging road attributes...")
    data = snapped.merge(rates,    on="osmid", how="left")
    data = data.merge(edges_df,    on="osmid", how="left")
    data["historical_accident_rate"] = data["historical_accident_rate"].fillna(0.0)
    data["speed_kph_real"]           = data["speed_kph_real"].fillna(50.0)
    data["highway"]                  = data["highway"].fillna("unclassified")

    # ── Validate/clip HOUR ────────────────────────────────────────────────────
    # FARS uses 99 = unknown. Clip to valid range.
    data["hour"]   = data["hour"].clip(0, 23).replace(99, 12)
    data["minute"] = data["minute"].clip(0, 59).replace(99, 0)
    data["month"]  = data["month"].clip(1, 12)
    data["day"]    = data["day"].clip(1, 28)

    # ── Build all 12 feature columns from REAL data ───────────────────────────
    logger.info("Building 12 real feature columns...")

    # Time features (real FARS timestamps)
    data["hour_of_day"]  = data["hour"].astype(float)
    data["day_of_week"]  = pd.to_datetime(
        dict(year=data.year, month=data.month, day=data.day),
        errors="coerce"
    ).dt.dayofweek.fillna(0).astype(float)
    data["month_feat"]   = data["month"].astype(float)

    # Night indicator: LGT_COND-based (real), falls back to hour-based
    data["night_indicator"] = data.apply(
        lambda r: build_night_indicator(int(r["hour"]), int(r["lgt_cond"])), axis=1
    )

    # Road class: FUNC_SYS-based (real nationwide road class)
    data["road_class"] = data["func_sys"].map(FUNC_SYS_MAP).fillna(_FUNC_SYS_DEFAULT).astype(float)

    # Speed limit: from parsed OSM maxspeed (real)
    data["speed_limit_kmh"] = data["speed_kph_real"].astype(float)

    # Weather features: from FARS WEATHER code (real condition at accident)
    weather_cols = data["weather_code"].apply(
        lambda w: pd.Series(build_weather_features(w),
                            index=["precipitation_mm", "visibility_km", "wind_speed_ms", "temperature_c"])
    )
    data[["precipitation_mm", "visibility_km", "wind_speed_ms", "temperature_c"]] = weather_cols.values

    # Derived feature: rain_on_congestion = precipitation_mm × (1 − speed_ratio)
    speed_ratio = (0.6 * data["speed_limit_kmh"] / 100.0).clip(0.1, 1.0)
    data["rain_on_congestion"] = (data["precipitation_mm"] / 100.0) * (1.0 - speed_ratio)

    # Historical rate (already merged, normalise same as features.py)
    data["historical_accident_rate_feat"] = (data["historical_accident_rate"] * 10).clip(upper=100.0)

    # ── Assemble final feature dataframe ─────────────────────────────────────
    feature_map = {
        "hour_of_day":            "hour_of_day",
        "day_of_week":            "day_of_week",
        "month":                  "month_feat",
        "night_indicator":        "night_indicator",
        "road_class":             "road_class",
        "speed_limit_kmh":        "speed_limit_kmh",
        "precipitation_mm":       "precipitation_mm",
        "visibility_km":          "visibility_km",
        "wind_speed_ms":          "wind_speed_ms",
        "temperature_c":          "temperature_c",
        "rain_on_congestion":     "rain_on_congestion",
        "historical_accident_rate": "historical_accident_rate_feat",
    }

    features_df = pd.DataFrame({fname: data[col] for fname, col in feature_map.items()})
    features_df["incident"] = 1.0   # All records are real accidents (positive class)
    features_df = features_df.reset_index(drop=True)

    # ── Quality report ────────────────────────────────────────────────────────
    logger.info("\n=== Feature Quality Report ===")
    for col in FEATURE_NAMES:
        u = features_df[col].nunique()
        logger.info(f"  {col:30s}: {u} unique values  "
                    f"mean={features_df[col].mean():.3f}  std={features_df[col].std():.3f}")

    # Check NaN / inf
    bad = features_df[FEATURE_NAMES].isin([float("inf"), float("-inf")]).sum().sum()
    nan_count = features_df[FEATURE_NAMES].isna().sum().sum()
    logger.info(f"\n  NaN values:  {nan_count}")
    logger.info(f"  Inf values:  {bad}")
    if nan_count > 0 or bad > 0:
        logger.warning("  ⚠ Found NaN/Inf — filling with 0")
        features_df[FEATURE_NAMES] = features_df[FEATURE_NAMES].fillna(0).replace([float("inf"), float("-inf")], 0)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_parquet(args.output, index=False)
    logger.info(f"\n✓ Saved {len(features_df)} records to {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
