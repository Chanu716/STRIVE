#!/usr/bin/env python3
"""
T-06: Chronological Train / Val / Test Split (70 / 15 / 15)

Reconstructs a balanced dataset (positives + negatives) from features.parquet
and accidents_snapped.parquet, adds a timestamp column, sorts chronologically,
then splits without shuffling to avoid temporal leakage.

Output: data/splits/train.parquet, val.parquet, test.parquet
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

SEED = 42
np.random.seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
features = pd.read_parquet("data/processed/features.parquet")
accidents = pd.read_parquet("data/processed/accidents_snapped.parquet")

print(f"  features shape : {features.shape}")
print(f"  accidents shape: {accidents.shape}")

# ── Build timestamp from accidents (year/month/day/hour) ─────────────────────
# accidents rows align 1-to-1 with features rows (both 116 329 rows)
accidents = accidents.reset_index(drop=True)
features  = features.reset_index(drop=True)

accidents["timestamp"] = pd.to_datetime(
    accidents[["year", "month", "day", "hour"]].rename(
        columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
    )
)
features["timestamp"] = accidents["timestamp"]

# ── Generate negative samples ─────────────────────────────────────────────────
# Strategy: for each positive, create one synthetic negative by randomising
# weather/time while keeping road attributes and historical rate the same.
# This gives a balanced 50/50 dataset.
print("Generating negative samples...")

neg = features.copy()
neg["incident"] = 0.0

# Randomise time features
neg["hour_of_day"]    = np.random.randint(0, 24,  size=len(neg)).astype(float)
neg["day_of_week"]    = np.random.randint(0, 7,   size=len(neg)).astype(float)
neg["month"]          = np.random.randint(1, 13,  size=len(neg)).astype(float)
neg["night_indicator"]= (neg["hour_of_day"].isin(range(20, 24)) |
                          neg["hour_of_day"].isin(range(0, 6))).astype(float)

# Randomise weather (mostly clear conditions for negatives)
neg["precipitation_mm"]  = np.random.exponential(0.5, size=len(neg)).clip(0, 50)
neg["visibility_km"]     = np.random.uniform(5, 15, size=len(neg))
neg["wind_speed_ms"]     = np.random.uniform(0, 10, size=len(neg))
neg["temperature_c"]     = np.random.uniform(5, 35, size=len(neg))

# Recompute derived feature
speed_ratio = (0.6 * neg["speed_limit_kmh"] / 100.0).clip(0.1, 1.0)
neg["rain_on_congestion"] = (neg["precipitation_mm"] / 100.0) * (1.0 - speed_ratio)

# Assign timestamps spread across the same date range as positives
date_range = pd.date_range(
    start=features["timestamp"].min(),
    end=features["timestamp"].max(),
    periods=len(neg)
)
neg["timestamp"] = date_range

# ── Combine and sort chronologically ─────────────────────────────────────────
df = pd.concat([features, neg], ignore_index=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"  Combined shape : {df.shape}")
print(f"  Label balance  : {df['incident'].value_counts().to_dict()}")

# ── Split 70 / 15 / 15 ───────────────────────────────────────────────────────
n = len(df)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

train = df.iloc[:n_train]
val   = df.iloc[n_train : n_train + n_val]
test  = df.iloc[n_train + n_val :]

print(f"\nSplit sizes:")
print(f"  train : {len(train):>7,}  ({len(train)/n:.1%})")
print(f"  val   : {len(val):>7,}  ({len(val)/n:.1%})")
print(f"  test  : {len(test):>7,}  ({len(test)/n:.1%})")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data/splits", exist_ok=True)
train.to_parquet("data/splits/train.parquet", index=False)
val.to_parquet("data/splits/val.parquet",     index=False)
test.to_parquet("data/splits/test.parquet",   index=False)

print("\n✓ Splits saved to data/splits/")
