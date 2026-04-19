#!/usr/bin/env python3
"""
T-05: Feature Engineering Pipeline

This module is the core of both training (M2) and inference (M3).
It transforms raw inputs (weather, time, road attributes) into a fixed-length
feature vector for XGBoost inference.

The pipeline MUST be importable as:
    from app.ml.features import build_feature_vector, FEATURE_NAMES

And work both in training (batch) and inference (single sample) modes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Feature Schema
# ============================================================================

FEATURE_NAMES: List[str] = [
    # Time features
    'hour_of_day',
    'day_of_week',
    'month',
    'night_indicator',
    # Road features
    'road_class',
    'speed_limit_kmh',
    # Weather features
    'precipitation_mm',
    'visibility_km',
    'wind_speed_ms',
    'temperature_c',
    # Derived features
    'rain_on_congestion',
    'historical_accident_rate',
]

# Feature index mapping (for easy reference)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Expected ranges for validation
FEATURE_RANGES = {
    'hour_of_day': (0, 23),
    'day_of_week': (0, 6),
    'month': (1, 12),
    'night_indicator': (0, 1),
    'road_class': (0, 5),
    'speed_limit_kmh': (0, 200),
    'precipitation_mm': (0, 100),
    'visibility_km': (0, 50),
    'wind_speed_ms': (0, 50),
    'temperature_c': (-50, 60),
    'rain_on_congestion': (0, 1),
    'historical_accident_rate': (0, 100),
}

# Road class mapping (OSM highway classification)
ROAD_CLASS_MAPPING = {
    'motorway': 0,
    'trunk': 0,
    'primary': 1,
    'secondary': 2,
    'tertiary': 3,
    'residential': 4,
    'unclassified': 4,
}


# ============================================================================
# Feature Extractors (modular functions)
# ============================================================================

def extract_time_features(timestamp: datetime) -> Dict[str, float]:
    """
    Extract time-based features from a timestamp.

    Args:
        timestamp: datetime object or ISO string

    Returns:
        Dict with hour_of_day, day_of_week, month, night_indicator
    """
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    month = timestamp.month

    # Night indicator: True if between 20:00 (20) and 06:00 (6)
    night_indicator = 1.0 if (hour >= 20 or hour < 6) else 0.0

    return {
        'hour_of_day': float(hour),
        'day_of_week': float(day_of_week),
        'month': float(month),
        'night_indicator': night_indicator,
    }


def extract_road_features(road_attrs: Dict) -> Dict[str, float]:
    """
    Extract road attribute features.

    Args:
        road_attrs: Dict with 'highway', 'speed_limit_kmh' keys

    Returns:
        Dict with road_class, speed_limit_kmh
    """
    # Road class
    highway_type = road_attrs.get('highway', 'unclassified')
    road_class = float(ROAD_CLASS_MAPPING.get(highway_type, 4))

    # Speed limit (default to 50 km/h if missing)
    speed_limit = float(road_attrs.get('speed_limit_kmh', 50))

    return {
        'road_class': road_class,
        'speed_limit_kmh': speed_limit,
    }


def extract_weather_features(weather: Dict) -> Dict[str, float]:
    """
    Extract weather features.

    Args:
        weather: Dict with precipitation_mm, visibility_km, wind_speed_ms, temperature_c

    Returns:
        Dict with weather features
    """
    return {
        'precipitation_mm': float(weather.get('precipitation_mm', 0.0)),
        'visibility_km': float(weather.get('visibility_km', 10.0)),
        'wind_speed_ms': float(weather.get('wind_speed_ms', 0.0)),
        'temperature_c': float(weather.get('temperature_c', 20.0)),
    }


def extract_historical_features(historical: Dict) -> Dict[str, float]:
    """
    Extract historical accident rate.

    Args:
        historical: Dict with 'historical_accident_rate' key

    Returns:
        Dict with historical_accident_rate (normalized to 0-100 scale)
    """
    rate = float(historical.get('historical_accident_rate', 0.0))
    # Normalize to 0-100 scale (rates rarely exceed 10 incidents/km/year)
    rate_normalized = min(rate * 10, 100.0)  # Cap at 100
    return {'historical_accident_rate': rate_normalized}


def compute_derived_features(
    weather: Dict[str, float],
    road: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute derived features that combine multiple inputs.

    Args:
        weather: Weather features (precipitation_mm)
        road: Road features (speed_limit_kmh)

    Returns:
        Dict with rain_on_congestion
    """
    # rain_on_congestion: interaction between rain and road speed
    # High value when raining AND moving slowly (congestion indicator)
    precipitation = weather.get('precipitation_mm', 0.0)
    speed_limit = road.get('speed_limit_kmh', 50.0)

    # Speed ratio: how much of the speed limit are we achieving?
    # We use a simple model: assume average traffic speed = 0.6 * speed_limit
    # When speed is low (congestion), speed_ratio is low, so interaction is high
    speed_ratio = max(0.6 * speed_limit / 100.0, 0.1)  # Clamp to avoid division issues
    rain_on_congestion = (precipitation / 100.0) * (1.0 - min(speed_ratio, 1.0))

    return {'rain_on_congestion': rain_on_congestion}


# ============================================================================
# Main Pipeline Function
# ============================================================================

def build_feature_vector(
    raw: Dict,
    validate: bool = True
) -> np.ndarray:
    """
    Build a fixed-length feature vector from raw inputs.

    This function is used in both training and inference:
    - Training: Batch processing via pandas (raw is dict-like)
    - Inference: Real-time API calls (raw is a single dict)

    Args:
        raw: Dict with keys:
            - 'timestamp': ISO string or datetime
            - 'latitude': float
            - 'longitude': float
            - 'road_class': int or 'highway' string
            - 'speed_limit_kmh': float
            - 'precipitation_mm': float
            - 'visibility_km': float
            - 'wind_speed_ms': float
            - 'temperature_c': float
            - 'historical_accident_rate': float
        validate: If True, validate output ranges

    Returns:
        Numpy array of length 12 in the order of FEATURE_NAMES
    """
    try:
        # Extract time features
        time_features = extract_time_features(raw['timestamp'])

        # Extract road features
        road_features = extract_road_features({
            'highway': raw.get('highway', 'unclassified'),
            'speed_limit_kmh': raw.get('speed_limit_kmh', 50),
        })

        # Extract weather features
        weather_features = extract_weather_features({
            'precipitation_mm': raw.get('precipitation_mm', 0.0),
            'visibility_km': raw.get('visibility_km', 10.0),
            'wind_speed_ms': raw.get('wind_speed_ms', 0.0),
            'temperature_c': raw.get('temperature_c', 20.0),
        })

        # Extract historical features
        historical_features = extract_historical_features({
            'historical_accident_rate': raw.get('historical_accident_rate', 0.0),
        })

        # Compute derived features
        derived_features = compute_derived_features(weather_features, road_features)

        # Merge all features
        all_features = {
            **time_features,
            **road_features,
            **weather_features,
            **derived_features,
            **historical_features,
        }

        # Build vector in the correct order
        feature_vector = np.array([all_features[name] for name in FEATURE_NAMES])

        # Validate
        if validate:
            validate_feature_vector(feature_vector)

        return feature_vector

    except KeyError as e:
        raise ValueError(f"Missing required key in raw input: {e}")
    except Exception as e:
        raise ValueError(f"Error building feature vector: {e}")


def validate_feature_vector(vector: np.ndarray) -> bool:
    """
    Validate that a feature vector is within expected ranges.

    Args:
        vector: Numpy array of features

    Returns:
        True if valid, raises ValueError otherwise
    """
    if len(vector) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(vector)}")

    for i, name in enumerate(FEATURE_NAMES):
        value = vector[i]
        min_val, max_val = FEATURE_RANGES[name]

        if not (min_val <= value <= max_val):
            logger.warning(
                f"Feature {name} (idx {i}) out of range: "
                f"{value} not in [{min_val}, {max_val}]"
            )

    return True


def build_feature_dataframe(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Batch build features from a DataFrame (for model training).

    Args:
        raw_data: DataFrame with columns matching build_feature_vector inputs

    Returns:
        DataFrame with columns from FEATURE_NAMES
    """
    logger.info(f"Building features for {len(raw_data)} rows...")

    features = []
    errors = 0

    for idx, row in raw_data.iterrows():
        try:
            raw_dict = row.to_dict()
            feature_vector = build_feature_vector(raw_dict, validate=False)
            features.append(feature_vector)
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"Row {idx}: {e}")

    if errors > 0:
        logger.warning(f"Total errors: {errors} / {len(raw_data)}")

    features_array = np.stack(features) if features else np.empty((0, len(FEATURE_NAMES)))
    features_df = pd.DataFrame(features_array, columns=FEATURE_NAMES)

    logger.info(f"✓ Built {len(features_df)} feature vectors")
    return features_df


# ============================================================================
# Utility Functions for Training Data
# ============================================================================

def create_training_dataset(
    snapped_accidents: pd.DataFrame,
    road_attributes: pd.DataFrame,
    weather_data: Optional[pd.DataFrame] = None,
    label_col: str = 'incident'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a training dataset with features and labels.

    This combines:
    - Snapped accidents (positive samples)
    - Random non-accident segments (negative samples)
    - Road attributes
    - Weather data (if provided, or use defaults)

    Args:
        snapped_accidents: DataFrame with accident records
        road_attributes: DataFrame with segment attributes
        weather_data: Optional weather data by timestamp
        label_col: Column name for the label

    Returns:
        Tuple of (features_df, labels_series)
    """
    # For MVP: just use accident records
    # In production, you'd create negative samples (non-accidents)

    logger.info(f"Creating training dataset from {len(snapped_accidents)} accidents...")

    # Add default weather if not provided
    if weather_data is None:
        snapped_accidents['precipitation_mm'] = 0.0
        snapped_accidents['visibility_km'] = 10.0
        snapped_accidents['wind_speed_ms'] = 0.0
        snapped_accidents['temperature_c'] = 20.0

    # Build features
    features_df = build_feature_dataframe(snapped_accidents)

    # Add label (1 = incident, 0 = no incident)
    labels = pd.Series(np.ones(len(features_df)), index=features_df.index)

    logger.info(f"✓ Created dataset: {len(features_df)} samples")
    return features_df, labels


if __name__ == "__main__":
    # Test the feature pipeline
    print("Testing Feature Engineering Pipeline...")
    print(f"Feature names: {FEATURE_NAMES}")
    print(f"Number of features: {len(FEATURE_NAMES)}\n")

    # Test with sample data
    test_input = {
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

    feature_vector = build_feature_vector(test_input)
    print(f"Test feature vector (shape {feature_vector.shape}):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:30} = {feature_vector[i]:10.4f}")
