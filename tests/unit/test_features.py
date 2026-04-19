#!/usr/bin/env python3
"""
Unit tests for M1 Data Engineer components.

Tests feature engineering pipeline and data validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ml.features import (
    build_feature_vector,
    FEATURE_NAMES,
    FEATURE_INDEX,
    extract_time_features,
    extract_road_features,
    extract_weather_features,
    validate_feature_vector,
)


class TestFeatureExtraction:
    """Tests for individual feature extractors."""

    def test_extract_time_features_day(self):
        """Test time feature extraction for daytime."""
        ts = datetime(2023, 6, 15, 14, 30)  # Thursday, 2:30 PM
        features = extract_time_features(ts)

        assert features['hour_of_day'] == 14
        assert features['day_of_week'] == 3  # Wednesday (0-indexed)
        assert features['month'] == 6
        assert features['night_indicator'] == 0.0

    def test_extract_time_features_night(self):
        """Test time feature extraction for nighttime."""
        ts = datetime(2023, 6, 15, 22, 0)  # 10 PM
        features = extract_time_features(ts)

        assert features['hour_of_day'] == 22
        assert features['night_indicator'] == 1.0

    def test_extract_time_features_early_morning(self):
        """Test time feature extraction for early morning."""
        ts = datetime(2023, 6, 15, 3, 30)  # 3:30 AM
        features = extract_time_features(ts)

        assert features['hour_of_day'] == 3
        assert features['night_indicator'] == 1.0

    def test_extract_road_features(self):
        """Test road feature extraction."""
        road_attrs = {'highway': 'secondary', 'speed_limit_kmh': 50.0}
        features = extract_road_features(road_attrs)

        assert features['road_class'] == 2  # secondary = 2
        assert features['speed_limit_kmh'] == 50.0

    def test_extract_road_features_default(self):
        """Test road feature defaults."""
        features = extract_road_features({})

        assert features['road_class'] == 4  # unclassified
        assert features['speed_limit_kmh'] == 50.0  # default

    def test_extract_weather_features(self):
        """Test weather feature extraction."""
        weather = {
            'precipitation_mm': 5.0,
            'visibility_km': 8.0,
            'wind_speed_ms': 3.0,
            'temperature_c': 25.0,
        }
        features = extract_weather_features(weather)

        assert features['precipitation_mm'] == 5.0
        assert features['visibility_km'] == 8.0
        assert features['wind_speed_ms'] == 3.0
        assert features['temperature_c'] == 25.0

    def test_extract_weather_features_defaults(self):
        """Test weather feature defaults."""
        features = extract_weather_features({})

        assert features['precipitation_mm'] == 0.0
        assert features['visibility_km'] == 10.0
        assert features['wind_speed_ms'] == 0.0
        assert features['temperature_c'] == 20.0


class TestFeatureVector:
    """Tests for complete feature vector building."""

    def test_build_feature_vector_basic(self):
        """Test basic feature vector construction."""
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

        vector = build_feature_vector(raw, validate=False)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(FEATURE_NAMES)
        assert not np.any(np.isnan(vector))

    def test_feature_order(self):
        """Test that features are in correct order."""
        raw = {
            'timestamp': datetime(2023, 6, 15, 14, 30),
            'highway': 'secondary',
            'speed_limit_kmh': 50.0,
            'precipitation_mm': 0.0,
            'visibility_km': 10.0,
            'wind_speed_ms': 0.0,
            'temperature_c': 20.0,
            'historical_accident_rate': 0.0,
        }

        vector = build_feature_vector(raw, validate=False)

        assert vector[FEATURE_INDEX['hour_of_day']] == 14
        assert vector[FEATURE_INDEX['day_of_week']] == 3
        assert vector[FEATURE_INDEX['month']] == 6
        assert vector[FEATURE_INDEX['night_indicator']] == 0.0
        assert vector[FEATURE_INDEX['road_class']] == 2

    def test_feature_vector_with_rain(self):
        """Test feature vector with rain conditions."""
        raw = {
            'timestamp': datetime(2023, 6, 15, 14, 30),
            'highway': 'secondary',
            'speed_limit_kmh': 50.0,
            'precipitation_mm': 10.0,  # Heavy rain
            'visibility_km': 5.0,  # Low visibility
            'wind_speed_ms': 5.0,  # Windy
            'temperature_c': 15.0,
            'historical_accident_rate': 2.0,  # High risk area
        }

        vector = build_feature_vector(raw, validate=False)

        # Check that rain-related features are non-zero
        assert vector[FEATURE_INDEX['precipitation_mm']] == 10.0
        assert vector[FEATURE_INDEX['visibility_km']] == 5.0
        assert vector[FEATURE_INDEX['rain_on_congestion']] > 0.0

    def test_feature_vector_missing_optional(self):
        """Test that missing optional fields use defaults."""
        raw = {
            'timestamp': datetime(2023, 6, 15, 14, 30),
            # Missing highway, weather, etc.
        }

        vector = build_feature_vector(raw, validate=False)

        assert len(vector) == len(FEATURE_NAMES)
        # Check defaults were applied
        assert vector[FEATURE_INDEX['road_class']] == 4  # unclassified
        assert vector[FEATURE_INDEX['speed_limit_kmh']] == 50.0
        assert vector[FEATURE_INDEX['precipitation_mm']] == 0.0

    def test_feature_vector_iso_timestamp(self):
        """Test that ISO string timestamps work."""
        raw = {
            'timestamp': '2023-06-15T14:30:00',
            'highway': 'primary',
            'speed_limit_kmh': 60.0,
            'precipitation_mm': 0.0,
            'visibility_km': 10.0,
            'wind_speed_ms': 0.0,
            'temperature_c': 20.0,
            'historical_accident_rate': 0.0,
        }

        vector = build_feature_vector(raw, validate=False)

        assert vector[FEATURE_INDEX['hour_of_day']] == 14
        assert vector[FEATURE_INDEX['month']] == 6

    def test_feature_vector_raises_on_missing_timestamp(self):
        """Test that missing timestamp raises error."""
        raw = {'highway': 'secondary'}

        with pytest.raises(ValueError):
            build_feature_vector(raw, validate=False)


class TestFeatureValidation:
    """Tests for feature validation."""

    def test_validate_normal_vector(self):
        """Test validation of normal feature vector."""
        raw = {
            'timestamp': datetime(2023, 6, 15, 14, 30),
            'highway': 'secondary',
            'speed_limit_kmh': 50.0,
            'precipitation_mm': 5.0,
            'visibility_km': 8.0,
            'wind_speed_ms': 3.0,
            'temperature_c': 25.0,
            'historical_accident_rate': 0.5,
        }

        vector = build_feature_vector(raw, validate=True)
        assert len(vector) == len(FEATURE_NAMES)

    def test_extreme_weather(self):
        """Test handling of extreme weather values."""
        raw = {
            'timestamp': datetime(2023, 6, 15, 14, 30),
            'highway': 'secondary',
            'speed_limit_kmh': 50.0,
            'precipitation_mm': 100.0,  # Extreme rain
            'visibility_km': 0.1,  # Extreme fog
            'wind_speed_ms': 40.0,  # High wind
            'temperature_c': -40.0,  # Extreme cold
            'historical_accident_rate': 5.0,
        }

        vector = build_feature_vector(raw, validate=False)
        assert len(vector) == len(FEATURE_NAMES)
        assert not np.any(np.isnan(vector))


class TestFeatureDataFrame:
    """Tests for batch feature engineering."""

    def test_build_features_from_dataframe(self):
        """Test building features from a DataFrame."""
        from app.ml.features import build_feature_dataframe

        data = pd.DataFrame([
            {
                'timestamp': datetime(2023, 6, 15, 14, 30),
                'highway': 'secondary',
                'speed_limit_kmh': 50.0,
                'precipitation_mm': 5.0,
                'visibility_km': 8.0,
                'wind_speed_ms': 3.0,
                'temperature_c': 25.0,
                'historical_accident_rate': 0.5,
            },
            {
                'timestamp': datetime(2023, 6, 15, 22, 0),
                'highway': 'primary',
                'speed_limit_kmh': 60.0,
                'precipitation_mm': 0.0,
                'visibility_km': 10.0,
                'wind_speed_ms': 0.0,
                'temperature_c': 20.0,
                'historical_accident_rate': 0.2,
            },
        ])

        features = build_feature_dataframe(data)

        assert len(features) == 2
        assert list(features.columns) == FEATURE_NAMES
        assert not features.isnull().any().any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
