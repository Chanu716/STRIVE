#!/usr/bin/env python3
"""
Integration tests for M1 data pipeline.

Tests the complete flow from raw data to features.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ml.features import build_feature_dataframe, build_feature_vector, FEATURE_NAMES


class TestDataPipelineIntegration:
    """Integration tests for M1 data pipeline."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_feature_vector_consistency(self):
        """Test that building features multiple times gives same result."""
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

        vector1 = build_feature_vector(raw, validate=False)
        vector2 = build_feature_vector(raw, validate=False)

        assert np.allclose(vector1, vector2), "Feature vectors should be identical"

    def test_batch_to_single_consistency(self):
        """Test that batch and single processing give same result."""
        from app.ml.features import build_feature_dataframe

        raw_dict = {
            'timestamp': datetime(2023, 6, 15, 14, 30),
            'highway': 'secondary',
            'speed_limit_kmh': 50.0,
            'precipitation_mm': 5.0,
            'visibility_km': 8.0,
            'wind_speed_ms': 3.0,
            'temperature_c': 25.0,
            'historical_accident_rate': 0.5,
        }

        # Single vector
        single = build_feature_vector(raw_dict, validate=False)

        # Batch
        df_input = pd.DataFrame([raw_dict])
        batch = build_feature_dataframe(df_input)

        assert np.allclose(single, batch.iloc[0].values), \
            "Single and batch features should match"

    def test_missing_segments_handling(self):
        """Test that snapping with missing data is handled gracefully."""
        # This simulates what happens when some accidents don't snap to segments
        raw_data = pd.DataFrame([
            {
                'timestamp': datetime(2023, 6, 15, 14, 30),
                'highway': 'primary',
                'speed_limit_kmh': 60.0,
                'precipitation_mm': 0.0,
                'visibility_km': 10.0,
                'wind_speed_ms': 0.0,
                'temperature_c': 20.0,
                'historical_accident_rate': 0.0,
            },
            {
                'timestamp': datetime(2023, 6, 15, 22, 0),
                'highway': None,  # Missing highway info
                'speed_limit_kmh': 50.0,
                'precipitation_mm': 5.0,
                'visibility_km': 8.0,
                'wind_speed_ms': 3.0,
                'temperature_c': 25.0,
                'historical_accident_rate': 0.5,
            },
        ])

        features = build_feature_dataframe(raw_data)

        assert len(features) == 2
        assert not features.isnull().any().any()

    def test_weather_extremes_realistic(self):
        """Test realistic but extreme weather scenarios."""
        scenarios = [
            # Clear day
            {
                'timestamp': datetime(2023, 6, 15, 14, 30),
                'precipitation_mm': 0.0,
                'visibility_km': 15.0,
                'wind_speed_ms': 0.5,
                'temperature_c': 28.0,
            },
            # Heavy rain, poor visibility, high wind
            {
                'timestamp': datetime(2023, 1, 15, 2, 0),
                'precipitation_mm': 30.0,
                'visibility_km': 0.5,
                'wind_speed_ms': 20.0,
                'temperature_c': 5.0,
            },
            # Heavy snow
            {
                'timestamp': datetime(2023, 12, 15, 6, 0),
                'precipitation_mm': 50.0,
                'visibility_km': 0.2,
                'wind_speed_ms': 25.0,
                'temperature_c': -10.0,
            },
        ]

        for scenario in scenarios:
            raw = {
                'highway': 'secondary',
                'speed_limit_kmh': 50.0,
                **scenario,
                'historical_accident_rate': 0.5,
            }
            vector = build_feature_vector(raw, validate=False)
            assert len(vector) == len(FEATURE_NAMES)
            assert not np.any(np.isnan(vector))

    def test_road_type_mapping(self):
        """Test all road type classifications."""
        from app.ml.features import ROAD_CLASS_MAPPING

        road_types = [
            ('motorway', 0),
            ('trunk', 0),
            ('primary', 1),
            ('secondary', 2),
            ('tertiary', 3),
            ('residential', 4),
            ('unknown', 4),  # default
        ]

        for highway_type, expected_class in road_types:
            raw = {
                'timestamp': datetime(2023, 6, 15, 14, 30),
                'highway': highway_type,
                'speed_limit_kmh': 50.0,
                'precipitation_mm': 0.0,
                'visibility_km': 10.0,
                'wind_speed_ms': 0.0,
                'temperature_c': 20.0,
                'historical_accident_rate': 0.0,
            }
            vector = build_feature_vector(raw, validate=False)
            assert vector[4] == expected_class, \
                f"Road class for {highway_type} should be {expected_class}"

    def test_time_patterns(self):
        """Test time feature patterns for different times of day."""
        test_cases = [
            # Daytime
            (datetime(2023, 6, 15, 9, 0), 0.0, "Morning commute"),
            (datetime(2023, 6, 15, 14, 0), 0.0, "Midday"),
            (datetime(2023, 6, 15, 18, 0), 0.0, "Evening commute"),
            # Nighttime
            (datetime(2023, 6, 15, 21, 0), 1.0, "Night (9 PM)"),
            (datetime(2023, 6, 15, 2, 0), 1.0, "Late night"),
            (datetime(2023, 6, 15, 5, 0), 1.0, "Early morning"),
            # Boundary
            (datetime(2023, 6, 15, 6, 0), 0.0, "Boundary (6 AM)"),
            (datetime(2023, 6, 15, 20, 0), 1.0, "Boundary (8 PM)"),
        ]

        for ts, expected_night, description in test_cases:
            raw = {
                'timestamp': ts,
                'highway': 'secondary',
                'speed_limit_kmh': 50.0,
                'precipitation_mm': 0.0,
                'visibility_km': 10.0,
                'wind_speed_ms': 0.0,
                'temperature_c': 20.0,
                'historical_accident_rate': 0.0,
            }
            vector = build_feature_vector(raw, validate=False)
            # night_indicator is at index 3
            assert vector[3] == expected_night, \
                f"Night indicator failed for {description} ({ts})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
