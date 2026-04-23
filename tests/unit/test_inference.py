"""Unit tests for model inference layer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.ml.features import FEATURE_NAMES
from app.ml.inference import _FallbackRiskModel, explain_prediction, run_inference


@pytest.fixture
def mock_model():
    """Ensure we test against the deterministic fallback model."""
    with patch("app.ml.inference.load_model", return_value=_FallbackRiskModel()):
        yield


def test_inference_returns_score_in_range(mock_model):
    """Verify that every prediction is bucketed [0, 100]."""
    dummy_vec = np.zeros(len(FEATURE_NAMES))
    score = run_inference(dummy_vec)
    assert isinstance(score, int)
    assert 0 <= score <= 100


def test_explanation_contains_all_features(mock_model):
    """Verify that SHAP explanations cover the full feature space."""
    dummy_vec = np.zeros(len(FEATURE_NAMES))
    result = explain_prediction(dummy_vec)
    assert len(result.shap_values) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in result.shap_values


def test_high_rain_night_logic(mock_model):
    """Verify that the fallback model correctly elevates risk for poor conditions."""
    vec = np.zeros(len(FEATURE_NAMES))
    vec[FEATURE_NAMES.index("precipitation_mm")] = 50.0  # Heavy rain
    vec[FEATURE_NAMES.index("night_indicator")] = 1.0   # Night
    
    result = explain_prediction(vec)
    # Expected logit: -2.2 + 50*0.08 + 1*0.45 = 2.25
    # Prob = 1/(1+exp(-2.25)) = 0.904 -> risk_score = 90
    assert result.risk_score >= 80
    assert result.shap_values["night_indicator"] > 0
    assert result.shap_values["precipitation_mm"] > 0
