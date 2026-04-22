"""Model loading, inference, and explainability helpers for STRIVE."""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from app.ml.features import FEATURE_NAMES


load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
FEATURE_CONFIG_PATH = Path(os.getenv("FEATURE_CONFIG_PATH", "models/feature_config.json"))


@dataclass
class ExplanationResult:
    """Full prediction payload returned by the inference layer."""

    risk_score: int
    probability: float
    shap_values: dict[str, float]
    expected_value: float


class _FallbackRiskModel:
    """Heuristic model used until the trained artefact from T-11 is available."""

    def __init__(self) -> None:
        self.feature_names_in_ = np.array(FEATURE_NAMES, dtype=object)
        self._weights = {
            "precipitation_mm": 0.08,
            "visibility_km": -0.05,
            "wind_speed_ms": 0.03,
            "temperature_c": -0.01,
            "historical_accident_rate": 0.04,
            "night_indicator": 0.45,
            "road_class": 0.08,
            "speed_limit_kmh": 0.006,
            "rain_on_congestion": 0.6,
            "hour_of_day": 0.0,
            "day_of_week": 0.0,
            "month": 0.0,
        }
        self._bias = -2.2

    def predict_proba(self, feature_matrix: np.ndarray) -> np.ndarray:
        row = feature_matrix[0]
        score = self._bias
        for index, name in enumerate(FEATURE_NAMES):
            score += float(row[index]) * self._weights.get(name, 0.0)
        probability = 1.0 / (1.0 + math.exp(-score))
        return np.array([[1.0 - probability, probability]], dtype=float)

    def explain(self, feature_vector: np.ndarray) -> ExplanationResult:
        shap_values = {
            name: float(feature_vector[idx]) * self._weights.get(name, 0.0)
            for idx, name in enumerate(FEATURE_NAMES)
        }
        probability = float(self.predict_proba(feature_vector.reshape(1, -1))[0][1])
        return ExplanationResult(
            risk_score=int(round(probability * 100)),
            probability=probability,
            shap_values=shap_values,
            expected_value=float(self._bias),
        )


@lru_cache(maxsize=1)
def load_feature_config() -> dict[str, Any]:
    """Load optional feature metadata emitted by the training pipeline."""
    if not FEATURE_CONFIG_PATH.exists():
        return {"feature_names": FEATURE_NAMES}
    with FEATURE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the trained model, or fall back to a deterministic heuristic model."""
    if not MODEL_PATH.exists():
        return _FallbackRiskModel()
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def _compute_native_shap(model: Any, feature_vector: np.ndarray) -> tuple[dict[str, float], float]:
    """Compute SHAP values with TreeExplainer when the dependency/model is available."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_vector.reshape(1, -1))
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[-1])[0]
    else:
        shap_row = np.asarray(shap_values)[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[-1]
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.item()

    return (
        {name: float(shap_row[idx]) for idx, name in enumerate(FEATURE_NAMES)},
        float(expected_value),
    )


def explain_prediction(feature_vector: np.ndarray) -> ExplanationResult:
    """Run inference and return a normalized explanation payload."""
    model = load_model()

    if hasattr(model, "explain"):
        return model.explain(feature_vector)

    probability = float(model.predict_proba(feature_vector.reshape(1, -1))[0][1])
    risk_score = int(round(probability * 100))

    try:
        shap_values, expected_value = _compute_native_shap(model, feature_vector)
    except Exception:
        fallback_model = _FallbackRiskModel()
        fallback = fallback_model.explain(feature_vector)
        shap_values = fallback.shap_values
        expected_value = fallback.expected_value

    return ExplanationResult(
        risk_score=risk_score,
        probability=probability,
        shap_values=shap_values,
        expected_value=expected_value,
    )


def run_inference(feature_vector: np.ndarray) -> int:
    """Return a 0-100 risk score for the given feature vector."""
    return explain_prediction(feature_vector).risk_score
