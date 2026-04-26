import numpy as np

from app.ml.features import FEATURE_NAMES
from app.ml.inference import ExplanationResult, explain_prediction, run_inference


class DummyModel:
    def explain(self, feature_vector):
        shap_values = {name: float(index + 1) * 0.01 for index, name in enumerate(FEATURE_NAMES)}
        return ExplanationResult(
            risk_score=88,
            probability=0.88,
            shap_values=shap_values,
            expected_value=0.12,
        )


def test_run_inference_returns_risk_score(monkeypatch):
    monkeypatch.setattr("app.ml.inference.load_model", lambda: DummyModel())
    score = run_inference(np.zeros(len(FEATURE_NAMES)))
    assert score == 88


def test_explain_prediction_returns_full_payload(monkeypatch):
    monkeypatch.setattr("app.ml.inference.load_model", lambda: DummyModel())
    result = explain_prediction(np.zeros(len(FEATURE_NAMES)))
    assert result.risk_score == 88
    assert set(result.shap_values) == set(FEATURE_NAMES)
    assert result.expected_value == 0.12
