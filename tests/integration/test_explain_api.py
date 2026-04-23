"""Integration tests for explainability endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_segment_explain_endpoint(client: TestClient):
    """Verify that a valid lat/lon returns a full SHAP explanation."""
    response = client.get("/v1/explain/segment", params={"lat": 34.05, "lon": -118.24})
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "shap_values" in data
    assert "expected_value" in data
    assert "risk_score" in data
    
    # Mathematical sanity check for SHAP: baseline + sum(shap) should reflect some logic
    # In tree models, logit(p) = expected_value + sum(shap)
    # result.risk_score approx (1 / (1 + exp(- (expected + sum)))) * 100
    expected = data["expected_value"]
    shap_sum = sum(data["shap_values"].values())
    assert abs(data["risk_score"] / 100.0 - 1.0 / (1.0 + pow(2.718, -(expected + shap_sum)))) < 0.1
