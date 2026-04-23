"""Integration tests for risk scoring endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_segment_risk_endpoint(client: TestClient):
    """Verify that a valid lat/lon returns a risk score payload."""
    response = client.get("/v1/risk/segment", params={"lat": 34.05, "lon": -118.24})
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert "shap_top_factors" in data
    assert len(data["shap_top_factors"]) > 0


def test_segment_risk_invalid_params(client: TestClient):
    """Verify that missing parameters return 422 Unprocessable Entity."""
    response = client.get("/v1/risk/segment", params={"lat": 34.05})
    assert response.status_code == 422


def test_risk_heatmap_endpoint(client: TestClient):
    """Verify that a bounding box returns a GeoJSON FeatureCollection."""
    bbox = "-118.3,34.0,-118.2,34.1"
    response = client.get("/v1/risk/heatmap", params={"bbox": bbox})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data


def test_risk_heatmap_malformed_bbox(client: TestClient):
    """Verify that a malformed bounding box returns an error."""
    # Only 3 values instead of 4
    response = client.get("/v1/risk/heatmap", params={"bbox": "34.0,-118.2,34.1"})
    assert response.status_code == 422
