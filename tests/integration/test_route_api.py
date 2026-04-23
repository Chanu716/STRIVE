"""Integration tests for routing endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_safe_route_endpoint(client: TestClient):
    """Verify that a POST route request returns a valid official route payload."""
    payload = {
        "origin": {"lat": 34.05, "lon": -118.24},
        "destination": {"lat": 34.051, "lon": -118.241},
        "alpha": 0.5
    }
    response = client.post("/v1/route/safe", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Official structure check
    assert "route" in data
    assert "fastest" in data
    assert "risk_reduction_pct" in data
    assert "segments" in data
    
    # Summary metrics
    assert "distance_km" in data["route"]
    assert "duration_min" in data["route"]
    assert "geometry" in data["route"]


def test_route_endpoint_invalid_payload(client: TestClient):
    """Verify that malformed JSON payload returns 422."""
    payload = {
        "origin": {"lat": 34.05}  # Missing lon
    }
    response = client.post("/v1/route/safe", json=payload)
    assert response.status_code == 422
