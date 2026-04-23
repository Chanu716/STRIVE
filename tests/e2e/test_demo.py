"""End-to-end demo validation for STRIVE."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_e2e_scoring_to_explanation_to_routing(client: TestClient):
    """
    Validation Scenario:
    1. Check risk for a known coordinate.
    2. Ensure explanation correlates with risk.
    3. Compute a safe route and check the response structure.
    """
    # 1. Scoring
    lat, lon = 34.05, -118.24
    risk_resp = client.get("/v1/risk/segment", params={"lat": lat, "lon": lon})
    assert risk_resp.status_code == 200
    risk_data = risk_resp.json()
    score = risk_data["risk_score"]
    
    # 2. Explanation
    explain_resp = client.get("/v1/explain/segment", params={"lat": lat, "lon": lon})
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()
    assert explain_data["risk_score"] == score
    
    # 3. Routing
    route_payload = {
        "origin": {"lat": 34.05, "lon": -118.24},
        "destination": {"lat": 34.055, "lon": -118.245},
        "alpha": 0.5
    }
    route_resp = client.post("/v1/route/safe", json=route_payload)
    assert route_resp.status_code == 200
    route_data = route_resp.json()
    
    assert "route" in route_data
    assert "risk_reduction_pct" in route_data
    assert "segments" in route_data
    
    # Verify GeoJSON
    assert route_data["route"]["geometry"]["type"] == "LineString"
    assert isinstance(route_data["route"]["geometry"]["coordinates"], list)
