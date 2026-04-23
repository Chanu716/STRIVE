"""Performance benchmarks for risk scoring and routing."""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient


def test_performance_benchmarks(client: TestClient):
    """
    Performance Targets:
    - Risk scoring (segment) <= 500ms
    - Routing (city-scale) <= 2000ms
    """
    # 1. Risk Scoring Performance
    lat, lon = 34.05, -118.24
    start_time = time.perf_counter()
    resp = client.get("/v1/risk/segment", params={"lat": lat, "lon": lon})
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    assert resp.status_code == 200
    print(f"\nRisk Scoring Latency: {duration_ms:.2f}ms")
    assert duration_ms <= 500, f"Risk scoring too slow: {duration_ms:.2f}ms"

    # 2. Routing Performance
    route_payload = {
        "origin": {"lat": 34.05, "lon": -118.24},
        "destination": {"lat": 34.055, "lon": -118.245},
        "alpha": 0.5
    }
    start_time = time.perf_counter()
    resp = client.post("/v1/route/safe", json=route_payload)
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    assert resp.status_code == 200
    print(f"Routing Latency: {duration_ms:.2f}ms")
    assert duration_ms <= 2000, f"Routing too slow: {duration_ms:.2f}ms"
