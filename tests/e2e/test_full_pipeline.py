import os

import httpx
import pytest


BASE_URL = os.getenv("STRIVE_E2E_BASE_URL", "http://localhost:8000")


@pytest.mark.skipif(os.getenv("RUN_LIVE_E2E") != "1", reason="Requires the live Docker stack.")
@pytest.mark.anyio
async def test_full_pipeline():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=15.0) as client:
        health = await client.get("/health")
        assert health.status_code == 200

        risk = await client.get("/v1/risk/segment", params={"lat": 34.05, "lon": -118.24})
        assert risk.status_code == 200
        assert risk.json()["shap_top_factors"]

        explain = await client.get("/v1/explain/segment", params={"lat": 34.05, "lon": -118.24})
        assert explain.status_code == 200
        assert "shap_values" in explain.json()

        route = await client.post(
            "/v1/route/safe",
            json={
                "origin": {"lat": 34.052, "lon": -118.243},
                "destination": {"lat": 34.073, "lon": -118.2},
                "alpha": 0.8,
            },
        )
        assert route.status_code == 200
        assert route.json()["risk_reduction_pct"] >= 0

        heatmap = await client.get("/v1/risk/heatmap", params={"bbox": "-118.3,34.0,-118.2,34.1"})
        assert heatmap.status_code == 200
        assert heatmap.json()["type"] == "FeatureCollection"
