"""Integration tests for the health check endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Verify that /health returns 200 OK with model and DB status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["db_connected"] is True
