import networkx as nx
from httpx import AsyncClient, ASGITransport
from shapely.geometry import LineString

from app import main
from app.ml.features import FEATURE_NAMES
from app.ml.inference import ExplanationResult
from app.routers import risk


def build_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_edge(1, 2, key=0, length=100.0, speed_kph=50.0, highway="primary", geometry=LineString([(0.0, 0.0), (1.0, 0.0)]))
    return graph


def patch_dependencies(monkeypatch):
    graph = build_graph()
    explanation = ExplanationResult(
        risk_score=72,
        probability=0.72,
        shap_values={name: float(index) / 100.0 for index, name in enumerate(FEATURE_NAMES)},
        expected_value=0.25,
    )
    monkeypatch.setattr(risk, "_load_graph", lambda: graph)
    monkeypatch.setattr(risk, "_find_segment_in_db", lambda db, segment_id: None)
    monkeypatch.setattr(risk, "get_weather", lambda lat, lon: {"precipitation_mm": 8.0, "visibility_km": 7.0, "wind_speed_ms": 2.0, "temperature_c": 18.0})
    monkeypatch.setattr(risk, "explain_prediction", lambda feature_vector: explanation)


async def test_explain_endpoint(monkeypatch):
    patch_dependencies(monkeypatch)
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/explain/segment", params={"lat": 0.1, "lon": 0.1, "datetime": "2024-06-15T22:00:00"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_score"] == 72
    assert set(payload["features"]) == set(FEATURE_NAMES)
    assert set(payload["shap_values"]) == set(FEATURE_NAMES)
