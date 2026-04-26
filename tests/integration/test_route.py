import networkx as nx
from httpx import AsyncClient, ASGITransport
from shapely.geometry import LineString

from app import main
from app.routers import route


def build_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_node(3, x=2.0, y=0.0)
    graph.add_node(4, x=1.0, y=1.0)
    graph.add_edge(1, 2, key=0, length=100.0, speed_kph=100.0, highway="primary", geometry=LineString([(0.0, 0.0), (1.0, 0.0)]))
    graph.add_edge(2, 3, key=0, length=100.0, speed_kph=100.0, highway="primary", geometry=LineString([(1.0, 0.0), (2.0, 0.0)]))
    graph.add_edge(1, 4, key=0, length=150.0, speed_kph=30.0, highway="residential", geometry=LineString([(0.0, 0.0), (1.0, 1.0)]))
    graph.add_edge(4, 3, key=0, length=150.0, speed_kph=30.0, highway="residential", geometry=LineString([(1.0, 1.0), (2.0, 0.0)]))
    return graph


def patch_dependencies(monkeypatch):
    graph = build_graph()
    monkeypatch.setattr(route, "get_graph", lambda: graph)
    nodes = iter([1, 3])
    monkeypatch.setattr(route, "nearest_node", lambda lat, lon: next(nodes))
    monkeypatch.setattr(route, "get_weather", lambda lat, lon: {"precipitation_mm": 8.0, "visibility_km": 7.0, "wind_speed_ms": 2.0, "temperature_c": 18.0})
    monkeypatch.setattr(
        route,
        "_score_graph_edges",
        lambda graph, weather, timestamp: {
            (1, 2, 0): 95.0,
            (2, 3, 0): 95.0,
            (1, 4, 0): 5.0,
            (4, 3, 0): 5.0,
            ("1", "2", "0"): 95.0,
            ("2", "3", "0"): 95.0,
            ("1", "4", "0"): 5.0,
            ("4", "3", "0"): 5.0,
        },
    )


async def test_route_endpoint(monkeypatch):
    patch_dependencies(monkeypatch)
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/route/safe",
            json={
                "origin": {"lat": 0.0, "lon": 0.0},
                "destination": {"lat": 2.0, "lon": 0.0},
                "alpha": 0.6,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"]["avg_risk_score"] < payload["fastest"]["avg_risk_score"]
    assert payload["risk_reduction_pct"] > 0
    assert len(payload["segments"]) >= 1
