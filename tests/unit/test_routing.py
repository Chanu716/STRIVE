import networkx as nx

from app.routing.astar import safe_route, travel_time_normalizer


def _sample_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_node(3, x=2.0, y=0.0)
    graph.add_node(4, x=1.0, y=1.0)

    graph.add_edge(1, 2, key=0, length=100.0, speed_kph=100.0)
    graph.add_edge(2, 3, key=0, length=100.0, speed_kph=100.0)
    graph.add_edge(1, 4, key=0, length=150.0, speed_kph=30.0)
    graph.add_edge(4, 3, key=0, length=150.0, speed_kph=30.0)
    return graph


def test_safe_route_uses_fastest_path_when_alpha_zero():
    graph = _sample_graph()
    risk_scores = {
        (1, 2, 0): 95.0,
        (2, 3, 0): 95.0,
        (1, 4, 0): 5.0,
        (4, 3, 0): 5.0,
    }

    assert safe_route(graph, 1, 3, 0.0, risk_scores) == [1, 2, 3]


def test_safe_route_uses_lower_risk_path_when_alpha_one():
    graph = _sample_graph()
    risk_scores = {
        (1, 2, 0): 95.0,
        (2, 3, 0): 95.0,
        (1, 4, 0): 5.0,
        (4, 3, 0): 5.0,
    }

    assert safe_route(graph, 1, 3, 1.0, risk_scores) == [1, 4, 3]


def test_travel_time_normalizer_is_positive():
    graph = _sample_graph()

    assert travel_time_normalizer(graph) > 0.0
