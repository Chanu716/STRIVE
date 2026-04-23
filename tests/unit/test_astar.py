"""Unit tests for safety-aware routing engine."""

from __future__ import annotations

import networkx as nx
import pytest

from app.routing.astar import safe_route


@pytest.fixture
def test_graph() -> nx.MultiDiGraph:
    """Create a 2x2 multi-di-graph for routing tests."""
    G = nx.MultiDiGraph()
    # Path 0-1-3
    G.add_edge(0, 1, key=0, length=100, speed_kph=50)
    G.add_edge(1, 3, key=0, length=100, speed_kph=50)
    # Path 0-2-3
    G.add_edge(0, 2, key=0, length=100, speed_kph=50)
    G.add_edge(2, 3, key=0, length=100, speed_kph=50)
    return G


def test_route_connectivity(test_graph: nx.MultiDiGraph):
    """Verify that A* finds a path between diagonally opposite nodes."""
    # alpha=0.5, risk_scores=empty (defaults to 50)
    path = safe_route(test_graph, 0, 3, alpha=0.5, risk_scores={})
    assert path[0] == 0
    assert path[-1] == 3
    assert len(path) == 3


def test_safety_prioritization(test_graph: nx.MultiDiGraph):
    """Verify that high alpha avoids risky edges."""
    # Use (u, v, key) tuples for risk_scores
    risk_scores = {
        (0, 1, 0): 90.0,
        (1, 3, 0): 90.0,
        (0, 2, 0): 10.0,
        (2, 3, 0): 10.0,
    }
    
    # With high alpha, it should take 0-2-3 (safe)
    path_safe = safe_route(test_graph, 0, 3, alpha=0.9, risk_scores=risk_scores)
    assert path_safe == [0, 2, 3]
    
    # Flip the risks: 0-1-3- becomes safe
    flipped_risks = {k: 100.0 - v for k, v in risk_scores.items()}
    path_safe_flipped = safe_route(test_graph, 0, 3, alpha=0.9, risk_scores=flipped_risks)
    assert path_safe_flipped == [0, 1, 3]


def test_travel_time_dominance(test_graph: nx.MultiDiGraph):
    """Verify that alpha=0 ignores risk and optimizes for speed/distance."""
    G = test_graph.copy()
    # Path 0-1-3 is very slow
    G[0][1][0]["speed_kph"] = 5
    G[1][3][0]["speed_kph"] = 5
    # Path 0-2-3 is fast
    G[0][2][0]["speed_kph"] = 100
    G[2][3][0]["speed_kph"] = 100
    
    # Even if 0-2-3 is riskier, alpha=0 should pick it
    risks = {(0, 2, 0): 50.0, (2, 3, 0): 50.0, (0, 1, 0): 1.0, (1, 3, 0): 1.0}
    
    path_fast = safe_route(G, 0, 3, alpha=0.0, risk_scores=risks)
    assert path_fast == [0, 2, 3]
    
    # alpha=1 should ignore speed
    path_safe = safe_route(G, 0, 3, alpha=1.0, risk_scores=risks)
    assert path_safe == [0, 1, 3]
