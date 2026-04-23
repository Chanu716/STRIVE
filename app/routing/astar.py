"""Safety-aware A* routing utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np


DEFAULT_SPEED_KPH = 50.0
DEFAULT_LENGTH_M = 50.0
DEFAULT_RISK_SCORE = 50.0


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    if isinstance(value, str):
        value = value.replace("mph", "").replace("km/h", "").strip()
        if ";" in value:
            value = value.split(";", 1)[0].strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def travel_time_seconds(edge_data: dict[str, Any]) -> float:
    """Estimate edge travel time from length and speed attributes."""
    length_m = _as_float(edge_data.get("length"), DEFAULT_LENGTH_M)
    speed_kph = _as_float(edge_data.get("speed_kph") or edge_data.get("maxspeed"), DEFAULT_SPEED_KPH)
    speed_mps = max(speed_kph / 3.6, 1.0)
    return max(length_m / speed_mps, 0.0)


def _iter_edge_data(graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    return [data for _, _, _, data in graph.edges(keys=True, data=True)]


@lru_cache(maxsize=8)
def max_travel_time(graph_id: int, edge_count: int, edge_signature: int, graph: nx.MultiDiGraph) -> float:
    """Compute a stable 95th percentile travel-time normalizer for a graph."""
    del graph_id, edge_count, edge_signature
    travel_times = [travel_time_seconds(data) for data in _iter_edge_data(graph)]
    if not travel_times:
        return 1.0
    percentile = float(np.percentile(travel_times, 95))
    return max(percentile, 1.0)


def travel_time_normalizer(graph: nx.MultiDiGraph) -> float:
    """Return the cached 95th percentile travel time for the graph."""
    edge_count = graph.number_of_edges()
    edge_signature = hash(tuple(graph.edges(keys=True))) if edge_count < 100_000 else edge_count
    return max_travel_time(id(graph), edge_count, edge_signature, graph)


def _edge_key(edge_data: dict[str, Any], default: int = 0) -> Any:
    return edge_data.get("key", default)


def _single_edge_weight(
    u: Any,
    v: Any,
    key: Any,
    edge_data: dict[str, Any],
    alpha: float,
    risk_scores: dict[tuple[Any, Any, Any], float],
    max_tt: float,
) -> float:
    travel_component = min(travel_time_seconds(edge_data) / max_tt, 1.0)
    risk = risk_scores.get((u, v, key), risk_scores.get((str(u), str(v), str(key)), DEFAULT_RISK_SCORE))
    risk_component = min(max(float(risk) / 100.0, 0.0), 1.0)
    return alpha * risk_component + (1.0 - alpha) * travel_component


def _weight_function(
    alpha: float,
    risk_scores: dict[tuple[Any, Any, Any], float],
    max_tt: float,
):
    def weight(u: Any, v: Any, data: dict[str, Any]) -> float:
        if data and all(isinstance(value, dict) for value in data.values()):
            return min(
                _single_edge_weight(u, v, key, edge_data, alpha, risk_scores, max_tt)
                for key, edge_data in data.items()
            )
        return _single_edge_weight(u, v, _edge_key(data), data, alpha, risk_scores, max_tt)

    return weight


def safe_route(
    graph: nx.MultiDiGraph,
    origin_node: Any,
    dest_node: Any,
    alpha: float,
    risk_scores: dict[tuple[Any, Any, Any], float],
) -> list[Any]:
    """Return ordered OSM node IDs for the risk-weighted A* route."""
    bounded_alpha = min(max(alpha, 0.0), 1.0)
    max_tt = travel_time_normalizer(graph)
    return nx.astar_path(
        graph,
        origin_node,
        dest_node,
        weight=_weight_function(bounded_alpha, risk_scores, max_tt),
    )
