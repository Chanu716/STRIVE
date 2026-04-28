"""Safety-aware routing endpoint for STRIVE."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.features import build_feature_vector
from app.ml.inference import load_model, run_inference
from app.routing.astar import safe_route, travel_time_seconds, travel_time_normalizer
from app.routing.graph import get_static_graph, nearest_node
from app.weather import get_weather


router = APIRouter(prefix="/v1/route", tags=["route"])

SEGMENT_RATES_PATH = Path("data/processed/segment_rates.parquet")

RISK_LEVELS = (
    (24, "LOW"),
    (49, "MODERATE"),
    (74, "HIGH"),
    (100, "CRITICAL"),
)


class Coordinate(BaseModel):
    """Latitude/longitude coordinate."""

    lat: float = Field(description="Latitude in decimal degrees.", examples=[34.052])
    lon: float = Field(description="Longitude in decimal degrees.", examples=[-118.243])


class SafeRouteRequest(BaseModel):
    """Request body for safety-aware routing."""

    origin: Coordinate = Field(description="Trip origin coordinate.")
    destination: Coordinate = Field(description="Trip destination coordinate.")
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Safety weight. 0 optimizes travel time only; 1 optimizes risk only.",
        examples=[0.6],
    )


class RouteSegment(BaseModel):
    """Risk score for one traversed route segment."""

    segment_id: str = Field(description="Road segment identifier.")
    risk_score: int = Field(description="Predicted risk score on a 0-100 scale.")
    risk_level: str = Field(description="Bucketed risk label for the segment.")


class RouteSummary(BaseModel):
    """Aggregate route metrics."""

    route_id: str = Field(description="Unique identifier for the route option.")
    geometry: dict[str, Any] = Field(description="Route geometry as GeoJSON LineString.")
    distance_km: float = Field(description="Total route distance in kilometres.")
    duration_min: float = Field(description="Estimated route duration in minutes.")
    avg_risk_score: int = Field(description="Mean edge risk score across the route.")
    is_safest: bool = Field(default=False, description="Flag indicating if this is the safest option.")
    segments: list[RouteSegment] = Field(description="Segments traversed by the route.")


class SafeRouteResponse(BaseModel):
    """Response contract for POST /v1/route/safe."""

    alternatives: list[RouteSummary] = Field(description="List of alternative routes.")


def _risk_level(score: int) -> str:
    for upper, label in RISK_LEVELS:
        if score <= upper:
            return label
    return "CRITICAL"


def _as_float(value: Any, default: float = 0.0) -> float:
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


def _edge_id(u: Any, v: Any) -> str:
    return f"{u}_{v}"


@lru_cache(maxsize=1)
def _load_segment_rates() -> dict[str, float]:
    if not SEGMENT_RATES_PATH.exists():
        return {}
    frame = pd.read_parquet(SEGMENT_RATES_PATH)
    return {str(row["osmid"]): float(row["historical_accident_rate"]) for _, row in frame.iterrows()}


def _road_class(edge_data: dict[str, Any]) -> str:
    highway = edge_data.get("highway", "unclassified")
    if isinstance(highway, (list, tuple)):
        return str(highway[0]) if highway else "unclassified"
    return str(highway)


def _speed_limit(edge_data: dict[str, Any]) -> float:
    return _as_float(edge_data.get("speed_kph") or edge_data.get("maxspeed"), 50.0)


def _historical_rate(u: Any, v: Any) -> float:
    rates = _load_segment_rates()
    segment_id = _edge_id(u, v)
    return rates.get(segment_id, rates.get(_edge_id(v, u), rates.get(str(segment_id), 0.0)))


def _edge_feature_input(
    u: Any,
    v: Any,
    edge_data: dict[str, Any],
    weather: dict[str, float],
    timestamp: datetime,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "highway": _road_class(edge_data),
        "speed_limit_kmh": _speed_limit(edge_data),
        "precipitation_mm": weather["precipitation_mm"],
        "visibility_km": weather["visibility_km"],
        "wind_speed_ms": weather["wind_speed_ms"],
        "temperature_c": weather["temperature_c"],
        "historical_accident_rate": _historical_rate(u, v),
    }


def _score_graph_edges(
    graph: nx.MultiDiGraph,
    weather: dict[str, float],
    timestamp: datetime,
    paths: list[list[Any]] = None,
) -> dict[tuple[Any, Any, Any], float]:
    risk_scores: dict[tuple[Any, Any, Any], float] = {}
    edge_refs: list[tuple[Any, Any, Any]] = []
    feature_vectors: list[np.ndarray] = []

    # If paths are provided, only score edges that appear in these paths.
    # This optimization allows for sub-second responses on large regional graphs.
    if paths:
        target_edges = set()
        for path in paths:
            if not path: continue
            for u, v in zip(path, path[1:]):
                if graph.has_edge(u, v):
                    for key in graph[u][v]:
                        target_edges.add((u, v, key))
        
        for u, v, key in target_edges:
            data = graph.get_edge_data(u, v, key)
            edge_refs.append((u, v, key))
            feature_vectors.append(build_feature_vector(_edge_feature_input(u, v, data, weather, timestamp)))
    else:
        for u, v, key, data in graph.edges(keys=True, data=True):
            edge_refs.append((u, v, key))
            feature_vectors.append(build_feature_vector(_edge_feature_input(u, v, data, weather, timestamp)))

    if not feature_vectors:
        return risk_scores

    feature_matrix = np.vstack(feature_vectors)
    try:
        probabilities = load_model().predict_proba(feature_matrix)[:, 1]
        scores = [float(round(probability * 100.0)) for probability in probabilities]
    except Exception:
        scores = [float(run_inference(feature_vector)) for feature_vector in feature_vectors]

    for (u, v, key), score in zip(edge_refs, scores):
        risk_scores[(u, v, key)] = score
        risk_scores[(str(u), str(v), str(key))] = score

    return risk_scores


def _node_coordinate(graph: nx.MultiDiGraph, node: Any) -> list[float]:
    data = graph.nodes[node]
    return [_as_float(data.get("x")), _as_float(data.get("y"))]


def _edge_coordinates(graph: nx.MultiDiGraph, u: Any, v: Any, edge_data: dict[str, Any]) -> list[list[float]]:
    geometry = edge_data.get("geometry")
    if hasattr(geometry, "coords"):
        return [[float(x), float(y)] for x, y in geometry.coords]
    if isinstance(geometry, str):
        try:
            from shapely import wkt

            parsed = wkt.loads(geometry)
            if hasattr(parsed, "coords"):
                return [[float(x), float(y)] for x, y in parsed.coords]
        except Exception:
            pass
    return [_node_coordinate(graph, u), _node_coordinate(graph, v)]


def _edge_choices(graph: nx.MultiDiGraph, u: Any, v: Any) -> list[tuple[Any, dict[str, Any]]]:
    edge_map = graph.get_edge_data(u, v) or {}
    if edge_map and all(isinstance(value, dict) for value in edge_map.values()):
        return list(edge_map.items())
    return [(0, edge_map)]


def _select_edge(
    graph: nx.MultiDiGraph,
    u: Any,
    v: Any,
    risk_scores: dict[tuple[Any, Any, Any], float],
    alpha: float,
) -> tuple[Any, dict[str, Any]]:
    max_tt = travel_time_normalizer(graph)

    def edge_cost(item: tuple[Any, dict[str, Any]]) -> float:
        key, data = item
        risk = risk_scores.get((u, v, key), risk_scores.get((str(u), str(v), str(key)), 50.0))
        risk_component = min(max(risk / 100.0, 0.0), 1.0)
        travel_component = min(travel_time_seconds(data) / max_tt, 1.0)
        return alpha * risk_component + (1.0 - alpha) * travel_component

    return min(_edge_choices(graph, u, v), key=edge_cost)


def _route_edges(
    graph: nx.MultiDiGraph,
    path: list[Any],
    risk_scores: dict[tuple[Any, Any, Any], float],
    alpha: float,
) -> list[tuple[Any, Any, Any, dict[str, Any]]]:
    edges: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    for u, v in zip(path, path[1:]):
        key, data = _select_edge(graph, u, v, risk_scores, alpha)
        edges.append((u, v, key, data))
    return edges


def _route_geometry(graph: nx.MultiDiGraph, edges: list[tuple[Any, Any, Any, dict[str, Any]]]) -> dict[str, Any]:
    coordinates: list[list[float]] = []
    for u, v, _, data in edges:
        edge_coords = _edge_coordinates(graph, u, v, data)
        if coordinates and edge_coords and coordinates[-1] == edge_coords[0]:
            coordinates.extend(edge_coords[1:])
        else:
            coordinates.extend(edge_coords)
    return {"type": "LineString", "coordinates": coordinates}


def _route_summary(
    graph: nx.MultiDiGraph,
    path: list[Any],
    risk_scores: dict[tuple[Any, Any, Any], float],
    alpha: float = 1.0,
    route_id: str = "route_0",
) -> RouteSummary:
    edges = _route_edges(graph, path, risk_scores, alpha)
    distance_m = sum(_as_float(data.get("length"), 0.0) for _, _, _, data in edges)
    duration_s = sum(travel_time_seconds(data) for _, _, _, data in edges)
    segment_scores = [
        int(round(risk_scores.get((u, v, key), risk_scores.get((str(u), str(v), str(key)), 50.0))))
        for u, v, key, _ in edges
    ]
    avg_risk = int(round(sum(segment_scores) / len(segment_scores))) if segment_scores else 0
    segments = [
        RouteSegment(segment_id=_edge_id(u, v), risk_score=score, risk_level=_risk_level(score))
        for (u, v, _, _), score in zip(edges, segment_scores)
    ]
    return RouteSummary(
        route_id=route_id,
        geometry=_route_geometry(graph, edges),
        distance_km=round(distance_m / 1000.0, 3),
        duration_min=round(duration_s / 60.0, 1),
        avg_risk_score=avg_risk,
        segments=segments,
    )


@router.post(
    "/safe",
    response_model=SafeRouteResponse,
    summary="Compute A Safety-Aware Route",
    description=(
        "Snap origin and destination coordinates to the road graph, score graph edges "
        "with the M3 inference layer, run K-shortest distinct physical paths, and score each."
    ),
)
def post_safe_route(request: SafeRouteRequest) -> SafeRouteResponse:
    from app.routing.graph import get_graph_for_points, nearest_node
    graph = get_graph_for_points(
        request.origin.lat, request.origin.lon, 
        request.destination.lat, request.destination.lon
    )
    
    if not graph or len(graph) == 0:
        raise HTTPException(
            status_code=503, 
            detail="Road network data unavailable for this area. Please check your internet connection or try a different region."
        )

    origin_node = nearest_node(graph, request.origin.lat, request.origin.lon)
    destination_node = nearest_node(graph, request.destination.lat, request.destination.lon)

    weather_lat = (request.origin.lat + request.destination.lat) / 2.0
    weather_lon = (request.origin.lon + request.destination.lon) / 2.0
    weather = get_weather(weather_lat, weather_lon)

    from app.routing.astar import alternative_paths
    paths = alternative_paths(graph, origin_node, destination_node, k=5)
    
    # Filter out empty paths or single-node "paths"
    valid_paths = [p for p in paths if len(p) > 1]
    
    if not valid_paths:
        if origin_node == destination_node:
             detail = "Origin and destination are too close (snapped to the same junction)."
        else:
             detail = "No drivable route found. Keep clicks within the LA/Santa Monica area."
        raise HTTPException(status_code=404, detail=detail)

    # Scoped optimization: Only score edges in our candidate paths
    risk_scores = _score_graph_edges(graph, weather, datetime.utcnow(), paths=valid_paths)

    alternatives = []
    for idx, path in enumerate(valid_paths):
        summary = _route_summary(graph, path, risk_scores, alpha=request.alpha, route_id=f"route_{idx}")
        alternatives.append(summary)

    if alternatives:
        best_idx = min(range(len(alternatives)), key=lambda i: alternatives[i].avg_risk_score)
        alternatives[best_idx].is_safest = True

    return SafeRouteResponse(alternatives=alternatives)

