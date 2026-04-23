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
from app.routing.graph import get_graph, nearest_node
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


class RouteSummary(BaseModel):
    """Aggregate route metrics."""

    geometry: dict[str, Any] = Field(description="Route geometry as GeoJSON LineString.")
    distance_km: float = Field(description="Total route distance in kilometres.")
    duration_min: float = Field(description="Estimated route duration in minutes.")
    avg_risk_score: int = Field(description="Mean edge risk score across the route.")


class RouteSegment(BaseModel):
    """Risk score for one traversed route segment."""

    segment_id: str = Field(description="Road segment identifier.")
    risk_score: int = Field(description="Predicted risk score on a 0-100 scale.")
    risk_level: str = Field(description="Bucketed risk label for the segment.")


class SafeRouteResponse(BaseModel):
    """Response contract for POST /v1/route/safe."""

    route: RouteSummary = Field(description="Safety-aware route metrics.")
    fastest: RouteSummary = Field(description="Fastest-route baseline metrics.")
    risk_reduction_pct: float = Field(description="Average risk reduction compared with the fastest route.")
    segments: list[RouteSegment] = Field(description="Segments traversed by the safety-aware route.")


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
) -> dict[tuple[Any, Any, Any], float]:
    risk_scores: dict[tuple[Any, Any, Any], float] = {}
    edge_refs: list[tuple[Any, Any, Any]] = []
    feature_vectors: list[np.ndarray] = []

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
    alpha: float,
) -> tuple[RouteSummary, list[RouteSegment]]:
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
    return (
        RouteSummary(
            geometry=_route_geometry(graph, edges),
            distance_km=round(distance_m / 1000.0, 3),
            duration_min=round(duration_s / 60.0, 1),
            avg_risk_score=avg_risk,
        ),
        segments,
    )


@router.post(
    "/safe",
    response_model=SafeRouteResponse,
    summary="Compute A Safety-Aware Route",
    description=(
        "Snap origin and destination coordinates to the road graph, score graph edges "
        "with the M3 inference layer, run risk-weighted A*, and compare against the fastest route."
    ),
)
def post_safe_route(request: SafeRouteRequest) -> SafeRouteResponse:
    graph = get_graph()
    origin_node = nearest_node(request.origin.lat, request.origin.lon)
    destination_node = nearest_node(request.destination.lat, request.destination.lon)

    weather_lat = (request.origin.lat + request.destination.lat) / 2.0
    weather_lon = (request.origin.lon + request.destination.lon) / 2.0
    weather = get_weather(weather_lat, weather_lon)
    risk_scores = _score_graph_edges(graph, weather, datetime.utcnow())

    try:
        safe_path = safe_route(graph, origin_node, destination_node, request.alpha, risk_scores)
        fastest_path = safe_route(graph, origin_node, destination_node, 0.0, risk_scores)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise HTTPException(status_code=404, detail="No drivable route found between the requested points.") from exc

    route_summary, segments = _route_summary(graph, safe_path, risk_scores, request.alpha)
    fastest_summary, _ = _route_summary(graph, fastest_path, risk_scores, 0.0)

    if fastest_summary.avg_risk_score <= 0:
        risk_reduction_pct = 0.0
    else:
        reduction = fastest_summary.avg_risk_score - route_summary.avg_risk_score
        risk_reduction_pct = round((reduction / fastest_summary.avg_risk_score) * 100.0, 1)

    return SafeRouteResponse(
        route=route_summary,
        fastest=fastest_summary,
        risk_reduction_pct=risk_reduction_pct,
        segments=segments,
    )
