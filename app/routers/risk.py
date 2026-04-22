"""Risk scoring endpoints for STRIVE."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import networkx as nx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import RoadSegment
from app.db.session import get_db
from app.ml.features import FEATURE_NAMES, build_feature_vector
from app.ml.inference import explain_prediction
from app.weather import get_weather


router = APIRouter(prefix="/v1/risk", tags=["risk"])

GRAPH_PATH = Path("data/raw/road_network.graphml")
SEGMENT_RATES_PATH = Path("data/processed/segment_rates.parquet")

RISK_LEVELS = (
    (24, "LOW"),
    (49, "MODERATE"),
    (74, "HIGH"),
    (100, "CRITICAL"),
)

FEATURE_LABELS = {
    "precipitation_mm": "heavy rain",
    "visibility_km": "reduced visibility",
    "wind_speed_ms": "strong wind",
    "temperature_c": "temperature conditions",
    "historical_accident_rate": "high historical crash rate",
    "night_indicator": "night-time conditions",
    "road_class": "road class",
    "speed_limit_kmh": "higher speed limit",
    "rain_on_congestion": "rain on slower corridors",
}


class ShapFactor(BaseModel):
    """Top explanatory factor for a risk score."""

    feature: str = Field(description="Model feature name.")
    value: float = Field(description="Feature value used for this prediction.")
    shap: float = Field(description="Feature contribution for the predicted score.")


class SegmentRiskResponse(BaseModel):
    """Response contract for GET /v1/risk/segment."""

    segment_id: str = Field(description="Resolved road-segment identifier.")
    risk_score: int = Field(description="Predicted risk score on a 0-100 scale.")
    risk_level: str = Field(description="Bucketed risk label derived from the score.")
    shap_top_factors: list[ShapFactor] = Field(description="Top four features ranked by absolute SHAP value.")
    shap_summary: str = Field(description="Plain-English summary of the most important drivers.")


class HeatmapFeature(BaseModel):
    """Single GeoJSON feature in the heatmap response."""

    type: str = Field(default="Feature", description="GeoJSON feature type.")
    geometry: dict[str, Any] = Field(description="Segment geometry as a GeoJSON LineString.")
    properties: dict[str, Any] = Field(description="Heatmap properties for this segment.")


class HeatmapResponse(BaseModel):
    """GeoJSON feature collection of segment risks inside the requested bounding box."""

    type: str = Field(default="FeatureCollection", description="GeoJSON collection type.")
    features: list[HeatmapFeature] = Field(description="Segments scored inside the bounding box.")


def _risk_level(score: int) -> str:
    for upper, label in RISK_LEVELS:
        if score <= upper:
            return label
    return "CRITICAL"


@lru_cache(maxsize=1)
def _load_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Road graph not found at {GRAPH_PATH}")
    return nx.read_graphml(GRAPH_PATH)


@lru_cache(maxsize=1)
def _load_segment_rates() -> dict[str, float]:
    if not SEGMENT_RATES_PATH.exists():
        return {}
    frame = pd.read_parquet(SEGMENT_RATES_PATH)
    return {str(row["osmid"]): float(row["historical_accident_rate"]) for _, row in frame.iterrows()}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_id(u: str, v: str) -> str:
    return f"{u}_{v}"


def _edge_geometry(graph: nx.MultiDiGraph, u: str, v: str) -> dict[str, Any]:
    node_u = graph.nodes[u]
    node_v = graph.nodes[v]
    return {
        "type": "LineString",
        "coordinates": [
            [_as_float(node_u.get("x")), _as_float(node_u.get("y"))],
            [_as_float(node_v.get("x")), _as_float(node_v.get("y"))],
        ],
    }


def _edge_distance_sq(graph: nx.MultiDiGraph, u: str, v: str, lat: float, lon: float) -> float:
    geometry = _edge_geometry(graph, u, v)
    coords = geometry["coordinates"]
    return min((lon - x) ** 2 + (lat - y) ** 2 for x, y in coords)


def _resolve_graph_segment(lat: float, lon: float) -> dict[str, Any]:
    graph = _load_graph()
    nearest: dict[str, Any] | None = None

    for u, v, key, data in graph.edges(keys=True, data=True):
        distance = _edge_distance_sq(graph, str(u), str(v), lat, lon)
        if nearest is None or distance < nearest["distance"]:
            nearest = {
                "segment_id": _segment_id(str(u), str(v)),
                "u": int(u),
                "v": int(v),
                "geometry": _edge_geometry(graph, str(u), str(v)),
                "road_class": str(data.get("highway", "unclassified")),
                "speed_limit_kmh": _as_float(data.get("speed_kph") or data.get("maxspeed"), 50.0),
                "length_m": _as_float(data.get("length"), 0.0),
                "distance": distance,
            }

    if nearest is None:
        raise HTTPException(status_code=404, detail="No road segment could be resolved for the given coordinates.")

    segment_rates = _load_segment_rates()
    nearest["historical_accident_rate"] = segment_rates.get(
        nearest["segment_id"],
        segment_rates.get(f"{nearest['v']}_{nearest['u']}", 0.0),
    )
    return nearest


def _find_segment_in_db(db: Session, segment_id: str) -> RoadSegment | None:
    try:
        return db.get(RoadSegment, segment_id)
    except Exception:
        return None


def _resolve_segment(db: Session, lat: float, lon: float) -> dict[str, Any]:
    segment = _resolve_graph_segment(lat, lon)
    db_row = _find_segment_in_db(db, segment["segment_id"])
    if db_row is None:
        return segment

    return {
        "segment_id": db_row.segment_id,
        "u": db_row.u,
        "v": db_row.v,
        "geometry": db_row.geometry,
        "road_class": db_row.road_class,
        "speed_limit_kmh": db_row.speed_limit_kmh,
        "length_m": db_row.length_m,
        "historical_accident_rate": db_row.historical_accident_rate,
    }


def _build_raw_input(segment: dict[str, Any], weather: dict[str, float], timestamp: datetime) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "highway": segment.get("road_class", "unclassified"),
        "speed_limit_kmh": segment.get("speed_limit_kmh", 50.0),
        "precipitation_mm": weather["precipitation_mm"],
        "visibility_km": weather["visibility_km"],
        "wind_speed_ms": weather["wind_speed_ms"],
        "temperature_c": weather["temperature_c"],
        "historical_accident_rate": segment.get("historical_accident_rate", 0.0),
    }


def _summarize_top_factors(top_factors: list[ShapFactor]) -> str:
    if not top_factors:
        return "Risk is driven by baseline model conditions."
    labels = [FEATURE_LABELS.get(item.feature, item.feature.replace("_", " ")) for item in top_factors[:2]]
    if len(labels) == 1:
        return f"Risk is mainly driven by {labels[0]}."
    return f"Risk is mainly driven by {labels[0]} and {labels[1]}."


def _top_factors(feature_vector: Any, shap_values: dict[str, float]) -> list[ShapFactor]:
    factors: list[ShapFactor] = []
    for index, feature in enumerate(FEATURE_NAMES):
        factors.append(
            ShapFactor(
                feature=feature,
                value=float(feature_vector[index]),
                shap=float(shap_values.get(feature, 0.0)),
            )
        )
    return sorted(factors, key=lambda item: abs(item.shap), reverse=True)[:4]


def _score_segment(segment: dict[str, Any], weather: dict[str, float], timestamp: datetime) -> tuple[int, str]:
    feature_vector = build_feature_vector(_build_raw_input(segment, weather, timestamp))
    score = explain_prediction(feature_vector).risk_score
    return score, _risk_level(score)


@router.get(
    "/segment",
    response_model=SegmentRiskResponse,
    summary="Score A Single Road Segment",
    description=(
        "Resolve the nearest road segment for the provided coordinates, fetch weather, "
        "build the feature vector, and return the predicted risk score plus the top SHAP factors."
    ),
)
def get_segment_risk(
    lat: Annotated[float, Query(description="Latitude of the query point.", examples=[34.05])],
    lon: Annotated[float, Query(description="Longitude of the query point.", examples=[-118.24])],
    dt: Annotated[
        datetime | None,
        Query(
            alias="datetime",
            description="Optional ISO timestamp used for feature engineering. Defaults to current UTC time.",
            examples=["2024-06-15T22:00:00"],
        ),
    ] = None,
    db: Session = Depends(get_db),
) -> SegmentRiskResponse:
    timestamp = dt or datetime.utcnow()
    segment = _resolve_segment(db, lat, lon)
    weather = get_weather(lat, lon)
    feature_vector = build_feature_vector(_build_raw_input(segment, weather, timestamp))
    explanation = explain_prediction(feature_vector)
    top_factors = _top_factors(feature_vector, explanation.shap_values)

    return SegmentRiskResponse(
        segment_id=segment["segment_id"],
        risk_score=explanation.risk_score,
        risk_level=_risk_level(explanation.risk_score),
        shap_top_factors=top_factors,
        shap_summary=_summarize_top_factors(top_factors),
    )


def _bbox_contains(geometry: dict[str, Any], min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
    for lon, lat in geometry.get("coordinates", []):
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True
    return False


def _segments_in_bbox(db: Session, bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    graph = _load_graph()
    segments: list[dict[str, Any]] = []
    segment_rates = _load_segment_rates()

    for u, v, key, data in graph.edges(keys=True, data=True):
        geometry = _edge_geometry(graph, str(u), str(v))
        if not _bbox_contains(geometry, min_lon, min_lat, max_lon, max_lat):
            continue
        segment_id = _segment_id(str(u), str(v))
        db_row = _find_segment_in_db(db, segment_id)
        if db_row is not None:
            segments.append(
                {
                    "segment_id": db_row.segment_id,
                    "u": db_row.u,
                    "v": db_row.v,
                    "geometry": db_row.geometry,
                    "road_class": db_row.road_class,
                    "speed_limit_kmh": db_row.speed_limit_kmh,
                    "length_m": db_row.length_m,
                    "historical_accident_rate": db_row.historical_accident_rate,
                }
            )
            continue

        segments.append(
            {
                "segment_id": segment_id,
                "u": int(u),
                "v": int(v),
                "geometry": geometry,
                "road_class": str(data.get("highway", "unclassified")),
                "speed_limit_kmh": _as_float(data.get("speed_kph") or data.get("maxspeed"), 50.0),
                "length_m": _as_float(data.get("length"), 0.0),
                "historical_accident_rate": segment_rates.get(segment_id, segment_rates.get(f"{v}_{u}", 0.0)),
            }
        )

    return segments


@router.get(
    "/heatmap",
    response_model=HeatmapResponse,
    summary="Build A Risk Heatmap",
    description=(
        "Score all road segments inside the requested bounding box and return a GeoJSON "
        "FeatureCollection for map rendering."
    ),
)
def get_risk_heatmap(
    bbox: Annotated[
        str,
        Query(
            description="Bounding box formatted as min_lon,min_lat,max_lon,max_lat.",
            examples=["-118.3,34.0,-118.2,34.1"],
        ),
    ],
    db: Session = Depends(get_db),
) -> HeatmapResponse:
    try:
        parsed_bbox = tuple(float(value.strip()) for value in bbox.split(","))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="bbox must be min_lon,min_lat,max_lon,max_lat") from exc

    if len(parsed_bbox) != 4:
        raise HTTPException(status_code=422, detail="bbox must contain exactly four comma-separated values")

    min_lon, min_lat, max_lon, max_lat = parsed_bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(status_code=422, detail="bbox min values must be lower than max values")

    centroid_lat = (min_lat + max_lat) / 2.0
    centroid_lon = (min_lon + max_lon) / 2.0
    weather = get_weather(centroid_lat, centroid_lon)
    timestamp = datetime.utcnow()

    features: list[HeatmapFeature] = []
    for segment in _segments_in_bbox(db, parsed_bbox):
        score, level = _score_segment(segment, weather, timestamp)
        features.append(
            HeatmapFeature(
                geometry=segment["geometry"],
                properties={
                    "segment_id": segment["segment_id"],
                    "risk_score": score,
                    "risk_level": level,
                },
            )
        )

    return HeatmapResponse(features=features)
