"""Explainability endpoints for STRIVE risk scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ml.features import FEATURE_NAMES, build_feature_vector
from app.ml.inference import explain_prediction
from app.routers.risk import _build_raw_input, _resolve_segment
from app.weather import get_weather


router = APIRouter(prefix="/v1/explain", tags=["explain"])


class SegmentExplainResponse(BaseModel):
    """Full explainability payload for a scored segment."""

    segment_id: str = Field(description="Resolved road-segment identifier.")
    risk_score: int = Field(description="Predicted risk score on a 0-100 scale.")
    features: dict[str, float] = Field(description="Feature values used for model inference.")
    shap_values: dict[str, float] = Field(description="SHAP value for every feature in the model input.")
    expected_value: float = Field(description="Model expected value used as the explanation baseline.")


@router.get(
    "/segment",
    response_model=SegmentExplainResponse,
    summary="Explain A Single Segment Score",
    description=(
        "Resolve the nearest road segment, score it with the risk model, and return every "
        "feature value and SHAP contribution for research inspection."
    ),
)
def explain_segment(
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
) -> SegmentExplainResponse:
    timestamp = dt or datetime.utcnow()
    segment = _resolve_segment(db, lat, lon)
    weather = get_weather(lat, lon)
    feature_vector = build_feature_vector(_build_raw_input(segment, weather, timestamp))
    explanation = explain_prediction(feature_vector)

    return SegmentExplainResponse(
        segment_id=segment["segment_id"],
        risk_score=explanation.risk_score,
        features={name: float(feature_vector[index]) for index, name in enumerate(FEATURE_NAMES)},
        shap_values=explanation.shap_values,
        expected_value=explanation.expected_value,
    )
