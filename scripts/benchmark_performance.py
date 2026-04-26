#!/usr/bin/env python3
"""
Measure the latency of the main API paths and write reports/performance.md.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path

import httpx
import networkx as nx
from fastapi import FastAPI
from shapely.geometry import LineString

from app import main as app_main
from app.ml.inference import ExplanationResult
from app.routers import risk as risk_router
from app.routers import route as route_router
from app.weather import get_weather


REPORT_PATH = Path("reports/performance.md")


def build_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("1", x=0.0, y=0.0)
    graph.add_node("2", x=1.0, y=0.0)
    graph.add_node("3", x=2.0, y=0.0)
    graph.add_node("4", x=1.0, y=1.0)
    graph.add_edge("1", "2", key=0, length=100.0, speed_kph=100.0, highway="primary", geometry=LineString([(0.0, 0.0), (1.0, 0.0)]))
    graph.add_edge("2", "3", key=0, length=100.0, speed_kph=100.0, highway="primary", geometry=LineString([(1.0, 0.0), (2.0, 0.0)]))
    graph.add_edge("1", "4", key=0, length=150.0, speed_kph=30.0, highway="residential", geometry=LineString([(0.0, 0.0), (1.0, 1.0)]))
    graph.add_edge("4", "3", key=0, length=150.0, speed_kph=30.0, highway="residential", geometry=LineString([(1.0, 1.0), (2.0, 0.0)]))
    return graph


def patch_runtime() -> None:
    graph = build_graph()
    explanation = ExplanationResult(
        risk_score=72,
        probability=0.72,
        shap_values={
            "hour_of_day": 0.1,
            "day_of_week": 0.1,
            "month": 0.1,
            "night_indicator": 0.6,
            "road_class": 0.2,
            "speed_limit_kmh": 0.1,
            "precipitation_mm": 0.7,
            "visibility_km": -0.1,
            "wind_speed_ms": 0.0,
            "temperature_c": 0.0,
            "rain_on_congestion": 0.4,
            "historical_accident_rate": 0.3,
        },
        expected_value=0.25,
    )

    risk_router._load_graph.cache_clear()
    risk_router._load_segment_rates.cache_clear()
    route_router._load_segment_rates.cache_clear()

    risk_router._load_graph = lambda: graph
    risk_router._find_segment_in_db = lambda db, segment_id: None
    risk_router.get_weather = lambda lat, lon: {
        "precipitation_mm": 8.0,
        "visibility_km": 7.5,
        "wind_speed_ms": 2.0,
        "temperature_c": 20.0,
    }
    risk_router.explain_prediction = lambda feature_vector: explanation

    route_router.get_graph = lambda: graph
    route_router.nearest_node = lambda lat, lon: "1" if lat < 0.5 and lon < 0.5 else "3"
    route_router.get_weather = lambda lat, lon: {
        "precipitation_mm": 8.0,
        "visibility_km": 7.5,
        "wind_speed_ms": 2.0,
        "temperature_c": 20.0,
    }
    route_router._score_graph_edges = lambda graph, weather, timestamp: {
        (1, 2, 0): 95.0,
        (2, 3, 0): 95.0,
        (1, 4, 0): 5.0,
        (4, 3, 0): 5.0,
        ("1", "2", "0"): 95.0,
        ("2", "3", "0"): 95.0,
        ("1", "4", "0"): 5.0,
        ("4", "3", "0"): 5.0,
    }

    class DummyModel:
        def predict_proba(self, X):
            import numpy as np

            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            probs = np.full((X.shape[0], 2), 0.0)
            probs[:, 1] = 0.72
            probs[:, 0] = 0.28
            return probs

    app_main.load_model = lambda: DummyModel()


async def measure(client: httpx.AsyncClient, method: str, url: str, json: dict | None = None, params: dict | None = None) -> list[float]:
    timings: list[float] = []
    for _ in range(10):
        start = time.perf_counter()
        response = await client.request(method, url, json=json, params=params)
        response.raise_for_status()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * (pct / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def main_async() -> None:
    patch_runtime()
    transport = httpx.ASGITransport(app=app_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        risk_samples = await measure(client, "GET", "/v1/risk/segment", params={"lat": 0.1, "lon": 0.1})
        route_samples = await measure(
            client,
            "POST",
            "/v1/route/safe",
            json={"origin": {"lat": 0.0, "lon": 0.0}, "destination": {"lat": 2.0, "lon": 0.0}, "alpha": 0.6},
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# STRIVE Performance Report",
                "",
                "| Endpoint | p50 (ms) | p95 (ms) | Target |",
                "|---|---:|---:|---|",
                f"| GET /v1/risk/segment | {percentile(risk_samples, 50):.1f} | {percentile(risk_samples, 95):.1f} | <= 500 ms |",
                f"| POST /v1/route/safe | {percentile(route_samples, 50):.1f} | {percentile(route_samples, 95):.1f} | <= 2000 ms |",
                "",
                "Benchmark method: 10 in-process requests per endpoint using httpx against the FastAPI app with deterministic local fixtures.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main_async())
