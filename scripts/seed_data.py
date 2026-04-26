#!/usr/bin/env python3
"""
Seed the STRIVE database from the processed parquet artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import osmnx as ox
from shapely.geometry import LineString, mapping
from sqlalchemy import delete, insert

from app.db.models import Accident, Base, RoadSegment
from app.db.session import engine, init_db


REPO_ROOT = Path(__file__).resolve().parent.parent


def _edge_geometry(data: dict[str, object], u: object, v: object, graph: object) -> dict:
    geometry = data.get("geometry")
    if hasattr(geometry, "__geo_interface__"):
        return mapping(geometry)
    if hasattr(geometry, "coords"):
        return mapping(geometry)
    node_u = graph.nodes[u]
    node_v = graph.nodes[v]
    return {
        "type": "LineString",
        "coordinates": [
            [float(node_u.get("x", 0.0)), float(node_u.get("y", 0.0))],
            [float(node_v.get("x", 0.0)), float(node_v.get("y", 0.0))],
        ],
    }


def _road_class(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unclassified"
    return str(value or "unclassified")


def _speed_limit(data: dict[str, object]) -> float:
    value = data.get("speed_kph") or data.get("maxspeed") or 50.0
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 50.0
    try:
        return float(str(value).replace("mph", "").replace("km/h", "").strip())
    except ValueError:
        return 50.0


def _segment_rows(graph_path: Path, rates: pd.DataFrame) -> list[dict[str, object]]:
    graph = ox.load_graphml(graph_path)
    rate_map = {str(row["osmid"]): float(row["historical_accident_rate"]) for _, row in rates.iterrows()}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for u, v, key, data in graph.edges(keys=True, data=True):
        segment_id = f"{u}_{v}"
        if segment_id in seen:
            continue
        seen.add(segment_id)
        rows.append(
            {
                "segment_id": segment_id,
                "u": int(u),
                "v": int(v),
                "geometry": _edge_geometry(data, u, v, graph),
                "road_class": _road_class(data.get("highway", "unclassified")),
                "speed_limit_kmh": _speed_limit(data),
                "length_m": float(data.get("length", 0.0)),
                "historical_accident_rate": rate_map.get(segment_id, rate_map.get(f"{v}_{u}", 0.0)),
            }
        )
    return rows


def _timestamp_frame(accidents: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(
        accidents[["year", "month", "day", "hour", "minute"]].rename(columns={"minute": "minute"}),
        errors="coerce",
    )
    return timestamp.fillna(pd.Timestamp("2022-01-01 00:00:00"))


def _severity_frame(accidents: pd.DataFrame) -> pd.Series:
    fatalities = accidents.get("fatalities", pd.Series([0] * len(accidents))).fillna(0)
    drunk = accidents.get("drunk_drivers", pd.Series([0] * len(accidents))).fillna(0)
    severity = pd.Series(3, index=accidents.index, dtype="int64")
    severity.loc[fatalities > 0] = 1
    severity.loc[(fatalities <= 0) & (drunk > 0)] = 2
    return severity


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the STRIVE database.")
    parser.add_argument("--snapped", default="data/processed/accidents_snapped.parquet")
    parser.add_argument("--rates", default="data/processed/segment_rates.parquet")
    parser.add_argument("--network", default="data/raw/road_network.graphml")
    args = parser.parse_args()

    snapped_path = REPO_ROOT / args.snapped
    rates_path = REPO_ROOT / args.rates
    network_path = REPO_ROOT / args.network

    snapped = pd.read_parquet(snapped_path)
    rates = pd.read_parquet(rates_path)

    init_db()
    Base.metadata.create_all(bind=engine)

    segment_rows = _segment_rows(network_path, rates)
    accident_rows = snapped.copy()
    accident_rows["timestamp"] = _timestamp_frame(accident_rows)
    accident_rows["severity"] = _severity_frame(accident_rows)

    road_segment_table = RoadSegment.__table__
    accident_table = Accident.__table__

    with engine.begin() as connection:
        connection.execute(delete(accident_table))
        connection.execute(delete(road_segment_table))
        if segment_rows:
            connection.execute(insert(road_segment_table), segment_rows)

        accident_payload = [
            {
                "segment_id": str(row["osmid"]),
                "timestamp": row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                "severity": int(row["severity"]),
            }
            for _, row in accident_rows.iterrows()
        ]
        if accident_payload:
            connection.execute(insert(accident_table), accident_payload)

    print(f"Seeded {len(segment_rows)} road segments and {len(accident_rows)} accidents.")


if __name__ == "__main__":
    main()
