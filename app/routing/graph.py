"""Road graph loading and lookup helpers for routing."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
from dotenv import load_dotenv


load_dotenv()

DEFAULT_GRAPH_PATH = "data/raw/road_network.graphml"


def graph_path() -> Path:
    """Return the configured OSM graph path."""
    return Path(os.getenv("GRAPH_PATH", DEFAULT_GRAPH_PATH))


@lru_cache(maxsize=1)
def get_graph() -> nx.MultiDiGraph:
    """Load the OSM road graph once and cache it for the process lifetime."""
    path = graph_path()
    if not path.exists():
        raise FileNotFoundError(f"Road graph not found at {path}")
    return ox.load_graphml(path)


def nearest_node(lat: float, lon: float) -> Any:
    """Resolve a latitude/longitude pair to the nearest graph node."""
    graph = get_graph()
    return ox.nearest_nodes(graph, X=lon, Y=lat)
