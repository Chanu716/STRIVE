"""Road graph loading and lookup helpers for routing."""
from __future__ import annotations

import os
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GRAPH_PATH = "data/raw/road_network.graphml"
CACHE_DIR = Path("data/cache/graphs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.timeout = 60

# ── Named-city bounding boxes ────────────────────────────────────────────────
# Any click inside these bounds loads the pre-downloaded file instantly.
CITY_BOUNDS = [
    # (name,       south,   north,   west,    east,          filename)
    ("LA",         33.70,   34.40,  -118.70, -118.00, "road_network.graphml",      True),  # Static file
    ("vijayawada", 16.40,   16.65,   80.40,   80.75,  "city_vijayawada.graphml",   False),
    ("vizag",      17.60,   17.85,   83.10,   83.45,  "city_vizag.graphml",        False),
    ("hyderabad",  17.25,   17.65,   78.25,   78.70,  "city_hyderabad.graphml",    False),
    ("delhi",      28.40,   28.85,   76.80,   77.45,  "city_delhi.graphml",        False),
    ("bengaluru",  12.80,   13.20,   77.40,   77.85,  "city_bengaluru.graphml",    False),
]

# In-process cache so we only load each file once per server lifecycle
_graph_cache: dict[str, nx.MultiDiGraph] = {}


def _load_named(filename: str, is_static: bool) -> nx.MultiDiGraph | None:
    """Load a named graph file from either the raw data dir or the cache dir."""
    if filename in _graph_cache:
        return _graph_cache[filename]

    path = Path("data/raw" if is_static else "data/cache/graphs") / filename
    if not path.exists():
        return None

    print(f"Loading named graph: {path}")
    G = ox.load_graphml(path)
    _graph_cache[filename] = G
    return G


def get_graph_for_points(lat1: float, lon1: float, lat2: float, lon2: float) -> nx.MultiDiGraph:
    """Return a graph that covers both points, fastest path first."""

    # 1. Check every pre-downloaded city region
    for name, south, north, west, east, fname, is_static in CITY_BOUNDS:
        if (south <= lat1 <= north and west <= lon1 <= east and
                south <= lat2 <= north and west <= lon2 <= east):
            G = _load_named(fname, is_static)
            if G:
                print(f"Using named city graph: {name}")
                return G

    # 2. Check generic bbox cache
    n = max(lat1, lat2) + 0.01
    s = min(lat1, lat2) - 0.01
    e = max(lon1, lon2) + 0.01
    w = min(lon1, lon2) - 0.01
    cache_key = hashlib.md5(f"{n}{s}{e}{w}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"bbox_{cache_key}.graphml"
    if cache_path.exists():
        print(f"Cache hit: {cache_path}")
        return ox.load_graphml(cache_path)

    # 3. Last resort — download from OSM (slow, first-time only)
    return _download_and_cache(lat1, lon1, lat2, lon2, cache_path)


def _download_and_cache(lat1, lon1, lat2, lon2, cache_path) -> nx.MultiDiGraph:
    import time
    center_lat = (lat1 + lat2) / 2.0
    center_lon = (lon1 + lon2) / 2.0
    try:
        dist_m = ox.distance.great_circle_vec(lat1, lon1, lat2, lon2)
    except Exception:
        dist_m = 1000.0
    radius = min(max(1500, int(dist_m * 1.3)), 8000)

    print(f"--- OSM DOWNLOAD: radius={radius}m @ {center_lat},{center_lon} ---")
    for attempt in range(2):
        try:
            G = ox.graph_from_point((center_lat, center_lon), dist=radius,
                                    network_type="drive", simplify=True)
            if G and len(G) > 0:
                ox.save_graphml(G, cache_path)
                print(f"--- SAVED to {cache_path} ---")
                return G
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    return nx.MultiDiGraph()


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> Any:
    if not graph or len(graph) == 0:
        return None
    return ox.nearest_nodes(graph, X=lon, Y=lat)
