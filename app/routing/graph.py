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
    # (name, south, north, west, east, filename, is_in_raw_dir)
    ("LA",              33.70, 34.40, -118.70, -118.00, "road_network.graphml",         True),
    ("vijayawada",      16.44, 16.62,  80.52,   80.80,  "city_vijayawada.graphml",      False),
    ("vijayawada_west", 16.40, 16.51,  80.46,   80.57,  "city_vijayawada_west.graphml", False),
    ("vizag",           17.60, 17.85,  83.10,   83.45,  "city_vizag.graphml",           False),
    ("hyderabad",       17.25, 17.65,  78.25,   78.70,  "city_hyderabad.graphml",       False),
    ("delhi",           28.40, 28.85,  76.80,   77.45,  "city_delhi.graphml",           False),
    ("bengaluru",       12.80, 13.20,  77.40,   77.85,  "city_bengaluru.graphml",       False),
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


def _graph_covers(G: nx.MultiDiGraph, lat: float, lon: float, margin: float = 0.005) -> bool:
    """True if a point is within the actual node extent of the graph."""
    if "bounds" not in G.graph:
        lons = [d['x'] for _, d in G.nodes(data=True)]
        lats = [d['y'] for _, d in G.nodes(data=True)]
        if not lons:
            return False
        G.graph["bounds"] = (min(lats), max(lats), min(lons), max(lons))
    
    min_lat, max_lat, min_lon, max_lon = G.graph["bounds"]
    return (min_lat - margin <= lat <= max_lat + margin and
            min_lon - margin <= lon <= max_lon + margin)


def get_graph_for_points(lat1: float, lon1: float, lat2: float, lon2: float) -> nx.MultiDiGraph:
    """Return a graph that covers both points, fastest path first."""

    # 1. Check every pre-downloaded city region (by bounding box first)
    # Use a tighter internal check to avoid 'edge effects' where nodes are sparse
    for name, south, north, west, east, fname, is_static in CITY_BOUNDS:
        if (south <= lat1 <= north and west <= lon1 <= east and
                south <= lat2 <= north and west <= lon2 <= east):
            G = _load_named(fname, is_static)
            if G:
                # Validate actual node coverage with a safety margin
                if _graph_covers(G, lat1, lon1, margin=0.0) and _graph_covers(G, lat2, lon2, margin=0.0):
                    print(f"Using named city graph: {name}")
                    return G
                else:
                    print(f"Points near boundary of {name}, falling back to download for safety.")

    # 2. Check generic bbox cache
    s_val, n_val = min(lat1, lat2), max(lat1, lat2)
    w_val, e_val = min(lon1, lon2), max(lon1, lon2)
    
    # --- Distance Guard ---
    # Inter-city routing (>50km) is too large for real-time OSM download
    from osmnx.distance import great_circle
    if great_circle(lat1, lon1, lat2, lon2) > 50000:
        print("Distance too large for live download fallback.")
        return nx.MultiDiGraph()

    # Add a massive 3km buffer (approx 0.03 degrees)
    # Optimized for fast downloads and high connectivity
    buffer = 0.03
    north, south, east, west = n_val + buffer, s_val - buffer, e_val + buffer, w_val - buffer
    
    # Cache key v4
    cache_key = hashlib.md5(f"v4_{north:.4f}{south:.4f}{east:.4f}{west:.4f}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"bbox_{cache_key}.graphml"
    
    if cache_path.exists():
        return ox.load_graphml(cache_path)

    # 3. Last resort — download from OSM with multi-server failover
    # Rotate Overpass servers to bypass 503 rate limits
    overpass_servers = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://overpass.osm.ch/api/interpreter"
    ]
    
    ox.settings.timeout = 180
    ox.settings.requests_timeout = 180
    ox.settings.user_agent = f"STRIVE_Intelligence_Platform_v1.2_{hash(str(lat1))}"
    
    # Calculate a center point and radius for point-based fallback (often more stable)
    center_lat, center_lon = (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0
    from osmnx.distance import great_circle
    radius = great_circle(lat1, lon1, lat2, lon2) / 2.0 + 3000 # dist/2 + 3km buffer

    return _download_with_failover(center_lat, center_lon, radius, cache_path, overpass_servers)


def _download_with_failover(lat, lon, dist, cache_path, servers) -> nx.MultiDiGraph:
    import time
    for server in servers:
        ox.settings.overpass_url = server
        print(f"--- ATTEMPTING OSM DOWNLOAD (Radius={dist:.0f}m) via {server} ---")
        for attempt in range(2):
            try:
                G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive", simplify=True)
                if G and len(G) > 0:
                    # Resilient Connectivity Filter
                    try:
                        G_sc = ox.utils_graph.get_largest_component(G, strongly=True)
                        if len(G_sc) > (len(G) * 0.1): G = G_sc
                    except: pass
                    
                    try:
                        G = ox.utils_graph.get_largest_component(G, strongly=False)
                    except: pass

                    ox.save_graphml(G, cache_path)
                    print(f"--- SUCCESS: Map saved from {server} ---")
                    return G
            except Exception as e:
                print(f"Attempt {attempt+1} on {server} failed: {e}")
                time.sleep(2)
    return nx.MultiDiGraph()
    return nx.MultiDiGraph()


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> Any:
    if not graph or len(graph) == 0:
        return None
    return ox.nearest_nodes(graph, X=lon, Y=lat)
