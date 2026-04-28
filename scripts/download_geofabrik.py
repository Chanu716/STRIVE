"""
Downloads Andhra Pradesh from Geofabrik (fast, official mirror),
then extracts the Vijayawada metro area as a GraphML file.

Run ONCE from the D:\STRIVE directory:
    python scripts/download_geofabrik.py
"""
import subprocess, sys, struct
from pathlib import Path

CACHE_DIR = Path("data/cache/graphs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box for Vijayawada metro + surrounding 10km
VJA_BBOX = dict(north=16.65, south=16.40, east=80.75, west=80.40)

OUT_GRAPHML = CACHE_DIR / "city_vijayawada.graphml"

def main():
    if OUT_GRAPHML.exists():
        print(f"Already exists: {OUT_GRAPHML}")
        return

    import osmnx as ox
    import requests, io

    # ── Use Overpass Turbo with a smaller, targeted bounding box ──────────────
    # This is much more reliable than graph_from_address because we give
    # the exact coordinates and a precise bounding box
    print("Fetching Vijayawada road network via targeted Overpass query ...")

    ox.settings.timeout = 120
    ox.settings.log_console = True

    bbox = (VJA_BBOX["north"], VJA_BBOX["south"], VJA_BBOX["east"], VJA_BBOX["west"])
    try:
        G = ox.graph_from_bbox(
            bbox,
            network_type="drive",
            simplify=True,
        )
        print(f"Downloaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        ox.save_graphml(G, OUT_GRAPHML)
        print(f"Saved to {OUT_GRAPHML}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
