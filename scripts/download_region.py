"""
Pre-downloads road networks for specified cities and saves them 
as named files. Run this script ONCE on the host machine.
Usage: python scripts/download_region.py
"""
import osmnx as ox
from pathlib import Path
import sys

CACHE_DIR = Path("data/cache/graphs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Add cities here: (name_for_filename, osm_search_address, radius_meters)
CITIES = [
    ("vijayawada",    "Vijayawada, Andhra Pradesh, India", 8000),
    ("vizag",         "Visakhapatnam, Andhra Pradesh, India", 8000),
    ("hyderabad",     "Hyderabad, Telangana, India", 8000),
    ("delhi",         "New Delhi, India", 8000),
    ("bengaluru",     "Bengaluru, Karnataka, India", 8000),
]

def download(slug, address, dist):
    out = CACHE_DIR / f"city_{slug}.graphml"
    if out.exists():
        print(f"[SKIP] {slug} already cached at {out}")
        return
    print(f"[DOWNLOADING] {address} (radius={dist}m) ...")
    try:
        G = ox.graph_from_address(address, dist=dist, network_type="drive", simplify=True)
        ox.save_graphml(G, out)
        print(f"[OK] Saved → {out}  ({G.number_of_nodes()} nodes)")
    except Exception as e:
        print(f"[FAIL] {address}: {e}", file=sys.stderr)

if __name__ == "__main__":
    for slug, addr, dist in CITIES:
        download(slug, addr, dist)
    print("\nAll done! Restart the Docker container.")
