import osmnx as ox
from pathlib import Path

CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CITIES = [
    {"name": "Vijayawada, India", "file": "vijayawada.graphml"},
    {"name": "Visakhapatnam, India", "file": "vizag.graphml"},
    {"name": "Delhi, India", "file": "delhi.graphml"},
    {"name": "Agra, India", "file": "agra.graphml"},
]

def warm_up():
    for city in CITIES:
        path = CACHE_DIR / city["file"]
        if path.exists():
            print(f"Skipping {city['name']}, already exists.")
            continue
            
        print(f"Pre-downloading {city['name']}...")
        try:
            # Download 10km radius for each major city
            G = ox.graph_from_address(city["name"], dist=10000, network_type="drive")
            ox.save_graphml(G, path)
            print(f"Successfully saved {city['name']} to {path}")
        except Exception as e:
            print(f"Failed to download {city['name']}: {e}")

if __name__ == "__main__":
    warm_up()
