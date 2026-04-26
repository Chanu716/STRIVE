import pandas as pd
import osmnx as ox

snapped = pd.read_parquet("data/processed/accidents_snapped.parquet")
print("Snapped cols:", list(snapped.columns))

for col in ["WEATHER","WEATHERNAME","LGT_COND","FUNC_SYS"]:
    print(f"  {col} in snapped? {col in snapped.columns}")

G = ox.load_graphml("data/raw/road_network.graphml")
sample = list(G.edges(data=True))[:2]
for u, v, d in sample:
    print("Edge keys:", list(d.keys()))
    print("  speed_kph:", d.get("speed_kph", "NOT FOUND"))
    print("  maxspeed:", d.get("maxspeed", "NOT FOUND"))
    print("  highway:", d.get("highway", "NOT FOUND"))
