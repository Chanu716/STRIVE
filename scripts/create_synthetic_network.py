#!/usr/bin/env python3
"""
Create a synthetic road network for development/testing.

This script generates a realistic NetworkX graph that simulates
the structure returned by OSMnx for testing purposes.
"""

import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString

# Set random seed for reproducibility
np.random.seed(42)

print("Creating synthetic road network...")

# Create a grid-based road network for a simulated LA area
# Bounds: approximately 34.0° to 34.1° N, -118.5° to -118.4° W
lat_min, lat_max = 34.0, 34.1
lon_min, lon_max = -118.5, -118.4

# Create grid nodes
G = nx.MultiDiGraph()

# Set CRS metadata (required by OSMnx)
G.graph["crs"] = "EPSG:4326"

lat_points = np.linspace(lat_min, lat_max, 8)
lon_points = np.linspace(lon_min, lon_max, 8)

# Add nodes
node_id = 0
nodes_map = {}
for lat in lat_points:
    for lon in lon_points:
        nodes_map[(lat, lon)] = node_id
        G.add_node(
            node_id,
            y=lat,
            x=lon,
            geometry=Point(lon, lat)
        )
        node_id += 1

# Add edges (connect adjacent nodes in grid)
road_classes = {
    0: ('motorway', 100),
    1: ('trunk', 90),
    2: ('primary', 80),
    3: ('secondary', 60),
    4: ('residential', 40),
}

edge_id = 0
for i, lat in enumerate(lat_points):
    for j, lon in enumerate(lon_points):
        current_node = nodes_map[(lat, lon)]

        # Determine road class based on position (main roads near edges)
        road_tier = min(i, len(lat_points) - 1 - i, j, len(lon_points) - 1 - j)
        road_class_key = min(road_tier, 4)
        highway_type, speed = road_classes[road_class_key]

        # Connect to right neighbor
        if j < len(lon_points) - 1:
            next_node = nodes_map[(lat, lon_points[j + 1])]
            length = 111.32 * (lon_points[j + 1] - lon) * 1000  # ~111km per degree lon

            G.add_edge(
                current_node, next_node,
                key=edge_id,
                geometry=LineString([(lon, lat), (lon_points[j + 1], lat)]),
                length=length,
                speed_kph=speed,
                highway=highway_type,
                name=f"{highway_type} {edge_id}",
            )
            G.add_edge(
                next_node, current_node,
                key=edge_id + 1,
                geometry=LineString([(lon_points[j + 1], lat), (lon, lat)]),
                length=length,
                speed_kph=speed,
                highway=highway_type,
                name=f"{highway_type} {edge_id}",
            )
            edge_id += 2

        # Connect to bottom neighbor
        if i < len(lat_points) - 1:
            next_node = nodes_map[(lat_points[i + 1], lon)]
            length = 110.57 * (lat_points[i + 1] - lat) * 1000  # ~110.57km per degree lat

            G.add_edge(
                current_node, next_node,
                key=edge_id,
                geometry=LineString([(lon, lat), (lon, lat_points[i + 1])]),
                length=length,
                speed_kph=speed,
                highway=highway_type,
                name=f"{highway_type} {edge_id}",
            )
            G.add_edge(
                next_node, current_node,
                key=edge_id + 1,
                geometry=LineString([(lon, lat_points[i + 1]), (lon, lat)]),
                length=length,
                speed_kph=speed,
                highway=highway_type,
                name=f"{highway_type} {edge_id}",
            )
            edge_id += 2

# Add some random variations
for u, v, k, data in list(G.edges(keys=True, data=True)):
    if np.random.random() < 0.3:
        # Add some one-way streets
        pass

    # Ensure all edges have required attributes
    if 'length' not in data:
        data['length'] = 500
    if 'speed_kph' not in data:
        data['speed_kph'] = 50
    if 'highway' not in data:
        data['highway'] = 'residential'

print(f"[OK] Created synthetic network: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Save
output_path = "data/raw/road_network.graphml"
ox.save_graphml(G, filepath=output_path)
print(f"[OK] Saved to {output_path}")

# Verify
G_loaded = ox.load_graphml(output_path)
print(f"[OK] Verified: {len(G_loaded.nodes)} nodes, {len(G_loaded.edges)} edges")
print("\nNetwork created successfully!")
print("Note: This is a synthetic network for testing. Use real OSM data in production.")
