#!/usr/bin/env python3
"""
Convert OSM PBF file to GraphML format for use with osmnx.
Handles large PBF files by streaming and filtering highways only.
"""

import osmium
import networkx as nx
from tqdm import tqdm
import sys

pbf_file = "data/raw/california-260418.osm.pbf"
output_file = "data/raw/road_network.graphml"

# Road types to include (exclude paths, footways, cycleways, etc.)
VALID_HIGHWAYS = {
    'motorway', 'motorway_link',
    'trunk', 'trunk_link',
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'unclassified', 'residential',
    'living_street', 'walkway'
}

class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.graph = nx.MultiDiGraph()
        self.node_coords = {}
        self.ways_count = 0
        self.edges_count = 0

    def node(self, n):
        """Store node coordinates."""
        self.node_coords[n.id] = (n.lat, n.lon)

    def way(self, w):
        """Process ways (roads)."""
        if 'highway' in w.tags:
            highway_type = w.tags['highway']
            # Only include valid road types
            if highway_type in VALID_HIGHWAYS:
                self.ways_count += 1

                # Process way nodes as edges
                node_ids = w.nd_ids
                for i in range(len(node_ids) - 1):
                    from_node = node_ids[i]
                    to_node = node_ids[i + 1]

                    # Both nodes must exist
                    if from_node in self.node_coords and to_node in self.node_coords:
                        lat1, lon1 = self.node_coords[from_node]
                        lat2, lon2 = self.node_coords[to_node]

                        # Add edge with attributes
                        self.graph.add_edge(
                            (lat1, lon1),
                            (lat2, lon2),
                            key=f"{from_node}_{to_node}",
                            highway=highway_type,
                            osmid=w.id
                        )
                        self.edges_count += 1

print("=" * 70)
print("Converting OSM PBF to GraphML")
print("=" * 70)
print(f"\nReading: {pbf_file}")
print("Processing: All roads (motorway, primary, secondary, residential, etc.)")
print("This may take 5-10 minutes for large files...\n")

try:
    handler = RoadHandler()
    handler.apply_file(pbf_file, locations=True)

    print(f"\n✓ Ways processed: {handler.ways_count:,}")
    print(f"✓ Edges created: {handler.edges_count:,}")
    print(f"✓ Graph nodes: {len(handler.graph.nodes):,}")
    print(f"✓ Graph edges: {len(handler.graph.edges):,}")

    print(f"\nSaving to: {output_file}")
    nx.write_graphml(handler.graph, output_file)

    import os
    size_mb = os.path.getsize(output_file) / 1024 / 1024

    print(f"✓ File size: {size_mb:.1f} MB")
    print("\n" + "=" * 70)
    print("SUCCESS! Ready for pipeline")
    print("=" * 70)
    print("\nNext step: Run the pipeline")
    print("  python run_pipeline_wait_for_osm.py")

except Exception as e:
    print(f"\n✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
