#!/usr/bin/env python3
"""
T-02: Download OpenStreetMap road network via OSMnx.

OSMnx allows us to download and process road networks from OpenStreetMap.
We'll fetch a "drive" network (driveable roads) for the target city/place.

Output: data/raw/road_network.graphml (NetworkX graph in GraphML format)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import osmnx as ox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_road_network(place: str, network_type: str = "drive", output_dir: str = "data/raw") -> str:
    """
    Download road network from OpenStreetMap for a given place.

    Args:
        place: Place name recognized by OSM (e.g., "Los Angeles, California, USA")
        network_type: "drive", "walk", "bike", or "all" (default: "drive")
        output_dir: Directory to save GraphML file

    Returns:
        Path to saved GraphML file
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading road network for '{place}'...")
    logger.info(f"  Network type: {network_type}")

    try:
        # Download graph from OSM
        G = ox.graph_from_place(place, network_type=network_type, simplify=True, retain_all=False)
        logger.info(f"✓ Downloaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # Add useful metadata to edges
        logger.info("Computing edge attributes...")
        for u, v, k, data in G.edges(keys=True, data=True):
            # Ensure speed_kph exists (default to 50 km/h if missing)
            if 'speed_kph' not in data:
                data['speed_kph'] = 50

            # Compute travel time in seconds
            if 'length' in data:
                length_km = data['length'] / 1000
                speed = data.get('speed_kph', 50)
                data['travel_time_sec'] = (length_km / speed) * 3600 if speed > 0 else 3600

        # Save to GraphML
        output_file = os.path.join(output_dir, "road_network.graphml")
        ox.save_graphml(G, filepath=output_file)
        logger.info(f"✓ Saved graph to {output_file}")

        # Summary statistics
        logger.info("\nNetwork Summary:")
        logger.info(f"  Nodes (intersections): {len(G.nodes)}")
        logger.info(f"  Edges (road segments): {len(G.edges)}")
        logger.info(f"  Average degree: {sum(dict(G.degree()).values()) / len(G.nodes):.2f}")

        # Check for required attributes
        sample_edge = list(G.edges(keys=True, data=True))[0]
        _, _, _, attrs = sample_edge
        logger.info(f"\nSample edge attributes:")
        for key in sorted(attrs.keys())[:5]:
            logger.info(f"    {key}: {attrs[key]}")

        return output_file

    except Exception as e:
        logger.error(f"Failed to download network: {e}")
        raise


def validate_road_network(graphml_file: str) -> bool:
    """
    Validate that the road network GraphML has required attributes.

    Returns:
        True if valid, False otherwise
    """
    try:
        G = ox.load_graphml(graphml_file)
        logger.info(f"✓ Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # Check for required attributes
        sample_edge = list(G.edges(keys=True, data=True))[0]
        _, _, _, attrs = sample_edge
        required = {'length', 'geometry'}
        optional = {'speed_kph', 'highway', 'name'}

        missing_required = required - set(attrs.keys())
        if missing_required:
            logger.warning(f"Missing required attributes: {missing_required}")

        found_optional = optional & set(attrs.keys())
        logger.info(f"Found optional attributes: {found_optional}")

        return len(missing_required) == 0

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download OpenStreetMap road network via OSMnx"
    )
    parser.add_argument(
        "--place",
        default="Los Angeles, California, USA",
        help="Place name recognized by OSM (e.g., 'Los Angeles, California, USA')"
    )
    parser.add_argument(
        "--network-type",
        choices=["drive", "walk", "bike", "all"],
        default="drive",
        help="Type of network to download (default: drive)"
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the downloaded network after saving"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("OSMnx Road Network Download")
    logger.info("=" * 60)

    # Download
    output_file = download_road_network(
        args.place,
        network_type=args.network_type,
        output_dir=args.output
    )

    # Validate
    if args.validate:
        logger.info("\nValidating network...")
        if validate_road_network(output_file):
            logger.info("✓ Network validation passed")
        else:
            logger.warning("✗ Network validation had warnings")

    logger.info("\n" + "=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
