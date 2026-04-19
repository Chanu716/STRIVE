#!/usr/bin/env python3
"""
T-03: Match FARS accident records to nearest OSM road segments.

This script:
1. Loads the FARS accident CSV files
2. Loads the OSM road network (GraphML)
3. For each accident lat/lon, finds the nearest edge
4. Keeps only matches within a distance threshold (e.g., 50 m)
5. Outputs snapped data to parquet

Output: data/processed/accidents_snapped.parquet
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fars_data(fars_files: List[str]) -> pd.DataFrame:
    """
    Load and combine FARS CSV files.

    Args:
        fars_files: List of paths to FARS CSV files

    Returns:
        Combined DataFrame with all accident records
    """
    dfs = []
    for f in fars_files:
        logger.info(f"Loading FARS data from {f}...")
        df = pd.read_csv(f)
        dfs.append(df)
        logger.info(f"  → {len(df)} records")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"✓ Combined FARS data: {len(combined)} total records")
    return combined


def load_road_network(graphml_file: str) -> nx.MultiDiGraph:
    """Load OSM road network from GraphML."""
    logger.info(f"Loading road network from {graphml_file}...")
    G = ox.load_graphml(graphml_file)
    logger.info(f"✓ Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def snap_accidents_to_edges(
    accidents: pd.DataFrame,
    G: nx.MultiDiGraph,
    threshold_m: float = 50.0
) -> pd.DataFrame:
    """
    Match each accident to the nearest road edge.

    Args:
        accidents: DataFrame with 'LATITUDE', 'LONGITUDE' columns
        G: Road network graph
        threshold_m: Maximum distance threshold in meters

    Returns:
        DataFrame with snapped edges and distances
    """
    logger.info(f"Snapping {len(accidents)} accidents to nearest edges...")
    logger.info(f"  Threshold: {threshold_m} m")

    snapped_records = []
    errors = 0

    for idx, row in accidents.iterrows():
        if idx % 500 == 0 and idx > 0:
            logger.info(f"  Processed {idx} / {len(accidents)} accidents...")

        try:
            lat, lon = float(row['LATITUDE']), float(row['LONGITUDE'])

            # Find nearest edge - try to handle different return formats
            nearest_edge_result = ox.nearest_edges(
                G,
                X=lon,
                Y=lat,
                return_dist=True
            )

            # Handle different tuple lengths
            # return_dist=True returns ((u, v, k), distance) tuple
            if isinstance(nearest_edge_result, tuple) and len(nearest_edge_result) == 2:
                edge_tuple, dist_m = nearest_edge_result
                u, v, k = edge_tuple
            elif isinstance(nearest_edge_result, tuple) and len(nearest_edge_result) == 3:
                u, v, k = nearest_edge_result
                dist_m = 0  # No distance returned
            else:
                logger.debug(f"Unexpected nearest_edges return: {nearest_edge_result}")
                errors += 1
                continue

            # Check distance threshold (convert m to meters if needed)
            dist_m_float = float(dist_m) if isinstance(dist_m, (int, float, np.number)) else dist_m

            if dist_m_float <= threshold_m:
                snapped_records.append({
                    'accident_id': str(row.get('FARS_ID', f"ACC_{idx}")),
                    'osmid': f"{u}_{v}",
                    'latitude': lat,
                    'longitude': lon,
                    'snap_distance_m': dist_m_float,
                    'year': int(row.get('YEAR', 0)),
                    'month': int(row.get('MONTH', 0)),
                    'day': int(row.get('DAY', 0)),
                    'hour': int(row.get('HOUR', 0)),
                    'minute': int(row.get('MINUTE', 0)),
                    'severity': str(row.get('SEVERITY', '')),
                    'fatalities': int(row.get('FATALITIES', 0)),
                    'injuries': int(row.get('INJURIES', 0)),
                })

        except Exception as e:
            errors += 1
            if errors <= 5:  # Log first 5 errors only
                logger.debug(f"Error snapping accident {idx}: {e}")

    logger.info(f"[OK] Snapped {len(snapped_records)} accidents (within threshold)")
    logger.info(f"  Discarded: {len(accidents) - len(snapped_records)} (outside threshold or error)")
    logger.info(f"  Errors: {errors}")

    return pd.DataFrame(snapped_records)


def validate_snapped_data(snapped: pd.DataFrame) -> bool:
    """Validate snapped data schema."""
    if len(snapped) == 0:
        logger.warning("Snapped data is empty - no accidents within threshold")
        return True  # Not a hard failure, just means threshold is too strict

    required_cols = {
        'accident_id', 'osmid', 'latitude', 'longitude',
        'snap_distance_m', 'year', 'month', 'day', 'hour'
    }
    missing = required_cols - set(snapped.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    logger.info(f"[OK] Schema validation passed")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Snap FARS accidents to OSM road segments"
    )
    parser.add_argument(
        "--fars-dir",
        default="data/raw",
        help="Directory containing FARS CSV files"
    )
    parser.add_argument(
        "--fars-pattern",
        default="fars_*.csv",
        help="Glob pattern for FARS files"
    )
    parser.add_argument(
        "--network",
        default="data/raw/road_network.graphml",
        help="Path to road network GraphML file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Distance threshold in meters"
    )
    parser.add_argument(
        "--output",
        default="data/processed/accidents_snapped.parquet",
        help="Output parquet file"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Snapping FARS Accidents to Road Segments")
    logger.info("=" * 60)

    # Find FARS files
    from glob import glob
    fars_files = sorted(glob(os.path.join(args.fars_dir, args.fars_pattern)))
    if not fars_files:
        logger.error(f"No FARS files found matching {args.fars_pattern}")
        return

    logger.info(f"Found {len(fars_files)} FARS files")

    # Load data
    accidents = load_fars_data(fars_files)
    G = load_road_network(args.network)

    # Snap
    snapped = snap_accidents_to_edges(accidents, G, threshold_m=args.threshold)

    # Validate
    if validate_snapped_data(snapped):
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        snapped.to_parquet(args.output)
        logger.info(f"✓ Saved {args.output}")

        # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  Input: {len(accidents)} accidents")
    logger.info(f"  Output: {len(snapped)} snapped accidents")
    logger.info(f"  Match rate: {100 * len(snapped) / len(accidents):.1f}%")
    if len(snapped) > 0:
        logger.info(f"  Mean snap distance: {snapped['snap_distance_m'].mean():.1f} m")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
