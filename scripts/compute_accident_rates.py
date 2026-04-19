#!/usr/bin/env python3
"""
T-04: Compute historical accident rate per road segment.

This script:
1. Loads snapped accidents
2. Loads road network to get segment lengths
3. Computes: rate = incidents / (length_km) / years
4. Outputs to parquet: osmid, historical_accident_rate

Output: data/processed/segment_rates.parquet
"""

import os
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import osmnx as ox
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_snapped_accidents(parquet_file: str) -> pd.DataFrame:
    """Load snapped accident data."""
    logger.info(f"Loading snapped accidents from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    logger.info(f"✓ Loaded {len(df)} snapped accidents")
    return df


def load_network_edges(graphml_file: str) -> pd.DataFrame:
    """
    Extract edge data from road network.

    Returns DataFrame with columns: osmid, length_m, highway, speed_kph
    """
    logger.info(f"Loading network edges from {graphml_file}...")
    G = ox.load_graphml(graphml_file)

    edges_data = []
    for u, v, k, data in G.edges(keys=True, data=True):
        edges_data.append({
            'osmid': f"{u}_{v}",
            'length_m': data.get('length', 0),
            'highway': data.get('highway', 'unknown'),
            'speed_kph': data.get('speed_kph', 50),
        })

    edges_df = pd.DataFrame(edges_data)
    logger.info(f"✓ Loaded {len(edges_df)} edges")
    return edges_df


def compute_accident_rates(
    snapped_accidents: pd.DataFrame,
    edges: pd.DataFrame,
    time_range_years: int = 3
) -> pd.DataFrame:
    """
    Compute historical accident rate per segment.

    Formula: rate = count / (length_km) / years

    Args:
        snapped_accidents: DataFrame with osmid column
        edges: DataFrame with osmid, length_m columns
        time_range_years: Number of years of historical data

    Returns:
        DataFrame with osmid, historical_accident_rate
    """
    logger.info(f"Computing accident rates (time span: {time_range_years} years)...")

    # Count accidents per segment
    accidents_per_segment = snapped_accidents.groupby('osmid').size().reset_index(name='count')
    logger.info(f"  Unique segments with accidents: {len(accidents_per_segment)}")

    # Merge with edge lengths
    rates = edges[['osmid', 'length_m']].merge(
        accidents_per_segment,
        on='osmid',
        how='left'
    )

    # Fill missing accident counts with 0
    rates['count'] = rates['count'].fillna(0)

    # Compute rate: incidents / (length in km) / years
    rates['length_km'] = rates['length_m'] / 1000
    rates['historical_accident_rate'] = (
        rates['count'] / rates['length_km'].clip(lower=0.001) / time_range_years
    )

    # Clean up
    result = rates[['osmid', 'historical_accident_rate', 'count', 'length_km']].copy()

    logger.info(f"✓ Computed rates for {len(result)} segments")
    logger.info(f"  Mean rate: {result['historical_accident_rate'].mean():.4f} incidents/km/year")
    logger.info(f"  Max rate: {result['historical_accident_rate'].max():.4f}")
    logger.info(f"  Segments with accidents: {(result['count'] > 0).sum()}")

    return result[['osmid', 'historical_accident_rate']]


def validate_rates(rates: pd.DataFrame) -> bool:
    """Validate rates data."""
    required_cols = {'osmid', 'historical_accident_rate'}
    missing = required_cols - set(rates.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False

    # Check for NaN
    if rates['historical_accident_rate'].isna().any():
        logger.error("Found NaN values in historical_accident_rate")
        return False

    # Check for negative values
    if (rates['historical_accident_rate'] < 0).any():
        logger.error("Found negative rates")
        return False

    logger.info("✓ Rates validation passed")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute historical accident rates per road segment"
    )
    parser.add_argument(
        "--snapped",
        default="data/processed/accidents_snapped.parquet",
        help="Path to snapped accidents parquet file"
    )
    parser.add_argument(
        "--network",
        default="data/raw/road_network.graphml",
        help="Path to road network GraphML file"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Time span in years for rate calculation"
    )
    parser.add_argument(
        "--output",
        default="data/processed/segment_rates.parquet",
        help="Output parquet file"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Computing Historical Accident Rates")
    logger.info("=" * 60)

    # Load data
    snapped = load_snapped_accidents(args.snapped)
    edges = load_network_edges(args.network)

    # Compute rates
    rates = compute_accident_rates(snapped, edges, time_range_years=args.years)

    # Validate
    if validate_rates(rates):
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        rates.to_parquet(args.output)
        logger.info(f"✓ Saved {args.output}")

        logger.info("\n" + "=" * 60)
        logger.info("Summary:")
        logger.info(f"  Segments with rates: {len(rates)}")
        logger.info(f"  Mean rate: {rates['historical_accident_rate'].mean():.4f} incidents/km/year")
        logger.info(f"  Median rate: {rates['historical_accident_rate'].median():.4f}")
        logger.info(f"  95th percentile: {rates['historical_accident_rate'].quantile(0.95):.4f}")
        logger.info("=" * 60)

    else:
        logger.error("Validation failed")


if __name__ == "__main__":
    main()
