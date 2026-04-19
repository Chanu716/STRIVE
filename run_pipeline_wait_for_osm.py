#!/usr/bin/env python3
"""
Automated Pipeline Runner - Waits for OSM network, then runs full pipeline
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_road_network(timeout_minutes=25):
    """Wait for road network to be downloaded."""
    logger.info("Waiting for road network download to complete...")
    logger.info("(This typically takes 10-20 minutes for California)")

    network_file = "data/raw/road_network.graphml"
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    last_size = 0
    check_count = 0

    while time.time() - start_time < timeout_seconds:
        if os.path.exists(network_file):
            current_size = os.path.getsize(network_file)

            # File exists and is reasonably sized (> 10 MB for California)
            if current_size > 10_000_000:
                elapsed = int((time.time() - start_time) / 60)
                logger.info(f"Road network ready! ({current_size / 1024 / 1024:.1f} MB, {elapsed} min)")
                return True

            # Show progress
            if current_size != last_size:
                size_mb = current_size / 1024 / 1024
                elapsed = int((time.time() - start_time) / 60)
                logger.info(f"  Progress: {size_mb:.1f} MB (elapsed: {elapsed} min)")
                last_size = current_size

        check_count += 1
        if check_count % 6 == 0:  # Every 30 seconds
            logger.info(f"  Still downloading... ({int((time.time() - start_time) / 60)} min elapsed)")

        time.sleep(5)

    logger.error(f"Timeout waiting for road network after {timeout_minutes} minutes")
    return False


def validate_road_network():
    """Validate that road network has correct format."""
    logger.info("Validating road network...")

    try:
        import osmnx as ox

        G = ox.load_graphml("data/raw/road_network.graphml")
        logger.info(f"  Nodes: {len(G.nodes):,}")
        logger.info(f"  Edges: {len(G.edges):,}")

        if len(G.nodes) < 10000:
            logger.warning(f"  Warning: Small network ({len(G.nodes)} nodes)")

        return True
    except Exception as e:
        logger.error(f"Network validation failed: {e}")
        return False


def run_snapping():
    """Run accident snapping pipeline."""
    logger.info("\n" + "="*60)
    logger.info("Running T-03: Snap Accidents to Road Segments")
    logger.info("="*60)

    cmd = [
        "python", "scripts/snap_accidents.py",
        "--fars-dir", "data/raw",
        "--network", "data/raw/road_network.graphml",
        "--threshold", "50.0",
        "--output", "data/processed/accidents_snapped.parquet"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("Snapping complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Snapping failed: {e}")
        return False


def run_rate_computation():
    """Run historical rate computation."""
    logger.info("\n" + "="*60)
    logger.info("Running T-04: Compute Historical Accident Rates")
    logger.info("="*60)

    cmd = [
        "python", "scripts/compute_accident_rates.py",
        "--snapped", "data/processed/accidents_snapped.parquet",
        "--network", "data/raw/road_network.graphml",
        "--years", "3",
        "--output", "data/processed/segment_rates.parquet"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("Rate computation complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Rate computation failed: {e}")
        return False


def verify_pipeline():
    """Verify that pipeline completed successfully."""
    logger.info("\n" + "="*60)
    logger.info("Verifying Pipeline Results")
    logger.info("="*60)

    import pandas as pd

    success = True

    # Check snapped accidents
    snapped_file = "data/processed/accidents_snapped.parquet"
    if os.path.exists(snapped_file):
        snapped = pd.read_parquet(snapped_file)
        logger.info(f"Snapped accidents: {len(snapped):,}")
        if len(snapped) < 10000:
            logger.warning(f"  Warning: Low snapping count")
    else:
        logger.error(f"Missing: {snapped_file}")
        success = False

    # Check segment rates
    rates_file = "data/processed/segment_rates.parquet"
    if os.path.exists(rates_file):
        rates = pd.read_parquet(rates_file)
        logger.info(f"Road segments with rates: {len(rates):,}")
        logger.info(f"  Mean accident rate: {rates['historical_accident_rate'].mean():.4f} incidents/km/year")
        logger.info(f"  Max rate: {rates['historical_accident_rate'].max():.4f}")
    else:
        logger.error(f"Missing: {rates_file}")
        success = False

    return success


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("M1 DATA PIPELINE - AUTOMATED RUNNER")
    logger.info("="*60)

    # Step 1: Wait for road network
    logger.info("\nStep 1: Waiting for Road Network Download...")
    if not wait_for_road_network():
        logger.error("Road network download timeout or failed")
        return 1

    # Step 2: Validate road network
    logger.info("\nStep 2: Validating Road Network...")
    if not validate_road_network():
        logger.error("Road network validation failed")
        return 1

    # Step 3: Run snapping
    logger.info("\nStep 3: Running Accident Snapping...")
    if not run_snapping():
        logger.error("Accident snapping failed")
        return 1

    # Step 4: Compute rates
    logger.info("\nStep 4: Computing Historical Rates...")
    if not run_rate_computation():
        logger.error("Rate computation failed")
        return 1

    # Step 5: Verify results
    logger.info("\nStep 5: Verifying Results...")
    if not verify_pipeline():
        logger.error("Verification failed")
        return 1

    # Success!
    logger.info("\n" + "="*60)
    logger.info("SUCCESS! Pipeline Complete")
    logger.info("="*60)
    logger.info("\nYour real data is ready for M2 (ML Engineer):")
    logger.info("  data/processed/accidents_snapped.parquet")
    logger.info("  data/processed/segment_rates.parquet")
    logger.info("  app/ml/features.py (feature pipeline)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
