#!/usr/bin/env python3
"""
M1 Data Preparation Pipeline - Complete Workflow

This script orchestrates all T-01 to T-05 tasks in sequence:
1. Download FARS data
2. Download OSM network
3. Snap accidents to segments
4. Compute accident rates
5. Build complete training features

Run: python data_prepare.py --city "Los Angeles, CA" --years 2021 2022 2023
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a subprocess command and log output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"  Error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during {description}")
        logger.error(f"  Error: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete M1 data preparation pipeline"
    )
    parser.add_argument(
        "--city",
        default="Los Angeles, California, USA",
        help="City and state for data download"
    )
    parser.add_argument(
        "--place",
        default="Los Angeles, California, USA",
        help="Place name for OSMnx (default: same as city)"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Years to download"
    )
    parser.add_argument(
        "--skip-fars",
        action="store_true",
        help="Skip FARS download (reuse existing)"
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip OSM download (reuse existing)"
    )

    args = parser.parse_args()

    logger.info("\n" + "="*60)
    logger.info("STRIVE M1 — Data Preparation Pipeline")
    logger.info("="*60)
    logger.info(f"City: {args.city}")
    logger.info(f"Years: {args.years}")
    logger.info("="*60 + "\n")

    all_success = True

    # T-01: Download FARS
    if not args.skip_fars:
        cmd = [
            "python", "scripts/download_fars_data.py",
            "--city", args.city.split(",")[0].strip(),
            "--state", args.city.split(",")[-1].strip() if "," in args.city else "CA",
            "--years", *map(str, args.years),
            "--output", "data/raw"
        ]
        all_success = run_command(cmd, "T-01: Download FARS Data") and all_success

    # T-02: Download OSM
    if not args.skip_osm:
        cmd = [
            "python", "scripts/download_osm_network.py",
            "--place", args.place,
            "--network-type", "drive",
            "--output", "data/raw",
            "--validate"
        ]
        all_success = run_command(cmd, "T-02: Download OSM Network") and all_success

    # T-03: Snap Accidents
    cmd = [
        "python", "scripts/snap_accidents.py",
        "--fars-dir", "data/raw",
        "--network", "data/raw/road_network.graphml",
        "--threshold", "50.0",
        "--output", "data/processed/accidents_snapped.parquet"
    ]
    all_success = run_command(cmd, "T-03: Snap Accidents to Road Segments") and all_success

    # T-04: Compute Rates
    cmd = [
        "python", "scripts/compute_accident_rates.py",
        "--snapped", "data/processed/accidents_snapped.parquet",
        "--network", "data/raw/road_network.graphml",
        "--years", str(len(args.years)),
        "--output", "data/processed/segment_rates.parquet"
    ]
    all_success = run_command(cmd, "T-04: Compute Historical Accident Rates") and all_success

    # Summary
    logger.info("\n" + "="*60)
    if all_success:
        logger.info("✓ ALL TASKS COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("\nGenerated files:")
        logger.info("  data/raw/fars_*.csv (FARS accident data)")
        logger.info("  data/raw/road_network.graphml (OSM road network)")
        logger.info("  data/processed/accidents_snapped.parquet (Snapped accidents)")
        logger.info("  data/processed/segment_rates.parquet (Accident rates)")
        logger.info("\nNext: M2 will use these to train the ML model")
        return 0
    else:
        logger.error("✗ SOME TASKS FAILED")
        logger.error("="*60)
        logger.error("\nPlease check errors above and re-run")
        return 1


if __name__ == "__main__":
    sys.exit(main())
