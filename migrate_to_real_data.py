#!/usr/bin/env python3
"""
Migration Script: Switch from Sample Data to Real Data

This script helps transition from synthetic sample data to real NHTSA FARS
and OpenStreetMap datasets.

Usage:
    python migrate_to_real_data.py --fars-dir "/path/to/real/fars" --download-osm
"""

import os
import sys
import logging
import shutil
from pathlib import Path
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_sample_data():
    """Backup existing sample data before replacing with real data."""
    logger.info("Backing up existing sample data...")

    backup_dir = "data/backup_sample_data"
    os.makedirs(backup_dir, exist_ok=True)

    # Backup raw FARS data
    for year in [2021, 2022, 2023]:
        src = f"data/raw/fars_{year}.csv"
        if os.path.exists(src):
            dst = f"{backup_dir}/fars_{year}.csv"
            shutil.copy(src, dst)
            logger.info(f"  Backed up {src}")

    # Backup road network
    if os.path.exists("data/raw/road_network.graphml"):
        shutil.copy("data/raw/road_network.graphml",
                   f"{backup_dir}/road_network_synthetic.graphml")
        logger.info("  Backed up road_network.graphml")

    # Backup processed data
    for file in ["accidents_snapped.parquet", "segment_rates.parquet"]:
        src = f"data/processed/{file}"
        if os.path.exists(src):
            dst = f"{backup_dir}/{file}"
            shutil.copy(src, dst)
            logger.info(f"  Backed up {src}")

    logger.info(f"Backup complete: {backup_dir}/")


def copy_real_fars_data(fars_dir):
    """Copy real FARS CSV files from user-provided directory."""
    logger.info(f"\nCopying real FARS data from {fars_dir}...")

    if not os.path.exists(fars_dir):
        logger.error(f"FARS directory not found: {fars_dir}")
        return False

    fars_files = []
    for file in os.listdir(fars_dir):
        if file.endswith(".csv") and "fars" in file.lower():
            fars_files.append(file)

    if not fars_files:
        logger.warning(f"No FARS CSV files found in {fars_dir}")
        logger.info("Expected files: fars_2021.csv, fars_2022.csv, fars_2023.csv")
        return False

    logger.info(f"Found {len(fars_files)} FARS files:")
    for file in fars_files:
        src = os.path.join(fars_dir, file)
        dst = f"data/raw/{file}"
        shutil.copy(src, dst)
        size_mb = os.path.getsize(dst) / 1024 / 1024
        logger.info(f"  ✓ Copied {file} ({size_mb:.1f} MB)")

    return True


def verify_fars_data():
    """Verify that FARS files have expected schema."""
    logger.info("\nVerifying FARS data schema...")

    import pandas as pd

    for year in [2021, 2022, 2023]:
        file = f"data/raw/fars_{year}.csv"
        if not os.path.exists(file):
            logger.warning(f"  FARS file not found: {file}")
            continue

        try:
            df = pd.read_csv(file, nrows=1)
            required_cols = {'LATITUDE', 'LONGITUDE', 'STATE', 'CITY'}
            missing = required_cols - set(df.columns)

            if missing:
                logger.error(f"  {file}: Missing columns {missing}")
                return False

            logger.info(f"  ✓ {file}: Valid schema ({len(df.columns)} columns)")
        except Exception as e:
            logger.error(f"  {file}: {e}")
            return False

    return True


def download_real_osm_network():
    """Download real OSM network using OSMnx."""
    logger.info("\nDownloading real OSM network (this may take 5-15 minutes)...")

    try:
        import osmnx as ox
    except ImportError:
        logger.error("OSMnx not installed. Run: pip install osmnx")
        return False

    try:
        logger.info("Downloading for Los Angeles, California...")
        G = ox.graph_from_place(
            "Los Angeles, California, USA",
            network_type="drive",
            simplify=True,
            retain_all=False
        )

        logger.info(f"✓ Downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

        # Save
        ox.save_graphml(G, "data/raw/road_network.graphml")
        size_mb = os.path.getsize("data/raw/road_network.graphml") / 1024 / 1024
        logger.info(f"✓ Saved to data/raw/road_network.graphml ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        logger.error(f"Failed to download OSM network: {e}")
        logger.info("\nFallback: Use offline OSM data")
        logger.info("1. Download from: https://download.geofabrik.de/")
        logger.info("2. Place california-latest.osm.pbf in data/raw/")
        logger.info("3. Run: python scripts/download_osm_network.py --use-offline")
        return False


def reprocess_data_pipeline():
    """Re-run the data pipeline with real data."""
    logger.info("\nRe-processing data pipeline with real data...")

    cmd = [
        "python", "data_prepare.py",
        "--city", "Los Angeles, CA",
        "--years", "2021", "2022", "2023",
        "--skip-fars",  # Already have real FARS
        "--skip-osm"    # Already have real OSM
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✓ Data pipeline complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate from sample data to real NHTSA FARS and OSM data"
    )
    parser.add_argument(
        "--fars-dir",
        help="Directory containing real FARS CSV files"
    )
    parser.add_argument(
        "--download-osm",
        action="store_true",
        help="Download real OSM network (takes 5-15 minutes)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup sample data before replacing (default: yes)"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=True,
        help="Re-run pipeline after data migration (default: yes)"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("STRIVE - Migrate to Real Data")
    logger.info("="*60)

    # Backup sample data
    if args.backup:
        backup_sample_data()

    success = True

    # Copy real FARS data
    if args.fars_dir:
        if not copy_real_fars_data(args.fars_dir):
            success = False
        elif not verify_fars_data():
            success = False
    else:
        logger.warning("\nNo FARS directory provided (--fars-dir)")
        logger.info("To use real FARS data:")
        logger.info("  1. Download from: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars")
        logger.info("  2. Run: python migrate_to_real_data.py --fars-dir /path/to/fars")

    # Download real OSM network
    if args.download_osm:
        if not download_real_osm_network():
            success = False
    else:
        logger.warning("\nOSM network not downloaded (--download-osm)")
        logger.info("To use real OSM data:")
        logger.info("  python migrate_to_real_data.py --download-osm")

    # Re-process pipeline
    if success and args.reprocess:
        if not reprocess_data_pipeline():
            success = False

    # Summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("✓ Migration complete!")
        logger.info("\nReal data is now in use:")
        logger.info("  data/raw/fars_*.csv (real NHTSA data)")
        logger.info("  data/raw/road_network.graphml (real OSM)")
        logger.info("  data/processed/*.parquet (reprocessed)")
        logger.info("\nSample data backed up to: data/backup_sample_data/")
    else:
        logger.error("✗ Migration incomplete - see errors above")
        logger.info("\nNext steps:")
        logger.info("  1. Check error messages above")
        logger.info("  2. Download missing data")
        logger.info("  3. Re-run this script")

    logger.info("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
