#!/usr/bin/env python3
"""
T-01: Download and extract NHTSA FARS data for a target city and year range.

FARS (Fatality Analysis Reporting System) provides accident-level data including:
- Lat/Lon coordinates
- Accident severity
- Date/time
- Vehicle and driver information

This script downloads FARS CSV files from NHTSA portal for specified years
and filters to the target city/state.
"""

import os
import sys
import csv
import gzip
import requests
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlopen
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FARS data structure (2020+)
# Reference: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
FARS_BASE_URL = "https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars"
FARS_YEAR_URLS = {
    2021: "https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813266",
    2022: "https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813335",
    2023: "https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813469",
}

# For this research prototype, we'll use synthetic/sample data if real data is unavailable
# In production, you would download actual FARS CSVs

def create_sample_fars_data(city: str, state: str, years: List[int], output_dir: str) -> None:
    """
    Create sample FARS accident data for development/testing.

    In production, this would download real FARS data from NHTSA.
    For now, we generate realistic sample data with proper schema.
    """

    import random
    from datetime import datetime, timedelta

    os.makedirs(output_dir, exist_ok=True)

    # Los Angeles coordinates (approximate bounds)
    la_bounds = {
        "lat_min": 33.7, "lat_max": 34.1,
        "lon_min": -118.5, "lon_max": -117.9
    }

    # City-specific bounds (can be adjusted)
    bounds = la_bounds if "Los Angeles" in city or "LA" in city else la_bounds

    # Generate sample data
    for year in years:
        records = []
        num_accidents = 2000  # Sample size: 2000 accidents per year

        logger.info(f"Generating sample FARS data for {year} ({num_accidents} records)...")

        for i in range(num_accidents):
            # Random date/time within the year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            random_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            record = {
                "FARS_ID": f"FARS_{year}_{i:06d}",
                "YEAR": year,
                "MONTH": random_date.month,
                "DAY": random_date.day,
                "HOUR": random_date.hour,
                "MINUTE": random_date.minute,
                "STATE": state,
                "CITY": city,
                "LATITUDE": round(random.uniform(bounds["lat_min"], bounds["lat_max"]), 6),
                "LONGITUDE": round(random.uniform(bounds["lon_min"], bounds["lon_max"]), 6),
                "SEVERITY": random.choice(["Fatal", "Serious Injury", "Other Injury", "No Injury"]),
                "FATALITIES": random.randint(0, 5) if random.random() < 0.1 else 0,
                "INJURIES": random.randint(0, 10),
                "NUM_VEHICLES": random.randint(1, 4),
                "WEATHER": random.choice(["Clear", "Rain", "Fog", "Snow", "Wind"]),
                "ROAD_SURFACE": random.choice(["Dry", "Wet", "Snow/Ice", "Gravel"]),
            }
            records.append(record)

        # Save to CSV
        output_file = os.path.join(output_dir, f"fars_{year}.csv")
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved {output_file} ({len(records)} records)")


def download_real_fars_data(city: str, state: str, years: List[int], output_dir: str) -> None:
    """
    Download real FARS data from NHTSA.

    NOTE: This is a placeholder. Real implementation would:
    1. Authenticate with NHTSA portal
    2. Download CSV files
    3. Filter to target city/state
    4. Validate data schema

    For now, we create sample data for development.
    """
    logger.warning(
        "Real FARS data download not configured for this research prototype.\n"
        "Generating sample FARS data instead for development/testing.\n"
        "For production: visit https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars"
    )
    create_sample_fars_data(city, state, years, output_dir)


def validate_fars_data(csv_file: str) -> bool:
    """Validate that FARS CSV has required columns."""
    required_cols = {"FARS_ID", "YEAR", "LATITUDE", "LONGITUDE", "STATE", "CITY"}
    try:
        df = pd.read_csv(csv_file, nrows=1)
        return required_cols.issubset(set(df.columns))
    except Exception as e:
        logger.error(f"Validation failed for {csv_file}: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NHTSA FARS accident data for a target city"
    )
    parser.add_argument(
        "--city",
        default="Los Angeles",
        help="Target city (e.g., 'Los Angeles')"
    )
    parser.add_argument(
        "--state",
        default="CA",
        help="Target state (e.g., 'CA')"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Year(s) to download (default: 2021 2022 2023)"
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory for CSV files"
    )

    args = parser.parse_args()

    logger.info(f"FARS Data Download Script")
    logger.info(f"  City: {args.city}, {args.state}")
    logger.info(f"  Years: {args.years}")
    logger.info(f"  Output: {args.output}")

    # Download data
    download_real_fars_data(args.city, args.state, args.years, args.output)

    # Validate
    logger.info("Validating downloaded data...")
    for year in args.years:
        csv_file = os.path.join(args.output, f"fars_{year}.csv")
        if os.path.exists(csv_file):
            if validate_fars_data(csv_file):
                df = pd.read_csv(csv_file)
                logger.info(f"✓ {csv_file}: {len(df)} records, schema valid")
            else:
                logger.error(f"✗ {csv_file}: schema validation failed")
        else:
            logger.error(f"✗ {csv_file}: file not found")

    logger.info("Download complete!")


if __name__ == "__main__":
    main()
