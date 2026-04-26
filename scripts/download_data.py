#!/usr/bin/env python3
"""
Download the raw FARS and OSM inputs used by the STRIVE pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from download_fars_data import download_real_fars_data
from download_osm_network import download_road_network


def _file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size_kb = path.stat().st_size / 1024.0
    return f"{size_kb:.1f} KB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the STRIVE raw data inputs.")
    parser.add_argument("--city", default="Los Angeles", help="Target city name.")
    parser.add_argument("--state", default="CA", help="Target state abbreviation.")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023], help="Years to fetch.")
    parser.add_argument(
        "--place",
        default="Los Angeles, California, USA",
        help="OSM place name for the road network download.",
    )
    parser.add_argument("--network-type", default="drive", choices=["drive", "walk", "bike", "all"])
    parser.add_argument("--output-dir", default="data/raw", help="Directory for the raw files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading FARS data...")
    download_real_fars_data(args.city, args.state, args.years, str(output_dir))

    print("Downloading road network...")
    graph_path = Path(download_road_network(args.place, network_type=args.network_type, output_dir=str(output_dir)))

    print("\nDownload summary:")
    for year in args.years:
        path = output_dir / f"fars_{year}.csv"
        print(f"- {path}: {_file_size(path)}")
    print(f"- {graph_path}: {_file_size(graph_path)}")


if __name__ == "__main__":
    main()
