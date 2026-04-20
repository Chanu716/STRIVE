#!/usr/bin/env python3
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ml.features import create_training_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser("Build Features Dataset")
    parser.add_argument("--snapped", default="data/processed/accidents_snapped.parquet")
    parser.add_argument("--rates", default="data/processed/segment_rates.parquet")
    parser.add_argument("--network", default="data/raw/road_network.graphml")
    parser.add_argument("--output", default="data/processed/features.parquet")
    args = parser.parse_args()

    logger.info("Loading inputs...")
    snapped = pd.read_parquet(args.snapped)
    rates = pd.read_parquet(args.rates)
    
    # Extract road attributes
    logger.info("Extracting road attributes from graph...")
    G = ox.load_graphml(args.network)
    edges_data = []
    for u, v, k, data in G.edges(keys=True, data=True):
        edges_data.append({
            'osmid': f"{u}_{v}",
            'highway': data.get('highway', 'unclassified'),
            'speed_limit_kmh': data.get('speed_kph', 50),
        })
    edges_df = pd.DataFrame(edges_data)
    
    logger.info("Merging datasets...")
    # Add timestamp to snapped accidents
    # Handle missing date parts
    for col in ['year', 'month', 'day', 'hour', 'minute']:
        if col not in snapped:
            snapped[col] = 1 if col in ['month', 'day'] else 0
    
    # some FARS data might have hour=99 (unknown). Clip/replace it.
    snapped['hour'] = snapped['hour'].clip(0, 23).replace(99, 12)
    snapped['minute'] = snapped['minute'].clip(0, 59).replace(99, 0)
    snapped['month'] = snapped['month'].clip(1, 12)
    snapped['day'] = snapped['day'].clip(1, 28) # simplify days to avoid e.g. Feb 30
    
    snapped['timestamp'] = pd.to_datetime(
        dict(year=snapped.year, month=snapped.month, day=snapped.day, hour=snapped.hour, minute=snapped.minute),
        errors='coerce'
    ).fillna(pd.Timestamp("2021-01-01 12:00:00"))

    # Merge with rates and road attributes
    # remove duplicate osm_id
    rates = rates.drop_duplicates(subset=['osmid'])
    edges_df = edges_df.drop_duplicates(subset=['osmid'])

    data = snapped.merge(rates, on='osmid', how='left')
    data = data.merge(edges_df, on='osmid', how='left')
    
    # Fill missing values
    data['historical_accident_rate'] = data['historical_accident_rate'].fillna(0)
    data['highway'] = data['highway'].fillna('unclassified')
    data['speed_limit_kmh'] = data['speed_limit_kmh'].fillna(50)
    
    # Create dataset
    logger.info("Building feature dataset using app.ml.features...")
    features_df, labels = create_training_dataset(data, edges_df)
    features_df['incident'] = labels
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_parquet(args.output)
    logger.info(f"Saved {len(features_df)} records to {args.output}")

if __name__ == "__main__":
    main()
