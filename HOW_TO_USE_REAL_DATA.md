#!/usr/bin/env python3
"""
Guide: How to Use Real Datasets Instead of Sample Data

This document explains how to replace the synthetic data with real NHTSA FARS 
and OpenStreetMap data for production use.
"""

# ============================================================================
# PART 1: REAL NHTSA FARS DATA
# ============================================================================

"""
Current Status: Using synthetic sample data (6,000 records)
Real Data Location: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars

Step 1: Download FARS Data Manually
-----------------------------------
1. Visit: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
2. Look for "Downloadable Research Files" section
3. Select years you want (e.g., 2021-2023)
4. Download accident-level CSV files
5. Extract and save to: data/raw/fars_YEAR.csv

Step 2: Update the download script (scripts/download_fars_data.py)
-------------------
Replace the create_sample_fars_data() call with real data loading:

BEFORE (current):
    download_real_fars_data(city, state, years, output_dir)
    # This creates sample data

AFTER (for real data):
    import os
    for year in years:
        input_file = f"data/raw/fars_{year}.csv"
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            logger.info(f"Loaded real FARS data: {input_file} ({len(df)} records)")
            # No modification needed - use as-is
        else:
            logger.error(f"FARS file not found: {input_file}")

Step 3: Important Notes About Real FARS Data
---------------------------------------------
- Records are at accident level (not sample-based like ours)
- Contains actual lat/lon coordinates (our sample used random coordinates)
- May include sensitive information (handle appropriately)
- File sizes: typically 2-5 MB per year
- Columns may vary by year (harmonize if needed)

Real FARS Schema:
  - ST_CASE: Case number
  - STATE: State code
  - CITY: City name
  - LATITUDE: Decimal degrees
  - LONGITUDE: Decimal degrees  
  - FATALS: Number of fatalities
  - DRUNK_DR: Number of drunk drivers
  - DAY: Day of week
  - MONTH: Month
  - HOUR: Hour of day
  - MINUTE: Minute
  - ...and ~50+ more fields

Step 4: Update accident_snapped.parquet workflow
---------------------------------------
Once you have real FARS files:

    python scripts/download_fars_data.py \\
        --city "Los Angeles" \\
        --state "CA" \\
        --years 2021 2022 2023 \\
        --output data/raw \\
        --use-existing  # NEW FLAG: use existing CSV files instead of generating

Step 5: Expected Results with Real Data
---------------------------------------
- Much higher accident counts (real LA has 1000s of fatal accidents per year)
- More realistic geographic clustering
- Historical rates will be more accurate
- Model training will produce better results
"""


# ============================================================================
# PART 2: REAL OPENSTREETMAP DATA
# ============================================================================

"""
Current Status: Using synthetic grid network (224 edges)
Real Data Location: OpenStreetMap (via OSMnx library)

Why Synthetic Network?
- OSM API had temporary issues during development
- OSMnx can take 5-10 minutes for large cities
- Synthetic network is reproducible for testing

How to Use Real OSM Data
------------------------

Option A: Use Real OSM (Recommended for Production)
===================================================

Step 1: Install/Update OSMnx
    pip install --upgrade osmnx networkx

Step 2: Update scripts/download_osm_network.py
    
    # Change this line in the script:
    ALREADY EXISTS - just run it!
    
    python scripts/download_osm_network.py \\
        --place "Los Angeles, California, USA" \\
        --network-type drive \\
        --output data/raw \\
        --validate

Step 3: Handle Large Downloads
    - Los Angeles: ~50,000 edges (takes ~5-10 minutes)
    - San Francisco: ~30,000 edges
    - New York: ~100,000+ edges
    
    For faster processing, use smaller regions:
    
    python scripts/download_osm_network.py \\
        --place "Downtown Los Angeles, California, USA" \\
        --network-type drive \\
        --output data/raw \\
        --validate

Step 4: Expected Output
    - Real road names and types (motorway, primary, secondary, etc.)
    - Accurate speed limits from OSM
    - Actual road geometry from OSM
    - Size: typically 5-50 MB depending on city

Option B: Download OSM Data Offline (Fast)
===========================================

If OSM API is slow/unavailable:

1. Download pre-made OSM extracts:
   - Geofabrik: https://download.geofabrik.de/
   - Download .osm.pbf for your region
   
2. Convert to GraphML:
   
   import osmnx as ox
   G = ox.graph_from_file(
       "california-latest.osm.pbf",
       simplify=True,
       retain_all=False,
       custom_filter='["highway"!~"path|footway|cycleway"]'
   )
   ox.save_graphml(G, "data/raw/road_network.graphml")

Step 5: Verify Real Network
   
   python << 'EOF'
   import osmnx as ox
   G = ox.load_graphml("data/raw/road_network.graphml")
   
   print(f"Nodes: {len(G.nodes)}")
   print(f"Edges: {len(G.edges)}")
   
   # Check attributes
   sample_edge = list(G.edges(keys=True, data=True))[0]
   u, v, k, data = sample_edge
   print(f"Sample edge: {u} -> {v}")
   print(f"Attributes: {list(data.keys())}")
   EOF
"""


# ============================================================================
# PART 3: STEP-BY-STEP MIGRATION GUIDE
# ============================================================================

"""
Complete Migration: Sample Data → Real Data
============================================

Timeline: ~20-30 minutes

Step 1: Prepare Real FARS Data (5 minutes)
-------------------------------------------
1. Visit NHTSA FARS website
2. Download 2021, 2022, 2023 accident-level CSV files
3. Save to: data/raw/fars_2021.csv, fars_2022.csv, fars_2023.csv
4. Verify files exist:
   
   ls -lh data/raw/fars_*.csv

Step 2: Download Real OSM Network (10-15 minutes)
--------------------------------------------------
1. Run OSM download (may take a while):
   
   python scripts/download_osm_network.py \\
       --place "Los Angeles, California, USA" \\
       --network-type drive \\
       --validate

2. Monitor progress (OSMnx will print status)
3. Verify output:
   
   ls -lh data/raw/road_network.graphml

Step 3: Re-run Data Pipeline (5 minutes)
-----------------------------------------
Now run the complete pipeline with real data:

   # Remove old synthetic data (optional, backup first)
   rm data/processed/*.parquet
   
   # Run pipeline with real data
   python data_prepare.py \\
       --city "Los Angeles, CA" \\
       --years 2021 2022 2023 \\
       --skip-fars \\
       --skip-osm

   # Or run individually:
   python scripts/snap_accidents.py
   python scripts/compute_accident_rates.py

Step 4: Verify New Results
--------------------------
Compare with sample data:

   python << 'EOF'
   import pandas as pd
   
   # Check snapped accidents
   snapped = pd.read_parquet("data/processed/accidents_snapped.parquet")
   print(f"Snapped accidents: {len(snapped)}")
   print(f"Snap distance stats (m):")
   print(snapped['snap_distance_m'].describe())
   
   # Check rates
   rates = pd.read_parquet("data/processed/segment_rates.parquet")
   print(f"\nSegments with rates: {len(rates)}")
   print(f"Rate statistics (incidents/km/year):")
   print(rates['historical_accident_rate'].describe())
   EOF
"""


# ============================================================================
# PART 4: TROUBLESHOOTING
# ============================================================================

"""
Common Issues & Solutions
===========================

Issue 1: OSM API Timeout / Rate Limiting
-----------------------------------------
Error: "No data elements in server response"

Solutions:
a) Try smaller area:
   --place "Downtown Los Angeles, CA" instead of full city

b) Use offline download:
   - Download from Geofabrik (https://download.geofabrik.de/)
   - Use ox.graph_from_file() instead of graph_from_place()

c) Use cached data:
   - OSMnx caches by default in ~/.cache/osmnx
   - Delete cache if needed: rm -rf ~/.cache/osmnx

d) Retry with delay:
   import time
   try:
       G = ox.graph_from_place(place)
   except:
       time.sleep(60)  # Wait 1 minute
       G = ox.graph_from_place(place)


Issue 2: NHTSA FARS File Format Changed
----------------------------------------
Error: "Missing columns" or KeyError

Solutions:
a) Check actual columns:
   
   import pandas as pd
   df = pd.read_csv("data/raw/fars_2023.csv")
   print(df.columns.tolist())

b) Update feature engineering if needed:
   - Rename columns to match expected schema
   - Add missing columns with defaults
   - Document changes in comments

c) Contact NHTSA if schema questions
   - https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars


Issue 3: Snapping Failures / Low Match Rate
---------------------------------------------
If snapping match rate drops significantly:

a) Check distance distribution:
   
   import pandas as pd
   snapped = pd.read_parquet("data/processed/accidents_snapped.parquet")
   print(snapped['snap_distance_m'].describe())

b) Increase threshold if needed:
   
   python scripts/snap_accidents.py --threshold 100.0

c) Verify coordinate format:
   - FARS uses decimal degrees (e.g., 34.05, -118.24)
   - OSM should also use EPSG:4326
   - Check for coordinate swaps (lat vs lon)

d) Debug with sample:
   
   python << 'EOF'
   import osmnx as ox
   import pandas as pd
   
   G = ox.load_graphml("data/raw/road_network.graphml")
   fars = pd.read_csv("data/raw/fars_2023.csv", nrows=100)
   
   for idx, row in fars.iterrows():
       try:
           result = ox.nearest_edges(
               G, X=row['LONGITUDE'], Y=row['LATITUDE'], return_dist=True
           )
           edge, dist = result
           print(f"{idx}: dist={dist:.1f}m")
       except Exception as e:
           print(f"{idx}: ERROR - {e}")
           break
   EOF
"""


# ============================================================================
# PART 5: PRODUCTION CHECKLIST
# ============================================================================

"""
Production Deployment Checklist
=================================

Data Quality:
  [ ] Real FARS data downloaded and verified
  [ ] Real OSM network downloaded and verified
  [ ] Historical accident rates computed for all segments
  [ ] Feature engineering pipeline tested with real data
  
Validation:
  [ ] Accident snapping accuracy ≥ 95%
  [ ] Feature vector generation works for all samples
  [ ] No NaN or infinite values in features
  [ ] Performance benchmarks met (< 1ms per feature)
  
Testing:
  [ ] Unit tests still pass with real data
  [ ] Integration tests still pass
  [ ] End-to-end pipeline runs successfully
  
Documentation:
  [ ] Update data source documentation
  [ ] Document any schema changes
  [ ] Record real dataset statistics
  [ ] Update README with real data instructions
  
Handoff:
  [ ] Notify M2 that real data is available
  [ ] Commit real data sources to version control
  [ ] Update environment documentation
"""


# ============================================================================
# QUICK REFERENCE: COMMANDS TO SWITCH TO REAL DATA
# ============================================================================

"""
Quick Reference Commands
=========================

1. Download real FARS (visit website manually)
   - https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
   - Save to: data/raw/fars_YEAR.csv

2. Download real OSM network:
   python scripts/download_osm_network.py \\
       --place "Los Angeles, California, USA" \\
       --network-type drive \\
       --validate

3. Run pipeline with real data:
   python data_prepare.py \\
       --city "Los Angeles, CA" \\
       --years 2021 2022 2023 \\
       --skip-fars --skip-osm

4. Or run individual steps:
   python scripts/snap_accidents.py
   python scripts/compute_accident_rates.py

5. Verify results:
   python -c "
   import pandas as pd
   snapped = pd.read_parquet('data/processed/accidents_snapped.parquet')
   rates = pd.read_parquet('data/processed/segment_rates.parquet')
   print(f'Accidents: {len(snapped)}')
   print(f'Segments: {len(rates)}')
   "
"""
