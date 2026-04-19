# Current Data Status & How to Switch to Real Data

## 📊 Current Setup: Sample Data

We currently have **synthetic/sample data** for development and testing:

| Component | Current | Size | Real Alternative |
|-----------|---------|------|------------------|
| **FARS Accidents** | Synthetic (6,000 records) | 578 KB | Real NHTSA FARS CSV |
| **Road Network** | Synthetic grid (224 segments) | 75 KB | Real OSM ~50,000 segments |
| **Snapped Data** | 100% match (synthetic) | 240 KB | Real data (varies) |
| **Rates** | Computed (all 224 segments) | 3.9 KB | Real data (all segments) |

**Why Sample Data?**
- OSM API had temporary issues during development
- Real FARS requires manual download from NHTSA website
- Sample data allows testing without long setup times
- Synthetic network is reproducible & deterministic

---

## 🔄 How to Switch to Real Data

### Option 1: Quick Switch (Recommended) - 20 minutes

```bash
# 1. Manually download real FARS data
#    Visit: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
#    Download: Accident-level CSV for 2021, 2022, 2023
#    Save to: /your/local/path/fars_files/

# 2. Run migration script
python migrate_to_real_data.py \
    --fars-dir "/your/local/path/fars_files" \
    --download-osm \
    --backup \
    --reprocess
```

**What This Does:**
- ✓ Backs up existing sample data to `data/backup_sample_data/`
- ✓ Copies real FARS CSVs to `data/raw/`
- ✓ Downloads real OSM network (~5-15 min)
- ✓ Verifies data schemas
- ✓ Re-runs entire pipeline automatically
- ✓ Generates new processed data files

**Result:** Real data replaces sample data, pipeline re-executes

---

### Option 2: Manual Step-by-Step (for debugging)

```bash
# Step 1: Get real FARS data
# Visit: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
# Download and save: data/raw/fars_2021.csv, fars_2022.csv, fars_2023.csv

# Step 2: Get real OSM network (takes 5-15 minutes)
python scripts/download_osm_network.py \
    --place "Los Angeles, California, USA" \
    --network-type drive \
    --output data/raw \
    --validate

# Step 3: Re-snap accidents
python scripts/snap_accidents.py \
    --fars-dir data/raw \
    --network data/raw/road_network.graphml \
    --threshold 50.0 \
    --output data/processed/accidents_snapped.parquet

# Step 4: Compute new accident rates
python scripts/compute_accident_rates.py \
    --snapped data/processed/accidents_snapped.parquet \
    --network data/raw/road_network.graphml \
    --years 3 \
    --output data/processed/segment_rates.parquet

# Done! Your real data is now in use.
```

---

## 📥 Step-by-Step: Get Real FARS Data

### From NHTSA Website

1. Visit: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
2. Click "Downloadable Research Files"
3. Select year (2021, 2022, 2023)
4. Download "Accident File (NMS05 File)" - CSV format
5. Repeat for each year
6. Save all to same directory (e.g., `~/Downloads/fars/`)

**Files you'll get:**
- `fars_2021.csv` (~3-5 MB)
- `fars_2022.csv` (~3-5 MB)
- `fars_2023.csv` (~3-5 MB)

**Or download programmatically** (if NHTSA allows):
```python
# Note: NHTSA may require authentication/registration
# Check their API documentation

import requests
import pandas as pd

# Example (check NHTSA for actual endpoints)
years = [2021, 2022, 2023]
for year in years:
    url = f"https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/..."
    df = pd.read_csv(url)
    df.to_csv(f"data/raw/fars_{year}.csv", index=False)
```

---

## 📥 Step-by-Step: Get Real OSM Data

### Automatic Download (Recommended)

```bash
# This will download real Los Angeles road network
python scripts/download_osm_network.py \
    --place "Los Angeles, California, USA" \
    --network-type drive \
    --validate

# Takes 5-15 minutes, downloads ~20-50 MB
# Result: data/raw/road_network.graphml (~50 MB)
```

### Offline Download (If OSM API is Down)

1. Visit: https://download.geofabrik.de/
2. Download: `california-latest.osm.pbf` (~700 MB)
3. Place in `data/raw/`
4. Run:
```python
import osmnx as ox

G = ox.graph_from_file(
    "data/raw/california-latest.osm.pbf",
    simplify=True,
    retain_all=False,
    custom_filter='["highway"!~"path|footway|cycleway|bridleway|steps|pedestrian"]'
)
ox.save_graphml(G, "data/raw/road_network.graphml")
```

---

## ✅ Verify Data Switch

After migration, check that real data is in use:

```bash
# 1. Check file sizes (real data will be much larger)
ls -lh data/raw/

# Expected for real data:
# - fars_2021.csv: 3-5 MB (vs 193 KB sample)
# - fars_2022.csv: 3-5 MB (vs 192 KB sample)
# - fars_2023.csv: 3-5 MB (vs 193 KB sample)
# - road_network.graphml: 20-50 MB (vs 75 KB sample)

# 2. Check accident counts
python << 'EOF'
import pandas as pd

snapped = pd.read_parquet('data/processed/accidents_snapped.parquet')
rates = pd.read_parquet('data/processed/segment_rates.parquet')

print(f"Total accidents: {len(snapped)}")
print(f"Road segments: {len(rates)}")
print(f"Accident rate stats:")
print(rates['historical_accident_rate'].describe())
EOF

# Expected for real data:
# - Total accidents: 10,000-50,000 (vs 6,000 sample)
# - Road segments: 50,000+ (vs 224 sample)
# - More varied accident rates
```

---

## 🚨 Troubleshooting

### Issue: OSM API Timeout

**Solution 1:** Use smaller area
```bash
python scripts/download_osm_network.py \
    --place "Downtown Los Angeles, CA" \
    --validate
```

**Solution 2:** Use offline Geofabrik data
- Download from: https://download.geofabrik.de/
- See "Offline Download" section above

**Solution 3:** Wait and retry
```bash
import time
import osmnx as ox

for attempt in range(3):
    try:
        G = ox.graph_from_place("Los Angeles, California, USA")
        break
    except:
        print(f"Attempt {attempt+1} failed, retrying...")
        time.sleep(60)
```

### Issue: FARS CSV Schema Mismatch

**Check columns:**
```python
import pandas as pd

df = pd.read_csv("data/raw/fars_2021.csv")
print(df.columns.tolist())
```

**Map column names if needed:**
```python
# If NHTSA changed column names, create mapping:
column_mapping = {
    'LAT': 'LATITUDE',
    'LON': 'LONGITUDE',
    'MO': 'MONTH',
    'HR': 'HOUR',
}
df.rename(columns=column_mapping, inplace=True)
```

### Issue: Low Snapping Match Rate

**Check why accidents don't match:**
```python
import osmnx as ox
import pandas as pd

G = ox.load_graphml("data/raw/road_network.graphml")
fars = pd.read_csv("data/raw/fars_2021.csv", nrows=100)

for idx, row in fars.iterrows():
    try:
        result = ox.nearest_edges(
            G, X=row['LONGITUDE'], Y=row['LATITUDE'], return_dist=True
        )
        edge, dist = result
        if dist > 100:  # > 100 meters
            print(f"Accident {idx}: far from road ({dist:.1f}m)")
    except Exception as e:
        print(f"Accident {idx}: ERROR - {e}")
        break
```

---

## 📚 Complete Guide

For detailed instructions and troubleshooting, see:
- `HOW_TO_USE_REAL_DATA.md` - Comprehensive guide
- `migrate_to_real_data.py` - Automated migration script

---

## 🎯 Summary

| Action | Command | Time |
|--------|---------|------|
| Quick migration | `python migrate_to_real_data.py --fars-dir /path --download-osm` | 20 min |
| Manual switch | See "Manual Step-by-Step" above | 15 min |
| Just download OSM | `python scripts/download_osm_network.py --validate` | 10-15 min |
| Just copy FARS | `python migrate_to_real_data.py --fars-dir /path` | 2 min |

**Default (what we use now):**
- Sample FARS data (synthetic, 6,000 records)
- Synthetic road network (224 segments)
- ✓ Good for testing & development
- ✓ Fast setup
- ✗ Not production data

**Real Data (what you should use):**
- Real NHTSA FARS (10,000s of records)
- Real OSM network (50,000+ segments)
- ✓ Production-quality
- ✓ More accurate results
- ✗ Slower setup (20-30 minutes)

