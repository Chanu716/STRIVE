# Quick Start: Your Real Data Pipeline

## 🎯 You're Here

```
Real FARS Data ✅ → OSM Download ⏳ → Snap Accidents → Compute Rates → Ready for M2
```

---

## ⏳ Currently Happening

**OSM California Road Network Download** (Running now)
- Estimated time: 10-20 minutes
- File will grow to: 50-100 MB
- Status: In background

**Check progress:**
```bash
ls -lh d:/STRIVE/data/raw/road_network.graphml
```

If it's still 75 KB after 5 minutes, download may need attention.

---

## 🚀 Next Command (Run After OSM Download)

Once the road network file is > 10 MB, run:

```bash
cd d:/STRIVE
python run_pipeline_wait_for_osm.py
```

This will:
- ✓ Automatically wait for network if not ready
- ✓ Validate the road network
- ✓ Snap 73,000 accidents to segments
- ✓ Compute historical rates
- ✓ Verify all outputs

**Expected time: 15-20 minutes total** (including waiting)

---

## 📊 What You'll See

```
================================================================
M1 DATA PIPELINE - AUTOMATED RUNNER
================================================================

Step 1: Waiting for Road Network Download...
  Validating road network...
    Nodes: 150,000
    Edges: 200,000

Step 2: Validating Road Network...

Step 3: Running Accident Snapping...
  Loading FARS data from data/raw...
  Loaded 73,000 snapped accidents
  Snapped 68,000 accidents (within threshold)
  Match rate: 93.2%

Step 4: Computing Historical Rates...
  Loaded graph: 200,000 nodes, 300,000 edges
  Computed rates for 200,000 segments
  Mean rate: 0.85 incidents/km/year

Step 5: Verifying Results...
  Snapped accidents: 68,000
  Road segments with rates: 200,000
  
================================================================
SUCCESS! Pipeline Complete
================================================================

Your real data is ready for M2 (ML Engineer):
  data/processed/accidents_snapped.parquet
  data/processed/segment_rates.parquet
  app/ml/features.py (feature pipeline)
```

---

## 🛠️ If Something Goes Wrong

### OSM Download Fails
```bash
# Try with a smaller region first
python scripts/download_osm_network.py \
    --place "Los Angeles, California, USA" \
    --validate

# Or use offline data from Geofabrik
# https://download.geofabrik.de/north-america/us/california.html
```

### Snapping has low match rate
```bash
# Check why accidents don't match
python << 'EOF'
import pandas as pd
snapped = pd.read_parquet("data/processed/accidents_snapped.parquet")
print(f"Snapped: {len(snapped)}")
print(f"Mean snap distance: {snapped['snap_distance_m'].mean():.1f} m")
EOF
```

### Feature pipeline not working
```bash
# Test it
cd d:/STRIVE
pytest tests/unit/test_features.py -v
```

---

## 📋 Checklist

- [ ] Real FARS data downloaded (3 files, 25 MB each) ✅
- [ ] OSM California network downloading ⏳
- [ ] Ready to run `python run_pipeline_wait_for_osm.py`
- [ ] All data processed and verified
- [ ] Git commit ready

---

## 💾 Files Location

```
d:/STRIVE/data/
├── raw/
│   ├── fars_2021.csv (25 MB) ✅
│   ├── fars_2022.csv (25 MB) ✅
│   ├── fars_2023.csv (25 MB) ✅
│   └── road_network.graphml (downloading...)
└── processed/
    ├── accidents_snapped.parquet (will be created)
    └── segment_rates.parquet (will be created)
```

---

## ✅ Manual Work Required

**NONE!** Everything is automated from here.

Just:
1. Wait for OSM download
2. Run the pipeline script
3. Done!

---

## 🎯 Timeline

- **Now:** OSM downloading (10-20 min)
- **+20 min:** Run pipeline script (15-20 min)
- **+40 min:** Data ready for M2 ✅

Total time from now: ~40 minutes

---

**Status:** Waiting for OSM download to complete...

Check back in 15-20 minutes and run:
```bash
python run_pipeline_wait_for_osm.py
```
