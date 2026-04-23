import pandas as pd
import os

# M1 outputs
snapped = pd.read_parquet("data/processed/accidents_snapped.parquet")
rates   = pd.read_parquet("data/processed/segment_rates.parquet")
feats   = pd.read_parquet("data/processed/features.parquet")

# M2 splits
train = pd.read_parquet("data/splits/train.parquet")
val   = pd.read_parquet("data/splits/val.parquet")
test  = pd.read_parquet("data/splits/test.parquet")

print("=== M1 Outputs ===")
print(f"  accidents_snapped : {snapped.shape}  cols: {list(snapped.columns)}")
print(f"  segment_rates     : {rates.shape}   cols: {list(rates.columns)}")
print(f"  features          : {feats.shape}   cols: {list(feats.columns)}")
print(f"  snap_distance stats (m): mean={snapped['snap_distance_m'].mean():.1f}  max={snapped['snap_distance_m'].max():.1f}")
print(f"  rate stats: mean={rates['historical_accident_rate'].mean():.4f}  max={rates['historical_accident_rate'].max():.1f}")

print()
print("=== M2 Splits ===")
total = len(train)+len(val)+len(test)
print(f"  train : {len(train):,}  ({len(train)/total:.1%})  incident rate={train['incident'].mean():.2%}")
print(f"  val   : {len(val):,}   ({len(val)/total:.1%})  incident rate={val['incident'].mean():.2%}")
print(f"  test  : {len(test):,}   ({len(test)/total:.1%})  incident rate={test['incident'].mean():.2%}")
print(f"  total combined : {total:,}")

print()
print("=== Models ===")
for f in ["models/baseline.pkl","models/model.pkl","models/best_params.json","models/feature_config.json"]:
    size = os.path.getsize(f)
    print(f"  {f}  ->  {size/1024:.1f} KB")

print()
print("=== Reports ===")
for f in os.listdir("reports"):
    size = os.path.getsize(f"reports/{f}")
    print(f"  reports/{f}  ->  {size/1024:.1f} KB")

print()
print("=== Missing deliverables check ===")
deliverables = {
    "T-01 data/raw/fars_2021.csv": "data/raw/fars_2021.csv",
    "T-01 data/raw/fars_2022.csv": "data/raw/fars_2022.csv",
    "T-01 data/raw/fars_2023.csv": "data/raw/fars_2023.csv",
    "T-02 data/raw/road_network.graphml": "data/raw/road_network.graphml",
    "T-03 accidents_snapped.parquet": "data/processed/accidents_snapped.parquet",
    "T-04 segment_rates.parquet": "data/processed/segment_rates.parquet",
    "T-05 features.parquet": "data/processed/features.parquet",
    "T-05 app/ml/features.py": "app/ml/features.py",
    "T-06 data/splits/train.parquet": "data/splits/train.parquet",
    "T-06 data/splits/val.parquet": "data/splits/val.parquet",
    "T-06 data/splits/test.parquet": "data/splits/test.parquet",
    "T-07 models/baseline.pkl": "models/baseline.pkl",
    "T-08 models/best_params.json": "models/best_params.json",
    "T-09 reports/evaluation.md": "reports/evaluation.md",
    "T-09 reports/roc_curve.png": "reports/roc_curve.png",
    "T-09 reports/pr_curve.png": "reports/pr_curve.png",
    "T-09 reports/calibration.png": "reports/calibration.png",
    "T-09 reports/confusion_matrix.png": "reports/confusion_matrix.png",
    "T-10 reports/shap_summary.png": "reports/shap_summary.png",
    "T-10 reports/shap_beeswarm.png": "reports/shap_beeswarm.png",
    "T-11 models/model.pkl": "models/model.pkl",
    "T-11 models/feature_config.json": "models/feature_config.json",
}
for label, path in deliverables.items():
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"  [{status}] {label}")
