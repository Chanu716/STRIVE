#!/usr/bin/env python3
"""
T-11: Save final model artefact and feature config.

models/model.pkl is already saved by T-09.
This script writes models/feature_config.json with the full config
consumed by M3 at inference time.
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

from app.ml.features import FEATURE_NAMES

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Derive optimal threshold from test set ────────────────────────────────────
test = pd.read_parquet("data/splits/test.parquet")
X_test = test[FEATURE_NAMES].values
y_test = test["incident"].values

proba = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
opt_idx = int(np.argmax(f1_scores[:-1]))
opt_threshold = float(thresholds[opt_idx])

auroc = roc_auc_score(y_test, proba)
print(f"  AUROC          : {auroc:.4f}")
print(f"  Opt threshold  : {opt_threshold:.4f}")

# ── Write feature_config.json ─────────────────────────────────────────────────
config = {
    "feature_names": FEATURE_NAMES,
    "n_features": len(FEATURE_NAMES),
    "threshold": opt_threshold,
    "xgboost_version": xgb.__version__,
    "risk_levels": {
        "LOW":      [0,  25],
        "MODERATE": [26, 50],
        "HIGH":     [51, 75],
        "CRITICAL": [76, 100],
    },
    "model_path": "models/model.pkl",
    "notes": "risk_score = round(100 * P(incident)). Threshold used for binary classification only.",
}

os.makedirs("models", exist_ok=True)
with open("models/feature_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n[OK] models/feature_config.json saved")
print("[OK] models/model.pkl already present")

# ── Verify both files exist ───────────────────────────────────────────────────
for path in ["models/model.pkl", "models/feature_config.json"]:
    size_kb = os.path.getsize(path) / 1024
    print(f"  {path}  ({size_kb:.1f} KB)")
