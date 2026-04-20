#!/usr/bin/env python3
"""
T-07: Train XGBoost Baseline Model

Trains with default-ish hyperparameters, logs AUROC to MLflow,
saves baseline artefact to models/baseline.pkl.
"""

import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import roc_auc_score

from app.ml.features import FEATURE_NAMES

SEED = 42

# ── Load splits ───────────────────────────────────────────────────────────────
print("Loading splits...")
train = pd.read_parquet("data/splits/train.parquet")
val   = pd.read_parquet("data/splits/val.parquet")

X_train, y_train = train[FEATURE_NAMES].values, train["incident"].values
X_val,   y_val   = val[FEATURE_NAMES].values,   val["incident"].values

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  train: {len(X_train):,}  |  val: {len(X_val):,}  |  scale_pos_weight: {pos_weight:.2f}")

# ── Train ─────────────────────────────────────────────────────────────────────
mlflow.set_experiment("strive_baseline")

with mlflow.start_run(run_name="baseline"):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=SEED,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, val_proba)

    mlflow.log_param("n_estimators", model.best_iteration)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_metric("val_auroc", auroc)

    print(f"\n  Best iteration : {model.best_iteration}")
    print(f"  Val AUROC      : {auroc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/baseline.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n[OK] Baseline model saved to models/baseline.pkl")
