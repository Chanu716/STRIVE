#!/usr/bin/env python3
"""
T-08: Optuna Hyperparameter Search (50 trials, 3-fold time-series CV)

Tunes XGBoost hyperparameters, logs each trial to MLflow,
saves best params to models/best_params.json.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from app.ml.features import FEATURE_NAMES

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED  = 42
N_TRIALS = 50
N_FOLDS  = 3
CV_SAMPLE = 40000   # subsample for CV speed (still chronological)

# ── Load train split ──────────────────────────────────────────────────────────
print("Loading train split...")
train = pd.read_parquet("data/splits/train.parquet")
X_train = train[FEATURE_NAMES].values
y_train = train["incident"].values
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"  train: {len(X_train):,}  |  scale_pos_weight: {pos_weight:.2f}")

# Subsample for CV (preserve chronological order — take last CV_SAMPLE rows)
if len(X_train) > CV_SAMPLE:
    X_cv = X_train[-CV_SAMPLE:]
    y_cv = y_train[-CV_SAMPLE:]
else:
    X_cv, y_cv = X_train, y_train
print(f"  CV sample: {len(X_cv):,}")

# ── Objective ─────────────────────────────────────────────────────────────────
mlflow.set_experiment("strive_optuna")

def objective(trial: optuna.Trial) -> float:
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": pos_weight,
        "eval_metric":      "auc",
        "random_state":     SEED,
        "verbosity":        0,
    }

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_aurocs = []

    for fold_train_idx, fold_val_idx in tscv.split(X_cv):
        Xf_tr, yf_tr = X_cv[fold_train_idx], y_cv[fold_train_idx]
        Xf_val, yf_val = X_cv[fold_val_idx], y_cv[fold_val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(Xf_tr, yf_tr, verbose=False)
        proba = model.predict_proba(Xf_val)[:, 1]
        fold_aurocs.append(roc_auc_score(yf_val, proba))

    mean_auroc = float(np.mean(fold_aurocs))

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_auroc", mean_auroc)

    return mean_auroc

# ── Run study ─────────────────────────────────────────────────────────────────
print(f"\nRunning Optuna search ({N_TRIALS} trials, {N_FOLDS}-fold time-series CV)...")

with mlflow.start_run(run_name="optuna_search"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
best["scale_pos_weight"] = pos_weight
best_auroc = study.best_value

print(f"\n  Best CV AUROC : {best_auroc:.4f}")
print(f"  Best params   : {json.dumps(best, indent=2)}")

mlflow.log_metric("best_cv_auroc", best_auroc)
mlflow.log_params({k: v for k, v in best.items() if k != "scale_pos_weight"})

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/best_params.json", "w") as f:
    json.dump(best, f, indent=2)

print("\n✓ Best params saved to models/best_params.json")
