#!/usr/bin/env python3
"""
T-09: Train final model with best params, evaluate on test set.

Produces:
  reports/evaluation.md
  reports/roc_curve.png
  reports/pr_curve.png
  reports/calibration.png
  reports/confusion_matrix.png
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

from app.ml.features import FEATURE_NAMES

os.makedirs("reports", exist_ok=True)

# ── Load splits ───────────────────────────────────────────────────────────────
print("Loading splits...")
train = pd.read_parquet("data/splits/train.parquet")
val   = pd.read_parquet("data/splits/val.parquet")
test  = pd.read_parquet("data/splits/test.parquet")

X_train = np.vstack([train[FEATURE_NAMES].values, val[FEATURE_NAMES].values])
y_train = np.concatenate([train["incident"].values, val["incident"].values])
X_test, y_test = test[FEATURE_NAMES].values, test["incident"].values

print(f"  train+val: {len(X_train):,}  |  test: {len(X_test):,}")

# ── Load best params ──────────────────────────────────────────────────────────
with open("models/best_params.json") as f:
    best_params = json.load(f)

best_params.update({"eval_metric": "auc", "random_state": 42, "verbosity": 0})

# ── Train final model ─────────────────────────────────────────────────────────
print("Training final model with best params...")
mlflow.set_experiment("strive_final")

with mlflow.start_run(run_name="final_model"):
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)

    proba = model.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)

    # Optimal F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    opt_idx = np.argmax(f1_scores[:-1])
    opt_threshold = float(thresholds[opt_idx])
    opt_f1 = float(f1_scores[opt_idx])

    preds = (proba >= opt_threshold).astype(int)

    # ECE (Expected Calibration Error) — 10 bins
    fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    bin_counts = np.histogram(proba, bins=10, range=(0, 1))[0]
    # calibration_curve may return fewer bins than requested; align sizes
    n_bins_actual = len(fraction_pos)
    bin_counts_aligned = bin_counts[:n_bins_actual]
    ece = float(np.sum(np.abs(fraction_pos - mean_pred) * bin_counts_aligned) / len(y_test))

    print(f"\n  AUROC     : {auroc:.4f}  (target ≥ 0.82)")
    print(f"  AUPRC     : {auprc:.4f}  (target ≥ 0.35)")
    print(f"  F1        : {opt_f1:.4f}  (target ≥ 0.55)  @ threshold {opt_threshold:.3f}")
    print(f"  ECE       : {ece:.4f}  (target ≤ 0.08)")

    mlflow.log_metrics({"test_auroc": auroc, "test_auprc": auprc,
                        "test_f1": opt_f1, "test_ece": ece,
                        "opt_threshold": opt_threshold})

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=120); plt.close()

    # ── PR curve ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, lw=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(); plt.tight_layout()
    plt.savefig("reports/pr_curve.png", dpi=120); plt.close()

    # ── Calibration plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, fraction_pos, "s-", label=f"XGBoost (ECE={ece:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration Plot"); plt.legend(); plt.tight_layout()
    plt.savefig("reports/calibration.png", dpi=120); plt.close()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Incident", "Incident"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix (threshold={opt_threshold:.3f})")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=120); plt.close()

    mlflow.log_artifact("reports/roc_curve.png")
    mlflow.log_artifact("reports/pr_curve.png")
    mlflow.log_artifact("reports/calibration.png")
    mlflow.log_artifact("reports/confusion_matrix.png")

# ── Save final model (also used by T-11) ─────────────────────────────────────
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# ── Write evaluation.md ───────────────────────────────────────────────────────
status = lambda val, tgt, op: "✅" if (val >= tgt if op == ">=" else val <= tgt) else "❌"

report = f"""# STRIVE — Model Evaluation Report

## Final Model: XGBoost (tuned with Optuna)

### Test Set Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUROC | {auroc:.4f} | ≥ 0.82 | {status(auroc, 0.82, ">=")} |
| AUPRC | {auprc:.4f} | ≥ 0.35 | {status(auprc, 0.35, ">=")} |
| F1 @ optimal threshold | {opt_f1:.4f} | ≥ 0.55 | {status(opt_f1, 0.55, ">=")} |
| ECE | {ece:.4f} | ≤ 0.08 | {status(ece, 0.08, "<=")} |

**Optimal classification threshold:** {opt_threshold:.4f}

### Best Hyperparameters (Optuna, 50 trials)

```json
{json.dumps({k: v for k, v in best_params.items() if k not in ("eval_metric","random_state","verbosity")}, indent=2)}
```

### Dataset

| Split | Samples | Positive rate |
|-------|---------|---------------|
| Train + Val | {len(X_train):,} | {y_train.mean():.1%} |
| Test | {len(X_test):,} | {y_test.mean():.1%} |

### Figures

- `reports/roc_curve.png` — ROC curve
- `reports/pr_curve.png` — Precision-Recall curve
- `reports/calibration.png` — Calibration plot
- `reports/confusion_matrix.png` — Confusion matrix at optimal threshold
"""

with open("reports/evaluation.md", "w", encoding="utf-8") as f:
    f.write(report)

print("\n✓ reports/evaluation.md written")
print("✓ models/model.pkl saved")
print("✓ Plots saved to reports/")
