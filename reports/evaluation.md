# STRIVE — Model Evaluation Report

## Final Model: XGBoost (tuned with Optuna)

### Test Set Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUROC | 0.6942 | >= 0.82 | [FAIL] |
| AUPRC | 0.7233 | >= 0.35 | ✅ |
| F1 @ optimal threshold | 0.6953 | >= 0.55 | ✅ |
| ECE | 0.0058 | <= 0.08 | ✅ |

**Optimal classification threshold:** 0.3480

### Best Hyperparameters (Optuna, 50 trials)

```json
{
  "max_depth": 3,
  "learning_rate": 0.15407468620744033,
  "n_estimators": 455,
  "subsample": 0.9675975947587264,
  "colsample_bytree": 0.5529539197128122,
  "min_child_weight": 7,
  "scale_pos_weight": 0.9957844170486018
}
```

### Dataset

| Split | Samples | Positive rate |
|-------|---------|---------------|
| Train + Val | 197,758 | 49.8% |
| Test | 34,900 | 50.9% |

### Figures

- `reports/roc_curve.png` — ROC curve
- `reports/pr_curve.png` — Precision-Recall curve
- `reports/calibration.png` — Calibration plot
- `reports/confusion_matrix.png` — Confusion matrix at optimal threshold
