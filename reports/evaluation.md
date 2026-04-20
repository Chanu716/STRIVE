# STRIVE — Model Evaluation Report

## Final Model: XGBoost (tuned with Optuna)

### Test Set Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUROC | 1.0000 | ≥ 0.82 | ✅ |
| AUPRC | 1.0000 | ≥ 0.35 | ✅ |
| F1 @ optimal threshold | 1.0000 | ≥ 0.55 | ✅ |
| ECE | 0.0000 | ≤ 0.08 | ✅ |

**Optimal classification threshold:** 0.9999

### Best Hyperparameters (Optuna, 50 trials)

```json
{
  "max_depth": 5,
  "learning_rate": 0.22648248189516848,
  "n_estimators": 466,
  "subsample": 0.7993292420985183,
  "colsample_bytree": 0.5780093202212182,
  "min_child_weight": 2,
  "scale_pos_weight": 0.9905641928229197
}
```

### Dataset

| Split | Samples | Positive rate |
|-------|---------|---------------|
| Train + Val | 197,758 | 50.0% |
| Test | 34,900 | 50.2% |

### Figures

- `reports/roc_curve.png` — ROC curve
- `reports/pr_curve.png` — Precision-Recall curve
- `reports/calibration.png` — Calibration plot
- `reports/confusion_matrix.png` — Confusion matrix at optimal threshold
