# STRIVE — Task Sheet: M2 ML Engineer

**Member role:** ML Engineer  
**Depends on:** M1 (`data/processed/features.parquet`, `app/ml/features.py`)  
**Produces for:** M3 (`models/model.pkl`, `models/feature_config.json`), M5 (evaluation results for README)

---

## Responsibility

Train, tune, and evaluate the XGBoost accident-risk model. Validate SHAP explainability. Package the final model artefact consumed by the live API. Own the automated training script and the final research report/slides.

---

## Task List

| ID | Phase | Description | Output |
|---|---|---|---|
| **T-06** | 1 | Create chronological train / val / test split (70 / 15 / 15) | `data/splits/` |
| **T-07** | 1 | Train XGBoost baseline model with default hyperparameters; log to MLflow | `models/baseline.pkl`, MLflow run |
| **T-08** | 1 | Run Optuna hyperparameter search (50 trials, 3-fold time-series CV) | `models/best_params.json`, MLflow experiment |
| **T-09** | 1 | Evaluate final model: AUROC, AUPRC, F1 @ optimal threshold, ECE | `reports/evaluation.md`, plots |
| **T-10** | 1 | Validate SHAP explanations — confirm top factors match domain expectations | `reports/shap_analysis.ipynb` |
| **T-11** | 1 | Save model artefact and feature config | `models/model.pkl`, `models/feature_config.json` |
| **T-34** | 4 | Write `scripts/train_model.py` — end-to-end training pipeline with MLflow logging | `scripts/train_model.py` |
| **T-41** | 4 | Prepare research report / slides: model evaluation, SHAP analysis, routing quality | `reports/research_report.pdf` |

**Total: 8 tasks**

---

## Detailed Task Notes

### T-06 — Chronological Train / Val / Test Split

- Load `data/processed/features.parquet`.
- Sort by `timestamp` ascending.
- Split: first 70 % → train, next 15 % → val, final 15 % → test.
- **No random shuffle** — chronological order must be preserved to avoid temporal leakage.
- Save splits as parquet: `data/splits/train.parquet`, `val.parquet`, `test.parquet`.

### T-07 — Baseline XGBoost Model

```python
import xgboost as xgb
from mlflow import log_metric, log_artifact

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric="auc",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
```

- Log AUROC on val set to MLflow.
- Save baseline artefact to `models/baseline.pkl`.

### T-08 — Optuna Hyperparameter Search

Tune the following parameters:
- `max_depth`: 3–10
- `learning_rate`: 1e-3 – 0.3 (log-scale)
- `n_estimators`: 100–1000
- `subsample`: 0.5–1.0
- `colsample_bytree`: 0.5–1.0
- `min_child_weight`: 1–10

Use 3-fold **time-series cross-validation** (no shuffle; each fold preserves temporal order).  
Objective: maximise AUROC on the val set.  
Run 50 trials; log each trial to MLflow. Save best params to `models/best_params.json`.

### T-09 — Model Evaluation

Evaluate on the held-out **test set** (never used during training or tuning):

| Metric | Target |
|---|---|
| AUROC | ≥ 0.82 |
| AUPRC | ≥ 0.35 |
| F1 (optimal threshold) | ≥ 0.55 |
| ECE (Expected Calibration Error) | ≤ 0.08 |

Produce:
- ROC and PR curves (saved as PNG).
- Calibration plot.
- Confusion matrix at optimal F1 threshold.
- `reports/evaluation.md` summarising all metrics and figures.

### T-10 — SHAP Validation

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES)
```

- Verify that **`precipitation_mm`** and **`historical_accident_rate`** appear in the top 5 SHAP features for a "wet night" test case.
- Verify that **`night_indicator`** has a positive contribution for night-time predictions.
- Save SHAP summary bar chart and beeswarm plot as `reports/shap_summary.png`, `reports/shap_beeswarm.png`.
- Document findings in `reports/shap_analysis.ipynb`.

### T-11 — Save Model Artefact

```python
import pickle, json
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

config = {
    "feature_names": FEATURE_NAMES,
    "threshold": optimal_threshold,
    "xgboost_version": xgb.__version__,
}
with open("models/feature_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

The `models/` directory must be committed (use `.gitignore` to exclude large data files but include model artefacts < 50 MB).

### T-34 — `scripts/train_model.py`

End-to-end script that:
1. Loads splits from `data/splits/`.
2. Runs Optuna search (or loads `models/best_params.json` if `--skip-tuning`).
3. Retrains final model on train + val with best params.
4. Evaluates on test set; prints and logs all metrics to MLflow.
5. Saves `models/model.pkl` and `models/feature_config.json`.

```bash
python scripts/train_model.py --skip-tuning   # use saved best params
python scripts/train_model.py --trials 50     # run full Optuna search
```

### T-41 — Research Report / Slides

The report should cover:
1. **Problem motivation** — why safety-aware routing matters.
2. **Dataset description** — FARS years, city, accident count, class imbalance ratio.
3. **Feature engineering** — which features were used and why.
4. **Modelling decisions** — why XGBoost; baseline vs tuned comparison.
5. **Evaluation results** — all four metrics, curves, and calibration.
6. **SHAP analysis** — top factors, domain interpretation, sample explanations.
7. **Routing quality** — case study showing safe route vs fastest route risk reduction.
8. **Limitations & future work**.

---

## Deliverables Checklist

- [ ] `data/splits/train.parquet`, `val.parquet`, `test.parquet`
- [ ] `models/baseline.pkl`
- [ ] `models/best_params.json`
- [ ] `models/model.pkl`
- [ ] `models/feature_config.json`
- [ ] `reports/evaluation.md` with all four metrics
- [ ] `reports/shap_analysis.ipynb`
- [ ] `reports/shap_summary.png`, `reports/shap_beeswarm.png`
- [ ] `scripts/train_model.py`
- [ ] `reports/research_report.pdf` (or `.pptx` slides)

---

## Dependencies from Others

| Needs | Provided by |
|---|---|
| `data/processed/features.parquet` | M1 (T-05) |
| `app/ml/features.py` (`FEATURE_NAMES`) | M1 (T-05) |

## What Others Depend On from M2

| Deliverable | Used by |
|---|---|
| `models/model.pkl` | M3 (inference in risk endpoints) |
| `models/feature_config.json` | M3 (feature ordering at inference time) |
| `reports/evaluation.md` | M5 (README update) |
