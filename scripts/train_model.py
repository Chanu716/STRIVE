#!/usr/bin/env python3
"""T-34: End-to-end STRIVE model training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from app.ml.features import FEATURE_NAMES

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised in lightweight local envs
    mlflow = None


SEED = 42
N_FOLDS = 3


def _mlflow_run(run_name: str):
    if mlflow is None:
        return nullcontext()
    return mlflow.start_run(run_name=run_name)


def _log_params(params: dict[str, Any]) -> None:
    if mlflow is not None:
        mlflow.log_params(params)


def _log_metrics(metrics: dict[str, float]) -> None:
    if mlflow is not None:
        mlflow.log_metrics(metrics)


def _log_artifact(path: Path) -> None:
    if mlflow is not None and path.exists():
        mlflow.log_artifact(str(path))


def _load_splits(splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(splits_dir / "train.parquet")
    val = pd.read_parquet(splits_dir / "val.parquet")
    test = pd.read_parquet(splits_dir / "test.parquet")
    return train, val, test


def _split_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame[FEATURE_NAMES].values, frame["incident"].values


def _scale_pos_weight(labels: np.ndarray) -> float:
    positives = max(int((labels == 1).sum()), 1)
    negatives = int((labels == 0).sum())
    return float(negatives / positives)


def _model_params(params: dict[str, Any]) -> dict[str, Any]:
    configured = dict(params)
    configured.update(
        {
            "eval_metric": "auc",
            "random_state": SEED,
            "verbosity": 0,
        }
    )
    return configured


def run_optuna_search(
    train: pd.DataFrame,
    trials: int,
    cv_sample: int,
) -> tuple[dict[str, Any], float]:
    X_train, y_train = _split_xy(train)
    pos_weight = _scale_pos_weight(y_train)

    if len(X_train) > cv_sample:
        X_cv = X_train[-cv_sample:]
        y_cv = y_train[-cv_sample:]
    else:
        X_cv, y_cv = X_train, y_train

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": pos_weight,
        }
        fold_scores: list[float] = []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=N_FOLDS).split(X_cv):
            model = xgb.XGBClassifier(**_model_params(params))
            model.fit(X_cv[train_idx], y_cv[train_idx], verbose=False)
            probabilities = model.predict_proba(X_cv[val_idx])[:, 1]
            fold_scores.append(float(roc_auc_score(y_cv[val_idx], probabilities)))
        score = float(np.mean(fold_scores))
        with _mlflow_run(f"trial_{trial.number}"):
            _log_params(params)
            _log_metrics({"cv_auroc": score})
        return score

    if mlflow is not None:
        mlflow.set_experiment("strive_end_to_end_training")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    with _mlflow_run("optuna_search"):
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
        _log_metrics({"best_cv_auroc": float(study.best_value)})

    best = dict(study.best_params)
    best["scale_pos_weight"] = pos_weight
    return best, float(study.best_value)


def load_or_tune_params(args: argparse.Namespace, train: pd.DataFrame) -> dict[str, Any]:
    params_path = args.models_dir / "best_params.json"
    if args.skip_tuning and params_path.exists():
        with params_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    best_params, best_score = run_optuna_search(train, args.trials, args.cv_sample)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(best_params, handle, indent=2)
    print(f"Best CV AUROC: {best_score:.4f}")
    return best_params


def train_final_model(train: pd.DataFrame, val: pd.DataFrame, params: dict[str, Any]) -> xgb.XGBClassifier:
    train_val = pd.concat([train, val], ignore_index=True)
    X_train, y_train = _split_xy(train_val)
    model = xgb.XGBClassifier(**_model_params(params))
    model.fit(X_train, y_train, verbose=False)
    return model


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    fraction_pos, mean_pred = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    counts = np.histogram(probabilities, bins=bins, range=(0, 1))[0][: len(fraction_pos)]
    return float(np.sum(np.abs(fraction_pos - mean_pred) * counts) / len(labels))


def evaluate_model(
    model: xgb.XGBClassifier,
    test: pd.DataFrame,
    reports_dir: Path,
    params: dict[str, Any],
) -> dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    X_test, y_test = _split_xy(test)
    probabilities = model.predict_proba(X_test)[:, 1]

    auroc = float(roc_auc_score(y_test, probabilities))
    auprc = float(average_precision_score(y_test, probabilities))
    precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    opt_idx = int(np.argmax(f1_scores[:-1]))
    opt_threshold = float(thresholds[opt_idx])
    opt_f1 = float(f1_scores[opt_idx])
    ece = expected_calibration_error(y_test, probabilities)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, lw=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curve.png", dpi=120)
    plt.close()

    fraction_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, fraction_pos, "s-", label=f"XGBoost (ECE={ece:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction Of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "calibration.png", dpi=120)
    plt.close()

    predictions = (probabilities >= opt_threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(matrix, display_labels=["No Incident", "Incident"])
    fig, ax = plt.subplots(figsize=(5, 4))
    display.plot(ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix (threshold={opt_threshold:.3f})")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    metrics = {
        "test_auroc": auroc,
        "test_auprc": auprc,
        "test_f1": opt_f1,
        "test_ece": ece,
        "opt_threshold": opt_threshold,
    }
    _write_evaluation_report(reports_dir, metrics, params, test)
    for artifact in ("roc_curve.png", "pr_curve.png", "calibration.png", "confusion_matrix.png", "evaluation.md"):
        _log_artifact(reports_dir / artifact)
    return metrics


def _target_status(value: float, target: float, op: str) -> str:
    passed = value >= target if op == ">=" else value <= target
    return "PASS" if passed else "MISS"


def _write_evaluation_report(
    reports_dir: Path,
    metrics: dict[str, float],
    params: dict[str, Any],
    test: pd.DataFrame,
) -> None:
    report = f"""# STRIVE Model Evaluation Report

## Final Model: XGBoost

| Metric | Value | Target | Status |
| --- | ---: | ---: | --- |
| AUROC | {metrics["test_auroc"]:.4f} | >= 0.82 | {_target_status(metrics["test_auroc"], 0.82, ">=")} |
| AUPRC | {metrics["test_auprc"]:.4f} | >= 0.35 | {_target_status(metrics["test_auprc"], 0.35, ">=")} |
| F1 @ optimal threshold | {metrics["test_f1"]:.4f} | >= 0.55 | {_target_status(metrics["test_f1"], 0.55, ">=")} |
| ECE | {metrics["test_ece"]:.4f} | <= 0.08 | {_target_status(metrics["test_ece"], 0.08, "<=")} |

Optimal classification threshold: `{metrics["opt_threshold"]:.4f}`

## Best Hyperparameters

```json
{json.dumps(params, indent=2)}
```

## Test Split

Samples: {len(test):,}
Positive rate: {test["incident"].mean():.1%}

## Figures

- `reports/roc_curve.png`
- `reports/pr_curve.png`
- `reports/calibration.png`
- `reports/confusion_matrix.png`
"""
    (reports_dir / "evaluation.md").write_text(report, encoding="utf-8")


def save_artifacts(
    model: xgb.XGBClassifier,
    models_dir: Path,
    threshold: float,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    config = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "threshold": threshold,
        "xgboost_version": xgb.__version__,
        "risk_levels": {
            "LOW": [0, 25],
            "MODERATE": [26, 50],
            "HIGH": [51, 75],
            "CRITICAL": [76, 100],
        },
        "model_path": str(models_dir / "model.pkl"),
        "notes": "risk_score = round(100 * P(incident)). Threshold used for binary classification only.",
    }
    with (models_dir / "feature_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the STRIVE XGBoost model.")
    parser.add_argument("--skip-tuning", action="store_true", help="Load models/best_params.json instead of Optuna.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials when tuning.")
    parser.add_argument("--cv-sample", type=int, default=40_000, help="Chronological sample size for CV tuning.")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, val, test = _load_splits(args.splits_dir)
    params = load_or_tune_params(args, train)

    if mlflow is not None:
        mlflow.set_experiment("strive_end_to_end_training")
    else:
        print("MLflow is not installed; continuing without experiment logging.")

    with _mlflow_run("final_model"):
        _log_params(params)
        model = train_final_model(train, val, params)
        metrics = evaluate_model(model, test, args.reports_dir, params)
        _log_metrics(metrics)
        save_artifacts(model, args.models_dir, metrics["opt_threshold"])

    print("Training pipeline complete.")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved model artifacts to {args.models_dir}")
    print(f"Saved evaluation artifacts to {args.reports_dir}")


if __name__ == "__main__":
    main()
