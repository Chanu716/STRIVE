#!/usr/bin/env python3
"""
End-to-end training pipeline for STRIVE.

This script stitches together the existing phase-specific helpers so one command
can prepare splits, tune hyperparameters, train the final model, generate SHAP
artifacts, and export the inference config.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_step(label: str, script: str, env: dict[str, str] | None = None) -> None:
    command = [PYTHON, script]
    print(f"\n==> {label}")
    print("    " + " ".join(command))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=REPO_ROOT, env=merged_env, check=True)


def required_outputs_exist() -> bool:
    return all((REPO_ROOT / path).exists() for path in [
        "data/splits/train.parquet",
        "data/splits/val.parquet",
        "data/splits/test.parquet",
        "models/best_params.json",
        "models/model.pkl",
        "models/feature_config.json",
        "reports/evaluation.md",
        "reports/shap_analysis.md",
        "reports/shap_summary.png",
        "reports/shap_beeswarm.png",
        "reports/research_report.pdf",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STRIVE training pipeline end to end.")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Reuse an existing models/best_params.json instead of rerunning Optuna.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials to request when tuning.",
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="Force regeneration of the train/val/test split files.",
    )
    args = parser.parse_args()

    splits_ready = all((REPO_ROOT / path).exists() for path in [
        "data/splits/train.parquet",
        "data/splits/val.parquet",
        "data/splits/test.parquet",
    ])

    if args.rebuild_splits or not splits_ready:
        run_step("Splitting dataset", "scripts/split_dataset.py")

    run_step("Training baseline model", "scripts/train_baseline.py")

    best_params_path = REPO_ROOT / "models" / "best_params.json"
    if not args.skip_tuning or not best_params_path.exists():
        run_step(
            "Hyperparameter tuning",
            "scripts/tune_hyperparams.py",
            env={"STRIVE_OPTUNA_TRIALS": str(args.trials)},
        )
    else:
        print("\n==> Hyperparameter tuning")
        print("    Reusing models/best_params.json")

    run_step("Final training and evaluation", "scripts/evaluate_model.py")
    run_step("SHAP analysis", "scripts/explain_shap.py")
    run_step("Exporting artefact config", "scripts/save_artefacts.py")
    run_step("Generating research report", "scripts/generate_research_report.py")

    print("\n==> Pipeline complete")
    for path in [
        "models/baseline.pkl",
        "models/best_params.json",
        "models/model.pkl",
        "models/feature_config.json",
        "reports/evaluation.md",
        "reports/shap_analysis.md",
        "reports/research_report.pdf",
    ]:
        exists = (REPO_ROOT / path).exists()
        print(f"    {path}: {'present' if exists else 'missing'}")

    if not required_outputs_exist():
        raise SystemExit("One or more expected outputs are still missing.")


if __name__ == "__main__":
    main()
