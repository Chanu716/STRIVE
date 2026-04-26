#!/usr/bin/env python3
"""
Generate a compact PDF research report from the latest evaluation artifacts.
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "research_report.pdf"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_metrics(report_text: str) -> dict[str, str]:
    metrics = {}
    patterns = {
        "AUROC": r"\| AUROC \| ([0-9.]+) \|",
        "AUPRC": r"\| AUPRC \| ([0-9.]+) \|",
        "F1": r"\| F1 @ optimal threshold \| ([0-9.]+) \|",
        "ECE": r"\| ECE \| ([0-9.]+) \|",
        "Threshold": r"\*\*Optimal classification threshold:\*\* ([0-9.]+)",
        "Train+Val": r"\| Train \+ Val \| ([0-9,]+) \| ([0-9.]+%) \|",
        "Test": r"\| Test \| ([0-9,]+) \| ([0-9.]+%) \|",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match:
            metrics[key] = " / ".join(match.groups())
    return metrics


def add_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        wrapped = wrap(line, width=92) or [""]
        for chunk in wrapped:
            ax.text(0.06, y, chunk, fontsize=10, va="top")
            y -= 0.018
        y -= 0.008
        if y < 0.08:
            break
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    evaluation = read_text(REPORTS_DIR / "evaluation.md")
    shap_report = read_text(REPORTS_DIR / "shap_analysis.md")

    metrics = parse_metrics(evaluation)
    top_lines = []
    for line in shap_report.splitlines():
        if line.startswith("| ") and "Feature" not in line and "Rank" not in line:
            top_lines.append(line)
        if len(top_lines) >= 8:
            break

    with PdfPages(OUTPUT_PATH) as pdf:
        add_page(
            pdf,
            "STRIVE Research Report",
            [
                "Problem: predict road-segment risk and produce a safer route recommendation from live weather, road context, and historical crash data.",
                f"Dataset: train+val {metrics.get('Train+Val', 'n/a')}; test {metrics.get('Test', 'n/a')}.",
                "Model: tuned XGBoost classifier with SHAP explanations and a risk-weighted A* routing layer.",
                f"Evaluation: AUROC {metrics.get('AUROC', 'n/a')}, AUPRC {metrics.get('AUPRC', 'n/a')}, F1 {metrics.get('F1', 'n/a')}, ECE {metrics.get('ECE', 'n/a')}.",
            ],
        )
        add_page(
            pdf,
            "Feature Engineering and SHAP Summary",
            [
                "Feature groups: time of day, day of week, month, night indicator, road class, speed limit, precipitation, visibility, wind speed, temperature, rain-on-congestion, historical accident rate.",
                "Global SHAP highlights from the latest report:",
                *top_lines,
                "Domain checks: precipitation in wet-night top-5 PASS; historical accident rate in top-5 FAIL; night indicator positive contribution FAIL.",
            ],
        )
        add_page(
            pdf,
            "Routing and Limitations",
            [
                "Routing layer: the /v1/route/safe endpoint computes a safety-aware path and compares it with the fastest-route baseline.",
                "The route response returns distance, duration, average risk, segment-level risk labels, and a risk-reduction percentage.",
                "Limitations: model quality still trails the AUROC target, SHAP domain checks need improvement, and live performance benchmarking is tracked separately.",
                "Future work: improve training data fidelity, calibrate the classifier, and validate routing against real road-trip case studies.",
            ],
        )

    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
