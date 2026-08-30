from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STAGE_DIR / "outputs"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
        }
    )


def plot_early_quality_quintiles() -> Path:
    data = pd.read_csv(OUTPUT_DIR / "early_quality_quintile_ic.csv")
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    colors = {"5d": "#2563a6", "20d": "#b42318", "1m": "#18794e"}
    for (frequency, horizon), group in data.groupby(["Frequency", "Horizon"]):
        label = f"{frequency} {horizon}"
        axis.plot(
            group["QualityQuintile"],
            group["SpearmanIC"],
            marker="o",
            linewidth=2.2,
            label=label,
            color=colors[horizon],
        )
    axis.axhline(0.0, color="#627184", linewidth=1.0)
    axis.set_xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    axis.set_xlabel("Stage 30 composite data-quality quintile")
    axis.set_ylabel("Raw ODS Spearman IC")
    axis.set_title("2007–2017: higher composite quality did not produce monotone IC")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    output_path = OUTPUT_DIR / "early_quality_quintile_ic.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_degradation_distribution() -> Path:
    replications = pd.read_csv(
        OUTPUT_DIR / "late_degradation_replications.csv"
    )
    comparison = pd.read_csv(
        OUTPUT_DIR / "late_degradation_comparison.csv"
    ).set_index("Metric")
    metrics = [
        ("CAGR", "CAGR", 100.0, "%"),
        ("Sharpe", "Sharpe", 1.0, ""),
        ("MDD", "MDD", 100.0, "%"),
        ("Calmar", "Calmar", 1.0, ""),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for axis, (column, label, scale, suffix) in zip(axes.ravel(), metrics):
        values = replications[column] * scale
        original = comparison.loc[label, "OriginalLate"] * scale
        axis.hist(values, bins=8, color="#7ca8c9", edgecolor="white")
        axis.axvline(
            original,
            color="#b42318",
            linewidth=2.2,
            label=f"original {original:.3f}{suffix}",
        )
        axis.axvline(
            values.mean(),
            color="#18794e",
            linewidth=2.2,
            linestyle="--",
            label=f"degraded mean {values.mean():.3f}{suffix}",
        )
        axis.set_title(label)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "2018–2026 Stage 30 after 20 early-quality degradation replications",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout()
    output_path = OUTPUT_DIR / "late_degradation_performance_distribution.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_quality_shift() -> Path:
    report = json.loads(
        (OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    original = report["degradation_test"]["original_late_quality"]
    degraded = report["degradation_test"]["degraded_late_quality_mean"]
    labels = ["Contracts", "Coverage", "IV invalid", "25Δ distance", "Quality Q"]
    keys = [
        "listed_contracts",
        "coverage_log_width",
        "invalid_iv_share",
        "put25_nearest_strike_distance",
        "quality",
    ]
    ratios = np.array([degraded[key] / original[key] for key in keys])
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    bars = axis.bar(labels, ratios, color=["#2563a6"] * 5)
    axis.axhline(1.0, color="#b42318", linewidth=1.5, linestyle="--")
    axis.set_ylabel("Degraded / original late-period value")
    axis.set_title("The counterfactual materially changed chain quality")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, ratios):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    figure.tight_layout()
    output_path = OUTPUT_DIR / "late_degradation_quality_shift.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    _set_style()
    for path in (
        plot_early_quality_quintiles(),
        plot_degradation_distribution(),
        plot_quality_shift(),
    ):
        print(path)


if __name__ == "__main__":
    main()
