from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "strategies" / "stage30_data_quality_validation"
OUTPUT_DIR = VALIDATION_DIR / "outputs"
CONSOLIDATED_CHAIN = ROOT / "raw_data" / "KOSPI200OptionPrice.csv"
EXPECTED_CHAIN_SHA256 = (
    "8687e09198e4951716fa87e301db7f98567fc681dd33c86c8c24d4ec6de0d497"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report() -> dict:
    return json.loads(
        (OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )


def test_raw_option_files_are_unchanged() -> None:
    report = _report()
    assert report["raw_files_unchanged"] is True
    assert report["raw_files_unchanged_after_secondary_diagnostic"] is True
    assert report["raw_manifest_before"] == report["raw_manifest_after"]
    assert _sha256(CONSOLIDATED_CHAIN) == EXPECTED_CHAIN_SHA256


def test_validation_pipeline_and_replications_completed() -> None:
    report = _report()
    assert report["verdict"] == "data_quality_hypothesis_not_supported"
    assert all(report["checks"].values())

    fidelity = pd.read_csv(OUTPUT_DIR / "pipeline_fidelity.csv")
    assert fidelity["MaxAbsoluteDifference"].max() < 1e-12

    replications = pd.read_csv(
        OUTPUT_DIR / "late_degradation_replications.csv"
    )
    assert len(replications) == 20
    assert replications["seed"].nunique() == 20
    assert replications["surface_success_rate"].mean() > 0.98


def test_counterfactual_matches_early_quality_scale_without_changing_dte() -> None:
    quality = _report()["degradation_test"]
    original = quality["original_late_quality"]
    degraded = quality["degraded_late_quality_mean"]

    assert 50.0 < degraded["listed_contracts"] < 75.0
    assert 0.20 < degraded["coverage_log_width"] < 0.40
    assert degraded["invalid_iv_share"] > original["invalid_iv_share"]
    assert degraded["put25_nearest_strike_distance"] > original[
        "put25_nearest_strike_distance"
    ]
    assert abs(degraded["dte"] - original["dte"]) < 0.10
    assert degraded["quality"] < original["quality"]


def test_degradation_does_not_reduce_mean_late_performance_or_20d_ic() -> None:
    performance = pd.read_csv(
        OUTPUT_DIR / "late_degradation_comparison.csv"
    ).set_index("Metric")
    assert performance.loc["CAGR", "MeanDeltaDegradedMinusOriginal"] > 0.0
    assert performance.loc["Sharpe", "MeanDeltaDegradedMinusOriginal"] > 0.0
    assert performance.loc["MDD", "MeanDeltaDegradedMinusOriginal"] > 0.0

    ic = pd.read_csv(
        OUTPUT_DIR / "late_degradation_ic_comparison.csv"
    ).set_index("Horizon")
    assert abs(ic.loc["20d", "MeanDeltaDegradedMinusOriginal"]) < 0.01
    assert ic.loc["5d", "MeanDeltaDegradedMinusOriginal"] > 0.0


def test_component_diagnostic_isolated_iv_validity_without_changing_verdict() -> None:
    component = pd.read_csv(
        OUTPUT_DIR / "early_component_quality_quintile_ic.csv"
    )
    required = {
        "composite_q",
        "measurement_q_no_roll",
        "listed_contracts",
        "coverage_log_width",
        "iv_validity",
        "put25_precision",
    }
    assert set(component["Measure"]) == required

    iv = component.loc[
        component["Measure"].eq("iv_validity")
        & component["Horizon"].eq("20d")
    ].set_index("QualityQuintile")
    assert iv.loc[5, "SpearmanIC"] - iv.loc[1, "SpearmanIC"] > 0.10

    interactions = pd.read_csv(
        OUTPUT_DIR / "early_component_quality_interaction_hac.csv"
    ).set_index("Measure")
    assert interactions.loc["iv_validity", "InteractionBeta"] > 0.0
    assert interactions.loc["iv_validity", "InteractionPValue"] < 0.05
    assert _report()["verdict"] == "data_quality_hypothesis_not_supported"


def test_html_report_and_local_charts_are_complete() -> None:
    html_path = VALIDATION_DIR / "stage30_data_quality_validation_report.html"
    html = html_path.read_text(encoding="utf-8")
    assert "Stage 30 옵션 체인 데이터 품질 원인검증" in html
    assert html.count("<section") >= 10
    assert html.count("<table") >= 8
    for filename in (
        "early_quality_quintile_ic.png",
        "late_degradation_quality_shift.png",
        "late_degradation_performance_distribution.png",
    ):
        assert f"outputs/{filename}" in html
        path = OUTPUT_DIR / filename
        assert path.is_file()
        assert path.stat().st_size > 50_000


def test_humanized_report_artifacts_pass_fidelity_review() -> None:
    run_dir = ROOT / "_workspace" / "2026-08-29-001"
    fidelity = json.loads(
        (run_dir / "04_fidelity_audit.json").read_text(encoding="utf-8")
    )
    naturalness = json.loads(
        (run_dir / "05_naturalness_review.json").read_text(encoding="utf-8")
    )
    assert fidelity["meta"]["audit_verdict"] == "full_pass"
    assert fidelity["meta"]["rollback_required"] == 0
    assert naturalness["meta"]["verdict"] == "accept"
    assert naturalness["meta"]["quality_level"] == "A"
    assert naturalness["meta"]["s1_residual"] == 0
