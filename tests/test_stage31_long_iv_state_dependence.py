from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "strategies" / "stage31_long_iv_state_dependence"
OUTPUT_DIR = STAGE_DIR / "outputs"
LONG_IV_XLSX = ROOT / "raw_data" / "260829_옵션내재변동성.xlsx"
K200_FUTURES_XLSX = ROOT / "raw_data" / "260829_K200선물데이터.xlsx"
EXPECTED_HASHES = {
    LONG_IV_XLSX: "967f0a612eb8ccde36e47a5ad870e1a27c58bd43685dc9234ac4f7b46e403d6c",
    K200_FUTURES_XLSX: "4d37798c636a4716b7a7d03b71549d7195067e35718635da0db2420f019d0818",
}


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


def test_source_workbooks_are_bitwise_unchanged() -> None:
    report = _report()
    assert report["raw_files_unchanged"] is True
    assert report["source_manifest_before"] == report["source_manifest_after"]
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected


def test_fixed_otm2_factor_and_prior_month_timing() -> None:
    daily = pd.read_csv(
        OUTPUT_DIR / "normalized_long_iv_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    complete = daily.dropna(
        subset=["near_put_otm2", "near_call_otm2", "wing_asym_near"]
    )
    assert np.allclose(
        complete["wing_asym_near"],
        complete["near_put_otm2"] - complete["near_call_otm2"],
    )
    assert np.allclose(
        complete["bucket_direction_near"], -complete["wing_asym_near"]
    )

    monthly = pd.read_csv(
        OUTPUT_DIR / "monthly_bucket_iv_signals.csv",
        index_col=0,
        parse_dates=["bucket_signal_date"],
    )
    target = pd.PeriodIndex(monthly.index, freq="M")
    signal_month = pd.PeriodIndex(monthly["bucket_signal_month"], freq="M")
    signal_date_month = monthly["bucket_signal_date"].dt.to_period("M")
    assert (signal_month < target).all()
    assert np.array_equal(signal_date_month, signal_month)


def test_long_iv_signal_is_not_a_replication_of_stage30_direction() -> None:
    futures = pd.read_csv(OUTPUT_DIR / "long_iv_era_diagnostics.csv")
    primary = futures.loc[
        futures["Signal"].eq("near_otm2_direction_primary")
    ].set_index("Period")
    assert primary.loc["full_1997_2026", "SpearmanIC"] < -0.10
    assert primary.loc["early_2007_2017", "SpearmanIC"] < 0.0
    assert primary.loc["late_2018_2026", "SpearmanIC"] < 0.0

    spot = pd.read_csv(
        OUTPUT_DIR / "long_iv_spot_return_robustness.csv"
    )
    spot_primary = spot.loc[
        spot["Signal"].eq("near_otm2_direction_primary")
    ].set_index("Period")
    assert spot_primary.loc["full_2000_2026", "SpearmanIC"] < -0.10

    ods = pd.read_csv(OUTPUT_DIR / "stage30_ods_era_diagnostics.csv")
    ods_score = ods.loc[
        ods["Signal"].eq("stage30_option_direction_score")
    ].set_index("Period")
    assert ods_score.loc["early_2007_2017", "SpearmanIC"] < 0.0
    assert ods_score.loc["late_2018_2026", "SpearmanIC"] > 0.0


def test_state_dependence_is_not_statistically_supported() -> None:
    report = _report()
    breaks = {row["Test"]: row for row in report["structural_break_tests"]}
    assert breaks["stage30_ods_2018_break"]["LateInteractionPValue"] > 0.10
    assert breaks["bucket_direction_near_2018_break"][
        "LateInteractionPValue"
    ] > 0.10

    interactions = pd.read_csv(
        OUTPUT_DIR / "continuous_interaction_hac.csv"
    ).set_index("Test")
    assert interactions.loc[
        "Stage30_ODS_x_MacroFragility", "InteractionPValue"
    ] > 0.10
    assert interactions.loc[
        "Stage30_ODS_x_VIX6Stress", "InteractionPValue"
    ] > 0.10


def test_reliability_formula_is_causal_and_parameter_free() -> None:
    signals = pd.read_csv(
        OUTPUT_DIR / "stage31_reliability_signals.csv", index_col=0
    )
    expected = np.square(signals["causal_calibration_z"]) / (
        1.0 + np.square(signals["causal_calibration_z"])
    )
    assert np.allclose(
        signals["causal_calibration_reliability"], expected
    )
    assert signals["causal_calibration_reliability"].between(0.0, 1.0).all()

    rolling = pd.read_csv(
        OUTPUT_DIR / "rolling_stage30_ods_beta_ic.csv", index_col=0
    )
    target = pd.PeriodIndex(rolling.index, freq="M")
    history_through = pd.PeriodIndex(rolling["history_through"], freq="M")
    assert (history_through < target).all()
    assert _report()["fixed_research_design"]["searched_parameters"] is None


def test_stage31_reliability_is_not_promoted_on_performance() -> None:
    performance = pd.read_csv(OUTPUT_DIR / "performance_comparison.csv")
    full = performance.loc[
        performance["Period"].eq("full_2007_2026")
    ].set_index("Strategy")
    stage30 = full.loc["Stage30_ODS"]
    stage31 = full.loc["Stage31_Reliability"]

    assert stage31["Sharpe"] > stage30["Sharpe"]
    assert stage31["CAGR"] < stage30["CAGR"]
    assert stage31["MDD"] < stage30["MDD"]
    assert stage31["CAGR"] < 0.10
    assert stage31["Sharpe"] < 1.0
    assert (
        _report()["decision"]
        == "retain_stage20_official_keep_stage30_research"
    )
    assert all(_report()["checks"].values())


def test_html_report_has_complete_local_assets() -> None:
    html_path = STAGE_DIR / "stage31_validation_report.html"
    html = html_path.read_text(encoding="utf-8")
    assert "Stage 31 장기 버킷 IV·상태의존성 검증" in html
    assert html.count("<section>") == 9
    assert html.count("<table>") == 4
    assert "outputs/rolling_beta_paths.png" in html
    assert (OUTPUT_DIR / "rolling_beta_paths.png").is_file()
    assert (OUTPUT_DIR / "rolling_beta_paths.png").stat().st_size > 100_000
