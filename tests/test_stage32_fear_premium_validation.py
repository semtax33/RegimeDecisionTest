from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "strategies" / "stage32_fear_premium_validation"
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


def test_source_and_frozen_strategy_files_are_unchanged() -> None:
    report = _report()
    assert report["source_files_unchanged"] is True
    assert report["frozen_strategies_unchanged"] is True
    assert report["source_manifest_before"] == report["source_manifest_after"]
    assert report["frozen_strategy_manifest_before"] == report[
        "frozen_strategy_manifest_after"
    ]
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected


def test_signal_formulas_and_t_minus_one_timing_are_fixed() -> None:
    daily = pd.read_csv(
        ROOT
        / "strategies"
        / "stage31_long_iv_state_dependence"
        / "outputs"
        / "normalized_long_iv_daily.csv",
        index_col=0,
    )
    near = daily.dropna(
        subset=["near_put_otm2", "near_call_otm2", "wing_asym_near"]
    )
    assert np.allclose(
        near["wing_asym_near"],
        near["near_put_otm2"] - near["near_call_otm2"],
    )

    frame = pd.read_csv(
        OUTPUT_DIR / "monthly_fear_premium_research_frame.csv", index_col=0
    )
    term = frame.dropna(
        subset=["wing_asym_near", "wing_asym_next", "fear_term_slope"]
    )
    assert np.allclose(
        term["fear_term_slope"],
        term["wing_asym_near"] - term["wing_asym_next"],
    )
    target = pd.PeriodIndex(frame.index, freq="M")
    signal_month = pd.PeriodIndex(frame["bucket_signal_month"], freq="M")
    assert (signal_month < target).all()


def test_research_design_did_not_search_buckets_or_windows() -> None:
    report = _report()
    assert report["fixed_design"]["deciles"] == 10
    assert report["fixed_design"]["horizons_months"] == [1, 3, 6]
    assert report["fixed_design"]["searched_parameters"] is None
    assert report["checks"]["no_winsorization"] is True
    assert report["checks"]["no_parameter_bucket_or_window_search"] is True
    deciles = pd.read_csv(OUTPUT_DIR / "decile_forward_returns.csv")
    assert set(deciles["HorizonMonths"]) == {1, 3, 6}
    assert set(deciles["Decile"]) == set(range(1, 11))


def test_raw_wing_effect_is_positive_but_not_strictly_monotone() -> None:
    decile = pd.read_csv(
        OUTPUT_DIR / "decile_monotonicity_summary.csv"
    ).set_index(["Signal", "HorizonMonths"])
    univariate = pd.read_csv(
        OUTPUT_DIR / "horizon_univariate_hac.csv"
    ).set_index(["Signal", "HorizonMonths"])
    for horizon in (1, 3, 6):
        row = decile.loc[("WingAsym_Near_OTM2", horizon)]
        assert row["Q10MinusQ1MeanReturn"] > 0.0
        assert row["StrictlyMonotoneIncreasing"] in (False, np.bool_(False))
        assert univariate.loc[
            ("WingAsym_Near_OTM2", horizon), "SpearmanIC"
        ] > 0.0
    assert univariate.loc[
        ("WingAsym_Near_OTM2", 3), "HACBetaPValue"
    ] < 0.01
    assert univariate.loc[
        ("WingAsym_Near_OTM2", 6), "HACBetaPValue"
    ] < 0.05


def test_era_split_and_vix6_absorption_are_reproduced() -> None:
    era = pd.read_csv(
        OUTPUT_DIR / "control_availability_era_tests.csv"
    ).set_index(["Signal", "Sample", "HorizonMonths"])
    early = era.loc[
        ("WingAsym_Near_OTM2", "pre_frozen_controls_1997_08_2006_03", 6)
    ]
    late = era.loc[
        (
            "WingAsym_Near_OTM2",
            "frozen_controls_available_2006_04_2026_07",
            6,
        )
    ]
    assert early["StandardizedBeta"] > late["StandardizedBeta"] > 0.0

    ladder = pd.read_csv(
        OUTPUT_DIR / "sequential_control_ladder.csv"
    ).set_index(["Signal", "HorizonMonths", "ControlStep"])
    raw = ladder.loc[("WingAsym_Near_OTM2", 6, "WingOnly_CommonSample")]
    plus_vix6 = ladder.loc[("WingAsym_Near_OTM2", 6, "Plus_VIX6")]
    assert raw["HACBetaPValue"] < 0.05
    assert plus_vix6["HACBetaPValue"] > 0.10
    assert abs(plus_vix6["StandardizedSignalBeta"]) < abs(
        raw["StandardizedSignalBeta"]
    )


def test_wing_does_not_survive_all_frozen_controls() -> None:
    controlled = pd.read_csv(
        OUTPUT_DIR / "controlled_predictive_regressions.csv"
    ).set_index(["Signal", "HorizonMonths"])
    wing = controlled.loc["WingAsym_Near_OTM2"]
    assert (wing["HACBetaPValue"] > 0.10).all()


def test_term_slope_fails_the_positive_rebound_hypothesis() -> None:
    controlled = pd.read_csv(
        OUTPUT_DIR / "controlled_predictive_regressions.csv"
    ).set_index(["Signal", "HorizonMonths"])
    for horizon in (1, 3):
        row = controlled.loc[("FearTermSlope", horizon)]
        assert row["ControlledStandardizedBeta"] < 0.0
        assert row["HACBetaPValue"] < 0.10
    term_decile = pd.read_csv(
        OUTPUT_DIR / "decile_monotonicity_summary.csv"
    ).set_index(["Signal", "HorizonMonths"])
    assert (
        term_decile.loc["FearTermSlope"]["Q10MinusQ1HACPValue"] > 0.10
    ).all()


def test_crisis_exclusions_do_not_remove_the_raw_effect() -> None:
    crisis = pd.read_csv(
        OUTPUT_DIR / "crisis_exclusion_robustness.csv"
    ).set_index(["Sample", "HorizonMonths"])
    excluded = crisis.loc["ExcludeAll_1998_2008_2020"]
    assert (excluded["SpearmanIC"] > 0.0).all()
    assert (excluded["Q10MinusQ1MeanReturn"] > 0.0).all()


def test_overlay_gate_fails_and_stage20_remains_frozen() -> None:
    report = _report()
    assert report["overlay_eligible"] is False
    assert report["decision"] == (
        "fear_premium_fails_overlay_gate_keep_stage20_frozen"
    )
    assert report["gate_results"] == {
        "one_month_decile_monotonicity": False,
        "positive_ic_at_1_3_6m_and_two_significant_horizons": True,
        "one_month_survives_all_controls": False,
        "one_month_survives_all_crisis_exclusions": True,
    }
    assert not (OUTPUT_DIR / "stage20_fear_premium_overlay.csv").exists()
    assert all(report["checks"].values())


def test_html_report_assets_and_humanized_copy_are_complete() -> None:
    html_path = STAGE_DIR / "stage32_fear_premium_validation_report.html"
    html = html_path.read_text(encoding="utf-8")
    assert "Stage 32" in html and "Fear Premium 검증" in html
    assert html.count("<section") >= 10
    assert html.count("<table") >= 8
    for filename in (
        "decile_forward_return_curves.png",
        "control_and_crisis_robustness.png",
    ):
        assert f"outputs/{filename}" in html
        path = OUTPUT_DIR / filename
        assert path.is_file()
        assert path.stat().st_size > 80_000

    run_dir = ROOT / "_workspace" / "2026-08-29-002"
    fidelity = json.loads(
        (run_dir / "04_fidelity_audit.json").read_text(encoding="utf-8")
    )
    naturalness = json.loads(
        (run_dir / "05_naturalness_review.json").read_text(encoding="utf-8")
    )
    assert fidelity["meta"]["audit_verdict"] == "full_pass"
    assert naturalness["meta"]["verdict"] == "accept"
    assert naturalness["meta"]["quality_level"] == "A"
