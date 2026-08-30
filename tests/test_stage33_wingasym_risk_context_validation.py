from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = (
    ROOT / "strategies" / "stage33_wingasym_risk_context_validation"
)
OUTPUT_DIR = STAGE_DIR / "outputs"
EXPECTED_RAW_HASHES = {
    ROOT / "raw_data" / "260829_옵션내재변동성.xlsx": (
        "967f0a612eb8ccde36e47a5ad870e1a27c58bd43685dc9234ac4f7b46e403d6c"
    ),
    ROOT / "raw_data" / "260829_K200선물데이터.xlsx": (
        "4d37798c636a4716b7a7d03b71549d7195067e35718635da0db2420f019d0818"
    ),
    ROOT / "raw_data" / "KOSPI200OptionPrice.csv": (
        "8687e09198e4951716fa87e301db7f98567fc681dd33c86c8c24d4ec6de0d497"
    ),
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


def test_raw_and_frozen_inputs_are_unchanged() -> None:
    report = _report()
    assert report["checks"]["source_files_unchanged"] is True
    assert report["checks"][
        "stage20_to_stage32_code_and_results_unchanged"
    ] is True
    assert report["source_manifest_before"] == report["source_manifest_after"]
    assert report["frozen_manifest_before"] == report["frozen_manifest_after"]
    for path, expected_hash in EXPECTED_RAW_HASHES.items():
        assert _sha256(path) == expected_hash


def test_stage33_uses_fixed_causal_timing_and_no_signal_search() -> None:
    report = _report()
    frame = pd.read_csv(
        OUTPUT_DIR / "monthly_stage33_research_frame.csv", index_col=0
    )
    target = pd.PeriodIndex(frame.index, freq="M")
    signal = pd.PeriodIndex(frame["bucket_signal_month"], freq="M")
    assert len(frame) == 232
    assert str(target.min()) == "2007-04"
    assert str(target.max()) == "2026-07"
    assert (signal < target).all()
    assert report["fixed_design"]["left_tail_quantile"] == 0.05
    assert report["fixed_design"]["minimum_causal_history_months"] == 60
    assert report["fixed_design"]["searched_parameters"] is None
    assert report["checks"][
        "no_threshold_window_bucket_or_sign_search"
    ] is True
    assert report["checks"]["no_residual_wingasym"] is True
    assert report["checks"]["fear_term_slope_retired"] is True


def test_causal_state_thresholds_are_prior_history_estimates() -> None:
    frame = pd.read_csv(
        OUTPUT_DIR / "monthly_stage33_research_frame.csv", index_col=0
    )
    state_rows = frame.dropna(
        subset=["causal_2x2_state", "causal_vix6_median", "causal_wing_median"]
    )
    assert len(state_rows) >= 200
    for _, row in state_rows.iloc[[0, -1]].iterrows():
        if "VIX6 High" in row["causal_2x2_state"]:
            assert row["vix6_stress_score"] >= row["causal_vix6_median"]
        else:
            assert row["vix6_stress_score"] < row["causal_vix6_median"]
        if "Wing High" in row["causal_2x2_state"]:
            assert row["wing_asym_near"] >= row["causal_wing_median"]
        else:
            assert row["wing_asym_near"] < row["causal_wing_median"]


def test_wingasym_adds_no_future_risk_information_after_controls() -> None:
    risk = pd.read_csv(
        OUTPUT_DIR / "future_risk_incremental_regressions.csv"
    )
    full = risk.loc[risk["Model"].eq("FullControls")].set_index("Target")
    assert set(full.index) == {
        "future_realized_vol_1m",
        "future_max_drawdown_1m",
        "future_max_drawdown_3m",
        "future_left_tail_1m",
    }
    assert (full["WingHACPValue"] > 0.80).all()
    assert int((full["WingStandardizedBeta"] > 0.0).sum()) < 3
    assert int((full["IncrementalFitScore"] > 0.0).sum()) <= 1
    report = _report()
    assert report["direct_risk_sensor_pass"] is False


def test_wingasym_worsens_expanding_tail_scores_vs_vix6_only() -> None:
    scores = pd.read_csv(
        OUTPUT_DIR / "left_tail_expanding_oos_scores.csv"
    ).set_index("Model")
    assert int(scores.loc["VIX6Only", "OOSEvents"]) == 5
    assert scores.loc["VIX6PlusWing", "AUC"] < scores.loc["VIX6Only", "AUC"]
    assert (
        scores.loc["VIX6PlusWing", "BrierScore"]
        > scores.loc["VIX6Only", "BrierScore"]
    )
    assert (
        scores.loc["VIX6PlusWing", "LogLoss"]
        > scores.loc["VIX6Only", "LogLoss"]
    )


def test_high_high_descriptive_effect_does_not_survive_interaction_test() -> None:
    states = pd.read_csv(
        OUTPUT_DIR / "causal_2x2_state_diagnostic.csv"
    ).set_index("State")
    high_high = states.loc["VIX6 High / Wing High"]
    assert int(high_high["Months"]) >= 15
    assert high_high["MeanFutureK200Return3M"] > 0.0
    assert high_high["MeanFutureK200Return6M"] > 0.0
    assert high_high["MeanStage20DefenseOpportunityCost3M"] > 0.0
    assert high_high["MeanStage20DefenseOpportunityCost6M"] > 0.0

    interactions = pd.read_csv(
        OUTPUT_DIR / "vix6_wing_context_interactions.csv"
    )
    returns = interactions.loc[
        interactions["TestFamily"].eq("K200ForwardReturn")
        & interactions["Model"].eq("FullControlsContext")
        & interactions["HorizonMonths"].isin([3, 6])
    ]
    assert (returns["InteractionStandardizedBeta"] < 0.0).all()
    assert (returns["InteractionHACPValue"] < 0.10).all()
    assert _report()["vix6_context_reentry_pass"] is False


def test_false_positive_interaction_is_not_significant() -> None:
    interactions = pd.read_csv(
        OUTPUT_DIR / "vix6_wing_context_interactions.csv"
    )
    row = interactions.loc[
        interactions["TestFamily"].eq("Stage20FalsePositive")
        & interactions["Model"].eq("FullControlsContext")
    ].iloc[0]
    assert int(row["Observations"]) == 204
    assert int(row["Events"]) == 106
    assert row["InteractionHACPValue"] > 0.10


def test_branch_is_closed_and_stage20_remains_frozen() -> None:
    report = _report()
    assert report["decision"] == (
        "close_wingasym_branch_move_to_independent_information_source"
    )
    assert report["direct_risk_sensor_pass"] is False
    assert report["vix6_context_reentry_pass"] is False
    assert report["checks"][
        "no_strategy_weights_or_expected_returns_changed"
    ] is True
    assert not (OUTPUT_DIR / "stage20_wingasym_overlay.csv").exists()


def test_html_and_chart_artifacts_are_complete() -> None:
    html = (
        STAGE_DIR / "stage33_wingasym_risk_context_report.html"
    ).read_text(encoding="utf-8")
    assert "Stage 33" in html
    assert "WingAsym 연구 가지 종료" in html
    assert html.count("<section") >= 10
    assert html.count("<table") >= 5
    for filename in (
        "incremental_risk_coefficients.png",
        "causal_2x2_context.png",
    ):
        assert f"outputs/{filename}" in html
        path = OUTPUT_DIR / filename
        assert path.is_file()
        assert path.stat().st_size > 40_000
