from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "strategies"


def _report(stage: str) -> dict:
    path = STRATEGIES / stage / "outputs" / "validation_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _performance(stage: str) -> pd.DataFrame:
    path = STRATEGIES / stage / "outputs" / "performance_comparison.csv"
    return pd.read_csv(path).set_index(["Strategy", "Period"])


def test_rejected_alpha_branches_preserve_stage36_and_are_causal() -> None:
    stage37 = _report("stage37_bond_curve_alpha")
    stage38 = _report("stage38_gold_state_alpha")
    stage39 = _report("stage39_calibrated_gold_alpha")

    assert stage37["decision"] == (
        "retain_stage36_and_treat_bond_curve_alpha_as_research_only"
    )
    assert stage38["decision"] == (
        "retain_stage36_and_treat_gold_state_as_research_only"
    )
    assert stage39["decision"] == "retain_stage36_and_close_gold_alpha_branch"
    assert {stage37["promoted_strategy"], stage38["promoted_strategy"],
            stage39["promoted_strategy"]} == {"Stage36_Frozen"}

    for report in (stage37, stage38, stage39):
        checks = report["checks"]
        assert checks["no_change_reproduces_stage36_returns"]
        assert checks["no_change_reproduces_stage36_weights"]
        assert checks["no_leverage_long_only_sum_to_one"]
        assert checks["all_candidate_solvers_feasible"]

    assert stage37["checks"]["bond_signal_precedes_target"]
    assert stage37["checks"]["no_2y_backfill"]
    assert stage38["checks"]["signal_month_precedes_target"]
    assert stage38["checks"]["release_dates_not_after_signal_month_end"]
    assert stage39["checks"]["signal_month_precedes_target"]
    assert stage39["checks"]["minimum_60_calibration_observations"]


def test_bond_curve_alpha_is_not_promoted_after_costs() -> None:
    perf = _performance("stage37_bond_curve_alpha")
    base = perf.loc[("Stage36_Frozen", "full_2007_2026")]
    candidate = perf.loc[("Stage37_BondCurveAlpha", "full_2007_2026")]

    assert candidate["CAGR"] < base["CAGR"]
    assert candidate["Sharpe"] < base["Sharpe"]
    assert candidate["MDD"] < base["MDD"]
    assert candidate["TotalCost"] > base["TotalCost"]


def test_gold_state_point_estimate_is_reported_without_promoting_it() -> None:
    perf = _performance("stage38_gold_state_alpha")
    full_base = perf.loc[("Stage36_Frozen", "full_2007_2026")]
    full_fx = perf.loc[("Stage38_FXState", "full_2007_2026")]
    locked_base = perf.loc[("Stage36_Frozen", "locked_2018_2026")]
    locked_fx = perf.loc[("Stage38_FXState", "locked_2018_2026")]

    assert full_fx["CAGR"] > full_base["CAGR"]
    assert full_fx["Sharpe"] > full_base["Sharpe"]
    assert full_fx["MDD"] > full_base["MDD"]
    assert locked_fx["CAGR"] < locked_base["CAGR"]

    bootstrap = pd.read_csv(
        STRATEGIES
        / "stage38_gold_state_alpha"
        / "outputs"
        / "paired_block_bootstrap_vs_stage36.csv"
    )
    full_cagr = bootstrap.loc[
        bootstrap["Candidate"].eq("Stage38_FXState")
        & bootstrap["Period"].eq("full_2007_2026")
        & bootstrap["Metric"].eq("delta_CAGR"),
        "ProbabilityPositive",
    ].iloc[0]
    assert full_cagr < 0.60


def test_causal_gold_calibration_does_not_restore_cagr() -> None:
    perf = _performance("stage39_calibrated_gold_alpha")
    for period in (
        "full_2007_2026",
        "common_2010_2026",
        "locked_2018_2026",
    ):
        base = perf.loc[("Stage36_Frozen", period)]
        candidate = perf.loc[("Stage39_CalibratedGoldAlpha", period)]
        assert candidate["CAGR"] < base["CAGR"]

    signals = pd.read_csv(
        STRATEGIES
        / "stage39_calibrated_gold_alpha"
        / "outputs"
        / "monthly_calibrated_gold_signals.csv"
    )
    active = signals["calibrated_gold_alpha_active"].astype(bool)
    assert signals.loc[active, "gold_calibration_observations"].min() >= 60
    assert (signals.loc[active, "gold_calibration_slope"] >= 0.0).all()
    assert np.allclose(
        signals.loc[~active, "calibrated_gold_mu_adjustment"], 0.0
    )
