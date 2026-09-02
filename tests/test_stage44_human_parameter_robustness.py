from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.stage44_human_parameter_robustness import (
    human_parameter_robustness as strategy,
)


def test_stage44_is_isolated_and_predeclares_the_four_clean_ablations() -> None:
    assert strategy.OUTPUT_DIR.parent.name == "stage44_human_parameter_robustness"
    assert strategy.ORIGINAL_VOL_CAP == pytest.approx(0.13)
    assert strategy.ORIGINAL_CDAR_LIMIT == pytest.approx(0.16)
    assert strategy.ORIGINAL_CDAR_CONFIDENCE == pytest.approx(0.90)
    assert {item.name for item in strategy.CORE_SPECIFICATIONS} == {
        strategy.CORE_BOTH,
        strategy.CORE_CDAR_ONLY,
        strategy.CORE_VOL_ONLY,
        strategy.CORE_NO_GUARDS,
    }
    assert strategy.VOL_SENSITIVITY == (0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
    assert strategy.CDAR_LIMIT_SENSITIVITY == (0.12, 0.14, 0.16, 0.18, 0.20)
    assert strategy.CDAR_CONFIDENCE_SENSITIVITY == (0.80, 0.85, 0.90, 0.95)


def test_objective_is_parameter_free_log_growth_and_semivariance_is_diagnostic() -> None:
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mu = np.array([0.01, 0.004, 0.006, 0.002])
    covariance = np.diag([0.0020, 0.0005, 0.0010, 0.0030])
    history = np.array(
        [
            [0.02, 0.00, 0.01, -0.01],
            [-0.01, 0.005, 0.00, 0.02],
            [0.01, -0.02, 0.01, -0.03],
            [-0.02, 0.01, -0.01, 0.01],
        ]
    )
    pretrade = weights.copy()
    values = strategy.portfolio_values(
        weights, mu, covariance, history, pretrade, 0.90
    )
    expected_cost = strategy.stage36.stage35.expected_transaction_cost(
        weights, pretrade
    )
    expected = float(weights @ mu) - 0.5 * float(
        weights @ covariance @ weights
    ) - expected_cost
    assert values["expected_monthly_log_growth_net"] == pytest.approx(expected)
    assert values["downside_semivariance_coefficient_in_objective"] == 0.0
    assert values["downside_semivariance_diagnostic"] > 0.0

    source = Path(strategy.__file__).read_text(encoding="utf-8")
    assert "log_growth = monthly_return - 0.5 * monthly_variance - transaction_cost" in source
    assert '"objective": "parameter_free_expected_log_growth"' in source


def test_core_ablation_full_period_regression() -> None:
    performance = pd.read_csv(strategy.OUTPUT_DIR / "core_ablation_performance.csv")
    full = performance.loc[performance["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )
    assert set(full.index) == {
        strategy.BASELINE_NAME,
        strategy.CORE_BOTH,
        strategy.CORE_CDAR_ONLY,
        strategy.CORE_VOL_ONLY,
        strategy.CORE_NO_GUARDS,
    }
    expected = {
        strategy.BASELINE_NAME: (0.10499388753644401, 1.1049112900976705, -0.12407087222855073),
        strategy.CORE_BOTH: (0.10657858747686322, 1.1006709893545301, -0.12326364165170356),
        strategy.CORE_CDAR_ONLY: (0.137081314550765, 0.8249675972350937, -0.2524790884161743),
        strategy.CORE_VOL_ONLY: (0.10659685467364177, 1.1008348538090529, -0.12326364171932025),
        strategy.CORE_NO_GUARDS: (0.1532807743967406, 0.8091237483480235, -0.2683561581574345),
    }
    for name, (cagr, sharpe, mdd) in expected.items():
        assert full.loc[name, "CAGR"] == pytest.approx(cagr)
        assert full.loc[name, "Sharpe"] == pytest.approx(sharpe)
        assert full.loc[name, "MDD"] == pytest.approx(mdd)

    assert abs(full.loc[strategy.CORE_BOTH, "CAGR"] - full.loc[strategy.BASELINE_NAME, "CAGR"]) < 0.002
    assert abs(full.loc[strategy.CORE_BOTH, "Sharpe"] - full.loc[strategy.BASELINE_NAME, "Sharpe"]) < 0.01
    assert abs(full.loc[strategy.CORE_BOTH, "MDD"] - full.loc[strategy.BASELINE_NAME, "MDD"]) < 0.01
    assert full.loc[strategy.CORE_CDAR_ONLY, "MDD"] < -0.25
    assert full.loc[strategy.CORE_NO_GUARDS, "Sharpe"] < 0.82


def test_saved_core_paths_are_causal_feasible_and_cover_232_months() -> None:
    specifications = {item.name: item for item in strategy.CORE_SPECIFICATIONS}
    for name, specification in specifications.items():
        path = pd.read_csv(strategy.OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        assert len(path) == 232, name
        assert path["month"].iloc[0] == "2007-04", name
        assert path["month"].iloc[-1] == "2026-07", name
        assert path["solver_success"].all(), name
        assert not path["used_fallback"].any(), name
        assert path["history_end_month"].lt(path["month"]).all(), name
        weights = path[[f"w_{asset}" for asset in strategy.ASSETS]]
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-8), name
        assert weights.min().min() >= -1e-9, name
        assert weights.max().max() <= 1.0 + 1e-9, name
        assert path["downside_semivariance_coefficient_in_objective"].eq(0.0).all()
        if specification.volatility_cap is not None:
            assert path["volatility_slack"].min() >= -1e-7, name
        if specification.cdar_limit is not None:
            assert path["cdar_slack"].min() >= -1e-7, name


def test_forecast_reproduction_and_stage36_freeze_are_machine_precision() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["stage"] == 44
    assert report["stage36_frozen_files_unchanged"] is True
    audit = report["stage36_forecast_reproduction_audit"]
    assert audit["months"] == 232
    assert audit["first_month"] == "2007-04"
    assert audit["last_month"] == "2026-07"
    assert audit["maximum_absolute_expected_return_error"] < 1e-15
    assert audit["maximum_absolute_expected_variance_error"] < 1e-15
    assert report["all_paths_causal"] is True
    assert report["all_solvers_successful"] is True
    assert report["total_fallbacks"] == 0
    assert report["anti_overfit"]["best_sensitivity_path_promoted"] is False
    assert report["anti_overfit"]["threshold_results_used_for_selection"] is False
    assert report["anti_overfit"]["input_edge_changed"] is False
    assert report["anti_overfit"]["mu_or_sigma_model_changed"] is False


def test_weight_paths_show_semivariance_and_cdar_are_not_performance_drivers() -> None:
    comparison = pd.read_csv(
        strategy.OUTPUT_DIR / "weight_path_comparison.csv"
    ).set_index("Candidate")
    both = comparison.loc[strategy.CORE_BOTH]
    vol_only = comparison.loc[strategy.CORE_VOL_ONLY]
    no_guards = comparison.loc[strategy.CORE_NO_GUARDS]
    assert both["ReturnCorrelation"] == pytest.approx(0.9972001741173824)
    assert both["MeanAbsoluteWeightDifference"] == pytest.approx(
        0.01170560062855153
    )
    assert vol_only["ReturnCorrelation"] > 0.999999
    assert vol_only["MeanAbsoluteWeightDifference"] < 0.0001
    assert no_guards["MeanAbsoluteWeightDifference"] > 0.18

    allocations = pd.read_csv(
        strategy.OUTPUT_DIR / "allocation_summary.csv"
    ).set_index("Strategy")
    assert allocations.loc[strategy.CORE_BOTH, "MonthsAbove90Percent"] == 2
    assert allocations.loc[strategy.CORE_NO_GUARDS, "MonthsAbove90Percent"] == 60
    assert allocations.loc[strategy.CORE_NO_GUARDS, "AverageWeight_BOND"] < 0.10


def test_volatility_sensitivity_is_a_smooth_tradeoff_not_a_13pct_peak() -> None:
    performance = pd.read_csv(strategy.OUTPUT_DIR / "sensitivity_performance.csv")
    full = performance.loc[
        performance["Period"].eq("full_2007_2026")
        & performance["Family"].eq("volatility_cap_sensitivity")
    ].sort_values("DisplayedValue")
    assert full["DisplayedValue"].tolist() == list(strategy.VOL_SENSITIVITY)
    assert np.all(np.diff(full["CAGR"]) > 0.0)
    assert np.all(np.diff(full["Sharpe"]) < 0.0)
    assert np.all(np.diff(full["MDD"]) < 0.0)
    original = full.loc[full["DisplayedValue"].eq(0.13)].iloc[0]
    assert original["CAGR"] < full["CAGR"].max()
    assert original["Sharpe"] < full["Sharpe"].max()
    assert original["MDD"] < full["MDD"].max()


def test_cdar_limit_and_confidence_sensitivity_are_narrow_and_not_selected() -> None:
    summary = pd.read_csv(strategy.OUTPUT_DIR / "sensitivity_summary.csv").set_index(
        "Family"
    )
    cdar_limit = summary.loc["cdar_limit_sensitivity"]
    confidence = summary.loc["cdar_confidence_sensitivity"]
    assert cdar_limit["CAGRRange"] < 0.002
    assert cdar_limit["SharpeRange"] < 0.02
    assert cdar_limit["MDDRange"] < 0.007
    assert confidence["CAGRRange"] < 0.002
    assert confidence["SharpeRange"] < 0.021
    assert confidence["MDDRange"] < 0.002
    assert (summary["SelectionUse"] == "report-only; no best path adopted").all()

    output = strategy.OUTPUT_DIR
    aliases = (
        ("stage44_volcap_13pct_monthly.csv", "stage44_pf_bothguards_monthly.csv"),
        ("stage44_cdarlimit_16pct_monthly.csv", "stage44_pf_bothguards_monthly.csv"),
        ("stage44_cdarconfidence_90pct_monthly.csv", "stage44_pf_bothguards_monthly.csv"),
    )
    for alias_name, reference_name in aliases:
        alias = pd.read_csv(output / alias_name)
        reference = pd.read_csv(output / reference_name)
        assert np.allclose(alias["return"], reference["return"])
        assert np.allclose(alias[strategy.WEIGHT_COLUMNS], reference[strategy.WEIGHT_COLUMNS])


def test_bootstrap_does_not_claim_semivariance_removal_is_a_new_alpha() -> None:
    bootstrap = pd.read_csv(
        strategy.OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv"
    )
    rows = bootstrap.loc[
        bootstrap["Candidate"].eq(strategy.CORE_BOTH)
        & bootstrap["Period"].eq("full_2007_2026")
    ].set_index("Metric")
    assert rows.loc["delta_CAGR", "ProbabilityPositive"] == pytest.approx(0.7665)
    assert rows.loc["delta_Sharpe", "ProbabilityPositive"] == pytest.approx(0.406)
    assert rows.loc["delta_MDD", "ProbabilityPositive"] == pytest.approx(0.3455)
    assert rows.loc["delta_CAGR", "P05"] < 0.0 < rows.loc["delta_CAGR", "P95"]


def test_readme_and_outputs_make_the_honest_boundary_explicit() -> None:
    for filename in (
        "core_ablation_performance.csv",
        "sensitivity_performance.csv",
        "sensitivity_summary.csv",
        "constraint_binding_summary.csv",
        "weight_path_comparison.csv",
        "allocation_summary.csv",
        "paired_block_bootstrap_vs_stage36.csv",
        "human_parameter_inventory.csv",
        "validation_report.json",
    ):
        assert (strategy.OUTPUT_DIR / filename).is_file(), filename

    readme = strategy.OUTPUT_DIR.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert readme.stat().st_size > 13_000
    for phrase in (
        "Human Parameter Robustness",
        "목적함수 정화",
        "핵심 A/B/C/D 설계",
        "downside semivariance 계수에 의존하지 않았다",
        "volatility cap은 실질적인 governance layer",
        "Volatility cap 10~15% 민감도",
        "CDaR limit -12~-20% 민감도",
        "CDaR confidence 80~95% 민감도",
        "가능한 주장과 불가능한 주장",
        "대체 threshold 중 성과가 가장 좋은 경로를 채택하지 않았다",
    ):
        assert phrase in text
