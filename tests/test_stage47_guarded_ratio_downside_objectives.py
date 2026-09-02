from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage47_guarded_ratio_downside_objectives import (
    guarded_ratio_downside_slsqp as strategy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "strategies"
    / "stage47_guarded_ratio_downside_objectives"
    / "outputs"
)


def _report() -> dict:
    return json.loads((OUTPUT / "validation_report.json").read_text("utf-8"))


def _performance() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "performance_comparison.csv").set_index(
        ["Strategy", "Period"]
    )


def test_objective_statistics_match_declared_formulas() -> None:
    weights = np.repeat(0.25, 4)
    expected_return = np.array([0.01, 0.02, 0.03, 0.04])
    covariance = np.diag([0.01, 0.02, 0.03, 0.04])
    history = np.array(
        [
            [-0.04, -0.02, 0.01, 0.03],
            [0.02, 0.01, -0.01, 0.00],
            [-0.03, -0.01, -0.02, -0.02],
        ]
    )
    cash_history = np.array([0.001, 0.001, 0.001])
    current_cash = 0.001
    statistics = strategy.objective_statistics(
        weights,
        expected_return,
        covariance,
        history,
        cash_history,
        current_cash,
        weights,
    )

    transaction_cost = strategy.stage36.stage35.expected_transaction_cost(
        weights, weights
    )
    expected_excess = (
        float(weights @ expected_return) - current_cash - transaction_cost
    )
    variance = float(weights @ covariance @ weights)
    historical_excess = history @ weights - cash_history
    downside = float(np.mean(np.minimum(historical_excess, 0.0) ** 2))
    assert np.isclose(
        statistics["estimated_transaction_cost"], transaction_cost
    )
    assert transaction_cost < 1e-8
    assert np.isclose(
        statistics["expected_excess_return_after_cost"], expected_excess
    )
    assert np.isclose(statistics["expected_monthly_variance"], variance)
    assert np.isclose(
        statistics["historical_downside_semivariance"], downside
    )
    assert np.isclose(
        statistics["ex_ante_sharpe_objective"],
        expected_excess / np.sqrt(variance),
    )
    assert np.isclose(
        statistics["downside_penalty_objective"],
        expected_excess - downside,
    )
    assert np.isclose(
        statistics["direct_sortino_objective"],
        expected_excess / np.sqrt(downside),
    )


def test_exactly_five_predeclared_paths_and_no_candidate_is_promoted() -> None:
    report = _report()
    performance = _performance()
    expected = {
        strategy.BASELINE_NAME,
        strategy.B_NAME,
        strategy.C_NAME,
        strategy.D_NAME,
        strategy.E_NAME,
    }
    assert set(performance.index.get_level_values("Strategy")) == expected
    assert report["checks"]["exactly_five_predeclared_performance_paths"]
    assert report["decision"] == "retain_stage36_no_objective_passed_all_gates"
    assert report["promoted_strategy"] == strategy.BASELINE_NAME
    assert all(
        result["pass"] is False
        for result in report["gate_results"].values()
    )


def test_candidate_outcomes_are_reported_without_cherry_picking() -> None:
    performance = _performance()
    for period in (
        "full_2007_2026",
        "common_2010_2026",
        "locked_2018_2026",
    ):
        baseline = performance.loc[(strategy.BASELINE_NAME, period)]
        downside = performance.loc[(strategy.D_NAME, period)]
        sharpe = performance.loc[(strategy.C_NAME, period)]
        sortino = performance.loc[(strategy.E_NAME, period)]
        assert downside["CAGR"] > baseline["CAGR"]
        assert downside["Sharpe"] < baseline["Sharpe"]
        assert downside["MDD"] < baseline["MDD"]
        assert sortino["Sortino"] < baseline["Sortino"]
        assert sharpe["CAGR"] < baseline["CAGR"]
        assert sharpe["MDD"] > baseline["MDD"]
    assert (
        performance.loc[(strategy.C_NAME, "full_2007_2026"), "Sharpe"]
        > performance.loc[
            (strategy.BASELINE_NAME, "full_2007_2026"), "Sharpe"
        ]
    )


def test_bootstrap_contains_every_candidate_period_and_metric() -> None:
    bootstrap = pd.read_csv(
        OUTPUT / "paired_block_bootstrap_vs_stage36.csv"
    )
    assert len(bootstrap) == 4 * 3 * 3
    assert set(bootstrap["Candidate"]) == {
        strategy.B_NAME,
        strategy.C_NAME,
        strategy.D_NAME,
        strategy.E_NAME,
    }
    assert set(bootstrap["Period"]) == {
        "full_2007_2026",
        "common_2010_2026",
        "locked_2018_2026",
    }
    assert set(bootstrap["Metric"]) == {
        "delta_CAGR",
        "delta_Sharpe",
        "delta_MDD",
    }
    assert bootstrap["Replications"].eq(2000).all()
    assert bootstrap["BlockMonths"].eq(12).all()
    assert bootstrap["ProbabilityPositive"].between(0.0, 1.0).all()


def test_guarded_paths_are_causal_long_only_fully_invested_and_tilt_limited() -> None:
    report = _report()
    filenames = {
        strategy.B_NAME: "stage47_b_guarded_stage36objective_lw_monthly.csv",
        strategy.C_NAME: "stage47_c_guarded_exante_sharpe_lw_monthly.csv",
        strategy.D_NAME: "stage47_d_guarded_downside_penalty_lw_monthly.csv",
        strategy.E_NAME: "stage47_e_guarded_direct_sortino_lw_monthly.csv",
    }
    for name, filename in filenames.items():
        path = pd.read_csv(OUTPUT / filename, index_col="month")
        weights = path[strategy.WEIGHT_COLUMNS]
        assert (weights.to_numpy() >= -1e-10).all()
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        assert path["applied_objective_tilt"].max() <= 0.05 + 1e-12
        assert path["drawdown_multiplier"].between(0.0, 1.0).all()
        assert report["causality_audit"][name][
            "all_covariance_dates_before_target"
        ]
        assert report["causality_audit"][name][
            "all_covariance_windows_have_252_rows"
        ]

    assert report["checks"]["all_shadow_solvers_feasible"]
    assert report["checks"]["all_deployments_long_only_fully_invested"]
    assert report["checks"][
        "stage36_stage45_stage46_frozen_files_unchanged"
    ]


def test_output_set_is_complete_and_fixed() -> None:
    expected = {
        "stage47_b_guarded_stage36objective_lw_monthly.csv",
        "stage47_b_stage36objective_lw_shadow.csv",
        "stage47_c_guarded_exante_sharpe_lw_monthly.csv",
        "stage47_c_exante_sharpe_lw_shadow.csv",
        "stage47_d_guarded_downside_penalty_lw_monthly.csv",
        "stage47_d_downside_penalty_lw_shadow.csv",
        "stage47_e_guarded_direct_sortino_lw_monthly.csv",
        "stage47_e_direct_sortino_lw_shadow.csv",
        "performance_comparison.csv",
        "paired_block_bootstrap_vs_stage36.csv",
        "validation_report.json",
    }
    assert expected == {path.name for path in OUTPUT.iterdir() if path.is_file()}
