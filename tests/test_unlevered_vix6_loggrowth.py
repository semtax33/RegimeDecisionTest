from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.stage12_unlevered_vix6_loggrowth import (
    unlevered_vix6_loggrowth as strategy,
)


def test_expected_log_growth_formula() -> None:
    expected_return = np.array([0.004, 0.009])
    covariance = np.array([[0.0002, 0.0001], [0.0001, 0.0008]])
    alpha = 0.40
    weights = np.array([1 - alpha, alpha])
    metrics = strategy.sleeve_metrics(
        alpha, expected_return, covariance
    )
    monthly_return = float(weights @ expected_return)
    monthly_variance = float(weights @ covariance @ weights)
    assert metrics["expected_log_growth"] == pytest.approx(
        12 * (monthly_return - 0.5 * monthly_variance)
    )


def test_optimizer_contains_required_constraints() -> None:
    source = inspect.getsource(strategy.optimize_growth_weight)
    assert '"expected_log_growth"' in source
    assert "FORECAST_SHARPE_FLOOR" in source
    assert "HISTORICAL_CDAR_FLOOR" in source
    assert 'method="SLSQP"' in source
    assert strategy.FORECAST_SHARPE_FLOOR == 1.05
    assert strategy.HISTORICAL_CDAR_FLOOR == -0.13


def test_daily_growth_sleeve_is_causal_and_unlevered() -> None:
    daily = pd.read_csv(
        strategy.OUTPUT_DIR / "unlevered_vix6_router_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    daily["signal_date"] = pd.to_datetime(daily["signal_date"])
    assert (daily["signal_date"] < daily.index).all()
    assert daily["gross_exposure"].max() <= 1.0 + 1e-12
    assert daily["gross_exposure"].min() >= 1.0 - 1e-12
    assert daily["source_gross_exposure"].max() > 1.50
    columns = [f"w_{asset}" for asset in ["KODEX200", "BOND", "GLD", "USO"]]
    assert np.allclose(daily[columns].sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "filename",
    ["selected_monthly.csv", "meta_loggrowth_put_candidate.csv"],
)
def test_selected_path_is_causal_and_uses_no_leverage(
    filename: str,
) -> None:
    selected = pd.read_csv(
        strategy.OUTPUT_DIR / filename, index_col=0
    )
    index = pd.PeriodIndex(selected.index, freq="M")
    signal = pd.PeriodIndex(selected["signal_month"], freq="M")
    assert (signal < index).all()
    allocation = [
        "w_stage10_core",
        "w_unlevered_vix6_growth",
        "w_option",
    ]
    assert np.allclose(selected[allocation].sum(axis=1), 1.0)
    assert selected["gross_exposure"].max() <= 1.0 + 1e-12
    assert selected[allocation].min().min() >= -1e-12
    assert selected[allocation].max().max() <= 1.0 + 1e-12


def test_report_confirms_target_and_solver_success() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["selected_strategy"] == (
        "UnleveredVIX6LogGrowth_NoOption"
    )
    assert not report["allocation"]["leverage_allowed"]
    assert report["allocation"][
        "maximum_observed_selected_gross_exposure"
    ] <= 1.0 + 1e-12
    assert all(report["target_checks"].values())
    audit = report["causality_and_solver"]
    assert audit["daily_router_signal_precedes_action"]
    assert audit["monthly_risk_signal_precedes_target"]
    assert audit["optimized_months"] == 220
    assert audit["solver_successes"] == 220
    assert audit["grid_fallbacks"] == 0
    assert audit["minimum_forecast_sharpe_slack"] >= -1e-10
    assert audit["minimum_historical_cdar_slack"] >= -1e-10


def test_physical_put_is_evaluated_but_not_promoted() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert not report["option_selected"]
    assert not report["option_gate"]["promoted"]
    for period in report["option_gate"]["periods"].values():
        assert not period["passes_all_three"]
        assert period["deltas"]["CAGR"] < 0
        assert period["deltas"]["Sharpe"] < 0


def test_selected_metrics_are_reproduced() -> None:
    comparison = pd.read_csv(
        strategy.OUTPUT_DIR / "performance_comparison.csv"
    )
    rows = comparison.loc[
        comparison["Strategy"].eq(
            "UnleveredVIX6LogGrowth_NoOption"
        )
    ].set_index("Period")
    full = rows.loc["full_2007_2026"]
    locked = rows.loc["locked_2018_2026"]
    assert int(full["Months"]) == 232
    assert full["CAGR"] == pytest.approx(
        0.10196019751706653, abs=1e-9
    )
    assert full["Sharpe"] == pytest.approx(
        1.3050287975008488, abs=1e-9
    )
    assert full["MDD"] == pytest.approx(
        -0.11002752357755297, abs=1e-9
    )
    assert int(locked["Months"]) == 103
    assert locked["CAGR"] == pytest.approx(
        0.1278499610647736, abs=1e-9
    )
    assert locked["Sharpe"] == pytest.approx(
        1.64850448891711, abs=1e-9
    )
    assert locked["MDD"] == pytest.approx(
        -0.0772023473637689, abs=1e-9
    )
