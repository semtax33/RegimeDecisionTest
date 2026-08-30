from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage11_mse_log_growth_slsqp100 import (
    mse_log_growth_slsqp100 as strategy,
)


def test_expected_metrics_reproduce_requested_formulas() -> None:
    weights = np.array([0.20, 0.50, 0.25, 0.05])
    expected_return = np.array([0.010, 0.003, 0.007, 0.004])
    covariance = np.array(
        [
            [0.0010, 0.0001, 0.0002, 0.0001],
            [0.0001, 0.0002, 0.0000, 0.0000],
            [0.0002, 0.0000, 0.0008, 0.0001],
            [0.0001, 0.0000, 0.0001, 0.0015],
        ]
    )
    metrics = strategy.expected_portfolio_metrics(
        weights, expected_return, covariance
    )
    monthly_return = float(weights @ expected_return)
    monthly_variance = float(weights @ covariance @ weights)
    assert metrics["expected_squared_loss"] == pytest.approx(
        (1 - monthly_return) ** 2 + monthly_variance
    )
    assert metrics["expected_log_growth"] == pytest.approx(
        12 * (monthly_return - 0.5 * monthly_variance)
    )


def test_optimizer_has_all_three_requested_objective_modes() -> None:
    source = inspect.getsource(strategy.optimize_mse_log_growth)
    assert 'if mode == "mse"' in source
    assert 'return metrics["expected_squared_loss"]' in source
    assert 'if mode == "log_growth"' in source
    assert 'return -metrics["expected_log_growth"]' in source
    assert "return -standardized_scores(weights)[0]" in source
    assert strategy.MSE_QUALITY_WEIGHT == 0.50
    assert strategy.LOG_GROWTH_WEIGHT == 0.50


def test_report_confirms_slsqp100_and_full_solver_success() -> None:
    report = json.loads(
        (
            strategy.OUTPUT_DIR / "slsqp_mse_loggrowth100_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["allocation"] == {
        "hard_share": 0.0,
        "slsqp_share": 1.0,
        "leverage": 1.0,
    }
    for name in ["combined", "mse_only", "log_growth_only"]:
        assert report["solver"][name]["months"] == 232
        assert report["solver"][name]["successes"] == 232
        assert report["solver"][name]["failures"] == 0
        assert report["solver"][name]["retries"] == 0
        assert report["solver"][name]["fallbacks"] == 0
        assert all(report["checks"][name].values())


@pytest.mark.parametrize(
    "filename",
    [
        "slsqp_mse_loggrowth100_monthly.csv",
        "slsqp_mse100_monthly.csv",
        "slsqp_loggrowth100_monthly.csv",
    ],
)
def test_saved_paths_respect_bounds_and_constraints(filename: str) -> None:
    path = pd.read_csv(strategy.OUTPUT_DIR / filename, index_col=0)
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    assert len(path) == 232
    assert path["solver_success"].all()
    assert not path["used_retry"].any()
    assert not path["used_fallback"].any()
    assert np.allclose(path[weight_columns].sum(axis=1), 1.0)
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()
    bounds = dict(zip(weight_columns, strategy.BOUNDS, strict=True))
    for column, (lower, upper) in bounds.items():
        assert path[column].min() >= lower - 1e-8
        assert path[column].max() <= upper + 1e-8


def test_full_and_locked_metrics_are_reproduced() -> None:
    comparison = pd.read_csv(
        strategy.OUTPUT_DIR / "performance_comparison.csv"
    )
    expected = {
        "MSELogGrowthObjective_SLSQP100": {
            "full_2007_2026": (
                0.08121806130710008,
                1.062478006177378,
                -0.13374718398394403,
            ),
            "locked_2018_2026": (
                0.0896156905852088,
                1.0478863903190236,
                -0.13374718398394414,
            ),
        },
        "MSEObjective_SLSQP100": {
            "full_2007_2026": (
                0.0812173001322396,
                1.0624704561680909,
                -0.1337492216168642,
            ),
            "locked_2018_2026": (
                0.08961337556969107,
                1.0478638025186549,
                -0.13374922161686453,
            ),
        },
        "LogGrowthObjective_SLSQP100": {
            "full_2007_2026": (
                0.08121808788501039,
                1.0624757517849124,
                -0.13374518927328904,
            ),
            "locked_2018_2026": (
                0.08961709833573206,
                1.0478982895833169,
                -0.13374518927328916,
            ),
        },
    }
    for strategy_name, periods in expected.items():
        rows = comparison.loc[
            comparison["Strategy"].eq(strategy_name)
        ].set_index("Period")
        for period, values in periods.items():
            row = rows.loc[period]
            assert int(row["Months"]) == (232 if period.startswith("full") else 103)
            assert row["CAGR"] == pytest.approx(values[0], abs=1e-9)
            assert row["Sharpe"] == pytest.approx(values[1], abs=1e-9)
            assert row["MDD"] == pytest.approx(values[2], abs=1e-9)


def test_the_three_objectives_produce_nearly_identical_weights() -> None:
    combined = pd.read_csv(
        strategy.OUTPUT_DIR / "slsqp_mse_loggrowth100_monthly.csv"
    )
    mse = pd.read_csv(strategy.OUTPUT_DIR / "slsqp_mse100_monthly.csv")
    log_growth = pd.read_csv(
        strategy.OUTPUT_DIR / "slsqp_loggrowth100_monthly.csv"
    )
    columns = [f"w_{asset}" for asset in ASSETS]
    assert np.max(np.abs(combined[columns] - mse[columns]).to_numpy()) < 5e-4
    assert (
        np.max(
            np.abs(combined[columns] - log_growth[columns]).to_numpy()
        )
        < 5e-4
    )
