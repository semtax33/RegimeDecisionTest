from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage10_slsqp_sharpe100 import slsqp_sharpe100 as strategy


def test_report_confirms_slsqp_is_one_hundred_percent() -> None:
    report = json.loads(
        (
            strategy.OUTPUT_DIR / "slsqp_sharpe_cagr100_report.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert report["strategy"] == "SharpeCAGRObjective_SLSQP100"
    assert report["allocation"] == {
        "hard_share": 0.0,
        "slsqp_share": 1.0,
        "leverage": 1.0,
    }
    assert report["solver"]["months"] == 232
    assert report["solver"]["successes"] == 232
    assert report["solver"]["failures"] == 0
    assert report["solver"]["fallbacks"] == 0
    assert all(report["checks"].values())


def test_objective_is_negative_expected_sharpe_only() -> None:
    source = inspect.getsource(strategy.maximize_expected_sharpe)
    assert "return -annualized_values(weights)[2]" in source
    for removed_term in (
        "return_reward",
        "vol_penalty",
        "cdar_penalty",
        "turnover_penalty",
        "tracking_penalty",
    ):
        assert removed_term not in source


def test_combined_objective_contains_equal_weight_sharpe_and_cagr() -> None:
    source = inspect.getsource(strategy.maximize_expected_sharpe_and_cagr)
    assert strategy.SHARPE_SCORE_WEIGHT == 0.50
    assert strategy.CAGR_SCORE_WEIGHT == 0.50
    assert "SHARPE_SCORE_WEIGHT * standardized[0]" in source
    assert "CAGR_SCORE_WEIGHT * standardized[1]" in source
    assert "12 * (monthly_return - 0.5 * monthly_variance)" in source
    assert "return -standardized_score(weights)[0]" in source
    for removed_term in (
        "vol_penalty",
        "cdar_penalty",
        "turnover_penalty",
        "tracking_penalty",
    ):
        assert removed_term not in source


@pytest.mark.parametrize(
    "filename",
    [
        "slsqp_sharpe_cagr100_monthly.csv",
        "slsqp_sharpe100_monthly.csv",
    ],
)
def test_saved_weights_and_constraints_are_valid(filename: str) -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / filename,
        index_col=0,
    )
    columns = [f"w_{asset}" for asset in ASSETS]
    assert len(path) == 232
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert np.allclose(path[columns].sum(axis=1), 1.0)
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()
    bounds = {
        "w_KODEX200": (0.02, 0.68),
        "w_BOND": (0.05, 0.88),
        "w_GLD": (0.02, 0.62),
        "w_USO": (0.00, 0.38),
    }
    for column, (lower, upper) in bounds.items():
        assert path[column].min() >= lower - 1e-8
        assert path[column].max() <= upper + 1e-8


def test_full_and_locked_metrics_are_reproduced() -> None:
    comparison = pd.read_csv(
        strategy.OUTPUT_DIR / "performance_comparison.csv"
    )
    combined = comparison.loc[
        comparison["Strategy"].eq("SharpeCAGRObjective_SLSQP100")
    ].set_index("Period")
    full = combined.loc["full_2007_2026"]
    locked = combined.loc["locked_2018_2026"]
    assert int(full["Months"]) == 232
    assert full["CAGR"] == pytest.approx(0.0685852252963858, abs=1e-9)
    assert full["Sharpe"] == pytest.approx(1.3293257462723729, abs=1e-9)
    assert full["MDD"] == pytest.approx(-0.0687831090912491, abs=1e-9)
    assert int(locked["Months"]) == 103
    assert locked["CAGR"] == pytest.approx(0.0698541699071659, abs=1e-9)
    assert locked["Sharpe"] == pytest.approx(1.1773216422131998, abs=1e-9)
    assert locked["MDD"] == pytest.approx(-0.0687831090912488, abs=1e-9)

    sharpe_only = comparison.loc[
        comparison["Strategy"].eq("SharpeOnlyObjective_SLSQP100")
    ].set_index("Period")
    full = sharpe_only.loc["full_2007_2026"]
    locked = sharpe_only.loc["locked_2018_2026"]
    assert int(full["Months"]) == 232
    assert full["CAGR"] == pytest.approx(0.0502590122718191, abs=1e-9)
    assert full["Sharpe"] == pytest.approx(1.3755414249516504, abs=1e-9)
    assert full["MDD"] == pytest.approx(-0.0667367917939356, abs=1e-9)
    assert int(locked["Months"]) == 103
    assert locked["CAGR"] == pytest.approx(0.0508102206848934, abs=1e-9)
    assert locked["Sharpe"] == pytest.approx(1.1371430265839768, abs=1e-9)
    assert locked["MDD"] == pytest.approx(-0.0667367917939355, abs=1e-9)
