from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage14_unconstrained_dynamic_risk_slsqp import (
    dynamic_risk_slsqp as strategy,
)


def test_simplex_projection_has_no_majority_cap() -> None:
    projected = strategy.project_to_long_only_simplex(
        np.array([1.2, -0.1, -0.05, -0.05])
    )
    assert np.allclose(projected, [1.0, 0.0, 0.0, 0.0])
    assert projected.sum() == pytest.approx(1.0)


def test_dynamic_lambda_uses_the_causal_percentile_without_search() -> None:
    assert strategy.STATIC_RISK_POLICY.risk_aversion(0.75) == 1.0
    assert strategy.DYNAMIC_RISK_POLICY.risk_aversion(0.0) == 1.0
    assert strategy.DYNAMIC_RISK_POLICY.risk_aversion(0.75) == 1.75
    assert strategy.DYNAMIC_RISK_POLICY.risk_aversion(1.0) == 2.0
    source = inspect.getsource(strategy.solve_weights)
    assert "- downside_lambda * downside_semivariance" in source
    assert "volatility_multiplier" not in source


def test_report_confirms_removed_asset_cap_and_no_leverage() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["allocation"] == {
        "slsqp_share": 1.0,
        "weight_sum": 1.0,
        "asset_bounds": [0.0, 1.0],
        "single_asset_majority_rule": False,
        "cash_asset": False,
        "leverage": 1.0,
    }
    assert report["risk_aversion"]["searched_alpha"] is None
    assert report["dynamic_lambda_concentration"][
        "months_above_50_percent"
    ] > 0
    assert report["dynamic_lambda_concentration"][
        "maximum_single_asset_weight"
    ] > 0.90
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    "filename",
    [
        "no_asset_cap_static_lambda_monthly.csv",
        "no_asset_cap_dynamic_lambda_monthly.csv",
    ],
)
def test_saved_paths_are_fully_invested_long_only_without_asset_cap(
    filename: str,
) -> None:
    path = pd.read_csv(strategy.OUTPUT_DIR / filename, index_col=0)
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    assert len(path) == 232
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert np.allclose(path[weight_columns].sum(axis=1), 1.0)
    assert (path[weight_columns] >= -1e-9).all().all()
    assert (path[weight_columns] <= 1.0 + 1e-9).all().all()
    assert path[weight_columns].max(axis=1).max() > 0.90
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()


def test_every_signal_precedes_the_investment_month() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "no_asset_cap_dynamic_lambda_monthly.csv"
    )
    target = pd.PeriodIndex(path["month"], freq="M")
    macro = pd.PeriodIndex(path["macro_signal_month"], freq="M")
    stress = pd.PeriodIndex(path["stress_signal_month"], freq="M")
    stress_date = pd.to_datetime(path["stress_signal_date"]).dt.to_period("M")
    assert (macro < target).all()
    assert (stress < target).all()
    assert (stress_date < target).all()


def test_dynamic_risk_attribution_reconciles() -> None:
    attribution = pd.read_csv(
        strategy.OUTPUT_DIR / "dynamic_risk_attribution.csv"
    )
    expected = (
        attribution["return_stress_aware"] - attribution["return_base"]
    )
    assert np.allclose(attribution["overlay_alpha"], expected)


def test_saved_full_period_metrics_are_reproduced() -> None:
    comparison = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    result = comparison.loc[
        comparison["Strategy"].eq("Stage14_NoAssetCap_DynamicLambda")
        & comparison["Period"].eq("full_2007_2026")
    ].iloc[0]
    assert int(result["Months"]) == 232
    assert result["CAGR"] == pytest.approx(0.1075840, abs=1e-6)
    assert result["Sharpe"] == pytest.approx(0.845965, abs=1e-6)
    assert result["MDD"] == pytest.approx(-0.229738, abs=1e-6)
