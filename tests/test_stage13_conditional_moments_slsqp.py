from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage13_conditional_moments_slsqp import (
    economic_conditional_slsqp as strategy,
)


def test_expanding_midrank_is_causal_and_handles_ties() -> None:
    values = pd.Series([2.0, 1.0, 2.0, 4.0])
    actual = strategy.causal_expanding_midrank(values)
    expected = pd.Series([0.5, 0.25, 2.0 / 3.0, 0.875])
    assert np.allclose(actual, expected)


def test_design_contract_has_no_model_or_parameter_search() -> None:
    source = inspect.getsource(strategy)
    assert "SparseJump" not in source
    assert "LogisticRegression" not in source
    assert "GridSearch" not in source
    assert "ParameterGrid" not in source
    assert "post_optimizer_overlay_share\": 0.0" in source
    assert strategy.MAX_SINGLE_ASSET_WEIGHT == 0.50
    assert strategy.CATASTROPHE_ANNUAL_VOLATILITY == 0.13


def test_report_records_economic_design_and_no_leverage() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "research_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["allocation"] == {
        "slsqp_share": 1.0,
        "hard_regime_share": 0.0,
        "post_optimizer_overlay_share": 0.0,
        "leverage": 1.0,
    }
    assert report["tunable_hyperparameters"] == []
    assert all(report["checks"].values())
    assert report["stress_aware_solver"]["months"] == 232
    assert report["stress_aware_solver"]["fallbacks"] == 0


@pytest.mark.parametrize(
    "filename",
    ["macro_conditional_monthly.csv", "macro_stress_conditional_monthly.csv"],
)
def test_saved_weights_are_unlevered_and_constraints_are_valid(
    filename: str,
) -> None:
    path = pd.read_csv(strategy.OUTPUT_DIR / filename, index_col=0)
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    assert len(path) == 232
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert np.allclose(path[weight_columns].sum(axis=1), 1.0)
    assert (path[weight_columns] >= -1e-9).all().all()
    assert (
        path[weight_columns] <= strategy.MAX_SINGLE_ASSET_WEIGHT + 1e-8
    ).all().all()
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()


def test_all_signals_precede_the_investment_month() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "macro_stress_conditional_monthly.csv"
    )
    target = pd.PeriodIndex(path["month"], freq="M")
    macro = pd.PeriodIndex(path["macro_signal_month"], freq="M")
    stress_month = pd.PeriodIndex(path["stress_signal_month"], freq="M")
    stress_date = pd.to_datetime(path["stress_signal_date"]).dt.to_period("M")
    assert (macro < target).all()
    assert (stress_month < target).all()
    assert (stress_date < target).all()


def test_overlay_attribution_reconciles_exactly() -> None:
    attribution = pd.read_csv(
        strategy.OUTPUT_DIR / "risk_overlay_attribution.csv"
    )
    expected = (
        attribution["return_stress_aware"] - attribution["return_base"]
    )
    assert np.allclose(attribution["overlay_alpha"], expected)


def test_regression_metrics_match_saved_research_result() -> None:
    comparison = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    result = comparison.loc[
        comparison["Strategy"].eq(
            "Stage13_MacroStressConditional_SLSQP100"
        )
        & comparison["Period"].eq("full_2007_2026")
    ].iloc[0]
    assert int(result["Months"]) == 232
    assert result["CAGR"] == pytest.approx(0.1150184, abs=1e-6)
    assert result["Sharpe"] == pytest.approx(0.900892, abs=1e-6)
    assert result["MDD"] == pytest.approx(-0.213193, abs=1e-6)
