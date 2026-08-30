from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS, load_monthly_asset_returns
from strategies.stage19_asset_trend_confidence import (
    asset_trend_confidence_slsqp as strategy,
)


def test_confidence_mapping_is_symmetric_fixed_and_parameter_free() -> None:
    history = pd.DataFrame(
        {
            "KODEX200": [0.01] * 12,
            "BOND": [-0.01] * 12,
            "GLD": [-0.02] * 12,
            "USO": [0.02] * 12,
        }
    )
    detail = strategy.asset_trend_confidence(
        history, np.array([0.02, 0.01, 0.03, 0.00])
    )
    assert detail["macro_neutral_return"] == pytest.approx(0.015)
    assert np.allclose(detail["trend_score"], [1.0, -1.0, -1.0, 1.0])
    assert np.allclose(detail["macro_confidence"], [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(
        detail["filtered_macro_expected_return"], [0.02, 0.01, 0.015, 0.015]
    )


def test_trend_is_a_confidence_filter_not_an_added_alpha() -> None:
    source = inspect.getsource(strategy.solve_weights)
    assert "expected_return = filtered_macro + stress_adjustment" in source
    assert "downside_risk_aversion_lambda\": 1.0" in source
    assert "concentration_penalty" not in source
    assert "risk_aversion(stress" not in source


def test_every_saved_trend_recomputes_from_strictly_prior_returns() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "asset_trend_confidence_monthly.csv",
        index_col=0,
    )
    path.index = pd.PeriodIndex(path.index, freq="M")
    returns, _ = load_monthly_asset_returns(False)
    assert strategy.verify_saved_trends_are_causal(path, returns)


def test_saved_path_is_causal_long_only_unlevered_and_static_lambda() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "asset_trend_confidence_monthly.csv"
    )
    target = pd.PeriodIndex(path["month"], freq="M")
    macro = pd.PeriodIndex(path["macro_signal_month"], freq="M")
    stress = pd.PeriodIndex(path["stress_signal_month"], freq="M")
    stress_date = pd.to_datetime(path["stress_signal_date"]).dt.to_period("M")
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert (macro < target).all()
    assert (stress < target).all()
    assert (stress_date < target).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-9).all().all()
    assert (weights <= 1.0 + 1e-9).all().all()
    assert np.allclose(path["downside_risk_aversion_lambda"], 1.0)
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()


def test_stage14_diagnostic_confirms_gold_macro_trend_conflict() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = report["stage14_gold_episode"]
    candidate = report["trend_strategy_gold_episode"]
    assert baseline["average_gold_weight"] == pytest.approx(0.4947023, abs=1e-7)
    assert baseline["average_gold_macro_mu_monthly"] > 0.009
    assert baseline["average_gold_trend_6m"] < -0.08
    assert baseline["average_gold_trend_12m"] < -0.15
    assert baseline["both_gold_trends_negative_months"] == 21
    assert candidate["average_gold_weight"] < 0.20
    assert candidate["average_gold_macro_confidence"] == pytest.approx(0.08)


def test_gold_drawdown_improves_but_full_success_gates_fail() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    episodes = pd.DataFrame(report["top_drawdown_episodes"])
    baseline_gold = episodes.loc[
        (episodes["Strategy"] == "Stage14_StaticLambda")
        & (episodes["UnderwaterStart"] == "2012-10")
    ].iloc[0]
    candidate_gold = episodes.loc[
        (episodes["Strategy"] == "Stage19_AssetTrendConfidence")
        & (episodes["UnderwaterStart"] == "2012-10")
    ].iloc[0]
    assert baseline_gold["EpisodeMDD"] == pytest.approx(-0.231921, abs=1e-6)
    assert candidate_gold["EpisodeMDD"] == pytest.approx(-0.112242, abs=1e-6)
    gates = report["success_gates"]
    assert not gates["cagr_at_least_10_5_percent"]
    assert not gates["sharpe_at_least_0_85"]
    assert gates["mdd_no_worse_than_18_percent"]
    assert gates["no_leverage"]
    assert gates["no_hard_regime"]


def test_saved_performance_and_concentration_tradeoff_are_explicit() -> None:
    comparison = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    baseline = full.loc["Stage14_StaticLambda"]
    candidate = full.loc["Stage19_AssetTrendConfidence"]
    assert candidate["CAGR"] == pytest.approx(0.0975759, abs=1e-6)
    assert candidate["Sharpe"] == pytest.approx(0.8392071, abs=1e-6)
    assert candidate["MDD"] == pytest.approx(-0.1589795, abs=1e-6)
    assert candidate["CAGR"] < baseline["CAGR"]
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_80 = next(
        row
        for row in report["stage14_concentration_thresholds"]
        if row["Threshold"] == 0.8
    )
    candidate_80 = next(
        row
        for row in report["trend_strategy_concentration_thresholds"]
        if row["Threshold"] == 0.8
    )
    assert baseline_80["Months"] == 10
    assert candidate_80["Months"] == 35
    assert candidate_80["BONDMonths"] == 25


def test_report_proves_one_hypothesis_and_no_search() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["confidence_formula"]["searched_parameters"] is None
    assert report["confidence_formula"]["candidate_count"] == 1
    assert all(report["checks"].values())
