from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.stage41_cagr_guardrail_objectives import (
    cagr_guardrail_slsqp as strategy,
)


def test_stage41_is_isolated_and_maximizes_cagr_in_both_modes() -> None:
    assert strategy.OUTPUT_DIR.parent.name == "stage41_cagr_guardrail_objectives"
    assert strategy.STRATEGIES == {
        "Stage41_CAGR_HardMDD14_Sharpe1": "hard_mdd",
        "Stage41_CAGR_CDaR16_Sharpe1": "cdar",
    }
    assert strategy.MDD_FLOOR == pytest.approx(-0.14)
    assert strategy.SHARPE_FLOOR == pytest.approx(1.0)
    assert strategy.CDAR_FLOOR == pytest.approx(-0.16)

    source = Path(strategy.__file__).read_text(encoding="utf-8")
    assert 'return -cagr if np.isfinite(cagr) else 1e12' in source
    assert '"objective": "maximize_causal_historical_cagr"' in source
    assert "calmar_or_sortino_ratio_objective" in source


def test_historical_max_drawdown_includes_initial_wealth_peak() -> None:
    result = strategy.historical_max_drawdown(np.array([0.10, -0.20, 0.05]))
    assert result == pytest.approx(-0.20)
    assert strategy.historical_max_drawdown(np.array([])) == 0.0


def test_historical_statistics_use_geometric_growth_and_prior_scenarios() -> None:
    weights = np.array([0.6, 0.4])
    history = np.array(
        [
            [0.02, 0.00],
            [-0.01, 0.01],
            [0.03, -0.02],
            [-0.02, 0.00],
        ]
    )
    scenario = history @ weights
    values = strategy.historical_statistics(weights, history)

    expected_cagr = np.expm1(12.0 * np.mean(np.log1p(scenario)))
    expected_sharpe = np.sqrt(12.0) * np.mean(scenario) / np.std(
        scenario, ddof=1
    )
    assert values["historical_cagr"] == pytest.approx(expected_cagr)
    assert values["historical_sharpe"] == pytest.approx(expected_sharpe)
    assert values["historical_mdd"] == pytest.approx(
        strategy.historical_max_drawdown(scenario)
    )
    assert values["historical_cdar"] == pytest.approx(
        strategy.stage36.stage35.cdar(
            scenario, strategy.stage36.stage35.CDAR_CONFIDENCE
        )
    )


def test_stage41_outputs_exist_and_cover_the_full_232_month_path() -> None:
    for filename in (
        "stage41_hard_mdd_monthly.csv",
        "stage41_cdar_monthly.csv",
        "performance_comparison.csv",
        "paired_block_bootstrap_vs_stage36.csv",
        "validation_report.json",
    ):
        assert (strategy.OUTPUT_DIR / filename).is_file(), filename

    for filename in ("stage41_hard_mdd_monthly.csv", "stage41_cdar_monthly.csv"):
        path = pd.read_csv(strategy.OUTPUT_DIR / filename)
        assert len(path) == 232
        assert path["month"].iloc[0] == "2007-04"
        assert path["month"].iloc[-1] == "2026-07"
        assert path["history_end_month"].lt(path["month"]).all()
        assert path["solver_success"].all()
        assert not path["infeasible_portfolio"].any()
        assert np.allclose(
            path[[f"w_{asset}" for asset in strategy.ASSETS]].sum(axis=1),
            1.0,
            atol=1e-8,
        )


def test_performance_regression_and_final_absolute_gates() -> None:
    performance = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    full = performance.loc[performance["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )
    assert set(full.index) == {"Stage36_Frozen", *strategy.STRATEGIES}

    assert full.loc["Stage36_Frozen", "CAGR"] == pytest.approx(0.1049938875)
    assert full.loc["Stage36_Frozen", "Sharpe"] == pytest.approx(1.1049112901)
    assert full.loc["Stage36_Frozen", "MDD"] == pytest.approx(-0.1240708722)

    hard = full.loc["Stage41_CAGR_HardMDD14_Sharpe1"]
    assert hard["CAGR"] == pytest.approx(0.0982923542)
    assert hard["Sharpe"] == pytest.approx(0.7065962472)
    assert hard["MDD"] == pytest.approx(-0.2501919847)

    cdar = full.loc["Stage41_CAGR_CDaR16_Sharpe1"]
    assert cdar["CAGR"] == pytest.approx(0.0964435227)
    assert cdar["Sharpe"] == pytest.approx(0.6572631327)
    assert cdar["MDD"] == pytest.approx(-0.2535834358)

    assert full.loc["Stage36_Frozen", "Sharpe"] >= strategy.SHARPE_FLOOR
    assert full.loc["Stage36_Frozen", "MDD"] >= strategy.MDD_FLOOR
    assert (full.loc[list(strategy.STRATEGIES), "Sharpe"] < 1.0).all()
    assert (full.loc[list(strategy.STRATEGIES), "MDD"] < -0.14).all()


def test_monthly_constraints_pass_but_dynamic_realized_path_fails() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["base"] == "Stage36_GVZ_OVXAssetRisk"
    assert report["objective"] == "maximize causal historical CAGR"
    assert report["stage36_frozen_files_unchanged"] is True
    assert report["anti_overfit"]["candidate_count"] == 2
    assert report["anti_overfit"]["threshold_grid_search"] is False
    assert report["anti_overfit"]["lookback_grid_search"] is False
    assert report["anti_overfit"]["infeasible_guardrail_relaxation"] is False
    assert report["deliberately_not_used"][
        "future_realized_returns_in_optimization"
    ] is True
    assert report["eligible_strategies"] == ["Stage36_Frozen"]
    assert report["selected_highest_cagr_strategy"] == "Stage36_Frozen"

    for name, check in report["checks"].items():
        assert check["months"] == 232, name
        assert check["all_solver_success"] is True, name
        assert check["infeasible_months"] == 0, name
        assert check["max_sum_error"] <= 1e-8, name
        assert check["minimum_weight"] >= -1e-9, name
        assert check["maximum_weight"] <= 1.0 + 1e-9, name
        assert check["minimum_historical_sharpe_slack"] >= -1e-7, name
        assert check["minimum_selected_guardrail_slack"] >= -1e-7, name
        assert check["all_history_ends_before_target"] is True, name

        gate = report["validation_gates"][name]
        assert gate["all_monthly_optimization_constraints_pass"] is True, name
        assert gate["full_mdd_at_least_minus_14pct"] is False, name
        assert gate["full_sharpe_at_least_one"] is False, name
        assert gate["full_validation_pass"] is False, name


def test_readme_explains_formulas_results_caveat_and_files() -> None:
    readme = strategy.OUTPUT_DIR.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert readme.stat().st_size > 8_000
    for phrase in (
        "CAGR 최대화 + MDD·Sharpe 절대 가드레일",
        "A안: Hard MDD + Sharpe",
        "B안: CDaR + Sharpe",
        "왜 월별 제약을 지켰는데 최종 MDD가 -25%인가",
        "Stage36에서 보존한 것과 바뀐 것",
        "불가능 해 처리",
        "과최적화 방지",
        "Stage36_Frozen",
    ):
        assert phrase in text

    for relative in (
        "cagr_guardrail_slsqp.py",
        "outputs/stage41_hard_mdd_monthly.csv",
        "outputs/stage41_cdar_monthly.csv",
        "outputs/performance_comparison.csv",
        "outputs/validation_report.json",
    ):
        assert (readme.parent / relative).is_file()
