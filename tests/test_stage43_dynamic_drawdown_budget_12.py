from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.stage43_dynamic_drawdown_budget_12 import (
    dynamic_drawdown_budget_12_slsqp as strategy,
)


def test_stage43_is_isolated_and_uses_the_declared_absolute_thresholds() -> None:
    assert strategy.OUTPUT_DIR.parent.name == "stage43_dynamic_drawdown_budget_12"
    assert strategy.STRATEGY_NAME == "Stage43_DynamicDD12_ExAnteSharpe11"
    assert strategy.NAV_FLOOR_RATIO == pytest.approx(0.88)
    assert strategy.MDD_FLOOR == pytest.approx(-0.12)
    assert strategy.EX_ANTE_SHARPE_FLOOR == pytest.approx(1.10)
    assert strategy.TAIL_CONFIDENCE == pytest.approx(0.90)

    source = Path(strategy.__file__).read_text(encoding="utf-8")
    assert "monthly_return - 0.5 * monthly_variance - transaction_cost" in source
    assert '"objective": "maximize_expected_geometric_growth_net"' in source
    assert "EX_ANTE_SHARPE_FLOOR" in source
    assert "remaining_loss_budget(current_drawdown)" in source


@pytest.mark.parametrize(
    ("drawdown", "expected_budget"),
    [
        (0.0, -0.12),
        (-0.05, -0.0736842105263158),
        (-0.08, -0.04347826086956519),
        (-0.09, -0.03296703296703296),
    ],
)
def test_remaining_loss_budget_is_derived_from_the_real_nav_state(
    drawdown: float, expected_budget: float
) -> None:
    budget = strategy.remaining_loss_budget(drawdown)
    assert budget == pytest.approx(expected_budget)
    assert (1.0 + drawdown) * (1.0 + budget) == pytest.approx(0.88)


def test_portfolio_statistics_use_expected_log_growth_and_annualized_sharpe() -> None:
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mu = np.array([0.01, 0.004, 0.006, 0.002])
    covariance = np.array(
        [
            [0.0020, 0.0001, 0.0000, 0.0001],
            [0.0001, 0.0005, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0010, 0.0001],
            [0.0001, 0.0000, 0.0001, 0.0030],
        ]
    )
    history = np.array(
        [
            [0.02, 0.00, 0.01, -0.01],
            [-0.01, 0.005, 0.00, 0.02],
            [0.01, -0.02, 0.01, -0.03],
            [-0.02, 0.01, -0.01, 0.01],
        ]
    )
    pretrade = weights.copy()
    values = strategy.portfolio_forecast_statistics(
        weights, mu, covariance, history, pretrade
    )
    expected_return = float(weights @ mu)
    expected_variance = float(weights @ covariance @ weights)
    expected_cost = strategy.stage36.stage35.expected_transaction_cost(
        weights, pretrade
    )
    expected_log_growth = expected_return - 0.5 * expected_variance - expected_cost
    expected_sharpe = (
        np.sqrt(12.0)
        * (expected_return - expected_cost)
        / np.sqrt(expected_variance)
    )
    assert values["expected_monthly_log_growth_net"] == pytest.approx(
        expected_log_growth
    )
    assert values["expected_annual_log_growth_net"] == pytest.approx(
        12.0 * expected_log_growth
    )
    assert values["ex_ante_sharpe"] == pytest.approx(expected_sharpe)
    assert values["historical_cdar90"] == pytest.approx(
        strategy.historical_cdar(history @ weights)
    )


def test_saved_path_stops_at_the_first_realized_floor_breach_without_relaxation() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "stage43_dynamic_dd12_budget_monthly.csv"
    )
    assert len(path) == 19
    assert path["month"].iloc[0] == "2007-04"
    assert path["month"].iloc[-1] == "2008-10"
    assert path["solver_success"].all()
    assert not path["infeasible_portfolio"].any()
    assert path["ex_ante_sharpe_slack"].min() >= -1e-7
    assert path["dynamic_tail_budget_slack"].min() >= -1e-7
    assert path["history_end_month"].lt(path["month"]).all()
    weights = path[[f"w_{asset}" for asset in strategy.ASSETS]]
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
    assert weights.min().min() >= -1e-9
    assert weights.max().max() <= 1.0 + 1e-9

    failure_month = path.iloc[-1]
    assert failure_month["current_drawdown_before_trade"] == pytest.approx(
        -0.07221323053947759
    )
    assert failure_month["remaining_loss_budget"] == pytest.approx(
        -0.05150619844289095
    )
    assert failure_month["historical_cdar90"] == pytest.approx(
        -0.0502154973191129
    )
    assert failure_month["return"] == pytest.approx(-0.07188568977666926)
    assert failure_month["drawdown"] == pytest.approx(-0.13890782242781496)
    assert failure_month["w_GLD"] == pytest.approx(0.608379155977103)
    assert failure_month["expected_annual_volatility"] == pytest.approx(
        0.18303250243192443
    )


def test_performance_marks_incomplete_periods_instead_of_filling_them() -> None:
    performance = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    full = performance.loc[performance["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )
    baseline = full.loc[strategy.STAGE36_NAME]
    candidate = full.loc[strategy.STRATEGY_NAME]
    assert bool(baseline["CompletePeriod"]) is True
    assert baseline["CAGR"] == pytest.approx(0.10499388753644401)
    assert baseline["Sharpe"] == pytest.approx(1.1049112900976705)
    assert baseline["MDD"] == pytest.approx(-0.12407087222855073)
    assert bool(candidate["CompletePeriod"]) is False
    assert int(candidate["Months"]) == 19
    assert pd.isna(candidate["CAGR"])
    assert pd.isna(candidate["Sharpe"])
    assert pd.isna(candidate["MDD"])

    partial = pd.read_csv(
        strategy.OUTPUT_DIR / "partial_failure_window_performance.csv"
    ).set_index("Strategy")
    assert partial.loc[strategy.STRATEGY_NAME, "CAGR"] == pytest.approx(
        0.0868331886614222
    )
    assert partial.loc[strategy.STRATEGY_NAME, "Sharpe"] == pytest.approx(
        0.5871581329828716
    )
    assert partial.loc[strategy.STRATEGY_NAME, "MDD"] == pytest.approx(
        -0.13890782242781496
    )
    assert partial.loc[strategy.STAGE36_NAME, "Sharpe"] == pytest.approx(
        1.3775453874460544
    )


def test_validation_report_proves_causality_failure_and_stage36_freeze() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["stage"] == 43
    assert report["base"] == "Stage36_GVZ_OVXAssetRisk"
    assert report["candidate_complete_full_period"] is False
    assert report["stage36_frozen_files_unchanged"] is True
    assert report["anti_overfit"]["candidate_count"] == 1
    assert report["anti_overfit"]["threshold_search"] is False
    assert report["anti_overfit"]["lookback_search"] is False
    assert report["anti_overfit"]["objective_weight_search"] is False
    assert report["anti_overfit"]["constraint_relaxation"] is False
    assert report["anti_overfit"]["future_return_in_optimizer"] is False
    assert report["eligible_strategies"] == []
    assert report["selected_highest_cagr_strategy"] is None

    check = report["monthly_checks"]
    assert check["months"] == 19
    assert check["all_solver_success"] is True
    assert check["minimum_ex_ante_sharpe_slack"] >= -1e-7
    assert check["minimum_dynamic_tail_budget_slack"] >= -1e-7
    assert check["all_history_ends_before_target"] is True
    assert check["minimum_realized_drawdown"] < -0.12

    event = report["infeasible_events"]
    assert len(event) == 1
    assert event[0]["month"] == "2008-11"
    assert event[0]["reason"] == "DrawdownFloorAlreadyBreached"
    assert event[0]["current_drawdown"] == pytest.approx(-0.13890782242781496)
    assert report["validation_gates"][strategy.STAGE36_NAME][
        "sharpe_at_least_1_1"
    ] is True
    assert report["validation_gates"][strategy.STAGE36_NAME][
        "mdd_at_least_minus_12pct"
    ] is False
    assert report["validation_gates"][strategy.STRATEGY_NAME][
        "full_validation_pass"
    ] is False


def test_stage43_outputs_and_readme_document_the_negative_result() -> None:
    for filename in (
        "stage43_dynamic_dd12_budget_monthly.csv",
        "performance_comparison.csv",
        "partial_failure_window_performance.csv",
        "infeasible_events.csv",
        "validation_report.json",
    ):
        assert (strategy.OUTPUT_DIR / filename).is_file(), filename

    readme = strategy.OUTPUT_DIR.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert readme.stat().st_size > 10_000
    for phrase in (
        "동적 MDD 기준을 -12%로 완화한 재검증",
        "expected geometric growth",
        "ex-ante Sharpe 제약",
        "2008-10 실패 분석",
        "왜 -12% 완화가 더 나빠졌나",
        "불가능 해와 전체성과 처리",
        "과최적화 방지",
        "Stage42에서 바꾼 것은 하나",
        "Stage36 13% 변동성 guard를 넣지 않은 이유",
    ):
        assert phrase in text
