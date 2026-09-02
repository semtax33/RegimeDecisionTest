from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.stage40_calmar_sortino_objectives import (
    ratio_objective_slsqp as strategy,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage40_is_isolated_and_has_both_maximization_modes() -> None:
    assert strategy.OUTPUT_DIR.parent.name == "stage40_calmar_sortino_objectives"
    assert strategy.OBJECTIVE_MODES == {
        "Stage40_CausalCalmar": "calmar",
        "Stage40_CausalSortino": "sortino",
    }
    source = Path(strategy.__file__).read_text(encoding="utf-8")
    assert 'objective_key = f"causal_{objective_mode}_objective"' in source
    assert "return -float(ratio)" in source


def test_historical_max_drawdown_includes_initial_wealth_peak() -> None:
    result = strategy.historical_max_drawdown(np.array([0.10, -0.20, 0.05]))
    assert result == pytest.approx(-0.20)
    assert strategy.historical_max_drawdown(np.array([])) == 0.0


def test_ratio_statistics_are_causal_ratio_definitions() -> None:
    weights = np.array([0.5, 0.5])
    expected_return = np.array([0.01, 0.005])
    covariance = np.array([[0.002, 0.0], [0.0, 0.001]])
    history = np.array([[0.02, 0.0], [-0.01, 0.005], [0.01, -0.02]])
    values = strategy.ratio_statistics(
        weights, expected_return, covariance, history, 0.0
    )
    scenario = history @ weights
    downside = np.sqrt(np.mean(np.minimum(scenario, 0.0) ** 2)) * np.sqrt(12)
    expected_cagr = np.expm1(12 * (0.0075 - 0.5 * 0.00075))
    assert values["causal_sortino_objective"] == pytest.approx(0.09 / downside)
    assert values["causal_calmar_objective"] == pytest.approx(
        expected_cagr / abs(strategy.historical_max_drawdown(scenario))
    )


def test_stage40_outputs_exist_and_cover_232_months() -> None:
    for filename in (
        "stage40_calmar_monthly.csv",
        "stage40_sortino_monthly.csv",
        "performance_comparison.csv",
        "paired_block_bootstrap_vs_stage36.csv",
        "validation_report.json",
    ):
        assert (strategy.OUTPUT_DIR / filename).is_file(), filename

    for filename in ("stage40_calmar_monthly.csv", "stage40_sortino_monthly.csv"):
        path = pd.read_csv(strategy.OUTPUT_DIR / filename)
        assert len(path) == 232
        assert path["month"].iloc[0] == "2007-04"
        assert path["month"].iloc[-1] == "2026-07"


def test_performance_and_full_sharpe_floor_regression() -> None:
    performance = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    full = performance.loc[performance["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )
    assert set(full.index) == {
        "Stage36_Frozen",
        "Stage40_CausalCalmar",
        "Stage40_CausalSortino",
    }
    assert full.loc["Stage40_CausalCalmar", "Sharpe"] == pytest.approx(
        1.2215186469
    )
    assert full.loc["Stage40_CausalSortino", "Sharpe"] == pytest.approx(
        1.0474003921
    )
    assert (full.loc[["Stage40_CausalCalmar", "Stage40_CausalSortino"], "Sharpe"] >= 1.0).all()
    assert full.loc["Stage40_CausalCalmar", "Calmar"] < full.loc[
        "Stage36_Frozen", "Calmar"
    ]
    assert full.loc["Stage40_CausalSortino", "Sortino"] < full.loc[
        "Stage36_Frozen", "Sortino"
    ]


def test_constraints_causality_and_stage36_freeze_audit() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["base"] == "Stage36_GVZ_OVXAssetRisk"
    assert report["change_scope"] == "SLSQP objective only"
    assert report["stage36_frozen_files_unchanged"] is True
    assert report["anti_overfit"]["future_returns_in_objective"] is False
    assert report["selected_strategy"] == "Stage36_Frozen"
    assert report["promoted_strategies"] == []
    assert set(report["full_sharpe_requirement_eligible_strategies"]) == {
        "Stage40_CausalCalmar",
        "Stage40_CausalSortino",
    }
    for name, check in report["checks"].items():
        assert check["months"] == 232, name
        assert check["all_solver_success"] is True, name
        assert check["all_signal_months_precede_target"] is True, name
        assert check["max_sum_error"] <= 1e-8, name
        assert check["minimum_weight"] >= -1e-9, name
        assert check["maximum_weight"] <= 1.0 + 1e-9, name
        assert check["minimum_volatility_slack"] >= -1e-7, name
        assert check["minimum_cdar_slack"] >= -1e-7, name


def test_ratio_maximization_exposes_bond_concentration_without_hidden_cap() -> None:
    calmar = pd.read_csv(strategy.OUTPUT_DIR / "stage40_calmar_monthly.csv")
    sortino = pd.read_csv(strategy.OUTPUT_DIR / "stage40_sortino_monthly.csv")
    assert calmar["w_BOND"].mean() == pytest.approx(0.8990548374)
    assert sortino["w_BOND"].mean() == pytest.approx(0.9528081770)
    assert calmar[[f"w_{asset}" for asset in strategy.ASSETS]].max(axis=1).max() > 0.90
    assert sortino[[f"w_{asset}" for asset in strategy.ASSETS]].max(axis=1).max() > 0.90


def test_readme_explains_maximization_results_and_files() -> None:
    readme = strategy.OUTPUT_DIR.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert readme.stat().st_size > 8_000
    for phrase in (
        "Calmar·Sortino 최대화 SLSQP",
        "`-Calmar`, `-Sortino`",
        "Stage36에서 보존한 구조",
        "왜 채권에 집중됐는가",
        "Sharpe 1 조건과 최종판정",
        "Stage36_Frozen",
        "fallback",
        "과최적화 방지",
    ):
        assert phrase in text

    for relative in (
        "ratio_objective_slsqp.py",
        "outputs/stage40_calmar_monthly.csv",
        "outputs/stage40_sortino_monthly.csv",
        "outputs/performance_comparison.csv",
        "outputs/validation_report.json",
    ):
        assert (readme.parent / relative).is_file()
