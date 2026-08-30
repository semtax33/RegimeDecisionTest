from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage18_program_flow_stress import program_flow_stress as strategy


def test_weekly_program_pressure_uses_only_volume_and_independent_categories() -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    rows = []
    for index, date in enumerate(dates):
        rows.extend(
            [
                {
                    "date": date,
                    "category": "차익",
                    "sell_volume": 20 + index,
                    "buy_volume": 10,
                },
                {
                    "date": date,
                    "category": "비차익",
                    "sell_volume": 10,
                    "buy_volume": 20 + index,
                },
            ]
        )
    features, _ = strategy.build_program_volume_features(pd.DataFrame(rows))
    expected_arbitrage = (
        sum((20 + index) - 10 for index in range(5))
        / sum((20 + index) + 10 for index in range(5))
    )
    assert features.iloc[4]["arbitrage_sell_pressure_5d"] == pytest.approx(
        expected_arbitrage
    )
    assert features.iloc[:4]["arbitrage_sell_pressure_5d"].isna().all()
    assert features.iloc[4:]["program_stress_component"].between(0.0, 1.0).all()
    assert not any("value" in column for column in features.columns)
    assert not any("total" in column for column in features.columns)


def test_saved_program_data_audit_is_complete_and_arithmetically_consistent() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    audit = report["data_audit"]
    assert audit["unique_dates"] == 6083
    assert audit["start"] == "2002-01-02"
    assert audit["end"] == "2026-08-28"
    assert audit["duplicate_keys"] == 0
    assert audit["days_without_three_categories"] == 0
    assert audit["missing_numeric_values"] == 0
    assert audit["net_volume_mismatches"] == 0
    assert audit["net_value_mismatches"] == 0
    assert not any(audit["total_vs_parts_mismatches"].values())


def test_saved_path_is_causal_fully_invested_and_long_only() -> None:
    path = pd.read_csv(
        strategy.OUTPUT_DIR / "program_stress_dynamic_monthly.csv"
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
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()


def test_saved_report_records_one_formula_and_no_parameter_search() -> None:
    report = json.loads(
        (strategy.OUTPUT_DIR / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["program_feature_formula"]["searched_parameters"] is None
    assert all(report["checks"].values())
    assert report["program_component_correlation_with_original_stress"] == pytest.approx(
        0.05956164, abs=1e-8
    )


def test_saved_result_does_not_falsely_claim_an_improvement() -> None:
    comparison = pd.read_csv(strategy.OUTPUT_DIR / "performance_comparison.csv")
    full = comparison.loc[comparison["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )
    baseline = full.loc["Stage14_Original_DynamicLambda"]
    candidate = full.loc["Stage18_ProgramStress_DynamicLambda"]
    assert baseline["CAGR"] == pytest.approx(0.1075840, abs=1e-6)
    assert candidate["CAGR"] == pytest.approx(0.1070512, abs=1e-6)
    assert candidate["Sharpe"] < baseline["Sharpe"]
    assert candidate["MDD"] < baseline["MDD"]

