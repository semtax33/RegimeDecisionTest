from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage28_option_directional_surface import (
    option_directional_surface_slsqp as stage28,
)
from strategies.stage29_option_two_axis_confirmation import (
    two_axis_option_direction_slsqp as stage29,
)


def test_second_thursday_and_predeclared_horizons() -> None:
    assert stage28._second_thursday(2026, 9) == pd.Timestamp("2026-09-10")
    assert stage28.TARGET_MATURITY_DAYS == 30
    assert stage28.FAST_DAYS == 5
    assert stage28.SLOW_DAYS == 20


def test_saved_daily_surface_has_declared_factor_arithmetic() -> None:
    daily = pd.read_csv(
        stage28.OUTPUT_DIR / "daily_option_direction_features.csv",
        parse_dates=["date"],
    ).set_index("date")
    valid = daily.dropna(
        subset=[
            "bear_pressure_fast",
            "bear_pressure_slow",
            "option_direction_fast",
            "option_direction_slow",
            "option_direction",
            "option_direction_score",
        ]
    )
    assert len(daily) == 4_997
    assert daily.index.min() == pd.Timestamp("2000-01-05")
    assert daily.index.max() == pd.Timestamp("2026-08-27")
    assert np.allclose(
        valid["bear_pressure_fast"],
        (
            valid["z_put_skew_change_5"]
            + valid["z_iva_change_5"]
            - valid["z_cpv_change_5"]
        )
        / 3.0,
    )
    assert np.allclose(
        valid["option_direction"],
        valid[["option_direction_fast", "option_direction_slow"]].mean(axis=1),
    )
    assert np.allclose(
        valid["option_direction_score"],
        valid["option_direction"] / (1.0 + valid["option_direction"].abs()),
    )
    assert valid["option_direction_score"].between(-1.0, 1.0).all()


def test_maturity_proxy_limitation_is_measured_and_disclosed() -> None:
    report = json.loads(
        (stage28.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    limitation = report["maturity_limitation"]
    assert limitation["requested_target_days"] == 30
    assert limitation["exact_or_interpolated_rows"] == 103
    assert limitation["nearest_listed_proxy_rows"] == 4_894
    assert report["data_audit"]["option_surface"]["night_session_excluded"]
    assert report["data_audit"]["option_surface"]["searched_parameters"] is None


def test_monthly_option_signal_is_last_known_prior_month_surface() -> None:
    daily = pd.read_csv(
        stage28.OUTPUT_DIR / "daily_option_direction_features.csv",
        index_col=0,
        parse_dates=True,
    )
    monthly = pd.read_csv(
        stage28.OUTPUT_DIR / "monthly_option_direction_signals.csv",
        index_col=0,
    )
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    assert stage28.verify_monthly_option_signals(monthly, daily)
    signal_period = pd.to_datetime(monthly["option_signal_date"]).dt.to_period("M")
    assert (signal_period.to_numpy() < monthly.index.to_numpy()).all()


def test_vix6_is_absent_from_candidate_stress_and_ods_changes_only_equity_mu() -> None:
    stress = pd.read_csv(stage28.OUTPUT_DIR / "daily_vkospi_only_stress.csv")
    assert not any("vix6" in column.lower() for column in stress.columns)

    no_ods = pd.read_csv(stage28.OUTPUT_DIR / "vkospi_only_no_ods_monthly.csv")
    ods = pd.read_csv(stage28.OUTPUT_DIR / "option_directional_surface_monthly.csv")
    assert np.allclose(no_ods["option_mu_adjustment_KODEX200"], 0.0)
    assert np.allclose(
        ods["option_mu_adjustment_KODEX200"],
        ods["option_direction_score"] * ods["option_return_scale"],
    )
    for asset in [asset for asset in ASSETS if asset != "KODEX200"]:
        assert np.allclose(
            ods[f"filtered_expected_mu_{asset}"],
            no_ods[f"filtered_expected_mu_{asset}"],
        )
    assert np.allclose(
        ods["filtered_expected_mu_KODEX200"]
        - no_ods["filtered_expected_mu_KODEX200"],
        ods["option_mu_adjustment_KODEX200"],
    )


def test_stage28_is_causal_long_only_unlevered_and_solver_clean() -> None:
    path = pd.read_csv(stage28.OUTPUT_DIR / "option_directional_surface_monthly.csv")
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert len(path) == 232
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-9).all().all()
    assert (weights <= 1.0 + 1e-9).all().all()
    assert np.allclose(path["downside_risk_aversion_lambda"], 1.0)
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-8).all()
    assert (path["cdar_slack"] >= -1e-8).all()
    report = json.loads(
        (stage28.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert all(report["checks"].values())


def test_stage28_performance_and_replacement_decision_are_reproduced() -> None:
    comparison = pd.read_csv(stage28.OUTPUT_DIR / "performance_comparison.csv")
    full = comparison.set_index(["Strategy", "Period"]).loc[
        ("Stage28_OptionDirectionalSurface", "full_2007_2026")
    ]
    locked = comparison.set_index(["Strategy", "Period"]).loc[
        ("Stage28_OptionDirectionalSurface", "locked_2018_2026")
    ]
    assert full["CAGR"] == pytest.approx(0.100328, abs=1e-6)
    assert full["Sharpe"] == pytest.approx(0.977942, abs=1e-6)
    assert full["MDD"] == pytest.approx(-0.180833, abs=1e-6)
    assert locked["CAGR"] == pytest.approx(0.174486, abs=1e-6)
    assert locked["Sharpe"] == pytest.approx(1.525486, abs=1e-6)
    assert locked["MDD"] == pytest.approx(-0.110361, abs=1e-6)

    decision = json.loads(
        (stage28.OUTPUT_DIR / "replacement_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["decision"] == "retain_stage20"
    assert not decision["full_period_pareto_replacements"]
    assert not decision["additional_feedback_assessment"][
        "signed_order_flow_available"
    ]
    assert not decision["additional_feedback_assessment"][
        "abnormal_surface_residual_implemented"
    ]


def test_two_axis_confirmation_implements_four_cell_table_without_tuned_cutoff() -> None:
    daily = pd.DataFrame(
        {
            "bear_pressure_fast": [-1.0, 1.0, 1.0, -1.0],
            "z_option_erp_fast": [1.0, -1.0, 1.0, -1.0],
            "bear_pressure_slow": [-1.0, 1.0, 1.0, -1.0],
            "z_option_erp_slow": [1.0, -1.0, 1.0, -1.0],
        }
    )
    output = stage29.apply_two_axis_confirmation(daily)
    assert output["two_axis_state_fast"].tolist() == [
        "bullish",
        "bearish",
        "fear_with_compensation",
        "uninformative",
    ]
    assert np.allclose(output["option_direction"], [1.0, -1.0, 0.0, 0.0])
    source = inspect.getsource(stage29.apply_two_axis_confirmation)
    assert "0.0" in source
    assert "quantile" not in source.lower()


def test_stage29_improves_stage28_drawdown_but_does_not_replace_stage20() -> None:
    comparison = pd.read_csv(stage29.OUTPUT_DIR / "performance_comparison.csv")
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    stage20 = full.loc["Stage20_VIX6Decomposition"]
    stage28_row = full.loc["Stage28_ODS_Difference"]
    stage29_row = full.loc["Stage29_OptionTwoAxisConfirmation"]
    assert stage29_row["MDD"] > stage28_row["MDD"]
    assert stage29_row["MDD"] < stage20["MDD"]
    assert stage29_row["Sharpe"] < stage20["Sharpe"]
    report = json.loads(
        (stage29.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert all(report["checks"].values())
