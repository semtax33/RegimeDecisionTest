from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage30_abnormal_surface_erp import (
    abnormal_surface_erp_slsqp as stage30,
)


def _read_output(name: str, period_index: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(stage30.OUTPUT_DIR / name, index_col=0)
    if period_index:
        frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def test_expanding_residual_is_strictly_causal() -> None:
    index = pd.date_range("2000-01-01", periods=16, freq="D")
    predictors = pd.DataFrame(
        {"x1": np.arange(16, dtype=float), "x2": np.sin(np.arange(16))},
        index=index,
    )
    target = pd.Series(1.0 + 0.2 * predictors["x1"], index=index)
    original = stage30.expanding_one_step_ols_residual(
        target, predictors, "test", min_history=5
    )
    altered_target = target.copy()
    altered_predictors = predictors.copy()
    altered_target.iloc[12:] += 1_000.0
    altered_predictors.iloc[12:, :] *= -500.0
    altered = stage30.expanding_one_step_ols_residual(
        altered_target, altered_predictors, "test", min_history=5
    )
    assert np.allclose(
        original.loc[: index[11], "residual_test"],
        altered.loc[: index[11], "residual_test"],
        equal_nan=True,
    )
    assert original.loc[index[5], "test_training_observations"] == 5


def test_order_flow_is_discarded_and_erp_is_diagnostic_only() -> None:
    surface_source = inspect.getsource(stage30._surface_for_expiry_without_order_flow)
    assert "trading_value" not in surface_source
    assert "volume" not in surface_source
    report = json.loads(
        (stage30.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    option_audit = report["data_audit"]["option"]
    assert option_audit["order_flow_used"] is False
    assert option_audit["erp_used_for_allocation"] is False
    assert report["directional_formula"]["order_flow"] == "excluded"
    assert "diagnostic only" in report["directional_formula"]["implied_erp"]


def test_saved_pure_direction_and_quality_formula_are_exact() -> None:
    daily = _read_output("daily_abnormal_surface_erp_features.csv")
    valid = daily.dropna(subset=["option_direction_score"])
    assert len(daily) == 4_995
    assert np.allclose(
        valid["abnormal_bear_pressure_fast"],
        valid[["z_residual_put_skew_fast", "z_residual_iva_fast"]].mean(axis=1),
    )
    assert np.allclose(
        valid["abnormal_bear_pressure_slow"],
        valid[["z_residual_put_skew_slow", "z_residual_iva_slow"]].mean(axis=1),
    )
    assert np.allclose(
        valid["pure_direction_raw"],
        -valid[
            ["abnormal_bear_pressure_fast", "abnormal_bear_pressure_slow"]
        ].mean(axis=1),
    )
    quality = valid[
        ["q_dte", "q_coverage", "q_parity", "q_arbitrage", "q_roll"]
    ].prod(axis=1)
    assert np.allclose(valid["data_quality_confidence"], quality)
    assert np.allclose(
        valid["option_direction"], valid["pure_direction_raw"] * quality
    )
    assert np.allclose(
        valid["option_direction_score"],
        valid["option_direction"] / (1.0 + valid["option_direction"].abs()),
    )


def test_rate_aware_parity_and_roll_audit_are_saved() -> None:
    daily = _read_output("daily_abnormal_surface_erp_features.csv")
    assert daily["discount_factor"].gt(0.0).all()
    assert not np.allclose(daily["discount_factor"], 1.0)
    assert daily["parity_nrmse"].ge(0.0).all()
    assert set(daily["q_roll"].dropna().unique()).issubset({0.0, 1.0})
    construction = pd.read_csv(
        stage30.OUTPUT_DIR / "option_construction_diagnostics.csv"
    )
    assert {"DTE_bucket_IC", "recent_roll_IC", "data_quality_period_mean"}.issubset(
        construction["Audit"]
    )
    assert {"2007_2017", "2018_2026"}.issubset(construction["Group"])


def test_causal_mu_calibration_never_reverses_signal_meaning() -> None:
    signals = _read_output("monthly_option_alpha_signals.csv", period_index=True)
    assert (signals["causal_calibration_slope"] >= 0.0).all()
    assert np.allclose(
        signals["causal_calibration_slope"],
        signals["causal_calibration_raw_slope"].clip(lower=0.0),
    )
    assert np.allclose(
        signals["calibrated_mu_adjustment_KODEX200"],
        signals["causal_calibration_slope"] * signals["option_direction_score"],
    )
    assert (
        signals["calibration_observations"]
        == np.arange(
            signals["calibration_observations"].iloc[0],
            signals["calibration_observations"].iloc[0] + len(signals),
        )
    ).all()


def test_stage30_retains_stage20_risk_and_changes_only_equity_mu() -> None:
    baseline = _read_output("stage20_vix6_monthly.csv", period_index=True)
    candidate = _read_output(
        "stage30_pureods_qualitycausal_monthly.csv", period_index=True
    )
    common = baseline.index.intersection(candidate.index)
    for asset in [asset for asset in ASSETS if asset != "KODEX200"]:
        assert np.allclose(
            baseline.loc[common, f"filtered_expected_mu_{asset}"],
            candidate.loc[common, f"filtered_expected_mu_{asset}"],
        )
    assert np.allclose(
        candidate.loc[common, "filtered_expected_mu_KODEX200"]
        - baseline.loc[common, "filtered_expected_mu_KODEX200"],
        candidate.loc[common, "option_mu_adjustment_KODEX200"],
    )
    report = json.loads(
        (stage30.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["unchanged_controls"]["risk_engine"] == (
        "Stage20 VKOSPI/VIX6 decomposition"
    )


def test_stage30_is_long_only_unlevered_and_solver_clean() -> None:
    path = _read_output("stage30_pureods_qualitycausal_monthly.csv")
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
        (stage30.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert all(report["checks"].values())


def test_stage30_performance_and_honest_decision_are_reproduced() -> None:
    comparison = pd.read_csv(stage30.OUTPUT_DIR / "performance_comparison.csv")
    table = comparison.set_index(["Strategy", "Period"])
    full = table.loc[("Stage30_PureODS_QualityCausal", "full_2007_2026")]
    early = table.loc[("Stage30_PureODS_QualityCausal", "early_2007_2017")]
    locked = table.loc[("Stage30_PureODS_QualityCausal", "locked_2018_2026")]
    assert full["CAGR"] == pytest.approx(0.0978195, abs=1e-7)
    assert full["Sharpe"] == pytest.approx(0.9795673, abs=1e-7)
    assert full["MDD"] == pytest.approx(-0.1359901, abs=1e-7)
    assert early["CAGR"] == pytest.approx(0.0652130, abs=1e-7)
    assert locked["CAGR"] == pytest.approx(0.1400682, abs=1e-7)
    report = json.loads(
        (stage30.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["decision"] == "retain_stage20"
    assert report["full_changes_vs_stage20"]["cagr"] > 0.0
    assert report["full_changes_vs_stage20"]["mdd"] > 0.0
    assert report["full_changes_vs_stage20"]["sharpe"] < 0.0


def test_quality_layer_materially_controls_the_no_quality_ablation() -> None:
    comparison = pd.read_csv(stage30.OUTPUT_DIR / "performance_comparison.csv")
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    no_quality = full.loc["Stage30_PureODS_NoQuality"]
    quality = full.loc["Stage30_PureODS_QualityCausal"]
    assert quality["CAGR"] > no_quality["CAGR"]
    assert quality["Sharpe"] > no_quality["Sharpe"]
    assert quality["MDD"] > no_quality["MDD"]
