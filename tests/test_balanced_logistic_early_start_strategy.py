from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def report() -> dict:
    return json.loads(
        (RESULTS / "balanced_logistic_early_start_validation.json").read_text(
            encoding="utf-8"
        )
    )


def test_bridge_supplies_a_probability_from_the_first_strategy_month() -> None:
    factor = pd.read_csv(
        RESULTS / "balanced_logistic_early_start_factor.csv", index_col=0
    )
    assert factor.index[0] == "2007-04"
    assert factor.index[-1] == "2026-07"
    assert len(factor) == 232
    assert factor["p_tail_raw"].notna().all()
    assert factor["p_tail_raw"].between(0, 1).all()
    assert (factor["prediction_mode"] == "vkospi_pretrained_bridge").sum() == 38
    assert factor.loc["2007-04", "prediction_mode"] == "vkospi_pretrained_bridge"


def test_first_2007_prediction_uses_only_preexisting_labeled_history() -> None:
    summary = report()["implementation"]["fit_stats"]
    assert summary["first_prediction_month"] == "2007-04"
    assert summary["first_train_observations"] == 36
    assert summary["first_train_positive"] == 8
    assert summary["first_train_negative"] == 28
    assert summary["convergence_warning_count"] == 0

    factor = pd.read_csv(
        RESULTS / "balanced_logistic_early_start_factor.csv", index_col=0
    )
    bridge = factor.loc[factor["prediction_mode"].eq("vkospi_pretrained_bridge")]
    prediction_month = pd.PeriodIndex(bridge.index, freq="M")
    latest_training = pd.PeriodIndex(
        bridge["bridge_latest_training_month"], freq="M"
    )
    assert (latest_training <= prediction_month - 3).all()


def test_bridge_features_are_point_in_time() -> None:
    panel = pd.read_csv(
        RESULTS / "balanced_logistic_early_start_bridge_panel.csv",
        index_col=0,
        parse_dates=["feature_cutoff"],
    )
    target = pd.PeriodIndex(panel.index, freq="M")
    cutoff = pd.DatetimeIndex(panel["feature_cutoff"]).to_period("M")
    assert panel.index[0] == "2004-02"
    assert (cutoff < target).all()


def test_mature_model_is_bitwise_preserved_after_the_bridge() -> None:
    early = pd.read_csv(
        RESULTS / "balanced_logistic_early_start_factor.csv", index_col=0
    )
    legacy = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_factor.csv", index_col=0
    )
    mature = legacy["p_tail_raw"].notna()
    assert legacy.index[mature][0] == "2010-06"
    assert np.array_equal(
        early.loc[mature, "p_tail_raw"].to_numpy(),
        legacy.loc[mature, "p_tail_raw"].to_numpy(),
    )
    assert report()["causality_audit"][
        "maximum_mature_probability_difference_vs_legacy"
    ] == 0.0


def test_performance_delta_is_positive_but_economically_tiny() -> None:
    comparison = pd.read_csv(
        RESULTS / "balanced_logistic_early_start_comparison.csv"
    )
    full = comparison.loc[comparison["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )
    early = full.loc["EarlyStart_BalancedLogistic_RobustVKOSPI"]
    legacy = full.loc["Legacy2010_BalancedLogistic_RobustVKOSPI"]
    assert early["CAGR"] == pytest.approx(0.1564291168472658)
    assert early["Sharpe"] == pytest.approx(1.1327947719975777)
    assert early["MDD"] == pytest.approx(-0.12958799769553864)
    assert early["CAGR"] > legacy["CAGR"]
    assert early["Sharpe"] > legacy["Sharpe"]
    assert early["MDD"] == pytest.approx(legacy["MDD"], abs=1e-12)
    assert early["CAGR"] - legacy["CAGR"] < 0.0001


def test_small_gain_is_not_misrepresented_as_robust_evidence() -> None:
    result = report()
    assert result["promotion_gate"]["noninferior"] is True
    assert result["promotion_gate"]["strict_improvement"] is True
    assert result["promotion_gate"]["bootstrap_supports_promotion"] is False
    assert result["promotion_gate"]["passes"] is False
    assert result["calibration_bootstrap"]["probability_all_three_improve"] == pytest.approx(
        0.0612
    )
    assert result["implementation"]["bridge_active_tilt_months"] == [
        "2008-11",
        "2008-12",
    ]
    assert result["prediction"]["bridge_vs_proxy_tail"]["roc_auc"] < 0.5
    assert "very small" in result["promotion_gate"]["research_caveat"]


def test_early_start_daily_execution_remains_lagged_and_finite() -> None:
    daily = pd.read_csv(
        RESULTS / "balanced_logistic_early_start_final_daily.csv",
        parse_dates=["date", "signal_date"],
    )
    valid = daily["signal_date"].notna()
    assert (daily.loc[valid, "signal_date"] < daily.loc[valid, "date"]).all()
    assert np.isfinite(daily["return"]).all()
