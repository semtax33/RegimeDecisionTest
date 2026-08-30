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
        (RESULTS / "balanced_logistic_no_sjm_validation.json").read_text(
            encoding="utf-8"
        )
    )


def test_requested_model_configuration_is_explicit() -> None:
    implementation = report()["implementation"]
    assert implementation["macro"]["sjm_weight"] == 0.0
    logistic = implementation["logistic"]
    assert logistic["class_weight"] == "balanced"
    assert logistic["penalty"] == "l2"
    assert logistic["solver"] == "liblinear"
    assert logistic["C"] == 0.1
    assert logistic["embargo_months"] == 2
    assert len(logistic["features"]) == 16


def test_deployed_reproduction_proves_only_sjm_path_changed() -> None:
    audit = report()["deployed_reproduction_audit"]
    assert audit["feature_observations"] == 232
    assert audit["max_absolute_domestic_feature_difference"] < 1e-12
    assert audit["label_disagreements"] == 0
    assert audit["max_absolute_probability_difference"] < 1e-12
    assert audit["max_absolute_medium_return_difference"] < 1e-12
    assert audit["max_absolute_final_return_difference"] < 1e-12


def test_signal_and_daily_execution_boundaries_are_causal() -> None:
    signals = pd.read_csv(RESULTS / "balanced_logistic_no_sjm_signals.csv", index_col=0)
    target = pd.PeriodIndex(signals.index, freq="M")
    known = pd.PeriodIndex(signals["signal_month"], freq="M")
    assert (known < target).all()

    daily = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_daily.csv",
        parse_dates=["date", "signal_date"],
    )
    valid = daily["signal_date"].notna()
    assert (daily.loc[valid, "signal_date"] < daily.loc[valid, "date"]).all()
    assert daily["stress"].between(0, 1).all()


def test_locked_results_and_bootstrap_are_preserved() -> None:
    locked = report()["locked_final"]
    variant = locked["variant"]
    deployed = locked["deployed"]
    assert variant["CAGR"] == pytest.approx(0.20907823332988862)
    assert variant["Sharpe"] == pytest.approx(1.4969186464577038)
    assert variant["MDD"] == pytest.approx(-0.09760421330851354)
    assert variant["CAGR"] > deployed["CAGR"]
    assert variant["Sharpe"] > deployed["Sharpe"]
    assert variant["MDD"] > deployed["MDD"]
    assert locked["bootstrap"]["probability_all_three_improve"] == pytest.approx(
        0.6486
    )


def test_all_variant_outputs_are_finite_and_aligned() -> None:
    factor = pd.read_csv(RESULTS / "balanced_logistic_no_sjm_factor.csv")
    predicted = factor["p_tail_raw"].dropna()
    assert len(predicted) == 194
    assert predicted.between(0, 1).all()

    medium = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_medium_backtest.csv", index_col=0
    )
    final = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv", index_col=0
    )
    assert len(medium) == 232
    assert len(final) == 232
    assert np.isfinite(medium["return"]).all()
    assert np.isfinite(final["return"]).all()


def test_prelock_gate_is_not_misrepresented_as_passed() -> None:
    comparison = pd.read_csv(RESULTS / "balanced_logistic_no_sjm_comparison.csv")
    view = comparison.loc[
        comparison["Strategy"].isin(
            [
                "NoSJM_BalancedLogistic_RobustVKOSPI",
                "Deployed_SJM10_BalancedLogistic_RobustVKOSPI",
            ]
        )
    ]
    calibration = view.loc[view["Period"].eq("calibration_2007_2017")].set_index(
        "Strategy"
    )
    variant = calibration.loc["NoSJM_BalancedLogistic_RobustVKOSPI"]
    deployed = calibration.loc[
        "Deployed_SJM10_BalancedLogistic_RobustVKOSPI"
    ]
    assert variant["CAGR"] < deployed["CAGR"]
    assert variant["Sharpe"] < deployed["Sharpe"]
