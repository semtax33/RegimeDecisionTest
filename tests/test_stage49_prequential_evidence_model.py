from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage49_prequential_evidence_model import (
    prequential_evidence_model as stage49,
)


OUTPUT_DIR = stage49.OUTPUT_DIR
FULL_PATH = OUTPUT_DIR / "stage49_full_prequential_monthly.csv"
RETURN_HISTORY = OUTPUT_DIR / "return_prequential_history.csv"
VARIANCE_HISTORY = OUTPUT_DIR / "variance_prequential_history.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"


def test_return_stacking_solves_the_oos_convex_mse_problem() -> None:
    perfect_full = pd.DataFrame(
        {
            "actual_return": [0.1, -0.2, 0.3],
            "base_forecast": [0.0, 0.0, 0.0],
            "full_forecast": [0.1, -0.2, 0.3],
        }
    )
    assert np.isclose(
        stage49.prequential_return_stacking_weight(perfect_full)["weight"],
        1.0,
    )
    harmful_full = perfect_full.copy()
    harmful_full["full_forecast"] *= -1.0
    assert np.isclose(
        stage49.prequential_return_stacking_weight(harmful_full)["weight"],
        0.0,
    )


def test_variance_stacking_minimizes_prior_qlike() -> None:
    frame = pd.DataFrame(
        {
            "realized_variance": [2.0, 4.0, 3.0],
            "base_variance_forecast": [1.0, 2.0, 1.5],
            "full_variance_forecast": [2.0, 4.0, 3.0],
        }
    )
    frame["base_qlike"] = [
        stage49.qlike_loss(y, h)
        for y, h in zip(
            frame["realized_variance"], frame["base_variance_forecast"]
        )
    ]
    frame["full_qlike"] = 0.0
    weight = stage49.prequential_variance_stacking_weight(frame)["weight"]
    assert weight > 1.0 - 1e-6


def test_credit_and_technical_have_separate_nested_evidence_blocks() -> None:
    for asset in stage49.ASSETS:
        specs = dict(stage49._return_model_specs(asset))
        assert "credit_widening_z" in specs["credit"]
        assert "credit_widening_z" not in specs["stress_recovery"]
        assert f"technical_direction_{asset}" in specs["technical"]
        assert f"technical_direction_{asset}" not in specs["credit"]


def test_stage48_heuristics_and_double_risk_penalty_are_absent() -> None:
    source = Path(stage49.__file__).read_text(encoding="utf-8")
    forbidden = (
        "positive_part_signal_reliability",
        "cross_sectional_neutral",
        "credit_stress_multiplier",
        "support_lower",
        "support_upper",
        "downside_semivariance",
        "raw_slope * reliability",
    )
    assert all(token not in source for token in forbidden)
    assert source.count("scaling @ base_covariance @ scaling") == 1
    assert "0.5 * fit.prediction_std**2" in source


def test_saved_paths_are_causal_feasible_and_psd() -> None:
    path = pd.read_csv(FULL_PATH)
    path["month"] = pd.PeriodIndex(path["month"], freq="M")
    weights = path[stage49.WEIGHT_COLUMNS]
    assert np.isfinite(weights.to_numpy()).all()
    assert weights.min().min() >= -1e-8
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-8)
    assert path["base_covariance_min_eigenvalue"].min() > 0.0
    assert path["adjusted_covariance_min_eigenvalue"].min() > 0.0
    assert path["volatility_slack"].min() >= -1e-7
    assert path["cdar_slack"].min() >= -1e-7
    for signal in (
        "macro_signal_month",
        "stress_signal_month",
        "technical_signal_month",
        "fundamental_signal_month",
        "asset_vol_signal_month",
    ):
        assert (
            pd.PeriodIndex(path[signal], freq="M") < path["month"]
        ).all()
    assert (
        pd.to_datetime(path["covariance_cutoff"])
        <= pd.DatetimeIndex(
            [(month - 1).to_timestamp("M") for month in path["month"]]
        )
    ).all()


def test_each_credit_signal_is_direct_and_individually_prequential() -> None:
    path = pd.read_csv(FULL_PATH)
    history = pd.read_csv(RETURN_HISTORY)
    assert set(history["block"]) == set(stage49.RETURN_BLOCKS)
    for asset in stage49.ASSETS:
        assert f"credit_direct_contribution_{asset}" in path
        weight = path[f"return_credit_{asset}_evidence_weight"]
        assert weight.between(0.0, 1.0).all()
        unavailable = path[f"return_credit_{asset}_evidence_observations"].eq(
            0
        )
        assert weight.loc[unavailable].eq(0.0).all()


def test_variance_forecasts_are_positive_and_prequential() -> None:
    path = pd.read_csv(FULL_PATH)
    history = pd.read_csv(VARIANCE_HISTORY)
    assert len(history) == len(path) * len(stage49.ASSETS)
    for asset in stage49.ASSETS:
        multiplier = path[f"sensor_variance_multiplier_{asset}"]
        weight = path[f"variance_evidence_weight_{asset}"]
        assert np.isfinite(multiplier).all()
        assert multiplier.min() > 0.0
        assert weight.between(0.0, 1.0).all()


def test_validation_and_performance_reporting_are_honest() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert all(report["checks"].values())
    gate = report["gate"]
    assert gate["promotion_pass"] == all(
        value for key, value in gate.items() if key != "promotion_pass"
    )
    performance = pd.read_csv(OUTPUT_DIR / "performance_comparison.csv")
    expected = {
        "Stage36_GVZ_OVXAssetRisk",
        "Stage48_EvidenceCalibrated",
        *stage49.MODES,
    }
    assert set(performance["Strategy"]) == expected
    assert np.isfinite(
        performance[["CAGR", "Volatility", "Sharpe", "MDD"]].to_numpy()
    ).all()


def test_new_folder_contains_no_unused_stage48_report_artifacts() -> None:
    names = [path.name for path in OUTPUT_DIR.parent.rglob("*")]
    assert not any(name.endswith(".ipynb") for name in names)
    assert not any(name.endswith(".html") for name in names)
    assert not any(name.endswith(".png") for name in names)
    assert "asset_risk_predictive_regressions.csv" not in names
