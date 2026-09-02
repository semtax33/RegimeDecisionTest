from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage48_evidence_calibrated_stage36 import (
    evidence_calibrated_stage36 as stage48,
)


OUTPUT_DIR = stage48.OUTPUT_DIR
MONTHLY_PATH = OUTPUT_DIR / "stage48_monthly.csv"
PERFORMANCE_PATH = OUTPUT_DIR / "performance_comparison.csv"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"


def test_positive_part_reliability_uses_coefficient_uncertainty() -> None:
    assert stage48.positive_part_signal_reliability(1.0, 2.0) == 0.0
    assert np.isclose(
        stage48.positive_part_signal_reliability(1.0, 0.5), 0.75
    )
    assert np.isclose(
        stage48.positive_part_signal_reliability(-1.0, 0.5), 0.75
    )


def test_conflict_only_filter_has_a_true_neutral_signal() -> None:
    expected = np.array([0.03, 0.02, 0.01, 0.00])
    unchanged, detail = stage48.conflict_only_technical_filter(
        expected, np.zeros(4)
    )
    np.testing.assert_allclose(unchanged, expected)
    np.testing.assert_allclose(detail["technical_confidence"], 1.0)

    opposite = -np.sign(expected - expected.mean())
    filtered, _ = stage48.conflict_only_technical_filter(expected, opposite)
    np.testing.assert_allclose(filtered, expected.mean())


def test_variance_calibration_is_neutral_before_60_months() -> None:
    features = {
        asset: {"atr_state": 0.2, "credit_state": -0.1, "iv_state": 0.3}
        for asset in stage48.ASSETS
    }
    multipliers, details = stage48.calibrated_variance_multipliers(
        [], features
    )
    np.testing.assert_allclose(multipliers, 1.0)
    assert all(not details[asset]["active"] for asset in stage48.ASSETS)


def test_challenged_stage36_heuristics_are_not_in_runtime_source() -> None:
    source = Path(stage48.__file__).read_text(encoding="utf-8")
    assert "apply_technical_inputs" not in source
    assert "credit_stress_multiplier" not in source
    assert "gvz_gld_variance_multiplier" not in source
    assert "ovx_uso_variance_multiplier" not in source
    assert "raw_slope * reliability" not in source
    assert source.count("scaling @ base_covariance @ scaling") == 1


def test_saved_path_is_feasible_psd_and_all_signals_are_causal() -> None:
    path = pd.read_csv(MONTHLY_PATH)
    path["month"] = pd.PeriodIndex(path["month"], freq="M")
    weights = path[stage48.WEIGHT_COLUMNS]
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
    for model in ("variance", "credit"):
        for asset in stage48.ASSETS:
            column = f"{model}_calibration_last_training_month_{asset}"
            available = path[column].notna()
            assert (
                pd.PeriodIndex(path.loc[available, column], freq="M")
                < pd.PeriodIndex(path.loc[available, "month"], freq="M")
            ).all()


def test_credit_is_estimated_for_every_asset_and_variance_is_single_map() -> None:
    path = pd.read_csv(MONTHLY_PATH)
    for asset in stage48.ASSETS:
        assert f"credit_mu_adjustment_{asset}" in path
        assert f"variance_multiplier_{asset}" in path
        assert np.isfinite(path[f"variance_multiplier_{asset}"]).all()
        assert path[f"variance_multiplier_{asset}"].min() > 0.0
        inactive = path[
            f"variance_calibration_observations_{asset}"
        ].lt(stage48.MIN_CALIBRATION_MONTHS)
        assert path.loc[inactive, f"variance_multiplier_{asset}"].eq(1.0).all()


def test_report_is_honest_and_stage36_files_remained_frozen() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert all(report["checks"].values())
    assert report["gate"]["causal_and_feasible"]
    assert report["gate"]["promotion_pass"] == all(
        value
        for key, value in report["gate"].items()
        if key != "promotion_pass"
    )
    performance = pd.read_csv(PERFORMANCE_PATH)
    assert set(performance["Strategy"]) == {
        "Stage36_GVZ_OVXAssetRisk",
        "Stage48_EvidenceCalibrated",
    }
    assert np.isfinite(
        performance[["CAGR", "Volatility", "Sharpe", "MDD"]].to_numpy()
    ).all()


def test_new_strategy_folder_omits_stage36_research_artifacts() -> None:
    files = [path.name for path in stage48.OUTPUT_DIR.parent.rglob("*")]
    assert not any(name.endswith(".ipynb") for name in files)
    assert not any(name.endswith(".html") for name in files)
    assert not any(name.endswith(".png") for name in files)
    assert "asset_risk_predictive_regressions.csv" not in files
