from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage45_volatility_targeted_shrinkage_mlp import (
    volatility_targeted_shrinkage_mlp as strategy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "strategies"
    / "stage45_volatility_targeted_shrinkage_mlp"
    / "outputs"
)


def _report() -> dict:
    return json.loads((OUTPUT / "validation_report.json").read_text("utf-8"))


def _path() -> pd.DataFrame:
    frame = pd.read_csv(OUTPUT / "stage45_monthly.csv", index_col="month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def test_constant_correlation_shrinkage_is_psd_and_retains_variances() -> None:
    rng = np.random.default_rng(20260730)
    values = rng.normal(size=(252, len(strategy.ASSETS)))
    covariance, shrinkage = strategy.ledoit_wolf_constant_correlation(values)
    centered = values - values.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / len(values)

    assert 0.0 <= shrinkage <= 1.0
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert np.allclose(
        np.diag(covariance),
        np.diag(sample) + strategy.NUMERICAL_EPSILON,
        atol=1e-12,
    )


def test_port_removes_cross_sectional_stock_selection_logic() -> None:
    source = inspect.getsource(strategy.build_asset_month_panel)
    assert ".rank(" not in source
    assert "rv.mean" not in source
    assert "target_excess_return" in source
    assert "returns.loc[month, asset] - cash_return" in source
    assert strategy.MIN_TRAIN_MONTHS == 36
    assert strategy.HIDDEN_UNITS == 4


def test_saved_path_is_causal_feasible_and_reproduces_warmup() -> None:
    report = _report()
    path = _path()
    active = path.loc[path["model_active"]]

    assert report["causality_audit"]["all_model_cutoffs_before_target"]
    assert report["causality_audit"]["all_covariance_dates_before_target"]
    assert report["causality_audit"][
        "all_covariance_windows_have_252_rows"
    ]
    assert report["checks"]["all_active_solvers_feasible"]
    assert report["checks"]["long_only_no_leverage"]
    assert report["checks"]["warmup_reproduces_stage36"]
    assert report["checks"]["stage36_frozen_files_unchanged"]

    weights = active[strategy.WEIGHT_COLUMNS]
    assert (weights.to_numpy() >= -1e-10).all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-8).all()
    assert np.allclose(
        weights.sum(axis=1) + active["cash_weight"], 1.0, atol=1e-8
    )
    assert active["volatility_slack"].min() >= -1e-7
    assert active["cdar_slack"].min() >= -1e-7


def test_performance_is_reported_honestly_and_stage36_is_retained() -> None:
    report = _report()
    performance = pd.read_csv(OUTPUT / "performance_comparison.csv").set_index(
        ["Strategy", "Period"]
    )
    baseline = performance.loc[
        ("Stage36_GVZ_OVXAssetRisk", "mlp_active")
    ]
    candidate = performance.loc[
        ("Stage45_TinyMLP_LW_VolTarget", "mlp_active")
    ]

    assert report["promote"] is False
    assert report["decision"] == (
        "retain_stage36_and_keep_stage45_as_research_only"
    )
    assert candidate["CAGR"] < baseline["CAGR"]
    assert candidate["Sharpe"] < baseline["Sharpe"]
    assert candidate["MDD"] < baseline["MDD"]
    assert candidate["AvgTurnover"] > baseline["AvgTurnover"]
    assert report["forecast_diagnostics"]["asset_month_predictions"] == 748


def test_all_documented_outputs_exist() -> None:
    expected = {
        "stage45_monthly.csv",
        "oos_forecasts.csv",
        "performance_comparison.csv",
        "paired_block_bootstrap_vs_stage36.csv",
        "validation_report.json",
    }
    assert expected.issubset({path.name for path in OUTPUT.iterdir()})
