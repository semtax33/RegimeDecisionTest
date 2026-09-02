from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage50_explainable_regime_allocator import (
    explainable_regime_allocator as strategy,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = ROOT / "strategies" / "stage50_explainable_regime_allocator"
OUTPUT_DIR = STRATEGY_DIR / "outputs"
MONTHLY_PATH = OUTPUT_DIR / "monthly_results.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation_report.json"
ABLATION_REPORT_PATH = OUTPUT_DIR / "ablation_validation_report.json"
ASSETS = ("KODEX200", "BOND", "GLD", "USO")
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_production_has_one_standalone_execution_path() -> None:
    source_path = STRATEGY_DIR / "explainable_regime_allocator.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    strategy_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("strategies.")
    ]

    assert strategy_imports == []
    assert "MODE_" not in source
    assert "MacroBayes" not in source
    assert "layer_ablation_validation" not in source
    assert not any("macro_bayes" in path.name.lower() for path in STRATEGY_DIR.rglob("*"))


def test_saved_production_path_is_causal_feasible_and_finite() -> None:
    path = pd.read_csv(MONTHLY_PATH)
    month = pd.PeriodIndex(path["month"], freq="M")
    signal_month = pd.PeriodIndex(path["macro_signal_month"], freq="M")
    covariance_cutoff = pd.to_datetime(path["covariance_cutoff"])
    prior_month_end = pd.DatetimeIndex(
        [(period - 1).to_timestamp("M") for period in month]
    )

    assert len(path) == 232
    assert np.isfinite(path[["return", "gross_return", *WEIGHT_COLUMNS]].to_numpy()).all()
    assert path[WEIGHT_COLUMNS].min().min() >= -1e-9
    np.testing.assert_allclose(path[WEIGHT_COLUMNS].sum(axis=1), 1.0, atol=1e-8)
    assert (signal_month < month).all()
    assert (covariance_cutoff <= prior_month_end).all()
    assert path["covariance_min_eigenvalue"].min() > 0.0
    assert path["volatility_slack"].min() >= -1e-7
    assert path["cdar_slack"].min() >= -1e-7


def test_refactor_preserves_the_frozen_stage49_macro_forecast() -> None:
    # The comparison file is evidence only.  The production module above neither
    # imports nor resolves this older Stage path.
    old = pd.read_csv(
        ROOT
        / "strategies"
        / "stage49_prequential_evidence_model"
        / "outputs"
        / "stage49_macro_bayes_monthly.csv"
    )
    new = pd.read_csv(MONTHLY_PATH)
    assert old["month"].tolist() == new["month"].tolist()

    for asset in ASSETS:
        np.testing.assert_allclose(
            new[f"expected_return_{asset}"],
            old[f"expected_mu_{asset}"],
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(new["return"], old["return"], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(new[WEIGHT_COLUMNS], old[WEIGHT_COLUMNS], rtol=0.0, atol=5e-6)


def test_production_validation_and_performance_are_complete() -> None:
    report = _json(VALIDATION_PATH)
    assert report["strategy"] == "Stage50_ExplainableRegimeAllocator"
    assert report["all_checks_pass"]
    assert all(report["checks"].values())
    assert report["months"] == 232
    assert not report["known_statistical_limitations"][
        "regime_scores_are_calibrated_probabilities"
    ]
    assert not report["known_statistical_limitations"][
        "macro_point_in_time_vintages_verified"
    ]

    performance = pd.read_csv(OUTPUT_DIR / "performance.csv")
    assert set(performance["Period"]) == {"full_common", "locked_2018_2026"}
    assert np.isfinite(
        performance[["CAGR", "Volatility", "Sharpe", "MDD"]].to_numpy()
    ).all()


def test_six_one_layer_ablations_and_stress_scope_are_frozen() -> None:
    expected = {
        "AllLayersReference",
        "NoStressExpectedReturn",
        "NoStressCovarianceBlend",
        "NoTechnicalConfidence",
        "NoATRCovarianceScaling",
        "NoCreditStressRiskScaling",
        "NoGVZOVXVarianceScaling",
        "StressKODEX200Only",
    }
    performance = pd.read_csv(OUTPUT_DIR / "layer_ablation_performance.csv")
    assert set(performance["Strategy"]) == expected
    assert set(performance["Period"]) == {"full_2007_2026", "locked_2018_2026"}
    assert np.isfinite(
        performance[["CAGR", "Volatility", "Sharpe", "MDD"]].to_numpy()
    ).all()

    report = _json(ABLATION_REPORT_PATH)
    assert report["all_checks_pass"]
    assert all(report["checks"].values())
    assert report["reference_reproduction"]["months"] == 232
    assert report["reference_reproduction"]["max_return_absolute_error"] < 1e-8
    assert report["reference_reproduction"]["max_weight_absolute_error"] < 1e-7


def test_rank_zscore_diagnostic_reports_separate_return_and_risk_losses() -> None:
    losses = pd.read_csv(OUTPUT_DIR / "credit_encoding_oos_losses.csv")
    assert set(losses["encoding"]) == {"rank", "zscore"}
    assert set(losses["role"]) == {
        "next_month_return",
        "next_month_log_squared_return",
    }
    assert set(losses["asset"]) == set(ASSETS)
    assert len(losses) == 16
    assert (losses["observations"] > 0).all()
    assert np.isfinite(losses["mse"]).all()

    policies = pd.read_csv(OUTPUT_DIR / "credit_encoding_policy_comparison.csv")
    assert set(policies["policy"]) == {
        "rank_only",
        "zscore_only",
        "current_mixed_z_return_rank_risk",
    }
    best = policies.loc[policies["mean_relative_mse"].idxmin(), "policy"]
    assert best == "rank_only"


def test_regime_beta_stability_reports_sign_and_magnitude_diagnostics() -> None:
    stability = pd.read_csv(OUTPUT_DIR / "regime_beta_stability.csv")
    assert len(stability) == 32
    assert set(stability["asset"]) == set(ASSETS)
    assert set(stability["beta_kind"]) == {"stress", "recovery"}
    assert int((stability["nonzero_sign_switches"] > 0).sum()) == 14
    assert int((stability["p90_to_median_magnitude"] > 3.0).sum()) == 28

    report = _json(ABLATION_REPORT_PATH)["beta_instability_flags"]
    assert report["series_total"] == 32
    assert report["series_with_nonzero_sign_switches"] == 14
    assert report["unconstrained_bond_gold_series_with_sign_switches"] == 14
    assert report["unconstrained_bond_gold_series_total"] == 16
