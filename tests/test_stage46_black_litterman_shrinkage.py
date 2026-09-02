from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage46_black_litterman_shrinkage import (
    black_litterman_shrinkage_slsqp as strategy,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "strategies" / "stage46_black_litterman_shrinkage" / "outputs"


def _report() -> dict:
    return json.loads((OUTPUT / "validation_report.json").read_text("utf-8"))


def _performance() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "performance_comparison.csv").set_index(
        ["Strategy", "Period"]
    )


def test_regime_confidence_has_exact_endpoints() -> None:
    columns = strategy.stage36.stage35.REGIME_COLUMNS
    uniform = pd.Series(dict(zip(columns, [0.25] * 4)))
    certain = pd.Series(dict(zip(columns, [1.0, 0.0, 0.0, 0.0])))
    assert strategy.regime_confidence(uniform) == 0.0
    assert strategy.regime_confidence(certain) == 1.0


def test_black_litterman_zero_confidence_equals_equilibrium() -> None:
    covariance = np.diag([0.01, 0.02, 0.03, 0.04])
    weights = np.repeat(0.25, 4)
    view = np.array([0.03, -0.01, 0.02, 0.00])
    posterior, detail = strategy.black_litterman_posterior(
        covariance, weights, view, 0.0, 2.0, 60
    )
    assert np.allclose(posterior, detail["equilibrium_return"])
    assert np.isinf(detail["omega_diagonal"]).all()


def test_guarded_bl_satisfies_all_requested_performance_gates() -> None:
    report = _report()
    result = report["gate_results"][strategy.BL_NAME]
    assert result["pass"] is True
    assert all(result["gates"].values())
    assert report["promoted_strategy"] == strategy.BL_NAME
    assert report["checks"]["guarded_bl_passes_every_performance_gate"]


def test_sharpe_mdd_cagr_and_turnover_hold_in_every_period() -> None:
    performance = _performance()
    for period in (
        "full_2007_2026",
        "common_2010_2026",
        "locked_2018_2026",
    ):
        baseline = performance.loc[(strategy.BASELINE_NAME, period)]
        candidate = performance.loc[(strategy.BL_NAME, period)]
        assert candidate["Sharpe"] >= baseline["Sharpe"] - 1e-12
        assert candidate["MDD"] >= baseline["MDD"] - 1e-12
        assert candidate["CAGR"] >= baseline["CAGR"] - strategy.CAGR_TOLERANCE
        assert candidate["AvgTurnover"] <= (
            baseline["AvgTurnover"] * (1.0 + strategy.TURNOVER_TOLERANCE)
            + 1e-10
        )


def test_bootstrap_probabilities_are_checked_and_majority_positive() -> None:
    bootstrap = pd.read_csv(
        OUTPUT / "paired_block_bootstrap_vs_stage36.csv"
    )
    guarded = bootstrap.loc[bootstrap["Candidate"].eq(strategy.BL_NAME)]
    assert set(guarded["Period"]) == {
        "full_2007_2026",
        "common_2010_2026",
        "locked_2018_2026",
    }
    for metric in ("delta_Sharpe", "delta_MDD"):
        values = guarded.loc[
            guarded["Metric"].eq(metric), "ProbabilityPositive"
        ]
        assert len(values) == 3
        assert (values >= 0.50).all()


def test_monthly_path_is_causal_long_only_and_guarded() -> None:
    report = _report()
    path = pd.read_csv(
        OUTPUT / "stage46_guarded_blacklitterman_lw_monthly.csv",
        index_col="month",
    )
    weights = path[strategy.WEIGHT_COLUMNS]
    assert (weights.to_numpy() >= -1e-10).all()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
    assert path["applied_bl_tilt"].max() <= 0.05 + 1e-12
    assert path["drawdown_multiplier"].between(0.0, 1.0).all()
    assert int(path["lw_overlay_veto"].sum()) == 20
    assert report["causality_audit"][strategy.BL_NAME][
        "all_covariance_dates_before_target"
    ]
    assert report["causality_audit"][strategy.BL_NAME][
        "all_covariance_windows_have_252_rows"
    ]
    assert report["checks"]["all_candidate_solvers_feasible"]
    assert report["checks"][
        "stage36_and_stage45_frozen_files_unchanged"
    ]


def test_only_three_ablation_paths_are_reported() -> None:
    performance = pd.read_csv(OUTPUT / "performance_comparison.csv")
    assert set(performance["Strategy"]) == {
        strategy.BASELINE_NAME,
        strategy.LW_NAME,
        strategy.BL_NAME,
    }
    expected = {
        "stage46_guarded_blacklitterman_lw_monthly.csv",
        "stage46_stage36mu_lw_monthly.csv",
        "stage46_blacklitterman_lw_shadow_monthly.csv",
        "performance_comparison.csv",
        "paired_block_bootstrap_vs_stage36.csv",
        "validation_report.json",
    }
    assert expected == {path.name for path in OUTPUT.iterdir() if path.is_file()}
