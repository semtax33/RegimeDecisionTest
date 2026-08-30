from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.stage15_hyg_substitution.hyg_substitution import (
    ASSETS,
    FOREIGN_ASSETS,
    HYG_DAILY_CACHE,
    OUTPUT_DIR,
    project_to_long_only_simplex,
    run_research,
)


@pytest.fixture(scope="module")
def research() -> dict:
    return run_research(save=False, refresh_hyg=False)


def test_new_universe_replaces_bond_with_hyg() -> None:
    assert ASSETS == ["KODEX200", "HYG", "GLD", "USO"]
    assert "BOND" not in ASSETS
    assert "HYG" in FOREIGN_ASSETS


def test_hyg_snapshot_is_local_and_covers_the_test() -> None:
    snapshot = pd.read_csv(HYG_DAILY_CACHE, parse_dates=["date"])
    assert len(snapshot) > 4_800
    assert snapshot["date"].min() == pd.Timestamp("2007-04-11")
    assert snapshot["date"].max() >= pd.Timestamp("2026-08-01")
    assert snapshot[["open", "close"]].notna().all().all()


def test_simplex_has_no_majority_cap() -> None:
    projected = project_to_long_only_simplex(np.array([4.0, 0.0, 0.0, 0.0]))
    assert np.allclose(projected, [1.0, 0.0, 0.0, 0.0])


def test_dynamic_path_is_causal_unlevered_and_feasible(research: dict) -> None:
    path = research["dynamic_path"]
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-10).all().all()
    assert (weights <= 1.0 + 1e-10).all().all()
    assert (path["macro_signal_month"] < path.index).all()
    assert (path["stress_signal_month"] < path.index).all()
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-7).all()
    assert (path["cdar_slack"] >= -1e-7).all()


def test_only_infeasible_crisis_month_relaxes_vol_guard(research: dict) -> None:
    path = research["dynamic_path"]
    relaxed = path[path["volatility_cap_relaxed"]]
    assert relaxed.index.tolist() == [pd.Period("2008-12", freq="M")]
    row = relaxed.iloc[0]
    assert row["minimum_feasible_annual_volatility"] > 0.13
    assert row["effective_annual_volatility_cap"] < 0.131


def test_performance_tradeoff_matches_saved_result(research: dict) -> None:
    delta = research["deltas"].set_index(["Candidate", "Period"])
    execution = delta.loc[("Stage15_HYG_ExecutionOnly", "common_full")]
    optimized = delta.loc[("Stage15_HYG_DynamicLambda", "common_full")]
    assert execution["CAGR_Delta"] == pytest.approx(0.008689, abs=1e-6)
    assert execution["Sharpe_Delta"] == pytest.approx(0.021333, abs=1e-6)
    assert execution["MDD_Delta"] < 0
    assert optimized["CAGR_Delta"] == pytest.approx(-0.005086, abs=1e-6)
    assert optimized["Sharpe_Delta"] == pytest.approx(0.053603, abs=1e-6)
    assert optimized["MDD_Delta"] == pytest.approx(0.014725, abs=1e-6)


def test_saved_report_checks_are_all_true() -> None:
    report_path = OUTPUT_DIR / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert all(report["checks"].values())
    assert report["solver"]["fallbacks"] == 0
    assert report["solver"]["volatility_cap_relaxation_months"] == 1
    assert HYG_DAILY_CACHE.is_file()

