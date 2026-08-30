from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage16_confirmed_crash_risk.confirmed_crash_risk_slsqp import (
    EVIDENCE_DEAD_ZONE,
    OUTPUT_DIR,
    run_research,
)


@pytest.fixture(scope="module")
def research() -> dict:
    return run_research(save=False)


def test_dead_zone_is_the_percentile_median_not_a_searched_cutoff() -> None:
    assert EVIDENCE_DEAD_ZONE == 0.5


def test_confirmed_signal_is_causal_and_normalization_turns_lambda_off(
    research: dict,
) -> None:
    path = research["confirmed_path"]
    assert (path["macro_signal_month"] < path.index).all()
    assert (path["stress_signal_month"] < path.index).all()
    normalization = path["normalization_state"]
    assert (path.loc[normalization, "crash_pressure"] == 0.0).all()
    assert (
        path.loc[normalization, "downside_risk_aversion_lambda"] == 1.0
    ).all()


def test_fast_slow_direction_macro_formula_has_no_fitted_weight(
    research: dict,
) -> None:
    signals = research["signals"]
    expected_fast_slow = np.minimum(
        signals["shock_component"], signals["persistence_component"]
    )
    expected_timing = signals[
        ["stress_direction_rank", "tail_component"]
    ].mean(axis=1)
    expected_evidence = pd.concat(
        [
            expected_fast_slow,
            expected_timing,
            signals["macro_vulnerability"],
        ],
        axis=1,
    ).mean(axis=1)
    assert np.allclose(signals["fast_slow_confirmation"], expected_fast_slow)
    assert np.allclose(
        signals["direction_tail_confirmation"], expected_timing
    )
    assert np.allclose(signals["crash_evidence"], expected_evidence)


def test_weights_are_fully_invested_unlevered_and_solvers_succeed(
    research: dict,
) -> None:
    path = research["confirmed_path"]
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-10).all().all()
    assert (weights <= 1.0 + 1e-10).all().all()
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-7).all()
    assert (path["cdar_slack"] >= -1e-7).all()


def test_lambda_activation_and_false_positives_are_reduced(research: dict) -> None:
    report = research["report"]
    direct = report["confirmed_direct_signal"]["full"]
    original = research["original_attribution"]
    confirmed = research["confirmed_attribution"]
    assert direct["lambda_activation_months"] == 119
    assert direct["maximum_lambda"] <= 2.0
    assert int(confirmed["risk_off_action"].sum()) < int(
        original["risk_off_action"].sum()
    )
    assert int(confirmed["false_positive"].sum()) < int(
        original["false_positive"].sum()
    )


def test_performance_tradeoff_is_recorded_without_cherry_picking(
    research: dict,
) -> None:
    delta = research["deltas"].set_index("Period")
    full = delta.loc["full_2007_2026"]
    locked = delta.loc["locked_2018_2026"]
    assert full["CAGR_Delta"] == pytest.approx(0.001734, abs=1e-6)
    assert full["Sharpe_Delta"] == pytest.approx(0.004582, abs=1e-6)
    assert full["MDD_Delta"] == pytest.approx(-0.002204, abs=1e-6)
    assert locked["CAGR_Delta"] == pytest.approx(0.001827, abs=1e-6)
    assert locked["MDD_Delta"] == pytest.approx(0.0, abs=1e-9)


def test_saved_validation_report_has_no_fitted_classifier_or_search() -> None:
    report = json.loads(
        (OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert all(report["checks"].values())
    assert report["parameter_policy"]["searched_parameters"] is None
    assert report["solver"]["fallbacks"] == 0

