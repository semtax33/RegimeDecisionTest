from __future__ import annotations

import json

import numpy as np
import pandas as pd

from strategies.core.regime_research import ASSETS
from strategies.stage10_vix6_router.vix6_conditional_router_strategy import (
    CALIBRATION_PATH,
    DAILY_PATH,
    OPTION_BY_STATE,
    RECONCILED_PATH,
    REPORT_PATH,
    STATES,
)


def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_six_states_and_option_mapping_are_complete() -> None:
    daily = pd.read_csv(DAILY_PATH)
    assert set(daily["state"]) == set(STATES)
    assert set(OPTION_BY_STATE) == set(STATES)
    assert report()["option_mapping"] == OPTION_BY_STATE
    assert all(report()["state_counts"][state] > 0 for state in STATES)


def test_daily_signal_strictly_precedes_action() -> None:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date", "signal_date"])
    valid = daily["signal_date"].notna()
    assert (daily.loc[valid, "signal_date"] < daily.loc[valid, "date"]).all()
    assert report()["lookahead_audit"]["signal_strictly_before_action"] is True


def test_four_asset_weights_are_finite_nonnegative_and_only_expected_assets() -> None:
    daily = pd.read_csv(DAILY_PATH)
    expected = {f"w_{asset}" for asset in ASSETS}
    actual = {column for column in daily if column.startswith("w_")}
    assert actual == expected
    weights = daily[sorted(expected)]
    assert np.isfinite(weights).all().all()
    assert weights.ge(-1e-12).all().all()


def test_candidate_grid_is_prelock_only_and_safe_fallback_is_selected() -> None:
    calibration = pd.read_csv(CALIBRATION_PATH)
    selection = report()["selection"]
    assert len(calibration) == 24
    assert selection["candidate_count"] == 24
    assert selection["strict_count"] == int(calibration["StrictAllThree"].sum()) == 0
    assert selection["retention_count"] == int(calibration["RetentionGate"].sum()) == 0
    assert selection["uses_locked_metrics"] is False
    assert selection["rule"].startswith("safe fallback")


def test_selected_router_preserves_existing_robust_vkospi_path() -> None:
    selected = pd.read_csv(RECONCILED_PATH, index_col=0)
    baseline = pd.read_csv(
        RECONCILED_PATH.parent / "balanced_logistic_no_sjm_final_reconciled.csv",
        index_col=0,
    )
    common = selected.index.intersection(baseline.index)
    difference = selected.loc[common, "return"] - baseline.loc[common, "return"]
    assert float(difference.abs().max()) < 1e-12
