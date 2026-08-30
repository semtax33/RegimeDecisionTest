from __future__ import annotations

import json

import numpy as np
import pandas as pd

from strategies.stage10_vix6_router.option_structure_overlay import (
    COMPARISON_PATH,
    PREMIUM_BUDGET,
    REPORT_PATH,
    SELECTED_PATH,
    TRADES_PATH,
)
from strategies.stage10_vix6_router.vix6_conditional_router_strategy import (
    RECONCILED_PATH,
)


def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def trades() -> pd.DataFrame:
    return pd.read_csv(
        TRADES_PATH,
        parse_dates=["entry_date", "entry_signal_date", "exit_date", "exit_signal_date"],
    )


def test_option_trade_schedule_is_causal_and_structures_are_bounded() -> None:
    frame = trades()
    assert len(frame) >= 10
    assert set(frame["structure"]) == {"PutSpread", "CallSpread", "CoveredCall"}
    assert (frame["entry_signal_date"] < frame["entry_date"]).all()
    assert (frame["exit_signal_date"] < frame["exit_date"]).all()
    assert (frame["entry_date"] < frame["exit_date"]).all()
    assert frame["leg1_entry_dte"].between(30, 60).all()


def test_debit_spreads_respect_premium_and_max_loss_budgets() -> None:
    frame = trades()
    spreads = frame[frame["structure"].isin(["PutSpread", "CallSpread"])]
    expected = spreads["state"].map(PREMIUM_BUDGET)
    assert np.allclose(spreads["premium_budget"], expected)
    assert np.allclose(spreads["max_loss_nav"], spreads["premium_budget"])
    assert spreads["premium_budget"].le(0.0075 + 1e-12).all()
    assert spreads["entry_net_debit_points"].gt(0).all()


def test_short_options_are_spread_legs_or_covered_calls_only() -> None:
    frame = trades()
    covered = frame["structure"].eq("CoveredCall")
    assert frame.loc[covered, "leg_count"].eq(1).all()
    assert frame.loc[covered, "leg1_direction"].eq(-1).all()
    spreads = frame.loc[~covered]
    assert spreads["leg_count"].eq(2).all()
    assert spreads["leg1_direction"].eq(1).all()
    assert spreads["leg2_direction"].eq(-1).all()
    assert frame.loc[covered, "max_loss_nav"].le(0.20 + 1e-12).all()


def test_greek_risk_book_is_finite() -> None:
    frame = trades()
    columns = [
        "delta_equivalent",
        "gamma_pnl_for_1pct_move",
        "vega_pnl_for_1vol_point",
        "max_loss_nav",
    ]
    assert np.isfinite(frame[columns]).all().all()


def test_failed_prelock_gate_keeps_four_asset_router() -> None:
    selection = report()["selection"]
    assert selection["prelock_all_three_pass"] is False
    assert selection["locked_not_used_for_selection"] is True
    assert selection["selected"] == "four_asset_router"
    selected = pd.read_csv(SELECTED_PATH, index_col=0)
    baseline = pd.read_csv(RECONCILED_PATH, index_col=0)
    common = selected.index.intersection(baseline.index)
    difference = selected.loc[common, "return"] - baseline.loc[common, "return"]
    assert float(difference.abs().max()) < 1e-12


def test_comparison_contains_all_four_declared_periods() -> None:
    comparison = pd.read_csv(COMPARISON_PATH)
    assert len(comparison) == 8
    assert set(comparison["Period"]) == {
        "calibration_2007_2012",
        "validation_2013_2017",
        "locked_2018_2026",
        "full_2007_2026",
    }
