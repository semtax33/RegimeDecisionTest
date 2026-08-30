from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from strategies.core.regime_research import performance_summary
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    CAL_END,
    TEST_START,
    paired_multiobjective_bootstrap,
)
from strategies.stage08_options.option_asset_slippage_experiment import (
    SCENARIOS,
    add_forward_and_delta,
    effective_slippage,
    stress_multiplier,
)
from strategies.stage08_options.vix6_case1_strategy import load_option_chain


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

TRADES_PATH = RESULTS / "vix6_router_option_trades.csv"
PATH = RESULTS / "vix6_router_option_candidate.csv"
SELECTED_PATH = RESULTS / "vix6_router_option_selected.csv"
COMPARISON_PATH = RESULTS / "vix6_router_option_comparison.csv"
REPORT_PATH = RESULTS / "vix6_router_option_validation.json"

VALIDATION_START = pd.Period("2013-01", freq="M")
CALIBRATION_END = pd.Period("2012-12", freq="M")
MIN_DTE = 30
MAX_DTE = 60
TARGET_DTE = 45
MIN_VOLUME = 1.0
MIN_HOLDING_CALENDAR_DAYS = 1
UNDERLYING_REALLOCATION_COST = 0.0015

PREMIUM_BUDGET = {
    "PreCrash": 0.0050,
    "DeflationCrisis": 0.0075,
    "InflationCrisis": 0.0025,
    "Recovery": 0.0050,
}
COVERED_CALL_EQUITY_FRACTION = 0.20


def build_liquid_option_universe() -> pd.DataFrame:
    chain = load_option_chain(require_iv=False)
    chain = chain.loc[
        chain["date"].dt.to_period("M").between(
            pd.Period("2007-01", freq="M"), pd.Period("2026-12", freq="M")
        )
    ].copy()
    universe = add_forward_and_delta(chain)
    universe = universe.loc[
        universe["dte"].between(MIN_DTE, MAX_DTE)
        & universe["volume"].ge(MIN_VOLUME)
        & universe["abs_delta"].between(0.03, 0.60)
        & universe["iv"].between(0.005, 3.0)
        & universe["forward"].gt(0)
    ].copy()
    return universe.sort_values(["date", "expiry", "option_type", "strike"])


def _closest_leg(
    frame: pd.DataFrame,
    option_type: str,
    target_delta: float,
) -> pd.Series | None:
    candidates = frame.loc[frame["option_type"].eq(option_type)].copy()
    if candidates.empty:
        return None
    candidates["delta_distance"] = (candidates["abs_delta"] - target_delta).abs()
    return candidates.sort_values(
        ["delta_distance", "volume"], ascending=[True, False]
    ).iloc[0]


def _entry_legs_for_date(
    day: pd.DataFrame,
    structure: str,
) -> list[tuple[pd.Series, int]] | None:
    expiries = (
        day[["expiry", "dte"]]
        .drop_duplicates()
        .assign(distance=lambda frame: (frame["dte"] - TARGET_DTE).abs())
        .sort_values("distance")
    )
    for expiry in expiries["expiry"]:
        frame = day.loc[day["expiry"].eq(expiry)]
        if structure == "CoveredCall":
            short_call = _closest_leg(frame, "C", 0.30)
            if short_call is not None:
                return [(short_call, -1)]
            continue

        option_type = "P" if structure == "PutSpread" else "C"
        long_leg = _closest_leg(frame, option_type, 0.30)
        short_leg = _closest_leg(frame, option_type, 0.10)
        if long_leg is None or short_leg is None or long_leg["code"] == short_leg["code"]:
            continue
        if structure == "PutSpread" and not long_leg["strike"] > short_leg["strike"]:
            continue
        if structure == "CallSpread" and not long_leg["strike"] < short_leg["strike"]:
            continue
        return [(long_leg, 1), (short_leg, -1)]
    return None


def _aligned_router_state(
    dates: pd.DatetimeIndex,
    router_daily: pd.DataFrame,
) -> pd.DataFrame:
    aligned = router_daily.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=5))
    aligned.index = dates
    return aligned


def select_monthly_structure_trades(
    universe: pd.DataFrame,
    router_daily: pd.DataFrame,
) -> pd.DataFrame:
    state_columns = [
        "state",
        "option_structure",
        "signal_date",
        "vkospi_percentile",
        "vkospi_shock5",
        "w_KODEX200",
    ]
    quote_dates = pd.DatetimeIndex(universe["date"].drop_duplicates().sort_values())
    state_by_date = _aligned_router_state(quote_dates, router_daily[state_columns])
    state_by_date = state_by_date.loc[
        state_by_date["option_structure"].isin(
            ["PutSpread", "CallSpread", "CoveredCall"]
        )
    ]
    rows: list[dict[str, object]] = []

    for month in sorted(state_by_date.index.to_period("M").unique()):
        month_dates = state_by_date.index[state_by_date.index.to_period("M") == month]
        selected: dict[str, object] | None = None
        entry_legs: list[tuple[pd.Series, int]] | None = None
        for entry_date in month_dates:
            state_row = state_by_date.loc[entry_date]
            if pd.isna(state_row["signal_date"]) or not pd.Timestamp(
                state_row["signal_date"]
            ) < entry_date:
                continue
            structure = str(state_row["option_structure"])
            day = universe.loc[universe["date"].eq(entry_date)]
            entry_legs = _entry_legs_for_date(day, structure)
            if entry_legs is None:
                continue
            selected = {
                "month": month,
                "state": str(state_row["state"]),
                "structure": structure,
                "entry_date": entry_date,
                "entry_signal_date": pd.Timestamp(state_row["signal_date"]),
                "entry_vkospi_percentile": float(state_row["vkospi_percentile"]),
                "entry_vkospi_shock5": float(state_row["vkospi_shock5"]),
                "entry_equity_weight": float(state_row["w_KODEX200"]),
            }
            break
        if selected is None or entry_legs is None:
            continue

        codes = [str(leg["code"]) for leg, _ in entry_legs]
        expiry = pd.Timestamp(entry_legs[0][0]["expiry"])
        exit_quotes = universe.loc[
            universe["code"].astype(str).isin(codes)
            & universe["date"].gt(selected["entry_date"])
            & universe["date"].dt.to_period("M").eq(month)
            & universe["date"].le(expiry)
        ].copy()
        common_dates = (
            exit_quotes.groupby("date")["code"].nunique().loc[lambda value: value == len(codes)].index
        )
        if len(common_dates) == 0:
            continue

        exit_state = _aligned_router_state(
            pd.DatetimeIndex(common_dates), router_daily[state_columns]
        )
        # The archived KOSPI200 option sample contains only a few adjacent quote
        # dates in many months.  Requiring three calendar days discarded nearly
        # every otherwise executable spread.  A strictly later quote still
        # prevents same-day hindsight while retaining the available sample.
        minimum_exit = pd.Timestamp(selected["entry_date"]) + pd.Timedelta(
            days=MIN_HOLDING_CALENDAR_DAYS
        )
        valid_dates = pd.DatetimeIndex(common_dates)[pd.DatetimeIndex(common_dates) >= minimum_exit]
        if len(valid_dates) == 0:
            continue
        exit_date = valid_dates[-1]
        for candidate_date in valid_dates:
            candidate_structure = str(exit_state.loc[candidate_date, "option_structure"])
            if candidate_structure != selected["structure"]:
                exit_date = candidate_date
                break
        exit_signal_date = pd.Timestamp(exit_state.loc[exit_date, "signal_date"])
        if not exit_signal_date < exit_date:
            continue

        selected.update(
            {
                "exit_date": exit_date,
                "exit_signal_date": exit_signal_date,
                "exit_vkospi_percentile": float(
                    exit_state.loc[exit_date, "vkospi_percentile"]
                ),
                "exit_vkospi_shock5": float(exit_state.loc[exit_date, "vkospi_shock5"]),
                "expiry": expiry,
            }
        )
        for number, (leg, direction) in enumerate(entry_legs, start=1):
            exit_leg = exit_quotes.loc[
                exit_quotes["date"].eq(exit_date)
                & exit_quotes["code"].astype(str).eq(str(leg["code"]))
            ].iloc[-1]
            selected.update(
                {
                    f"leg{number}_code": str(leg["code"]),
                    f"leg{number}_direction": int(direction),
                    f"leg{number}_option_type": str(leg["option_type"]),
                    f"leg{number}_strike": float(leg["strike"]),
                    f"leg{number}_entry_price": float(leg["price"]),
                    f"leg{number}_entry_iv": float(leg["iv"]),
                    f"leg{number}_entry_delta": float(leg["abs_delta"]),
                    f"leg{number}_entry_forward": float(leg["forward"]),
                    f"leg{number}_entry_dte": float(leg["dte"]),
                    f"leg{number}_exit_price": float(exit_leg["price"]),
                    f"leg{number}_exit_delta": float(exit_leg["abs_delta"]),
                    f"leg{number}_exit_volume": float(exit_leg["volume"]),
                }
            )
        selected["leg_count"] = len(entry_legs)
        rows.append(selected)

    trades = pd.DataFrame(rows).set_index("month").sort_index()
    if not trades.empty:
        assert (trades["entry_signal_date"] < trades["entry_date"]).all()
        assert (trades["exit_signal_date"] < trades["exit_date"]).all()
        assert (trades["entry_date"] < trades["exit_date"]).all()
    return trades


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _greeks(
    option_type: str,
    forward: float,
    strike: float,
    iv: float,
    dte: float,
) -> tuple[float, float, float]:
    term = max(dte / 365.0, 1e-9)
    root = math.sqrt(term)
    d1 = (math.log(forward / strike) + 0.5 * iv * iv * term) / (iv * root)
    delta = float(ndtr(d1)) if option_type == "C" else float(ndtr(d1) - 1.0)
    gamma = _normal_pdf(d1) / (forward * iv * root)
    vega_one_vol_point = forward * _normal_pdf(d1) * root * 0.01
    return delta, gamma, vega_one_vol_point


def price_and_size_trades(trades: pd.DataFrame) -> pd.DataFrame:
    scenario = next(item for item in SCENARIOS if item.name == "Base")
    rows: list[dict[str, object]] = []
    for month, trade in trades.iterrows():
        entry_multiplier = stress_multiplier(
            float(trade["entry_vkospi_percentile"]),
            float(trade["entry_vkospi_shock5"]),
        )
        exit_multiplier = stress_multiplier(
            float(trade["exit_vkospi_percentile"]),
            float(trade["exit_vkospi_shock5"]),
        )
        entry_net = 0.0
        exit_value = 0.0
        delta_per_structure = 0.0
        gamma_per_structure = 0.0
        vega_per_structure = 0.0
        mean_forward = 0.0
        for number in range(1, int(trade["leg_count"]) + 1):
            direction = int(trade[f"leg{number}_direction"])
            entry_price = float(trade[f"leg{number}_entry_price"])
            exit_price = float(trade[f"leg{number}_exit_price"])
            entry_abs_delta = float(trade[f"leg{number}_entry_delta"])
            exit_abs_delta = float(trade[f"leg{number}_exit_delta"])
            entry_slip = effective_slippage(
                entry_price, entry_abs_delta, scenario, entry_multiplier
            )[0]
            exit_slip = effective_slippage(
                exit_price, exit_abs_delta, scenario, exit_multiplier
            )[0]
            entry_execution = entry_price * (1.0 + entry_slip if direction > 0 else 1.0 - entry_slip)
            exit_execution = exit_price * (1.0 - exit_slip if direction > 0 else 1.0 + exit_slip)
            entry_net += direction * entry_execution
            exit_value += direction * exit_execution

            forward = float(trade[f"leg{number}_entry_forward"])
            mean_forward = forward
            delta, gamma, vega = _greeks(
                str(trade[f"leg{number}_option_type"]),
                forward,
                float(trade[f"leg{number}_strike"]),
                float(trade[f"leg{number}_entry_iv"]),
                float(trade[f"leg{number}_entry_dte"]),
            )
            delta_per_structure += direction * delta
            gamma_per_structure += direction * gamma
            vega_per_structure += direction * vega

        structure = str(trade["structure"])
        state = str(trade["state"])
        if structure in {"PutSpread", "CallSpread"}:
            if entry_net <= 0:
                continue
            premium_budget = float(PREMIUM_BUDGET[state])
            quantity = premium_budget / entry_net
            liquidation_value = max(exit_value, 0.0)
            option_return = liquidation_value / entry_net - 1.0
            option_pnl_nav = quantity * (liquidation_value - entry_net)
            premium_credit = 0.0
            max_loss_nav = premium_budget
        else:
            covered_notional = min(
                max(float(trade["entry_equity_weight"]), 0.0)
                * COVERED_CALL_EQUITY_FRACTION,
                0.20,
            )
            quantity = covered_notional / mean_forward
            premium_budget = 0.0
            premium_credit = quantity * max(-entry_net, 0.0)
            option_pnl_nav = quantity * (-entry_net + exit_value)
            option_return = np.nan
            max_loss_nav = covered_notional

        rows.append(
            {
                **trade.to_dict(),
                "month": month,
                "premium_budget": premium_budget,
                "premium_credit": premium_credit,
                "quantity_per_nav": quantity,
                "option_return_on_debit": option_return,
                "option_pnl_nav": option_pnl_nav,
                "delta_equivalent": quantity * delta_per_structure * mean_forward,
                "gamma_pnl_for_1pct_move": quantity
                * gamma_per_structure
                * (0.01 * mean_forward) ** 2,
                "vega_pnl_for_1vol_point": quantity * vega_per_structure,
                "max_loss_nav": max_loss_nav,
                "entry_net_debit_points": entry_net,
                "exit_liquidation_points": exit_value,
            }
        )
    return pd.DataFrame(rows).set_index("month").sort_index()


def apply_option_overlay(
    baseline: pd.DataFrame,
    router_daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    output = baseline.copy()
    output["baseline_return"] = output["return"]
    for column in (
        "option_pnl_nav",
        "premium_budget",
        "premium_credit",
        "delta_equivalent",
        "gamma_pnl_for_1pct_move",
        "vega_pnl_for_1vol_point",
        "max_loss_nav",
    ):
        output[column] = 0.0
    output["option_structure"] = "None"
    output["option_trade_available"] = False

    for month, trade in trades.iterrows():
        if month not in output.index:
            continue
        held = router_daily.loc[
            (router_daily.index > trade["entry_date"])
            & (router_daily.index <= trade["exit_date"]),
            "return",
        ]
        holding_growth = float((1 + held).prod()) if len(held) else 1.0
        if trade["structure"] in {"PutSpread", "CallSpread"}:
            budget = float(trade["premium_budget"])
            option_growth = 1.0 + float(trade["option_return_on_debit"])
            mixed_growth = (
                (1.0 - budget) * holding_growth
                + budget * option_growth
                - 2.0 * UNDERLYING_REALLOCATION_COST * budget
            )
        else:
            mixed_growth = holding_growth + float(trade["option_pnl_nav"])
        relative_factor = mixed_growth / holding_growth
        output.loc[month, "return"] = (
            (1.0 + output.loc[month, "baseline_return"]) * relative_factor - 1.0
        )
        for column in (
            "option_pnl_nav",
            "premium_budget",
            "premium_credit",
            "delta_equivalent",
            "gamma_pnl_for_1pct_move",
            "vega_pnl_for_1vol_point",
            "max_loss_nav",
        ):
            output.loc[month, column] = float(trade[column])
        output.loc[month, "option_structure"] = str(trade["structure"])
        output.loc[month, "option_trade_available"] = True

    wealth = (1 + output["return"]).cumprod()
    output["nav"] = wealth
    output["drawdown"] = wealth / wealth.cummax() - 1.0
    return output


def _metric_row(
    period: str,
    strategy: str,
    path: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, object]:
    view = path
    if start is not None:
        view = view.loc[start:]
    if end is not None:
        view = view.loc[:end]
    metrics = performance_summary(view["return"])
    return {"Period": period, "Strategy": strategy, **metrics.to_dict()}


def main() -> None:
    baseline = pd.read_csv(RESULTS / "vix6_router_reconciled.csv", index_col=0)
    baseline.index = pd.PeriodIndex(baseline.index, freq="M")
    router_daily = pd.read_csv(
        RESULTS / "vix6_router_daily.csv", index_col=0, parse_dates=True
    )
    universe = build_liquid_option_universe()
    selected = select_monthly_structure_trades(universe, router_daily)
    trades = price_and_size_trades(selected)
    candidate = apply_option_overlay(baseline, router_daily, trades)

    periods = (
        ("calibration_2007_2012", None, CALIBRATION_END),
        ("validation_2013_2017", VALIDATION_START, CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", None, None),
    )
    comparison = pd.DataFrame(
        [
            _metric_row(period, strategy, path, start, end)
            for period, start, end in periods
            for strategy, path in (
                ("VIX6_Router_4Asset", baseline),
                ("VIX6_Router_WithOptions", candidate),
            )
        ]
    )
    indexed = comparison.set_index(["Period", "Strategy"])
    prelock_periods = ("calibration_2007_2012", "validation_2013_2017")
    prelock_pass = all(
        indexed.loc[(period, "VIX6_Router_WithOptions"), "CAGR"]
        > indexed.loc[(period, "VIX6_Router_4Asset"), "CAGR"]
        and indexed.loc[(period, "VIX6_Router_WithOptions"), "Sharpe"]
        > indexed.loc[(period, "VIX6_Router_4Asset"), "Sharpe"]
        and indexed.loc[(period, "VIX6_Router_WithOptions"), "MDD"]
        >= indexed.loc[(period, "VIX6_Router_4Asset"), "MDD"] - 1e-12
        for period in prelock_periods
    )
    locked = indexed.loc["locked_2018_2026"]
    locked_deltas = {
        metric: float(
            locked.loc["VIX6_Router_WithOptions", metric]
            - locked.loc["VIX6_Router_4Asset", metric]
        )
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    locked_pass = bool(
        locked_deltas["CAGR"] > 0
        and locked_deltas["Sharpe"] > 0
        and locked_deltas["MDD"] >= -1e-12
    )
    selected_path = candidate if prelock_pass else baseline
    decision = (
        "Adopt option structures"
        if prelock_pass
        else "Keep four-asset router; option structures remain research diagnostics"
    )

    common = baseline.loc[TEST_START:].index.intersection(candidate.loc[TEST_START:].index)
    bootstrap = paired_multiobjective_bootstrap(
        baseline.loc[common, "return"], candidate.loc[common, "return"]
    )
    structure_counts = {
        name: int(count) for name, count in trades["structure"].value_counts().items()
    }
    report = {
        "objective": (
            "Use VIX6 as a crisis-type router for Put Spread, Call Spread and "
            "Covered Call rather than as a fifth simple asset."
        ),
        "budgets": {
            "premium_budget_by_state": PREMIUM_BUDGET,
            "covered_call_equity_fraction": COVERED_CALL_EQUITY_FRACTION,
            "debit_spread_max_loss_equals_premium_budget": True,
            "naked_short_options": False,
        },
        "execution": {
            "target_dte": TARGET_DTE,
            "dte_range": [MIN_DTE, MAX_DTE],
            "minimum_holding_calendar_days": MIN_HOLDING_CALENDAR_DAYS,
            "exit_rule": "first later archived quote after state changes, otherwise last common quote in entry month",
            "long_spread_delta": 0.30,
            "short_spread_delta": 0.10,
            "covered_call_delta": 0.30,
            "slippage_scenario": "Base with VKOSPI stress multiplier and tick floor",
            "fractional_contract_assumption": True,
        },
        "risk_fields": [
            "premium_budget",
            "delta_equivalent",
            "gamma_pnl_for_1pct_move",
            "vega_pnl_for_1vol_point",
            "max_loss_nav",
        ],
        "trade_audit": {
            "completed_trades": int(len(trades)),
            "structure_counts": structure_counts,
            "entry_signal_precedes_entry": bool(
                (trades["entry_signal_date"] < trades["entry_date"]).all()
            ),
            "exit_signal_precedes_exit": bool(
                (trades["exit_signal_date"] < trades["exit_date"]).all()
            ),
            "entry_precedes_exit": bool((trades["entry_date"] < trades["exit_date"]).all()),
            "maximum_debit_spread_loss_budget": float(
                trades.loc[trades["structure"].isin(["PutSpread", "CallSpread"]), "max_loss_nav"].max()
            ),
            "maximum_absolute_delta_equivalent": float(trades["delta_equivalent"].abs().max()),
        },
        "selection": {
            "prelock_all_three_pass": bool(prelock_pass),
            "locked_all_three_pass": locked_pass,
            "selected": "candidate" if prelock_pass else "four_asset_router",
            "decision": decision,
            "locked_not_used_for_selection": True,
            "development_status": "post-lock exploratory; locked results were visible in the broader project",
        },
        "locked": {
            "deltas_candidate_minus_four_asset": locked_deltas,
            "bootstrap": bootstrap,
        },
        "comparison": comparison.to_dict(orient="records"),
    }

    trades.to_csv(TRADES_PATH, index_label="month")
    candidate.to_csv(PATH, index_label="month")
    selected_path.to_csv(SELECTED_PATH, index_label="month")
    comparison.to_csv(COMPARISON_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(comparison.loc[comparison["Period"].eq("full_2007_2026"), ["Strategy", "CAGR", "Sharpe", "MDD"]].to_string(index=False))
    print(json.dumps(report["trade_audit"], ensure_ascii=False, indent=2))
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
