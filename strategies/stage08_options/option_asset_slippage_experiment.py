from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from strategies.core.regime_research import performance_summary
from strategies.stage08_options.vix6_case1_strategy import (
    build_final_medium_reference,
    build_vix6_features,
    load_option_chain,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import CAL_END, TEST_START
from strategies.stage06_vkospi.vkospi_robust_dynamic_experiment import build_robust_daily_features


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

OPTION_UNIVERSE_PATH = RESULTS / "option_asset_liquid_put_universe.csv"
OPTION_TRADES_PATH = RESULTS / "option_asset_monthly_trades.csv"
OPTION_RETURNS_PATH = RESULTS / "option_asset_monthly_returns_by_slippage.csv"
COMPARISON_PATH = RESULTS / "option_asset_slippage_comparison_2007_2026.csv"
BEST_PATH = RESULTS / "option_asset_best_candidate_monthly.csv"
SELECTED_PATH = RESULTS / "option_asset_selected_strategy_monthly.csv"
REPORT_PATH = RESULTS / "option_asset_slippage_validation.json"

FULL_START = pd.Period("2007-01", freq="M")
FULL_END = pd.Period("2026-12", freq="M")
TARGET_DELTA = 0.10
MIN_DELTA = 0.05
MAX_DELTA = 0.15
MIN_DTE = 30
MAX_DTE = 60
TARGET_DTE = 45
VIX6_ENTRY_THRESHOLD = 0.20
VIX6_RECOVERY_THRESHOLD = 0.35
UNDERLYING_REALLOCATION_COST = 0.0015


@dataclass(frozen=True)
class SlippageScenario:
    name: str
    atm: float
    shoulder: float
    tail: float
    deep_tail: float


SCENARIOS = (
    SlippageScenario("Optimistic", atm=0.01, shoulder=0.02, tail=0.05, deep_tail=0.10),
    SlippageScenario("Base", atm=0.015, shoulder=0.03, tail=0.075, deep_tail=0.15),
    SlippageScenario("Conservative", atm=0.02, shoulder=0.05, tail=0.10, deep_tail=0.20),
)
TRIGGERS = (
    "vix6_score",
    "vix6_sqrt",
    "vix6_squared",
    "vix6_plus_logistic",
)
MAX_WEIGHTS = (0.005, 0.010, 0.020, 0.030)


def option_tick(price: float) -> float:
    """Current KRX KOSPI200 option tick rule used as a conservative V1 floor."""
    return 0.01 if price < 10.0 else 0.05


def delta_bucket_rate(abs_delta: float, scenario: SlippageScenario) -> float:
    if abs_delta < 0.05:
        return scenario.deep_tail
    if abs_delta <= 0.15:
        return scenario.tail
    if abs_delta < 0.45:
        return scenario.shoulder
    return scenario.atm


def stress_multiplier(percentile: float, shock_5: float) -> float:
    """Apply the pasted feedback's VKOSPI crisis liquidity penalty."""
    if (np.isfinite(percentile) and percentile >= 0.97) or (
        np.isfinite(shock_5) and shock_5 >= 2.5
    ):
        return 2.0
    if np.isfinite(percentile) and percentile >= 0.90:
        return 1.5
    if np.isfinite(percentile) and percentile >= 0.80:
        return 1.25
    return 1.0


def effective_slippage(
    price: float,
    abs_delta: float,
    scenario: SlippageScenario,
    multiplier: float,
) -> tuple[float, float, float, bool]:
    bucket = delta_bucket_rate(abs_delta, scenario) * multiplier
    tick_floor = option_tick(price) / price
    effective = min(max(bucket, tick_floor), 1.0)
    return float(effective), float(bucket), float(tick_floor), bool(tick_floor >= bucket)


def _put_price_from_forward(
    forward: np.ndarray,
    strike: np.ndarray,
    term: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    root_term = np.sqrt(term)
    d1 = (np.log(forward / strike) + 0.5 * np.square(sigma) * term) / (
        sigma * root_term
    )
    d2 = d1 - sigma * root_term
    return strike * ndtr(-d2) - forward * ndtr(-d1)


def implied_put_volatility(
    price: np.ndarray,
    forward: np.ndarray,
    strike: np.ndarray,
    term: np.ndarray,
) -> np.ndarray:
    """Vectorized zero-rate forward Black implied volatility for EOD put prices."""
    result = np.full(len(price), np.nan, dtype=float)
    intrinsic = np.maximum(strike - forward, 0.0)
    valid = (
        np.isfinite(price)
        & (price > intrinsic + 1e-10)
        & np.isfinite(forward)
        & (forward > 0)
        & np.isfinite(strike)
        & (strike > 0)
        & np.isfinite(term)
        & (term > 0)
        & (price < strike)
    )
    if not valid.any():
        return result
    p = price[valid]
    f = forward[valid]
    k = strike[valid]
    t = term[valid]
    low = np.full(len(p), 0.005)
    high = np.full(len(p), 3.0)
    attainable = _put_price_from_forward(f, k, t, high) >= p
    for _ in range(60):
        midpoint = 0.5 * (low + high)
        below = _put_price_from_forward(f, k, t, midpoint) < p
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)
    solved = 0.5 * (low + high)
    positions = np.flatnonzero(valid)
    result[positions[attainable]] = solved[attainable]
    return result


def add_forward_and_delta(chain: pd.DataFrame) -> pd.DataFrame:
    """Estimate same-expiry forwards from put-call parity and compute absolute delta."""
    quotes = chain.pivot_table(
        index=["date", "expiry", "strike"],
        columns="option_type",
        values="price",
        aggfunc="last",
    )
    if not {"C", "P"}.issubset(quotes.columns):
        raise ValueError("Both call and put observations are required")
    paired = quotes.dropna(subset=["C", "P"]).reset_index()
    paired = paired.loc[paired["C"].gt(0) & paired["P"].gt(0)].copy()
    paired["parity_gap"] = (paired["C"] - paired["P"]).abs()
    paired["synthetic_forward"] = paired["strike"] + paired["C"] - paired["P"]
    nearest = (
        paired.sort_values(["date", "expiry", "parity_gap"])
        .groupby(["date", "expiry"], sort=False)
        .head(3)
    )
    forwards = (
        nearest.groupby(["date", "expiry"])["synthetic_forward"]
        .median()
        .rename("forward")
        .reset_index()
    )
    output = chain.merge(forwards, on=["date", "expiry"], how="left")
    term = output["dte"].to_numpy(dtype=float) / 365.0
    forward = output["forward"].to_numpy(dtype=float)
    strike = output["strike"].to_numpy(dtype=float)
    price = output["price"].to_numpy(dtype=float)
    is_put = output["option_type"].eq("P").to_numpy(dtype=bool)
    sigma = output["iv"].to_numpy(dtype=float)
    sigma[is_put] = implied_put_volatility(
        price[is_put],
        forward[is_put],
        strike[is_put],
        term[is_put],
    )
    output["iv"] = sigma
    output["iv_source"] = np.where(is_put, "price_inverted", "reported")
    valid = (
        np.isfinite(term)
        & (term > 0)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(forward)
        & (forward > 0)
        & np.isfinite(strike)
        & (strike > 0)
    )
    d1 = np.full(len(output), np.nan, dtype=float)
    d1[valid] = (
        np.log(forward[valid] / strike[valid])
        + 0.5 * np.square(sigma[valid]) * term[valid]
    ) / (sigma[valid] * np.sqrt(term[valid]))
    output["abs_delta"] = np.where(
        output["option_type"].eq("P"),
        ndtr(-d1),
        ndtr(d1),
    )
    return output.replace([np.inf, -np.inf], np.nan)


def build_liquid_put_universe(force: bool = False) -> pd.DataFrame:
    if OPTION_UNIVERSE_PATH.exists() and not force:
        cached = pd.read_csv(
            OPTION_UNIVERSE_PATH,
            parse_dates=["date", "expiry"],
            dtype={"code": str},
        )
        return cached.sort_values(["date", "expiry", "strike"])
    chain = add_forward_and_delta(load_option_chain(require_iv=False))
    universe = chain.loc[
        chain["option_type"].eq("P")
        & chain["volume"].gt(0)
        & chain["abs_delta"].between(0.01, 0.99)
        & chain["date"].dt.to_period("M").between(FULL_START, FULL_END),
        [
            "date",
            "expiry",
            "dte",
            "code",
            "strike",
            "price",
            "iv",
            "volume",
            "forward",
            "abs_delta",
        ],
    ].copy()
    universe.to_csv(OPTION_UNIVERSE_PATH, index=False)
    return universe


def build_vix6_trade_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Create one-observation-lagged VIX6 entry and recovery signals."""
    observed = features.sort_index().shift(1)
    output = pd.DataFrame(index=observed.index)

    def ramp(column: str, threshold: float, width: float) -> pd.Series:
        return ((observed[column] - threshold) / width).clip(0, 1)

    left_tail = ramp("left_tail", 0.25, 1.75)
    asymmetry = ramp("asymmetry", 0.25, 1.75)
    impulse = ramp("left_impulse_z", 0.25, 1.75)
    breadth = ramp("breadth_z", 0.00, 2.00)
    output["entry_score"] = (
        pd.concat([left_tail, asymmetry, impulse], axis=1).max(axis=1)
        * (0.50 + 0.50 * breadth)
    ).clip(0, 1)
    output["recovery_score"] = (
        0.50 * ((-observed["left_change_5"]) / 1.50).clip(0, 1)
        + 0.30 * ((-observed["left_impulse_z"]) / 1.50).clip(0, 1)
        + 0.20
        * ((observed["right_tail"] - observed["left_tail"]) / 1.50).clip(0, 1)
    ).clip(0, 1)
    output["signal_date"] = pd.Series(features.index, index=features.index).shift(1)
    return output.replace([np.inf, -np.inf], np.nan)


def _align_vix6_signals(
    dates: pd.Series,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    aligned = signals.reindex(
        pd.DatetimeIndex(dates),
        method="ffill",
        tolerance=pd.Timedelta(days=7),
    )
    aligned.index = dates.index
    return aligned


def select_monthly_put_trades(
    universe: pd.DataFrame,
    vix6_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Trade a liquid 10-delta put only when lagged VIX6 signals permit it."""
    eligible = universe.loc[
        universe["dte"].between(MIN_DTE, MAX_DTE)
        & universe["abs_delta"].between(MIN_DELTA, MAX_DELTA)
    ].copy()
    eligible["month"] = eligible["date"].dt.to_period("M")
    entry_signals = _align_vix6_signals(eligible["date"], vix6_signals)
    eligible["entry_signal_score"] = entry_signals["entry_score"]
    eligible["entry_signal_date"] = entry_signals["signal_date"]
    eligible = eligible.loc[
        eligible["entry_signal_score"].ge(VIX6_ENTRY_THRESHOLD)
        & eligible["entry_signal_date"].lt(eligible["date"])
    ].copy()
    first_dates = eligible.groupby("month")["date"].min().rename("entry_date")
    entries = eligible.join(first_dates, on="month")
    entries = entries.loc[entries["date"].eq(entries["entry_date"])].copy()
    entries["delta_distance"] = (entries["abs_delta"] - TARGET_DELTA).abs()
    entries["dte_distance"] = (entries["dte"] - TARGET_DTE).abs()
    entries = (
        entries.sort_values(
            ["month", "delta_distance", "dte_distance", "volume"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates("month", keep="first")
        .copy()
    )

    keys = entries[["month", "entry_date", "code", "expiry"]]
    exits = universe.copy()
    exits["month"] = exits["date"].dt.to_period("M")
    exits = exits.merge(keys, on=["month", "code", "expiry"], how="inner")
    exits = exits.loc[exits["date"].gt(exits["entry_date"])].copy()
    exit_signals = _align_vix6_signals(exits["date"], vix6_signals)
    exits["exit_recovery_score"] = exit_signals["recovery_score"]
    exits["exit_signal_date"] = exit_signals["signal_date"]
    valid_recovery = exits.loc[
        exits["exit_recovery_score"].ge(VIX6_RECOVERY_THRESHOLD)
        & exits["exit_signal_date"].lt(exits["date"])
    ]
    recovery_exits = (
        valid_recovery.sort_values(["month", "date"])
        .groupby("month", sort=False)
        .head(1)
        .assign(exit_reason="vix6_recovery", exit_priority=0)
    )
    forced_exits = (
        exits.sort_values(["month", "date"])
        .groupby("month", sort=False)
        .tail(1)
        .assign(exit_reason="month_end_roll", exit_priority=1)
    )
    exits = pd.concat([recovery_exits, forced_exits]).sort_values(
        ["month", "exit_priority"]
    ).drop_duplicates("month", keep="first")

    entry_columns = {
        "date": "entry_date",
        "dte": "entry_dte",
        "strike": "strike",
        "price": "entry_close",
        "iv": "entry_iv",
        "volume": "entry_volume",
        "forward": "entry_forward",
        "abs_delta": "entry_abs_delta",
        "entry_signal_score": "entry_signal_score",
        "entry_signal_date": "entry_signal_date",
    }
    exit_columns = {
        "date": "exit_date",
        "dte": "exit_dte",
        "price": "exit_close",
        "iv": "exit_iv",
        "volume": "exit_volume",
        "forward": "exit_forward",
        "abs_delta": "exit_abs_delta",
        "exit_recovery_score": "exit_recovery_score",
        "exit_signal_date": "exit_signal_date",
        "exit_reason": "exit_reason",
    }
    entry_frame = entries[
        ["month", "code", "expiry", *entry_columns.keys()]
    ].rename(columns=entry_columns)
    exit_frame = exits[["month", "code", "expiry", *exit_columns.keys()]].rename(
        columns=exit_columns
    )
    trades = entry_frame.merge(exit_frame, on=["month", "code", "expiry"], how="inner")
    trades = trades.sort_values("month").set_index("month")
    if not trades["entry_dte"].between(MIN_DTE, MAX_DTE).all():
        raise AssertionError("Entry DTE outside requested range")
    if not trades["entry_abs_delta"].between(MIN_DELTA, MAX_DELTA).all():
        raise AssertionError("Entry delta outside requested liquid tail range")
    if not (trades[["entry_volume", "exit_volume"]].gt(0).all(axis=1)).all():
        raise AssertionError("Zero-volume option trade detected")
    if not (trades["exit_date"] > trades["entry_date"]).all():
        raise AssertionError("Option exits must occur after entries")
    if not (trades["entry_signal_date"] < trades["entry_date"]).all():
        raise AssertionError("VIX6 entry signals must precede option entries")
    if not (trades["exit_signal_date"] < trades["exit_date"]).all():
        raise AssertionError("VIX6 exit signals must precede option exits")
    return trades


def align_prior_vkospi(trades: pd.DataFrame) -> pd.DataFrame:
    """Align prior-observation VKOSPI state to option entry and exit dates."""
    robust = build_robust_daily_features()[
        ["percentile_252", "percentile_126", "shock_5"]
    ].shift(1)
    robust["percentile"] = robust["percentile_252"].fillna(
        robust["percentile_126"]
    )
    output = trades.copy()
    for side in ("entry", "exit"):
        dates = pd.DatetimeIndex(output[f"{side}_date"])
        aligned = robust[["percentile", "shock_5"]].reindex(
            dates,
            method="ffill",
            tolerance=pd.Timedelta(days=7),
        )
        output[f"{side}_vk_percentile"] = aligned["percentile"].to_numpy(dtype=float)
        output[f"{side}_vk_shock5"] = aligned["shock_5"].to_numpy(dtype=float)
        output[f"{side}_stress_multiplier"] = [
            stress_multiplier(percentile, shock)
            for percentile, shock in zip(
                output[f"{side}_vk_percentile"],
                output[f"{side}_vk_shock5"],
            )
        ]
    return output


def price_trade_scenarios(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, trade in trades.iterrows():
        for scenario in SCENARIOS:
            entry_slip, entry_bucket, entry_tick, entry_tick_binds = effective_slippage(
                float(trade["entry_close"]),
                float(trade["entry_abs_delta"]),
                scenario,
                float(trade["entry_stress_multiplier"]),
            )
            exit_slip, exit_bucket, exit_tick, exit_tick_binds = effective_slippage(
                float(trade["exit_close"]),
                float(trade["exit_abs_delta"]),
                scenario,
                float(trade["exit_stress_multiplier"]),
            )
            buy = float(trade["entry_close"]) * (1.0 + entry_slip)
            sell = float(trade["exit_close"]) * (1.0 - exit_slip)
            option_return = sell / buy - 1.0
            rows.append(
                {
                    "month": month,
                    "scenario": scenario.name,
                    "code": trade["code"],
                    "entry_date": trade["entry_date"],
                    "exit_date": trade["exit_date"],
                    "entry_close": float(trade["entry_close"]),
                    "exit_close": float(trade["exit_close"]),
                    "entry_abs_delta": float(trade["entry_abs_delta"]),
                    "exit_abs_delta": float(trade["exit_abs_delta"]),
                    "entry_dte": float(trade["entry_dte"]),
                    "exit_dte": float(trade["exit_dte"]),
                    "entry_volume": float(trade["entry_volume"]),
                    "exit_volume": float(trade["exit_volume"]),
                    "entry_signal_date": trade["entry_signal_date"],
                    "entry_signal_score": float(trade["entry_signal_score"]),
                    "exit_signal_date": trade["exit_signal_date"],
                    "exit_recovery_score": float(trade["exit_recovery_score"]),
                    "exit_reason": str(trade["exit_reason"]),
                    "entry_stress_multiplier": float(
                        trade["entry_stress_multiplier"]
                    ),
                    "exit_stress_multiplier": float(trade["exit_stress_multiplier"]),
                    "entry_bucket_slippage": entry_bucket,
                    "exit_bucket_slippage": exit_bucket,
                    "entry_tick_floor": entry_tick,
                    "exit_tick_floor": exit_tick,
                    "entry_tick_binds": entry_tick_binds,
                    "exit_tick_binds": exit_tick_binds,
                    "entry_effective_slippage": entry_slip,
                    "exit_effective_slippage": exit_slip,
                    "buy_price": buy,
                    "sell_price": sell,
                    "option_return": float(option_return),
                }
            )
    return pd.DataFrame(rows).set_index(["month", "scenario"]).sort_index()


def build_trigger_intensities(
    baseline: pd.DataFrame,
    medium: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, pd.Series]:
    index = baseline.index
    logistic = (-medium["risk_score"].reindex(index)).clip(0, 1).fillna(0.0)
    vix6_score = trades["entry_signal_score"].reindex(index).fillna(0.0).clip(0, 1)
    return {
        "vix6_score": vix6_score,
        "vix6_sqrt": np.sqrt(vix6_score),
        "vix6_squared": np.square(vix6_score),
        "vix6_plus_logistic": pd.concat([vix6_score, logistic], axis=1).max(axis=1),
    }


def build_baseline_holding_growth(
    trades: pd.DataFrame,
    daily_baseline: pd.DataFrame,
) -> pd.Series:
    """Return growth of the four-asset sleeve while each option is actually held."""
    values: dict[pd.Period, float] = {}
    for month, trade in trades.iterrows():
        held = daily_baseline.loc[
            (daily_baseline.index > trade["entry_date"])
            & (daily_baseline.index <= trade["exit_date"]),
            "return",
        ]
        values[month] = float((1.0 + held).prod()) if len(held) else 1.0
    return pd.Series(values, dtype=float).sort_index()


def run_option_allocation(
    baseline: pd.DataFrame,
    option_return: pd.Series,
    intensity: pd.Series,
    max_weight: float,
    baseline_holding_growth: pd.Series,
) -> pd.DataFrame:
    index = baseline.index.intersection(option_return.index.union(baseline.index))
    base_return = baseline.loc[index, "return"].astype(float)
    available_return = option_return.reindex(index)
    signal = intensity.reindex(index).fillna(0.0).clip(0, 1)
    option_weight = max_weight * signal
    option_weight = option_weight.where(available_return.notna(), 0.0)
    sleeve_weight = 1.0 - option_weight
    holding_growth = baseline_holding_growth.reindex(index).fillna(1.0)
    option_growth = 1.0 + available_return.fillna(0.0)
    reallocation_cost = 2.0 * UNDERLYING_REALLOCATION_COST * option_weight
    mixed_holding_growth = (
        sleeve_weight * holding_growth
        + option_weight * option_growth
        - reallocation_cost
    )
    relative_factor = mixed_holding_growth / holding_growth
    net_return = (1.0 + base_return) * relative_factor - 1.0
    nav = (1.0 + net_return).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return pd.DataFrame(
        {
            "return": net_return,
            "baseline_return": base_return,
            "option_return": available_return,
            "baseline_holding_growth": holding_growth,
            "option_holding_growth": option_growth,
            "option_relative_factor": relative_factor,
            "signal_intensity": signal,
            "w_existing_four_asset_sleeve": sleeve_weight,
            "w_KOSPI200_put_option": option_weight,
            "option_reallocation_cost": reallocation_cost,
            "option_trade_available": available_return.notna(),
            "nav": nav,
            "drawdown": drawdown,
        },
        index=index,
    )


def _metrics(path: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.Series:
    return performance_summary(path.loc[start:end, "return"])


def _comparison_row(
    scenario: str,
    trigger: str,
    max_weight: float,
    path: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario,
        "trigger": trigger,
        "max_option_weight": max_weight,
        "candidate": f"{scenario}_{trigger}_w{max_weight:.3f}",
        "active_months": int(path["w_KOSPI200_put_option"].gt(0).sum()),
        "average_option_weight": float(path["w_KOSPI200_put_option"].mean()),
        "annual_option_budget": float(path["w_KOSPI200_put_option"].mean() * 12),
    }
    periods = (
        ("full", FULL_START, FULL_END),
        ("pre2018", FULL_START, CAL_END),
        ("locked", TEST_START, FULL_END),
    )
    for prefix, start, end in periods:
        candidate_metrics = _metrics(path, start, end)
        baseline_metrics = _metrics(baseline, start, end)
        for metric in ("CAGR", "Sharpe", "MDD"):
            row[f"{prefix}_{metric}"] = float(candidate_metrics[metric])
            row[f"{prefix}_{metric}_delta"] = float(
                candidate_metrics[metric] - baseline_metrics[metric]
            )
    row["full_score"] = (
        float(row["full_CAGR_delta"]) / 0.01
        + float(row["full_Sharpe_delta"]) / 0.05
        + float(row["full_MDD_delta"]) / 0.01
    )
    row["pre2018_score"] = (
        float(row["pre2018_CAGR_delta"]) / 0.01
        + float(row["pre2018_Sharpe_delta"]) / 0.05
        + float(row["pre2018_MDD_delta"]) / 0.01
    )
    row["full_all_three_improve"] = bool(
        row["full_CAGR_delta"] > 0
        and row["full_Sharpe_delta"] > 0
        and row["full_MDD_delta"] >= -1e-12
    )
    return row


def run_experiment(force_universe: bool = False) -> dict[str, object]:
    universe = build_liquid_put_universe(force_universe)
    vix6_signals = build_vix6_trade_signals(build_vix6_features(False))
    trades = align_prior_vkospi(
        select_monthly_put_trades(universe, vix6_signals)
    )
    priced = price_trade_scenarios(trades)
    trades.to_csv(OPTION_TRADES_PATH)
    priced.to_csv(OPTION_RETURNS_PATH)

    baseline = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv",
        index_col=0,
    )
    baseline.index = pd.PeriodIndex(baseline.index, freq="M")
    baseline = baseline.loc[FULL_START:FULL_END].copy()
    baseline["nav"] = (1.0 + baseline["return"]).cumprod()
    baseline["drawdown"] = baseline["nav"] / baseline["nav"].cummax() - 1.0
    daily_baseline = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_daily.csv",
        index_col=0,
        parse_dates=True,
    )
    baseline_holding_growth = build_baseline_holding_growth(trades, daily_baseline)
    medium, _ = build_final_medium_reference()
    intensities = build_trigger_intensities(baseline, medium, trades)

    rows: list[dict[str, object]] = []
    paths: dict[tuple[str, str, float], pd.DataFrame] = {}
    for scenario in SCENARIOS:
        option_return = priced.xs(scenario.name, level="scenario")["option_return"]
        option_return.index = pd.PeriodIndex(option_return.index, freq="M")
        for trigger in TRIGGERS:
            for max_weight in MAX_WEIGHTS:
                path = run_option_allocation(
                    baseline,
                    option_return,
                    intensities[trigger],
                    max_weight,
                    baseline_holding_growth,
                )
                paths[(scenario.name, trigger, max_weight)] = path
                row = _comparison_row(
                    scenario.name,
                    trigger,
                    max_weight,
                    path,
                    baseline,
                )
                rows.append(row)
                print(
                    f"{row['candidate']}: CAGR={row['full_CAGR']:.4f} "
                    f"Sharpe={row['full_Sharpe']:.4f} MDD={row['full_MDD']:.4f}",
                    flush=True,
                )

    comparison = pd.DataFrame(rows).sort_values(
        ["scenario", "full_score"],
        ascending=[True, False],
    )
    comparison.to_csv(COMPARISON_PATH, index=False)

    base_candidates = comparison.loc[comparison["scenario"].eq("Base")]
    best_base = base_candidates.sort_values("full_score", ascending=False).iloc[0]
    best_key = (
        "Base",
        str(best_base["trigger"]),
        float(best_base["max_option_weight"]),
    )
    best_path = paths[best_key]
    best_path.to_csv(BEST_PATH)

    robust_keys: list[tuple[str, float]] = []
    for trigger in TRIGGERS:
        for max_weight in MAX_WEIGHTS:
            pair = comparison.loc[
                comparison["trigger"].eq(trigger)
                & comparison["max_option_weight"].eq(max_weight)
                & comparison["scenario"].isin(["Base", "Conservative"])
            ]
            if len(pair) == 2 and pair["full_all_three_improve"].astype(bool).all():
                robust_keys.append((trigger, max_weight))

    if robust_keys:
        trigger, max_weight = max(
            robust_keys,
            key=lambda key: float(
                comparison.loc[
                    comparison["scenario"].eq("Conservative")
                    & comparison["trigger"].eq(key[0])
                    & comparison["max_option_weight"].eq(key[1]),
                    "full_score",
                ].iloc[0]
            ),
        )
        selected_strategy = f"OptionAsset_{trigger}_w{max_weight:.3f}"
        selected_path = paths[("Conservative", trigger, max_weight)]
        decision = (
            "Adoptable only as a post-lock exploratory candidate: both Base and "
            "Conservative slippage improved full-period CAGR, Sharpe and MDD."
        )
    else:
        selected_strategy = "Existing_Final_RobustVKOSPI"
        selected_path = baseline[
            ["return", "nav", "drawdown"]
        ].copy()
        decision = (
            "Keep the existing strategy: no option allocation improved full-period "
            "CAGR, Sharpe and MDD under both Base and Conservative slippage."
        )
    selected_path.to_csv(SELECTED_PATH)

    baseline_metrics = _metrics(baseline, FULL_START, FULL_END)
    scenario_summary: dict[str, object] = {}
    for scenario in SCENARIOS:
        scenario_rows = comparison.loc[comparison["scenario"].eq(scenario.name)]
        winner = scenario_rows.sort_values("full_score", ascending=False).iloc[0]
        scenario_trades = priced.xs(scenario.name, level="scenario")
        scenario_summary[scenario.name] = {
            "best_candidate": str(winner["candidate"]),
            "CAGR": float(winner["full_CAGR"]),
            "Sharpe": float(winner["full_Sharpe"]),
            "MDD": float(winner["full_MDD"]),
            "all_three_improve": bool(winner["full_all_three_improve"]),
            "mean_entry_effective_slippage": float(
                scenario_trades["entry_effective_slippage"].mean()
            ),
            "mean_exit_effective_slippage": float(
                scenario_trades["exit_effective_slippage"].mean()
            ),
            "entry_tick_floor_binding_rate": float(
                scenario_trades["entry_tick_binds"].mean()
            ),
            "exit_tick_floor_binding_rate": float(
                scenario_trades["exit_tick_binds"].mean()
            ),
        }

    report: dict[str, object] = {
        "objective": (
            "Add a physically priced KOSPI200 put option as a fifth asset sleeve and "
            "stress test delta-bucket, one-tick and VKOSPI-dependent slippage."
        ),
        "evaluation_status": (
            "Post-lock exploratory walk-forward comparison; 2007-2026 is not an "
            "untouched holdout after candidate inspection."
        ),
        "period": {
            "requested": "2007-2026",
            "actual_start": str(baseline.index.min()),
            "actual_end": str(baseline.index.max()),
            "months": int(len(baseline)),
        },
        "option_contract": {
            "type": "long KOSPI200 monthly put",
            "entry_delta_range": [MIN_DELTA, MAX_DELTA],
            "target_delta": TARGET_DELTA,
            "entry_dte_range": [MIN_DTE, MAX_DTE],
            "target_dte": TARGET_DTE,
            "funding": (
                "Option premium weight scales down the existing four-asset strategy "
                "sleeve, so sleeve weights sum to one."
            ),
            "volume_rule": "entry and exit volume must both be positive",
            "roll_rule": (
                "enter the first liquid quote after a lagged VIX6 tail signal; exit "
                "on the first lagged VIX6 recovery signal or the last liquid "
                "same-contract quote in the month"
            ),
            "delta_iv_estimation": (
                "same-expiry put-call-parity forward and price-inverted Black put IV"
            ),
            "fractional_contract_assumption": True,
        },
        "vix6_allocation": {
            "entry_components": [
                "left_tail",
                "asymmetry",
                "left_impulse_z",
                "breadth_z",
            ],
            "entry_threshold": VIX6_ENTRY_THRESHOLD,
            "exit_components": [
                "left_change_5",
                "left_impulse_z",
                "right_tail_minus_left_tail",
            ],
            "recovery_threshold": VIX6_RECOVERY_THRESHOLD,
            "weight_rules": list(TRIGGERS),
            "timing": "all VIX6 features are shifted one observation before action",
        },
        "slippage": {
            "scenarios": {scenario.name: asdict(scenario) for scenario in SCENARIOS},
            "tick_rule": "0.01 point below 10 premium points; 0.05 at or above 10",
            "effective_formula": "max(delta bucket rate * stress multiplier, tick / close)",
            "buy": "close * (1 + effective slippage)",
            "sell": "close * (1 - effective slippage)",
            "stress_multiplier": {
                "normal": 1.0,
                "VKOSPI_percentile_80": 1.25,
                "VKOSPI_percentile_90": 1.5,
                "VKOSPI_percentile_97_or_shock5_2.5": 2.0,
            },
            "signal_timing": "prior VKOSPI observation, shifted one row before trade date",
        },
        "trade_audit": {
            "liquid_put_universe_rows": int(len(universe)),
            "completed_monthly_trades": int(len(trades)),
            "entry_volume_zero_count": int(trades["entry_volume"].eq(0).sum()),
            "exit_volume_zero_count": int(trades["exit_volume"].eq(0).sum()),
            "entry_delta_min": float(trades["entry_abs_delta"].min()),
            "entry_delta_max": float(trades["entry_abs_delta"].max()),
            "entry_dte_min": float(trades["entry_dte"].min()),
            "entry_dte_max": float(trades["entry_dte"].max()),
            "entry_strictly_before_exit": bool(
                (trades["entry_date"] < trades["exit_date"]).all()
            ),
            "vix6_entry_signal_precedes_trade": bool(
                (trades["entry_signal_date"] < trades["entry_date"]).all()
            ),
            "vix6_exit_signal_precedes_trade": bool(
                (trades["exit_signal_date"] < trades["exit_date"]).all()
            ),
            "vix6_recovery_exits": int(
                trades["exit_reason"].eq("vix6_recovery").sum()
            ),
            "month_end_roll_exits": int(
                trades["exit_reason"].eq("month_end_roll").sum()
            ),
        },
        "candidate_count": int(len(comparison)),
        "baseline": {
            "CAGR": float(baseline_metrics["CAGR"]),
            "Sharpe": float(baseline_metrics["Sharpe"]),
            "MDD": float(baseline_metrics["MDD"]),
        },
        "scenario_summary": scenario_summary,
        "best_base_candidate": {
            "candidate": str(best_base["candidate"]),
            "CAGR": float(best_base["full_CAGR"]),
            "Sharpe": float(best_base["full_Sharpe"]),
            "MDD": float(best_base["full_MDD"]),
            "CAGR_delta": float(best_base["full_CAGR_delta"]),
            "Sharpe_delta": float(best_base["full_Sharpe_delta"]),
            "MDD_delta": float(best_base["full_MDD_delta"]),
            "all_three_improve": bool(best_base["full_all_three_improve"]),
        },
        "selection": {
            "selected_strategy": selected_strategy,
            "decision": decision,
            "robust_candidate_count": int(len(robust_keys)),
            "rule": (
                "An option allocation must improve full-period CAGR, Sharpe and MDD "
                "under both Base and Conservative slippage."
            ),
        },
        "artifacts": {
            "liquid_put_universe": str(OPTION_UNIVERSE_PATH.relative_to(ROOT)),
            "monthly_trades": str(OPTION_TRADES_PATH.relative_to(ROOT)),
            "option_returns": str(OPTION_RETURNS_PATH.relative_to(ROOT)),
            "comparison": str(COMPARISON_PATH.relative_to(ROOT)),
            "best_candidate": str(BEST_PATH.relative_to(ROOT)),
            "selected_strategy": str(SELECTED_PATH.relative_to(ROOT)),
        },
        "official_reference": (
            "https://global.krx.co.kr/contents/GLB/02/0201/0201040202/"
            "GLB0201040202.jsp"
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-universe", action="store_true")
    args = parser.parse_args()
    run_experiment(force_universe=args.force_universe)


if __name__ == "__main__":
    main()
