from __future__ import annotations

import json
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import ASSETS
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    CAL_END,
    TEST_START,
    build_daily_vkospi_signals,
    load_daily_open_levels,
    paired_multiobjective_bootstrap,
    performance_summary,
    prepare_arrays,
    reconcile_to_monthly_reference,
    simulate,
)
from strategies.stage08_options.vix6_case1_strategy import (
    build_aligned_case1_inputs,
    build_final_medium_reference,
    build_vix6_features,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

DAILY_PATH = RESULTS / "vix6_router_daily.csv"
MONTHLY_PATH = RESULTS / "vix6_router_monthly.csv"
RECONCILED_PATH = RESULTS / "vix6_router_reconciled.csv"
COMPARISON_PATH = RESULTS / "vix6_router_comparison.csv"
STATE_SUMMARY_PATH = RESULTS / "vix6_router_state_summary.csv"
CALIBRATION_PATH = RESULTS / "vix6_router_calibration.csv"
REPORT_PATH = RESULTS / "vix6_router_validation.json"

VALIDATION_START = pd.Period("2013-01", freq="M")
REBALANCE_BAND = 0.20
FINANCING_RATE = 0.04

STATES = (
    "RiskOn",
    "HighIVRange",
    "PreCrash",
    "DeflationCrisis",
    "InflationCrisis",
    "Recovery",
)

OPTION_BY_STATE = {
    "RiskOn": "None",
    "HighIVRange": "CoveredCall",
    "PreCrash": "PutSpread",
    "DeflationCrisis": "PutSpread",
    "InflationCrisis": "PutSpread",
    "Recovery": "CallSpread",
}


@dataclass(frozen=True)
class RouterConfig:
    routing_preset: str = "legacy_gold"
    confirmation_scale: float = 0.0
    recovery_relief: float = 0.0

    @property
    def name(self) -> str:
        return (
            f"{self.routing_preset}_confirm{self.confirmation_scale:.2f}"
            f"_recovery{self.recovery_relief:.2f}"
        )


ROUTING_PRESETS = {
    "legacy_gold": {
        "precrash_deflation_bond_share": 0.0,
        "deflation_bond_share": 0.0,
        "inflation_bond_share": 0.0,
        "inflation_oil_cut": 1.0,
        "inflation_bond_tilt": 0.0,
    },
    "macro_soft": {
        "precrash_deflation_bond_share": 0.50,
        "deflation_bond_share": 0.50,
        "inflation_bond_share": 0.0,
        "inflation_oil_cut": 0.25,
        "inflation_bond_tilt": 0.05,
    },
    "macro_strict": {
        "precrash_deflation_bond_share": 0.75,
        "deflation_bond_share": 0.75,
        "inflation_bond_share": 0.0,
        "inflation_oil_cut": 0.0,
        "inflation_bond_tilt": 0.10,
    },
}


def _values(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return np.nan_to_num(
        frame[column].to_numpy(dtype=float),
        nan=default,
        posinf=default,
        neginf=default,
    )


def _ramp(values: np.ndarray, threshold: float, width: float) -> np.ndarray:
    return np.clip((values - threshold) / max(width, 1e-9), 0.0, 1.0)


def classify_vix6_states(aligned: pd.DataFrame) -> pd.DataFrame:
    """Classify six economically interpretable states with fixed causal rules."""
    vkospi_stress = np.clip(_values(aligned, "base_vkospi_stress"), 0, 1)
    vkospi_shock5 = _values(aligned, "vk_shock_5")
    percentile = _values(aligned, "vk_percentile_252", np.nan)
    fallback = _values(aligned, "vk_percentile_126", 0.5)
    percentile = np.where(np.isfinite(percentile), percentile, fallback)

    left_tail = _values(aligned, "left_tail")
    right_tail = _values(aligned, "right_tail")
    asymmetry = _values(aligned, "asymmetry")
    left_impulse = _values(aligned, "left_impulse_z")
    breadth_z = _values(aligned, "breadth_z")
    left_change = _values(aligned, "left_change_5")
    tracker_5 = _values(aligned, "tracker_return_5")
    tracker_21 = _values(aligned, "tracker_return_21")
    inflation = np.clip(_values(aligned, "p_inflation_high", 0.5), 0, 1)
    growth = np.clip(_values(aligned, "p_growth_high", 0.5), 0, 1)

    left_score = _ramp(
        np.maximum.reduce([left_tail, asymmetry, left_impulse]), 0.25, 1.75
    )
    breadth = _ramp(breadth_z, 0.0, 2.0)
    price_down = np.maximum(
        _ramp(-tracker_5, 0.01, 0.05),
        _ramp(-tracker_21, 0.02, 0.08),
    )
    price_up = np.maximum(
        _ramp(tracker_5, 0.01, 0.05),
        _ramp(tracker_21, 0.02, 0.08),
    )
    right_dominance = _ramp(right_tail - left_tail, 0.25, 1.75)
    fading_left = _ramp(-left_change, 0.25, 1.75)
    high_iv = _ramp(percentile, 0.70, 0.25)

    pre_crash = left_score * (1.0 - np.maximum(vkospi_stress, breadth)) * (
        1.0 - price_down
    )
    crisis = left_score * np.maximum(vkospi_stress, breadth) * (
        0.35 + 0.65 * price_down
    )
    recovery = fading_left * price_up * (0.50 + 0.50 * right_dominance)
    high_iv_range = (
        high_iv
        * (1.0 - left_score)
        * (1.0 - price_down)
        * (1.0 - price_up)
    )
    risk_on = (1.0 - high_iv) * (1.0 - left_score) * (
        0.50 + 0.50 * np.maximum(price_up, growth)
    )

    state = np.full(len(aligned), "RiskOn", dtype=object)
    state[high_iv_range >= 0.35] = "HighIVRange"
    state[pre_crash >= 0.35] = "PreCrash"
    crisis_mask = crisis >= 0.35
    state[crisis_mask & (inflation < 0.55)] = "DeflationCrisis"
    state[crisis_mask & (inflation >= 0.55)] = "InflationCrisis"
    state[recovery >= 0.35] = "Recovery"

    available = aligned["option_available"].to_numpy(dtype=bool)
    state = np.where(available, state, "RiskOn")
    option_structure = np.asarray([OPTION_BY_STATE[str(item)] for item in state])

    return pd.DataFrame(
        {
            "state": state,
            "option_structure": option_structure,
            "vkospi_stress": vkospi_stress,
            "vkospi_percentile": percentile,
            "vkospi_shock5": vkospi_shock5,
            "left_tail_score": left_score,
            "breadth_confirmation": breadth,
            "price_down_confirmation": price_down,
            "price_up_confirmation": price_up,
            "pre_crash_score": pre_crash,
            "crisis_score": crisis,
            "recovery_score": recovery,
            "high_iv_range_score": high_iv_range,
            "risk_on_score": risk_on,
            "p_growth_high": growth,
            "p_inflation_high": inflation,
            "option_available": available,
        },
        index=aligned.index,
    )


def _remove_fraction(
    weights: np.ndarray,
    donors: dict[int, float],
) -> tuple[np.ndarray, float]:
    output = weights.copy()
    removed = 0.0
    for index, fraction in donors.items():
        amount = max(float(output[index]), 0.0) * float(np.clip(fraction, 0, 1))
        output[index] -= amount
        removed += amount
    return output, removed


def route_asset_weights(
    base: np.ndarray,
    state: str,
    row: pd.Series,
    config: RouterConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a bounded VIX6 direction tilt; Macro/VKOSPI retain size control."""
    stress = float(np.clip(row["vkospi_stress"], 0, 1))
    pre_crash = float(np.clip(row["pre_crash_score"], 0, 1))
    crisis = float(np.clip(row["crisis_score"], 0, 1))
    recovery = float(np.clip(row["recovery_score"], 0, 1))
    inflation = float(np.clip(row["p_inflation_high"], 0, 1))
    routing = ROUTING_PRESETS[config.routing_preset]

    # Existing Robust VKOSPI is the risk-budget anchor: at full stress it removes
    # at most 35% of donor weights. VIX6 may alter the destination and add only a
    # bounded confirmation increment.
    base_cut = 0.35 * stress
    equity_cut = base_cut
    oil_cut = base_cut
    bond_share = 0.0

    if state == "PreCrash":
        equity_cut = max(base_cut, config.confirmation_scale * pre_crash)
        oil_cut = max(base_cut, 0.75 * config.confirmation_scale * pre_crash)
        bond_share = float(routing["precrash_deflation_bond_share"]) * (
            1.0 - inflation
        )
    elif state == "DeflationCrisis":
        equity_cut = max(base_cut, 1.50 * config.confirmation_scale * crisis)
        oil_cut = max(base_cut, 1.50 * config.confirmation_scale * crisis)
        bond_share = float(routing["deflation_bond_share"])
    elif state == "InflationCrisis":
        equity_cut = max(base_cut, 1.25 * config.confirmation_scale * crisis)
        oil_cut = base_cut * float(routing["inflation_oil_cut"])
        bond_share = float(routing["inflation_bond_share"])
    elif state == "Recovery":
        equity_cut = base_cut * (1.0 - config.recovery_relief * recovery)
        oil_cut = base_cut * (1.0 - config.recovery_relief * recovery)

    desired, removed = _remove_fraction(
        base,
        {0: float(np.clip(equity_cut, 0, 0.45)), 3: float(np.clip(oil_cut, 0, 0.45))},
    )
    bond_share = float(np.clip(bond_share, 0, 1))
    desired[1] += removed * bond_share
    desired[2] += removed * (1.0 - bond_share)

    # Inflation crises also move a bounded slice of bonds toward gold/oil.
    inflation_bond_tilt = 0.0
    if state == "InflationCrisis":
        inflation_bond_tilt = min(
            max(desired[1], 0.0)
            * float(routing["inflation_bond_tilt"])
            * crisis,
            0.05,
        )
        desired[1] -= inflation_bond_tilt
        desired[2] += 0.65 * inflation_bond_tilt
        desired[3] += 0.35 * inflation_bond_tilt

    # Recovery only relaxes the existing VKOSPI cut. It never adds leverage.
    recovery_tilt = 0.0

    assert np.isfinite(desired).all()
    assert (desired >= -1e-12).all()
    assert abs(float(desired.sum() - base.sum())) < 1e-10
    return desired, {
        "equity_cut_fraction": equity_cut,
        "oil_cut_fraction": oil_cut,
        "bond_receiver_share": bond_share,
        "inflation_bond_tilt": inflation_bond_tilt,
        "recovery_equity_tilt": recovery_tilt,
    }


def simulate_router(
    arrays: dict[str, object],
    diagnostics: pd.DataFrame,
    config: RouterConfig,
    keep_daily: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(arrays["dates"])
    months = pd.PeriodIndex(arrays["months"], freq="M")
    returns = np.asarray(arrays["returns"], dtype=float)
    base_weights = np.asarray(arrays["base_weights"], dtype=float)
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])

    pretrade = np.zeros(len(ASSETS))
    previous_month: pd.Period | None = None
    first_trade = True
    nav = 1.0
    peak = 1.0
    rf_daily = (1 + FINANCING_RATE) ** (1 / 252) - 1
    rows: list[dict[str, object]] = []

    for position, date in enumerate(dates):
        month = months[position]
        diagnostic = diagnostics.iloc[position]
        state = str(diagnostic["state"])
        desired, routing = route_asset_weights(
            base_weights[position].copy(), state, diagnostic, config
        )
        month_boundary = previous_month is None or month != previous_month
        desired_turnover = 0.5 * float(np.abs(desired - pretrade).sum())
        rebalance = month_boundary or desired_turnover >= REBALANCE_BAND
        weights = desired if rebalance else pretrade.copy()
        delta = weights - pretrade
        turnover = (
            float(np.abs(delta).sum())
            if first_trade
            else 0.5 * float(np.abs(delta).sum())
        )
        trade_cost = float(np.abs(delta).sum()) * 0.0015
        fx_cost = abs((weights[2] + weights[3]) - (pretrade[2] + pretrade[3])) * 0.0005
        debt_weight = 1.0 - float(weights.sum())
        asset_return = returns[position]
        gross_return = float(weights @ asset_return + debt_weight * rf_daily)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1 + asset_return) / (1 + gross_return)

        rows.append(
            {
                "date": date,
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "state": state,
                "option_structure": diagnostic["option_structure"],
                "transfer_fraction": max(
                    float(routing["equity_cut_fraction"]),
                    float(routing["oil_cut_fraction"]),
                ),
                "signal_date": signal_dates[position],
                **{column: diagnostic[column] for column in diagnostics.columns if column not in {"state", "option_structure"}},
                **routing,
                **{f"w_{asset}": float(weights[index]) for index, asset in enumerate(ASSETS)},
            }
        )
        previous_month = month
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda values: float(np.prod(1 + values))),
        gross_factor=("gross_return", lambda values: float(np.prod(1 + values))),
        turnover=("turnover", "sum"),
        trade_cost=("trade_cost", "sum"),
        fx_cost=("fx_cost", "sum"),
        avg_stress=("vkospi_stress", "mean"),
        max_stress=("vkospi_stress", "max"),
        avg_transfer=("transfer_fraction", "mean"),
    )
    monthly["return"] = monthly["return_factor"] - 1
    monthly["gross_return"] = monthly["gross_factor"] - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    return (daily if keep_daily else pd.DataFrame()), monthly


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
    return {
        "Period": period,
        "Strategy": strategy,
        **{key: float(value) for key, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean()) if "turnover" in view else np.nan,
    }


def main() -> None:
    medium, signals = build_final_medium_reference()
    levels = load_daily_open_levels()
    arrays = prepare_arrays(levels, medium, build_daily_vkospi_signals())
    aligned = build_aligned_case1_inputs(
        arrays, levels, signals, build_vix6_features(False)
    )
    diagnostics = classify_vix6_states(aligned)
    _, neutral_monthly = simulate(arrays, None, keep_daily=False)

    baseline = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv", index_col=0
    )
    baseline.index = pd.PeriodIndex(baseline.index, freq="M")
    calibration_end = pd.Period("2012-12", freq="M")
    baseline_cal = performance_summary(baseline.loc[:calibration_end, "return"])
    baseline_val = performance_summary(
        baseline.loc[VALIDATION_START:CAL_END, "return"]
    )

    candidate_rows: list[dict[str, object]] = []
    configs = [
        RouterConfig(routing, confirmation, relief)
        for routing, confirmation, relief in itertools.product(
            ROUTING_PRESETS,
            (0.0, 0.10),
            (0.0, 0.25, 0.50, 0.75),
        )
    ]
    for config in configs:
        _, candidate_monthly = simulate_router(
            arrays, diagnostics, config, keep_daily=False
        )
        candidate = reconcile_to_monthly_reference(
            medium, neutral_monthly, candidate_monthly
        )
        candidate_cal = performance_summary(candidate.loc[:calibration_end, "return"])
        candidate_val = performance_summary(
            candidate.loc[VALIDATION_START:CAL_END, "return"]
        )
        row: dict[str, object] = {"Name": config.name, **asdict(config)}
        for prefix, metrics, reference in (
            ("Cal", candidate_cal, baseline_cal),
            ("Validation", candidate_val, baseline_val),
        ):
            for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
                row[f"{prefix}_{metric}"] = float(metrics[metric])
                row[f"{prefix}_{metric}Delta"] = float(
                    metrics[metric] - reference[metric]
                )
        cal_score = (
            float(row["Cal_CAGRDelta"]) / 0.01
            + float(row["Cal_SharpeDelta"]) / 0.05
            + float(row["Cal_MDDDelta"]) / 0.01
        )
        validation_score = (
            float(row["Validation_CAGRDelta"]) / 0.01
            + float(row["Validation_SharpeDelta"]) / 0.05
            + float(row["Validation_MDDDelta"]) / 0.01
        )
        row["MultiObjectiveScore"] = min(cal_score, validation_score) + 0.25 * (
            cal_score + validation_score
        )
        row["StrictAllThree"] = bool(
            row["Cal_CAGRDelta"] > 0
            and row["Cal_SharpeDelta"] > 0
            and row["Cal_MDDDelta"] >= -1e-12
            and row["Validation_CAGRDelta"] > 0
            and row["Validation_SharpeDelta"] > 0
            and row["Validation_MDDDelta"] >= -1e-12
        )
        row["RetentionGate"] = bool(
            row["Cal_CAGR"] >= 0.995 * baseline_cal["CAGR"]
            and row["Cal_SharpeDelta"] > 0
            and row["Cal_MDDDelta"] >= -1e-12
            and row["Validation_CAGR"] >= 0.995 * baseline_val["CAGR"]
            and row["Validation_SharpeDelta"] > 0
            and row["Validation_MDDDelta"] >= -1e-12
        )
        candidate_rows.append(row)

    calibration = pd.DataFrame(candidate_rows)
    eligible = calibration.loc[calibration["StrictAllThree"]].copy()
    selection_rule = "strict all-three improvement in 2007-2012 and 2013-2017"
    if eligible.empty:
        eligible = calibration.loc[calibration["RetentionGate"]].copy()
        selection_rule = "99.5% CAGR retention with Sharpe/MDD improvement in both pre-lock windows"
    if eligible.empty:
        # Safe fallback reproduces the existing Robust VKOSPI allocation exactly.
        winner = RouterConfig("legacy_gold", 0.0, 0.0)
        selection_rule = "safe fallback: exact existing Robust VKOSPI allocation"
    else:
        winner_row = eligible.sort_values(
            ["MultiObjectiveScore", "Validation_Sharpe", "Cal_Sharpe"],
            ascending=False,
        ).iloc[0]
        winner = RouterConfig(
            routing_preset=str(winner_row["routing_preset"]),
            confirmation_scale=float(winner_row["confirmation_scale"]),
            recovery_relief=float(winner_row["recovery_relief"]),
        )

    daily, monthly = simulate_router(arrays, diagnostics, winner)
    reconciled = reconcile_to_monthly_reference(medium, neutral_monthly, monthly)
    periods = (
        ("calibration_2007_2012", None, calibration_end),
        ("validation_2013_2017", VALIDATION_START, CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", None, None),
    )
    comparison = pd.DataFrame(
        [
            _metric_row(period, strategy, path, start, end)
            for period, start, end in periods
            for strategy, path in (
                ("Existing_RobustVKOSPI", baseline),
                ("VIX6_ConditionalRouter", reconciled),
            )
        ]
    )

    state_summary = (
        daily.groupby("state")
        .agg(
            Days=("return", "size"),
            MeanDailyReturn=("return", "mean"),
            MeanVKOSPIStress=("vkospi_stress", "mean"),
            MeanEquityWeight=("w_KODEX200", "mean"),
            MeanBondWeight=("w_BOND", "mean"),
            MeanGoldWeight=("w_GLD", "mean"),
            MeanOilWeight=("w_USO", "mean"),
        )
        .reindex(STATES, fill_value=0)
    )

    locked_common = baseline.loc[TEST_START:].index.intersection(
        reconciled.loc[TEST_START:].index
    )
    bootstrap = paired_multiobjective_bootstrap(
        baseline.loc[locked_common, "return"],
        reconciled.loc[locked_common, "return"],
    )
    locked = comparison.loc[comparison["Period"].eq("locked_2018_2026")].set_index(
        "Strategy"
    )
    deltas = {
        metric: float(
            locked.loc["VIX6_ConditionalRouter", metric]
            - locked.loc["Existing_RobustVKOSPI", metric]
        )
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    signal_valid = daily["signal_date"].notna()
    report = {
        "objective": (
            "Macro sets base allocation; KOSPI/VKOSPI sets risk budget; VIX6 "
            "routes crisis type into bounded four-asset tilts and option structures."
        ),
        "states": list(STATES),
        "option_mapping": OPTION_BY_STATE,
        "parameters": {
            "rebalance_band": REBALANCE_BAND,
            "financing_rate": FINANCING_RATE,
            "maximum_equity_or_oil_fraction_cut": 0.45,
            "maximum_recovery_equity_tilt": 0.05,
            "classification": "fixed economic state rules; 24 bounded routing presets are ranked only on pre-2018 windows",
            "routing_winner": asdict(winner),
        },
        "selection": {
            "candidate_count": int(len(calibration)),
            "strict_count": int(calibration["StrictAllThree"].sum()),
            "retention_count": int(calibration["RetentionGate"].sum()),
            "rule": selection_rule,
            "uses_locked_metrics": False,
            "development_status": "post-lock exploratory; locked results were already visible in the broader project",
        },
        "data": {
            "daily_observations": int(len(daily)),
            "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()),
            "option_available_days": int(daily["option_available"].sum()),
        },
        "lookahead_audit": {
            "signal_strictly_before_action": bool(
                (daily.index[signal_valid].to_numpy() > pd.DatetimeIndex(daily.loc[signal_valid, "signal_date"]).to_numpy()).all()
            ),
            "vix6_state_uses_aligned_prior_signal": True,
            "selection_uses_locked_metrics": False,
        },
        "state_counts": {
            state: int(count) for state, count in daily["state"].value_counts().items()
        },
        "locked": {
            "existing": locked.loc["Existing_RobustVKOSPI", ["CAGR", "Sharpe", "MDD", "Calmar"]].to_dict(),
            "router": locked.loc["VIX6_ConditionalRouter", ["CAGR", "Sharpe", "MDD", "Calmar"]].to_dict(),
            "deltas": deltas,
            "all_three_improve": bool(
                deltas["CAGR"] > 0 and deltas["Sharpe"] > 0 and deltas["MDD"] >= -1e-12
            ),
            "bootstrap": bootstrap,
        },
        "comparison": comparison.to_dict(orient="records"),
    }

    daily.to_csv(DAILY_PATH, index_label="date")
    monthly.to_csv(MONTHLY_PATH, index_label="month")
    reconciled.to_csv(RECONCILED_PATH, index_label="month")
    comparison.to_csv(COMPARISON_PATH, index=False)
    state_summary.to_csv(STATE_SUMMARY_PATH, index_label="state")
    calibration.to_csv(CALIBRATION_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(comparison.loc[comparison["Period"].eq("full_2007_2026"), ["Strategy", "CAGR", "Sharpe", "MDD"]].to_string(index=False))
    print(json.dumps(report["state_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(report["locked"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
