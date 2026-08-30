from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import ASSETS, cdar, performance_summary
from strategies.stage10_vix6_router.vix6_conditional_router_strategy import (
    build_daily_vkospi_signals,
    build_final_medium_reference,
    load_daily_open_levels,
    prepare_arrays,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CORE_PATH = (
    ROOT
    / "strategies"
    / "stage10_slsqp_sharpe100"
    / "outputs"
    / "slsqp_sharpe_cagr100_monthly.csv"
)
STAGE11_PATH = (
    ROOT
    / "strategies"
    / "stage11_mse_log_growth_slsqp100"
    / "outputs"
    / "slsqp_mse_loggrowth100_monthly.csv"
)
ROUTER_DAILY_PATH = ROOT / "results" / "vix6_router_daily.csv"
OPTION_PATH = ROOT / "results" / "option_asset_best_candidate_monthly.csv"

FULL_START = pd.Period("2007-04", freq="M")
CALIBRATION_END = pd.Period("2012-12", freq="M")
VALIDATION_START = pd.Period("2013-01", freq="M")
VALIDATION_END = pd.Period("2017-12", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")

NORMAL_GROWTH_CAP = 1.00
ALERT_GROWTH_CAP = 0.75
CRISIS_GROWTH_CAP = 0.25
NEGATIVE_MOMENTUM_CAP = 0.50
MONTHLY_GROWTH_CHANGE_CAP = 0.25
FORECAST_SHARPE_FLOOR = 1.05
HISTORICAL_CDAR_FLOOR = -0.13
META_SWITCH_COST = 0.0005
MIN_HISTORY_MONTHS = 12


def load_monthly_path(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index()


def reconstruct_unlevered_router() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reprice the causal router after normalizing daily gross exposure to one."""
    medium, _ = build_final_medium_reference()
    arrays = prepare_arrays(
        load_daily_open_levels(), medium, build_daily_vkospi_signals()
    )
    dates = pd.DatetimeIndex(arrays["dates"])
    asset_returns = np.asarray(arrays["returns"], dtype=float)
    source = pd.read_csv(
        ROUTER_DAILY_PATH, index_col="date", parse_dates=True
    ).reindex(dates)
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    required = [
        *weight_columns,
        "signal_date",
        "state",
        "vkospi_stress",
        "crisis_score",
        "pre_crash_score",
    ]
    if source[required].isna().any().any():
        missing = source.index[source[required].isna().any(axis=1)]
        raise ValueError(
            f"Router data do not align with daily asset returns: {missing[:3]}"
        )

    raw_weights = source[weight_columns].to_numpy(dtype=float)
    raw_exposure = raw_weights.sum(axis=1)
    if np.any(raw_exposure <= 0):
        raise ValueError("Router contains non-positive gross exposure.")
    normalized_weights = raw_weights / raw_exposure[:, None]

    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    rows: list[dict[str, object]] = []
    for position, date in enumerate(dates):
        weights = normalized_weights[position]
        delta = weights - pretrade
        turnover = (
            float(np.abs(delta).sum())
            if position == 0
            else 0.5 * float(np.abs(delta).sum())
        )
        trade_cost = float(np.abs(delta).sum()) * 0.0015
        fx_cost = (
            abs(
                (weights[2] + weights[3])
                - (pretrade[2] + pretrade[3])
            )
            * 0.0005
        )
        gross_return = float(weights @ asset_returns[position])
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = (
            weights
            * (1 + asset_returns[position])
            / (1 + gross_return)
        )
        rows.append(
            {
                "date": date,
                "month": pd.Period(date, freq="M"),
                "signal_date": pd.Timestamp(source.iloc[position]["signal_date"]),
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "state": str(source.iloc[position]["state"]),
                "vkospi_stress": float(
                    source.iloc[position]["vkospi_stress"]
                ),
                "crisis_score": float(
                    source.iloc[position]["crisis_score"]
                ),
                "pre_crash_score": float(
                    source.iloc[position]["pre_crash_score"]
                ),
                "source_gross_exposure": float(raw_exposure[position]),
                "gross_exposure": float(weights.sum()),
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda values: float(np.prod(1 + values))),
        gross_factor=(
            "gross_return", lambda values: float(np.prod(1 + values))
        ),
        turnover=("turnover", "sum"),
        trade_cost=("trade_cost", "sum"),
        fx_cost=("fx_cost", "sum"),
        avg_stress=("vkospi_stress", "mean"),
        max_stress=("vkospi_stress", "max"),
        max_crisis_score=("crisis_score", "max"),
        max_pre_crash_score=("pre_crash_score", "max"),
        maximum_source_exposure=("source_gross_exposure", "max"),
        maximum_gross_exposure=("gross_exposure", "max"),
        **{
            f"avg_w_{asset}": (f"w_{asset}", "mean")
            for asset in ASSETS
        },
    )
    monthly["return"] = monthly.pop("return_factor") - 1
    monthly["gross_return"] = monthly.pop("gross_factor") - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return daily, monthly


def build_causal_monthly_risk_signals(
    router_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Use month-m daily diagnostics only for the following month allocation."""
    source = router_daily.groupby("month").agg(
        average_vkospi_stress=("vkospi_stress", "mean"),
        maximum_vkospi_stress=("vkospi_stress", "max"),
        maximum_crisis_score=("crisis_score", "max"),
        maximum_pre_crash_score=("pre_crash_score", "max"),
        last_source_date=("signal_date", "max"),
    )
    source["signal_month"] = source.index
    source.index = pd.PeriodIndex(source.index, freq="M") + 1
    source.index.name = "month"
    return source


def ewma_covariance(history: pd.DataFrame) -> np.ndarray:
    recent = history.tail(84)
    ages = np.arange(len(recent) - 1, -1, -1)
    weights = np.exp(-math.log(2) / 12 * ages)
    weights /= weights.sum()
    values = recent.to_numpy(dtype=float)
    center = (weights[:, None] * values).sum(axis=0)
    demeaned = values - center
    return (
        (demeaned * weights[:, None]).T @ demeaned
        + np.eye(values.shape[1]) * 1e-8
    )


def expected_sleeve_moments(
    history: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    long_mean = history.mean().to_numpy(dtype=float)
    recent_mean = (
        history.tail(84)
        .ewm(halflife=24, adjust=False)
        .mean()
        .iloc[-1]
        .to_numpy(dtype=float)
    )
    expected_return = np.clip(
        0.80 * long_mean + 0.20 * recent_mean,
        -0.006,
        0.020,
    )
    return expected_return, ewma_covariance(history)


def growth_cap(
    risk_signal: pd.Series,
    relative_momentum_6m: float,
) -> tuple[float, str]:
    if (
        float(risk_signal["maximum_vkospi_stress"]) >= 0.55
        or float(risk_signal["maximum_crisis_score"]) >= 0.55
    ):
        cap = CRISIS_GROWTH_CAP
        state = "Crisis"
    elif (
        float(risk_signal["maximum_vkospi_stress"]) >= 0.30
        or float(risk_signal["maximum_pre_crash_score"]) >= 0.45
    ):
        cap = ALERT_GROWTH_CAP
        state = "Alert"
    else:
        cap = NORMAL_GROWTH_CAP
        state = "Normal"
    if relative_momentum_6m <= 0:
        cap = min(cap, NEGATIVE_MOMENTUM_CAP)
        state += "_NegativeMomentum"
    return cap, state


def sleeve_metrics(
    growth_weight: float,
    expected_return: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    weights = np.array([1.0 - growth_weight, growth_weight])
    monthly_return = float(weights @ expected_return)
    monthly_variance = max(float(weights @ covariance @ weights), 0.0)
    annual_return = 12 * monthly_return
    annual_volatility = math.sqrt(12 * monthly_variance)
    expected_sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 1e-12
        else -1e9
    )
    expected_log_growth = annual_return - 6 * monthly_variance
    return {
        "expected_annual_return": annual_return,
        "expected_annual_volatility": annual_volatility,
        "expected_sharpe": expected_sharpe,
        "expected_log_growth": expected_log_growth,
    }


def optimize_growth_weight(
    history: pd.DataFrame,
    risk_signal: pd.Series,
    previous_growth_weight: float,
) -> tuple[float, dict[str, object]]:
    expected_return, covariance = expected_sleeve_moments(history)
    recent = history.tail(84).to_numpy(dtype=float)
    relative_momentum_6m = float(
        np.prod(1 + history.tail(6)["growth"])
        / np.prod(1 + history.tail(6)["core"])
        - 1
    )
    cap, risk_state = growth_cap(risk_signal, relative_momentum_6m)
    lower = max(0.0, previous_growth_weight - MONTHLY_GROWTH_CHANGE_CAP)
    upper = min(cap, previous_growth_weight + MONTHLY_GROWTH_CHANGE_CAP)
    if lower > upper:
        # A newly triggered risk cap can reduce exposure immediately.
        lower = upper

    def objective(value: np.ndarray) -> float:
        return -sleeve_metrics(
            float(value[0]), expected_return, covariance
        )["expected_log_growth"]

    def sharpe_constraint(value: np.ndarray) -> float:
        return (
            sleeve_metrics(float(value[0]), expected_return, covariance)[
                "expected_sharpe"
            ]
            - FORECAST_SHARPE_FLOOR
        )

    def cdar_constraint(value: np.ndarray) -> float:
        alpha = float(value[0])
        weights = np.array([1.0 - alpha, alpha])
        return cdar(recent @ weights, 0.90) - HISTORICAL_CDAR_FLOOR

    initial = np.array(
        [float(np.clip(previous_growth_weight, lower, upper))]
    )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(lower, upper)],
        constraints=[
            {"type": "ineq", "fun": sharpe_constraint},
            {"type": "ineq", "fun": cdar_constraint},
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    used_grid_fallback = False
    if result.success and np.isfinite(result.x).all():
        growth_weight = float(np.clip(result.x[0], lower, upper))
    else:
        grid = np.linspace(lower, upper, 201)
        feasible = [
            value
            for value in grid
            if sharpe_constraint(np.array([value])) >= -1e-10
            and cdar_constraint(np.array([value])) >= -1e-10
        ]
        growth_weight = (
            max(feasible, key=lambda value: -objective(np.array([value])))
            if feasible
            else 0.0
        )
        used_grid_fallback = True
    metrics = sleeve_metrics(growth_weight, expected_return, covariance)
    weights = np.array([1.0 - growth_weight, growth_weight])
    historical_cdar = cdar(recent @ weights, 0.90)
    return growth_weight, {
        "solver_success": bool(result.success),
        "used_grid_fallback": used_grid_fallback,
        "solver_status": int(getattr(result, "status", 0)),
        "solver_message": str(getattr(result, "message", "fixed_bound")),
        "solver_iterations": int(getattr(result, "nit", 0)),
        "risk_state": risk_state,
        "relative_momentum_6m": relative_momentum_6m,
        "growth_cap": cap,
        "growth_lower_bound": lower,
        "growth_upper_bound": upper,
        **metrics,
        "historical_cdar": historical_cdar,
        "sharpe_slack": metrics["expected_sharpe"] - FORECAST_SHARPE_FLOOR,
        "cdar_slack": historical_cdar - HISTORICAL_CDAR_FLOOR,
    }


def run_meta_strategy(
    core: pd.DataFrame,
    growth: pd.DataFrame,
    risk_signals: pd.DataFrame,
) -> pd.DataFrame:
    months = core.index.intersection(growth.index).sort_values()
    sleeve_returns = pd.concat(
        {
            "core": core.loc[months, "return"],
            "growth": growth.loc[months, "return"],
        },
        axis=1,
    )
    rows: list[dict[str, object]] = []
    previous_growth_weight = 0.0
    nav = 1.0
    peak = 1.0
    for month in months:
        history = sleeve_returns.loc[sleeve_returns.index < month]
        if len(history) < MIN_HISTORY_MONTHS or month not in risk_signals.index:
            growth_weight = 0.0
            detail: dict[str, object] = {
                "solver_success": True,
                "used_grid_fallback": False,
                "solver_status": 0,
                "solver_message": "warmup_core_only",
                "solver_iterations": 0,
                "risk_state": "Warmup",
                "relative_momentum_6m": np.nan,
                "growth_cap": 0.0,
                "growth_lower_bound": 0.0,
                "growth_upper_bound": 0.0,
                "expected_annual_return": np.nan,
                "expected_annual_volatility": np.nan,
                "expected_sharpe": np.nan,
                "expected_log_growth": np.nan,
                "historical_cdar": np.nan,
                "sharpe_slack": np.nan,
                "cdar_slack": np.nan,
            }
            signal_month = month - 1
        else:
            growth_weight, detail = optimize_growth_weight(
                history,
                risk_signals.loc[month],
                previous_growth_weight,
            )
            signal_month = risk_signals.loc[month, "signal_month"]
        core_weight = 1.0 - growth_weight
        switch_turnover = abs(growth_weight - previous_growth_weight)
        switch_cost = META_SWITCH_COST * switch_turnover
        net_return = float(
            core_weight * sleeve_returns.loc[month, "core"]
            + growth_weight * sleeve_returns.loc[month, "growth"]
            - switch_cost
        )
        nav *= 1 + net_return
        peak = max(peak, nav)
        internal_turnover = float(
            core_weight * core.loc[month, "turnover"]
            + growth_weight * growth.loc[month, "turnover"]
        )
        rows.append(
            {
                "month": month,
                "signal_month": signal_month,
                "return": net_return,
                "core_return": float(sleeve_returns.loc[month, "core"]),
                "growth_return": float(sleeve_returns.loc[month, "growth"]),
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": internal_turnover + switch_turnover,
                "meta_switch_turnover": switch_turnover,
                "meta_switch_cost": switch_cost,
                "w_stage10_core": core_weight,
                "w_unlevered_vix6_growth": growth_weight,
                "w_option": 0.0,
                "gross_exposure": core_weight + growth_weight,
                **detail,
            }
        )
        previous_growth_weight = growth_weight
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def apply_physical_put_candidate(meta: pd.DataFrame) -> pd.DataFrame:
    """Fund the existing causal VIX6 long-put trade from portfolio capital."""
    option = pd.read_csv(OPTION_PATH, index_col=0)
    option.index = pd.PeriodIndex(option.index, freq="M")
    output = meta.copy()
    aligned = option.reindex(output.index)
    option_weight = aligned["w_KOSPI200_put_option"].fillna(0.0).clip(0, 1)
    option_return = aligned["option_return"].fillna(0.0)
    reallocation_cost = aligned["option_reallocation_cost"].fillna(0.0)
    sleeve_weight = 1.0 - option_weight
    output["return"] = (
        sleeve_weight * meta["return"]
        + option_weight * option_return
        - reallocation_cost
    )
    output["w_stage10_core"] = sleeve_weight * meta["w_stage10_core"]
    output["w_unlevered_vix6_growth"] = (
        sleeve_weight * meta["w_unlevered_vix6_growth"]
    )
    output["w_option"] = option_weight
    output["option_return"] = aligned["option_return"]
    output["option_reallocation_cost"] = reallocation_cost
    output["option_trade_available"] = aligned[
        "option_trade_available"
    ].fillna(False).astype(bool)
    output["turnover"] = meta["turnover"] + 2.0 * option_weight
    output["gross_exposure"] = (
        output["w_stage10_core"]
        + output["w_unlevered_vix6_growth"]
        + output["w_option"]
    )
    wealth = (1 + output["return"]).cumprod()
    output["nav"] = wealth
    output["drawdown"] = wealth / wealth.cummax() - 1
    return output


def performance_row(
    name: str,
    period: str,
    path: pd.DataFrame,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, object]:
    view = path.loc[start:end]
    metrics = performance_summary(view["return"])
    return {
        "Strategy": name,
        "Period": period,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{key: float(value) for key, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean())
        if "turnover" in view
        else np.nan,
    }


def option_promotion_gate(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    all_pass = True
    for name, start, end in [
        ("calibration_2007_2012", FULL_START, CALIBRATION_END),
        ("validation_2013_2017", VALIDATION_START, VALIDATION_END),
    ]:
        base = performance_summary(baseline.loc[start:end, "return"])
        option = performance_summary(candidate.loc[start:end, "return"])
        deltas = {
            metric: float(option[metric] - base[metric])
            for metric in ["CAGR", "Sharpe", "MDD"]
        }
        passes = bool(all(value >= -1e-12 for value in deltas.values()))
        checks[name] = {"deltas": deltas, "passes_all_three": passes}
        all_pass &= passes
    return {"periods": checks, "promoted": bool(all_pass)}


def run_research(save: bool = True) -> dict[str, object]:
    core = load_monthly_path(CORE_PATH)
    stage11 = load_monthly_path(STAGE11_PATH)
    router_daily, unlevered_growth = reconstruct_unlevered_router()
    risk_signals = build_causal_monthly_risk_signals(router_daily)
    meta = run_meta_strategy(core, unlevered_growth, risk_signals)
    option_candidate = apply_physical_put_candidate(meta)
    option_gate = option_promotion_gate(meta, option_candidate)
    selected = option_candidate if option_gate["promoted"] else meta
    selected_name = (
        "UnleveredVIX6LogGrowth_WithPut"
        if option_gate["promoted"]
        else "UnleveredVIX6LogGrowth_NoOption"
    )

    paths = {
        selected_name: selected,
        "UnleveredVIX6LogGrowth_NoOption": meta,
        "UnleveredVIX6LogGrowth_PutCandidate": option_candidate,
        "UnleveredVIX6GrowthSleeve": unlevered_growth,
        "Stage10_SharpeCAGR_Core": core,
        "Stage11_MSELogGrowth": stage11,
    }
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, object]] = []
    for period, start, end in [
        ("calibration_2007_2012", FULL_START, CALIBRATION_END),
        ("validation_2013_2017", VALIDATION_START, VALIDATION_END),
        ("prelock_2007_2017", FULL_START, VALIDATION_END),
        ("locked_2018_2026", LOCKED_START, common_end),
        ("full_2007_2026", FULL_START, common_end),
    ]:
        period_end = min(end, common_end)
        for name, path in paths.items():
            rows.append(
                performance_row(name, period, path, start, period_end)
            )
    comparison = pd.DataFrame(rows).drop_duplicates(
        ["Strategy", "Period"], keep="first"
    )

    full_selected = performance_summary(
        selected.loc[FULL_START:common_end, "return"]
    )
    target_stage11 = performance_summary(
        stage11.loc[FULL_START:common_end, "return"]
    )
    allocation_columns = [
        "w_stage10_core",
        "w_unlevered_vix6_growth",
        "w_option",
    ]
    solver_months = meta["solver_message"].ne("warmup_core_only")
    report = {
        "selected_strategy": selected_name,
        "objective": (
            "maximize causal expected log growth subject to a forecast "
            "Sharpe floor, historical CDaR floor, VIX6/VKOSPI risk caps, "
            "and zero leverage"
        ),
        "allocation": {
            "leverage_allowed": False,
            "maximum_observed_selected_gross_exposure": float(
                selected["gross_exposure"].max()
            ),
            "maximum_observed_growth_daily_gross_exposure": float(
                router_daily["gross_exposure"].max()
            ),
            "maximum_source_router_exposure_before_normalization": float(
                router_daily["source_gross_exposure"].max()
            ),
            "average_selected_weights": {
                column: float(selected[column].mean())
                for column in allocation_columns
            },
        },
        "fixed_constraints": {
            "forecast_sharpe_floor": FORECAST_SHARPE_FLOOR,
            "historical_cdar_floor": HISTORICAL_CDAR_FLOOR,
            "normal_growth_cap": NORMAL_GROWTH_CAP,
            "alert_growth_cap": ALERT_GROWTH_CAP,
            "crisis_growth_cap": CRISIS_GROWTH_CAP,
            "negative_momentum_cap": NEGATIVE_MOMENTUM_CAP,
            "monthly_growth_change_cap": MONTHLY_GROWTH_CHANGE_CAP,
        },
        "option_gate": option_gate,
        "option_selected": bool(option_gate["promoted"]),
        "full_selected": {
            key: float(full_selected[key])
            for key in ["CAGR", "Volatility", "Sharpe", "MDD", "Calmar"]
        },
        "full_stage11_reference": {
            key: float(target_stage11[key])
            for key in ["CAGR", "Volatility", "Sharpe", "MDD", "Calmar"]
        },
        "target_checks": {
            "cagr_at_least_10pct": bool(full_selected["CAGR"] >= 0.10),
            "sharpe_at_least_stage11": bool(
                full_selected["Sharpe"] >= target_stage11["Sharpe"]
            ),
            "mdd_no_worse_than_stage11": bool(
                full_selected["MDD"] >= target_stage11["MDD"]
            ),
        },
        "causality_and_solver": {
            "daily_router_signal_precedes_action": bool(
                (router_daily["signal_date"] < router_daily.index).all()
            ),
            "monthly_risk_signal_precedes_target": bool(
                (meta["signal_month"] < meta.index).all()
            ),
            "optimized_months": int(solver_months.sum()),
            "solver_successes": int(
                meta.loc[solver_months, "solver_success"].sum()
            ),
            "grid_fallbacks": int(
                meta.loc[solver_months, "used_grid_fallback"].sum()
            ),
            "minimum_forecast_sharpe_slack": float(
                meta.loc[solver_months, "sharpe_slack"].min()
            ),
            "minimum_historical_cdar_slack": float(
                meta.loc[solver_months, "cdar_slack"].min()
            ),
        },
        "development_status": (
            "post-lock exploratory research; 2018-2026 outcomes were already "
            "visible elsewhere in the project, so they are not an untouched "
            "holdout"
        ),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        router_daily.to_csv(OUTPUT_DIR / "unlevered_vix6_router_daily.csv")
        unlevered_growth.to_csv(
            OUTPUT_DIR / "unlevered_vix6_growth_monthly.csv"
        )
        risk_signals.to_csv(OUTPUT_DIR / "monthly_risk_signals.csv")
        meta.to_csv(OUTPUT_DIR / "meta_loggrowth_no_option.csv")
        option_candidate.to_csv(
            OUTPUT_DIR / "meta_loggrowth_put_candidate.csv"
        )
        selected.to_csv(OUTPUT_DIR / "selected_monthly.csv")
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "router_daily": router_daily,
        "unlevered_growth": unlevered_growth,
        "risk_signals": risk_signals,
        "meta": meta,
        "option_candidate": option_candidate,
        "selected": selected,
        "comparison": comparison,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print("saved", OUTPUT_DIR)


if __name__ == "__main__":
    main()
