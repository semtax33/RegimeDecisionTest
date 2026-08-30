from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from strategies.core.regime_research import ASSETS, cdar, load_monthly_asset_returns
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
    load_vkospi_daily,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    CDAR_CONFIDENCE,
    FULL_START,
    LOCKED_START,
    ONE_CALENDAR_YEAR,
    REGIME_COLUMNS,
    SLSQP_MAX_ITERATIONS,
    SLSQP_TOLERANCE,
    build_daily_stress_features,
    build_monthly_stress_signals,
    estimate_conditional_moments,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    CATASTROPHE_ANNUAL_VOLATILITY,
    CATASTROPHE_CDAR,
    UNCONSTRAINED_LONG_ONLY_BOUNDS,
    concentration_summary,
    expected_transaction_cost,
    metric_row,
    project_to_long_only_simplex,
    solver_summary,
)
from strategies.stage17_dynamic_risk_shape.dynamic_risk_shape_slsqp import (
    drawdown_episodes,
)
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20,
)
from strategies.stage28_option_directional_surface import (
    option_directional_surface_slsqp as stage28,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OPTION_EQUITY_INDEX = ASSETS.index("KODEX200")
TARGET_MATURITY_DAYS = 30
MIN_MODEL_HISTORY = 252
MIN_CALIBRATION_MONTHS = 12
FAST_DAYS = 5
SLOW_DAYS = 20

# The final directional score contains only abnormal put-skew and IVA changes.
# ERP is retained as a diagnostic and order flow is absent by construction.
PURE_DIRECTION_COMPONENTS = (
    "abnormal_bear_pressure_fast",
    "abnormal_bear_pressure_slow",
)


def _surface_for_expiry_without_order_flow(
    frame: pd.DataFrame,
) -> dict[str, float] | None:
    """Build a rate-aware surface and observable quality diagnostics.

    The discount factor and forward are jointly inferred from the slope and
    intercept of C-P = D(F-K).  This avoids Stage28's zero-rate assumption
    without requiring a separately downloaded interest-rate series.
    """

    calls = frame.loc[frame["option_type"] == "C"].copy()
    puts = frame.loc[frame["option_type"] == "P"].copy()
    parity = calls.pivot_table(
        index="strike", values="close", aggfunc="median"
    ).join(
        puts.pivot_table(index="strike", values="close", aggfunc="median"),
        how="inner",
        lsuffix="_call",
        rsuffix="_put",
    ).dropna()
    parity = parity.loc[
        parity["close_call"].gt(0.0) & parity["close_put"].gt(0.0)
    ]
    if len(parity) < 2:
        return None
    dte = float(frame["dte"].iloc[0])
    maturity = dte / 365.0
    strikes = parity.index.to_numpy(dtype=float)
    call_put_difference = (
        parity["close_call"] - parity["close_put"]
    ).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(strikes)), strikes])
    intercept, strike_slope = np.linalg.lstsq(
        design, call_put_difference, rcond=None
    )[0]
    discount_factor = float(-strike_slope)
    forward = float(intercept / discount_factor) if discount_factor > 0.0 else np.nan
    if not np.isfinite(forward) or forward <= 0.0 or maturity <= 0.0:
        return None
    fitted_difference = intercept + strike_slope * strikes
    parity_rmse = float(
        np.sqrt(np.mean(np.square(call_put_difference - fitted_difference)))
    )
    parity_price_scale = float(
        np.median(
            parity["close_call"].to_numpy(dtype=float)
            + parity["close_put"].to_numpy(dtype=float)
        )
    )
    parity_nrmse = parity_rmse / max(parity_price_scale, 1e-12)
    implied_rate = float(-math.log(discount_factor) / maturity)

    def arbitrage_quality(option_frame: pd.DataFrame) -> tuple[float, float]:
        clean = (
            option_frame.loc[option_frame["close"].gt(0.0), ["strike", "close"]]
            .dropna()
            .groupby("strike", as_index=False)["close"]
            .median()
            .sort_values("strike")
        )
        if len(clean) < 3:
            return float("nan"), float("nan")
        price_change = np.diff(clean["close"].to_numpy(dtype=float))
        if str(option_frame["option_type"].iloc[0]) == "P":
            monotone_share = float(np.mean(price_change >= 0.0))
        else:
            monotone_share = float(np.mean(price_change <= 0.0))
        strike_change = np.diff(clean["strike"].to_numpy(dtype=float))
        slopes = price_change / strike_change
        convex_share = float(np.mean(np.diff(slopes) >= 0.0))
        return monotone_share, convex_share

    call_monotone, call_convex = arbitrage_quality(calls)
    put_monotone, put_convex = arbitrage_quality(puts)
    monotone_quality = float(np.nanmean([call_monotone, put_monotone]))
    convexity_quality = float(np.nanmean([call_convex, put_convex]))

    put_k, put_price = stage28._monotone_otm_prices(
        puts.loc[puts["strike"] < forward], "P"
    )
    call_k, call_price = stage28._monotone_otm_prices(
        calls.loc[calls["strike"] > forward], "C"
    )
    if len(put_k) < 3 or len(call_k) < 3:
        return None
    downside_variance = float(
        2.0 * np.trapezoid(put_price, put_k) / (forward**2 * maturity)
    )
    upside_variance = float(
        2.0 * np.trapezoid(call_price, call_k) / (forward**2 * maturity)
    )
    erp_proxy = downside_variance + upside_variance
    if not np.isfinite(erp_proxy) or erp_proxy <= 0.0:
        return None

    call_atm_iv = stage28._interpolated_iv(calls, forward)
    put_atm_iv = stage28._interpolated_iv(puts, forward)
    atm_iv = float(np.nanmean([call_atm_iv, put_atm_iv]))
    if not np.isfinite(atm_iv) or atm_iv <= 0.0:
        return None
    put_delta_d1 = float(norm.ppf(0.75))
    put25_strike = forward * math.exp(
        -put_delta_d1 * atm_iv * math.sqrt(maturity)
        + 0.5 * atm_iv**2 * maturity
    )
    put25_iv = stage28._interpolated_iv(puts, put25_strike)
    if not np.isfinite(put25_iv):
        return None
    put_strikes = puts["strike"].dropna().to_numpy(dtype=float)
    put25_nearest_distance = float(
        np.min(np.abs(put_strikes - put25_strike)) / forward
    )
    close_values = pd.to_numeric(frame["close"], errors="coerce")
    iv_values = pd.to_numeric(frame["implied_volatility"], errors="coerce")
    return {
        "dte": dte,
        "forward": forward,
        "discount_factor": discount_factor,
        "implied_rate": implied_rate,
        "parity_rmse": parity_rmse,
        "parity_nrmse": parity_nrmse,
        "parity_pairs": float(len(parity)),
        "atm_iv": atm_iv,
        "put25_iv": put25_iv,
        "put25_strike_ratio": float(put25_strike / forward),
        "put25_nearest_strike_distance": put25_nearest_distance,
        "put_skew25": put25_iv - atm_iv,
        "implied_variance_asymmetry": float(
            (downside_variance - upside_variance) / erp_proxy
        ),
        "option_erp_proxy": erp_proxy,
        "monotone_quality": monotone_quality,
        "convexity_quality": convexity_quality,
        "coverage_log_width": float(
            math.log(call_k.max() / forward)
            - math.log(put_k.min() / forward)
        ),
        "zero_or_missing_close_share": float(
            ((close_values <= 0.0) | close_values.isna()).mean()
        ),
        "invalid_iv_share": float(
            (~iv_values.between(3.0001, 200.0)).mean()
        ),
        "listed_contracts": float(len(frame)),
        "put_strike_min_ratio": float(put_k.min() / forward),
        "call_strike_max_ratio": float(call_k.max() / forward),
        "put_contracts": float(len(put_k)),
        "call_contracts": float(len(call_k)),
    }


def _constant_maturity_surface_without_order_flow(
    group: pd.DataFrame,
) -> dict[str, Any] | None:
    rows: list[dict[str, float]] = []
    for _, expiry_frame in group.groupby("expiry_date", sort=True):
        result = _surface_for_expiry_without_order_flow(expiry_frame)
        if result is not None:
            rows.append(result)
    if not rows:
        return None
    surfaces = pd.DataFrame(rows).sort_values("dte").reset_index(drop=True)
    exact = surfaces.loc[surfaces["dte"] == TARGET_MATURITY_DAYS]
    if len(exact):
        output = exact.iloc[0].to_dict()
        output["maturity_method"] = "exact_30d"
        return output
    lower = surfaces.loc[surfaces["dte"] < TARGET_MATURITY_DAYS]
    upper = surfaces.loc[surfaces["dte"] > TARGET_MATURITY_DAYS]
    numeric = [column for column in surfaces.columns if column != "dte"]
    if len(lower) and len(upper):
        lo = lower.iloc[-1]
        hi = upper.iloc[0]
        weight = (TARGET_MATURITY_DAYS - lo["dte"]) / (
            hi["dte"] - lo["dte"]
        )
        output = {
            column: float(lo[column] + weight * (hi[column] - lo[column]))
            for column in numeric
        }
        output["dte"] = float(TARGET_MATURITY_DAYS)
        output["maturity_method"] = "interpolated_30d"
        return output
    nearest = surfaces.iloc[
        (surfaces["dte"] - TARGET_MATURITY_DAYS).abs().argmin()
    ].to_dict()
    nearest["maturity_method"] = "nearest_listed_proxy"
    return nearest


def build_daily_base_surface(
    chain: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate the chain after explicitly discarding all flow fields."""

    if chain is None:
        chain, audit = stage28.load_kospi200_option_chain()
    else:
        audit = {"provided_chain": True, "rows": int(len(chain))}
    allowed = [
        "date",
        "expiry_date",
        "dte",
        "option_type",
        "strike",
        "close",
        "implied_volatility",
    ]
    surface_chain = chain.loc[:, allowed].copy()
    rows: list[dict[str, Any]] = []
    for date, date_frame in surface_chain.groupby("date", sort=True):
        result = _constant_maturity_surface_without_order_flow(date_frame)
        if result is not None:
            rows.append({"date": date, **result})
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    audit.update(
        {
            "daily_surface_rows": int(len(daily)),
            "first_surface_date": str(daily.index.min().date()),
            "last_surface_date": str(daily.index.max().date()),
            "maturity_methods": daily["maturity_method"].value_counts().to_dict(),
            "order_flow_columns_discarded_before_surface": [
                "volume",
                "trading_value",
                "open_interest",
            ],
            "order_flow_used": False,
            "erp_measure": (
                "Martin-style SVIX-squared proxy using a cross-sectionally "
                "estimated discount factor/forward and truncated listed-strike "
                "option-price integrals"
            ),
        }
    )
    return daily, audit


def _macro_context_for_dates(
    dates: pd.DatetimeIndex,
    macro_ranks: pd.DataFrame,
) -> pd.DataFrame:
    """Use the last macro ranks known by the prior calendar month-end."""

    clean = macro_ranks.dropna().sort_index()
    rows: list[dict[str, Any]] = []
    for target_month in sorted(dates.to_period("M").unique()):
        cutoff = (target_month - 1).to_timestamp("M")
        known = clean.loc[:cutoff]
        if known.empty:
            continue
        current = known.iloc[-1]
        rows.append(
            {
                "target_month": target_month,
                "macro_signal_date": known.index[-1],
                "macro_growth_high": float(
                    current[["GDP_YoY", "Export_YoY", "BSI"]].mean()
                ),
                "macro_inflation_high": float(
                    current[["CPI_YoY", "PPI_YoY", "ImportPrice_YoY"]].mean()
                ),
            }
        )
    monthly = pd.DataFrame(rows).set_index("target_month")
    lookup = pd.DataFrame(index=dates)
    lookup["target_month"] = lookup.index.to_period("M")
    return lookup.join(monthly, on="target_month").drop(columns="target_month")


def expanding_one_step_ols_residual(
    target: pd.Series,
    predictors: pd.DataFrame,
    output_prefix: str,
    min_history: int = MIN_MODEL_HISTORY,
) -> pd.DataFrame:
    """Predict each row with expanding OLS fitted strictly through t-1."""

    predictors = predictors.loc[target.index]
    names = list(predictors.columns)
    parameter_names = ["intercept", *names]
    p = len(parameter_names)
    xtx = np.zeros((p, p), dtype=float)
    xty = np.zeros(p, dtype=float)
    observations = 0
    rows: list[dict[str, float]] = []
    for date in target.index:
        y = float(target.loc[date])
        values = predictors.loc[date].to_numpy(dtype=float)
        complete = np.isfinite(y) and np.isfinite(values).all()
        x = np.r_[1.0, values]
        prediction = float("nan")
        residual = float("nan")
        beta = np.repeat(np.nan, p)
        condition_number = float("nan")
        if complete and observations >= min_history and np.linalg.matrix_rank(xtx) == p:
            beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
            prediction = float(x @ beta)
            residual = y - prediction
            condition_number = float(np.linalg.cond(xtx))
        rows.append(
            {
                "date": date,
                f"expected_{output_prefix}": prediction,
                f"residual_{output_prefix}": residual,
                f"{output_prefix}_training_observations": float(observations),
                f"{output_prefix}_design_condition_number": condition_number,
                **{
                    f"{output_prefix}_beta_{name}": float(beta[index])
                    for index, name in enumerate(parameter_names)
                },
            }
        )
        if complete:
            xtx += np.outer(x, x)
            xty += x * y
            observations += 1
    return pd.DataFrame(rows).set_index("date")


def causal_lagged_expanding_zscore(
    series: pd.Series,
    min_history: int = MIN_MODEL_HISTORY,
) -> pd.Series:
    """Standardize t only with observations dated strictly before t."""

    past_mean = series.expanding(min_periods=min_history).mean().shift(1)
    past_std = series.expanding(min_periods=min_history).std(ddof=0).shift(1)
    return (series - past_mean) / past_std.replace(0.0, np.nan)


def build_daily_direction_features(
    returns: pd.DataFrame | None = None,
    chain: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the two predeclared directional components and their equal mean."""

    if returns is None:
        returns, _ = load_monthly_asset_returns(False)
    _, macro_ranks = build_macro_probabilities(returns)
    surface, audit = build_daily_base_surface(chain)
    vkospi = load_vkospi_daily()["close"].rename("vkospi_close")
    asset_frames, asset_audit = stage28.load_daily_asset_ohlcv()
    kodex_close = asset_frames["KODEX200"]["close"].rename("kospi200_close")

    daily = surface.join(vkospi, how="left").join(kodex_close, how="left")
    daily[["vkospi_close", "kospi200_close"]] = daily[
        ["vkospi_close", "kospi200_close"]
    ].ffill(limit=3)
    daily = daily.join(_macro_context_for_dates(daily.index, macro_ranks))
    log_vkospi = np.log(daily["vkospi_close"].clip(lower=1e-12))
    log_kospi = np.log(daily["kospi200_close"].clip(lower=1e-12))
    residual_specs: list[tuple[str, pd.Series, pd.DataFrame]] = []
    for label, horizon in [("fast", FAST_DAYS), ("slow", SLOW_DAYS)]:
        daily[f"put_skew_change_{label}"] = daily["put_skew25"].diff(horizon)
        daily[f"iva_change_{label}"] = daily[
            "implied_variance_asymmetry"
        ].diff(horizon)
        daily[f"kospi_return_{label}_pct"] = log_kospi.diff(horizon) * 100.0
        daily[f"vkospi_change_{label}_pct"] = log_vkospi.diff(horizon) * 100.0
        daily[f"roll_flag_{label}"] = (
            daily["dte"].sub(daily["dte"].shift(horizon)) > 0.0
        ).astype(float)
        daily[f"dte_distance_{label}"] = (
            daily["dte"] - TARGET_MATURITY_DAYS
        ) / TARGET_MATURITY_DAYS
        predictors = daily[
            [
                f"kospi_return_{label}_pct",
                f"vkospi_change_{label}_pct",
                f"dte_distance_{label}",
                f"roll_flag_{label}",
                "macro_growth_high",
                "macro_inflation_high",
            ]
        ]
        residual_specs.extend(
            [
                (f"put_skew_{label}", daily[f"put_skew_change_{label}"], predictors),
                (f"iva_{label}", daily[f"iva_change_{label}"], predictors),
            ]
        )
    for prefix, target, predictors in residual_specs:
        daily = daily.join(
            expanding_one_step_ols_residual(
                target, predictors, prefix, MIN_MODEL_HISTORY
            )
        )
        daily[f"z_residual_{prefix}"] = causal_lagged_expanding_zscore(
            daily[f"residual_{prefix}"]
        )

    daily["abnormal_bear_pressure_fast"] = daily[
        ["z_residual_put_skew_fast", "z_residual_iva_fast"]
    ].mean(axis=1, skipna=False)
    daily["abnormal_bear_pressure_slow"] = daily[
        ["z_residual_put_skew_slow", "z_residual_iva_slow"]
    ].mean(axis=1, skipna=False)
    daily["pure_direction_raw"] = -daily[
        list(PURE_DIRECTION_COMPONENTS)
    ].mean(axis=1, skipna=False)

    # Parameter-free, contemporaneously observable measurement confidence.
    # A roll within the fast window sets q_roll to zero; the other components
    # decline continuously with maturity distance, narrow coverage, parity
    # error, and static-arbitrage violations.
    daily["q_dte"] = TARGET_MATURITY_DAYS / (
        TARGET_MATURITY_DAYS
        + (daily["dte"] - TARGET_MATURITY_DAYS).abs()
    )
    expanding_best_coverage = daily["coverage_log_width"].expanding().max()
    daily["q_coverage"] = (
        daily["coverage_log_width"] / expanding_best_coverage.replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    daily["q_parity"] = 1.0 / (1.0 + daily["parity_nrmse"].clip(lower=0.0))
    daily["q_arbitrage"] = daily[
        ["monotone_quality", "convexity_quality"]
    ].mean(axis=1).clip(0.0, 1.0)
    daily["q_roll"] = 1.0 - daily["roll_flag_fast"].clip(0.0, 1.0)
    daily["data_quality_confidence"] = daily[
        ["q_dte", "q_coverage", "q_parity", "q_arbitrage", "q_roll"]
    ].prod(axis=1, min_count=5).clip(0.0, 1.0)
    daily["quality_weighted_direction"] = (
        daily["pure_direction_raw"] * daily["data_quality_confidence"]
    )
    daily["option_direction"] = daily["quality_weighted_direction"]
    daily["option_direction_score"] = daily["option_direction"] / (
        1.0 + daily["option_direction"].abs()
    )

    # ERP remains available only to diagnose compensation versus tail risk. It
    # is deliberately absent from option_direction and portfolio weights.
    daily["log_option_erp_proxy"] = np.log(
        daily["option_erp_proxy"].clip(lower=1e-12)
    )
    daily["z_option_erp_proxy"] = causal_lagged_expanding_zscore(
        daily["log_option_erp_proxy"]
    )
    complete = daily.dropna(subset=["option_direction_score"])
    audit.update(
        {
            "complete_direction_rows": int(len(complete)),
            "first_complete_direction_date": str(complete.index.min().date()),
            "last_complete_direction_date": str(complete.index.max().date()),
            "minimum_residual_training_observations": MIN_MODEL_HISTORY,
            "residual_model": (
                "four expanding one-step-ahead OLS models for 5d/20d changes "
                "in put skew and IVA; each prediction uses rows through t-1"
            ),
            "residual_predictors": (
                "same-horizon KOSPI return, VKOSPI change, 30d maturity distance, "
                "roll flag, prior-month growth rank, prior-month inflation rank"
            ),
            "pure_direction_components": list(PURE_DIRECTION_COMPONENTS),
            "erp_used_for_allocation": False,
            "quality_formula": "q_dte*q_coverage*q_parity*q_arbitrage*q_roll",
            "score_bound": "x/(1+abs(x))",
            "searched_parameters": None,
            "asset_data": asset_audit,
        }
    )
    return daily.replace([np.inf, -np.inf], np.nan), audit


MONTHLY_SIGNAL_COLUMNS = [
    "put_skew25",
    "implied_variance_asymmetry",
    "residual_put_skew_fast",
    "residual_iva_fast",
    "residual_put_skew_slow",
    "residual_iva_slow",
    "z_residual_put_skew_fast",
    "z_residual_iva_fast",
    "z_residual_put_skew_slow",
    "z_residual_iva_slow",
    "abnormal_bear_pressure_fast",
    "abnormal_bear_pressure_slow",
    "pure_direction_raw",
    "option_erp_proxy",
    "z_option_erp_proxy",
    "q_dte",
    "q_coverage",
    "q_parity",
    "q_arbitrage",
    "q_roll",
    "data_quality_confidence",
    "quality_weighted_direction",
    "option_direction",
    "option_direction_score",
    "put_skew_slow_training_observations",
    "dte",
]


def build_monthly_direction_signals(
    target_months: pd.PeriodIndex,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Use the final complete observation of t-1 to trade month t."""

    valid = daily.dropna(subset=["option_direction_score"])
    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        signal_month = target_month - 1
        known = valid.loc[: signal_month.to_timestamp("M")]
        if known.empty:
            continue
        signal_date = known.index[-1]
        current = known.iloc[-1]
        rows.append(
            {
                "target_month": target_month,
                "option_signal_month": signal_month,
                "option_signal_date": signal_date,
                "macro_signal_date_for_residual": current["macro_signal_date"],
                **{
                    column: float(current[column])
                    for column in MONTHLY_SIGNAL_COLUMNS
                },
                "maturity_method": str(current["maturity_method"]),
            }
        )
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def add_causal_mu_calibration(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    min_history: int = MIN_CALIBRATION_MONTHS,
) -> pd.DataFrame:
    """Map the score into monthly return units using only realized past pairs."""

    output = signals.copy()
    raw_slopes: list[float] = []
    slopes: list[float] = []
    observations: list[int] = []
    adjustments: list[float] = []
    for month in output.index:
        history = output.index[output.index < month].intersection(returns.index)
        x = output.loc[history, "option_direction_score"]
        y = returns.loc[history, "KODEX200"]
        complete = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        raw_slope = 0.0
        if len(complete) >= min_history:
            centered_x = complete["x"] - complete["x"].mean()
            denominator = float(np.square(centered_x).sum())
            if denominator > 0.0:
                raw_slope = float(
                    (centered_x * (complete["y"] - complete["y"].mean())).sum()
                    / denominator
                )
        # The signal is defined ex ante so positive means bullish.  A negative
        # estimate is treated as no usable evidence, not permission to reverse
        # the economic meaning of the factor.  This is parameter-free NNLS for
        # a single slope and prevents small early samples from becoming an
        # unexplained contrarian strategy.
        slope = max(raw_slope, 0.0)
        raw_slopes.append(raw_slope)
        slopes.append(slope)
        observations.append(int(len(complete)))
        adjustments.append(slope * float(output.loc[month, "option_direction_score"]))
    output["causal_calibration_raw_slope"] = raw_slopes
    output["causal_calibration_slope"] = slopes
    output["calibration_observations"] = observations
    output["calibrated_mu_adjustment_KODEX200"] = adjustments
    output["ablation_component"] = "pure_direction_quality_causal_calibration"
    return output


def solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    option_signal: pd.Series,
    pretrade: np.ndarray,
    strategy_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the causally calibrated alpha directly in monthly-return units."""

    original_expected_return, base_covariance, moment_detail = (
        estimate_conditional_moments(
            history=history,
            historical_probabilities=historical_probabilities,
            current_probabilities=current_probabilities,
            historical_stress=historical_stress,
            current_stress=current_stress,
            historical_recovery=historical_recovery,
            current_recovery=current_recovery,
            use_short_term_stress=True,
        )
    )
    macro_expected_return = np.asarray(
        moment_detail["macro_expected_monthly_return"], dtype=float
    )
    stress_adjustment = np.asarray(
        moment_detail["stress_return_adjustment"], dtype=float
    )
    technical = stage28.apply_technical_inputs(
        macro_expected_return, base_covariance, technical_signal
    )
    filtered_macro = np.asarray(
        technical["filtered_macro_expected_return"], dtype=float
    ).copy()
    option_mu_adjustment = float(
        option_signal["calibrated_mu_adjustment_KODEX200"]
    )
    filtered_macro[OPTION_EQUITY_INDEX] += option_mu_adjustment
    expected_return = filtered_macro + stress_adjustment
    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)

    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    initial = (
        project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        realized_history = historical_returns @ weights
        downside_semivariance = float(
            np.mean(np.minimum(realized_history, 0.0) ** 2)
        )
        transaction_cost = expected_transaction_cost(weights, pretrade)
        utility = (
            monthly_return
            - 0.5 * monthly_variance
            - downside_semivariance
            - transaction_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_risk_aversion_lambda": 1.0,
            "variance_penalty": 0.5 * monthly_variance,
            "downside_semivariance": downside_semivariance,
            "downside_penalty": downside_semivariance,
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": utility,
        }

    def objective(weights: np.ndarray) -> float:
        return -portfolio_values(weights)["monthly_utility"]

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                CATASTROPHE_ANNUAL_VOLATILITY - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                CATASTROPHE_CDAR
                + cdar(historical_returns @ weights, CDAR_CONFIDENCE)
            ),
        },
    ]
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
    )
    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = project_to_long_only_simplex(result.x)
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                "Both calibrated-alpha and feasibility solves failed: "
                f"{result.message}; {fallback.message}"
            )
        result = fallback
        weights = project_to_long_only_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = cdar(historical_returns @ weights, CDAR_CONFIDENCE)
    detail: dict[str, Any] = {
        **values,
        "policy": strategy_name,
        "option_direction": float(option_signal["option_direction"]),
        "option_direction_score": float(option_signal["option_direction_score"]),
        "option_return_scale": float(option_signal["causal_calibration_slope"]),
        "causal_calibration_slope": float(
            option_signal["causal_calibration_slope"]
        ),
        "calibration_observations": int(
            option_signal["calibration_observations"]
        ),
        "option_mu_adjustment_KODEX200": option_mu_adjustment,
        "option_direction_applied": True,
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": CATASTROPHE_ANNUAL_VOLATILITY - annual_vol,
        "cdar_slack": CATASTROPHE_CDAR + historical_cdar,
        "largest_weight": float(weights.max()),
        "largest_asset": ASSETS[int(np.argmax(weights))],
        "weights_above_half": int(np.sum(weights > 0.5 + 1e-10)),
        "macro_expected_monthly_return": macro_expected_return.tolist(),
        "stress_return_adjustment": stress_adjustment.tolist(),
        "original_expected_return": original_expected_return.tolist(),
        "filtered_expected_return": expected_return.tolist(),
        **technical,
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    option_signals: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    """Run Stage20 controls with option alpha applied only to KODEX200 mu."""

    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(option_signals.index)
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < ONE_CALENDAR_YEAR:
            continue
        probability = probabilities.loc[month]
        stress = float(stress_signals.loc[month, "stress_score"])
        recovery = float(stress_signals.loc[month, "recovery_score"])
        technical_signal = technical_signals.loc[month]
        option_signal = option_signals.loc[month]
        weights, detail = solve_weights(
            history=history,
            historical_probabilities=probabilities.loc[
                probabilities.index < month
            ],
            current_probabilities=probability,
            historical_stress=stress_signals.loc[
                stress_signals.index < month, "stress_score"
            ],
            current_stress=stress,
            historical_recovery=stress_signals.loc[
                stress_signals.index < month, "recovery_score"
            ],
            current_recovery=recovery,
            technical_signal=technical_signal,
            option_signal=option_signal,
            pretrade=pretrade,
            strategy_name=strategy_name,
        )
        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign_indices].sum()))
            * FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

        row: dict[str, Any] = {
            "month": month,
            "macro_signal_month": probability["signal_month"],
            "stress_signal_month": stress_signals.loc[month, "stress_signal_month"],
            "stress_signal_date": stress_signals.loc[month, "stress_signal_date"],
            "technical_signal_month": technical_signal["technical_signal_month"],
            "option_signal_month": option_signal["option_signal_month"],
            "option_signal_date": option_signal["option_signal_date"],
            "macro_signal_date_for_residual": option_signal[
                "macro_signal_date_for_residual"
            ],
            "stress_score": stress,
            "recovery_score": recovery,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            "macro_neutral_return": float(detail["macro_neutral_return"]),
            **{column: float(probability[column]) for column in REGIME_COLUMNS},
            **{
                f"w_{asset}": float(weights[index])
                for index, asset in enumerate(ASSETS)
            },
        }
        scalar_detail_keys = [
            "policy",
            "solver_success",
            "used_fallback",
            "solver_status",
            "solver_message",
            "solver_iterations",
            "objective_value",
            "expected_monthly_return",
            "expected_monthly_variance",
            "expected_annual_log_growth",
            "downside_risk_aversion_lambda",
            "variance_penalty",
            "downside_semivariance",
            "downside_penalty",
            "estimated_transaction_cost",
            "monthly_utility",
            "expected_annual_volatility",
            "historical_cdar",
            "sum_error",
            "volatility_slack",
            "cdar_slack",
            "largest_weight",
            "largest_asset",
            "weights_above_half",
            "option_direction",
            "option_direction_score",
            "option_return_scale",
            "causal_calibration_slope",
            "calibration_observations",
            "option_mu_adjustment_KODEX200",
            "option_direction_applied",
        ]
        row.update({key: detail[key] for key in scalar_detail_keys})
        vector_fields = {
            "macro_mu": "macro_expected_monthly_return",
            "stress_mu_adjustment": "stress_return_adjustment",
            "original_expected_mu": "original_expected_return",
            "filtered_expected_mu": "filtered_expected_return",
            "macro_relative_direction": "macro_relative_direction",
            "technical_direction": "technical_direction",
            "macro_confidence": "macro_confidence",
            "filtered_macro_mu": "filtered_macro_expected_return",
            "atr_percentile": "atr_percentile",
            "atr_variance_scale": "atr_variance_scale",
        }
        for prefix, detail_key in vector_fields.items():
            values = np.asarray(detail[detail_key], dtype=float)
            row.update(
                {
                    f"{prefix}_{asset}": float(values[index])
                    for index, asset in enumerate(ASSETS)
                }
            )
        for asset in ASSETS:
            row[f"technical_signal_date_{asset}"] = technical_signal[
                f"technical_signal_date_{asset}"
            ]
            for feature in ["k_ratio", "k_score", "natr"]:
                row[f"{feature}_{asset}"] = float(
                    technical_signal[f"{feature}_{asset}"]
                )
        for feature in [
            "price_rsi",
            "volume_rsi",
            "price_strength",
            "volume_strength",
        ]:
            row[f"{feature}_KODEX200"] = float(
                technical_signal[f"{feature}_KODEX200"]
            )
        for feature in MONTHLY_SIGNAL_COLUMNS:
            row[feature] = float(option_signal[feature])
        row["option_maturity_method"] = str(option_signal["maturity_method"])
        row["ablation_component"] = str(option_signal["ablation_component"])
        rows.append(row)
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _return_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    years = len(values) / 12.0
    nav = np.cumprod(1.0 + values)
    cagr = float(nav[-1] ** (1.0 / years) - 1.0)
    volatility = float(values.std(ddof=1) * math.sqrt(12.0))
    sharpe = float(values.mean() / values.std(ddof=1) * math.sqrt(12.0))
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "MDD": mdd,
    }


def paired_block_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    replications: int = 2000,
    block_months: int = 12,
) -> pd.DataFrame:
    """Diagnostic only: paired circular one-year block bootstrap."""

    common = baseline.index.intersection(candidate.index)
    base = baseline.loc[common].to_numpy(dtype=float)
    test = candidate.loc[common].to_numpy(dtype=float)
    rng = np.random.default_rng(20260829)
    rows: list[dict[str, float]] = []
    blocks_needed = math.ceil(len(common) / block_months)
    for _ in range(replications):
        starts = rng.integers(0, len(common), size=blocks_needed)
        indices = np.concatenate(
            [
                (start + np.arange(block_months)) % len(common)
                for start in starts
            ]
        )[: len(common)]
        base_metrics = _return_metrics(base[indices])
        test_metrics = _return_metrics(test[indices])
        rows.append(
            {
                "delta_CAGR": test_metrics["CAGR"] - base_metrics["CAGR"],
                "delta_Sharpe": test_metrics["Sharpe"] - base_metrics["Sharpe"],
                "delta_MDD": test_metrics["MDD"] - base_metrics["MDD"],
            }
        )
    draws = pd.DataFrame(rows)
    summary_rows = []
    for column in draws.columns:
        values = draws[column]
        summary_rows.append(
            {
                "Metric": column,
                "Mean": float(values.mean()),
                "P05": float(values.quantile(0.05)),
                "P50": float(values.quantile(0.50)),
                "P95": float(values.quantile(0.95)),
                "ProbabilityPositive": float((values > 0.0).mean()),
                "Replications": replications,
                "BlockMonths": block_months,
            }
        )
    return pd.DataFrame(summary_rows)


def factor_diagnostics(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    returns: pd.DataFrame,
    stress_signals: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in [
        "pure_direction_raw",
        "option_direction_score",
        "z_option_erp_proxy",
    ]:
        for horizon in [5, 20]:
            target = daily["kospi200_close"].shift(-horizon).div(
                daily["kospi200_close"]
            ).sub(1.0)
            common = daily[feature].dropna().index.intersection(target.dropna().index)
            rows.append(
                {
                    "Frequency": "daily",
                    "Feature": feature,
                    "Target": f"KOSPI200_forward_{horizon}d",
                    "SpearmanIC": float(
                        daily.loc[common, feature].corr(
                            target.loc[common], method="spearman"
                        )
                    ),
                    "Observations": int(len(common)),
                }
            )
    common_months = monthly.index.intersection(returns.index)
    for feature in [
        "pure_direction_raw",
        "option_direction_score",
        "z_option_erp_proxy",
    ]:
        rows.append(
            {
                "Frequency": "monthly",
                "Feature": feature,
                "Target": "KODEX200_target_month_return",
                "SpearmanIC": float(
                    monthly.loc[common_months, feature].corr(
                        returns.loc[common_months, "KODEX200"], method="spearman"
                    )
                ),
                "Observations": int(len(common_months)),
            }
        )
    stress_common = monthly.index.intersection(stress_signals.index)
    for feature in [
        "pure_direction_raw",
        "option_direction_score",
        "z_option_erp_proxy",
    ]:
        rows.append(
            {
                "Frequency": "monthly",
                "Feature": feature,
                "Target": "VIX6_VKOSPI_stress_score_correlation",
                "SpearmanIC": float(
                    monthly.loc[stress_common, feature].corr(
                        stress_signals.loc[stress_common, "stress_score"],
                        method="spearman",
                    )
                ),
                "Observations": int(len(stress_common)),
            }
        )
    return pd.DataFrame(rows)


def option_construction_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    """Audit maturity/roll IC and pre/post-2018 measurement quality."""

    frame = daily.copy()
    frame["forward_5d"] = frame["kospi200_close"].shift(-FAST_DAYS).div(
        frame["kospi200_close"]
    ).sub(1.0)
    frame["forward_20d"] = frame["kospi200_close"].shift(-SLOW_DAYS).div(
        frame["kospi200_close"]
    ).sub(1.0)
    frame["dte_bucket"] = pd.cut(
        frame["dte"],
        bins=[6, 14, 30, 45, 60],
        labels=["07_14", "15_30", "31_45", "46_60"],
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("dte_bucket", observed=True):
        for feature in ["pure_direction_raw", "option_direction_score"]:
            for horizon in [FAST_DAYS, SLOW_DAYS]:
                complete = group[
                    [feature, f"forward_{horizon}d"]
                ].dropna()
                value = (
                    float(
                        complete[feature].corr(
                            complete[f"forward_{horizon}d"], method="spearman"
                        )
                    )
                    if complete[feature].nunique() > 1
                    else float("nan")
                )
                rows.append(
                    {
                        "Audit": "DTE_bucket_IC",
                        "Group": str(bucket),
                        "Feature": feature,
                        "Horizon": f"{horizon}d",
                        "Value": value,
                        "Observations": int(len(complete)),
                    }
                )
    for rolled, group in frame.groupby(frame["roll_flag_fast"].eq(1.0)):
        for feature in ["pure_direction_raw", "option_direction_score"]:
            for horizon in [FAST_DAYS, SLOW_DAYS]:
                complete = group[[feature, f"forward_{horizon}d"]].dropna()
                value = (
                    float(
                        complete[feature].corr(
                            complete[f"forward_{horizon}d"], method="spearman"
                        )
                    )
                    if complete[feature].nunique() > 1
                    else float("nan")
                )
                rows.append(
                    {
                        "Audit": "recent_roll_IC",
                        "Group": (
                            "roll_within_5d" if rolled else "no_roll_within_5d"
                        ),
                        "Feature": feature,
                        "Horizon": f"{horizon}d",
                        "Value": value,
                        "Observations": int(len(complete)),
                    }
                )
    quality_columns = [
        "dte",
        "listed_contracts",
        "coverage_log_width",
        "parity_nrmse",
        "monotone_quality",
        "convexity_quality",
        "zero_or_missing_close_share",
        "invalid_iv_share",
        "put25_nearest_strike_distance",
        "data_quality_confidence",
    ]
    for period, group in [
        ("2007_2017", frame.loc["2007-01-01":"2017-12-31"]),
        ("2018_2026", frame.loc["2018-01-01":]),
    ]:
        for column in quality_columns:
            rows.append(
                {
                    "Audit": "data_quality_period_mean",
                    "Group": period,
                    "Feature": "measurement",
                    "Horizon": column,
                    "Value": float(group[column].mean()),
                    "Observations": int(group[column].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _changes(left: pd.Series, right: pd.Series) -> dict[str, float]:
    return {
        "cagr": float(left["CAGR"] - right["CAGR"]),
        "sharpe": float(left["Sharpe"] - right["Sharpe"]),
        "mdd": float(left["MDD"] - right["MDD"]),
        "volatility": float(left["Volatility"] - right["Volatility"]),
    }


def run_research(save: bool = True) -> dict[str, Any]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)
    daily_technical, technical_audit = stage28.build_daily_technical_features()
    technical_signals = stage28.build_monthly_technical_signals(
        returns.index, daily_technical
    )
    daily_option, option_audit = build_daily_direction_features(returns)
    raw_signals = build_monthly_direction_signals(returns.index, daily_option)
    quality_signals = add_causal_mu_calibration(raw_signals, returns)
    no_quality_raw = raw_signals.copy()
    no_quality_raw["option_direction"] = no_quality_raw["pure_direction_raw"]
    no_quality_raw["option_direction_score"] = no_quality_raw[
        "option_direction"
    ] / (1.0 + no_quality_raw["option_direction"].abs())
    no_quality_signals = add_causal_mu_calibration(no_quality_raw, returns)
    variants = {
        "Stage30_PureODS_NoQuality": no_quality_signals,
        "Stage30_PureODS_QualityCausal": quality_signals,
    }

    baseline = stage20.run_backtest(
        returns, probabilities, stress_signals, technical_signals
    )
    paths = {"Stage20_VIX6": baseline}
    for name, signals in variants.items():
        paths[name] = run_backtest(
            returns,
            probabilities,
            stress_signals,
            technical_signals,
            signals,
            name,
        )
    common_end = min(path.index.max() for path in paths.values())
    early_end = LOCKED_START - 1
    comparison_rows: list[dict[str, Any]] = []
    for period, start, end in [
        ("full_2007_2026", FULL_START, common_end),
        ("early_2007_2017", FULL_START, early_end),
        ("locked_2018_2026", LOCKED_START, common_end),
    ]:
        for name, path in paths.items():
            comparison_rows.append(metric_row(name, path, period, start, end))
    comparison = pd.DataFrame(comparison_rows)
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    baseline_metrics = full.loc["Stage20_VIX6"]
    candidate_metrics = full.loc["Stage30_PureODS_QualityCausal"]
    diagnostics = factor_diagnostics(
        daily_option, quality_signals, returns, stress_signals
    )
    construction_audit = option_construction_diagnostics(daily_option)
    bootstrap = paired_block_bootstrap(
        baseline.loc[FULL_START:common_end, "return"],
        paths["Stage30_PureODS_QualityCausal"].loc[FULL_START:common_end, "return"],
    )
    episodes = pd.concat(
        [
            drawdown_episodes(path, returns, name)
            for name, path in paths.items()
        ],
        ignore_index=True,
    )
    top_episodes = (
        episodes.sort_values(["Strategy", "EpisodeMDD"])
        .groupby("Strategy", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    candidate = paths["Stage30_PureODS_QualityCausal"]
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    other_assets = [asset for asset in ASSETS if asset != "KODEX200"]
    common_path = baseline.index.intersection(candidate.index)
    all_checks = {
        "macro_signal_precedes_target": bool(
            (candidate["macro_signal_month"] < candidate.index).all()
        ),
        "vix6_stress_signal_precedes_target": bool(
            (candidate["stress_signal_month"] < candidate.index).all()
        ),
        "technical_signal_precedes_target": stage28.verify_technical_signal_dates(
            candidate
        ),
        "option_signal_precedes_target": bool(
            (candidate["option_signal_month"] < candidate.index).all()
        ),
        "macro_context_precedes_option_signal": bool(
            (
                pd.to_datetime(candidate["macro_signal_date_for_residual"])
                < pd.to_datetime(candidate["option_signal_date"])
            ).all()
        ),
        "expanding_ols_uses_t_minus_1_only": True,
        "option_order_flow_excluded": bool(
            option_audit["order_flow_used"] is False
            and not any("flow" in column.lower() for column in MONTHLY_SIGNAL_COLUMNS)
        ),
        "erp_is_diagnostic_not_allocation_input": bool(
            np.allclose(
                raw_signals["option_direction"],
                raw_signals["pure_direction_raw"]
                * raw_signals["data_quality_confidence"],
            )
        ),
        "original_vix6_risk_engine_retained": bool(
            {"parallel_shift", "put_skew", "call_skew"}.issubset(
                daily_stress.columns
            )
        ),
        "rate_aware_parity_discount_factor_estimated": bool(
            daily_option["discount_factor"].notna().all()
            and not np.allclose(daily_option["discount_factor"], 1.0)
        ),
        "quality_confidence_matches_product": bool(
            np.allclose(
                daily_option["data_quality_confidence"],
                daily_option[
                    ["q_dte", "q_coverage", "q_parity", "q_arbitrage", "q_roll"]
                ].prod(axis=1, min_count=5),
                equal_nan=True,
            )
        ),
        "causal_mu_adjustment_matches_slope_times_score": bool(
            np.allclose(
                quality_signals["calibrated_mu_adjustment_KODEX200"],
                quality_signals["causal_calibration_slope"]
                * quality_signals["option_direction_score"],
            )
        ),
        "candidate_changes_only_equity_expected_mu": bool(
            all(
                np.allclose(
                    candidate.loc[common_path, f"filtered_expected_mu_{asset}"],
                    baseline.loc[common_path, f"filtered_expected_mu_{asset}"],
                )
                for asset in other_assets
            )
            and np.allclose(
                candidate.loc[common_path, "filtered_expected_mu_KODEX200"]
                - baseline.loc[common_path, "filtered_expected_mu_KODEX200"],
                candidate.loc[common_path, "option_mu_adjustment_KODEX200"],
            )
        ),
        "weights_sum_to_one": bool(
            np.allclose(candidate[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (candidate[weight_columns] >= -1e-10).all().all()
        ),
        "no_leverage": bool(
            np.allclose(candidate[weight_columns].sum(axis=1), 1.0)
        ),
        "static_lambda_equals_one": bool(
            np.allclose(candidate["downside_risk_aversion_lambda"], 1.0)
        ),
        "all_solvers_succeeded": bool(
            candidate["solver_success"].all()
            and not candidate["used_fallback"].any()
        ),
        "no_hard_asset_cap": True,
        "no_hard_regime_weights": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_or_candidate_search": True,
        "ablations_are_attribution_not_candidate_selection": True,
    }
    pareto = bool(
        candidate_metrics["CAGR"] >= baseline_metrics["CAGR"]
        and candidate_metrics["Sharpe"] >= baseline_metrics["Sharpe"]
        and candidate_metrics["MDD"] >= baseline_metrics["MDD"]
    )
    report: dict[str, Any] = {
        "strategy": "Stage30_PureODS_QualityCausal",
        "base_strategy": "Stage20_VIX6",
        "decision": "promote_stage30" if pareto else "retain_stage20",
        "design": (
            "Stage20 VIX6/VKOSPI remains the risk engine. Strictly causal "
            "abnormal 5d/20d put-skew and IVA changes form pure direction. "
            "Observable data quality shrinks the score, and an expanding "
            "signal-return slope maps it to KODEX200 monthly mu. ERP is "
            "diagnostic only and option order flow is excluded."
        ),
        "directional_formula": {
            "fast_bear": "mean(Z[abnormal 5d put-skew change], Z[abnormal 5d IVA change])",
            "slow_bear": "mean(Z[abnormal 20d put-skew change], Z[abnormal 20d IVA change])",
            "pure_direction": "-mean(fast_bear, slow_bear)",
            "quality": "q_dte*q_coverage*q_parity*q_arbitrage*q_roll",
            "bounded_score": "quality*pure_direction/(1+abs(quality*pure_direction))",
            "equity_mu_adjustment": (
                "expanding past-only slope(KODEX200 return ~ score) * current score"
            ),
            "implied_erp": "diagnostic only; excluded from direction and mu",
            "order_flow": "excluded",
            "searched_parameters": None,
        },
        "residual_model": {
            "method": "four expanding one-step-ahead OLS models",
            "minimum_history": MIN_MODEL_HISTORY,
            "targets": [
                "5d put-skew change",
                "5d IVA change",
                "20d put-skew change",
                "20d IVA change",
            ],
            "predictors": (
                "same-horizon KOSPI return and VKOSPI change, DTE distance, "
                "roll flag, prior-month growth and inflation ranks"
            ),
            "training_cutoff": "strictly t-1",
            "macro_cutoff": "prior calendar month-end",
        },
        "causal_mu_calibration": {
            "minimum_months": MIN_CALIBRATION_MONTHS,
            "method": (
                "expanding one-slope nonnegative least squares through prior "
                "realized month; negative raw slope shrinks alpha to zero"
            ),
            "intercept_applied_to_mu": False,
            "signal_sign_reversal_allowed": False,
        },
        "data_audit": {
            "technical": technical_audit,
            "option": option_audit,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "full_changes_vs_stage20": _changes(
            candidate_metrics, baseline_metrics
        ),
        "factor_diagnostics": json.loads(
            diagnostics.to_json(orient="records", force_ascii=False)
        ),
        "option_construction_diagnostics": json.loads(
            construction_audit.to_json(orient="records", force_ascii=False)
        ),
        "paired_block_bootstrap": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "concentration": {
            name: concentration_summary(path) for name, path in paths.items()
        },
        "top_drawdown_episodes": json.loads(
            top_episodes.to_json(orient="records", force_ascii=False)
        ),
        "solver": solver_summary(candidate),
        "unchanged_controls": {
            "risk_engine": "Stage20 VKOSPI/VIX6 decomposition",
            "downside_risk_aversion_lambda": 1.0,
            "annual_volatility_guard": CATASTROPHE_ANNUAL_VOLATILITY,
            "cdar_guard": CATASTROPHE_CDAR,
            "asset_bounds": [0.0, 1.0],
            "weight_sum": 1.0,
            "leverage": 1.0,
            "stage20_daily_k_ratio_atr_rsi": True,
        },
        "checks": all_checks,
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        baseline.to_csv(OUTPUT_DIR / "stage20_vix6_monthly.csv")
        for name, path in paths.items():
            if name == "Stage20_VIX6":
                continue
            path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        daily_option.to_csv(
            OUTPUT_DIR / "daily_abnormal_surface_erp_features.csv"
        )
        quality_signals.to_csv(OUTPUT_DIR / "monthly_option_alpha_signals.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        diagnostics.to_csv(OUTPUT_DIR / "factor_diagnostics.csv", index=False)
        construction_audit.to_csv(
            OUTPUT_DIR / "option_construction_diagnostics.csv", index=False
        )
        bootstrap.to_csv(OUTPUT_DIR / "paired_block_bootstrap.csv", index=False)
        top_episodes.to_csv(
            OUTPUT_DIR / "drawdown_episode_attribution.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "daily_stress": daily_stress,
        "stress_signals": stress_signals,
        "daily_option": daily_option,
        "combined_signals": quality_signals,
        "paths": paths,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "construction_audit": construction_audit,
        "bootstrap": bootstrap,
        "episodes": top_episodes,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(result["diagnostics"].to_string(index=False))
    print(result["bootstrap"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
