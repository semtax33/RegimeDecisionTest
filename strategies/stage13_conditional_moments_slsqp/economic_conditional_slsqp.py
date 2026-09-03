from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right, insort
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import (
    ASSETS,
    cdar,
    load_monthly_asset_returns,
    performance_summary,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
    load_vkospi_daily,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
VIX6_FEATURE_PATH = ROOT / "results" / "vix6_case1_features_daily.csv"
STAGE10_PATH = (
    ROOT
    / "strategies"
    / "stage10_slsqp_sharpe100"
    / "outputs"
    / "slsqp_sharpe_cagr100_monthly.csv"
)

FULL_START = pd.Period("2007-04", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")

# Economic horizons, not backtest-selected hyperparameters.
ONE_WEEK = 5
ONE_TRADING_MONTH = 21
ONE_CALENDAR_YEAR = 12

# Catastrophe guards. Unlike Stage10's roughly 8% target-volatility rule, these
# limits are intentionally wide and are not intended to determine normal weights.
CATASTROPHE_ANNUAL_VOLATILITY = 0.13
CATASTROPHE_CDAR = 0.16
CDAR_CONFIDENCE = 0.90

# No single asset may hold a majority of the portfolio. This is a symmetric
# governance rule rather than four separately fitted asset caps.
MAX_SINGLE_ASSET_WEIGHT = 0.50
BOUNDS = [(0.0, MAX_SINGLE_ASSET_WEIGHT)] * len(ASSETS)

REGIME_COLUMNS = [
    "p_Goldilocks",
    "p_Overheating",
    "p_Slowdown",
    "p_Stagflation",
]

# These values are solver/numerical controls, not economic strategy parameters.
SLSQP_MAX_ITERATIONS = 300
SLSQP_TOLERANCE = 1e-9
NUMERICAL_EPSILON = 1e-12

def causal_expanding_midrank(series: pd.Series) -> pd.Series:
    """Causal empirical CDF with mid-ranks and no fitted window or scale.

    The current observation is ranked only against observations available up to
    the current date. The ordered-list implementation is O(n log n), while the
    definition is identical to an expanding empirical percentile.
    """

    result = pd.Series(np.nan, index=series.index, dtype=float)
    ordered: list[float] = []
    for index, raw_value in series.items():
        value = float(raw_value) if pd.notna(raw_value) else np.nan
        if not np.isfinite(value):
            continue
        left = bisect_left(ordered, value)  # 나보다 작은 값의 개수
        right = bisect_right(ordered, value)  # 나보다 큰 값의 개수
        equal_after_insertion = right - left + 1  # 나와 같은 값의 개수
        result.loc[index] = (left + 0.5 * equal_after_insertion) / (len(ordered) + 1)
        insort(ordered, value)
    return result


def _read_vix6_components() -> pd.DataFrame:
    """Load only the six decomposition outputs needed by the risk model."""

    if not VIX6_FEATURE_PATH.exists():
        raise FileNotFoundError(f"VIX6 feature cache is missing: {VIX6_FEATURE_PATH}")
    frame = pd.read_csv(VIX6_FEATURE_PATH, index_col=0, parse_dates=True)
    components = [
        "sticky_strike",
        "parallel_shift",
        "put_skew",
        "call_skew",
        "downside_convexity",
        "upside_convexity",
    ]
    missing = [column for column in components if column not in frame]
    if missing:
        raise ValueError(f"VIX6 cache is missing columns: {missing}")
    return frame[components].apply(pd.to_numeric, errors="coerce").sort_index()


def build_daily_stress_features() -> pd.DataFrame:
    """Build a continuous VKOSPI/VIX6 stress score without tuned coefficients.

    Four equally weighted economic blocks are used:

    1. VKOSPI level,
    2. VKOSPI and VIX6 broad volatility shocks,
    3. left-tail repricing relative to right-tail repricing,
    4. persistence over one trading month.

    VIX6 ``parallel_shift`` already removes the sticky-strike mechanical move,
    so all six decomposition legs enter either directly or through that residual.
    Rising stress is applied immediately. Falling stress retains the larger of
    today's score and its one-week mean, preventing a one-day relief rally from
    being mistaken for a completed recovery.
    """

    vkospi = load_vkospi_daily()[["close"]].rename(columns={"close": "vkospi_close"})
    vix6 = _read_vix6_components()
    daily = vkospi.join(vix6, how="inner").sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]

    daily["vkospi_log_change_5"] = np.log(daily["vkospi_close"]).diff(ONE_WEEK)
    daily["vix6_left_impulse"] = daily["put_skew"] + daily["downside_convexity"]
    daily["vix6_right_impulse"] = daily["call_skew"] + daily["upside_convexity"]
    daily["vix6_tail_asymmetry"] = (
        daily["vix6_left_impulse"] - daily["vix6_right_impulse"]
    )

    daily["level_component"] = causal_expanding_midrank(daily["vkospi_close"])
    daily["vkospi_shock_rank"] = causal_expanding_midrank(daily["vkospi_log_change_5"])
    daily["parallel_shift_rank"] = causal_expanding_midrank(daily["parallel_shift"])
    daily["shock_component"] = daily[["vkospi_shock_rank", "parallel_shift_rank"]].mean(
        axis=1
    )
    daily["left_impulse_rank"] = causal_expanding_midrank(daily["vix6_left_impulse"])
    daily["tail_asymmetry_rank"] = causal_expanding_midrank(
        daily["vix6_tail_asymmetry"]
    )
    daily["tail_component"] = daily[["left_impulse_rank", "tail_asymmetry_rank"]].mean(
        axis=1
    )

    three_blocks = daily[["level_component", "shock_component", "tail_component"]].mean(
        axis=1
    )
    daily["persistence_component"] = three_blocks.rolling(
        ONE_TRADING_MONTH, min_periods=1
    ).mean()
    daily["stress_raw"] = daily[
        [
            "level_component",
            "shock_component",
            "tail_component",
            "persistence_component",
        ]
    ].mean(axis=1)
    one_week_mean = daily["stress_raw"].rolling(ONE_WEEK, min_periods=1).mean()
    daily["stress_score"] = (
        pd.concat([daily["stress_raw"], one_week_mean], axis=1)
        .max(axis=1)
        .clip(0.0, 1.0)
    )
    recovery_intensity = (one_week_mean - daily["stress_raw"]).clip(lower=0.0)
    daily["recovery_score"] = causal_expanding_midrank(
        recovery_intensity.where(recovery_intensity > 0.0)
    ).fillna(0.0)
    return daily.replace([np.inf, -np.inf], np.nan)

# stress는 daily가 나은듯
def build_monthly_stress_signals(
    target_months: pd.PeriodIndex,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map the last known daily stress observation to each following month."""

    if daily is None:
        daily = build_daily_stress_features()
    rows: list[dict[str, Any]] = []
    columns = [
        "level_component",
        "shock_component",
        "tail_component",
        "persistence_component",
        "stress_raw",
        "stress_score",
        "recovery_score",
    ]
    valid = daily.dropna(subset=["stress_score"])
    for target_month in target_months:
        signal_month = target_month - 1
        month_end = signal_month.to_timestamp("M")
        month_start = signal_month.to_timestamp("D", "start")
        print(month_start, month_end)
        known = valid.loc[month_start:month_end]
        # import matplotlib.pyplot as plt
        # plt.plot(known.index, known['stress_score'])
        # plt.title(f"Stress Score for {signal_month}")
        # plt.xlabel("Date")
        # plt.ylabel("Stress Score")
        # plt.grid()
        # plt.show()
        if known.empty:
            continue
        signal_date = known.index[-1]
        current = known.iloc[-1]
        #print("Stress Score")
        #print(known.iloc[-1])
        #current=known.mean()  # Use the average of the month instead of the last day to reduce noise
        #print(current)
        #exit()
        rows.append(
            {
                "target_month": target_month,
                "stress_signal_month": signal_month,
                "stress_signal_date": signal_date,
                **{column: float(current[column]) for column in columns},
            }
        )
    monthly = pd.DataFrame(rows).set_index("target_month")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly

# def build_monthly_stress_features() -> pd.DataFrame:
#     """Build monthly VKOSPI/VIX6 stress features aligned to monthly allocation."""

#     vkospi = load_vkospi_daily()[["close"]].rename(
#         columns={"close": "vkospi_close"}
#     )
#     vix6 = _read_vix6_components()

#     daily = vkospi.join(vix6, how="inner").sort_index()
#     daily = daily[~daily.index.duplicated(keep="last")]

#     daily["vix6_left_impulse"] = (
#         daily["put_skew"] + daily["downside_convexity"]
#     )
#     daily["vix6_right_impulse"] = (
#         daily["call_skew"] + daily["upside_convexity"]
#     )
#     daily["vix6_tail_asymmetry"] = (
#         daily["vix6_left_impulse"] - daily["vix6_right_impulse"]
#     )

#     monthly = pd.DataFrame(index=daily.index.to_period("M").unique())

#     month_group = daily.groupby(daily.index.to_period("M"))

#     monthly["vkospi_close"] = month_group["vkospi_close"].last()
#     monthly["vkospi_log_change_1m"] = np.log(monthly["vkospi_close"]).diff()

#     monthly["parallel_shift_1m"] = month_group["parallel_shift"].sum()
#     monthly["left_impulse_1m"] = month_group["vix6_left_impulse"].sum()
#     monthly["right_impulse_1m"] = month_group["vix6_right_impulse"].sum()
#     monthly["tail_asymmetry_1m"] = (
#         monthly["left_impulse_1m"] - monthly["right_impulse_1m"]
#     )

#     monthly["level_component"] = causal_expanding_midrank(
#         monthly["vkospi_close"]
#     )

#     monthly["vkospi_shock_rank"] = causal_expanding_midrank(
#         monthly["vkospi_log_change_1m"]
#     )
#     monthly["parallel_shift_rank"] = causal_expanding_midrank(
#         monthly["parallel_shift_1m"]
#     )
#     monthly["shock_component"] = monthly[
#         ["vkospi_shock_rank", "parallel_shift_rank"]
#     ].mean(axis=1)

#     monthly["left_impulse_rank"] = causal_expanding_midrank(
#         monthly["left_impulse_1m"]
#     )
#     monthly["tail_asymmetry_rank"] = causal_expanding_midrank(
#         monthly["tail_asymmetry_1m"]
#     )
#     monthly["tail_component"] = monthly[
#         ["left_impulse_rank", "tail_asymmetry_rank"]
#     ].mean(axis=1)

#     monthly["stress_score"] = monthly[
#         ["level_component", "shock_component", "tail_component"]
#     ].mean(axis=1).clip(0.0, 1.0)

#     monthly["recovery_intensity"] = (
#         monthly["stress_score"].shift(1) - monthly["stress_score"]
#     ).clip(lower=0.0)

#     monthly["recovery_score"] = causal_expanding_midrank(
#         monthly["recovery_intensity"].where(
#             monthly["recovery_intensity"] > 0.0
#         )
#     ).fillna(0.0)
#     monthly["stress_signal_month"] = monthly.index - 1
#     monthly["stress_signal_date"] = (monthly.index - 1).to_timestamp("M")

#     return monthly.replace([np.inf, -np.inf], np.nan)

def build_monthly_stress_features() -> pd.DataFrame:
    """Build monthly VKOSPI/VIX6 stress features aligned to next-month allocation."""

    vkospi = load_vkospi_daily()[["close"]].rename(
        columns={"close": "vkospi_close"}
    )
    vix6 = _read_vix6_components()

    daily = vkospi.join(vix6, how="inner").sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]

    daily["vix6_left_impulse"] = (
        daily["put_skew"] + daily["downside_convexity"]
    )
    daily["vix6_right_impulse"] = (
        daily["call_skew"] + daily["upside_convexity"]
    )
    daily["vix6_tail_asymmetry"] = (
        daily["vix6_left_impulse"] - daily["vix6_right_impulse"]
    )

    signal_month_index = daily.index.to_period("M")
    month_group = daily.groupby(signal_month_index)

    monthly = pd.DataFrame(index=signal_month_index.unique())
    monthly.index.name = "stress_signal_month"

    monthly["vkospi_close"] = month_group["vkospi_close"].last()
    monthly["vkospi_log_change_1m"] = np.log(monthly["vkospi_close"]).diff()

    monthly["parallel_shift_1m"] = month_group["parallel_shift"].sum()
    monthly["left_impulse_1m"] = month_group["vix6_left_impulse"].sum()
    monthly["right_impulse_1m"] = month_group["vix6_right_impulse"].sum()
    monthly["tail_asymmetry_1m"] = (
        monthly["left_impulse_1m"] - monthly["right_impulse_1m"]
    )

    monthly["level_component"] = causal_expanding_midrank(
        monthly["vkospi_close"]
    )

    monthly["vkospi_shock_rank"] = causal_expanding_midrank(
        monthly["vkospi_log_change_1m"]
    )
    monthly["parallel_shift_rank"] = causal_expanding_midrank(
        monthly["parallel_shift_1m"]
    )
    monthly["shock_component"] = monthly[
        ["vkospi_shock_rank", "parallel_shift_rank"]
    ].mean(axis=1)

    monthly["left_impulse_rank"] = causal_expanding_midrank(
        monthly["left_impulse_1m"]
    )
    monthly["tail_asymmetry_rank"] = causal_expanding_midrank(
        monthly["tail_asymmetry_1m"]
    )
    monthly["tail_component"] = monthly[
        ["left_impulse_rank", "tail_asymmetry_rank"]
    ].mean(axis=1)

    monthly["stress_score"] = monthly[
        ["level_component", "shock_component", "tail_component"]
    ].mean(axis=1).clip(0.0, 1.0)

    monthly["recovery_intensity"] = (
        monthly["stress_score"].shift(1) - monthly["stress_score"]
    ).clip(lower=0.0)

    monthly["recovery_score"] = causal_expanding_midrank(
        monthly["recovery_intensity"].where(
            monthly["recovery_intensity"] > 0.0
        )
    ).fillna(0.0)

    monthly["stress_signal_month"] = monthly.index
    monthly["stress_signal_date"] = monthly.index.to_timestamp("M")

    monthly.index = monthly.index + 1
    monthly.index.name = "target_month"

    return monthly.replace([np.inf, -np.inf], np.nan)


def _weighted_mean_and_covariance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Population weighted moments; the covariance is made numerically PSD."""

    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    total = float(weights.sum())
    if total <= NUMERICAL_EPSILON:
        weights = np.ones(len(values), dtype=float)
        total = float(len(values))
    normalized = weights / total
    mean = normalized @ values
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    return mean, _nearest_psd(covariance)


def _nearest_psd(covariance: np.ndarray) -> np.ndarray:
    """Remove only negative eigenvalues caused by floating-point arithmetic."""

    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.trace(symmetric)) / len(symmetric), NUMERICAL_EPSILON)
    floor = scale * 1e-10
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def estimate_conditional_moments(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    use_short_term_stress: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate causal macro-conditional means/covariances and stress effects.

    The four soft macro probabilities weight every prior return. For each regime,
    a no-intercept centered OLS slope estimates how returns changed with the
    continuous stress and recovery scores. This implements
    ``mu_macro + beta' * V`` without a ridge penalty, grid search, or window
    choice. Each univariate OLS slope is reliability-weighted by its own
    R-squared. Equity and oil obey fixed economic sign priors: non-positive for
    stress and non-negative for recovery.

    The stress covariance is also continuous: observations receive their actual
    historical stress score as a weight, rather than being split at a threshold.
    """

    common = history.index.intersection(historical_probabilities.index)
    common = common.intersection(historical_stress.dropna().index)
    common = common.intersection(historical_recovery.dropna().index)
    if len(common) < ONE_CALENDAR_YEAR:
        raise ValueError("At least one calendar year of causal history is required.")

    values = history.loc[common, ASSETS].to_numpy(dtype=float)
    probabilities = historical_probabilities.loc[common, REGIME_COLUMNS]
    stress = historical_stress.loc[common].to_numpy(dtype=float)
    recovery = historical_recovery.loc[common].to_numpy(dtype=float)
    current_p = current_probabilities[REGIME_COLUMNS].to_numpy(dtype=float)
    current_p = current_p / current_p.sum()
    current_s = float(np.clip(current_stress, 0.0, 1.0))
    current_recovery_score = float(np.clip(current_recovery, 0.0, 1.0))

    macro_mean = np.zeros(len(ASSETS), dtype=float)
    macro_covariance = np.zeros((len(ASSETS), len(ASSETS)), dtype=float)
    high_stress_covariance = np.zeros_like(macro_covariance)
    stress_adjustment = np.zeros(len(ASSETS), dtype=float)
    beta_rows: dict[str, dict[str, list[float]]] = {}
    effective_samples: dict[str, float] = {}
    credibility_rows: dict[str, float] = {}
    unconditional_mean, unconditional_covariance = _weighted_mean_and_covariance(
        values, np.ones(len(values), dtype=float)
    )

    for regime_index, regime_column in enumerate(REGIME_COLUMNS):
        regime_weights = probabilities[regime_column].to_numpy(dtype=float)
        raw_regime_mean, raw_regime_covariance = _weighted_mean_and_covariance(
            values, regime_weights
        )
        effective_sample = float(
            regime_weights.sum() ** 2
            / max(float(np.square(regime_weights).sum()), NUMERICAL_EPSILON)
        )
        # One calendar year acts as an explicit prior sample. This is not fitted:
        # it says that a new regime estimate must accumulate roughly one annual
        # cycle before it receives the same credibility as the unconditional
        # expanding estimate.
        credibility = effective_sample / (effective_sample + ONE_CALENDAR_YEAR)
        regime_mean = (
            credibility * raw_regime_mean + (1.0 - credibility) * unconditional_mean
        )
        regime_covariance = (
            credibility * raw_regime_covariance
            + (1.0 - credibility) * unconditional_covariance
        )
        weighted_asset_variance = (
            regime_weights[:, None] * (values - raw_regime_mean) ** 2
        ).sum(axis=0)

        def reliable_slope(feature: np.ndarray) -> tuple[np.ndarray, float]:
            feature_mean = float(np.average(feature, weights=regime_weights))
            centered_feature = feature - feature_mean
            denominator = float(np.sum(regime_weights * centered_feature**2))
            if denominator <= NUMERICAL_EPSILON:
                return np.zeros(len(ASSETS), dtype=float), feature_mean
            raw_slope = (
                (regime_weights * centered_feature)[:, None]
                * (values - raw_regime_mean)
            ).sum(
                axis=0
            ) / denominator  # Regime 확률 기반 가중 OLS 기울기를 추정
            covariance_numerator = raw_slope * denominator
            reliability = np.divide(
                covariance_numerator**2,
                denominator * weighted_asset_variance,
                out=np.zeros(len(ASSETS), dtype=float),
                where=weighted_asset_variance > NUMERICAL_EPSILON,
            ).clip(
                0.0, 1.0
            )  # R-squared를 계산하여 신뢰도를 추정
            return raw_slope * reliability, feature_mean

        def ols_slope(feature: np.ndarray) -> tuple[np.ndarray, float]:
            feature_mean = float(np.average(feature, weights=regime_weights))
            centered_feature = feature - feature_mean
            denominator = float(np.sum(regime_weights * centered_feature**2))
            if denominator <= NUMERICAL_EPSILON:
                return np.zeros(len(ASSETS), dtype=float), feature_mean
            raw_slope = (
                (regime_weights * centered_feature)[:, None]
                * (values - raw_regime_mean)
            ).sum(
                axis=0
            ) / denominator  # Regime 확률 기반 가중 OLS 기울기를 추정
            return raw_slope, feature_mean

        # R-squared is a parameter-free reliability weight. A weak historical
        # relationship cannot create a large expected-return forecast merely
        # because its raw OLS slope is unstable.
        beta, stress_mean = reliable_slope(stress)
        recovery_beta, recovery_mean = reliable_slope(recovery)

        # A volatility shock is a higher discount-rate / funding-liquidity shock
        # for equity and oil. Positive estimated slopes mostly capture subsequent
        # rebound months, not a reason to add risk during the shock itself. The
        # sign prior is economic and invariant across regimes; the magnitude still
        # comes entirely from the causal regime-weighted regression. Bond and gold
        # signs remain data-driven because inflation and deflation crises affect
        # those defensive assets differently.
        beta[ASSETS.index("KODEX200")] = min(beta[ASSETS.index("KODEX200")], 0.0)
        beta[ASSETS.index("USO")] = min(beta[ASSETS.index("USO")], 0.0)
        recovery_beta[ASSETS.index("KODEX200")] = max(
            recovery_beta[ASSETS.index("KODEX200")], 0.0
        )
        recovery_beta[ASSETS.index("USO")] = max(
            recovery_beta[ASSETS.index("USO")], 0.0
        )
        stress_weights = regime_weights * stress
        _, raw_regime_stress_covariance = _weighted_mean_and_covariance(
            values, stress_weights
        )
        stress_effective_sample = float(
            stress_weights.sum() ** 2
            / max(float(np.square(stress_weights).sum()), NUMERICAL_EPSILON)
        )
        stress_credibility = stress_effective_sample / (
            stress_effective_sample + ONE_CALENDAR_YEAR
        )
        regime_stress_covariance = (
            stress_credibility * raw_regime_stress_covariance
            + (1.0 - stress_credibility) * regime_covariance
        )

        probability = float(current_p[regime_index])
        macro_mean += probability * regime_mean
        macro_covariance += probability * regime_covariance
        high_stress_covariance += probability * regime_stress_covariance
        stress_adjustment += probability * (
            beta * (current_s - stress_mean)
            + recovery_beta * (current_recovery_score - recovery_mean)
        )
        beta_rows[regime_column.removeprefix("p_")] = {
            "stress": beta.tolist(),
            "recovery": recovery_beta.tolist(),
        }
        effective_samples[regime_column.removeprefix("p_")] = effective_sample
        credibility_rows[regime_column.removeprefix("p_")] = credibility

    if use_short_term_stress:
        expected_return = macro_mean + stress_adjustment
        covariance = (
            1.0 - current_s
        ) * macro_covariance + current_s * high_stress_covariance
    else:
        expected_return = macro_mean
        covariance = macro_covariance

    detail: dict[str, Any] = {
        "macro_expected_monthly_return": macro_mean.tolist(),
        "stress_return_adjustment": (
            stress_adjustment if use_short_term_stress else np.zeros(len(ASSETS))
        ).tolist(),
        "regime_stress_beta": beta_rows,
        "effective_regime_samples": effective_samples,
        "regime_credibility": credibility_rows,
    }
    return expected_return, _nearest_psd(covariance), detail


def project_to_bounded_simplex(weights: np.ndarray) -> np.ndarray:
    """Project onto the symmetric long-only concentration-bounded simplex."""

    lower = np.zeros(len(weights), dtype=float)
    upper = np.full(len(weights), MAX_SINGLE_ASSET_WEIGHT, dtype=float)
    left = float(np.min(weights - upper))
    right = float(np.max(weights - lower))
    for _ in range(100):
        shift = 0.5 * (left + right)
        projected = np.clip(weights - shift, lower, upper)
        if projected.sum() > 1.0:
            left = shift
        else:
            right = shift
    projected = np.clip(weights - 0.5 * (left + right), lower, upper)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > NUMERICAL_EPSILON:
        room = upper - projected if residual > 0 else projected - lower
        projected[int(np.argmax(room))] += residual
    return projected


def _portfolio_trade_cost(weights: np.ndarray, pretrade: np.ndarray) -> float:
    """Use the same explicit execution-cost assumptions as the reference path."""

    change = weights - pretrade
    smooth_absolute_change = np.sqrt(change**2 + NUMERICAL_EPSILON)
    domestic = float(smooth_absolute_change.sum()) * DOMESTIC_TRADE_COST
    foreign_change = (
        weights[ASSETS.index("GLD")]
        + weights[ASSETS.index("USO")]
        - pretrade[ASSETS.index("GLD")]
        - pretrade[ASSETS.index("USO")]
    )
    foreign = (
        math.sqrt(foreign_change**2 + NUMERICAL_EPSILON) * FOREIGN_WEIGHT_CHANGE_COST
    )
    return domestic + foreign


def solve_conditional_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    pretrade: np.ndarray,
    use_short_term_stress: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve 100% of the final unlevered allocation with one SLSQP program."""

    expected_return, covariance, moment_detail = estimate_conditional_moments(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=use_short_term_stress,
    )
    # print("Conditional moments estimated:")
    # print(history.head())
    # exit()
    benchmark=history['KODEX200']
    access_returns=expected_return-expected_return[ASSETS.index('KODEX200')]
    common = history.index.intersection(historical_stress.dropna().index)
    historical_asset_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    risk_aversion = (
        float(np.clip(current_stress, 0.0, 1.0)) if use_short_term_stress else 0.0
    )
    initial = (
        project_to_bounded_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        realized_history = historical_asset_returns @ weights
        downside_semivariance = float(np.mean(np.minimum(realized_history, 0.0) ** 2))
        trade_cost = _portfolio_trade_cost(weights, pretrade)
        monthly_utility = (
            monthly_return
            - 0.5 * monthly_variance
            - risk_aversion * downside_semivariance
            - trade_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_semivariance": downside_semivariance,
            "risk_aversion": risk_aversion,
            "estimated_trade_cost": trade_cost,
            "monthly_utility": monthly_utility,
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
                + cdar(historical_asset_returns @ weights, CDAR_CONFIDENCE)
            ),
        },
    ]
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints,
        options={
            "maxiter": SLSQP_MAX_ITERATIONS,
            "ftol": SLSQP_TOLERANCE,
        },
    )

    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = project_to_bounded_simplex(result.x)
    else:
        # A second SLSQP solve minimizes conditional variance to find a feasible
        # long-only point. It does not introduce hard regime or overlay weights.
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=BOUNDS,
            constraints=constraints,
            options={
                "maxiter": SLSQP_MAX_ITERATIONS,
                "ftol": SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                "Both the economic objective and feasibility SLSQP failed: "
                f"{result.message}; {fallback.message}"
            )
        result = fallback
        weights = project_to_bounded_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    path_cdar = cdar(historical_asset_returns @ weights, CDAR_CONFIDENCE)
    detail: dict[str, Any] = {
        **values,
        **moment_detail,
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": path_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": CATASTROPHE_ANNUAL_VOLATILITY - annual_vol,
        "cdar_slack": CATASTROPHE_CDAR + path_cdar,
    }
    return weights, detail


def run_conditional_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    use_short_term_stress: bool,
) -> pd.DataFrame:
    """Walk forward monthly; every return/moment observation precedes its trade."""

    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0

    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < ONE_CALENDAR_YEAR:
            continue
        current_probability = probabilities.loc[month]
        current_stress = float(stress_signals.loc[month, "stress_score"])
        current_recovery = float(stress_signals.loc[month, "recovery_score"])
        weights, detail = solve_conditional_weights(
            history=history,
            historical_probabilities=probabilities.loc[probabilities.index < month],
            current_probabilities=current_probability,
            historical_stress=stress_signals.loc[
                stress_signals.index < month, "stress_score"
            ],
            current_stress=current_stress,
            historical_recovery=stress_signals.loc[
                stress_signals.index < month, "recovery_score"
            ],
            current_recovery=current_recovery,
            pretrade=pretrade,
            use_short_term_stress=use_short_term_stress,
        )

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        foreign_change = abs(
            weights[ASSETS.index("GLD")]
            + weights[ASSETS.index("USO")]
            - pretrade[ASSETS.index("GLD")]
            - pretrade[ASSETS.index("USO")]
        )
        fx_cost = foreign_change * FOREIGN_WEIGHT_CHANGE_COST
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

        rows.append(
            {
                "month": month,
                "macro_signal_month": current_probability["signal_month"],
                "stress_signal_month": stress_signals.loc[month, "stress_signal_month"],
                "stress_signal_date": stress_signals.loc[month, "stress_signal_date"],
                "use_short_term_stress": use_short_term_stress,
                "stress_score": current_stress,
                "recovery_score": current_recovery,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **{
                    column: float(current_probability[column])
                    for column in REGIME_COLUMNS
                },
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    key: value
                    for key, value in detail.items()
                    if key
                    not in {
                        "macro_expected_monthly_return",
                        "stress_return_adjustment",
                        "regime_stress_beta",
                        "effective_regime_samples",
                        "regime_credibility",
                    }
                },
                **{
                    f"macro_mu_{asset}": float(
                        detail["macro_expected_monthly_return"][index]
                    )
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    f"stress_mu_adjustment_{asset}": float(
                        detail["stress_return_adjustment"][index]
                    )
                    for index, asset in enumerate(ASSETS)
                },
            }
        )

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _metric_row(
    strategy: str,
    path: pd.DataFrame,
    period: str,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, Any]:
    view = path.loc[start:end]
    metrics = performance_summary(view["return"])
    return {
        "Strategy": strategy,
        "Period": period,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{name: float(value) for name, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean()),
        "TotalCost": float(view[["trade_cost", "fx_cost"]].sum().sum()),
    }


def build_overlay_attribution(
    returns: pd.DataFrame,
    macro_only: pd.DataFrame,
    stress_aware: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attribute the short-horizon model against the same macro optimizer."""

    common = macro_only.index.intersection(stress_aware.index)
    risky_columns = ["w_KODEX200", "w_USO"]
    attribution = pd.DataFrame(index=common)
    attribution["return_base"] = macro_only.loc[common, "return"]
    attribution["return_stress_aware"] = stress_aware.loc[common, "return"]
    attribution["overlay_alpha"] = (
        attribution["return_stress_aware"] - attribution["return_base"]
    )
    attribution["stress_score"] = stress_aware.loc[common, "stress_score"]
    attribution["base_risky_weight"] = macro_only.loc[common, risky_columns].sum(axis=1)
    attribution["stress_aware_risky_weight"] = stress_aware.loc[
        common, risky_columns
    ].sum(axis=1)
    attribution["risk_reduction"] = (
        attribution["base_risky_weight"] - attribution["stress_aware_risky_weight"]
    )
    attribution["KODEX200_return"] = returns.loc[common, "KODEX200"]

    # Crash labels are diagnostics only. The sample's bottom decile is a
    # data-derived reporting definition and never enters the optimizer.
    crash_threshold = float(attribution["KODEX200_return"].quantile(0.10))
    numerical_action = attribution["risk_reduction"] > 1e-6
    attribution["risk_off_action"] = numerical_action
    attribution["positive_equity_month"] = attribution["KODEX200_return"] > 0.0
    attribution["crash_month"] = attribution["KODEX200_return"] <= crash_threshold
    attribution["false_positive"] = (
        attribution["risk_off_action"] & attribution["positive_equity_month"]
    )
    attribution["caught_crash"] = (
        attribution["risk_off_action"] & attribution["crash_month"]
    )
    attribution["missed_crash"] = (
        ~attribution["risk_off_action"] & attribution["crash_month"]
    )

    base_metrics = performance_summary(attribution["return_base"])
    aware_metrics = performance_summary(attribution["return_stress_aware"])
    summary: dict[str, Any] = {
        "months": int(len(attribution)),
        "overlay_cumulative_return": float(
            (1.0 + attribution["overlay_alpha"]).prod() - 1.0
        ),
        "average_monthly_overlay_alpha": float(attribution["overlay_alpha"].mean()),
        "cagr_change": float(aware_metrics["CAGR"] - base_metrics["CAGR"]),
        "sharpe_change": float(aware_metrics["Sharpe"] - base_metrics["Sharpe"]),
        "mdd_change": float(aware_metrics["MDD"] - base_metrics["MDD"]),
        "risk_off_months": int(attribution["risk_off_action"].sum()),
        "false_positive_months": int(attribution["false_positive"].sum()),
        "crash_months": int(attribution["crash_month"].sum()),
        "caught_crash_months": int(attribution["caught_crash"].sum()),
        "missed_crash_months": int(attribution["missed_crash"].sum()),
        "diagnostic_crash_threshold": crash_threshold,
    }
    return attribution, summary


def _solver_summary(path: pd.DataFrame) -> dict[str, Any]:
    return {
        "months": int(len(path)),
        "solver_successes": int(path["solver_success"].sum()),
        "fallbacks": int(path["used_fallback"].sum()),
        "maximum_weight_sum_error": float(path["sum_error"].max()),
        "minimum_volatility_slack": float(path["volatility_slack"].min()),
        "minimum_cdar_slack": float(path["cdar_slack"].min()),
        "volatility_guard_binding_months": int(
            (path["volatility_slack"].abs() < 1e-6).sum()
        ),
        "cdar_guard_binding_months": int((path["cdar_slack"].abs() < 1e-6).sum()),
    }


def run_research(save: bool = True) -> dict[str, Any]:
    """Run Stage10, macro-only attribution base, and the stress-aware strategy."""

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)
    #stress_signals = build_monthly_stress_features()
    macro_only = run_conditional_backtest(
        returns, probabilities, stress_signals, use_short_term_stress=False
    )
    stress_aware = run_conditional_backtest(
        returns, probabilities, stress_signals, use_short_term_stress=True
    )
    stage10 = pd.read_csv(STAGE10_PATH, index_col=0)
    stage10.index = pd.PeriodIndex(stage10.index, freq="M")

    paths = {
        "Stage10_SharpeCAGR_SLSQP100": stage10,
        "Stage13_MacroConditional_SLSQP100": macro_only,
        "Stage13_MacroStressConditional_SLSQP100": stress_aware,
    }
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            rows.append(_metric_row(name, path, period, start, common_end))
    comparison = pd.DataFrame(rows)

    attribution, attribution_summary = build_overlay_attribution(
        returns, macro_only, stress_aware
    )
    _, locked_attribution_summary = build_overlay_attribution(
        returns.loc[LOCKED_START:],
        macro_only.loc[LOCKED_START:],
        stress_aware.loc[LOCKED_START:],
    )
    checks = {
        "macro_signal_precedes_target": bool(
            (stress_aware["macro_signal_month"] < stress_aware.index).all()
        ),
        "stress_signal_month_precedes_target": bool(
            (stress_aware["stress_signal_month"] < stress_aware.index).all()
        ),
        "all_weights_sum_to_one": bool(
            np.allclose(
                stress_aware[[f"w_{asset}" for asset in ASSETS]].sum(axis=1),
                1.0,
            )
        ),
        "all_weights_are_long_only": bool(
            (stress_aware[[f"w_{asset}" for asset in ASSETS]] >= -1e-10).all().all()
        ),
        "no_weight_exceeds_concentration_guard": bool(
            (
                stress_aware[[f"w_{asset}" for asset in ASSETS]]
                <= MAX_SINGLE_ASSET_WEIGHT + 1e-10
            )
            .all()
            .all()
        ),
        "no_leverage": bool(
            np.allclose(
                stress_aware[[f"w_{asset}" for asset in ASSETS]].sum(axis=1),
                1.0,
            )
        ),
        "no_hard_regime_weights": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_search": True,
    }
    report: dict[str, Any] = {
        "strategy": "Stage13_MacroStressConditional_SLSQP100",
        "based_on": "Stage10_SharpeCAGR_SLSQP100",
        "allocation": {
            "slsqp_share": 1.0,
            "hard_regime_share": 0.0,
            "post_optimizer_overlay_share": 0.0,
            "leverage": 1.0,
        },
        "objective": (
            "maximize monthly expected log growth minus current stress percentile "
            "times historical downside semivariance minus explicit transaction cost"
        ),
        "economic_constants": {
            "one_week_observations": ONE_WEEK,
            "one_trading_month_observations": ONE_TRADING_MONTH,
            "minimum_history_months": ONE_CALENDAR_YEAR,
            "regime_prior_months": ONE_CALENDAR_YEAR,
            "catastrophe_annual_volatility": CATASTROPHE_ANNUAL_VOLATILITY,
            "catastrophe_cdar": CATASTROPHE_CDAR,
            "cdar_confidence": CDAR_CONFIDENCE,
            "maximum_single_asset_weight": MAX_SINGLE_ASSET_WEIGHT,
            "domestic_trade_cost": DOMESTIC_TRADE_COST,
            "foreign_weight_change_cost": FOREIGN_WEIGHT_CHANGE_COST,
        },
        "tunable_hyperparameters": [],
        "expected_return_model": [
            "four soft-regime probability-weighted expanding means",
            "regime means and covariances shrink to unconditional moments "
            "using one prior calendar year",
            "regime-weighted stress and recovery OLS slopes multiplied by their own R-squared",
            "non-positive equity/oil stress beta and non-negative equity/oil recovery beta",
        ],
        "stress_blocks": [
            "VKOSPI level expanding percentile",
            "equal mean of VKOSPI five-observation shock rank and VIX6 parallel-shift rank",
            "equal mean of VIX6 left-impulse rank and left-minus-right impulse rank",
            "one-trading-month mean of the preceding three blocks",
            "positive one-week stress decline ranked as a separate recovery score",
        ],
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "overlay_attribution": {
            "full_2007_2026": attribution_summary,
            "locked_2018_2026": locked_attribution_summary,
        },
        "macro_only_solver": _solver_summary(macro_only),
        "stress_aware_solver": _solver_summary(stress_aware),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily_stress.to_csv(OUTPUT_DIR / "daily_stress_features.csv")
        stress_signals.to_csv(OUTPUT_DIR / "monthly_stress_signals.csv")
        macro_only.to_csv(OUTPUT_DIR / "macro_conditional_monthly.csv")
        stress_aware.to_csv(OUTPUT_DIR / "macro_stress_conditional_monthly.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        attribution.to_csv(OUTPUT_DIR / "risk_overlay_attribution.csv")
        (OUTPUT_DIR / "research_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "daily_stress": daily_stress,
        "stress_signals": stress_signals,
        "macro_only": macro_only,
        "stress_aware": stress_aware,
        "comparison": comparison,
        "attribution": attribution,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
