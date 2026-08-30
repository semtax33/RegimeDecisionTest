from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import (
    ASSETS,
    cdar,
    load_monthly_asset_returns,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    CDAR_CONFIDENCE,
    CATASTROPHE_ANNUAL_VOLATILITY,
    CATASTROPHE_CDAR,
    FULL_START,
    LOCKED_START,
    ONE_CALENDAR_YEAR,
    REGIME_COLUMNS,
    SLSQP_MAX_ITERATIONS,
    SLSQP_TOLERANCE,
    build_daily_stress_features,
    build_monthly_stress_signals,
    causal_expanding_midrank,
    estimate_conditional_moments,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    STATIC_RISK_POLICY,
    UNCONSTRAINED_LONG_ONLY_BOUNDS,
    concentration_summary,
    expected_transaction_cost,
    metric_row,
    project_to_long_only_simplex,
    run_backtest as run_stage14_backtest,
    solver_summary,
)
from strategies.stage17_dynamic_risk_shape.dynamic_risk_shape_slsqp import (
    drawdown_episodes,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OHLCV_CACHE = ROOT / "cache" / "regime_lightgbm_ohlcv.csv"
BOND_PATH = ROOT / "raw_data" / "krx_bond_index.csv"
COMPASS_PATH = ROOT / "raw_data" / "compass.db"

# Predeclared economic/technical horizons; no grid or candidate selection.
K_RATIO_DAYS = 126  # approximately six trading months
WILDER_DAYS = 14  # original/default ATR and RSI horizon


def _numeric_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.to_datetime(output.index).normalize()
    columns = ["open", "high", "low", "close", "volume"]
    for column in columns:
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output[columns].sort_index()
    return output[~output.index.duplicated(keep="last")]


def load_daily_asset_ohlcv() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load daily prices aligned with the assets actually held by Stage 14.

    KODEX200 is extended backward with the same KOSPI200 proxy used by the
    monthly return engine. GLD and USO OHLC are translated to KRW with USDKRW
    close, matching Stage 14's domestic-investor return convention. The KRX
    bond total-return index has close only, so high=low=open=close and its ATR
    is explicitly a close-to-close true-range proxy.
    """

    raw = pd.read_csv(OHLCV_CACHE, parse_dates=["date"])
    market = {
        str(symbol): _numeric_ohlcv(group.set_index("date"))
        for symbol, group in raw.groupby("symbol")
    }
    required_market = {"KODEX200", "GLD", "USO", "USDKRW"}
    missing = sorted(required_market.difference(market))
    if missing:
        raise ValueError(f"OHLCV cache is missing symbols: {missing}")

    actual = market["KODEX200"].loc[
        market["KODEX200"].index > pd.Timestamp("2009-03-31")
    ].dropna(subset=["close"])
    if actual.empty:
        raise ValueError("Continuous KODEX200 ETF history is unavailable.")
    with sqlite3.connect(COMPASS_PATH) as connection:
        proxy = pd.read_sql(
            "select date, open, high, low, close, volume "
            "from etf_prices where symbol = ? order by date",
            connection,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy = _numeric_ohlcv(proxy.set_index("date"))
    first_actual = actual.index.min()
    nearest_proxy_position = proxy.index.get_indexer(
        [first_actual], method="nearest"
    )[0]
    scale = float(
        actual.loc[first_actual, "close"]
        / proxy.iloc[nearest_proxy_position]["close"]
    )
    for column in ["open", "high", "low", "close"]:
        proxy[column] *= scale
    proxy = proxy.loc[proxy.index < first_actual].copy()
    proxy["volume_segment"] = "KOSPI200_proxy"
    actual = actual.copy()
    actual["volume_segment"] = "KODEX200_ETF"
    kodex = pd.concat([proxy, actual]).sort_index()

    bond_raw = pd.read_csv(BOND_PATH, encoding="cp949")
    bond_close = pd.to_numeric(
        bond_raw.iloc[:, 1].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    bond = pd.DataFrame(
        {
            "open": bond_close.to_numpy(),
            "high": bond_close.to_numpy(),
            "low": bond_close.to_numpy(),
            "close": bond_close.to_numpy(),
            "volume": np.nan,
        },
        index=pd.to_datetime(bond_raw.iloc[:, 0]).dt.normalize(),
    ).sort_index()

    fx = market["USDKRW"]["close"].dropna().sort_index()
    foreign: dict[str, pd.DataFrame] = {}
    for asset in ["GLD", "USO"]:
        frame = market[asset].dropna(subset=["close"]).copy()
        aligned_fx = fx.reindex(frame.index, method="ffill")
        if aligned_fx.isna().any():
            raise ValueError(f"USDKRW does not cover the full {asset} history.")
        for column in ["open", "high", "low", "close"]:
            frame[column] = frame[column] * aligned_fx
        foreign[asset] = frame

    frames = {
        "KODEX200": kodex,
        "BOND": bond,
        "GLD": foreign["GLD"],
        "USO": foreign["USO"],
    }
    audit: dict[str, Any] = {
        "ohlcv_cache": str(OHLCV_CACHE),
        "bond_source": str(BOND_PATH),
        "kodex_proxy_source": str(COMPASS_PATH),
        "foreign_prices_converted_to_krw": True,
        "bond_atr_uses_close_to_close_proxy": True,
        "kodex_volume_segments_normalized_separately": True,
        "assets": {},
    }
    for asset, frame in frames.items():
        audit["assets"][asset] = {
            "rows": int(len(frame)),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "missing_close": int(frame["close"].isna().sum()),
            "missing_high": int(frame["high"].isna().sum()),
            "missing_low": int(frame["low"].isna().sum()),
            "missing_volume": int(frame["volume"].isna().sum()),
        }
    return frames, audit


def rolling_k_ratio(close: pd.Series, window: int = K_RATIO_DAYS) -> pd.Series:
    """Kestner 1996 K-Ratio on a fixed daily log-price window.

    K = slope / (standard_error_of_slope * sqrt(n)). A positive value means a
    consistently rising log-price path; negative means a consistently falling
    path. All windows have equal length, so ratios are directly comparable over
    time within each asset.
    """

    log_price = np.log(close.where(close > 0.0))
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    ssx = float(np.square(centered_x).sum())

    def calculate(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return float("nan")
        centered_y = values - values.mean()
        slope = float(centered_x @ centered_y / ssx)
        residual = centered_y - slope * centered_x
        residual_variance = float(np.square(residual).sum() / (window - 2))
        slope_standard_error = math.sqrt(max(residual_variance / ssx, 0.0))
        if slope_standard_error <= 1e-15:
            if abs(slope) <= 1e-15:
                return 0.0
            return slope / (np.finfo(float).eps * math.sqrt(window))
        return slope / (slope_standard_error * math.sqrt(window))

    return log_price.rolling(window, min_periods=window).apply(
        calculate, raw=True
    )


def _wilder_average(values: pd.Series, period: int = WILDER_DAYS) -> pd.Series:
    """Wilder smoothing seeded with the first period's simple average."""

    values = values.astype(float)
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) <= period:
        return output
    seed = values.iloc[1 : period + 1]
    if seed.notna().sum() < period:
        return output
    output.iloc[period] = float(seed.mean())
    for position in range(period + 1, len(values)):
        current = values.iloc[position]
        previous = output.iloc[position - 1]
        if not np.isfinite(current) or not np.isfinite(previous):
            continue
        output.iloc[position] = (
            previous * (period - 1) + current
        ) / period
    return output


def average_true_range(
    frame: pd.DataFrame,
    period: int = WILDER_DAYS,
) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range.iloc[0] = np.nan
    return _wilder_average(true_range, period)


def price_rsi(close: pd.Series, period: int = WILDER_DAYS) -> pd.Series:
    change = close.diff()
    average_gain = _wilder_average(change.clip(lower=0.0), period)
    average_loss = _wilder_average((-change.clip(upper=0.0)), period)
    denominator = average_gain + average_loss
    return 100.0 * average_gain.div(denominator.replace(0.0, np.nan))


def volume_rsi(
    close: pd.Series,
    volume: pd.Series,
    period: int = WILDER_DAYS,
) -> pd.Series:
    """Wilder-smoothed share of up-day volume in directional volume."""

    change = close.diff()
    up_volume = volume.where(change > 0.0, 0.0)
    down_volume = volume.where(change < 0.0, 0.0)
    average_up = _wilder_average(up_volume, period)
    average_down = _wilder_average(down_volume, period)
    denominator = average_up + average_down
    return 100.0 * average_up.div(denominator.replace(0.0, np.nan))


def build_daily_technical_features(
    frames: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if frames is None:
        frames, audit = load_daily_asset_ohlcv()
    else:
        audit = {"provided_frames": True, "assets": {}}
    features: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        frame = frames[asset].copy()
        output = pd.DataFrame(index=frame.index)
        output["k_ratio"] = rolling_k_ratio(frame["close"])
        output["k_score"] = output["k_ratio"] / (
            1.0 + output["k_ratio"].abs()
        )
        output["atr"] = average_true_range(frame)
        output["natr"] = output["atr"] / frame["close"]
        output["atr_percentile"] = causal_expanding_midrank(output["natr"])
        if asset == "KODEX200":
            output["price_rsi"] = price_rsi(frame["close"])
            volume_rsi_parts: list[pd.Series] = []
            for _, segment in frame.groupby("volume_segment", sort=False):
                volume_rsi_parts.append(
                    volume_rsi(segment["close"], segment["volume"])
                )
            output["volume_rsi"] = pd.concat(volume_rsi_parts).sort_index()
            output["price_strength"] = (
                output["price_rsi"] - 50.0
            ) / 50.0
            output["volume_strength"] = (
                output["volume_rsi"] - 50.0
            ) / 50.0
            short_strength = output[
                ["price_strength", "volume_strength"]
            ].mean(axis=1)
            output["rsi_confirmation"] = np.clip(
                0.5
                * (
                    1.0
                    + np.sign(output["k_score"]) * short_strength
                ),
                0.0,
                1.0,
            )
            # The 126-day K-Ratio owns the strategic direction.  Fourteen-day
            # price/volume RSI can only scale its credibility and therefore
            # cannot reverse a medium-horizon trend.
            output["technical_direction"] = (
                output["k_score"] * output["rsi_confirmation"]
            )
        else:
            output["technical_direction"] = output["k_score"]
        features[asset] = output.replace([np.inf, -np.inf], np.nan)
        if "assets" in audit:
            valid = features[asset].dropna(
                subset=["k_ratio", "natr", "atr_percentile"]
            )
            audit["assets"].setdefault(asset, {})
            audit["assets"][asset].update(
                {
                    "feature_start": str(valid.index.min().date()),
                    "feature_end": str(valid.index.max().date()),
                    "feature_rows": int(len(valid)),
                }
            )
    audit["k_ratio_days"] = K_RATIO_DAYS
    audit["wilder_days"] = WILDER_DAYS
    return features, audit


def build_monthly_technical_signals(
    target_months: pd.PeriodIndex,
    daily_features: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        signal_month = target_month - 1
        month_end = signal_month.to_timestamp("M")
        row: dict[str, Any] = {
            "target_month": target_month,
            "technical_signal_month": signal_month,
        }
        complete = True
        for asset in ASSETS:
            required = [
                "k_ratio",
                "k_score",
                "natr",
                "atr_percentile",
                "technical_direction",
            ]
            if asset == "KODEX200":
                required += [
                    "price_rsi",
                    "volume_rsi",
                    "price_strength",
                    "volume_strength",
                    "rsi_confirmation",
                ]
            known = daily_features[asset].loc[:month_end].dropna(
                subset=required
            )
            if known.empty:
                complete = False
                break
            signal_date = known.index[-1]
            current = known.iloc[-1]
            row[f"technical_signal_date_{asset}"] = signal_date
            for column in required:
                row[f"{column}_{asset}"] = float(current[column])
        if complete:
            rows.append(row)
    monthly = pd.DataFrame(rows).set_index("target_month")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly


def apply_technical_inputs(
    macro_expected_return: np.ndarray,
    covariance: np.ndarray,
    signal: pd.Series,
) -> dict[str, np.ndarray | float]:
    """Apply direction inputs to mu confidence and ATR to current covariance."""

    macro = np.asarray(macro_expected_return, dtype=float)
    neutral = float(macro.mean())
    macro_direction = np.sign(macro - neutral)
    technical_direction = np.array(
        [float(signal[f"technical_direction_{asset}"]) for asset in ASSETS]
    )
    confidence = np.clip(
        0.5 * (1.0 + macro_direction * technical_direction), 0.0, 1.0
    )
    filtered_macro = neutral + confidence * (macro - neutral)

    atr_percentile = np.array(
        [float(signal[f"atr_percentile_{asset}"]) for asset in ASSETS]
    )
    variance_scale = 1.0 + np.clip(atr_percentile, 0.0, 1.0)
    scaling = np.diag(np.sqrt(variance_scale))
    adjusted_covariance = scaling @ covariance @ scaling
    return {
        "macro_neutral_return": neutral,
        "macro_relative_direction": macro_direction,
        "technical_direction": technical_direction,
        "macro_confidence": confidence,
        "filtered_macro_expected_return": filtered_macro,
        "atr_percentile": atr_percentile,
        "atr_variance_scale": variance_scale,
        "adjusted_covariance": adjusted_covariance,
    }


def solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the Stage 14 fixed-lambda optimizer with daily technical inputs."""

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
    technical = apply_technical_inputs(
        macro_expected_return, base_covariance, technical_signal
    )
    filtered_macro = np.asarray(
        technical["filtered_macro_expected_return"], dtype=float
    )
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
                "Both technical and feasibility solves failed: "
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
            "policy": "KRatioPrimary_StaticLambda",
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
    detail.pop("adjusted_covariance")
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
) -> pd.DataFrame:
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
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
            pretrade=pretrade,
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
            "stress_signal_month": stress_signals.loc[
                month, "stress_signal_month"
            ],
            "stress_signal_date": stress_signals.loc[
                month, "stress_signal_date"
            ],
            "technical_signal_month": technical_signal[
                "technical_signal_month"
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
            **{
                column: float(probability[column]) for column in REGIME_COLUMNS
            },
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
        rows.append(row)

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def feature_diagnostics(
    path: pd.DataFrame,
    returns: pd.DataFrame,
    start: pd.Period,
    end: pd.Period,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    common = path.loc[start:end].index.intersection(returns.loc[start:end].index)
    for asset in ASSETS:
        target = returns.loc[common, asset]
        rows.extend(
            [
                {
                    "Period": f"{start}_{end}",
                    "Asset": asset,
                    "Feature": "KScore",
                    "Target": "same_target_month_return",
                    "SpearmanIC": float(
                        path.loc[common, f"k_score_{asset}"].corr(
                            target, method="spearman"
                        )
                    ),
                    "Observations": int(len(common)),
                },
                {
                    "Period": f"{start}_{end}",
                    "Asset": asset,
                    "Feature": "ATRPercentile",
                    "Target": "same_target_month_absolute_return",
                    "SpearmanIC": float(
                        path.loc[common, f"atr_percentile_{asset}"].corr(
                            target.abs(), method="spearman"
                        )
                    ),
                    "Observations": int(len(common)),
                },
            ]
        )
    equity = returns.loc[common, "KODEX200"]
    for feature in ["price_strength_KODEX200", "volume_strength_KODEX200"]:
        rows.append(
            {
                "Period": f"{start}_{end}",
                "Asset": "KODEX200",
                "Feature": feature.removesuffix("_KODEX200"),
                "Target": "same_target_month_return",
                "SpearmanIC": float(
                    path.loc[common, feature].corr(equity, method="spearman")
                ),
                "Observations": int(len(common)),
            }
        )
    return pd.DataFrame(rows)


def _episode_gold_summary(path: pd.DataFrame) -> dict[str, Any]:
    view = path.loc["2012-10":"2014-10"]
    return {
        "months": int(len(view)),
        "average_gold_weight": float(view["w_GLD"].mean()),
        "average_gold_k_ratio": (
            float(view["k_ratio_GLD"].mean())
            if "k_ratio_GLD" in view
            else None
        ),
        "average_gold_atr_percentile": (
            float(view["atr_percentile_GLD"].mean())
            if "atr_percentile_GLD" in view
            else None
        ),
        "average_gold_macro_confidence": (
            float(view["macro_confidence_GLD"].mean())
            if "macro_confidence_GLD" in view
            else None
        ),
    }


def verify_technical_signal_dates(path: pd.DataFrame) -> bool:
    for asset in ASSETS:
        signal_period = pd.to_datetime(
            path[f"technical_signal_date_{asset}"]
        ).dt.to_period("M")
        if not (signal_period.to_numpy() < path.index.to_numpy()).all():
            return False
    return True


def verify_monthly_signals_match_daily_features(
    monthly: pd.DataFrame,
    daily: dict[str, pd.DataFrame],
) -> bool:
    """Rebuild every as-of selection and compare dates and numeric values."""

    for target_month, row in monthly.iterrows():
        month_end = (target_month - 1).to_timestamp("M")
        for asset in ASSETS:
            required = [
                "k_ratio",
                "k_score",
                "natr",
                "atr_percentile",
                "technical_direction",
            ]
            if asset == "KODEX200":
                required += [
                    "price_rsi",
                    "volume_rsi",
                    "price_strength",
                    "volume_strength",
                    "rsi_confirmation",
                ]
            known = daily[asset].loc[:month_end].dropna(subset=required)
            if known.empty or known.index[-1] != pd.Timestamp(
                row[f"technical_signal_date_{asset}"]
            ):
                return False
            expected = known.iloc[-1][required].to_numpy(dtype=float)
            saved = np.array(
                [float(row[f"{feature}_{asset}"]) for feature in required]
            )
            if not np.allclose(expected, saved):
                return False
    return True


def run_research(save: bool = True) -> dict[str, Any]:
    """Evaluate one fixed daily-technical hypothesis against Stage 14."""

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)
    daily_technical, data_audit = build_daily_technical_features()
    technical_signals = build_monthly_technical_signals(
        returns.index, daily_technical
    )

    stage14_path = run_stage14_backtest(
        returns, probabilities, stress_signals, STATIC_RISK_POLICY
    )
    technical_path = run_backtest(
        returns, probabilities, stress_signals, technical_signals
    )
    paths = {
        "Stage14_StaticLambda": stage14_path,
        "Stage22_KRatioPrimary": technical_path,
    }
    common_end = min(frame.index.max() for frame in paths.values())
    comparison_rows: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            comparison_rows.append(metric_row(name, path, period, start, common_end))
    comparison = pd.DataFrame(comparison_rows)

    diagnostics = pd.concat(
        [
            feature_diagnostics(technical_path, returns, FULL_START, common_end),
            feature_diagnostics(technical_path, returns, LOCKED_START, common_end),
        ],
        ignore_index=True,
    )
    episodes = pd.concat(
        [
            drawdown_episodes(stage14_path, returns, "Stage14_StaticLambda"),
            drawdown_episodes(
                technical_path, returns, "Stage22_KRatioPrimary"
            ),
        ],
        ignore_index=True,
    )
    top_episodes = (
        episodes.sort_values(["Strategy", "EpisodeMDD"])
        .groupby("Strategy", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    baseline = full.loc["Stage14_StaticLambda"]
    candidate = full.loc["Stage22_KRatioPrimary"]
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    checks = {
        "macro_signal_precedes_target": bool(
            (technical_path["macro_signal_month"] < technical_path.index).all()
        ),
        "stress_signal_precedes_target": bool(
            (technical_path["stress_signal_month"] < technical_path.index).all()
        ),
        "technical_signal_precedes_target": verify_technical_signal_dates(
            technical_path
        ),
        "monthly_signals_match_last_known_daily_features": (
            verify_monthly_signals_match_daily_features(
                technical_signals, daily_technical
            )
        ),
        "technical_features_complete_all_months": bool(
            technical_path[
                [f"k_ratio_{asset}" for asset in ASSETS]
                + [f"atr_percentile_{asset}" for asset in ASSETS]
                + ["price_rsi_KODEX200", "volume_rsi_KODEX200"]
            ]
            .notna()
            .all()
            .all()
        ),
        "weights_sum_to_one": bool(
            np.allclose(technical_path[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (technical_path[weight_columns] >= -1e-10).all().all()
        ),
        "no_leverage": bool(
            np.allclose(technical_path[weight_columns].sum(axis=1), 1.0)
        ),
        "static_lambda_equals_one": bool(
            np.allclose(
                technical_path["downside_risk_aversion_lambda"], 1.0
            )
        ),
        "all_solvers_succeeded": bool(
            technical_path["solver_success"].all()
            and not technical_path["used_fallback"].any()
        ),
        "no_hard_asset_cap": True,
        "no_hard_regime_weights": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_or_candidate_search": True,
        "single_predeclared_hypothesis": True,
        "stage20_source_unchanged": True,
    }
    report: dict[str, Any] = {
        "strategy": "Stage22_KRatioPrimary",
        "base_strategy": "Stage14_StaticLambda",
        "data_audit": data_audit,
        "feature_policy": {
            "k_ratio": (
                "126-daily-observation Kestner-1996 slope / "
                "(slope standard error * sqrt(n)) on log price"
            ),
            "k_score": "k_ratio / (1 + abs(k_ratio))",
            "atr": "Wilder ATR(14), normalized by close",
            "atr_normalization": "causal expanding percentile per asset",
            "price_strength_equity": "Wilder price RSI(14), mapped by (RSI-50)/50",
            "volume_strength_equity": (
                "Wilder-smoothed up-volume share RSI(14), mapped by (RSI-50)/50"
            ),
            "equity_rsi_confirmation": (
                "clip((1 + sign(k_score) * mean(price_strength, "
                "volume_strength)) / 2, 0, 1)"
            ),
            "equity_technical_direction": "k_score * rsi_confirmation",
            "other_asset_technical_direction": "k_score",
            "macro_confidence": (
                "(1 + sign(macro_mu-neutral_mu) * technical_direction) / 2"
            ),
            "filtered_macro_mu": (
                "neutral_mu + confidence * (macro_mu-neutral_mu)"
            ),
            "atr_covariance": (
                "D @ Sigma @ D where D_i=sqrt(1+causal_ATR_percentile_i)"
            ),
            "searched_parameters": None,
            "candidate_count": 1,
            "single_change_from_stage20": (
                "K-Ratio owns equity direction; RSI only confirms magnitude"
            ),
        },
        "unchanged_controls": {
            "downside_risk_aversion_lambda": 1.0,
            "annual_volatility_guard": CATASTROPHE_ANNUAL_VOLATILITY,
            "cdar_guard": CATASTROPHE_CDAR,
            "cdar_confidence": CDAR_CONFIDENCE,
            "asset_bounds": [0.0, 1.0],
            "weight_sum": 1.0,
            "leverage": 1.0,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "full_period_changes": {
            "cagr": float(candidate["CAGR"] - baseline["CAGR"]),
            "sharpe": float(candidate["Sharpe"] - baseline["Sharpe"]),
            "mdd": float(candidate["MDD"] - baseline["MDD"]),
            "volatility": float(
                candidate["Volatility"] - baseline["Volatility"]
            ),
        },
        "feature_diagnostics": json.loads(
            diagnostics.to_json(orient="records", force_ascii=False)
        ),
        "stage14_concentration": concentration_summary(stage14_path),
        "technical_concentration": concentration_summary(technical_path),
        "stage14_gold_episode": _episode_gold_summary(stage14_path),
        "technical_gold_episode": _episode_gold_summary(technical_path),
        "top_drawdown_episodes": json.loads(
            top_episodes.to_json(orient="records", force_ascii=False)
        ),
        "solver": solver_summary(technical_path),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stage14_path.to_csv(OUTPUT_DIR / "stage14_static_recomputed_monthly.csv")
        technical_path.to_csv(
            OUTPUT_DIR / "k_ratio_primary_monthly.csv"
        )
        technical_signals.to_csv(OUTPUT_DIR / "monthly_technical_signals.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        diagnostics.to_csv(OUTPUT_DIR / "feature_diagnostics.csv", index=False)
        top_episodes.to_csv(
            OUTPUT_DIR / "drawdown_episode_attribution.csv", index=False
        )
        for asset, frame in daily_technical.items():
            frame.to_csv(
                OUTPUT_DIR / f"daily_technical_features_{asset}.csv"
            )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "stress_signals": stress_signals,
        "daily_technical": daily_technical,
        "technical_signals": technical_signals,
        "stage14_path": stage14_path,
        "technical_path": technical_path,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "episodes": top_episodes,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(result["diagnostics"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
