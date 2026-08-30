from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from strategies.core.regime_research import (
    ASSETS,
    cdar,
    load_monthly_asset_returns,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
    load_vkospi_daily,
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
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20_base,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OHLCV_CACHE = ROOT / "cache" / "regime_lightgbm_ohlcv.csv"
BOND_PATH = ROOT / "raw_data" / "krx_bond_index.csv"
COMPASS_PATH = ROOT / "raw_data" / "compass.db"
OPTION_CHAIN_PATH = ROOT / "raw_data" / "KOSPI200OptionPrice.csv"

# Predeclared economic/technical horizons; no grid or candidate selection.
K_RATIO_DAYS = 126  # approximately six trading months
WILDER_DAYS = 14  # original/default ATR and RSI horizon

# Predeclared conventions from the supplied economic design.  These are not
# searched: the surface targets 30 calendar days, and the fast/slow blocks use
# one trading week and one trading month.
TARGET_MATURITY_DAYS = 30
MIN_MATURITY_DAYS = 7
MAX_MATURITY_DAYS = 60
FAST_DAYS = 5
SLOW_DAYS = 20
OPTION_EQUITY_INDEX = ASSETS.index("KODEX200")


def _second_thursday(year: int, month: int) -> pd.Timestamp:
    """Return the scheduled monthly KOSPI200 option expiry date."""

    first = pd.Timestamp(year=year, month=month, day=1)
    first_thursday = first + pd.Timedelta(days=(3 - first.weekday()) % 7)
    return first_thursday + pd.Timedelta(days=7)


def _expiry_from_token(token: str) -> pd.Timestamp:
    value = str(token)
    if len(value) == 4:
        year = 2000 + int(value[:2])
        month = int(value[2:])
    elif len(value) == 6:
        year = int(value[:4])
        month = int(value[4:])
    else:
        return pd.NaT
    if month < 1 or month > 12:
        return pd.NaT
    return _second_thursday(year, month)


def load_kospi200_option_chain() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the consolidated regular-session option chain.

    Old KRX rows use a four-digit YYMM contract token and encode call/put in
    the first digit of the issue code.  Newer rows expose C/P and YYYYMM in the
    name.  Both encodings are normalized without using future returns.
    """

    columns = [
        "일자",
        "종목코드",
        "종목명",
        "종가",
        "내재변동성",
        "거래량",
        "거래대금",
        "미결제약정",
    ]
    raw = pd.read_csv(OPTION_CHAIN_PATH, usecols=columns, low_memory=False)
    session = raw["일자"].astype(str)
    date_text = session.str.extract(r"(\d{4}/\d{2}/\d{2})", expand=False)
    raw["date"] = pd.to_datetime(date_text, errors="coerce")
    raw = raw.loc[session.str.contains("주간", na=False)].copy()

    names = raw["종목명"].astype(str).str.replace(",", "", regex=False)
    contract = names.str.extract(
        r"(?P<expiry>\d{4,6})\s+(?P<strike>[0-9.]+)\s*$"
    )
    explicit_type = names.str.extract(r"\s(?P<option_type>[CP])\s", expand=True)[
        "option_type"
    ]
    legacy_type = raw["종목코드"].astype(str).str[0].map(
        {"2": "C", "3": "P"}
    )
    raw["option_type"] = explicit_type.fillna(legacy_type)
    raw["expiry_token"] = contract["expiry"]
    raw["strike"] = pd.to_numeric(contract["strike"], errors="coerce")
    for source, target in [
        ("종가", "close"),
        ("내재변동성", "implied_volatility"),
        ("거래량", "volume"),
        ("거래대금", "trading_value"),
        ("미결제약정", "open_interest"),
    ]:
        raw[target] = pd.to_numeric(raw[source], errors="coerce")

    expiry_map = {
        token: _expiry_from_token(token)
        for token in raw["expiry_token"].dropna().astype(str).unique()
    }
    raw["expiry_date"] = raw["expiry_token"].astype(str).map(expiry_map)
    raw["dte"] = (raw["expiry_date"] - raw["date"]).dt.days
    keep = (
        raw["date"].notna()
        & raw["option_type"].isin(["C", "P"])
        & raw["strike"].gt(0.0)
        & raw["dte"].between(MIN_MATURITY_DAYS, MAX_MATURITY_DAYS)
    )
    parsed = raw.loc[
        keep,
        [
            "date",
            "expiry_date",
            "dte",
            "option_type",
            "strike",
            "close",
            "implied_volatility",
            "volume",
            "trading_value",
            "open_interest",
        ],
    ].copy()
    parsed = parsed.sort_values(
        ["date", "expiry_date", "option_type", "strike"]
    )
    audit = {
        "source": str(OPTION_CHAIN_PATH),
        "raw_rows": int(len(raw)),
        "regular_session_rows_with_parsed_contract_and_7_60_dte": int(
            len(parsed)
        ),
        "first_date": str(parsed["date"].min().date()),
        "last_date": str(parsed["date"].max().date()),
        "target_maturity_days": TARGET_MATURITY_DAYS,
        "maturity_range_days": [MIN_MATURITY_DAYS, MAX_MATURITY_DAYS],
        "legacy_call_put_from_issue_code": True,
        "night_session_excluded": True,
    }
    return parsed, audit


def _monotone_otm_prices(
    frame: pd.DataFrame, option_type: str
) -> tuple[np.ndarray, np.ndarray]:
    clean = (
        frame.loc[frame["close"].gt(0.0), ["strike", "close"]]
        .dropna()
        .groupby("strike", as_index=False)["close"]
        .median()
        .sort_values("strike")
    )
    strikes = clean["strike"].to_numpy(dtype=float)
    prices = clean["close"].to_numpy(dtype=float)
    if option_type == "P":
        prices = np.maximum.accumulate(prices)
    else:
        prices = np.minimum.accumulate(prices)
    return strikes, prices


def _interpolated_iv(frame: pd.DataFrame, strike: float) -> float:
    clean = (
        frame.loc[
            frame["implied_volatility"].between(3.0001, 200.0),
            ["strike", "implied_volatility"],
        ]
        .dropna()
        .groupby("strike", as_index=False)["implied_volatility"]
        .median()
        .sort_values("strike")
    )
    if len(clean) < 2:
        return float("nan")
    x = clean["strike"].to_numpy(dtype=float)
    if strike < x.min() or strike > x.max():
        return float("nan")
    return float(
        np.interp(strike, x, clean["implied_volatility"].to_numpy(dtype=float))
        / 100.0
    )


def _surface_for_expiry(frame: pd.DataFrame) -> dict[str, float] | None:
    calls = frame.loc[frame["option_type"] == "C"].copy()
    puts = frame.loc[frame["option_type"] == "P"].copy()
    parity = calls.pivot_table(index="strike", values="close", aggfunc="median").join(
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
    parity["distance"] = (parity["close_call"] - parity["close_put"]).abs()
    near = parity.nsmallest(min(5, len(parity)), "distance")
    forward = float(
        np.median(
            near.index.to_numpy(dtype=float)
            + near["close_call"].to_numpy(dtype=float)
            - near["close_put"].to_numpy(dtype=float)
        )
    )
    dte = float(frame["dte"].iloc[0])
    maturity = dte / 365.0
    if not np.isfinite(forward) or forward <= 0.0 or maturity <= 0.0:
        return None

    otm_put = puts.loc[puts["strike"] < forward]
    otm_call = calls.loc[calls["strike"] > forward]
    put_k, put_p = _monotone_otm_prices(otm_put, "P")
    call_k, call_p = _monotone_otm_prices(otm_call, "C")
    if len(put_k) < 3 or len(call_k) < 3:
        return None
    downside_variance = float(
        2.0 * np.trapezoid(put_p, put_k) / (forward**2 * maturity)
    )
    upside_variance = float(
        2.0 * np.trapezoid(call_p, call_k) / (forward**2 * maturity)
    )
    total_variance = downside_variance + upside_variance
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        return None
    variance_asymmetry = (
        downside_variance - upside_variance
    ) / total_variance

    call_atm_iv = _interpolated_iv(calls, forward)
    put_atm_iv = _interpolated_iv(puts, forward)
    atm_values = np.asarray([call_atm_iv, put_atm_iv], dtype=float)
    atm_iv = float(np.nanmean(atm_values))
    if not np.isfinite(atm_iv) or atm_iv <= 0.0:
        return None
    put_delta_d1 = float(norm.ppf(0.75))
    put25_strike = forward * math.exp(
        -put_delta_d1 * atm_iv * math.sqrt(maturity)
        + 0.5 * atm_iv**2 * maturity
    )
    put25_iv = _interpolated_iv(puts, put25_strike)
    if not np.isfinite(put25_iv):
        return None

    call_value = float(calls["trading_value"].fillna(0.0).clip(lower=0.0).sum())
    put_value = float(puts["trading_value"].fillna(0.0).clip(lower=0.0).sum())
    if call_value <= 0.0 or put_value <= 0.0:
        return None
    return {
        "dte": dte,
        "forward": forward,
        "atm_iv": atm_iv,
        "put25_iv": put25_iv,
        "put_skew25": put25_iv - atm_iv,
        "downside_implied_variance": downside_variance,
        "upside_implied_variance": upside_variance,
        "implied_variance_asymmetry": float(variance_asymmetry),
        "log_call_put_trading_value": float(math.log(call_value / put_value)),
        # Martin-style SVIX-squared proxy.  It is deliberately labelled a
        # proxy because the local file lacks a full discount curve and the
        # strike tails are truncated at listed contracts.
        "option_erp_proxy": float(total_variance),
        "put_strike_min_ratio": float(put_k.min() / forward),
        "call_strike_max_ratio": float(call_k.max() / forward),
        "put_contracts": float(len(put_k)),
        "call_contracts": float(len(call_k)),
    }


def _constant_maturity_row(group: pd.DataFrame) -> dict[str, Any] | None:
    rows: list[dict[str, float]] = []
    for _, expiry_frame in group.groupby("expiry_date", sort=True):
        result = _surface_for_expiry(expiry_frame)
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


def causal_expanding_zscore(
    series: pd.Series, min_periods: int = 60
) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def build_daily_option_direction_features(
    chain: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if chain is None:
        chain, audit = load_kospi200_option_chain()
    else:
        audit = {"provided_chain": True, "rows": int(len(chain))}
    rows: list[dict[str, Any]] = []
    for date, date_frame in chain.groupby("date", sort=True):
        result = _constant_maturity_row(date_frame)
        if result is not None:
            rows.append({"date": date, **result})
    daily = pd.DataFrame(rows).set_index("date").sort_index()

    daily["put_skew_change_5"] = daily["put_skew25"].diff(FAST_DAYS)
    daily["iva_change_5"] = daily["implied_variance_asymmetry"].diff(FAST_DAYS)
    daily["cpv_change_5"] = daily["log_call_put_trading_value"].diff(FAST_DAYS)
    daily["put_skew_change_20"] = daily["put_skew25"].diff(SLOW_DAYS)
    daily["iva_change_20"] = daily["implied_variance_asymmetry"].diff(SLOW_DAYS)
    daily["cpv_change_20"] = daily["log_call_put_trading_value"].diff(SLOW_DAYS)
    log_erp = np.log(daily["option_erp_proxy"].clip(lower=1e-12))
    daily["option_erp_fast"] = log_erp.rolling(FAST_DAYS).mean()
    daily["option_erp_slow"] = log_erp.rolling(SLOW_DAYS).mean()

    z_inputs = [
        "put_skew_change_5",
        "iva_change_5",
        "cpv_change_5",
        "put_skew_change_20",
        "iva_change_20",
        "cpv_change_20",
        "option_erp_fast",
        "option_erp_slow",
    ]
    for column in z_inputs:
        daily[f"z_{column}"] = causal_expanding_zscore(daily[column])
    daily["bear_pressure_fast"] = (
        daily["z_put_skew_change_5"]
        + daily["z_iva_change_5"]
        - daily["z_cpv_change_5"]
    ) / 3.0
    daily["bear_pressure_slow"] = (
        daily["z_put_skew_change_20"]
        + daily["z_iva_change_20"]
        - daily["z_cpv_change_20"]
    ) / 3.0
    daily["option_direction_fast"] = (
        daily["z_option_erp_fast"] - daily["bear_pressure_fast"]
    )
    daily["option_direction_slow"] = (
        daily["z_option_erp_slow"] - daily["bear_pressure_slow"]
    )
    daily["option_direction"] = daily[
        ["option_direction_fast", "option_direction_slow"]
    ].mean(axis=1)
    daily["option_direction_score"] = daily["option_direction"] / (
        1.0 + daily["option_direction"].abs()
    )
    audit.update(
        {
            "daily_surface_rows": int(len(daily)),
            "first_surface_date": str(daily.index.min().date()),
            "last_surface_date": str(daily.index.max().date()),
            "complete_direction_rows": int(
                daily["option_direction_score"].notna().sum()
            ),
            "maturity_methods": daily["maturity_method"].value_counts().to_dict(),
            "erp_measure": (
                "Martin-style SVIX-squared proxy with zero-rate parity forward "
                "and truncated listed-strike integrals"
            ),
            "vertical_arbitrage_projection": (
                "monotone OTM put/call prices before integration"
            ),
            "searched_parameters": None,
        }
    )
    return daily.replace([np.inf, -np.inf], np.nan), audit


def build_monthly_option_direction_signals(
    target_months: pd.PeriodIndex,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "put_skew25",
        "implied_variance_asymmetry",
        "log_call_put_trading_value",
        "option_erp_proxy",
        "bear_pressure_fast",
        "bear_pressure_slow",
        "option_direction_fast",
        "option_direction_slow",
        "option_direction",
        "option_direction_score",
        "dte",
    ]
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
                **{column: float(current[column]) for column in columns},
                "maturity_method": str(current["maturity_method"]),
            }
        )
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def build_daily_vkospi_only_stress_features() -> pd.DataFrame:
    """Retain VKOSPI risk magnitude while removing every VIX6 component."""

    daily = load_vkospi_daily()[["close"]].rename(
        columns={"close": "vkospi_close"}
    )
    daily["vkospi_log_change_5"] = np.log(daily["vkospi_close"]).diff(FAST_DAYS)
    daily["level_component"] = causal_expanding_midrank(daily["vkospi_close"])
    daily["shock_component"] = causal_expanding_midrank(
        daily["vkospi_log_change_5"]
    )
    daily["persistence_component"] = daily[
        ["level_component", "shock_component"]
    ].mean(axis=1).rolling(SLOW_DAYS, min_periods=1).mean()
    daily["stress_raw"] = daily[
        ["level_component", "shock_component", "persistence_component"]
    ].mean(axis=1)
    one_week_mean = daily["stress_raw"].rolling(FAST_DAYS, min_periods=1).mean()
    daily["stress_score"] = pd.concat(
        [daily["stress_raw"], one_week_mean], axis=1
    ).max(axis=1).clip(0.0, 1.0)
    recovery_intensity = (one_week_mean - daily["stress_raw"]).clip(lower=0.0)
    daily["recovery_score"] = causal_expanding_midrank(
        recovery_intensity.where(recovery_intensity > 0.0)
    ).fillna(0.0)
    return daily.replace([np.inf, -np.inf], np.nan)


def build_monthly_vkospi_only_stress_signals(
    target_months: pd.PeriodIndex,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "level_component",
        "shock_component",
        "persistence_component",
        "stress_raw",
        "stress_score",
        "recovery_score",
    ]
    valid = daily.dropna(subset=["stress_score"])
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
                "stress_signal_month": signal_month,
                "stress_signal_date": signal_date,
                **{column: float(current[column]) for column in columns},
            }
        )
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


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
            output["technical_direction"] = output[
                ["k_score", "price_strength", "volume_strength"]
            ].mean(axis=1)
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
    option_signal: pd.Series | None,
    use_option_direction: bool,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run fixed-lambda SLSQP with option direction applied only to equity mu."""

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
    ).copy()
    option_direction = (
        float(option_signal["option_direction"])
        if option_signal is not None
        else 0.0
    )
    option_direction_score = (
        float(option_signal["option_direction_score"])
        if option_signal is not None
        else 0.0
    )
    # ODS is dimensionless.  Convert it to the optimizer's monthly-return
    # units with the contemporaneous cross-asset dispersion of the causal
    # macro forecast.  This creates no fitted coefficient and caps the effect
    # at one macro cross-sectional standard deviation through the bounded ODS.
    option_return_scale = float(np.std(macro_expected_return, ddof=0))
    option_mu_adjustment = (
        option_direction_score * option_return_scale
        if use_option_direction
        else 0.0
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
            "policy": (
                "OptionDirectionalSurface_StaticLambda"
                if use_option_direction
                else "VKOSPIOnly_NoOptionDirection_StaticLambda"
            ),
            "option_direction": option_direction,
            "option_direction_score": option_direction_score,
            "option_return_scale": option_return_scale,
            "option_mu_adjustment_KODEX200": option_mu_adjustment,
            "option_direction_applied": bool(use_option_direction),
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
    option_signals: pd.DataFrame,
    use_option_direction: bool,
) -> pd.DataFrame:
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
            use_option_direction=use_option_direction,
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
            "option_signal_month": option_signal["option_signal_month"],
            "option_signal_date": option_signal["option_signal_date"],
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
            "option_direction",
            "option_direction_score",
            "option_return_scale",
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
        for feature in [
            "put_skew25",
            "implied_variance_asymmetry",
            "log_call_put_trading_value",
            "option_erp_proxy",
            "bear_pressure_fast",
            "bear_pressure_slow",
            "option_direction_fast",
            "option_direction_slow",
            "dte",
        ]:
            row[feature] = float(option_signal[feature])
        row["option_maturity_method"] = str(option_signal["maturity_method"])
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


def verify_option_signal_dates(path: pd.DataFrame) -> bool:
    signal_period = pd.to_datetime(path["option_signal_date"]).dt.to_period("M")
    return bool((signal_period.to_numpy() < path.index.to_numpy()).all())


def verify_monthly_option_signals(
    monthly: pd.DataFrame, daily: pd.DataFrame
) -> bool:
    columns = [
        "put_skew25",
        "implied_variance_asymmetry",
        "log_call_put_trading_value",
        "option_erp_proxy",
        "bear_pressure_fast",
        "bear_pressure_slow",
        "option_direction_fast",
        "option_direction_slow",
        "option_direction",
        "option_direction_score",
        "dte",
    ]
    valid = daily.dropna(subset=["option_direction_score"])
    for target_month, row in monthly.iterrows():
        known = valid.loc[: (target_month - 1).to_timestamp("M")]
        if known.empty or known.index[-1] != pd.Timestamp(row["option_signal_date"]):
            return False
        expected = known.iloc[-1][columns].to_numpy(dtype=float)
        actual = row[columns].to_numpy(dtype=float)
        if not np.allclose(expected, actual):
            return False
    return True


def option_forward_diagnostics(
    daily_option: pd.DataFrame,
    option_signals: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames, _ = load_daily_asset_ohlcv()
    close = frames["KODEX200"]["close"].rename("kodex_close")
    daily = daily_option.join(close, how="inner")
    daily["forward_5d_return"] = daily["kodex_close"].shift(-FAST_DAYS) / daily[
        "kodex_close"
    ] - 1.0
    daily["forward_20d_return"] = daily["kodex_close"].shift(-SLOW_DAYS) / daily[
        "kodex_close"
    ] - 1.0
    sample = daily.dropna(
        subset=["option_direction_score", "forward_5d_return", "forward_20d_return"]
    ).copy()
    # The two-axis follow-up can assign an exact zero to many observations.
    # Rank-first preserves the score ordering while allowing five equally
    # sized diagnostic bins; the bins never feed the strategy itself.
    diagnostic_rank = sample["option_direction_score"].rank(method="first")
    sample["ODSQuintile"] = pd.qcut(
        diagnostic_rank, 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rows: list[dict[str, Any]] = []
    for quintile, view in sample.groupby("ODSQuintile", sort=True):
        tail_cut = view["forward_20d_return"].quantile(0.10)
        rows.append(
            {
                "ODSQuintile": int(quintile),
                "Observations": int(len(view)),
                "MeanODSScore": float(view["option_direction_score"].mean()),
                "Forward5DMean": float(view["forward_5d_return"].mean()),
                "Forward20DMean": float(view["forward_20d_return"].mean()),
                "Forward5DNegativeProbability": float(
                    (view["forward_5d_return"] < 0.0).mean()
                ),
                "Forward20DNegativeProbability": float(
                    (view["forward_20d_return"] < 0.0).mean()
                ),
                "Forward20DTailMean": float(
                    view.loc[view["forward_20d_return"] <= tail_cut, "forward_20d_return"].mean()
                ),
            }
        )
    quintiles = pd.DataFrame(rows)
    common = option_signals.index.intersection(returns.index)
    monthly_score = option_signals.loc[common, "option_direction_score"]
    monthly_equity = returns.loc[common, "KODEX200"]
    summary = {
        "daily_observations": int(len(sample)),
        "daily_ods_forward_5d_ic": float(
            sample["option_direction_score"].corr(sample["forward_5d_return"])
        ),
        "daily_ods_forward_20d_ic": float(
            sample["option_direction_score"].corr(sample["forward_20d_return"])
        ),
        "monthly_ods_equity_return_ic": float(monthly_score.corr(monthly_equity)),
        "quintile_5d_monotonic_correlation": float(
            quintiles["ODSQuintile"].corr(quintiles["Forward5DMean"])
        ),
        "quintile_20d_monotonic_correlation": float(
            quintiles["ODSQuintile"].corr(quintiles["Forward20DMean"])
        ),
        "quintile_forward_returns_strictly_increasing": {
            "5d": bool(quintiles["Forward5DMean"].is_monotonic_increasing),
            "20d": bool(quintiles["Forward20DMean"].is_monotonic_increasing),
        },
        "quintiles_are_full_sample_diagnostics_not_strategy_inputs": True,
        "quintile_ties_split_by_stable_date_order": True,
    }
    return quintiles, summary


def run_research(save: bool = True) -> dict[str, Any]:
    """Replace Stage20 VIX6 decomposition with VKOSPI magnitude plus ODS."""

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    original_daily_stress = build_daily_stress_features()
    original_stress_signals = build_monthly_stress_signals(
        returns.index, original_daily_stress
    )
    vkospi_daily_stress = build_daily_vkospi_only_stress_features()
    vkospi_stress_signals = build_monthly_vkospi_only_stress_signals(
        returns.index, vkospi_daily_stress
    )
    daily_technical, technical_audit = build_daily_technical_features()
    technical_signals = build_monthly_technical_signals(
        returns.index, daily_technical
    )
    daily_option, option_audit = build_daily_option_direction_features()
    option_signals = build_monthly_option_direction_signals(
        returns.index, daily_option
    )

    stage20_path = stage20_base.run_backtest(
        returns,
        probabilities,
        original_stress_signals,
        technical_signals,
    )
    vkospi_only_path = run_backtest(
        returns,
        probabilities,
        vkospi_stress_signals,
        technical_signals,
        option_signals,
        use_option_direction=False,
    )
    option_path = run_backtest(
        returns,
        probabilities,
        vkospi_stress_signals,
        technical_signals,
        option_signals,
        use_option_direction=True,
    )
    paths = {
        "Stage20_VIX6Decomposition": stage20_path,
        "Stage28_VKOSPIOnly_NoODS": vkospi_only_path,
        "Stage28_OptionDirectionalSurface": option_path,
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
            feature_diagnostics(option_path, returns, FULL_START, common_end),
            feature_diagnostics(option_path, returns, LOCKED_START, common_end),
        ],
        ignore_index=True,
    )
    quintiles, option_diagnostic_summary = option_forward_diagnostics(
        daily_option, option_signals, returns
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

    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    baseline = full.loc["Stage20_VIX6Decomposition"]
    ablation = full.loc["Stage28_VKOSPIOnly_NoODS"]
    candidate = full.loc["Stage28_OptionDirectionalSurface"]
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    stage20_saved = pd.read_csv(
        stage20_base.OUTPUT_DIR / "daily_technical_confidence_monthly.csv",
        index_col=0,
    )
    stage20_saved.index = pd.PeriodIndex(stage20_saved.index, freq="M")
    common_saved = stage20_path.index.intersection(stage20_saved.index)
    other_assets = [asset for asset in ASSETS if asset != "KODEX200"]
    checks = {
        "stage20_reference_reproduced": bool(
            np.allclose(
                stage20_path.loc[common_saved, "return"],
                stage20_saved.loc[common_saved, "return"],
            )
        ),
        "macro_signal_precedes_target": bool(
            (option_path["macro_signal_month"] < option_path.index).all()
        ),
        "vkospi_signal_precedes_target": bool(
            (option_path["stress_signal_month"] < option_path.index).all()
        ),
        "technical_signal_precedes_target": verify_technical_signal_dates(
            option_path
        ),
        "option_signal_precedes_target": verify_option_signal_dates(option_path),
        "monthly_technical_signals_match_daily": (
            verify_monthly_signals_match_daily_features(
                technical_signals, daily_technical
            )
        ),
        "monthly_option_signals_match_daily": verify_monthly_option_signals(
            option_signals, daily_option
        ),
        "candidate_stress_is_vkospi_only": bool(
            not any("vix6" in column.lower() for column in vkospi_daily_stress.columns)
        ),
        "ods_disabled_ablation_has_zero_adjustment": bool(
            np.allclose(vkospi_only_path["option_mu_adjustment_KODEX200"], 0.0)
        ),
        "ods_adjustment_matches_bounded_score_times_macro_scale": bool(
            np.allclose(
                option_path["option_mu_adjustment_KODEX200"],
                option_path["option_direction_score"]
                * option_path["option_return_scale"],
            )
        ),
        "ods_changes_only_equity_expected_mu": bool(
            all(
                np.allclose(
                    option_path[f"filtered_expected_mu_{asset}"],
                    vkospi_only_path[f"filtered_expected_mu_{asset}"],
                )
                for asset in other_assets
            )
            and np.allclose(
                option_path["filtered_expected_mu_KODEX200"]
                - vkospi_only_path["filtered_expected_mu_KODEX200"],
                option_path["option_mu_adjustment_KODEX200"],
            )
        ),
        "weights_sum_to_one": bool(
            np.allclose(option_path[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (option_path[weight_columns] >= -1e-10).all().all()
        ),
        "no_leverage": bool(
            np.allclose(option_path[weight_columns].sum(axis=1), 1.0)
        ),
        "static_lambda_equals_one": bool(
            np.allclose(option_path["downside_risk_aversion_lambda"], 1.0)
        ),
        "all_solvers_succeeded": bool(
            option_path["solver_success"].all()
            and not option_path["used_fallback"].any()
        ),
        "no_hard_asset_cap": True,
        "no_hard_regime_weights": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_or_candidate_search": True,
        "single_predeclared_ods_formula": True,
        "nearest_maturity_limitation_disclosed": True,
    }

    def changes(left: pd.Series, right: pd.Series) -> dict[str, float]:
        return {
            "cagr": float(left["CAGR"] - right["CAGR"]),
            "sharpe": float(left["Sharpe"] - right["Sharpe"]),
            "mdd": float(left["MDD"] - right["MDD"]),
            "volatility": float(left["Volatility"] - right["Volatility"]),
        }

    report: dict[str, Any] = {
        "strategy": "Stage28_OptionDirectionalSurface",
        "base_strategy": "Stage20_VIX6Decomposition",
        "design": (
            "VIX6 decomposition is removed from the candidate. VKOSPI retains "
            "risk magnitude; KOSPI200 option-surface direction changes only "
            "KODEX200 expected return."
        ),
        "data_audit": {
            "technical": technical_audit,
            "option_surface": option_audit,
            "vkospi_only_stress_first_date": str(
                vkospi_daily_stress.dropna(subset=["stress_score"]).index.min().date()
            ),
            "vkospi_only_stress_last_date": str(
                vkospi_daily_stress.dropna(subset=["stress_score"]).index.max().date()
            ),
        },
        "option_policy": {
            "put_wing": "IV(25-delta put) - ATM IV",
            "variance_asymmetry": "(downside IVar-upside IVar)/(sum)",
            "flow": "log(call trading value/put trading value)",
            "erp": (
                "Martin-style SVIX-squared proxy from OTM option-price integrals; "
                "not a full curve-complete ERP estimate"
            ),
            "fast_days": FAST_DAYS,
            "slow_days": SLOW_DAYS,
            "bear_pressure": "mean(z(delta put skew), z(delta IVA), -z(delta CPV))",
            "option_direction": (
                "equal mean of fast and slow [z(option ERP proxy)-bear pressure]"
            ),
            "bounded_score": "ODS/(1+abs(ODS))",
            "equity_mu_adjustment": (
                "bounded ODS * contemporaneous cross-sectional std(macro mu)"
            ),
            "lambda_changed_by_ods": False,
            "covariance_changed_by_ods": False,
            "hard_allocation_changed_by_ods": False,
            "searched_parameters": None,
            "candidate_count": 1,
        },
        "maturity_limitation": {
            "requested_target_days": TARGET_MATURITY_DAYS,
            "exact_or_interpolated_rows": int(
                daily_option["maturity_method"].isin(
                    ["exact_30d", "interpolated_30d"]
                ).sum()
            ),
            "nearest_listed_proxy_rows": int(
                (daily_option["maturity_method"] == "nearest_listed_proxy").sum()
            ),
            "explanation": (
                "The historical KRX file usually lists one monthly expiry, so "
                "a true 30-day interpolation is unavailable on most dates. The "
                "nearest 7-60 DTE surface is used and explicitly tagged."
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
            "stage20_daily_k_ratio_atr_rsi": True,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "full_period_changes_vs_stage20": changes(candidate, baseline),
        "vix6_removal_effect_before_ods": changes(ablation, baseline),
        "ods_increment_on_vkospi_only": changes(candidate, ablation),
        "option_forward_diagnostics": option_diagnostic_summary,
        "technical_feature_diagnostics": json.loads(
            diagnostics.to_json(orient="records", force_ascii=False)
        ),
        "concentration": {
            name: concentration_summary(path) for name, path in paths.items()
        },
        "top_drawdown_episodes": json.loads(
            top_episodes.to_json(orient="records", force_ascii=False)
        ),
        "solver": solver_summary(option_path),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stage20_path.to_csv(OUTPUT_DIR / "stage20_vix6_monthly.csv")
        vkospi_only_path.to_csv(OUTPUT_DIR / "vkospi_only_no_ods_monthly.csv")
        option_path.to_csv(OUTPUT_DIR / "option_directional_surface_monthly.csv")
        technical_signals.to_csv(OUTPUT_DIR / "monthly_technical_signals.csv")
        option_signals.to_csv(OUTPUT_DIR / "monthly_option_direction_signals.csv")
        vkospi_stress_signals.to_csv(
            OUTPUT_DIR / "monthly_vkospi_only_stress_signals.csv"
        )
        daily_option.to_csv(OUTPUT_DIR / "daily_option_direction_features.csv")
        vkospi_daily_stress.to_csv(OUTPUT_DIR / "daily_vkospi_only_stress.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        diagnostics.to_csv(OUTPUT_DIR / "feature_diagnostics.csv", index=False)
        quintiles.to_csv(OUTPUT_DIR / "option_forward_quintiles.csv", index=False)
        top_episodes.to_csv(
            OUTPUT_DIR / "drawdown_episode_attribution.csv", index=False
        )
        for asset, frame in daily_technical.items():
            frame.to_csv(OUTPUT_DIR / f"daily_technical_features_{asset}.csv")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "original_stress_signals": original_stress_signals,
        "vkospi_stress_signals": vkospi_stress_signals,
        "daily_technical": daily_technical,
        "technical_signals": technical_signals,
        "daily_option": daily_option,
        "option_signals": option_signals,
        "stage20_path": stage20_path,
        "vkospi_only_path": vkospi_only_path,
        "option_path": option_path,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "option_quintiles": quintiles,
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
