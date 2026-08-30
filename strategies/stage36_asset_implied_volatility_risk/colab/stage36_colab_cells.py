# %% [01_environment]
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import unicodedata
import zipfile
from bisect import bisect_left, bisect_right, insort
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import spearmanr


ASSETS = ["KODEX200", "BOND", "GLD", "USO"]
REGIME_COLUMNS = [
    "p_Goldilocks",
    "p_Overheating",
    "p_Slowdown",
    "p_Stagflation",
]
FULL_START = pd.Period("2007-04", freq="M")
COMMON_START = pd.Period("2010-01", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")
RESEARCH_END = pd.Period("2026-07", freq="M")
ONE_WEEK = 5
ONE_TRADING_MONTH = 21
ONE_CALENDAR_YEAR = 12
MIN_CAUSAL_MONTHS = 60
MIN_SENSOR_HISTORY = 252
K_RATIO_DAYS = 126
WILDER_DAYS = 14
CREDIT_CHANGE_DAYS = 20
CATASTROPHE_ANNUAL_VOLATILITY = 0.13
CATASTROPHE_CDAR = 0.16
CDAR_CONFIDENCE = 0.90
DOMESTIC_TRADE_COST = 0.0015
FOREIGN_WEIGHT_CHANGE_COST = 0.0005
SLSQP_MAX_ITERATIONS = 300
SLSQP_TOLERANCE = 1e-9
NUMERICAL_EPSILON = 1e-12
UNCONSTRAINED_LONG_ONLY_BOUNDS = [(0.0, 1.0)] * len(ASSETS)
EQUITY_INDEX = ASSETS.index("KODEX200")
GOLD_INDEX = ASSETS.index("GLD")
OIL_INDEX = ASSETS.index("USO")
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]

DATA_ROOT: Path | None = None
RAW_DIR: Path | None = None
CACHE_DIR: Path | None = None
RESULTS_DIR: Path | None = None
OUTPUT_DIR: Path | None = None


def configure_data_root(data_root: str | Path) -> Path:
    """Point every loader at the extracted data-only bundle."""

    global DATA_ROOT, RAW_DIR, CACHE_DIR, RESULTS_DIR, OUTPUT_DIR
    DATA_ROOT = Path(data_root).resolve()
    RAW_DIR = DATA_ROOT / "raw_data"
    CACHE_DIR = DATA_ROOT / "cache"
    RESULTS_DIR = DATA_ROOT / "results"
    OUTPUT_DIR = DATA_ROOT / "colab_outputs"
    for path in (RAW_DIR, CACHE_DIR, RESULTS_DIR):
        if not path.is_dir():
            raise FileNotFoundError(f"Required data directory is missing: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_path(directory: Path, filename: str) -> Path:
    """Resolve Korean filenames regardless of NFC/NFD normalization."""

    target = unicodedata.normalize("NFC", filename)
    for path in directory.iterdir():
        if unicodedata.normalize("NFC", path.name) == target:
            return path
    raise FileNotFoundError(filename)


def validate_data_bundle(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
        elif path.stat().st_size != record["bytes"]:
            failures.append(f"size: {record['path']}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"sha256: {record['path']}")
    if failures:
        raise ValueError("Bundle validation failed: " + ", ".join(failures))
    return {
        "bundle": manifest["bundle"],
        "files": len(manifest["files"]),
        "bytes": int(sum(row["bytes"] for row in manifest["files"])),
        "all_hashes_match": True,
        "code_included": bool(manifest["code_included"]),
    }


# %% [02_upload_and_extract]
def locate_or_upload_bundle() -> Path:
    """Use one uploaded ZIP in Colab; use an environment path in local QA."""

    local_override = os.environ.get("STAGE36_DATA_ZIP")
    if local_override:
        path = Path(local_override).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    try:
        from google.colab import files  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "Local execution requires STAGE36_DATA_ZIP to point to "
            "stage36_colab_data.zip"
        ) from error
    print("stage36_colab_data.zip 파일 하나를 업로드하세요.")
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(zip_names) != 1:
        raise ValueError("ZIP 파일을 정확히 하나만 업로드해야 합니다.")
    path = Path(zip_names[0]).resolve()
    path.write_bytes(uploaded[zip_names[0]])
    return path


def extract_bundle(bundle_zip: str | Path) -> Path:
    """Extract to a clean work directory and validate all source hashes."""

    default_work = "/content/stage36_workspace"
    work_root = Path(os.environ.get("STAGE36_WORK_DIR", default_work)).resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip) as archive:
        archive.extractall(work_root)
    data_root = work_root / "stage36_data"
    configure_data_root(data_root)
    audit = validate_data_bundle(data_root)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return data_root


# %% [03_causal_transforms]
def causal_expanding_percentile(series: pd.Series) -> pd.Series:
    """Empirical midrank using observations available through the current row."""

    output = pd.Series(np.nan, index=series.index, dtype=float)
    history: list[float] = []
    for index, value in series.items():
        if not np.isfinite(value):
            continue
        history.append(float(value))
        reference = np.asarray(history, dtype=float)
        less = float(np.sum(reference < value))
        equal = float(np.sum(reference == value))
        output.loc[index] = (less + 0.5 * equal) / len(reference)
    return output


def causal_expanding_midrank(series: pd.Series) -> pd.Series:
    """O(n log n) causal empirical CDF with equal-value midranks."""

    result = pd.Series(np.nan, index=series.index, dtype=float)
    ordered: list[float] = []
    for index, raw_value in series.items():
        value = float(raw_value) if pd.notna(raw_value) else np.nan
        if not np.isfinite(value):
            continue
        left = bisect_left(ordered, value)
        right = bisect_right(ordered, value)
        equal_after_insertion = right - left + 1
        result.loc[index] = (
            left + 0.5 * equal_after_insertion
        ) / (len(ordered) + 1)
        insort(ordered, value)
    return result


def causal_zscore(series: pd.Series, minimum: int = MIN_CAUSAL_MONTHS) -> pd.Series:
    values = series.astype(float)
    prior = values.shift(1)
    mean = prior.expanding(min_periods=minimum).mean()
    std = prior.expanding(min_periods=minimum).std(ddof=1)
    return ((values - mean) / std.where(std > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )


def rank_after_prior_history(
    series: pd.Series, minimum: int = MIN_SENSOR_HISTORY
) -> tuple[pd.Series, pd.Series]:
    rank = causal_expanding_midrank(series)
    prior_count = (
        series.notna().shift(1).fillna(False).astype(int).cumsum().astype(int)
    )
    return rank.where(prior_count >= minimum), prior_count


# %% [04_market_returns]
def load_market_cache() -> pd.DataFrame:
    assert CACHE_DIR is not None
    path = CACHE_DIR / "market_daily.csv"
    return pd.read_csv(path, parse_dates=["date"])


def load_monthly_asset_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the exact monthly open-to-next-open KRW return panel."""

    assert RAW_DIR is not None
    market = load_market_cache()
    with sqlite3.connect(get_path(RAW_DIR, "compass.db")) as connection:
        proxy = pd.read_sql(
            "select date, open, close from etf_prices "
            "where symbol = ? order by date",
            connection,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy[["open", "close"]] = proxy[["open", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )

    actual = market.loc[market["symbol"].eq("KODEX200")].copy()
    actual = actual.dropna(subset=["open"])
    actual = actual.loc[actual["date"] > pd.Timestamp("2009-03-31")]
    first_actual = actual["date"].min()
    actual_anchor = float(
        actual.loc[actual["date"].eq(first_actual), "open"].iloc[0]
    )
    proxy_anchor = proxy.loc[proxy["date"].eq(first_actual), "open"]
    if proxy_anchor.empty:
        nearest = proxy.iloc[(proxy["date"] - first_actual).abs().argsort()[:1]]
        proxy_anchor_value = float(nearest["open"].iloc[0])
    else:
        proxy_anchor_value = float(proxy_anchor.iloc[0])
    for column in ("open", "close"):
        proxy[column] *= actual_anchor / proxy_anchor_value
    proxy = proxy.loc[proxy["date"] < first_actual].copy()
    proxy["symbol"] = "KODEX200"
    kodex = pd.concat(
        [proxy[["date", "symbol", "open", "close"]], actual],
        ignore_index=True,
    )

    bond = pd.read_csv(get_path(RAW_DIR, "krx_bond_index.csv"), encoding="cp949")
    bond["date"] = pd.to_datetime(bond.iloc[:, 0])
    bond["open"] = (
        bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    bond["close"] = bond["open"]
    bond["symbol"] = "BOND"

    fx = (
        market.loc[market["symbol"].eq("USDKRW")]
        .set_index("date")["close"]
        .sort_index()
    )
    fx = fx.reindex(pd.date_range(fx.index.min(), fx.index.max(), freq="D")).ffill()
    first_open: dict[str, pd.Series] = {}
    for symbol, data in {
        "KODEX200": kodex,
        "BOND": bond,
        "GLD": market.loc[market["symbol"].eq("GLD")],
        "USO": market.loc[market["symbol"].eq("USO")],
    }.items():
        temp = data.dropna(subset=["open"]).sort_values("date").copy()
        temp["month"] = temp["date"].dt.to_period("M")
        first = temp.groupby("month", sort=True).first()
        value = first["open"].astype(float)
        if symbol in {"GLD", "USO"}:
            aligned = fx.reindex(pd.DatetimeIndex(first["date"]), method="ffill")
            value = value * aligned.to_numpy()
        first_open[symbol] = value
    levels = pd.concat(first_open, axis=1).sort_index()
    returns = levels.shift(-1).div(levels).sub(1.0).dropna(how="any")
    return returns[ASSETS], levels[ASSETS]


# %% [05_macro_probabilities]
def load_macro_levels() -> pd.DataFrame:
    """Load six published macro levels with the original release lags."""

    assert RAW_DIR is not None
    gdp = pd.read_excel(
        get_path(RAW_DIR, "GDP 성장률.xlsx"), index_col=0, skiprows=6
    )
    gdp.columns = ["GDP_QoQ", "GDP_YoY"]
    gdp.index = (
        pd.PeriodIndex(gdp.index, freq="Q")
        .asfreq("M", how="end")
        .to_timestamp("M")
        + pd.offsets.MonthEnd(1)
    )
    gdp = gdp.resample("ME").ffill()

    trade = pd.read_excel(
        get_path(RAW_DIR, "수출입 총괄_20260816.xlsx"),
        index_col=0,
        skiprows=4,
    )
    trade = trade[["수출 금액", "수입금액"]].iloc[1:].copy()
    for column in trade:
        trade[column] = (
            trade[column].astype(str).str.replace(",", "", regex=False).astype(float)
        )
    trade.index = pd.to_datetime(trade.index, format="%Y.%m") + pd.offsets.MonthEnd(1)
    trade["Export_YoY"] = trade["수출 금액"].pct_change(12) * 100.0

    bsi = pd.read_csv(
        get_path(RAW_DIR, "기업경기조사(전망).csv"), encoding="cp949"
    )
    bsi = bsi.loc[
        bsi["업종코드별"].eq("제 조 업")
        & bsi["BSI코드별"].eq("업황전망BSI 1)")
    ].iloc[:, 2:4]
    bsi["시점"] = (
        bsi["시점"]
        .str.replace("월", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    bsi["시점"] = pd.to_datetime(bsi["시점"], format="%Y.%m") + pd.offsets.MonthEnd(1)
    bsi = bsi.set_index("시점")
    bsi.columns = ["BSI"]

    cpi = pd.read_excel(
        get_path(RAW_DIR, "소비자물가 상승률.xlsx"), index_col=0, skiprows=6
    )
    cpi.columns = ["CPI_QoQ", "CPI_YoY"]
    cpi.index = pd.to_datetime(cpi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    ppi = pd.read_excel(
        get_path(RAW_DIR, "생산자물가 상승률.xlsx"), index_col=0, skiprows=6
    )
    ppi.columns = ["PPI_QoQ", "PPI_YoY"]
    ppi.index = pd.to_datetime(ppi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    prices = pd.read_excel(
        get_path(RAW_DIR, "수출입물가 상승률.xlsx"), index_col=0, skiprows=6
    )
    prices.columns = ["ExportPrice_YoY", "ImportPrice_YoY"]
    prices.index = pd.to_datetime(prices.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    return pd.concat(
        [
            gdp["GDP_YoY"],
            trade["Export_YoY"],
            bsi["BSI"],
            cpi["CPI_YoY"],
            ppi["PPI_YoY"],
            prices["ImportPrice_YoY"],
        ],
        axis=1,
    ).sort_index()


def build_macro_probabilities(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = load_macro_levels()
    ranks = levels.apply(causal_expanding_percentile)
    rows: list[dict[str, Any]] = []
    for target_month in returns.index:
        signal_month = target_month - 1
        known = ranks.loc[: signal_month.to_timestamp("M")]
        if known.empty:
            continue
        current = known.iloc[-1]
        if current.isna().any():
            continue
        growth = float(current[["GDP_YoY", "Export_YoY", "BSI"]].mean())
        inflation = float(
            current[["CPI_YoY", "PPI_YoY", "ImportPrice_YoY"]].mean()
        )
        rows.append(
            {
                "target_month": target_month,
                "signal_month": signal_month,
                "p_growth_high": growth,
                "p_inflation_high": inflation,
            }
        )
    probabilities = pd.DataFrame(rows).set_index("target_month")
    probabilities.index = pd.PeriodIndex(probabilities.index, freq="M")
    growth = probabilities["p_growth_high"]
    inflation = probabilities["p_inflation_high"]
    probabilities["p_Goldilocks"] = growth * (1.0 - inflation)
    probabilities["p_Overheating"] = growth * inflation
    probabilities["p_Slowdown"] = (1.0 - growth) * (1.0 - inflation)
    probabilities["p_Stagflation"] = (1.0 - growth) * inflation
    return probabilities, ranks


# %% [06_vkospi_vix6_stress]
def load_vkospi_daily() -> pd.DataFrame:
    assert RAW_DIR is not None
    raw = pd.read_csv(RAW_DIR / "VKOSPIData.csv", encoding="utf-8-sig")
    daily = raw.iloc[:, :7].copy()
    daily.columns = ["date", "close", "change", "return_pct", "open", "high", "low"]
    daily["date"] = pd.to_datetime(daily["date"], format="%Y/%m/%d", errors="coerce")
    daily["close"] = pd.to_numeric(
        daily["close"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    daily = daily.dropna(subset=["date", "close"]).set_index("date").sort_index()
    return daily.loc[~daily.index.duplicated(keep="last")]


def read_vix6_components() -> pd.DataFrame:
    assert RESULTS_DIR is not None
    path = RESULTS_DIR / "vix6_case1_features_daily.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
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
    vkospi = load_vkospi_daily()[["close"]].rename(columns={"close": "vkospi_close"})
    daily = vkospi.join(read_vix6_components(), how="inner").sort_index()
    daily = daily.loc[~daily.index.duplicated(keep="last")]
    daily["vkospi_log_change_5"] = np.log(daily["vkospi_close"]).diff(ONE_WEEK)
    daily["vix6_left_impulse"] = daily["put_skew"] + daily["downside_convexity"]
    daily["vix6_right_impulse"] = daily["call_skew"] + daily["upside_convexity"]
    daily["vix6_tail_asymmetry"] = (
        daily["vix6_left_impulse"] - daily["vix6_right_impulse"]
    )
    daily["level_component"] = causal_expanding_midrank(daily["vkospi_close"])
    daily["vkospi_shock_rank"] = causal_expanding_midrank(daily["vkospi_log_change_5"])
    daily["parallel_shift_rank"] = causal_expanding_midrank(daily["parallel_shift"])
    daily["shock_component"] = daily[
        ["vkospi_shock_rank", "parallel_shift_rank"]
    ].mean(axis=1)
    daily["left_impulse_rank"] = causal_expanding_midrank(daily["vix6_left_impulse"])
    daily["tail_asymmetry_rank"] = causal_expanding_midrank(
        daily["vix6_tail_asymmetry"]
    )
    daily["tail_component"] = daily[
        ["left_impulse_rank", "tail_asymmetry_rank"]
    ].mean(axis=1)
    three_blocks = daily[
        ["level_component", "shock_component", "tail_component"]
    ].mean(axis=1)
    daily["persistence_component"] = three_blocks.rolling(
        ONE_TRADING_MONTH, min_periods=1
    ).mean()
    daily["stress_raw"] = daily[
        ["level_component", "shock_component", "tail_component", "persistence_component"]
    ].mean(axis=1)
    one_week_mean = daily["stress_raw"].rolling(ONE_WEEK, min_periods=1).mean()
    daily["stress_score"] = pd.concat(
        [daily["stress_raw"], one_week_mean], axis=1
    ).max(axis=1).clip(0.0, 1.0)
    recovery_intensity = (one_week_mean - daily["stress_raw"]).clip(lower=0.0)
    daily["recovery_score"] = causal_expanding_midrank(
        recovery_intensity.where(recovery_intensity > 0.0)
    ).fillna(0.0)
    return daily.replace([np.inf, -np.inf], np.nan)


def build_monthly_stress_signals(
    target_months: pd.PeriodIndex, daily: pd.DataFrame
) -> pd.DataFrame:
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
    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        signal_month = target_month - 1
        known = valid.loc[: signal_month.to_timestamp("M")]
        if known.empty:
            continue
        current = known.iloc[-1]
        rows.append(
            {
                "target_month": target_month,
                "stress_signal_month": signal_month,
                "stress_signal_date": known.index[-1],
                **{column: float(current[column]) for column in columns},
            }
        )
    monthly = pd.DataFrame(rows).set_index("target_month")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly


# %% [07_conditional_moments]
def nearest_psd(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.trace(symmetric)) / len(symmetric), NUMERICAL_EPSILON)
    floor = scale * 1e-10
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def weighted_mean_and_covariance(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    return mean, nearest_psd(covariance)


def estimate_conditional_moments(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    current_r = float(np.clip(current_recovery, 0.0, 1.0))
    macro_mean = np.zeros(len(ASSETS), dtype=float)
    macro_covariance = np.zeros((len(ASSETS), len(ASSETS)), dtype=float)
    high_stress_covariance = np.zeros_like(macro_covariance)
    stress_adjustment = np.zeros(len(ASSETS), dtype=float)
    unconditional_mean, unconditional_covariance = weighted_mean_and_covariance(
        values, np.ones(len(values), dtype=float)
    )
    effective_samples: dict[str, float] = {}
    credibility_rows: dict[str, float] = {}
    for regime_index, regime_column in enumerate(REGIME_COLUMNS):
        regime_weights = probabilities[regime_column].to_numpy(dtype=float)
        raw_mean, raw_covariance = weighted_mean_and_covariance(values, regime_weights)
        effective_sample = float(
            regime_weights.sum() ** 2
            / max(float(np.square(regime_weights).sum()), NUMERICAL_EPSILON)
        )
        credibility = effective_sample / (effective_sample + ONE_CALENDAR_YEAR)
        regime_mean = credibility * raw_mean + (1.0 - credibility) * unconditional_mean
        regime_covariance = (
            credibility * raw_covariance
            + (1.0 - credibility) * unconditional_covariance
        )
        weighted_asset_variance = (
            regime_weights[:, None] * (values - raw_mean) ** 2
        ).sum(axis=0)

        def reliable_slope(feature: np.ndarray) -> tuple[np.ndarray, float]:
            feature_mean = float(np.average(feature, weights=regime_weights))
            centered = feature - feature_mean
            denominator = float(np.sum(regime_weights * centered**2))
            if denominator <= NUMERICAL_EPSILON:
                return np.zeros(len(ASSETS), dtype=float), feature_mean
            raw_slope = (
                (regime_weights * centered)[:, None] * (values - raw_mean)
            ).sum(axis=0) / denominator
            covariance_numerator = raw_slope * denominator
            reliability = np.divide(
                covariance_numerator**2,
                denominator * weighted_asset_variance,
                out=np.zeros(len(ASSETS), dtype=float),
                where=weighted_asset_variance > NUMERICAL_EPSILON,
            ).clip(0.0, 1.0)
            return raw_slope * reliability, feature_mean

        beta, stress_mean = reliable_slope(stress)
        recovery_beta, recovery_mean = reliable_slope(recovery)
        beta[EQUITY_INDEX] = min(beta[EQUITY_INDEX], 0.0)
        beta[OIL_INDEX] = min(beta[OIL_INDEX], 0.0)
        recovery_beta[EQUITY_INDEX] = max(recovery_beta[EQUITY_INDEX], 0.0)
        recovery_beta[OIL_INDEX] = max(recovery_beta[OIL_INDEX], 0.0)
        stress_weights = regime_weights * stress
        _, raw_stress_covariance = weighted_mean_and_covariance(values, stress_weights)
        stress_effective = float(
            stress_weights.sum() ** 2
            / max(float(np.square(stress_weights).sum()), NUMERICAL_EPSILON)
        )
        stress_credibility = stress_effective / (stress_effective + ONE_CALENDAR_YEAR)
        regime_stress_covariance = (
            stress_credibility * raw_stress_covariance
            + (1.0 - stress_credibility) * regime_covariance
        )
        probability = float(current_p[regime_index])
        macro_mean += probability * regime_mean
        macro_covariance += probability * regime_covariance
        high_stress_covariance += probability * regime_stress_covariance
        stress_adjustment += probability * (
            beta * (current_s - stress_mean)
            + recovery_beta * (current_r - recovery_mean)
        )
        effective_samples[regime_column] = effective_sample
        credibility_rows[regime_column] = credibility
    covariance = (
        (1.0 - current_s) * macro_covariance
        + current_s * high_stress_covariance
    )
    detail = {
        "macro_expected_monthly_return": macro_mean.tolist(),
        "stress_return_adjustment": stress_adjustment.tolist(),
        "effective_regime_samples": effective_samples,
        "regime_credibility": credibility_rows,
    }
    return macro_mean + stress_adjustment, nearest_psd(covariance), detail


# %% [08_daily_technical_inputs]
def numeric_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.to_datetime(output.index).normalize()
    columns = ["open", "high", "low", "close", "volume"]
    for column in columns:
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output[columns].sort_index()
    return output.loc[~output.index.duplicated(keep="last")]


def load_daily_asset_ohlcv() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    assert CACHE_DIR is not None and RAW_DIR is not None
    raw = pd.read_csv(CACHE_DIR / "regime_lightgbm_ohlcv.csv", parse_dates=["date"])
    market = {
        str(symbol): numeric_ohlcv(group.set_index("date"))
        for symbol, group in raw.groupby("symbol")
    }
    required = {"KODEX200", "GLD", "USO", "USDKRW"}
    missing = sorted(required.difference(market))
    if missing:
        raise ValueError(f"OHLCV cache is missing symbols: {missing}")
    actual = market["KODEX200"].loc[
        market["KODEX200"].index > pd.Timestamp("2009-03-31")
    ].dropna(subset=["close"])
    with sqlite3.connect(RAW_DIR / "compass.db") as connection:
        proxy = pd.read_sql(
            "select date, open, high, low, close, volume "
            "from etf_prices where symbol = ? order by date",
            connection,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy = numeric_ohlcv(proxy.set_index("date"))
    first_actual = actual.index.min()
    nearest_position = proxy.index.get_indexer([first_actual], method="nearest")[0]
    scale = float(actual.loc[first_actual, "close"] / proxy.iloc[nearest_position]["close"])
    for column in ["open", "high", "low", "close"]:
        proxy[column] *= scale
    proxy = proxy.loc[proxy.index < first_actual].copy()
    proxy["volume_segment"] = "KOSPI200_proxy"
    actual = actual.copy()
    actual["volume_segment"] = "KODEX200_ETF"
    kodex = pd.concat([proxy, actual]).sort_index()

    bond_raw = pd.read_csv(RAW_DIR / "krx_bond_index.csv", encoding="cp949")
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
            raise ValueError(f"USDKRW does not cover {asset} history")
        for column in ["open", "high", "low", "close"]:
            frame[column] *= aligned_fx
        foreign[asset] = frame
    frames = {
        "KODEX200": kodex,
        "BOND": bond,
        "GLD": foreign["GLD"],
        "USO": foreign["USO"],
    }
    audit = {
        "foreign_prices_converted_to_krw": True,
        "bond_atr_uses_close_to_close_proxy": True,
        "assets": {
            asset: {
                "rows": int(len(frame)),
                "start": str(frame.index.min().date()),
                "end": str(frame.index.max().date()),
            }
            for asset, frame in frames.items()
        },
    }
    return frames, audit


def rolling_k_ratio(close: pd.Series, window: int = K_RATIO_DAYS) -> pd.Series:
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
        slope_se = math.sqrt(max(residual_variance / ssx, 0.0))
        if slope_se <= 1e-15:
            if abs(slope) <= 1e-15:
                return 0.0
            return slope / (np.finfo(float).eps * math.sqrt(window))
        return slope / (slope_se * math.sqrt(window))

    return log_price.rolling(window, min_periods=window).apply(calculate, raw=True)


def wilder_average(values: pd.Series, period: int = WILDER_DAYS) -> pd.Series:
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
        if np.isfinite(current) and np.isfinite(previous):
            output.iloc[position] = (previous * (period - 1) + current) / period
    return output


def average_true_range(frame: pd.DataFrame, period: int = WILDER_DAYS) -> pd.Series:
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
    return wilder_average(true_range, period)


def price_rsi(close: pd.Series, period: int = WILDER_DAYS) -> pd.Series:
    change = close.diff()
    gain = wilder_average(change.clip(lower=0.0), period)
    loss = wilder_average(-change.clip(upper=0.0), period)
    return 100.0 * gain.div((gain + loss).replace(0.0, np.nan))


def volume_rsi(
    close: pd.Series, volume: pd.Series, period: int = WILDER_DAYS
) -> pd.Series:
    change = close.diff()
    up = wilder_average(volume.where(change > 0.0, 0.0), period)
    down = wilder_average(volume.where(change < 0.0, 0.0), period)
    return 100.0 * up.div((up + down).replace(0.0, np.nan))


def build_daily_technical_features(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        frame = frames[asset].copy()
        output = pd.DataFrame(index=frame.index)
        output["k_ratio"] = rolling_k_ratio(frame["close"])
        output["k_score"] = output["k_ratio"] / (1.0 + output["k_ratio"].abs())
        output["atr"] = average_true_range(frame)
        output["natr"] = output["atr"] / frame["close"]
        output["atr_percentile"] = causal_expanding_midrank(output["natr"])
        if asset == "KODEX200":
            output["price_rsi"] = price_rsi(frame["close"])
            parts = [
                volume_rsi(segment["close"], segment["volume"])
                for _, segment in frame.groupby("volume_segment", sort=False)
            ]
            output["volume_rsi"] = pd.concat(parts).sort_index()
            output["price_strength"] = (output["price_rsi"] - 50.0) / 50.0
            output["volume_strength"] = (output["volume_rsi"] - 50.0) / 50.0
            output["technical_direction"] = output[
                ["k_score", "price_strength", "volume_strength"]
            ].mean(axis=1)
        else:
            output["technical_direction"] = output["k_score"]
        features[asset] = output.replace([np.inf, -np.inf], np.nan)
    return features


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
                "k_ratio", "k_score", "natr", "atr_percentile", "technical_direction"
            ]
            if asset == "KODEX200":
                required += ["price_rsi", "volume_rsi", "price_strength", "volume_strength"]
            known = daily_features[asset].loc[:month_end].dropna(subset=required)
            if known.empty:
                complete = False
                break
            current = known.iloc[-1]
            row[f"technical_signal_date_{asset}"] = known.index[-1]
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
) -> dict[str, Any]:
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
    return {
        "filtered_macro_expected_return": filtered_macro,
        "adjusted_covariance": scaling @ covariance @ scaling,
        "macro_confidence": confidence,
        "atr_variance_scale": variance_scale,
    }


# %% [09_fundamental_inputs]
def load_fundamental_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    assert RAW_DIR is not None
    earnings = pd.read_excel(
        RAW_DIR / "260829_fwdPE.EPS.rev.xlsx",
        header=13,
        usecols=range(5),
        engine="openpyxl",
    )
    earnings.columns = [
        "date", "forward_pe_12m", "forward_eps_12m",
        "eps_revision_1w_pct", "eps_revision_1m_pct",
    ]
    earnings["date"] = pd.to_datetime(earnings["date"], errors="coerce")
    earnings = earnings.dropna(subset=["date"]).set_index("date").sort_index()
    earnings = earnings.apply(pd.to_numeric, errors="coerce")
    earnings = earnings.loc[~earnings.index.duplicated(keep="last")]
    positive_eps = earnings["forward_eps_12m"].where(earnings["forward_eps_12m"] > 0.0)
    earnings["computed_eps_revision_21d_pct"] = np.log(positive_eps).diff(21) * 100.0

    credit = pd.read_excel(
        RAW_DIR / "260829_국고채.회사채.xlsx",
        header=None,
        skiprows=14,
        usecols=range(11),
        engine="openpyxl",
    )
    credit.columns = [
        "date", "ktb_1y_pct", "ktb_2y_pct", "ktb_3y_pct", "ktb_5y_pct",
        "ktb_10y_pct", "ktb_20y_pct", "ktb_30y_pct", "ktb_50y_pct",
        "corp_aa_minus_3y_pct", "corp_bbb_minus_3y_pct",
    ]
    credit["date"] = pd.to_datetime(credit["date"], errors="coerce")
    credit = credit.dropna(subset=["date"]).set_index("date").sort_index()
    credit = credit.apply(pd.to_numeric, errors="coerce")
    credit = credit.loc[~credit.index.duplicated(keep="last")]
    credit["aa_credit_spread_pctpt"] = (
        credit["corp_aa_minus_3y_pct"] - credit["ktb_3y_pct"]
    )
    credit["aa_spread_widening_20d_pctpt"] = credit[
        "aa_credit_spread_pctpt"
    ].diff(CREDIT_CHANGE_DAYS)
    daily = earnings.join(credit, how="outer")
    daily["earnings_yield_gap"] = (
        1.0 / daily["forward_pe_12m"].where(daily["forward_pe_12m"] > 0.0)
        - daily["ktb_10y_pct"] / 100.0
    )
    audit = {
        "earnings_first_valid": str(daily["eps_revision_1m_pct"].dropna().index.min().date()),
        "credit_first_valid": str(daily["aa_spread_widening_20d_pctpt"].dropna().index.min().date()),
        "winsorization": False,
        "parameter_grid": False,
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit


def build_monthly_fundamental_signals(daily: pd.DataFrame) -> pd.DataFrame:
    required = [
        "forward_pe_12m", "forward_eps_12m", "eps_revision_1m_pct",
        "computed_eps_revision_21d_pct", "ktb_3y_pct", "ktb_10y_pct",
        "corp_aa_minus_3y_pct", "aa_credit_spread_pctpt",
        "aa_spread_widening_20d_pctpt", "earnings_yield_gap",
    ]
    rows: list[dict[str, Any]] = []
    for signal_month, group in daily.groupby(daily.index.to_period("M")):
        complete = group.dropna(
            subset=["eps_revision_1m_pct", "aa_spread_widening_20d_pctpt", "earnings_yield_gap"]
        )
        if complete.empty:
            continue
        current = complete.iloc[-1]
        rows.append(
            {
                "target_month": signal_month + 1,
                "fundamental_signal_month": signal_month,
                "fundamental_signal_date": complete.index[-1],
                **{column: float(current[column]) for column in required},
            }
        )
    signals = pd.DataFrame(rows).set_index("target_month").sort_index()
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    signals["eps_revision_z"] = causal_zscore(signals["eps_revision_1m_pct"])
    signals["credit_widening_z"] = causal_zscore(signals["aa_spread_widening_20d_pctpt"])
    signals["credit_easing_z"] = -signals["credit_widening_z"]
    signals["valuation_gap_z"] = causal_zscore(signals["earnings_yield_gap"])
    stress_rank = causal_expanding_midrank(signals["aa_spread_widening_20d_pctpt"])
    prior_count = (
        signals["aa_spread_widening_20d_pctpt"].notna().shift(1)
        .fillna(False).astype(int).cumsum()
    )
    signals["credit_stress_rank"] = stress_rank.where(prior_count >= MIN_CAUSAL_MONTHS)
    signals["credit_stress_multiplier"] = 2.0 * signals["credit_stress_rank"]
    return signals.replace([np.inf, -np.inf], np.nan)


def nonnegative_univariate_slope(feature: pd.Series, target: pd.Series) -> float:
    complete = pd.concat([feature, target], axis=1).dropna()
    if len(complete) < MIN_CAUSAL_MONTHS:
        return 0.0
    x = complete.iloc[:, 0].to_numpy(dtype=float)
    y = complete.iloc[:, 1].to_numpy(dtype=float)
    x_std = float(x.std(ddof=1))
    if not np.isfinite(x_std) or x_std <= 0.0:
        return 0.0
    x = (x - x.mean()) / x_std
    denominator = float(x @ x)
    if denominator <= 0.0:
        return 0.0
    raw = float(x @ (y - y.mean()) / denominator)
    return max(raw, 0.0)


def forward_compound(series: pd.Series, horizon: int) -> pd.Series:
    legs = [series.shift(-offset) for offset in range(horizon)]
    frame = pd.concat(legs, axis=1)
    valid = frame.notna().all(axis=1)
    return frame.add(1.0).prod(axis=1).sub(1.0).where(valid)


def add_causal_return_calibration(
    signals: pd.DataFrame, equity_returns: pd.Series
) -> pd.DataFrame:
    output = signals.copy()
    forward_12m_return = forward_compound(equity_returns, 12)
    rows: list[dict[str, Any]] = []
    for month in output.index:
        history = output.index[output.index < month].intersection(equity_returns.index)
        eps_feature = output.loc[history, "eps_revision_1m_pct"]
        credit_feature = -output.loc[history, "aa_spread_widening_20d_pctpt"]
        target = equity_returns.loc[history]
        eps_slope = nonnegative_univariate_slope(eps_feature, target)
        valuation_history = output.index[output.index <= month - 12].intersection(
            forward_12m_return.index
        )
        valuation_feature = output.loc[valuation_history, "earnings_yield_gap"]
        valuation_target = forward_12m_return.loc[valuation_history]
        valuation_slope = nonnegative_univariate_slope(valuation_feature, valuation_target)
        rows.append(
            {
                "target_month": month,
                "calibration_observations": int(
                    pd.concat([eps_feature, credit_feature, target], axis=1).dropna().shape[0]
                ),
                "eps_calibration_slope": eps_slope,
                "valuation_calibration_slope_12m": valuation_slope,
                "eps_mu_adjustment_KODEX200": eps_slope * float(output.loc[month, "eps_revision_z"]),
                "valuation_mu_adjustment_KODEX200": (
                    valuation_slope * float(output.loc[month, "valuation_gap_z"]) / 12.0
                ),
            }
        )
    calibrated = pd.DataFrame(rows).set_index("target_month")
    calibrated.index = pd.PeriodIndex(calibrated.index, freq="M")
    return output.join(calibrated)


# %% [10_gvz_ovx_overlay]
def load_fred_series(path: Path, value_column: str) -> pd.Series:
    raw = pd.read_csv(path)
    required = {"observation_date", value_column}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing: {sorted(missing)}")
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw[value_column] = pd.to_numeric(raw[value_column], errors="coerce")
    series = raw.dropna(subset=["observation_date"]).set_index("observation_date")[value_column].sort_index()
    series = series.loc[~series.index.duplicated(keep="last")]
    return series.where(series > 0.0)


def load_asset_implied_volatility_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    assert RAW_DIR is not None
    gvz = load_fred_series(RAW_DIR / "GVZCLS.csv", "GVZCLS").rename("gvz")
    ovx = load_fred_series(RAW_DIR / "OVXCLS.csv", "OVXCLS").rename("ovx")
    daily = pd.concat([gvz, ovx], axis=1).sort_index()
    for sensor in ("gvz", "ovx"):
        rank, count = rank_after_prior_history(daily[sensor])
        daily[f"{sensor}_causal_rank"] = rank
        daily[f"{sensor}_prior_valid_observations"] = count
    audit = {
        "gvz_first_valid": str(gvz.dropna().index.min().date()),
        "gvz_last_valid": str(gvz.dropna().index.max().date()),
        "gvz_valid_observations": int(gvz.notna().sum()),
        "ovx_first_valid": str(ovx.dropna().index.min().date()),
        "ovx_last_valid": str(ovx.dropna().index.max().date()),
        "ovx_valid_observations": int(ovx.notna().sum()),
        "minimum_prior_observations": MIN_SENSOR_HISTORY,
        "directional_mu_effect": False,
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit


def build_monthly_asset_volatility_signals(
    daily: pd.DataFrame, target_months: pd.PeriodIndex
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        signal_month = target_month - 1
        month_end = signal_month.to_timestamp("M")
        row: dict[str, Any] = {
            "target_month": target_month,
            "asset_vol_signal_month": signal_month,
        }
        for sensor, asset in (("gvz", "GLD"), ("ovx", "USO")):
            known = daily.loc[
                :month_end,
                [sensor, f"{sensor}_causal_rank", f"{sensor}_prior_valid_observations"],
            ].dropna(subset=[sensor])
            if known.empty:
                value, rank, count, signal_date = np.nan, np.nan, 0, pd.NaT
            else:
                current = known.iloc[-1]
                value = float(current[sensor])
                rank = float(current[f"{sensor}_causal_rank"])
                count = int(current[f"{sensor}_prior_valid_observations"])
                signal_date = known.index[-1]
            active = bool(np.isfinite(rank) and count >= MIN_SENSOR_HISTORY)
            row.update(
                {
                    f"{sensor}_signal_date": signal_date,
                    f"{sensor}_level": value,
                    f"{sensor}_causal_rank": rank,
                    f"{sensor}_prior_valid_observations": count,
                    f"{sensor}_active": active,
                    f"{sensor}_{asset.lower()}_variance_multiplier": 1.0 + rank if active else 1.0,
                }
            )
        rows.append(row)
    signals = pd.DataFrame(rows).set_index("target_month")
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    return signals.replace([np.inf, -np.inf], np.nan)


# %% [11_slsqp_optimizer]
def project_to_long_only_simplex(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Weights must be finite")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    rho = int(np.flatnonzero(candidates)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    return projected / projected.sum()


def cdar(returns: np.ndarray, alpha: float = CDAR_CONFIDENCE) -> float:
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[-len(wealth):] - 1.0
    count = max(1, int(math.ceil((1.0 - alpha) * len(drawdown))))
    return float(np.mean(np.sort(drawdown)[:count]))


def expected_transaction_cost(weights: np.ndarray, pretrade: np.ndarray) -> float:
    change = weights - pretrade
    smooth_absolute_change = np.sqrt(change**2 + NUMERICAL_EPSILON)
    trade_cost = float(smooth_absolute_change.sum()) * DOMESTIC_TRADE_COST
    foreign_indices = [GOLD_INDEX, OIL_INDEX]
    foreign_change = float(change[foreign_indices].sum())
    fx_cost = math.sqrt(foreign_change**2 + NUMERICAL_EPSILON) * FOREIGN_WEIGHT_CHANGE_COST
    return trade_cost + fx_cost


def mode_uses_gvz(mode: str) -> bool:
    return mode in {"gvz_gold_risk", "gvz_ovx_asset_risk"}


def mode_uses_ovx(mode: str) -> bool:
    return mode in {"ovx_oil_risk", "gvz_ovx_asset_risk"}


def solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    fundamental_signal: pd.Series,
    asset_vol_signal: pd.Series,
    pretrade: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, base_covariance, moment_detail = estimate_conditional_moments(
        history,
        historical_probabilities,
        current_probabilities,
        historical_stress,
        current_stress,
        historical_recovery,
        current_recovery,
    )
    macro_expected_return = np.asarray(
        moment_detail["macro_expected_monthly_return"], dtype=float
    )
    stress_adjustment = np.asarray(
        moment_detail["stress_return_adjustment"], dtype=float
    ).copy()
    technical = apply_technical_inputs(
        macro_expected_return, base_covariance, technical_signal
    )
    filtered_macro = np.asarray(
        technical["filtered_macro_expected_return"], dtype=float
    ).copy()

    eps_mu = float(fundamental_signal["eps_mu_adjustment_KODEX200"])
    valuation_mu = float(fundamental_signal["valuation_mu_adjustment_KODEX200"])
    credit_stress_multiplier = float(fundamental_signal["credit_stress_multiplier"])
    stress_adjustment[EQUITY_INDEX] *= credit_stress_multiplier
    filtered_macro[EQUITY_INDEX] += eps_mu + valuation_mu
    expected_return = filtered_macro + stress_adjustment

    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)
    credit_variance_multiplier = 1.0 + float(fundamental_signal["credit_stress_rank"])
    credit_scaling = np.eye(len(ASSETS), dtype=float)
    credit_scaling[EQUITY_INDEX, EQUITY_INDEX] = math.sqrt(credit_variance_multiplier)
    covariance = credit_scaling @ covariance @ credit_scaling

    gvz_multiplier = (
        float(asset_vol_signal["gvz_gld_variance_multiplier"])
        if mode_uses_gvz(mode)
        else 1.0
    )
    ovx_multiplier = (
        float(asset_vol_signal["ovx_uso_variance_multiplier"])
        if mode_uses_ovx(mode)
        else 1.0
    )
    asset_scaling = np.eye(len(ASSETS), dtype=float)
    asset_scaling[GOLD_INDEX, GOLD_INDEX] = math.sqrt(gvz_multiplier)
    asset_scaling[OIL_INDEX, OIL_INDEX] = math.sqrt(ovx_multiplier)
    covariance = asset_scaling @ covariance @ asset_scaling

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
        downside_semivariance = float(np.mean(np.minimum(realized_history, 0.0) ** 2))
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
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": utility,
        }

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: CATASTROPHE_ANNUAL_VOLATILITY - annual_volatility(weights),
        },
        {
            "type": "ineq",
            "fun": lambda weights: CATASTROPHE_CDAR
            + cdar(historical_returns @ weights, CDAR_CONFIDENCE),
        },
    ]
    result = minimize(
        lambda weights: -portfolio_values(weights)["monthly_utility"],
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
            raise RuntimeError(f"Both SLSQP solves failed: {result.message}; {fallback.message}")
        result = fallback
        weights = project_to_long_only_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = cdar(historical_returns @ weights, CDAR_CONFIDENCE)
    return weights, {
        **values,
        "policy": f"Stage36_{mode}",
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_iterations": int(result.nit),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": CATASTROPHE_ANNUAL_VOLATILITY - annual_vol,
        "cdar_slack": CATASTROPHE_CDAR + historical_cdar,
        "eps_mu_adjustment_KODEX200": eps_mu,
        "valuation_mu_adjustment_KODEX200": valuation_mu,
        "credit_stress_confirmation_multiplier": credit_stress_multiplier,
        "credit_equity_variance_multiplier": credit_variance_multiplier,
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
        "gvz_mu_adjustment_GLD": 0.0,
        "ovx_mu_adjustment_USO": 0.0,
        "expected_mu_GLD": float(expected_return[GOLD_INDEX]),
        "expected_mu_USO": float(expected_return[OIL_INDEX]),
    }


# %% [12_monthly_backtest]
def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
        "credit_stress_rank",
    ]
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(asset_vol_signals.index)
    months = months.intersection(
        fundamental_signals.dropna(subset=required_fundamental).index
    )
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav, peak = 1.0, 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < ONE_CALENDAR_YEAR:
            continue
        probability = probabilities.loc[month]
        asset_vol_signal = asset_vol_signals.loc[month]
        weights, detail = solve_weights(
            history,
            probabilities.loc[probabilities.index < month],
            probability,
            stress_signals.loc[stress_signals.index < month, "stress_score"],
            float(stress_signals.loc[month, "stress_score"]),
            stress_signals.loc[stress_signals.index < month, "recovery_score"],
            float(stress_signals.loc[month, "recovery_score"]),
            technical_signals.loc[month],
            fundamental_signals.loc[month],
            asset_vol_signal,
            pretrade,
            mode,
        )
        change = weights - pretrade
        turnover = float(np.abs(change).sum()) if first_trade else 0.5 * float(np.abs(change).sum())
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        fx_cost = abs(float(change[[GOLD_INDEX, OIL_INDEX]].sum())) * FOREIGN_WEIGHT_CHANGE_COST
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
                "asset_vol_signal_month": asset_vol_signal["asset_vol_signal_month"],
                "gvz_signal_date": asset_vol_signal["gvz_signal_date"],
                "ovx_signal_date": asset_vol_signal["ovx_signal_date"],
                "gvz_level": float(asset_vol_signal["gvz_level"]),
                "ovx_level": float(asset_vol_signal["ovx_level"]),
                "gvz_causal_rank": float(asset_vol_signal["gvz_causal_rank"]),
                "ovx_causal_rank": float(asset_vol_signal["ovx_causal_rank"]),
                "gvz_active": bool(asset_vol_signal["gvz_active"]),
                "ovx_active": bool(asset_vol_signal["ovx_active"]),
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **{f"w_{asset}": float(weights[i]) for i, asset in enumerate(ASSETS)},
                **detail,
            }
        )
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


# %% [13_performance_metrics]
def performance_summary(returns: pd.Series) -> pd.Series:
    values = pd.Series(returns).dropna()
    wealth = (1.0 + values).cumprod()
    years = len(values) / 12.0
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    volatility = values.std(ddof=1) * math.sqrt(12.0)
    sharpe = (
        values.mean() / values.std(ddof=1) * math.sqrt(12.0)
        if values.std(ddof=1) > 0
        else np.nan
    )
    drawdown = wealth / wealth.cummax() - 1.0
    mdd = float(drawdown.min())
    downside = np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)) * math.sqrt(12.0)
    return pd.Series(
        {
            "Months": len(values),
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "Sortino": values.mean() * 12.0 / downside if downside > 0 else np.nan,
            "MDD": mdd,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "FinalMultiple": wealth.iloc[-1],
            "PositiveMonths": (values > 0.0).mean(),
        }
    )


def metric_row(
    name: str, path: pd.DataFrame, period: str, start: pd.Period, end: pd.Period
) -> dict[str, Any]:
    view = path.loc[start:end]
    metrics = performance_summary(view["return"])
    return {
        "Strategy": name,
        "Period": period,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{key: float(value) for key, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean()),
        "TotalCost": float(view[["trade_cost", "fx_cost"]].sum().sum()),
    }


def performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_end = min(path.index.max() for path in paths.values())
    periods = {
        "full_2007_2026": (FULL_START, common_end),
        "common_2010_2026": (COMMON_START, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    return pd.DataFrame(
        [
            metric_row(name, path, period, start, end)
            for name, path in paths.items()
            for period, (start, end) in periods.items()
        ]
    )


def solver_summary(path: pd.DataFrame) -> dict[str, Any]:
    return {
        "months": int(len(path)),
        "successes": int(path["solver_success"].sum()),
        "fallbacks": int(path["used_fallback"].sum()),
        "maximum_weight_sum_error": float(path["sum_error"].max()),
        "minimum_volatility_slack": float(path["volatility_slack"].min()),
        "minimum_cdar_slack": float(path["cdar_slack"].min()),
    }


def return_metrics(values: np.ndarray) -> dict[str, float]:
    years = len(values) / 12.0
    nav = np.cumprod(1.0 + values)
    return {
        "CAGR": float(nav[-1] ** (1.0 / years) - 1.0),
        "Volatility": float(values.std(ddof=1) * math.sqrt(12.0)),
        "Sharpe": float(values.mean() / values.std(ddof=1) * math.sqrt(12.0)),
        "MDD": float(np.min(nav / np.maximum.accumulate(nav) - 1.0)),
    }


def paired_block_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    replications: int = 2000,
    block_months: int = 12,
) -> pd.DataFrame:
    common = baseline.index.intersection(candidate.index)
    base = baseline.loc[common].to_numpy(dtype=float)
    test = candidate.loc[common].to_numpy(dtype=float)
    rng = np.random.default_rng(20260829)
    rows: list[dict[str, float]] = []
    blocks_needed = math.ceil(len(common) / block_months)
    for _ in range(replications):
        starts = rng.integers(0, len(common), size=blocks_needed)
        indices = np.concatenate(
            [(start + np.arange(block_months)) % len(common) for start in starts]
        )[: len(common)]
        base_metrics = return_metrics(base[indices])
        test_metrics = return_metrics(test[indices])
        rows.append(
            {
                "delta_CAGR": test_metrics["CAGR"] - base_metrics["CAGR"],
                "delta_Sharpe": test_metrics["Sharpe"] - base_metrics["Sharpe"],
                "delta_MDD": test_metrics["MDD"] - base_metrics["MDD"],
            }
        )
    draws = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "Metric": column,
                "Mean": float(draws[column].mean()),
                "P05": float(draws[column].quantile(0.05)),
                "P50": float(draws[column].quantile(0.50)),
                "P95": float(draws[column].quantile(0.95)),
                "ProbabilityPositive": float((draws[column] > 0.0).mean()),
                "Replications": replications,
                "BlockMonths": block_months,
            }
            for column in draws.columns
        ]
    )


# %% [14_future_risk_diagnostics]
def realized_volatility_signal(close: pd.Series) -> pd.Series:
    daily = np.log(close.where(close > 0.0)).diff()
    rolling = daily.rolling(21, min_periods=15).std(ddof=1) * math.sqrt(252.0)
    monthly = rolling.groupby(rolling.index.to_period("M")).last()
    monthly.index = pd.PeriodIndex(monthly.index + 1, freq="M")
    return monthly.rename("realized_vol_21d")


def forward_risk_targets(close: pd.Series) -> pd.DataFrame:
    close = close.dropna().sort_index()
    log_returns = np.log(close).diff()
    periods = pd.period_range(
        close.index.min().to_period("M"), RESEARCH_END, freq="M"
    )
    rows: list[dict[str, Any]] = []
    for period in periods:
        month_returns = log_returns.loc[
            log_returns.index.to_period("M") == period
        ].dropna()
        row: dict[str, Any] = {
            "target_month": period,
            "future_realized_vol_1m": (
                float(month_returns.std(ddof=1) * math.sqrt(252.0))
                if len(month_returns) >= 15
                else np.nan
            ),
        }
        for horizon, minimum_prices in ((1, 16), (3, 46)):
            end_period = period + horizon - 1
            before = close.loc[close.index < period.start_time]
            within = close.loc[
                (close.index >= period.start_time)
                & (close.index <= end_period.end_time)
            ]
            if before.empty or len(within) < minimum_prices - 1:
                value = np.nan
            else:
                path = pd.concat([before.iloc[[-1]], within])
                wealth = path / float(path.iloc[0])
                value = float(-(wealth / wealth.cummax() - 1.0).min())
            row[f"future_max_drawdown_{horizon}m"] = value
        rows.append(row)
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def causal_tail_event(monthly_return: pd.Series) -> pd.Series:
    threshold = (
        monthly_return.shift(1)
        .expanding(min_periods=MIN_CAUSAL_MONTHS)
        .quantile(0.05)
    )
    return pd.Series(
        np.where(
            monthly_return.notna() & threshold.notna(),
            (monthly_return <= threshold).astype(float),
            np.nan,
        ),
        index=monthly_return.index,
        name="future_left_tail_1m",
    )


def build_asset_risk_research_frame(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    market: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frame = signals.copy()
    frame["vix6_stress_score"] = stress["stress_score"]
    frame["macro_fragility"] = (
        probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    )
    for sensor, asset in (("gvz", "GLD"), ("ovx", "USO")):
        prefix = asset.lower()
        close = market[asset]["close"].dropna()
        frame[f"{prefix}_recent_1m_return"] = returns[asset].shift(1)
        frame[f"{prefix}_realized_vol_21d"] = realized_volatility_signal(close)
        risk_targets = forward_risk_targets(close).rename(
            columns={
                "future_realized_vol_1m": f"{prefix}_future_realized_vol_1m",
                "future_max_drawdown_1m": f"{prefix}_future_max_drawdown_1m",
                "future_max_drawdown_3m": f"{prefix}_future_max_drawdown_3m",
            }
        )
        frame = frame.join(risk_targets, how="left")
        monthly_close = close.groupby(close.index.to_period("M")).last()
        monthly_return = monthly_close.pct_change().loc[:RESEARCH_END]
        frame[f"{prefix}_future_left_tail_1m"] = causal_tail_event(
            monthly_return
        )
    return frame.loc[FULL_START:RESEARCH_END].replace(
        [np.inf, -np.inf], np.nan
    )


def standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        std = float(frame[column].std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"No usable variation in {column}")
        output[column] = (frame[column] - frame[column].mean()) / std
    return output


def asset_risk_predictive_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "available_full": (FULL_START, RESEARCH_END),
        "common_2010_2026": (COMMON_START, RESEARCH_END),
        "locked_2018_2026": (LOCKED_START, RESEARCH_END),
    }
    targets = (
        ("future_realized_vol_1m", 1),
        ("future_max_drawdown_1m", 1),
        ("future_max_drawdown_3m", 3),
        ("future_left_tail_1m", 1),
    )
    for sensor, asset in (("gvz", "GLD"), ("ovx", "USO")):
        prefix = asset.lower()
        feature = f"{sensor}_causal_rank"
        controls = [
            f"{prefix}_recent_1m_return",
            f"{prefix}_realized_vol_21d",
            "vix6_stress_score",
            "macro_fragility",
        ]
        for period_name, (start, end) in periods.items():
            view = frame.loc[start:end]
            for target_suffix, lags in targets:
                target = f"{prefix}_{target_suffix}"
                for model, predictors in (
                    ("SensorOnly", [feature]),
                    ("FullControls", [feature, *controls]),
                ):
                    complete = view[[target, *predictors]].dropna()
                    if len(complete) < 36:
                        continue
                    standardized = standardize(complete, predictors)
                    fit = sm.OLS(
                        complete[target], sm.add_constant(standardized)
                    ).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
                    ic, ic_p = spearmanr(
                        complete[feature], complete[target], nan_policy="omit"
                    )
                    rows.append(
                        {
                            "Sensor": sensor.upper(),
                            "Asset": asset,
                            "Period": period_name,
                            "Target": target_suffix,
                            "Model": model,
                            "Observations": int(len(complete)),
                            "SensorStandardizedBeta": float(
                                fit.params[feature]
                            ),
                            "SensorHACPValue": float(fit.pvalues[feature]),
                            "SensorSpearmanIC": float(ic),
                            "SensorICPValue": float(ic_p),
                            "AdjustedR2": float(fit.rsquared_adj),
                            "HACLags": lags,
                        }
                    )
    return pd.DataFrame(rows)


# %% [15_research_orchestration]
def run_stage36_research(
    save: bool = True, bootstrap_replications: int = 2000
) -> dict[str, Any]:
    assert OUTPUT_DIR is not None
    returns, _ = load_monthly_asset_returns()
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress = build_monthly_stress_signals(returns.index, daily_stress)
    market, market_audit = load_daily_asset_ohlcv()
    technical_features = build_daily_technical_features(market)
    technical = build_monthly_technical_signals(
        returns.index, technical_features
    )
    raw_fundamental, fundamental_audit = load_fundamental_daily()
    fundamental = build_monthly_fundamental_signals(raw_fundamental)
    equity_close = market["KODEX200"]["close"].dropna()
    equity_monthly_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    calibrated = add_causal_return_calibration(
        fundamental, equity_monthly_close.pct_change()
    )
    daily_asset_vol, asset_vol_audit = load_asset_implied_volatility_daily()
    asset_vol = build_monthly_asset_volatility_signals(
        daily_asset_vol, returns.index
    )

    modes = {
        "Stage35_Frozen": "baseline_reproduction",
        "Stage36_GVZGoldRisk": "gvz_gold_risk",
        "Stage36_OVXOilRisk": "ovx_oil_risk",
        "Stage36_GVZ_OVXAssetRisk": "gvz_ovx_asset_risk",
    }
    paths = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated,
            asset_vol,
            mode,
        )
        for name, mode in modes.items()
    }
    performance = performance_table(paths)
    boot_rows: list[pd.DataFrame] = []
    baseline = paths["Stage35_Frozen"]
    candidate = paths["Stage36_GVZ_OVXAssetRisk"]
    for period_name, start in (
        ("full_2007_2026", FULL_START),
        ("common_2010_2026", COMMON_START),
    ):
        summary = paired_block_bootstrap(
            baseline.loc[start:RESEARCH_END, "return"],
            candidate.loc[start:RESEARCH_END, "return"],
            replications=bootstrap_replications,
        )
        summary.insert(0, "Period", period_name)
        summary.insert(0, "Candidate", "Stage36_GVZ_OVXAssetRisk")
        boot_rows.append(summary)
    bootstrap = pd.concat(boot_rows, ignore_index=True)

    risk_frame = build_asset_risk_research_frame(
        asset_vol, returns, probabilities, stress, market
    )
    risk_tests = asset_risk_predictive_regressions(risk_frame)
    source_checks = {
        "signal_month_precedes_target": bool(
            (asset_vol["asset_vol_signal_month"] < asset_vol.index).all()
        ),
        "no_backfill_before_252": bool(
            asset_vol.loc[
                ~asset_vol["gvz_active"],
                "gvz_gld_variance_multiplier",
            ].eq(1.0).all()
            and asset_vol.loc[
                ~asset_vol["ovx_active"],
                "ovx_uso_variance_multiplier",
            ].eq(1.0).all()
        ),
        "variance_multipliers_between_one_and_two": bool(
            asset_vol[
                [
                    "gvz_gld_variance_multiplier",
                    "ovx_uso_variance_multiplier",
                ]
            ].ge(1.0).all().all()
            and asset_vol[
                [
                    "gvz_gld_variance_multiplier",
                    "ovx_uso_variance_multiplier",
                ]
            ].le(2.0).all().all()
        ),
        "no_leverage_long_only_sum_to_one": bool(
            all(
                np.allclose(path[WEIGHT_COLUMNS].sum(axis=1), 1.0)
                and (path[WEIGHT_COLUMNS] >= -1e-10).all().all()
                and (path[WEIGHT_COLUMNS] <= 1.0 + 1e-10).all().all()
                for path in paths.values()
            )
        ),
        "no_directional_gvz_ovx_mu": bool(
            all(
                path["gvz_mu_adjustment_GLD"].eq(0.0).all()
                and path["ovx_mu_adjustment_USO"].eq(0.0).all()
                for path in paths.values()
            )
        ),
        "all_solvers_feasible": bool(
            all(
                path["solver_success"].all()
                and not path["used_fallback"].any()
                and path["volatility_slack"].min() >= -1e-7
                and path["cdar_slack"].min() >= -1e-7
                for path in paths.values()
            )
        ),
    }
    report = {
        "performance": performance,
        "bootstrap": bootstrap,
        "risk_tests": risk_tests,
        "paths": paths,
        "asset_vol_signals": asset_vol,
        "daily_asset_vol": daily_asset_vol,
        "source_checks": source_checks,
        "solver_audit": {
            name: solver_summary(path) for name, path in paths.items()
        },
        "data_audit": {
            "asset_vol": asset_vol_audit,
            "fundamental": fundamental_audit,
            "market": market_audit,
        },
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage35.csv", index=False
        )
        risk_tests.to_csv(
            OUTPUT_DIR / "asset_risk_predictive_regressions.csv", index=False
        )
        asset_vol.to_csv(
            OUTPUT_DIR / "monthly_asset_volatility_signals.csv"
        )
        daily_asset_vol.to_csv(OUTPUT_DIR / "normalized_gvz_ovx_daily.csv")
        for name, path in paths.items():
            path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        serializable = {
            "source_checks": source_checks,
            "solver_audit": report["solver_audit"],
            "data_audit": report["data_audit"],
        }
        (OUTPUT_DIR / "colab_validation_report.json").write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


# %% [16_results_and_charts]
def display_stage36_results(report: dict[str, Any]) -> None:
    performance = report["performance"]
    columns = [
        "Strategy",
        "Period",
        "CAGR",
        "Volatility",
        "Sharpe",
        "MDD",
        "AvgTurnover",
    ]
    display_table = performance[columns].copy()
    for column in ["CAGR", "Volatility", "MDD", "AvgTurnover"]:
        display_table[column] = display_table[column].map(
            lambda value: f"{value * 100:.3f}%"
        )
    print("\n성과 비교")
    try:
        from IPython.display import display

        display(display_table)
    except ImportError:
        print(display_table.to_string(index=False))

    paths = report["paths"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {
        "Stage35_Frozen": "#62788d",
        "Stage36_GVZGoldRisk": "#b38a32",
        "Stage36_OVXOilRisk": "#b35b45",
        "Stage36_GVZ_OVXAssetRisk": "#177b6d",
    }
    for name, path in paths.items():
        nav = (1.0 + path["return"]).cumprod()
        axes[0].plot(
            nav.index.to_timestamp(),
            nav,
            label=name,
            color=colors[name],
        )
        axes[1].plot(
            path.index.to_timestamp(),
            path["drawdown"] * 100.0,
            label=name,
            color=colors[name],
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Net NAV (log scale)")
    axes[1].set_title("Monthly drawdown (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    print("\n인과성·제약 검사")
    print(json.dumps(report["source_checks"], ensure_ascii=False, indent=2))
    print("\nSLSQP 검사")
    print(json.dumps(report["solver_audit"], ensure_ascii=False, indent=2))


def zip_colab_outputs() -> Path:
    assert OUTPUT_DIR is not None
    archive = OUTPUT_DIR.parent / "stage36_colab_outputs.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as handle:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(OUTPUT_DIR.parent))
    return archive
