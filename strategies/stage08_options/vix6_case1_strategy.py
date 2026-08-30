from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy import (
    balanced_logistic_spec,
    build_domestic_features,
    build_no_sjm_signals,
    fixed_robust_overlay,
    forward_path_loss,
    run_neutral_factor_blend,
)
from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    load_monthly_asset_returns,
    run_backtest,
)
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
from strategies.stage06_vkospi.vkospi_extended_diagnostics import DOMESTIC_FEATURES, OAP_COMPOSITES
from strategies.stage06_vkospi.vkospi_model_robustness import (
    fit_logistic_candidate,
    make_tail_factor,
    run_factor_vol_target,
)
from strategies.stage06_vkospi.vkospi_robust_dynamic_experiment import (
    RobustStressConfig,
    align_features_to_arrays,
    build_robust_daily_features,
    stress_from_features,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw_data"
RESULTS = ROOT / "results"
OPTION_PATH = RAW_DIR / "KOSPI200OptionPrice.csv"
SURFACE_PATH = RESULTS / "vix6_case1_surface_daily.csv"
FEATURE_PATH = RESULTS / "vix6_case1_features_daily.csv"
DAILY_PATH = RESULTS / "vix6_case1_daily.csv"
MONTHLY_PATH = RESULTS / "vix6_case1_monthly.csv"
RECONCILED_PATH = RESULTS / "vix6_case1_reconciled.csv"
CALIBRATION_PATH = RESULTS / "vix6_case1_calibration.csv"
COMPARISON_PATH = RESULTS / "vix6_case1_comparison.csv"
REPORT_PATH = RESULTS / "vix6_case1_validation.json"

CALIBRATION_END = pd.Period("2012-12", freq="M")
VALIDATION_START = pd.Period("2013-01", freq="M")
TARGET_DTE = 30.0
RANDOM_STATE = 20260828


@dataclass(frozen=True)
class Case1Config:
    tail_threshold: float = 1.0
    breadth_threshold: float = 0.25
    early_scale: float = 0.20
    confirmed_scale: float = 0.70
    recovery_relief: float = 0.35
    right_tail_relief: float = 0.35
    panic_relief: float = 0.20
    max_risk_transfer: float = 0.35
    rebalance_band: float = 0.20
    oil_cut_deflation: float = 1.0
    oil_cut_inflation: float = 1.0
    bond_share_deflation: float = 0.0
    bond_share_inflation: float = 0.0
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return (
            f"tt{self.tail_threshold:.2f}_bt{self.breadth_threshold:.2f}"
            f"_ew{self.early_scale:.2f}_cw{self.confirmed_scale:.2f}"
            f"_rr{self.recovery_relief:.2f}_ur{self.right_tail_relief:.2f}"
            f"_pr{self.panic_relief:.2f}_rt{self.max_risk_transfer:.2f}"
            f"_rb{self.rebalance_band:.2f}_od{self.oil_cut_deflation:.2f}"
            f"_oi{self.oil_cut_inflation:.2f}_bd{self.bond_share_deflation:.2f}"
            f"_bi{self.bond_share_inflation:.2f}"
        )


ROUTING_PRESETS = {
    "legacy_gold": {
        "oil_cut_deflation": 1.0,
        "oil_cut_inflation": 1.0,
        "bond_share_deflation": 0.0,
        "bond_share_inflation": 0.0,
    },
    "equity_only_gold": {
        "oil_cut_deflation": 0.0,
        "oil_cut_inflation": 0.0,
        "bond_share_deflation": 0.0,
        "bond_share_inflation": 0.0,
    },
    "macro_soft": {
        "oil_cut_deflation": 1.0,
        "oil_cut_inflation": 0.25,
        "bond_share_deflation": 0.50,
        "bond_share_inflation": 0.0,
    },
    "macro_strict": {
        "oil_cut_deflation": 1.0,
        "oil_cut_inflation": 0.0,
        "bond_share_deflation": 0.75,
        "bond_share_inflation": 0.0,
    },
    "macro_mixed": {
        "oil_cut_deflation": 1.0,
        "oil_cut_inflation": 0.25,
        "bond_share_deflation": 0.75,
        "bond_share_inflation": 0.25,
    },
}


def _second_thursday(month_start: pd.Series) -> pd.Series:
    offset = ((3 - month_start.dt.weekday) % 7) + 7
    return month_start + pd.to_timedelta(offset, unit="D")


def load_option_chain(
    path: Path = OPTION_PATH,
    require_iv: bool = True,
) -> pd.DataFrame:
    required = ["일자", "종목코드", "종목명", "종가", "내재변동성", "거래량"]
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, usecols=required)
    frame = raw.rename(
        columns={
            "일자": "date_session",
            "종목코드": "code",
            "종목명": "name",
            "종가": "price",
            "내재변동성": "iv_pct",
            "거래량": "volume",
        }
    )
    frame["date"] = pd.to_datetime(
        frame["date_session"].str.slice(0, 10), format="%Y/%m/%d", errors="coerce"
    )
    frame["session"] = frame["date_session"].str.extract(r"\(([^)]+)\)", expand=False)
    frame = frame.loc[frame["session"].eq("주간")].copy()

    name = frame["name"].fillna("")
    frame["option_type"] = np.where(
        name.str.contains(r"(?<![A-Z])C(?![A-Z])|콜", regex=True), "C", "P"
    )
    frame["strike"] = pd.to_numeric(
        name.str.extract(r"(\d+(?:\.\d+)?)\s*$", expand=False), errors="coerce"
    )
    full_expiry = name.str.extract(r"(?<!\d)(20\d{4})(?!\d)", expand=False)
    short_expiry = name.str.extract(r"(?<!\d)(\d{4})(?!\d)", expand=False)
    expiry_code = full_expiry.fillna(short_expiry)
    full_mask = full_expiry.notna()
    expiry_year = pd.to_numeric(expiry_code.str.slice(0, -2), errors="coerce")
    expiry_month = pd.to_numeric(expiry_code.str.slice(-2), errors="coerce")
    trade_year = frame["date"].dt.year
    inferred_year = (trade_year // 100) * 100 + expiry_year
    inferred_year = inferred_year.where(inferred_year >= trade_year - 1, inferred_year + 100)
    inferred_year = inferred_year.where(inferred_year <= trade_year + 5, inferred_year - 100)
    expiry_year = expiry_year.where(full_mask, inferred_year)
    expiry_month_start = pd.to_datetime(
        {
            "year": expiry_year,
            "month": expiry_month,
            "day": np.ones(len(frame), dtype=int),
        },
        errors="coerce",
    )
    frame["expiry"] = _second_thursday(expiry_month_start)
    frame["dte"] = (frame["expiry"] - frame["date"]).dt.days.astype(float)

    for column in ("price", "iv_pct", "volume"):
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    frame["iv"] = frame["iv_pct"] / 100.0
    valid = (
        frame["date"].notna()
        & frame["expiry"].notna()
        & frame["strike"].gt(0)
        & frame["price"].gt(0)
        & frame["dte"].between(5, 70)
    )
    if require_iv:
        valid &= frame["iv"].between(0.01, 2.0)
    output = frame.loc[
        valid,
        [
            "date",
            "expiry",
            "dte",
            "code",
            "option_type",
            "strike",
            "price",
            "iv",
            "volume",
        ],
    ].copy()
    output["volume"] = output["volume"].fillna(0.0).clip(lower=0.0)
    output = output.sort_values(["date", "expiry", "option_type", "strike"])
    if output.empty:
        raise ValueError("No usable day-session KOSPI200 option observations")
    return output


def _weighted_node(
    frame: pd.DataFrame,
    option_type: str | None,
    low_delta: float,
    high_delta: float,
    target_delta: float,
) -> tuple[float, float, int]:
    mask = frame["abs_delta"].between(low_delta, high_delta, inclusive="both")
    if option_type is not None:
        mask &= frame["option_type"].eq(option_type)
    sample = frame.loc[mask, ["iv", "log_moneyness", "abs_delta", "volume"]].dropna()
    if sample.empty:
        return np.nan, np.nan, 0
    median = float(sample["iv"].median())
    mad = float((sample["iv"] - median).abs().median())
    if mad > 0:
        sample = sample.loc[(sample["iv"] - median).abs() <= 4.5 * 1.4826 * mad]
    if sample.empty:
        return np.nan, np.nan, 0
    bandwidth = max((high_delta - low_delta) / 3.0, 0.025)
    proximity = np.exp(
        -0.5 * ((sample["abs_delta"] - target_delta) / bandwidth) ** 2
    )
    liquidity = 1.0 + np.log1p(sample["volume"].clip(upper=1_000_000)) / 12.0
    weights = (proximity * liquidity).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(sample))
    iv = float(np.average(sample["iv"], weights=weights))
    log_moneyness = float(np.average(sample["log_moneyness"], weights=weights))
    return iv, log_moneyness, int(len(sample))


def _expiry_surface(group: pd.DataFrame) -> dict[str, float] | None:
    deduped = group.sort_values(["option_type", "strike"]).drop_duplicates(
        ["option_type", "strike"], keep="last"
    )
    price = deduped.pivot(index="strike", columns="option_type", values="price")
    if not {"C", "P"}.issubset(price.columns):
        return None
    paired = price.dropna(subset=["C", "P"])
    paired = paired.loc[paired["C"].gt(0) & paired["P"].gt(0)]
    if paired.empty:
        return None
    closest = (paired["C"] - paired["P"]).abs().nsmallest(min(3, len(paired))).index
    synthetic_forwards = (
        closest.to_numpy(dtype=float)
        + paired.loc[closest, "C"].to_numpy(dtype=float)
        - paired.loc[closest, "P"].to_numpy(dtype=float)
    )
    forward = float(np.median(synthetic_forwards))
    if not np.isfinite(forward) or forward <= 0:
        return None

    work = deduped.copy()
    t = float(work["dte"].iloc[0] / 365.0)
    sqrt_t = math.sqrt(max(t, 1e-9))
    strike = work["strike"].to_numpy(dtype=float)
    sigma = work["iv"].to_numpy(dtype=float)
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * t) / (sigma * sqrt_t)
    work["abs_delta"] = np.where(work["option_type"].eq("C"), ndtr(d1), ndtr(-d1))
    work["log_moneyness"] = np.log(strike / forward)
    otm = (
        (work["option_type"].eq("C") & work["strike"].ge(forward))
        | (work["option_type"].eq("P") & work["strike"].le(forward))
    )
    work = work.loc[otm].copy()
    if work.empty:
        return None

    atm = _weighted_node(work, None, 0.35, 0.55, 0.48)
    put_shoulder = _weighted_node(work, "P", 0.15, 0.45, 0.30)
    call_shoulder = _weighted_node(work, "C", 0.15, 0.45, 0.30)
    put_tail = _weighted_node(work, "P", 0.01, 0.15, 0.08)
    call_tail = _weighted_node(work, "C", 0.01, 0.15, 0.08)
    nodes = (atm, put_shoulder, call_shoulder, put_tail, call_tail)
    if any(not np.isfinite(node[0]) for node in nodes):
        return None
    return {
        "dte": float(group["dte"].iloc[0]),
        "forward": forward,
        "atm_iv": atm[0],
        "atm_logm": atm[1],
        "atm_count": atm[2],
        "put_shoulder_iv": put_shoulder[0],
        "put_shoulder_logm": put_shoulder[1],
        "put_shoulder_count": put_shoulder[2],
        "call_shoulder_iv": call_shoulder[0],
        "call_shoulder_logm": call_shoulder[1],
        "call_shoulder_count": call_shoulder[2],
        "put_tail_iv": put_tail[0],
        "put_tail_logm": put_tail[1],
        "put_tail_count": put_tail[2],
        "call_tail_iv": call_tail[0],
        "call_tail_logm": call_tail[1],
        "call_tail_count": call_tail[2],
    }


def build_expiry_surfaces(option_chain: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = option_chain.groupby(["date", "expiry"], sort=True, observed=True)
    for (date, expiry), group in grouped:
        surface = _expiry_surface(group)
        if surface is not None:
            rows.append({"date": date, "expiry": expiry, **surface})
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("Could not construct any complete option surfaces")
    return output.sort_values(["date", "dte"]).reset_index(drop=True)


def _interpolate_iv(iv1: float, dte1: float, iv2: float, dte2: float) -> float:
    if dte1 == dte2:
        return float(iv1)
    weight2 = (TARGET_DTE - dte1) / (dte2 - dte1)
    weight1 = 1.0 - weight2
    total_variance = (
        weight1 * iv1 * iv1 * dte1 / 365.0
        + weight2 * iv2 * iv2 * dte2 / 365.0
    )
    return float(math.sqrt(max(total_variance / (TARGET_DTE / 365.0), 0.0)))


def interpolate_30d_surface(expiry_surfaces: pd.DataFrame) -> pd.DataFrame:
    iv_columns = [
        "atm_iv",
        "put_shoulder_iv",
        "call_shoulder_iv",
        "put_tail_iv",
        "call_tail_iv",
    ]
    linear_columns = [
        "forward",
        "atm_logm",
        "put_shoulder_logm",
        "call_shoulder_logm",
        "put_tail_logm",
        "call_tail_logm",
    ]
    count_columns = [column for column in expiry_surfaces if column.endswith("_count")]
    rows: list[dict[str, object]] = []
    for date, group in expiry_surfaces.groupby("date", sort=True):
        group = group.sort_values("dte")
        lower = group.loc[group["dte"].le(TARGET_DTE)].tail(1)
        upper = group.loc[group["dte"].ge(TARGET_DTE)].head(1)
        if not lower.empty and not upper.empty and lower.index[0] != upper.index[0]:
            left = lower.iloc[0]
            right = upper.iloc[0]
            dte1 = float(left["dte"])
            dte2 = float(right["dte"])
            weight2 = (TARGET_DTE - dte1) / (dte2 - dte1)
            weight1 = 1.0 - weight2
            row: dict[str, object] = {
                "date": date,
                "lower_dte": dte1,
                "upper_dte": dte2,
                "tenor_method": "bracketed_variance",
            }
            for column in iv_columns:
                row[column] = _interpolate_iv(
                    float(left[column]), dte1, float(right[column]), dte2
                )
            for column in linear_columns:
                row[column] = float(
                    weight1 * float(left[column]) + weight2 * float(right[column])
                )
            for column in count_columns:
                row[column] = float(min(float(left[column]), float(right[column])))
            rows.append(row)
            continue

        nearest = group.iloc[(group["dte"] - TARGET_DTE).abs().argsort()[:1]].iloc[0]
        if abs(float(nearest["dte"]) - TARGET_DTE) > 21:
            continue
        row = {
            "date": date,
            "lower_dte": float(nearest["dte"]),
            "upper_dte": float(nearest["dte"]),
            "tenor_method": "nearest_tenor",
        }
        for column in iv_columns + linear_columns + count_columns:
            row[column] = float(nearest[column])
        rows.append(row)

    output = pd.DataFrame(rows).set_index("date").sort_index()
    if output.empty:
        raise ValueError("No 30-day option surfaces were available")
    return output


def robust_zscore(series: pd.Series, window: int = 252, minimum: int = 126) -> pd.Series:
    median = series.rolling(window, min_periods=minimum).median()
    absolute = (series - median).abs()
    mad = absolute.rolling(window, min_periods=minimum).median()
    return ((series - median) / (1.4826 * mad.replace(0, np.nan))).clip(-6, 6)


def decompose_surface(surface: pd.DataFrame) -> pd.DataFrame:
    output = surface.copy()
    for column in [
        "atm_iv",
        "put_shoulder_iv",
        "call_shoulder_iv",
        "put_tail_iv",
        "call_tail_iv",
    ]:
        output[column] = 100.0 * output[column]

    output["put_skew_level"] = output["put_shoulder_iv"] - output["atm_iv"]
    output["call_skew_level"] = output["call_shoulder_iv"] - output["atm_iv"]
    put_ratio = output["put_tail_logm"].div(output["put_shoulder_logm"])
    call_ratio = output["call_tail_logm"].div(output["call_shoulder_logm"])
    expected_put_tail = output["atm_iv"] + put_ratio * output["put_skew_level"]
    expected_call_tail = output["atm_iv"] + call_ratio * output["call_skew_level"]
    output["downside_convexity_level"] = output["put_tail_iv"] - expected_put_tail
    output["upside_convexity_level"] = output["call_tail_iv"] - expected_call_tail

    shoulder_span = output["call_shoulder_logm"] - output["put_shoulder_logm"]
    fixed_strike_slope = (
        output["call_shoulder_iv"] - output["put_shoulder_iv"]
    ).div(shoulder_span.replace(0, np.nan))
    forward_return = np.log(output["forward"]).diff()
    output["sticky_strike"] = fixed_strike_slope.shift(1) * forward_return
    atm_change = output["atm_iv"].diff()
    output["parallel_shift"] = atm_change - output["sticky_strike"]
    output["put_skew"] = output["put_skew_level"].diff()
    output["call_skew"] = output["call_skew_level"].diff()
    output["downside_convexity"] = output["downside_convexity_level"].diff()
    output["upside_convexity"] = output["upside_convexity_level"].diff()

    for column in (
        "put_skew_level",
        "call_skew_level",
        "downside_convexity_level",
        "upside_convexity_level",
    ):
        smooth = output[column].rolling(5, min_periods=3).median()
        output[f"z_{column}"] = robust_zscore(smooth)

    output["left_tail"] = 0.5 * (
        output["z_put_skew_level"] + output["z_downside_convexity_level"]
    )
    output["right_tail"] = 0.5 * (
        output["z_call_skew_level"] + output["z_upside_convexity_level"]
    )
    output["asymmetry"] = output["left_tail"] - output["right_tail"]
    output["breadth_5"] = output["parallel_shift"].rolling(5, min_periods=3).sum()
    output["reaction_5"] = output["sticky_strike"].rolling(5, min_periods=3).sum()
    output["left_impulse_5"] = (
        output["put_skew"] + output["downside_convexity"]
    ).rolling(5, min_periods=3).sum()
    output["breadth_z"] = robust_zscore(output["breadth_5"])
    output["reaction_z"] = robust_zscore(output["reaction_5"])
    output["left_impulse_z"] = robust_zscore(output["left_impulse_5"])
    output["left_change_5"] = output["left_tail"].diff(5)
    output["forward_return_5"] = output["forward"].pct_change(5, fill_method=None)
    output["forward_return_21"] = output["forward"].pct_change(21, fill_method=None)
    output["decomposition_residual"] = (
        atm_change - output["sticky_strike"] - output["parallel_shift"]
    )
    return output.replace([np.inf, -np.inf], np.nan)


def build_vix6_features(force: bool = False) -> pd.DataFrame:
    if FEATURE_PATH.exists() and SURFACE_PATH.exists() and not force:
        cached = pd.read_csv(FEATURE_PATH, index_col=0, parse_dates=True)
        cached.index.name = "date"
        return cached.sort_index()
    option_chain = load_option_chain()
    expiry_surfaces = build_expiry_surfaces(option_chain)
    surface = interpolate_30d_surface(expiry_surfaces)
    features = decompose_surface(surface)
    SURFACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(SURFACE_PATH)
    features.to_csv(FEATURE_PATH)
    return features


def build_final_medium_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the latest no-SJM balanced-logistic strategy before its VKOSPI overlay."""
    returns, _ = load_monthly_asset_returns(False)
    signals, _ = build_no_sjm_signals(returns)
    defensive = run_backtest(returns, signals, StrategyConfig(), mode="proposed")
    domestic = build_domestic_features(signals, returns)
    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    composites.index = pd.PeriodIndex(composites.index, freq="M")
    neutral = run_neutral_factor_blend(returns, signals, defensive)
    data = domestic[DOMESTIC_FEATURES].join(composites[OAP_COMPOSITES], how="left")
    data = data.loc[data.index.intersection(neutral.index)].copy()
    path_loss = forward_path_loss(neutral.loc[data.index, "return"], horizon=2)
    data["tail_event"] = (path_loss < -0.05).where(path_loss.notna()).astype(float)
    probability, _ = fit_logistic_candidate(data, balanced_logistic_spec())
    factor = make_tail_factor(probability, data["tail_event"])
    medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    return medium, signals


def build_aligned_case1_inputs(
    arrays: dict[str, object],
    levels: pd.DataFrame,
    signals: pd.DataFrame,
    option_features: pd.DataFrame,
) -> pd.DataFrame:
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])
    aligned = pd.DataFrame(index=pd.RangeIndex(len(signal_dates)))
    valid = ~signal_dates.isna()
    if valid.any():
        target_dates = signal_dates[valid]
        option_aligned = option_features.reindex(
            target_dates,
            method="ffill",
            tolerance=pd.Timedelta(days=5),
        )
        option_aligned.index = np.flatnonzero(valid)
        aligned = aligned.join(option_aligned, how="left")

    kospi = levels["KODEX200"].astype(float)
    kospi_features = pd.DataFrame(index=kospi.index)
    kospi_features["tracker_return_5"] = kospi.pct_change(5, fill_method=None)
    kospi_features["tracker_return_21"] = kospi.pct_change(21, fill_method=None)
    kospi_features["tracker_distance_high_21"] = (
        kospi / kospi.rolling(21, min_periods=10).max() - 1.0
    )
    if valid.any():
        tracker_aligned = kospi_features.reindex(
            signal_dates[valid],
            method="ffill",
            tolerance=pd.Timedelta(days=5),
        )
        tracker_aligned.index = np.flatnonzero(valid)
        for column in tracker_aligned:
            aligned.loc[tracker_aligned.index, column] = tracker_aligned[column]

    robust = align_features_to_arrays(build_robust_daily_features(), arrays)
    report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(encoding="utf-8")
    )
    robust_config = RobustStressConfig(**report["winner"])
    aligned["base_vkospi_stress"] = stress_from_features(
        robust,
        robust_config.mode,
        robust_config.level_threshold,
        robust_config.shock_threshold,
    )
    for column in (
        "percentile_252",
        "percentile_126",
        "shock_5",
        "acceleration_z5",
        "close_location21",
        "distance_high21",
    ):
        aligned[f"vk_{column}"] = robust[column].to_numpy(dtype=float)

    months = pd.PeriodIndex(arrays["months"], freq="M")
    macro = signals.reindex(months)
    aligned["p_growth_high"] = macro["p_growth_high"].to_numpy(dtype=float)
    aligned["p_inflation_high"] = macro["p_inflation_high"].to_numpy(dtype=float)
    aligned["option_available"] = aligned.get(
        "asymmetry", pd.Series(np.nan, index=aligned.index)
    ).notna()
    aligned["signal_date"] = signal_dates.to_numpy()
    return aligned.replace([np.inf, -np.inf], np.nan)


def _ramp(values: np.ndarray, threshold: float, width: float = 2.0) -> np.ndarray:
    return np.clip((values - threshold) / max(width, 1e-9), 0.0, 1.0)


def case1_stress(aligned: pd.DataFrame, config: Case1Config) -> pd.DataFrame:
    index = aligned.index

    def values(column: str, default: float = 0.0) -> np.ndarray:
        if column not in aligned:
            return np.full(len(aligned), default, dtype=float)
        return np.nan_to_num(
            aligned[column].to_numpy(dtype=float),
            nan=default,
            posinf=default,
            neginf=default,
        )

    base = np.clip(values("base_vkospi_stress"), 0, 1)
    asymmetry = values("asymmetry")
    left_impulse = values("left_impulse_z")
    tail_score = np.maximum(asymmetry, left_impulse)
    tail_warning = _ramp(tail_score, config.tail_threshold)
    breadth = _ramp(values("breadth_z"), config.breadth_threshold)
    reaction = _ramp(values("reaction_z"), 0.75)
    right_dominance = _ramp(-asymmetry, 0.50)

    forward_break = np.maximum(
        _ramp(-values("forward_return_5"), 0.01, 0.05),
        _ramp(-values("forward_return_21"), 0.02, 0.08),
    )
    tracker_break = np.maximum(
        _ramp(-values("tracker_return_5"), 0.01, 0.05),
        _ramp(-values("tracker_return_21"), 0.02, 0.08),
    )
    price_break = np.maximum(forward_break, tracker_break)

    early_warning = tail_warning * (1.0 - breadth) * (1.0 - base) * (1.0 - price_break)
    confirmed = tail_warning * np.maximum(breadth, base) * (0.35 + 0.65 * price_break)
    fading = (
        np.clip(-values("left_change_5") / 2.0, 0.0, 1.0)
        * np.maximum(breadth, base)
        * _ramp(values("left_tail"), 0.25)
    )
    panic = tail_warning * base * reaction
    recovery = np.clip(
        fading * (1.0 - np.clip(price_break - 0.50, 0.0, 0.50)), 0.0, 1.0
    )

    option_addition = np.maximum(
        config.early_scale * early_warning,
        config.confirmed_scale * confirmed,
    )
    stress = base + (1.0 - base) * option_addition
    stress *= 1.0 - config.right_tail_relief * right_dominance * (1.0 - price_break)
    stress *= 1.0 - config.recovery_relief * recovery
    stress *= 1.0 - config.panic_relief * panic * fading
    available = aligned["option_available"].to_numpy(dtype=bool)
    stress = np.where(available, stress, base)
    stress = np.nan_to_num(stress, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)

    state = np.full(len(aligned), "Calm", dtype=object)
    state[early_warning >= 0.35] = "EarlyWarning"
    state[confirmed >= 0.35] = "ConfirmedRiskOff"
    state[panic >= 0.40] = "Panic"
    state[recovery >= 0.35] = "Recovery"
    return pd.DataFrame(
        {
            "stress": stress,
            "base_vkospi_stress": base,
            "tail_warning": tail_warning,
            "breadth_confirmation": breadth,
            "price_confirmation": price_break,
            "early_warning": early_warning,
            "confirmed_risk_off": confirmed,
            "panic": panic,
            "recovery": recovery,
            "right_tail_relief": right_dominance,
            "state": state,
        },
        index=index,
    )


def simulate_case1(
    arrays: dict[str, object],
    aligned: pd.DataFrame,
    config: Case1Config,
    diagnostics: pd.DataFrame | None = None,
    keep_daily: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = case1_stress(aligned, config) if diagnostics is None else diagnostics
    dates = pd.DatetimeIndex(arrays["dates"])
    months = pd.PeriodIndex(arrays["months"], freq="M")
    returns = np.asarray(arrays["returns"], dtype=float)
    base_weights = np.asarray(arrays["base_weights"], dtype=float)
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])
    inflation = np.nan_to_num(
        aligned["p_inflation_high"].to_numpy(dtype=float), nan=0.5
    ).clip(0, 1)
    stress = diagnostics["stress"].to_numpy(dtype=float)
    diagnostic_arrays = {
        column: diagnostics[column].to_numpy()
        for column in (
            "state",
            "early_warning",
            "confirmed_risk_off",
            "panic",
            "recovery",
            "right_tail_relief",
        )
    }

    pretrade = np.zeros(len(ASSETS))
    previous_month: pd.Period | None = None
    nav = 1.0
    peak = 1.0
    first_trade = True
    rf_daily = (1 + config.financing_rate) ** (1 / 252) - 1
    rows: list[dict[str, object]] = []
    for position, date in enumerate(dates):
        month = months[position]
        base = base_weights[position].copy()
        severity = float(stress[position])
        transfer_fraction = config.max_risk_transfer * severity
        inflation_probability = float(inflation[position])
        oil_cut = (
            config.oil_cut_deflation * (1.0 - inflation_probability)
            + config.oil_cut_inflation * inflation_probability
        )
        bond_share = (
            config.bond_share_deflation * (1.0 - inflation_probability)
            + config.bond_share_inflation * inflation_probability
        )
        oil_cut = float(np.clip(oil_cut, 0, 1))
        bond_share = float(np.clip(bond_share, 0, 1))

        desired = base.copy()
        removed_equity = desired[0] * transfer_fraction
        removed_oil = desired[3] * transfer_fraction * oil_cut
        desired[0] -= removed_equity
        desired[3] -= removed_oil
        removed = removed_equity + removed_oil
        desired[1] += removed * bond_share
        desired[2] += removed * (1.0 - bond_share)

        month_boundary = previous_month is None or month != previous_month
        desired_turnover = 0.5 * float(np.abs(desired - pretrade).sum())
        rebalance = month_boundary or desired_turnover >= config.rebalance_band
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
                "stress": severity,
                "transfer_fraction": transfer_fraction,
                "oil_cut": oil_cut,
                "bond_share": bond_share,
                "state": diagnostic_arrays["state"][position],
                "signal_date": signal_dates[position],
                **{
                    column: float(diagnostic_arrays[column][position])
                    for column in (
                        "early_warning",
                        "confirmed_risk_off",
                        "panic",
                        "recovery",
                        "right_tail_relief",
                    )
                },
                **{f"w_{asset}": float(weights[i]) for i, asset in enumerate(ASSETS)},
            }
        )
        previous_month = month
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda x: float(np.prod(1 + x))),
        gross_factor=("gross_return", lambda x: float(np.prod(1 + x))),
        turnover=("turnover", "sum"),
        trade_cost=("trade_cost", "sum"),
        fx_cost=("fx_cost", "sum"),
        avg_stress=("stress", "mean"),
        max_stress=("stress", "max"),
        avg_transfer=("transfer_fraction", "mean"),
        avg_early_warning=("early_warning", "mean"),
        avg_confirmed_risk_off=("confirmed_risk_off", "mean"),
        avg_panic=("panic", "mean"),
        avg_recovery=("recovery", "mean"),
        avg_right_tail_relief=("right_tail_relief", "mean"),
    )
    monthly["return"] = monthly["return_factor"] - 1
    monthly["gross_return"] = monthly["gross_factor"] - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    return (daily if keep_daily else pd.DataFrame()), monthly


def _view(path: pd.DataFrame, start: pd.Period | None, end: pd.Period | None) -> pd.DataFrame:
    output = path
    if start is not None:
        output = output.loc[start:]
    if end is not None:
        output = output.loc[:end]
    return output


def _metrics(
    path: pd.DataFrame, start: pd.Period | None, end: pd.Period | None
) -> pd.Series:
    return performance_summary(_view(path, start, end)["return"])


def _metric_row(
    period: str,
    strategy: str,
    path: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, object]:
    view = _view(path, start, end)
    metrics = performance_summary(view["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **{key: float(value) for key, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean()),
        "TotalCost": float(view[["trade_cost", "fx_cost"]].sum().sum()),
    }


def _evaluate_config(
    config: Case1Config,
    arrays: dict[str, object],
    aligned: pd.DataFrame,
    neutral_monthly: pd.DataFrame,
    medium: pd.DataFrame,
    baseline_metrics: dict[str, pd.Series],
    stress_cache: dict[tuple[float, ...], pd.DataFrame],
    stage: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    stress_key = (
        config.tail_threshold,
        config.breadth_threshold,
        config.early_scale,
        config.confirmed_scale,
        config.recovery_relief,
        config.right_tail_relief,
        config.panic_relief,
    )
    diagnostics = stress_cache.setdefault(stress_key, case1_stress(aligned, config))
    _, monthly = simulate_case1(
        arrays, aligned, config, diagnostics=diagnostics, keep_daily=False
    )
    reconciled = reconcile_to_monthly_reference(medium, neutral_monthly, monthly)
    cal = _metrics(reconciled, None, CALIBRATION_END)
    validation = _metrics(reconciled, VALIDATION_START, CAL_END)
    row: dict[str, object] = {
        "Stage": stage,
        "Name": config.name,
        **asdict(config),
    }
    for prefix, metrics, baseline in (
        ("Cal", cal, baseline_metrics["calibration"]),
        ("Validation", validation, baseline_metrics["validation"]),
    ):
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
            row[f"{prefix}_{metric}"] = float(metrics[metric])
            row[f"{prefix}_{metric}Delta"] = float(metrics[metric] - baseline[metric])
    cal_score = (
        row["Cal_CAGRDelta"] / 0.01
        + row["Cal_SharpeDelta"] / 0.05
        + row["Cal_MDDDelta"] / 0.01
    )
    validation_score = (
        row["Validation_CAGRDelta"] / 0.01
        + row["Validation_SharpeDelta"] / 0.05
        + row["Validation_MDDDelta"] / 0.01
    )
    row["MultiObjectiveScore"] = float(
        min(cal_score, validation_score) + 0.25 * (cal_score + validation_score)
    )
    # MDD deltas at the 1e-16 scale are floating-point reconciliation noise,
    # not an economically meaningful deterioration.
    mdd_tolerance = -1e-12
    row["StrictAllThree"] = bool(
        row["Cal_CAGRDelta"] > 0
        and row["Cal_SharpeDelta"] > 0
        and row["Cal_MDDDelta"] >= mdd_tolerance
        and row["Validation_CAGRDelta"] > 0
        and row["Validation_SharpeDelta"] > 0
        and row["Validation_MDDDelta"] >= mdd_tolerance
    )
    row["RetentionGate"] = bool(
        row["Cal_CAGR"] >= 0.995 * baseline_metrics["calibration"]["CAGR"]
        and row["Cal_SharpeDelta"] > 0
        and row["Cal_MDDDelta"] >= mdd_tolerance
        and row["Validation_CAGR"] >= 0.995 * baseline_metrics["validation"]["CAGR"]
        and row["Validation_SharpeDelta"] > 0
        and row["Validation_MDDDelta"] >= mdd_tolerance
    )
    return row, reconciled


def calibrate_case1(
    arrays: dict[str, object],
    aligned: pd.DataFrame,
    neutral_monthly: pd.DataFrame,
    medium: pd.DataFrame,
    baseline: pd.DataFrame,
    quick: bool = False,
) -> tuple[Case1Config, pd.DataFrame, str]:
    baseline_metrics = {
        "calibration": _metrics(baseline, None, CALIBRATION_END),
        "validation": _metrics(baseline, VALIDATION_START, CAL_END),
    }
    stress_cache: dict[tuple[float, ...], pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    relief_configs = [
        Case1Config(
            tail_threshold=tail_threshold,
            breadth_threshold=0.25,
            early_scale=0.0,
            confirmed_scale=0.0,
            recovery_relief=recovery_relief,
            right_tail_relief=right_tail_relief,
            panic_relief=panic_relief,
        )
        for tail_threshold, recovery_relief, right_tail_relief, panic_relief in itertools.product(
            (0.50, 1.00, 1.50),
            (0.00, 0.25, 0.50, 0.75),
            (0.00, 0.25, 0.50, 0.75),
            (0.00, 0.25),
        )
    ]
    addition_configs = [
        Case1Config(
            tail_threshold=tail_threshold,
            breadth_threshold=breadth_threshold,
            early_scale=early_scale,
            confirmed_scale=confirmed_scale,
            recovery_relief=recovery_relief,
            right_tail_relief=right_tail_relief,
            panic_relief=0.25,
        )
        for tail_threshold, breadth_threshold, early_scale, confirmed_scale, recovery_relief, right_tail_relief in itertools.product(
            (0.50, 1.00, 1.50),
            (0.00, 0.50),
            (0.05, 0.10),
            (0.25, 0.50),
            (0.25, 0.50),
            (0.25, 0.50),
        )
    ]
    stress_configs = relief_configs + addition_configs
    if quick:
        stress_configs = relief_configs[::12] + addition_configs[::12]
    for config in stress_configs:
        row, _ = _evaluate_config(
            config,
            arrays,
            aligned,
            neutral_monthly,
            medium,
            baseline_metrics,
            stress_cache,
            "stress",
        )
        rows.append(row)

    first_stage = pd.DataFrame(rows)
    candidates = first_stage.loc[first_stage["RetentionGate"]].copy()
    if candidates.empty:
        candidates = first_stage.sort_values("MultiObjectiveScore", ascending=False).head(12)
    else:
        candidates = candidates.sort_values("MultiObjectiveScore", ascending=False).head(15)
    if quick:
        candidates = candidates.head(4)

    seen = {row["Name"] for row in rows}
    for _, candidate in candidates.iterrows():
        base = Case1Config(
            **{
                field: candidate[field]
                for field in Case1Config.__dataclass_fields__
            }
        )
        routing_items = list(ROUTING_PRESETS.items())
        transfers = (0.25, 0.35, 0.45)
        bands = (0.15, 0.20, 0.25)
        if quick:
            routing_items = routing_items[:3]
            transfers = (0.35,)
            bands = (0.20,)
        for _, routing in routing_items:
            for transfer, band in itertools.product(transfers, bands):
                config = replace(
                    base,
                    max_risk_transfer=transfer,
                    rebalance_band=band,
                    **routing,
                )
                if config.name in seen:
                    continue
                seen.add(config.name)
                row, _ = _evaluate_config(
                    config,
                    arrays,
                    aligned,
                    neutral_monthly,
                    medium,
                    baseline_metrics,
                    stress_cache,
                    "allocation",
                )
                rows.append(row)

    calibration = pd.DataFrame(rows)
    strict = calibration.loc[calibration["StrictAllThree"]]
    if not strict.empty:
        eligible = strict
        selection_rule = "CAGR, Sharpe and MDD all improve in 2007-2012 and 2013-2017"
    else:
        retention = calibration.loc[calibration["RetentionGate"]]
        if not retention.empty:
            eligible = retention
            selection_rule = "Sharpe and MDD improve with >=99.5% CAGR retention in both pre-lock windows"
        else:
            eligible = calibration
            selection_rule = "fallback to highest pre-lock multi-objective score"
    winner_row = eligible.sort_values(
        ["MultiObjectiveScore", "Validation_Sharpe", "Validation_CAGR", "Validation_MDD"],
        ascending=False,
    ).iloc[0]
    winner = Case1Config(
        **{field: winner_row[field] for field in Case1Config.__dataclass_fields__}
    )
    calibration["Selected"] = calibration["Name"].eq(winner.name)
    return winner, calibration, selection_rule


def run_experiment(force_features: bool = False, quick: bool = False) -> dict[str, object]:
    option_features = build_vix6_features(force=force_features)
    medium, signals = build_final_medium_reference()
    levels = load_daily_open_levels()
    arrays = prepare_arrays(levels, medium, build_daily_vkospi_signals())
    aligned = build_aligned_case1_inputs(arrays, levels, signals, option_features)
    _, neutral_monthly = simulate(arrays, None, keep_daily=False)
    baseline_daily, _, baseline = fixed_robust_overlay(medium)

    winner, calibration, selection_rule = calibrate_case1(
        arrays,
        aligned,
        neutral_monthly,
        medium,
        baseline,
        quick=quick,
    )
    winner_diagnostics = case1_stress(aligned, winner)
    winner_daily, winner_monthly = simulate_case1(
        arrays,
        aligned,
        winner,
        diagnostics=winner_diagnostics,
        keep_daily=True,
    )
    winner_reconciled = reconcile_to_monthly_reference(
        medium, neutral_monthly, winner_monthly
    )
    for column in (
        "avg_early_warning",
        "avg_confirmed_risk_off",
        "avg_panic",
        "avg_recovery",
        "avg_right_tail_relief",
    ):
        winner_reconciled[column] = winner_monthly.loc[winner_reconciled.index, column]

    periods = (
        ("calibration_2007_2012", None, CALIBRATION_END),
        ("validation_2013_2017", VALIDATION_START, CAL_END),
        ("prelock_2007_2017", None, CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", None, None),
    )
    comparison = pd.DataFrame(
        [
            _metric_row(period, strategy, path, start, end)
            for period, start, end in periods
            for strategy, path in (
                ("Existing_Final_RobustVKOSPI", baseline),
                ("VIX6_Case1", winner_reconciled),
            )
        ]
    )

    locked = comparison.loc[comparison["Period"].eq("locked_2018_2026")].set_index(
        "Strategy"
    )
    locked_delta = {
        metric: float(
            locked.loc["VIX6_Case1", metric]
            - locked.loc["Existing_Final_RobustVKOSPI", metric]
        )
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover")
    }
    common_locked = baseline.loc[TEST_START:].index.intersection(
        winner_reconciled.loc[TEST_START:].index
    )
    bootstrap = paired_multiobjective_bootstrap(
        baseline.loc[common_locked, "return"],
        winner_reconciled.loc[common_locked, "return"],
    )

    action_dates = pd.DatetimeIndex(arrays["dates"])
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])
    valid_signal = ~signal_dates.isna()
    strict_lag = bool((action_dates[valid_signal] > signal_dates[valid_signal]).all())
    feature_available = aligned["option_available"].astype(bool)
    residual = option_features["decomposition_residual"].dropna().abs()
    state_counts = winner_daily["state"].value_counts().to_dict()
    report: dict[str, object] = {
        "objective": (
            "Cboe-inspired VIX six-factor close-IV proxy used only as a Case 1 "
            "four-asset risk-budget overlay"
        ),
        "methodology": {
            "official_reference": "https://cdn.cboe.com/resources/vix/VIX-Decomposition-2025-08-01.pdf",
            "status": "close-IV proxy, not exact official VIX-point attribution",
            "reason": "KOSPI200 option bid/ask quotes and official contribution weights are unavailable",
            "target_tenor_days": TARGET_DTE,
            "forward": "median of the three strikes with the smallest absolute call-put close difference",
            "sticky_strike": "prior fixed-strike shoulder slope times the log forward move",
            "parallel_shift": "ATM IV change less sticky-strike expected move",
            "put_call_skew": "daily change in 15-45 delta shoulder IV relative to ATM",
            "up_down_convexity": "daily change in 1-15 delta tail IV beyond linear shoulder extrapolation",
            "left_tail": "mean robust z-score of put-skew and downside-convexity levels",
            "right_tail": "mean robust z-score of call-skew and upside-convexity levels",
            "asymmetry": "left-tail minus right-tail",
            "breadth": "robust z-score of five-day parallel-shift sum",
            "reaction": "robust z-score of five-day sticky-strike sum",
            "option_is_allocated_asset": False,
            "tradable_assets": list(ASSETS),
        },
        "winner": asdict(winner),
        "selection": {
            "rule": selection_rule,
            "candidate_count": int(len(calibration)),
            "strict_all_three_count": int(calibration["StrictAllThree"].sum()),
            "retention_gate_count": int(calibration["RetentionGate"].sum()),
            "uses_locked_period_for_selection": False,
            "calibration_window": "2007-2012",
            "validation_window": "2013-2017",
            "locked_window": "2018-2026",
            "development_status": (
                "Post-lock exploratory research requested by the user. Candidate "
                "ranking uses pre-2018 data, but locked results were observed during "
                "the broader model iteration."
            ),
        },
        "data": {
            "option_source_rows": 860380,
            "surface_days": int(len(option_features)),
            "surface_start": str(option_features.index.min().date()),
            "surface_end": str(option_features.index.max().date()),
            "aligned_action_days": int(len(aligned)),
            "option_available_action_days": int(feature_available.sum()),
            "option_coverage": float(feature_available.mean()),
            "day_session_only": True,
            "vkospi_used": True,
            "kospi200_confirmation": "KOSPI200 option-implied forward plus investable KODEX200 tracker trend",
        },
        "lookahead_audit": {
            "signal_strictly_before_action_open": strict_lag,
            "signal_observations": int(valid_signal.sum()),
            "maximum_decomposition_identity_residual": float(residual.max()),
            "selection_uses_locked_metrics": False,
            "locked_metrics_observed_during_development": True,
        },
        "locked_comparison": {
            "existing": {
                key: float(locked.loc["Existing_Final_RobustVKOSPI", key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover")
            },
            "vix6_case1": {
                key: float(locked.loc["VIX6_Case1", key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover")
            },
            "delta_vix6_minus_existing": locked_delta,
            "all_three_improve": bool(
                locked_delta["CAGR"] > 0
                and locked_delta["Sharpe"] > 0
                and locked_delta["MDD"] >= 0
            ),
            "bootstrap": bootstrap,
        },
        "state_counts": {str(key): int(value) for key, value in state_counts.items()},
        "comparison": json.loads(comparison.to_json(orient="records")),
    }

    calibration.to_csv(CALIBRATION_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    winner_daily.to_csv(DAILY_PATH)
    winner_monthly.to_csv(MONTHLY_PATH)
    winner_reconciled.to_csv(RECONCILED_PATH)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== VIX6 CASE 1 WINNER ===")
    print(json.dumps(report["winner"], ensure_ascii=False, indent=2))
    print("\n=== LOCKED 2018-2026 ===")
    print(json.dumps(report["locked_comparison"], ensure_ascii=False, indent=2))
    print("\n=== PERIOD COMPARISON ===")
    print(
        comparison[
            ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    features = build_vix6_features(force=args.force_features)
    if args.features_only:
        print(
            json.dumps(
                {
                    "rows": int(len(features)),
                    "start": str(features.index.min().date()),
                    "end": str(features.index.max().date()),
                    "complete_meta_signals": int(
                        features[
                            [
                                "left_tail",
                                "right_tail",
                                "asymmetry",
                                "breadth_z",
                                "reaction_z",
                            ]
                        ].dropna().shape[0]
                    ),
                },
                indent=2,
            )
        )
        return
    run_experiment(force_features=False, quick=args.quick)


if __name__ == "__main__":
    main()
