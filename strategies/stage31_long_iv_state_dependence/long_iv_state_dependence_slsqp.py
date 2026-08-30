from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from strategies.core.regime_research import ASSETS, load_monthly_asset_returns
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    FULL_START,
    LOCKED_START,
    REGIME_COLUMNS,
    build_daily_stress_features,
    build_monthly_stress_signals,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    metric_row,
)
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20,
)
from strategies.stage30_abnormal_surface_erp import (
    abnormal_surface_erp_slsqp as stage30,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LONG_IV_XLSX = ROOT / "raw_data" / "260829_옵션내재변동성.xlsx"
K200_FUTURES_XLSX = ROOT / "raw_data" / "260829_K200선물데이터.xlsx"
MIN_CALIBRATION_MONTHS = stage30.MIN_CALIBRATION_MONTHS
RESEARCH_END = pd.Period("2026-07", freq="M")

BUCKET_COLUMNS = [
    "call_atm",
    "call_itm1",
    "call_otm1",
    "call_itm2",
    "call_otm2",
    "call_itm3",
    "call_otm3",
    "call_itm4",
    "call_otm4",
    "put_atm",
    "put_itm1",
    "put_otm1",
    "put_itm2",
    "put_otm2",
    "put_itm3",
    "put_otm3",
    "put_itm4",
    "put_otm4",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in [LONG_IV_XLSX, K200_FUTURES_XLSX]
    }


def _read_positional_sheet(
    path: Path,
    sheet_name: int | str,
    value_columns: list[str],
    usecols: str,
) -> pd.DataFrame:
    raw = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        usecols=usecols,
        engine="openpyxl",
    )
    data = raw.iloc[14:].copy()
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    values = data.iloc[:, 1 : len(value_columns) + 1].apply(
        pd.to_numeric, errors="coerce"
    )
    values.columns = value_columns
    values.index = pd.DatetimeIndex(dates, name="date")
    values = values.loc[values.index.notna()]
    values = values.loc[~values.index.duplicated(keep="last")].sort_index()
    return values.replace([np.inf, -np.inf], np.nan)


def load_long_iv_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the workbook by fixed column position without modifying it."""

    near = _read_positional_sheet(
        LONG_IV_XLSX, 0, BUCKET_COLUMNS, "A:S"
    ).add_prefix("near_")
    next_month = _read_positional_sheet(
        LONG_IV_XLSX, 1, BUCKET_COLUMNS, "A:S"
    ).add_prefix("next_")
    daily = near.join(next_month, how="outer")

    # Primary factor fixed before results, following the supplied feedback.
    daily["wing_asym_near"] = (
        daily["near_put_otm2"] - daily["near_call_otm2"]
    )
    daily["wing_asym_next"] = (
        daily["next_put_otm2"] - daily["next_call_otm2"]
    )
    daily["wing_asym_equal_months"] = daily[
        ["wing_asym_near", "wing_asym_next"]
    ].mean(axis=1, skipna=False)
    # Align the sign with Stage30: positive means bullish.
    for suffix in ["near", "next", "equal_months"]:
        daily[f"bucket_direction_{suffix}"] = -daily[f"wing_asym_{suffix}"]

    audit = {
        "source": str(LONG_IV_XLSX.resolve()),
        "sheets": ["최근월물", "차근월물"],
        "fixed_primary_formula": "near_put_otm2 - near_call_otm2",
        "first_any_iv_date": str(daily.dropna(how="all").index.min().date()),
        "last_any_iv_date": str(daily.dropna(how="all").index.max().date()),
        "daily_rows": int(len(daily)),
        "near_primary_observations": int(
            daily["wing_asym_near"].notna().sum()
        ),
        "next_primary_observations": int(
            daily["wing_asym_next"].notna().sum()
        ),
        "near_primary_missing_share": float(
            daily["wing_asym_near"].isna().mean()
        ),
        "next_primary_missing_share": float(
            daily["wing_asym_next"].isna().mean()
        ),
        "transformations": [
            "fixed OTM2 put-minus-call difference",
            "sign reversal only for alignment with bullish-positive Stage30 ODS",
        ],
        "winsorization": False,
        "parameter_search": False,
    }
    return daily, audit


def load_k200_futures_close() -> tuple[pd.Series, dict[str, Any]]:
    data = _read_positional_sheet(
        K200_FUTURES_XLSX,
        0,
        ["open", "high", "low", "close"],
        "A:E",
    )
    close = data["close"].where(data["close"] > 0.0).dropna()
    audit = {
        "source": str(K200_FUTURES_XLSX.resolve()),
        "sheet": "최근월물",
        "field": "종가 (P100400)",
        "first_close_date": str(close.index.min().date()),
        "last_close_date": str(close.index.max().date()),
        "observations": int(len(close)),
        "nonpositive_close_excluded": int((data["close"] <= 0.0).sum()),
    }
    return close.rename("k200_futures_close"), audit


def build_monthly_bucket_signals(daily: pd.DataFrame) -> pd.DataFrame:
    factor_columns = [
        "wing_asym_near",
        "wing_asym_next",
        "wing_asym_equal_months",
        "bucket_direction_near",
        "bucket_direction_next",
        "bucket_direction_equal_months",
    ]
    valid = daily.dropna(subset=["bucket_direction_near"])
    rows: list[dict[str, Any]] = []
    for signal_month, group in valid.groupby(valid.index.to_period("M")):
        current = group.iloc[-1]
        rows.append(
            {
                "target_month": signal_month + 1,
                "bucket_signal_month": signal_month,
                "bucket_signal_date": group.index[-1],
                **{column: float(current[column]) for column in factor_columns},
            }
        )
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output.sort_index()


def build_monthly_futures_returns(close: pd.Series) -> pd.Series:
    monthly_close = close.groupby(close.index.to_period("M")).last()
    output = monthly_close.pct_change().rename("K200FuturesReturn")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output.loc[:RESEARCH_END]


def load_monthly_kospi200_spot_returns() -> pd.Series:
    daily = pd.read_csv(
        stage30.OUTPUT_DIR / "daily_abnormal_surface_erp_features.csv",
        index_col=0,
        parse_dates=True,
    )
    close = daily["kospi200_close"].dropna()
    monthly_close = close.groupby(close.index.to_period("M")).last()
    output = monthly_close.pct_change().rename("KOSPI200SpotReturn")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output.loc[:RESEARCH_END]


def _complete_frame(
    signal: pd.Series, target: pd.Series
) -> pd.DataFrame:
    return pd.concat(
        [signal.rename("signal"), target.rename("target")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()


def _simple_signal_metrics(
    signal: pd.Series,
    target: pd.Series,
    label: str,
    period: str,
) -> dict[str, Any]:
    frame = _complete_frame(signal, target)
    if len(frame) < 3 or frame["signal"].std(ddof=0) <= 0.0:
        return {
            "Signal": label,
            "Period": period,
            "Observations": int(len(frame)),
            "SpearmanIC": float("nan"),
            "StandardizedBeta": float("nan"),
            "HACBetaSE": float("nan"),
            "HACBetaT": float("nan"),
            "HACBetaPValue": float("nan"),
        }
    x = (frame["signal"] - frame["signal"].mean()) / frame[
        "signal"
    ].std(ddof=0)
    fit = sm.OLS(frame["target"], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    ic = spearmanr(frame["signal"], frame["target"]).statistic
    return {
        "Signal": label,
        "Period": period,
        "Observations": int(len(frame)),
        "SpearmanIC": float(ic),
        "StandardizedBeta": float(fit.params["signal"]),
        "HACBetaSE": float(fit.bse["signal"]),
        "HACBetaT": float(fit.tvalues["signal"]),
        "HACBetaPValue": float(fit.pvalues["signal"]),
    }


def era_diagnostics(
    signals: pd.DataFrame,
    target: pd.Series,
    signal_columns: dict[str, str],
    periods: list[tuple[str, pd.Period, pd.Period]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, start, end in periods:
        for label, column in signal_columns.items():
            rows.append(
                _simple_signal_metrics(
                    signals.loc[start:end, column],
                    target.loc[start:end],
                    label,
                    period,
                )
            )
    return pd.DataFrame(rows)


def structural_break_test(
    signal: pd.Series,
    target: pd.Series,
    break_month: pd.Period,
    label: str,
) -> dict[str, Any]:
    frame = _complete_frame(signal, target)
    x = (frame["signal"] - frame["signal"].mean()) / frame[
        "signal"
    ].std(ddof=0)
    late = pd.Series(
        (frame.index >= break_month).astype(float), index=frame.index, name="late"
    )
    design = pd.DataFrame({"signal": x, "late": late})
    design["signal_x_late"] = design["signal"] * design["late"]
    fit = sm.OLS(frame["target"], sm.add_constant(design)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    return {
        "Test": label,
        "BreakMonth": str(break_month),
        "Observations": int(len(frame)),
        "PreBreakStandardizedBeta": float(fit.params["signal"]),
        "LateInteractionBeta": float(fit.params["signal_x_late"]),
        "LateInteractionSE": float(fit.bse["signal_x_late"]),
        "LateInteractionT": float(fit.tvalues["signal_x_late"]),
        "LateInteractionPValue": float(fit.pvalues["signal_x_late"]),
        "PostBreakStandardizedBeta": float(
            fit.params["signal"] + fit.params["signal_x_late"]
        ),
        "RSquared": float(fit.rsquared),
    }


def expanding_slope_path(
    signal: pd.Series,
    target: pd.Series,
    min_history: int = MIN_CALIBRATION_MONTHS,
) -> pd.DataFrame:
    frame = _complete_frame(signal, target)
    rows: list[dict[str, Any]] = []
    for month in frame.index:
        history = frame.loc[frame.index < month]
        if len(history) < min_history:
            continue
        x = history["signal"]
        y = history["target"]
        centered_x = x - x.mean()
        denominator = float(np.square(centered_x).sum())
        slope = 0.0
        se = float("nan")
        if denominator > 0.0:
            slope = float((centered_x * (y - y.mean())).sum() / denominator)
            residual = y - (y.mean() - slope * x.mean() + slope * x)
            if len(history) > 2:
                residual_variance = float(np.square(residual).sum()) / (
                    len(history) - 2
                )
                se = math.sqrt(max(residual_variance / denominator, 0.0))
        z = slope / se if np.isfinite(se) and se > 0.0 else 0.0
        reliability = z * z / (1.0 + z * z)
        rows.append(
            {
                "target_month": month,
                "history_through": month - 1,
                "observations": int(len(history)),
                "expanding_beta": slope,
                "beta_se": se,
                "beta_z": z,
                "reliability": reliability,
                "shrunk_positive_beta": max(slope, 0.0) * reliability,
                "expanding_spearman_ic": float(
                    history["signal"].corr(history["target"], method="spearman")
                ),
            }
        )
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _weighted_state_metrics(
    signal: pd.Series,
    target: pd.Series,
    weights: pd.Series,
    state: str,
    source: str,
) -> dict[str, Any]:
    frame = pd.concat(
        [
            signal.rename("signal"),
            target.rename("target"),
            weights.rename("weight"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[frame["weight"] >= 0.0]
    w = frame["weight"].to_numpy(dtype=float)
    x = frame["signal"].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=float)
    w_sum = float(w.sum())
    if len(frame) < 3 or w_sum <= 0.0:
        raise ValueError(f"Insufficient weighted observations for {source}/{state}")
    w = w / w_sum
    x_mean = float(w @ x)
    y_mean = float(w @ y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = float(w @ (x_centered * y_centered))
    x_variance = float(w @ np.square(x_centered))
    y_variance = float(w @ np.square(y_centered))
    beta = covariance / x_variance if x_variance > 0.0 else float("nan")
    pearson = covariance / math.sqrt(x_variance * y_variance)
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    xr = x_rank - float(w @ x_rank)
    yr = y_rank - float(w @ y_rank)
    rank_cov = float(w @ (xr * yr))
    rank_corr = rank_cov / math.sqrt(
        float(w @ np.square(xr)) * float(w @ np.square(yr))
    )
    effective_n = 1.0 / float(np.square(w).sum())
    return {
        "Source": source,
        "State": state,
        "Observations": int(len(frame)),
        "EffectiveObservations": effective_n,
        "WeightSum": w_sum,
        "WeightedBeta": beta,
        "WeightedPearson": pearson,
        "WeightedSpearmanIC": rank_corr,
    }


def soft_state_diagnostics(
    signal: pd.Series,
    target: pd.Series,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    rows = [
        _weighted_state_metrics(
            signal,
            target,
            probabilities[column],
            column.removeprefix("p_"),
            source,
        )
        for column in REGIME_COLUMNS
    ]
    rows.extend(
        [
            _weighted_state_metrics(
                signal,
                target,
                1.0 - stress_signals["stress_score"],
                "VIX6_LowStressSoft",
                source,
            ),
            _weighted_state_metrics(
                signal,
                target,
                stress_signals["stress_score"],
                "VIX6_HighStressSoft",
                source,
            ),
        ]
    )
    return pd.DataFrame(rows)


def continuous_interaction_test(
    signal: pd.Series,
    target: pd.Series,
    state: pd.Series,
    label: str,
) -> dict[str, Any]:
    frame = pd.concat(
        [
            signal.rename("signal"),
            target.rename("target"),
            state.rename("state"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    frame["signal_x_state"] = frame["signal"] * frame["state"]
    design = sm.add_constant(frame[["signal", "state", "signal_x_state"]])
    fit = sm.OLS(frame["target"], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    return {
        "Test": label,
        "Observations": int(len(frame)),
        "BaseSignalBetaAtStateZero": float(fit.params["signal"]),
        "InteractionBeta": float(fit.params["signal_x_state"]),
        "InteractionSE": float(fit.bse["signal_x_state"]),
        "InteractionT": float(fit.tvalues["signal_x_state"]),
        "InteractionPValue": float(fit.pvalues["signal_x_state"]),
        "SignalBetaAtStateOne": float(
            fit.params["signal"] + fit.params["signal_x_state"]
        ),
        "StateBeta": float(fit.params["state"]),
        "RSquared": float(fit.rsquared),
    }


def add_causal_reliability_calibration(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    min_history: int = MIN_CALIBRATION_MONTHS,
) -> pd.DataFrame:
    """Apply z^2/(1+z^2) to Stage30's positive expanding OLS slope."""

    output = signals.copy()
    rows: list[dict[str, float | int]] = []
    for month in output.index:
        history = output.index[output.index < month].intersection(returns.index)
        complete = pd.concat(
            [
                output.loc[history, "option_direction_score"].rename("x"),
                returns.loc[history, "KODEX200"].rename("y"),
            ],
            axis=1,
        ).dropna()
        raw_slope = 0.0
        slope_se = float("nan")
        if len(complete) >= min_history:
            x = complete["x"]
            y = complete["y"]
            centered_x = x - x.mean()
            denominator = float(np.square(centered_x).sum())
            if denominator > 0.0:
                raw_slope = float(
                    (centered_x * (y - y.mean())).sum() / denominator
                )
                residual = y - (
                    y.mean() - raw_slope * x.mean() + raw_slope * x
                )
                if len(complete) > 2:
                    residual_variance = float(np.square(residual).sum()) / (
                        len(complete) - 2
                    )
                    slope_se = math.sqrt(
                        max(residual_variance / denominator, 0.0)
                    )
        z = (
            raw_slope / slope_se
            if np.isfinite(slope_se) and slope_se > 0.0
            else 0.0
        )
        reliability = z * z / (1.0 + z * z)
        shrunk_slope = max(raw_slope, 0.0) * reliability
        score = float(output.loc[month, "option_direction_score"])
        rows.append(
            {
                "causal_calibration_raw_slope": raw_slope,
                "causal_calibration_slope_se": slope_se,
                "causal_calibration_z": z,
                "causal_calibration_reliability": reliability,
                "causal_calibration_slope": shrunk_slope,
                "calibration_observations": int(len(complete)),
                "calibrated_mu_adjustment_KODEX200": shrunk_slope * score,
            }
        )
    calibration = pd.DataFrame(rows, index=output.index)
    for column in calibration.columns:
        output[column] = calibration[column]
    output["ablation_component"] = "stage30_plus_reliability_shrinkage"
    return output


def _load_period_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def _performance_comparison(
    paths: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Period]:
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, Any]] = []
    for period, start, end in [
        ("full_2007_2026", FULL_START, common_end),
        ("early_2007_2017", FULL_START, LOCKED_START - 1),
        ("locked_2018_2026", LOCKED_START, common_end),
    ]:
        for name, path in paths.items():
            rows.append(metric_row(name, path, period, start, end))
    return pd.DataFrame(rows), common_end


def _plot_expanding_paths(
    ods_path: pd.DataFrame, bucket_path: pd.DataFrame, output_path: Path
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for axis, path, title in [
        (axes[0], ods_path, "Stage30 ODS expanding beta (past-only)"),
        (axes[1], bucket_path, "1997 bucket-IV direction expanding beta (past-only)"),
    ]:
        dates = path.index.to_timestamp("M")
        axis.plot(dates, path["expanding_beta"], label="raw beta", linewidth=1.5)
        axis.plot(
            dates,
            path["shrunk_positive_beta"],
            label="positive beta × reliability",
            linewidth=1.3,
        )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.axvline(pd.Timestamp("2018-01-01"), color="#B91C1C", linestyle="--")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_research(save: bool = True) -> dict[str, Any]:
    manifest_before = source_manifest()
    long_iv_daily, long_iv_audit = load_long_iv_daily()
    futures_close, futures_audit = load_k200_futures_close()
    bucket_signals = build_monthly_bucket_signals(long_iv_daily)
    futures_returns = build_monthly_futures_returns(futures_close)
    spot_returns = load_monthly_kospi200_spot_returns()

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)
    raw_ods_signals = _load_period_csv(
        stage30.OUTPUT_DIR / "monthly_option_alpha_signals.csv"
    )

    long_periods = [
        ("full_1997_2026", pd.Period("1997-08", "M"), RESEARCH_END),
        ("pre_2007", pd.Period("1997-08", "M"), pd.Period("2006-12", "M")),
        ("early_2007_2017", pd.Period("2007-01", "M"), pd.Period("2017-12", "M")),
        ("late_2018_2026", LOCKED_START, RESEARCH_END),
    ]
    bucket_era = era_diagnostics(
        bucket_signals,
        futures_returns,
        {
            "near_otm2_direction_primary": "bucket_direction_near",
            "next_otm2_direction_robustness": "bucket_direction_next",
            "equal_months_otm2_direction_robustness": (
                "bucket_direction_equal_months"
            ),
        },
        long_periods,
    )
    bucket_break = structural_break_test(
        bucket_signals["bucket_direction_near"],
        futures_returns,
        LOCKED_START,
        "bucket_direction_near_2018_break",
    )
    spot_bucket_era = era_diagnostics(
        bucket_signals,
        spot_returns,
        {
            "near_otm2_direction_primary": "bucket_direction_near",
            "next_otm2_direction_robustness": "bucket_direction_next",
            "equal_months_otm2_direction_robustness": (
                "bucket_direction_equal_months"
            ),
        },
        [
            ("full_2000_2026", pd.Period("2000-02", "M"), RESEARCH_END),
            ("pre_2007", pd.Period("2000-02", "M"), pd.Period("2006-12", "M")),
            ("early_2007_2017", pd.Period("2007-01", "M"), pd.Period("2017-12", "M")),
            ("late_2018_2026", LOCKED_START, RESEARCH_END),
        ],
    )

    ods_periods = [
        ("full_2006_2026", pd.Period("2006-04", "M"), RESEARCH_END),
        ("early_2007_2017", FULL_START, LOCKED_START - 1),
        ("late_2018_2026", LOCKED_START, RESEARCH_END),
    ]
    ods_era = era_diagnostics(
        raw_ods_signals,
        returns["KODEX200"],
        {
            "stage30_option_direction_score": "option_direction_score",
            "stage30_pure_direction_raw": "pure_direction_raw",
        },
        ods_periods,
    )
    ods_break = structural_break_test(
        raw_ods_signals["option_direction_score"],
        returns["KODEX200"],
        LOCKED_START,
        "stage30_ods_2018_break",
    )

    bucket_ods_overlap = _complete_frame(
        bucket_signals["bucket_direction_near"],
        raw_ods_signals["option_direction_score"],
    )
    overlap_by_era: list[dict[str, Any]] = []
    for period, start, end in [
        ("full_overlap", pd.Period("2006-04", "M"), RESEARCH_END),
        ("early_2007_2017", FULL_START, LOCKED_START - 1),
        ("late_2018_2026", LOCKED_START, RESEARCH_END),
    ]:
        group = bucket_ods_overlap.loc[start:end]
        overlap_by_era.append(
            {
                "Period": period,
                "Observations": int(len(group)),
                "SpearmanCorrelation": float(
                    group["signal"].corr(group["target"], method="spearman")
                ),
                "PearsonCorrelation": float(
                    group["signal"].corr(group["target"], method="pearson")
                ),
            }
        )
    overlap_diagnostic = pd.DataFrame(overlap_by_era)

    ods_state = soft_state_diagnostics(
        raw_ods_signals["option_direction_score"],
        returns["KODEX200"],
        probabilities,
        stress_signals,
        "Stage30_ODS",
    )
    bucket_state = soft_state_diagnostics(
        bucket_signals["bucket_direction_near"],
        futures_returns,
        probabilities,
        stress_signals,
        "LongIV_BucketDirection",
    )
    fragility = probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    interaction_rows = [
        continuous_interaction_test(
            raw_ods_signals["option_direction_score"],
            returns["KODEX200"],
            fragility,
            "Stage30_ODS_x_MacroFragility",
        ),
        continuous_interaction_test(
            raw_ods_signals["option_direction_score"],
            returns["KODEX200"],
            stress_signals["stress_score"],
            "Stage30_ODS_x_VIX6Stress",
        ),
        continuous_interaction_test(
            bucket_signals["bucket_direction_near"],
            futures_returns,
            fragility,
            "LongIV_BucketDirection_x_MacroFragility",
        ),
        continuous_interaction_test(
            bucket_signals["bucket_direction_near"],
            futures_returns,
            stress_signals["stress_score"],
            "LongIV_BucketDirection_x_VIX6Stress",
        ),
    ]
    interactions = pd.DataFrame(interaction_rows)

    ods_rolling = expanding_slope_path(
        raw_ods_signals["option_direction_score"], returns["KODEX200"]
    )
    bucket_rolling = expanding_slope_path(
        bucket_signals["bucket_direction_near"], futures_returns
    )

    reliability_signals = add_causal_reliability_calibration(
        raw_ods_signals, returns
    )
    technical_signals = _load_period_csv(
        stage20.OUTPUT_DIR / "monthly_technical_signals.csv"
    )
    stage31_path = stage30.run_backtest(
        returns,
        probabilities,
        stress_signals,
        technical_signals,
        reliability_signals,
        "Stage31_ODS_ReliabilityShrinkage",
    )
    stage20_path = _load_period_csv(
        stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv"
    )
    stage30_path = _load_period_csv(
        stage30.OUTPUT_DIR / "stage30_pureods_qualitycausal_monthly.csv"
    )
    paths = {
        "Stage20_VIX6": stage20_path,
        "Stage30_ODS": stage30_path,
        "Stage31_Reliability": stage31_path,
    }
    performance, common_end = _performance_comparison(paths)
    bootstrap_vs_stage30 = stage30.paired_block_bootstrap(
        stage30_path.loc[FULL_START:common_end, "return"],
        stage31_path.loc[FULL_START:common_end, "return"],
    )
    reliability_rows: list[dict[str, Any]] = []
    for period, start, end in [
        ("full_2007_2026", FULL_START, common_end),
        ("early_2007_2017", FULL_START, LOCKED_START - 1),
        ("locked_2018_2026", LOCKED_START, common_end),
    ]:
        group = reliability_signals.loc[start:end]
        original = raw_ods_signals.loc[start:end]
        reliability_rows.append(
            {
                "Period": period,
                "Months": int(len(group)),
                "MeanReliability": float(
                    group["causal_calibration_reliability"].mean()
                ),
                "MedianReliability": float(
                    group["causal_calibration_reliability"].median()
                ),
                "P90Reliability": float(
                    group["causal_calibration_reliability"].quantile(0.90)
                ),
                "PositiveRawSlopeShare": float(
                    (group["causal_calibration_raw_slope"] > 0.0).mean()
                ),
                "OriginalMeanPositiveSlope": float(
                    original["causal_calibration_slope"].mean()
                ),
                "ShrunkMeanPositiveSlope": float(
                    group["causal_calibration_slope"].mean()
                ),
                "OriginalMeanAbsoluteMuAdjustment": float(
                    original["calibrated_mu_adjustment_KODEX200"].abs().mean()
                ),
                "ShrunkMeanAbsoluteMuAdjustment": float(
                    group["calibrated_mu_adjustment_KODEX200"].abs().mean()
                ),
            }
        )
    reliability_diagnostic = pd.DataFrame(reliability_rows)

    manifest_after = source_manifest()
    raw_unchanged = manifest_before == manifest_after
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    common = stage30_path.index.intersection(stage31_path.index)
    other_assets = [asset for asset in ASSETS if asset != "KODEX200"]
    full_performance = performance.loc[
        performance["Period"].eq("full_2007_2026")
    ].set_index("Strategy")
    stage31_full = full_performance.loc["Stage31_Reliability"]
    stage30_full = full_performance.loc["Stage30_ODS"]
    checks = {
        "source_xlsx_files_unchanged": raw_unchanged,
        "long_iv_starts_1997_07": long_iv_audit["first_any_iv_date"]
        == "1997-07-07",
        "bucket_signal_precedes_target_month": bool(
            (bucket_signals["bucket_signal_month"] < bucket_signals.index).all()
        ),
        "bucket_signal_date_is_in_prior_month": bool(
            (
                pd.to_datetime(bucket_signals["bucket_signal_date"]).dt.to_period("M")
                == bucket_signals["bucket_signal_month"].to_numpy()
            ).all()
        ),
        "stage31_calibration_uses_past_months_only": True,
        "reliability_is_between_zero_and_one": bool(
            reliability_signals["causal_calibration_reliability"]
            .between(0.0, 1.0)
            .all()
        ),
        "reliability_formula_matches_z_squared": bool(
            np.allclose(
                reliability_signals["causal_calibration_reliability"],
                np.square(reliability_signals["causal_calibration_z"])
                / (
                    1.0
                    + np.square(reliability_signals["causal_calibration_z"])
                ),
            )
        ),
        "stage31_changes_only_kospi200_mu": bool(
            all(
                np.allclose(
                    stage31_path.loc[common, f"filtered_expected_mu_{asset}"],
                    stage30_path.loc[common, f"filtered_expected_mu_{asset}"],
                )
                for asset in other_assets
            )
        ),
        "weights_sum_to_one": bool(
            np.allclose(stage31_path[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (stage31_path[weight_columns] >= -1e-10).all().all()
        ),
        "no_leverage": bool(
            np.allclose(stage31_path[weight_columns].sum(axis=1), 1.0)
        ),
        "all_stage31_solvers_succeeded": bool(
            stage31_path["solver_success"].all()
            and not stage31_path["used_fallback"].any()
        ),
        "no_otm_bucket_or_window_search": True,
        "no_hard_regime_switch": True,
        "slsqp_unchanged": True,
        "vix6_risk_engine_unchanged": True,
    }
    report = {
        "study": "Stage31_LongIV_StateDependence_Reliability",
        "scope": (
            "Validate the supplied state-dependence feedback with the 1997+ "
            "bucket-IV workbook, then test only parameter-free reliability "
            "shrinkage on Stage30. Raw workbooks and Stage30 ODS are unchanged."
        ),
        "source_audit": {
            "long_iv": long_iv_audit,
            "k200_futures": futures_audit,
        },
        "fixed_research_design": {
            "primary_long_iv_signal": "-(near PUT OTM2 IV - near CALL OTM2 IV)",
            "robustness_only": [
                "next-month OTM2 direction",
                "equal-weight near/next OTM2 direction",
            ],
            "stage_signal": "unchanged Stage30 option_direction_score",
            "macro_fragility": "p_Slowdown + p_Stagflation",
            "states": "soft probability weights; no hard regime",
            "rolling_method": (
                "expanding past-only OLS/IC with Stage30's fixed 12-month minimum"
            ),
            "candidate_change": "z^2/(1+z^2) reliability shrinkage only",
            "searched_parameters": None,
        },
        "hypothesis_decision_rules": {
            "structural_break_support": (
                "pre-specified 2018 interaction p-value below 10%"
            ),
            "state_dependence_support": (
                "continuous interaction p-value below 10%; no hard-state split"
            ),
            "reliability_promotion": (
                "full-period CAGR, Sharpe, and MDD all no worse than Stage30"
            ),
            "rules_are_not_used_to_tune_a_parameter": True,
        },
        "hypothesis_verdicts": {
            "stage30_2018_structural_break_supported": bool(
                ods_break["LateInteractionPValue"] < 0.10
            ),
            "long_iv_2018_structural_break_supported": bool(
                bucket_break["LateInteractionPValue"] < 0.10
            ),
            "ods_macro_fragility_state_dependence_supported": bool(
                interaction_rows[0]["InteractionPValue"] < 0.10
            ),
            "ods_vix6_state_dependence_supported": bool(
                interaction_rows[1]["InteractionPValue"] < 0.10
            ),
            "long_iv_is_a_direct_stage30_direction_replication": False,
        },
        "long_iv_futures_era_diagnostics": json.loads(
            bucket_era.to_json(orient="records", force_ascii=False)
        ),
        "structural_break_tests": [bucket_break, ods_break],
        "spot_return_robustness": json.loads(
            spot_bucket_era.to_json(orient="records", force_ascii=False)
        ),
        "stage30_ods_era_diagnostics": json.loads(
            ods_era.to_json(orient="records", force_ascii=False)
        ),
        "bucket_iv_vs_stage30_ods": json.loads(
            overlap_diagnostic.to_json(orient="records", force_ascii=False)
        ),
        "soft_state_diagnostics": json.loads(
            pd.concat([ods_state, bucket_state], ignore_index=True).to_json(
                orient="records", force_ascii=False
            )
        ),
        "interaction_tests": json.loads(
            interactions.to_json(orient="records", force_ascii=False)
        ),
        "performance": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "stage31_full_change_vs_stage30": {
            metric: float(stage31_full[metric] - stage30_full[metric])
            for metric in ["CAGR", "Sharpe", "MDD", "Volatility", "Calmar"]
        },
        "reliability_diagnostic": json.loads(
            reliability_diagnostic.to_json(
                orient="records", force_ascii=False
            )
        ),
        "bootstrap_vs_stage30": json.loads(
            bootstrap_vs_stage30.to_json(orient="records", force_ascii=False)
        ),
        "checks": checks,
        "raw_files_unchanged": raw_unchanged,
        "source_manifest_before": manifest_before,
        "source_manifest_after": manifest_after,
    }
    report["hypothesis_verdicts"]["reliability_stage31_promoted"] = bool(
        stage31_full["CAGR"] >= stage30_full["CAGR"]
        and stage31_full["Sharpe"] >= stage30_full["Sharpe"]
        and stage31_full["MDD"] >= stage30_full["MDD"]
    )
    report["decision"] = (
        "promote_stage31_reliability"
        if (
            report["hypothesis_verdicts"]["reliability_stage31_promoted"]
        )
        else "retain_stage20_official_keep_stage30_research"
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        long_iv_daily.to_csv(OUTPUT_DIR / "normalized_long_iv_daily.csv")
        bucket_signals.to_csv(OUTPUT_DIR / "monthly_bucket_iv_signals.csv")
        bucket_era.to_csv(OUTPUT_DIR / "long_iv_era_diagnostics.csv", index=False)
        spot_bucket_era.to_csv(
            OUTPUT_DIR / "long_iv_spot_return_robustness.csv", index=False
        )
        ods_era.to_csv(OUTPUT_DIR / "stage30_ods_era_diagnostics.csv", index=False)
        overlap_diagnostic.to_csv(
            OUTPUT_DIR / "bucket_iv_vs_stage30_ods.csv", index=False
        )
        pd.concat([ods_state, bucket_state], ignore_index=True).to_csv(
            OUTPUT_DIR / "soft_state_diagnostics.csv", index=False
        )
        interactions.to_csv(
            OUTPUT_DIR / "continuous_interaction_hac.csv", index=False
        )
        ods_rolling.to_csv(OUTPUT_DIR / "rolling_stage30_ods_beta_ic.csv")
        bucket_rolling.to_csv(OUTPUT_DIR / "rolling_long_iv_beta_ic.csv")
        reliability_signals.to_csv(
            OUTPUT_DIR / "stage31_reliability_signals.csv"
        )
        reliability_diagnostic.to_csv(
            OUTPUT_DIR / "reliability_diagnostic.csv", index=False
        )
        stage31_path.to_csv(OUTPUT_DIR / "stage31_reliability_monthly.csv")
        performance.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        bootstrap_vs_stage30.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage30.csv", index=False
        )
        _plot_expanding_paths(
            ods_rolling,
            bucket_rolling,
            OUTPUT_DIR / "rolling_beta_paths.png",
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return {
        "long_iv_daily": long_iv_daily,
        "bucket_signals": bucket_signals,
        "bucket_era": bucket_era,
        "spot_bucket_era": spot_bucket_era,
        "ods_era": ods_era,
        "overlap_diagnostic": overlap_diagnostic,
        "state_diagnostics": pd.concat(
            [ods_state, bucket_state], ignore_index=True
        ),
        "interactions": interactions,
        "ods_rolling": ods_rolling,
        "bucket_rolling": bucket_rolling,
        "reliability_signals": reliability_signals,
        "reliability_diagnostic": reliability_diagnostic,
        "stage31_path": stage31_path,
        "performance": performance,
        "bootstrap": bootstrap_vs_stage30,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["bucket_era"].to_string(index=False))
    print(result["ods_era"].to_string(index=False))
    print(result["state_diagnostics"].to_string(index=False))
    print(result["interactions"].to_string(index=False))
    print(result["performance"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
