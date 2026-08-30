from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from strategies.stage04_ml_feedback.final_blend_crash_meta_experiment import (
    DOMESTIC_FEATURES,
    make_factor,
    predictive_metrics,
    walk_forward_probability,
)
from strategies.stage04_ml_feedback.market_structure_feature_experiment import causal_zscore
from strategies.stage04_ml_feedback.market_structure_robustness import run_factor_vol_target
from strategies.stage05_openassetpricing.openassetpricing_signal_experiment import (
    blend_risk_factors,
    forward_path_loss,
    walk_forward_probability_embargo,
)
from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import (
    CAL_END,
    TEST_START,
    FactorBlendConfig,
    paired_block_bootstrap,
    run_factor_blend,
)
from strategies.core.regime_research import (
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
VKOSPI_PATH = ROOT / "raw_data" / "VKOSPIData.csv"
VKOSPI_INTERPRETATION = "Korean equity implied-volatility index; domestic analogue of VIX"
OAP_SOURCE = "https://openassetpricing.com/SignalDoc-Browser.html"
OAP_SIGNAL_MAP = {
    "momentum": ["Mom12m", "Mom6m", "IntMom", "High52"],
    "reversal": ["STreversal", "MRreversal", "LRreversal"],
    "low_risk_tail": ["RealizedVol", "IdioVol3F", "MaxRet", "ReturnSkew", "betaVIX"],
    "excluded_no_volume": ["DolVol", "Illiquidity", "VolumeTrend", "VolSD"],
}

RAW_FEATURES = [
    "vkospi_raw_close",
    "vkospi_raw_change",
    "vkospi_raw_return_pct",
    "vkospi_raw_open",
    "vkospi_raw_high",
    "vkospi_raw_low",
]

DERIVED_FEATURES = [
    "vkospi_return_5",
    "vkospi_return_21",
    "vkospi_return_63",
    "vkospi_mean_ratio_5",
    "vkospi_mean_ratio_21",
    "vkospi_level_z_63",
    "vkospi_level_z_252",
    "vkospi_return_vol_21",
    "vkospi_positive_fraction_21",
    "vkospi_intraday_range_21",
    "vkospi_close_location_21",
    "vkospi_month_mean",
    "vkospi_month_max",
    "vkospi_month_min",
    "vkospi_month_range",
    "vkospi_month_return",
]

OAP_FEATURES = [
    "vkospi_oap_mom12m",
    "vkospi_oap_mom6m",
    "vkospi_oap_intmom",
    "vkospi_oap_streversal",
    "vkospi_oap_mrreversal",
    "vkospi_oap_lrreversal",
    "vkospi_oap_high52",
    "vkospi_oap_realized_vol_21",
    "vkospi_oap_max_return_21",
    "vkospi_oap_return_skew_21",
    "vkospi_oap_beta_21",
    "vkospi_oap_residual_vol_21",
]

COMPOSITE_FEATURES = [
    "vkospi_level_stress",
    "vkospi_momentum_stress",
    "vkospi_range_stress",
    "vkospi_relative_vix_stress",
]

STRUCTURE_FEATURES = [
    "systemic_correlation_stress",
    "breadth_dispersion_stress",
    "volume_stress",
    "tail_shape_stress",
    "sector_linkage_stress",
    "index_volume_proxy_stress",
]


def load_vkospi_daily(path: Path = VKOSPI_PATH) -> pd.DataFrame:
    """Load the KRX export by position so Korean header encoding is irrelevant."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if raw.shape[1] < 7:
        raise ValueError(f"VKOSPI file must contain seven columns: {path}")
    daily = raw.iloc[:, :7].copy()
    daily.columns = ["date", "close", "change", "return_pct", "open", "high", "low"]
    daily["date"] = pd.to_datetime(daily["date"], format="%Y/%m/%d", errors="coerce")
    for column in daily.columns[1:]:
        daily[column] = pd.to_numeric(
            daily[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    daily = daily.dropna(subset=["date", "close"]).set_index("date").sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    if daily.empty or not daily.index.is_monotonic_increasing:
        raise ValueError("VKOSPI data is empty or not chronologically sortable")
    return daily


def _prior_zscore(values: pd.Series, window: int, minimum: int) -> float:
    values = values.dropna()
    if len(values) < minimum + 1:
        return np.nan
    current = float(values.iloc[-1])
    reference = values.iloc[:-1].tail(window)
    if len(reference) < minimum:
        return np.nan
    scale = float(reference.std(ddof=1))
    return (current - float(reference.mean())) / scale if scale > 0 else np.nan


def _lagged_return(close: pd.Series, recent_lag: int, old_lag: int) -> float:
    """Match the SignalDoc convention: price(t-recent)/price(t-old)-1."""
    if len(close) <= old_lag:
        return np.nan
    recent = float(close.iloc[-recent_lag - 1]) if recent_lag else float(close.iloc[-1])
    old = float(close.iloc[-old_lag - 1])
    return recent / old - 1 if old != 0 else np.nan


def load_kodex200_close() -> pd.Series:
    market = pd.read_csv(ROOT / "cache" / "market_daily.csv", parse_dates=["date"])
    close = market.loc[market["symbol"] == "KODEX200", ["date", "close"]].copy()
    close["close"] = pd.to_numeric(close["close"], errors="coerce")
    close = close.dropna().drop_duplicates("date", keep="last").set_index("date")["close"]
    return close.sort_index().rename("KODEX200")


def build_vkospi_features(
    target_months: pd.PeriodIndex,
    daily: pd.DataFrame | None = None,
    kospi_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Build features known before each target month with a two-day safety buffer."""
    daily = load_vkospi_daily() if daily is None else daily.copy()
    kospi_close = load_kodex200_close() if kospi_close is None else kospi_close.copy()
    aligned_kospi = kospi_close.reindex(daily.index).ffill(limit=3)
    kospi_return = aligned_kospi.pct_change(fill_method=None)
    daily_return = daily["close"].pct_change(fill_method=None)
    previous_close = daily["close"].shift(1)
    intraday_range = (daily["high"] - daily["low"]).div(previous_close)
    close_location = (daily["close"] - daily["low"]).div(
        (daily["high"] - daily["low"]).replace(0, np.nan)
    )
    rows: list[dict[str, float | pd.Period | pd.Timestamp]] = []

    for month in target_months:
        cutoff = month.to_timestamp(how="start") - pd.Timedelta(days=2)
        history = daily.loc[:cutoff]
        if len(history) < 63:
            continue
        last = history.iloc[-1]
        close = history["close"]
        returns = daily_return.loc[history.index]
        signal_month = month - 1
        month_view = history.loc[history.index.to_period("M") == signal_month]
        if month_view.empty:
            month_view = history.tail(21)

        row: dict[str, float | pd.Period | pd.Timestamp] = {
            "month": month,
            "vkospi_signal_date": history.index[-1],
            "vkospi_raw_close": float(last["close"]),
            "vkospi_raw_change": float(last["change"]),
            "vkospi_raw_return_pct": float(last["return_pct"]),
            "vkospi_raw_open": float(last["open"]),
            "vkospi_raw_high": float(last["high"]),
            "vkospi_raw_low": float(last["low"]),
        }
        for window in (5, 21, 63):
            row[f"vkospi_return_{window}"] = (
                float(close.iloc[-1] / close.iloc[-window - 1] - 1)
                if len(close) > window
                else np.nan
            )
        for window in (5, 21):
            mean = float(close.tail(window).mean())
            row[f"vkospi_mean_ratio_{window}"] = float(close.iloc[-1] / mean - 1)
        row["vkospi_level_z_63"] = _prior_zscore(close, 63, 42)
        row["vkospi_level_z_252"] = _prior_zscore(close, 252, 126)
        recent_return = returns.tail(21).dropna()
        row["vkospi_return_vol_21"] = (
            float(recent_return.std(ddof=1) * math.sqrt(252))
            if len(recent_return) >= 10
            else np.nan
        )
        row["vkospi_positive_fraction_21"] = (
            float((recent_return > 0).mean()) if len(recent_return) >= 10 else np.nan
        )
        row["vkospi_intraday_range_21"] = float(
            intraday_range.loc[history.index].tail(21).mean()
        )
        row["vkospi_close_location_21"] = float(
            close_location.loc[history.index].tail(21).mean()
        )
        row["vkospi_month_mean"] = float(month_view["close"].mean())
        row["vkospi_month_max"] = float(month_view["high"].max())
        row["vkospi_month_min"] = float(month_view["low"].min())
        row["vkospi_month_range"] = float(
            (month_view["high"].max() - month_view["low"].min())
            / month_view["close"].mean()
        )
        row["vkospi_month_return"] = float(
            month_view["close"].iloc[-1] / month_view["close"].iloc[0] - 1
        )

        # Open Asset Pricing SignalDoc analogues. These preserve the documented
        # lookback/skip-month definitions, translated from a stock cross-section
        # to the domestic implied-volatility series. Positive VKOSPI momentum
        # means rising fear, not positive equity-price momentum; the model learns
        # that direction and the direct factor below fixes it as higher risk.
        row["vkospi_oap_mom12m"] = _lagged_return(close, 21, 252)
        row["vkospi_oap_mom6m"] = _lagged_return(close, 21, 126)
        row["vkospi_oap_intmom"] = _lagged_return(close, 126, 252)
        row["vkospi_oap_streversal"] = _lagged_return(close, 0, 21)
        row["vkospi_oap_mrreversal"] = _lagged_return(close, 252, 378)
        row["vkospi_oap_lrreversal"] = _lagged_return(close, 252, 756)
        row["vkospi_oap_high52"] = float(close.iloc[-1] / close.tail(252).max())
        row["vkospi_oap_realized_vol_21"] = row["vkospi_return_vol_21"]
        row["vkospi_oap_max_return_21"] = (
            float(recent_return.max()) if len(recent_return) >= 10 else np.nan
        )
        row["vkospi_oap_return_skew_21"] = (
            float(recent_return.skew()) if len(recent_return) >= 10 else np.nan
        )

        # betaVIX analogue: sensitivity of KODEX200 daily returns to changes in
        # domestic implied volatility. IdioVol3F analogue: residual volatility
        # from the same domestic systematic-volatility regression.
        regression = pd.concat(
            [
                kospi_return.loc[history.index].rename("kospi_return"),
                daily["close"].diff().loc[history.index].rename("vkospi_change"),
            ],
            axis=1,
        ).tail(21).dropna()
        if len(regression) >= 15 and float(regression["vkospi_change"].var(ddof=1)) > 0:
            x = regression["vkospi_change"].to_numpy(dtype=float)
            y = regression["kospi_return"].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ coefficients
            row["vkospi_oap_beta_21"] = float(coefficients[1])
            row["vkospi_oap_residual_vol_21"] = float(
                np.std(residual, ddof=1) * math.sqrt(252)
            )
        else:
            row["vkospi_oap_beta_21"] = np.nan
            row["vkospi_oap_residual_vol_21"] = np.nan
        rows.append(row)

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output.replace([np.inf, -np.inf], np.nan)


def build_vkospi_composites(data: pd.DataFrame) -> pd.DataFrame:
    """Economically signed, expanding-safe monthly VKOSPI stress composites."""
    z_columns = [
        "vkospi_raw_close",
        "vkospi_month_mean",
        "vkospi_month_max",
        "vkospi_return_5",
        "vkospi_return_21",
        "vkospi_month_return",
        "vkospi_return_vol_21",
        "vkospi_intraday_range_21",
        "vkospi_month_range",
        "vkospi_vix_ratio",
    ]
    z = data[z_columns].apply(causal_zscore)
    output = pd.DataFrame(index=data.index)
    output["vkospi_level_stress"] = z[
        ["vkospi_raw_close", "vkospi_month_mean", "vkospi_month_max"]
    ].mean(axis=1)
    output["vkospi_momentum_stress"] = z[
        ["vkospi_return_5", "vkospi_return_21", "vkospi_month_return"]
    ].mean(axis=1)
    output["vkospi_range_stress"] = z[
        ["vkospi_return_vol_21", "vkospi_intraday_range_21", "vkospi_month_range"]
    ].mean(axis=1)
    output["vkospi_relative_vix_stress"] = z["vkospi_vix_ratio"]
    return output.replace([np.inf, -np.inf], np.nan)


def direct_level_factor(data: pd.DataFrame) -> pd.DataFrame:
    """A model-free VKOSPI risk factor using only the causal monthly z-score."""
    percentile = data["vkospi_level_stress"].expanding(min_periods=24).rank(pct=True)
    severity = ((percentile - 0.80) / 0.20).clip(0, 1)
    output = pd.DataFrame(index=data.index)
    output["risk_percentile"] = percentile
    output["risk_severity"] = severity
    output["p_up"] = 0.5 - 0.15 * severity
    output["score"] = -severity
    output["factor_name"] = "vkospi_direct_level"
    return output


def metric_record(period: str, strategy: str, path: pd.DataFrame) -> dict[str, object]:
    metrics = performance_summary(path["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        "Months": len(path),
        **metrics.to_dict(),
        "AvgTurnover": float(path["turnover"].mean()),
    }


def univariate_audit(data: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, start, end in (
        ("calibration", data.index.min(), CAL_END),
        ("locked", TEST_START, data.index.max()),
    ):
        for column in columns:
            view = data.loc[start:end, [target, column]].dropna()
            if view.empty or view[target].nunique() < 2 or view[column].nunique() < 2:
                continue
            auc = float(roc_auc_score(view[target].astype(int), view[column]))
            rows.append(
                {
                    "Period": period,
                    "Target": target,
                    "Feature": column,
                    "AUC": auc,
                    "DirectionalAUC": max(auc, 1 - auc),
                    "HighMeansRisk": auc >= 0.5,
                    "Observations": len(view),
                    "Events": int(view[target].sum()),
                }
            )
    return pd.DataFrame(rows)


def _add_candidate_rows(
    rows: list[dict[str, object]],
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    name: str,
    factor: pd.DataFrame,
    path: pd.DataFrame,
    **parameters: object,
) -> None:
    candidates[name] = (factor, path)
    metrics = performance_summary(path.loc[:CAL_END, "return"])
    rows.append(
        {
            "Strategy": name,
            **parameters,
            **metrics.to_dict(),
            "AvgTurnover": float(path.loc[:CAL_END, "turnover"].mean()),
        }
    )


def main() -> None:
    base_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    base_data.index = pd.PeriodIndex(base_data.index, freq="M")
    vkospi = build_vkospi_features(base_data.index)
    if (vkospi["vkospi_signal_date"].dt.to_period("M") >= vkospi.index).any():
        raise AssertionError("VKOSPI feature timing leaked into the target month")
    structure = pd.read_csv(RESULTS / "market_structure_composites.csv", index_col=0)
    structure.index = pd.PeriodIndex(structure.index, freq="M")
    data = base_data.join(vkospi, how="left").join(structure, how="left")
    data["vkospi_vix_ratio"] = data["vkospi_raw_close"].div(
        data["VIX_last"].replace(0, np.nan)
    )
    composites = build_vkospi_composites(data)
    data = data.join(composites, how="left")

    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")
    neutral = pd.DataFrame({"p_up": 0.5}, index=data.index)
    final_blend = run_factor_blend(
        asset_returns, signals, defensive, neutral, FactorBlendConfig(max_shift=0.0)
    )
    data = data.loc[data.index.intersection(final_blend.index)].copy()
    baseline_return = final_blend.loc[data.index, "return"]
    data["final_loss3"] = (baseline_return < -0.03).astype(float)
    data["final_loss4"] = (baseline_return < -0.04).astype(float)
    for name, threshold in (("path_loss_2m_4", -0.04), ("path_loss_2m_5", -0.05)):
        path_loss = forward_path_loss(baseline_return, 2)
        data[name] = (path_loss < threshold).where(path_loss.notna()).astype(float)

    model_specs: dict[str, tuple[str, int, list[str]]] = {
        "loss3_domestic_raw": ("final_loss3", 1, DOMESTIC_FEATURES + RAW_FEATURES),
        "loss3_domestic_derived": (
            "final_loss3",
            1,
            DOMESTIC_FEATURES + DERIVED_FEATURES + COMPOSITE_FEATURES,
        ),
        "loss3_domestic_all": (
            "final_loss3",
            1,
            DOMESTIC_FEATURES
            + RAW_FEATURES
            + DERIVED_FEATURES
            + OAP_FEATURES
            + COMPOSITE_FEATURES,
        ),
        "loss3_domestic_oap": (
            "final_loss3",
            1,
            DOMESTIC_FEATURES + OAP_FEATURES,
        ),
        "loss3_structure_all": (
            "final_loss3",
            1,
            DOMESTIC_FEATURES
            + STRUCTURE_FEATURES
            + RAW_FEATURES
            + DERIVED_FEATURES
            + OAP_FEATURES
            + COMPOSITE_FEATURES,
        ),
        "loss4_structure_all": (
            "final_loss4",
            1,
            DOMESTIC_FEATURES
            + STRUCTURE_FEATURES
            + RAW_FEATURES
            + DERIVED_FEATURES
            + OAP_FEATURES
            + COMPOSITE_FEATURES,
        ),
        "path2m4_structure_all": (
            "path_loss_2m_4",
            2,
            DOMESTIC_FEATURES
            + STRUCTURE_FEATURES
            + RAW_FEATURES
            + DERIVED_FEATURES
            + OAP_FEATURES
            + COMPOSITE_FEATURES,
        ),
        "path2m5_structure_all": (
            "path_loss_2m_5",
            2,
            DOMESTIC_FEATURES
            + STRUCTURE_FEATURES
            + RAW_FEATURES
            + DERIVED_FEATURES
            + OAP_FEATURES
            + COMPOSITE_FEATURES,
        ),
    }

    factors: dict[str, pd.DataFrame] = {"direct_level": direct_level_factor(data)}
    prediction_report: dict[str, object] = {}
    for name, (target, embargo, columns) in model_specs.items():
        if embargo == 1:
            probability = walk_forward_probability(data, target, columns)
        else:
            probability = walk_forward_probability_embargo(
                data, target, columns, embargo_months=embargo
            )
        factor = make_factor(probability, data[target], name)
        factors[name] = factor
        factor.to_csv(RESULTS / f"vkospi_{name}_factor.csv")
        prediction_report[name] = {
            "target": target,
            "embargo_months": embargo,
            "feature_count": len(columns),
            "calibration": predictive_metrics(factor.loc[:CAL_END]),
            "locked": predictive_metrics(factor, TEST_START),
        }

    reference_factor = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0
    )
    reference_factor.index = pd.PeriodIndex(reference_factor.index, freq="M")
    reference_path = run_factor_vol_target(
        asset_returns,
        signals,
        defensive,
        reference_factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    reference_metrics = performance_summary(reference_path.loc[:CAL_END, "return"])

    calibration_rows: list[dict[str, object]] = []
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    _add_candidate_rows(
        calibration_rows,
        candidates,
        "ReferenceMediumHorizonOAPVol15",
        reference_factor,
        reference_path,
        source="reference",
        vkospi_weight=0.0,
        max_shift=0.20,
        target_vol=0.15,
    )
    for factor_name, factor in factors.items():
        for shift in (0.10, 0.15, 0.20, 0.25):
            for target_vol in (0.145, 0.15, 0.155):
                name = f"VKOSPI_{factor_name}_s{shift:.3f}_v{target_vol:.3f}"
                path = run_factor_vol_target(
                    asset_returns,
                    signals,
                    defensive,
                    factor,
                    max_shift=shift,
                    target_vol=target_vol,
                )
                _add_candidate_rows(
                    calibration_rows,
                    candidates,
                    name,
                    factor,
                    path,
                    source=factor_name,
                    vkospi_weight=1.0,
                    max_shift=shift,
                    target_vol=target_vol,
                )

        for weight in (0.25, 0.50, 0.75):
            blended = blend_risk_factors(reference_factor, factor, weight)
            for shift in (0.15, 0.20, 0.25):
                for target_vol in (0.145, 0.15, 0.155):
                    name = (
                        f"Blend_{factor_name}_w{weight:.2f}_s{shift:.3f}_v{target_vol:.3f}"
                    )
                    path = run_factor_vol_target(
                        asset_returns,
                        signals,
                        defensive,
                        blended,
                        max_shift=shift,
                        target_vol=target_vol,
                    )
                    _add_candidate_rows(
                        calibration_rows,
                        candidates,
                        name,
                        blended,
                        path,
                        source=factor_name,
                        vkospi_weight=weight,
                        max_shift=shift,
                        target_vol=target_vol,
                    )

    calibration = pd.DataFrame(calibration_rows)
    eligible = calibration[
        (calibration["CAGR"] >= float(reference_metrics["CAGR"]) - 1e-12)
        & (calibration["Sharpe"] >= float(reference_metrics["Sharpe"]) - 1e-12)
        & (calibration["MDD"] >= float(reference_metrics["MDD"]) - 1e-12)
    ].copy()
    for metric in ("CAGR", "Sharpe", "MDD"):
        eligible[f"{metric}Rank"] = eligible[metric].rank(pct=True, method="average")
    eligible["MultiObjectiveScore"] = eligible[
        ["CAGRRank", "SharpeRank", "MDDRank"]
    ].mean(axis=1)
    winner_row = eligible.sort_values(
        ["MultiObjectiveScore", "Calmar", "Sharpe", "CAGR"], ascending=False
    ).iloc[0]
    winner_name = str(winner_row["Strategy"])
    winner_factor, winner_path = candidates[winner_name]

    reference_locked = reference_path.loc[TEST_START:]
    winner_locked = winner_path.loc[TEST_START:]
    reference_locked_metrics = performance_summary(reference_locked["return"])
    winner_locked_metrics = performance_summary(winner_locked["return"])
    locked_deltas = {
        metric: float(winner_locked_metrics[metric] - reference_locked_metrics[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    # The locked period is an acceptance gate, never a way to choose another
    # candidate. A failed candidate leaves the previously validated reference
    # untouched, preventing a research improvement from degrading production.
    improves_all_three = bool(
        locked_deltas["CAGR"] >= 0
        and locked_deltas["Sharpe"] >= 0
        and locked_deltas["MDD"] >= 0
        and (
            locked_deltas["CAGR"] > 0
            or locked_deltas["Sharpe"] > 0
            or locked_deltas["MDD"] > 0
        )
    )
    deployed_name = winner_name if improves_all_three else "ReferenceMediumHorizonOAPVol15"
    deployed_factor = winner_factor if improves_all_three else reference_factor
    deployed_path = winner_path if improves_all_three else reference_path

    comparison_rows: list[dict[str, object]] = []
    selected_paths = {
        "FinalBlend": final_blend,
        "ReferenceMediumHorizonOAPVol15": reference_path,
        "VKOSPICalibrationCandidate": winner_path,
        "DeployedStrategy": deployed_path,
    }
    for period, start, end in (
        ("calibration_2007_2017", final_blend.index.min(), CAL_END),
        ("locked_2018_2026", TEST_START, final_blend.index.max()),
        ("full_2007_2026", final_blend.index.min(), final_blend.index.max()),
    ):
        for strategy, path in selected_paths.items():
            comparison_rows.append(metric_record(period, strategy, path.loc[start:end]))
    comparison = pd.DataFrame(comparison_rows)

    audit = univariate_audit(
        data,
        RAW_FEATURES + DERIVED_FEATURES + OAP_FEATURES + COMPOSITE_FEATURES,
        "final_loss3",
    )

    vkospi.to_csv(RESULTS / "vkospi_features.csv")
    composites.to_csv(RESULTS / "vkospi_composites.csv")
    audit.to_csv(RESULTS / "vkospi_univariate_audit.csv", index=False)
    calibration.to_csv(RESULTS / "vkospi_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "vkospi_comparison.csv", index=False)
    winner_factor.to_csv(RESULTS / "vkospi_candidate_factor.csv")
    winner_path.to_csv(RESULTS / "vkospi_candidate_backtest.csv")
    deployed_factor.to_csv(RESULTS / "vkospi_selected_factor.csv")
    deployed_path.to_csv(RESULTS / "vkospi_selected_backtest.csv")

    report = {
        "data": {
            "path": str(VKOSPI_PATH),
            "interpretation": VKOSPI_INTERPRETATION,
            "daily_start": str(load_vkospi_daily().index.min().date()),
            "daily_end": str(load_vkospi_daily().index.max().date()),
            "daily_rows": len(load_vkospi_daily()),
            "raw_features": RAW_FEATURES,
            "derived_features": DERIVED_FEATURES,
            "open_asset_pricing_source": OAP_SOURCE,
            "open_asset_pricing_signal_map": OAP_SIGNAL_MAP,
            "open_asset_pricing_analog_features": OAP_FEATURES,
            "composite_features": COMPOSITE_FEATURES,
            "timing": "target-month start minus two calendar days; no target-month observation",
        },
        "selection": {
            "calibration_end": str(CAL_END),
            "locked_start": str(TEST_START),
            "rule": "CAGR, Sharpe, and MDD must all match or beat the reference; equal-weight percentile rank selects among eligible candidates",
            "reference": "ReferenceMediumHorizonOAPVol15",
            "calibration_candidate": winner_name,
            "candidate_parameters": {
                key: (float(winner_row[key]) if key != "source" else str(winner_row[key]))
                for key in ("source", "vkospi_weight", "max_shift", "target_vol")
            },
            "eligible_candidates": len(eligible),
            "total_candidates": len(calibration),
            "locked_acceptance_rule": "candidate must match or improve CAGR, Sharpe, and MDD, with at least one strict improvement",
            "candidate_promoted": improves_all_three,
            "deployed_strategy": deployed_name,
        },
        "locked_test": {
            "reference": reference_locked_metrics.to_dict(),
            "calibration_candidate": winner_locked_metrics.to_dict(),
            "candidate_deltas": locked_deltas,
            "improves_all_three": improves_all_three,
            "deployed": performance_summary(deployed_path.loc[TEST_START:, "return"]).to_dict(),
            "bootstrap": paired_block_bootstrap(
                reference_locked["return"], winner_locked["return"]
            ),
        },
        "prediction": prediction_report,
        "notes": [
            "All model, blend, shift, and volatility-target selection uses data through 2017 only.",
            "The 2018-2026 locked period never chooses a candidate or hyperparameters; it is only a one-way promotion gate.",
            "A locked-test failure automatically retains the prior validated reference strategy.",
            "Raw KRX fields and engineered VKOSPI fields are tested separately before being combined.",
            "Open Asset Pricing price/risk signals are time-series analogues, not exact stock-level replications.",
            "Open Asset Pricing volume/liquidity signals are excluded because the VKOSPI export has no volume field.",
            "Historical simulation does not guarantee future performance.",
        ],
    }
    (RESULTS / "vkospi_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== VKOSPI COVERAGE ===")
    print(vkospi.notna().mean().sort_values().to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== CALIBRATION CANDIDATE ===")
    print(winner_row.to_string())
    print("\n=== COMPARISON ===")
    print(
        comparison[
            ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\n=== LOCKED DELTAS VS REFERENCE ===")
    print(json.dumps(locked_deltas, indent=2))
    print(f"\nPROMOTED={improves_all_three} DEPLOYED={deployed_name}")


if __name__ == "__main__":
    main()
