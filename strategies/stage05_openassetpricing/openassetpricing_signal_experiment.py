from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from strategies.stage04_ml_feedback.final_blend_crash_meta_experiment import (
    DOMESTIC_FEATURES,
    make_factor,
    make_model,
    predictive_metrics,
    walk_forward_probability,
)
from strategies.stage04_ml_feedback.market_structure_feature_experiment import (
    KOSPI200,
    SECTORS,
    causal_zscore,
    load_index_history,
)
from strategies.stage04_ml_feedback.market_structure_robustness import run_factor_vol_target
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
from strategies.stage04_ml_feedback.short_regime_tail_risk_experiment import causal_percentile


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

OAP_SOURCE = "https://openassetpricing.com/SignalDoc-Browser.html"
OAP_SIGNAL_MAP = {
    "momentum_trend": ["Mom12m", "Mom6m", "IntMom", "High52", "IndMom", "MomVol"],
    "reversal_crowding": ["STreversal", "MRreversal", "LRreversal"],
    "low_risk_tail": ["RealizedVol", "IdioVol3F", "MaxRet", "ReturnSkew", "Beta"],
    "liquidity_activity": ["Illiquidity", "DolVol", "VolumeTrend", "VolSD"],
}


def finite_summary(values: pd.Series, statistic: str, minimum: int = 4) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < minimum:
        return np.nan
    if statistic == "median":
        return float(clean.median())
    if statistic == "mean":
        return float(clean.mean())
    if statistic == "std":
        return float(clean.std(ddof=1))
    raise ValueError(statistic)


def lagged_return(history: pd.DataFrame, recent_lag: int, old_lag: int) -> pd.Series:
    if len(history) <= old_lag:
        return pd.Series(np.nan, index=history.columns)
    recent = history.iloc[-1 - recent_lag]
    old = history.iloc[-1 - old_lag]
    return recent.div(old).sub(1.0).replace([np.inf, -np.inf], np.nan)


def linear_trend_scaled(series: pd.Series, window: int, minimum: int) -> float:
    clean = series.dropna().tail(window)
    if len(clean) < minimum:
        return np.nan
    values = clean.to_numpy(dtype=float)
    mean = float(np.mean(values))
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return np.nan
    time = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(time, values, 1)[0])
    return slope * len(values) / mean


def monthly_volume_statistics(history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    monthly = history.resample("ME").sum(min_count=1)
    trends = pd.Series(
        {column: linear_trend_scaled(monthly[column], 60, 30) for column in monthly.columns},
        dtype=float,
    )
    variation: dict[str, float] = {}
    for column in monthly.columns:
        clean = monthly[column].dropna().tail(36)
        if len(clean) < 24 or abs(float(clean.mean())) < 1e-12:
            variation[column] = np.nan
        else:
            variation[column] = float(clean.std(ddof=1) / clean.mean())
    return trends, pd.Series(variation, dtype=float)


def capm_cross_section(window_returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    betas: dict[str, float] = {}
    residual_volatility: dict[str, float] = {}
    for sector in SECTORS:
        view = window_returns[[KOSPI200, sector]].dropna()
        if len(view) < 40:
            betas[sector] = np.nan
            residual_volatility[sector] = np.nan
            continue
        market = view[KOSPI200].to_numpy(dtype=float)
        asset = view[sector].to_numpy(dtype=float)
        variance = float(np.var(market, ddof=1))
        if variance < 1e-12:
            betas[sector] = np.nan
            residual_volatility[sector] = np.nan
            continue
        beta = float(np.cov(asset, market, ddof=1)[0, 1] / variance)
        alpha = float(np.mean(asset) - beta * np.mean(market))
        residual = asset - alpha - beta * market
        betas[sector] = beta
        residual_volatility[sector] = float(np.std(residual, ddof=1) * np.sqrt(252))
    return pd.Series(betas, dtype=float), pd.Series(residual_volatility, dtype=float)


def build_oap_features(target_months: pd.PeriodIndex) -> pd.DataFrame:
    close, volume = load_index_history()
    columns = [KOSPI200] + SECTORS
    close = close.reindex(columns=columns)
    volume = volume.reindex(columns=columns).replace(0, np.nan)
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    dollar_volume = close * volume
    illiquidity = returns.abs().div(dollar_volume.replace(0, np.nan))
    rows: list[dict[str, float | pd.Period]] = []

    for month in target_months:
        cutoff = month.to_timestamp(how="start") - pd.Timedelta(days=2)
        history_close = close.loc[:cutoff]
        history_return = returns.loc[:cutoff]
        history_volume = volume.loc[:cutoff]
        history_dollar_volume = dollar_volume.loc[:cutoff]
        history_illiquidity = illiquidity.loc[:cutoff]
        if len(history_close) < 756:
            continue

        sector_close = history_close[SECTORS]
        sector_return = history_return[SECTORS]
        mom12 = lagged_return(sector_close, 21, 252)
        mom6 = lagged_return(sector_close, 21, 126)
        intermediate_momentum = lagged_return(sector_close, 126, 252)
        short_reversal = lagged_return(sector_close, 0, 21)
        medium_reversal = lagged_return(sector_close, 252, 378)
        long_reversal = lagged_return(sector_close, 252, 756)
        high52 = sector_close.iloc[-1].div(sector_close.tail(252).max()).replace(
            [np.inf, -np.inf], np.nan
        )

        recent21 = sector_return.tail(21)
        realized_volatility = recent21.std(ddof=1) * np.sqrt(252)
        maximum_return = recent21.max()
        return_skewness = recent21.skew()
        betas, residual_volatility = capm_cross_section(history_return.tail(252))

        recent_dollar_volume = history_dollar_volume[SECTORS].tail(21).mean()
        valid_weight = recent_dollar_volume.where(mom6.notna()).dropna()
        if len(valid_weight) >= 4 and float(valid_weight.sum()) > 0:
            valid_weight = valid_weight / valid_weight.sum()
            volume_weighted_momentum = float((mom6.loc[valid_weight.index] * valid_weight).sum())
            mom_volume_correlation = float(
                mom6.loc[valid_weight.index].corr(np.log1p(recent_dollar_volume.loc[valid_weight.index]))
            )
        else:
            volume_weighted_momentum = np.nan
            mom_volume_correlation = np.nan

        illiquidity63 = history_illiquidity[SECTORS].tail(63).mean()
        illiquidity21 = history_illiquidity[SECTORS].tail(21).mean()
        illiquidity252 = history_illiquidity[SECTORS].tail(252).mean()
        illiquidity_ratio = illiquidity21.div(illiquidity252).replace([np.inf, -np.inf], np.nan)
        log_illiquidity63 = np.log10(illiquidity63.clip(lower=1e-18))

        log_dollar_volume63 = np.log1p(history_dollar_volume[SECTORS].tail(63))
        dollar_volume_z63 = (
            log_dollar_volume63.iloc[-1] - log_dollar_volume63.mean()
        ).div(log_dollar_volume63.std(ddof=1).replace(0, np.nan))
        volume_trend, volume_variation = monthly_volume_statistics(history_volume[SECTORS])

        market_illiquidity = history_illiquidity[KOSPI200].dropna()
        market_illiq_ratio = (
            float(market_illiquidity.tail(21).mean() / market_illiquidity.tail(252).mean())
            if len(market_illiquidity) >= 252
            else np.nan
        )
        market_monthly_volume = history_volume[KOSPI200].resample("ME").sum(min_count=1)
        market_volume_trend = linear_trend_scaled(market_monthly_volume, 60, 30)
        market_volume36 = market_monthly_volume.dropna().tail(36)
        market_volume_variation = (
            float(market_volume36.std(ddof=1) / market_volume36.mean())
            if len(market_volume36) >= 24 and float(market_volume36.mean()) > 0
            else np.nan
        )
        market_close = history_close[[KOSPI200]]

        row: dict[str, float | pd.Period] = {
            "month": month,
            # Momentum, 52-week-high, and industry-momentum analogues.
            "oap_sector_mom12_median": finite_summary(mom12, "median"),
            "oap_sector_mom6_median": finite_summary(mom6, "median"),
            "oap_sector_intmom_median": finite_summary(intermediate_momentum, "median"),
            "oap_sector_high52_median": finite_summary(high52, "median"),
            "oap_sector_mom6_breadth": float((mom6.dropna() > 0).mean())
            if len(mom6.dropna()) >= 4
            else np.nan,
            "oap_sector_mom12_dispersion": finite_summary(mom12, "std"),
            "oap_sector_mom6_dolvol_weighted": volume_weighted_momentum,
            "oap_sector_mom6_dolvol_corr": mom_volume_correlation,
            "oap_kospi_mom12": float(lagged_return(market_close, 21, 252).iloc[0]),
            "oap_kospi_high52": float(
                market_close.iloc[-1, 0] / market_close.tail(252).max().iloc[0]
            ),
            # Reversal and crowding analogues.
            "oap_sector_streversal_median": finite_summary(short_reversal, "median"),
            "oap_sector_streversal_dispersion": finite_summary(short_reversal, "std"),
            "oap_sector_mrreversal_median": finite_summary(medium_reversal, "median"),
            "oap_sector_lrreversal_median": finite_summary(long_reversal, "median"),
            "oap_kospi_streversal": float(lagged_return(market_close, 0, 21).iloc[0]),
            # Low-risk, volatility, maximum-return, skewness, and beta analogues.
            "oap_sector_realizedvol21_median": finite_summary(realized_volatility, "median"),
            "oap_sector_idiovol252_median": finite_summary(residual_volatility, "median"),
            "oap_sector_maxret21_median": finite_summary(maximum_return, "median"),
            "oap_sector_returnskew21_median": finite_summary(return_skewness, "median"),
            "oap_sector_beta252_mean": finite_summary(betas, "mean"),
            "oap_sector_beta252_dispersion": finite_summary(betas, "std"),
            # Amihud, dollar-volume, volume-trend, and volume-variance analogues.
            "oap_sector_log_illiquidity63_median": finite_summary(
                log_illiquidity63, "median"
            ),
            "oap_sector_illiquidity_ratio21_252_median": finite_summary(
                illiquidity_ratio, "median"
            ),
            "oap_kospi_illiquidity_ratio21_252": market_illiq_ratio,
            "oap_sector_dolvol_z63_median": finite_summary(dollar_volume_z63, "median"),
            "oap_sector_volume_trend60_median": finite_summary(volume_trend, "median"),
            "oap_sector_volume_cv36_median": finite_summary(volume_variation, "median"),
            "oap_kospi_volume_trend60": market_volume_trend,
            "oap_kospi_volume_cv36": market_volume_variation,
        }
        rows.append(row)

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output.replace([np.inf, -np.inf], np.nan)


def build_oap_composites(features: pd.DataFrame) -> pd.DataFrame:
    z = features.apply(causal_zscore)
    output = pd.DataFrame(index=features.index)
    output["oap_momentum_trend_stress"] = (
        -z["oap_sector_mom12_median"]
        - z["oap_sector_mom6_median"]
        - z["oap_sector_intmom_median"]
        - z["oap_sector_high52_median"]
        - z["oap_sector_mom6_breadth"]
        - z["oap_sector_mom6_dolvol_weighted"]
    ) / 6
    output["oap_reversal_crowding_stress"] = (
        -z["oap_sector_streversal_median"]
        - z["oap_kospi_streversal"]
        + z["oap_sector_streversal_dispersion"]
        + z["oap_sector_mom12_dispersion"]
    ) / 4
    output["oap_low_risk_tail_stress"] = (
        z["oap_sector_realizedvol21_median"]
        + z["oap_sector_idiovol252_median"]
        + z["oap_sector_maxret21_median"]
        - z["oap_sector_returnskew21_median"]
        + z["oap_sector_beta252_mean"]
        + z["oap_sector_beta252_dispersion"]
    ) / 6
    liquidity_parts = pd.concat(
        [
            z["oap_sector_log_illiquidity63_median"],
            z["oap_sector_illiquidity_ratio21_252_median"],
            z["oap_kospi_illiquidity_ratio21_252"],
            -z["oap_sector_dolvol_z63_median"],
            -z["oap_sector_volume_trend60_median"],
            z["oap_sector_volume_cv36_median"],
        ],
        axis=1,
    )
    output["oap_liquidity_activity_stress"] = liquidity_parts.mean(axis=1).where(
        liquidity_parts.notna().sum(axis=1) >= 4
    )
    return output.replace([np.inf, -np.inf], np.nan)


def forward_path_loss(returns: pd.Series, horizon: int) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index, dtype=float)
    for position in range(len(returns)):
        future = returns.iloc[position : position + horizon]
        if len(future) < horizon:
            continue
        path = (1 + future).cumprod() - 1
        output.iloc[position] = float(path.min())
    return output


def walk_forward_probability_embargo(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    embargo_months: int,
    min_train: int = 36,
) -> pd.Series:
    probability = pd.Series(np.nan, index=data.index, dtype=float)
    template = make_model()
    for number in range(len(data)):
        # The label consumes `embargo_months` future returns. Excluding the same
        # number of trailing rows prevents overlapping future returns entering
        # the training set for the prediction month.
        train_end = number - embargo_months
        if train_end < min_train:
            continue
        train = data.iloc[:train_end].dropna(subset=[target])
        y = train[target].astype(int)
        if y.sum() < 4 or (len(y) - y.sum()) < 12:
            continue
        model = clone(template)
        model.fit(train[features], y)
        probability.iloc[number] = float(
            model.predict_proba(data.iloc[[number]][features])[:, 1][0]
        )
    return probability


def metric_record(period: str, strategy: str, backtest: pd.DataFrame) -> dict[str, object]:
    metrics = performance_summary(backtest["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        "Months": len(backtest),
        **metrics.to_dict(),
        "AvgTurnover": float(backtest["turnover"].mean()),
    }


def load_factor(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"market_structure_{name}_factor.csv")
    frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
    return frame.set_index("month")


def blend_risk_factors(first: pd.DataFrame, second: pd.DataFrame, second_weight: float) -> pd.DataFrame:
    index = first.index.union(second.index).sort_values()
    first_severity = ((0.5 - first.reindex(index)["p_up"]) / 0.15).clip(0, 1)
    second_severity = ((0.5 - second.reindex(index)["p_up"]) / 0.15).clip(0, 1)
    severity = (1 - second_weight) * first_severity + second_weight * second_severity
    output = pd.DataFrame(index=index)
    output["risk_severity"] = severity
    output["p_up"] = 0.5 - 0.15 * severity
    output["score"] = -severity
    output["factor_name"] = f"blend_{second_weight:.2f}"
    return output


def make_direct_trend_factor(composites: pd.DataFrame) -> pd.DataFrame:
    # High momentum percentile is an upside signal. The percentile is causal and
    # therefore remains comparable across the changing KRX sector-index universe.
    strength = -composites["oap_momentum_trend_stress"]
    percentile = causal_percentile(strength)
    score = (2 * percentile - 1).clip(-1, 1)
    output = pd.DataFrame(index=composites.index)
    output["trend_percentile"] = percentile
    output["score"] = score
    output["p_up"] = 0.5 + 0.15 * score
    output["factor_name"] = "oap_direct_momentum"
    return output


def compose_risk_override_trend(
    existing: pd.DataFrame,
    trend: pd.DataFrame,
    trend_weight: float,
    mode: str,
) -> pd.DataFrame:
    index = existing.index.union(trend.index).sort_values()
    risk = ((0.5 - existing.reindex(index)["p_up"]) / 0.15).clip(0, 1).fillna(0)
    trend_score = ((trend.reindex(index)["p_up"] - 0.5) / 0.15).clip(-1, 1).fillna(0)
    if mode == "risk_on_only":
        upside = trend_score.clip(lower=0)
    elif mode == "symmetric":
        upside = trend_score
    else:
        raise ValueError(mode)
    # Existing crash-risk defense has priority. OAP trend only uses the residual
    # risk budget left when that defense is inactive.
    score = (-risk + (1 - risk) * trend_weight * upside).clip(-1, 1)
    output = pd.DataFrame(index=index)
    output["risk_severity"] = risk
    output["trend_score"] = trend_score
    output["score"] = score
    output["p_up"] = 0.5 + 0.15 * score
    output["factor_name"] = f"risk_override_{mode}_{trend_weight:.2f}"
    return output


def choose_shift(
    factor: pd.DataFrame,
    asset_returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    baseline_calibration: pd.Series,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    paths: dict[float, pd.DataFrame] = {}
    for shift in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
        path = run_factor_blend(
            asset_returns, signals, defensive, factor, FactorBlendConfig(max_shift=shift)
        )
        paths[shift] = path
        metrics = performance_summary(path.loc[:CAL_END, "return"])
        rows.append({"max_shift": shift, **metrics.to_dict()})
    table = pd.DataFrame(rows)
    eligible = table[
        (table["MDD"] >= -0.15)
        & (table["CAGR"] >= 0.95 * float(baseline_calibration["CAGR"]))
    ]
    pool = eligible if not eligible.empty else table
    winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
    shift = float(winner["max_shift"])
    return shift, paths[shift], table


def choose_signal_blend(
    existing: pd.DataFrame,
    oap: pd.DataFrame,
    asset_returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    baseline_calibration: pd.Series,
    existing_shift: float,
) -> tuple[float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    candidates: dict[tuple[float, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
    # Weight zero preserves the existing strategy and allows an honest no-change
    # decision when the OAP probability signal has no incremental value.
    for oap_weight in (0.0, 0.25, 0.50, 0.75):
        factor = blend_risk_factors(existing, oap, oap_weight)
        shift_grid = (existing_shift,) if oap_weight == 0 else (0.05, 0.10, 0.15, 0.20)
        for shift in shift_grid:
            path = run_factor_vol_target(
                asset_returns,
                signals,
                defensive,
                factor,
                max_shift=shift,
                target_vol=0.15,
            )
            candidates[(oap_weight, shift)] = (factor, path)
            metrics = performance_summary(path.loc[:CAL_END, "return"])
            rows.append(
                {
                    "oap_weight": oap_weight,
                    "max_shift": shift,
                    **metrics.to_dict(),
                }
            )
    table = pd.DataFrame(rows)
    eligible = table[
        (table["MDD"] >= -0.15)
        & (table["CAGR"] >= 0.95 * float(baseline_calibration["CAGR"]))
    ]
    pool = eligible if not eligible.empty else table
    winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
    weight = float(winner["oap_weight"])
    shift = float(winner["max_shift"])
    factor, path = candidates[(weight, shift)]
    return weight, shift, factor, path, table


def choose_risk_override_trend(
    existing: pd.DataFrame,
    existing_shift: float,
    trend: pd.DataFrame,
    asset_returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    reference_calibration: pd.Series,
) -> tuple[str, float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    candidates: dict[tuple[str, float, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
    baseline_factor = compose_risk_override_trend(existing, trend, 0.0, "risk_on_only")
    baseline_path = run_factor_vol_target(
        asset_returns,
        signals,
        defensive,
        baseline_factor,
        max_shift=existing_shift,
        target_vol=0.15,
    )
    candidates[("existing", 0.0, existing_shift)] = (baseline_factor, baseline_path)
    baseline_metrics = performance_summary(baseline_path.loc[:CAL_END, "return"])
    rows.append(
        {
            "mode": "existing",
            "trend_weight": 0.0,
            "max_shift": existing_shift,
            **baseline_metrics.to_dict(),
        }
    )
    for mode in ("risk_on_only", "symmetric"):
        for trend_weight in (0.25, 0.50, 0.75, 1.00):
            factor = compose_risk_override_trend(existing, trend, trend_weight, mode)
            for shift in (0.05, 0.10, 0.15, 0.20):
                path = run_factor_vol_target(
                    asset_returns,
                    signals,
                    defensive,
                    factor,
                    max_shift=shift,
                    target_vol=0.15,
                )
                candidates[(mode, trend_weight, shift)] = (factor, path)
                metrics = performance_summary(path.loc[:CAL_END, "return"])
                rows.append(
                    {
                        "mode": mode,
                        "trend_weight": trend_weight,
                        "max_shift": shift,
                        **metrics.to_dict(),
                    }
                )
    table = pd.DataFrame(rows)
    # Preserve both return and drawdown quality of the existing strategy on the
    # calibration sample; then rank the surviving choices by Calmar/Sharpe/CAGR.
    eligible = table[
        (table["CAGR"] >= 0.99 * float(reference_calibration["CAGR"]))
        & (table["MDD"] >= float(reference_calibration["MDD"]) - 0.005)
    ]
    pool = eligible if not eligible.empty else table
    winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
    mode = str(winner["mode"])
    weight = float(winner["trend_weight"])
    shift = float(winner["max_shift"])
    factor, path = candidates[(mode, weight, shift)]
    return mode, weight, shift, factor, path, table


def choose_risk_committee(
    factors: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    reference_calibration: pd.Series,
) -> tuple[str, float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.PeriodIndex([], freq="M")
    for factor in factors.values():
        index = index.union(factor.index)
    severities = pd.concat(
        {
            name: ((0.5 - factor.reindex(index)["p_up"]) / 0.15).clip(0, 1).fillna(0)
            for name, factor in factors.items()
        },
        axis=1,
    )
    aggregations = {
        "minimum_consensus": severities.min(axis=1),
        "median_committee": severities.median(axis=1),
        "mean_committee": severities.mean(axis=1),
        "maximum_union": severities.max(axis=1),
    }
    rows: list[dict[str, object]] = []
    candidates: dict[tuple[str, float, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for method, severity in aggregations.items():
        factor = pd.DataFrame(index=index)
        factor["risk_severity"] = severity
        factor["p_up"] = 0.5 - 0.15 * severity
        factor["score"] = -severity
        factor["factor_name"] = method
        for shift in (0.10, 0.15, 0.20):
            for target_vol in (0.145, 0.150, 0.155):
                path = run_factor_vol_target(
                    asset_returns,
                    signals,
                    defensive,
                    factor,
                    max_shift=shift,
                    target_vol=target_vol,
                )
                candidates[(method, shift, target_vol)] = (factor, path)
                metrics = performance_summary(path.loc[:CAL_END, "return"])
                rows.append(
                    {
                        "method": method,
                        "max_shift": shift,
                        "target_vol": target_vol,
                        **metrics.to_dict(),
                    }
                )
    table = pd.DataFrame(rows)
    eligible = table[
        (table["CAGR"] >= 0.99 * float(reference_calibration["CAGR"]))
        & (table["MDD"] >= float(reference_calibration["MDD"]) - 0.005)
    ]
    pool = eligible if not eligible.empty else table
    winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
    method = str(winner["method"])
    shift = float(winner["max_shift"])
    target_vol = float(winner["target_vol"])
    factor, path = candidates[(method, shift, target_vol)]
    return method, shift, target_vol, factor, path, table


def choose_medium_horizon_oap(
    data: pd.DataFrame,
    feature_columns: list[str],
    configurations: list[tuple[str, int]],
    asset_returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    reference_calibration: pd.Series,
) -> tuple[str, int, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    candidates: dict[tuple[str, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target, horizon in configurations:
        probability = walk_forward_probability_embargo(
            data,
            target,
            feature_columns,
            embargo_months=horizon,
        )
        factor = make_factor(probability, data[target], target)
        for shift in (0.05, 0.10, 0.15, 0.20):
            path = run_factor_vol_target(
                asset_returns,
                signals,
                defensive,
                factor,
                max_shift=shift,
                target_vol=0.15,
            )
            candidates[(target, shift)] = (factor, path)
            metrics = performance_summary(path.loc[:CAL_END, "return"])
            rows.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "max_shift": shift,
                    "events": int(data.loc[:CAL_END, target].sum()),
                    **metrics.to_dict(),
                }
            )
    table = pd.DataFrame(rows)
    # The multi-objective goal is explicit: only candidates that improve all
    # three calibration objectives over the existing strategy are eligible.
    eligible = table[
        (table["CAGR"] >= float(reference_calibration["CAGR"]))
        & (table["Sharpe"] >= float(reference_calibration["Sharpe"]))
        & (table["MDD"] >= float(reference_calibration["MDD"]) - 1e-8)
    ]
    pool = eligible if not eligible.empty else table
    winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
    target = str(winner["target"])
    horizon = int(winner["horizon"])
    shift = float(winner["max_shift"])
    factor, path = candidates[(target, shift)]
    return target, horizon, shift, factor, path, table


def main() -> None:
    base_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    base_data.index = pd.PeriodIndex(base_data.index, freq="M")
    features = build_oap_features(base_data.index)
    composites = build_oap_composites(features)
    direct_trend_factor = make_direct_trend_factor(composites)
    features.to_csv(RESULTS / "openassetpricing_features.csv")
    composites.to_csv(RESULTS / "openassetpricing_composites.csv")
    direct_trend_factor.to_csv(RESULTS / "openassetpricing_direct_momentum_factor.csv")

    structure = pd.read_csv(RESULTS / "market_structure_composites.csv", index_col=0)
    structure.index = pd.PeriodIndex(structure.index, freq="M")
    core_structure = [
        "systemic_correlation_stress",
        "breadth_dispersion_stress",
        "volume_stress",
        "tail_shape_stress",
        "sector_linkage_stress",
    ]
    index_volume = ["index_volume_proxy_stress"]
    oap_columns = list(composites.columns)
    data = base_data.join(features, how="left").join(composites, how="left").join(
        structure, how="left"
    )

    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")
    neutral = pd.DataFrame({"p_up": 0.5}, index=data.index)
    baseline = run_factor_blend(
        asset_returns, signals, defensive, neutral, FactorBlendConfig(max_shift=0.0)
    )
    data = data.loc[data.index.intersection(baseline.index)].copy()
    baseline_return = baseline.loc[data.index, "return"]
    data["final_loss3"] = (baseline_return < -0.03).astype(int)
    medium_targets = [
        ("oap_path_loss_2m_4", 2, -0.04),
        ("oap_path_loss_2m_5", 2, -0.05),
        ("oap_path_loss_3m_5", 3, -0.05),
        ("oap_path_loss_3m_6", 3, -0.06),
    ]
    for target, horizon, threshold in medium_targets:
        path_loss = forward_path_loss(baseline_return, horizon)
        data[target] = (path_loss < threshold).where(path_loss.notna()).astype(float)

    specifications = {
        "oap_momentum_domestic": DOMESTIC_FEATURES + ["oap_momentum_trend_stress"],
        "oap_reversal_domestic": DOMESTIC_FEATURES + ["oap_reversal_crowding_stress"],
        "oap_lowrisk_domestic": DOMESTIC_FEATURES + ["oap_low_risk_tail_stress"],
        "oap_liquidity_domestic": DOMESTIC_FEATURES + ["oap_liquidity_activity_stress"],
        "oap_all_domestic": DOMESTIC_FEATURES + oap_columns,
        "structure_plus_oap_domestic": DOMESTIC_FEATURES + core_structure + oap_columns,
        "structure_indexvolume_plus_oap_domestic": (
            DOMESTIC_FEATURES + core_structure + index_volume + oap_columns
        ),
    }
    factors: dict[str, pd.DataFrame] = {}
    for name, selected in specifications.items():
        probability = walk_forward_probability(data, "final_loss3", selected)
        factor = make_factor(probability, data["final_loss3"], name)
        factors[name] = factor
        factor.to_csv(RESULTS / f"openassetpricing_{name}_factor.csv")

    baseline_calibration = performance_summary(baseline.loc[:CAL_END, "return"])
    winners: dict[str, tuple[float, pd.DataFrame]] = {}
    calibration_rows: list[pd.DataFrame] = []
    for name, factor in factors.items():
        shift, path, table = choose_shift(
            factor, asset_returns, signals, defensive, baseline_calibration
        )
        winners[name] = (shift, path)
        table.insert(0, "Strategy", name)
        calibration_rows.append(table)
        path.to_csv(RESULTS / f"openassetpricing_{name}_backtest.csv")

    standalone_names = [name for name in specifications if name.startswith("oap_")]
    standalone_selection = []
    for name in standalone_names:
        shift, path = winners[name]
        metrics = performance_summary(path.loc[:CAL_END, "return"])
        standalone_selection.append(
            {"Strategy": name, "max_shift": shift, **metrics.to_dict()}
        )
    oap_winner = (
        pd.DataFrame(standalone_selection)
        .sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False)
        .iloc[0]["Strategy"]
    )
    oap_factor = factors[str(oap_winner)]

    existing_structure_factor = load_factor("loss3_composite_domestic")
    existing_proxy_factor = load_factor("loss3_composite_plus_index_volume_domestic")
    existing_structure_vol = run_factor_vol_target(
        asset_returns,
        signals,
        defensive,
        existing_structure_factor,
        max_shift=0.15,
        target_vol=0.15,
    )
    existing_proxy_vol = run_factor_vol_target(
        asset_returns,
        signals,
        defensive,
        existing_proxy_factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    existing_structure_calibration = performance_summary(
        existing_structure_vol.loc[:CAL_END, "return"]
    )
    existing_proxy_calibration = performance_summary(existing_proxy_vol.loc[:CAL_END, "return"])

    (
        medium_target,
        medium_horizon,
        medium_shift,
        medium_factor,
        medium_path,
        medium_grid,
    ) = choose_medium_horizon_oap(
        data,
        DOMESTIC_FEATURES + oap_columns,
        [(target, horizon) for target, horizon, _ in medium_targets],
        asset_returns,
        signals,
        defensive,
        existing_structure_calibration,
    )
    medium_factor.to_csv(RESULTS / "openassetpricing_medium_horizon_factor.csv")
    medium_path.to_csv(RESULTS / "openassetpricing_medium_horizon_backtest.csv")

    blend_weight, blend_shift, blend_factor, blend_path, blend_grid = choose_signal_blend(
        existing_structure_factor,
        oap_factor,
        asset_returns,
        signals,
        defensive,
        existing_structure_calibration,
        0.15,
    )
    (
        proxy_blend_weight,
        proxy_blend_shift,
        proxy_blend_factor,
        proxy_blend_path,
        proxy_blend_grid,
    ) = choose_signal_blend(
        existing_proxy_factor,
        oap_factor,
        asset_returns,
        signals,
        defensive,
        existing_proxy_calibration,
        0.20,
    )
    blend_factor.to_csv(RESULTS / "openassetpricing_signal_blend_structure_factor.csv")
    blend_path.to_csv(RESULTS / "openassetpricing_signal_blend_structure_backtest.csv")
    proxy_blend_factor.to_csv(RESULTS / "openassetpricing_signal_blend_proxy_factor.csv")
    proxy_blend_path.to_csv(RESULTS / "openassetpricing_signal_blend_proxy_backtest.csv")

    (
        trend_structure_mode,
        trend_structure_weight,
        trend_structure_shift,
        trend_structure_factor,
        trend_structure_path,
        trend_structure_grid,
    ) = choose_risk_override_trend(
        existing_structure_factor,
        0.15,
        direct_trend_factor,
        asset_returns,
        signals,
        defensive,
        existing_structure_calibration,
    )
    (
        trend_proxy_mode,
        trend_proxy_weight,
        trend_proxy_shift,
        trend_proxy_factor,
        trend_proxy_path,
        trend_proxy_grid,
    ) = choose_risk_override_trend(
        existing_proxy_factor,
        0.20,
        direct_trend_factor,
        asset_returns,
        signals,
        defensive,
        existing_proxy_calibration,
    )
    trend_structure_factor.to_csv(
        RESULTS / "openassetpricing_trend_override_structure_factor.csv"
    )
    trend_structure_path.to_csv(
        RESULTS / "openassetpricing_trend_override_structure_backtest.csv"
    )
    trend_proxy_factor.to_csv(RESULTS / "openassetpricing_trend_override_proxy_factor.csv")
    trend_proxy_path.to_csv(
        RESULTS / "openassetpricing_trend_override_proxy_backtest.csv"
    )

    joint_structure_shift = winners["structure_plus_oap_domestic"][0]
    joint_structure_vol = run_factor_vol_target(
        asset_returns,
        signals,
        defensive,
        factors["structure_plus_oap_domestic"],
        max_shift=joint_structure_shift,
        target_vol=0.15,
    )
    joint_proxy_shift = winners["structure_indexvolume_plus_oap_domestic"][0]
    joint_proxy_vol = run_factor_vol_target(
        asset_returns,
        signals,
        defensive,
        factors["structure_indexvolume_plus_oap_domestic"],
        max_shift=joint_proxy_shift,
        target_vol=0.15,
    )
    (
        committee_method,
        committee_shift,
        committee_target_vol,
        committee_factor,
        committee_path,
        committee_grid,
    ) = choose_risk_committee(
        {
            "existing_structure": existing_structure_factor,
            "existing_index_volume": existing_proxy_factor,
            "joint_oap": factors["structure_indexvolume_plus_oap_domestic"],
        },
        asset_returns,
        signals,
        defensive,
        existing_structure_calibration,
    )
    committee_factor.to_csv(RESULTS / "openassetpricing_committee_factor.csv")
    committee_path.to_csv(RESULTS / "openassetpricing_committee_backtest.csv")

    standard: dict[str, pd.DataFrame] = {
        "FinalBlend": baseline,
        **{name: path for name, (_, path) in winners.items()},
        "ExistingStructureVol15": existing_structure_vol,
        "ExistingStructureIndexVolumeVol15": existing_proxy_vol,
        "JointStructureOAPVol15": joint_structure_vol,
        "JointStructureIndexVolumeOAPVol15": joint_proxy_vol,
        "SignalBlendStructureOAPVol15": blend_path,
        "SignalBlendStructureIndexVolumeOAPVol15": proxy_blend_path,
        "TrendOverrideStructureOAPVol15": trend_structure_path,
        "TrendOverrideStructureIndexVolumeOAPVol15": trend_proxy_path,
        "CalibratedOAPRiskCommittee": committee_path,
        "SelectedMediumHorizonOAPVol15": medium_path,
    }
    comparison_rows: list[dict[str, object]] = []
    periods = (
        ("calibration_2007_2017", baseline.index.min(), CAL_END),
        ("locked_2018_2026", TEST_START, baseline.index.max()),
        ("full_2007_2026", baseline.index.min(), baseline.index.max()),
        ("subperiod_2007_2012", baseline.index.min(), pd.Period("2012-12", freq="M")),
        ("subperiod_2013_2017", pd.Period("2013-01", freq="M"), CAL_END),
        ("subperiod_2018_2021", TEST_START, pd.Period("2021-12", freq="M")),
        ("subperiod_2022_2026", pd.Period("2022-01", freq="M"), baseline.index.max()),
    )
    for label, start, end in periods:
        for name, path in standard.items():
            comparison_rows.append(metric_record(label, name, path.loc[start:end]))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RESULTS / "openassetpricing_comparison.csv", index=False)

    cost_rows: list[dict[str, object]] = []
    cost_strategies = {
        "ExistingStructureVol15": (existing_structure_factor, 0.15, 0.15),
        "ExistingStructureIndexVolumeVol15": (existing_proxy_factor, 0.20, 0.15),
        "JointStructureOAPVol15": (
            factors["structure_plus_oap_domestic"],
            joint_structure_shift,
            0.15,
        ),
        "JointStructureIndexVolumeOAPVol15": (
            factors["structure_indexvolume_plus_oap_domestic"],
            joint_proxy_shift,
            0.15,
        ),
        "SignalBlendStructureOAPVol15": (blend_factor, blend_shift, 0.15),
        "SignalBlendStructureIndexVolumeOAPVol15": (
            proxy_blend_factor,
            proxy_blend_shift,
            0.15,
        ),
        "TrendOverrideStructureOAPVol15": (
            trend_structure_factor,
            trend_structure_shift,
            0.15,
        ),
        "TrendOverrideStructureIndexVolumeOAPVol15": (
            trend_proxy_factor,
            trend_proxy_shift,
            0.15,
        ),
        "CalibratedOAPRiskCommittee": (
            committee_factor,
            committee_shift,
            committee_target_vol,
        ),
        "SelectedMediumHorizonOAPVol15": (medium_factor, medium_shift, 0.15),
    }
    for cost in (0.5, 1.0, 2.0, 3.0):
        for name, (factor, shift, target_vol) in cost_strategies.items():
            path = run_factor_vol_target(
                asset_returns,
                signals,
                defensive,
                factor,
                max_shift=shift,
                target_vol=target_vol,
                cost_multiplier=cost,
            )
            cost_rows.append(metric_record(f"cost_{cost:.1f}x_full", name, path))
            cost_rows.append(
                metric_record(f"cost_{cost:.1f}x_locked", name, path.loc[TEST_START:])
            )
    pd.DataFrame(cost_rows).to_csv(
        RESULTS / "openassetpricing_cost_robustness.csv", index=False
    )

    selected_locked = {
        "JointStructureOAPVol15": joint_structure_vol.loc[TEST_START:, "return"],
        "JointStructureIndexVolumeOAPVol15": joint_proxy_vol.loc[TEST_START:, "return"],
        "SignalBlendStructureOAPVol15": blend_path.loc[TEST_START:, "return"],
        "SignalBlendStructureIndexVolumeOAPVol15": proxy_blend_path.loc[
            TEST_START:, "return"
        ],
        "TrendOverrideStructureOAPVol15": trend_structure_path.loc[
            TEST_START:, "return"
        ],
        "TrendOverrideStructureIndexVolumeOAPVol15": trend_proxy_path.loc[
            TEST_START:, "return"
        ],
        "CalibratedOAPRiskCommittee": committee_path.loc[TEST_START:, "return"],
        "SelectedMediumHorizonOAPVol15": medium_path.loc[TEST_START:, "return"],
    }
    bootstrap = {
        name: paired_block_bootstrap(
            existing_structure_vol.loc[TEST_START:, "return"], returns
        )
        for name, returns in selected_locked.items()
    }
    validation = {
        "source": OAP_SOURCE,
        "signal_map": OAP_SIGNAL_MAP,
        "selection": {
            "calibration_end": str(CAL_END),
            "locked_start": str(TEST_START),
            "standalone_oap_winner": str(oap_winner),
            "standalone_oap_shift": winners[str(oap_winner)][0],
            "joint_structure_shift": joint_structure_shift,
            "joint_proxy_shift": joint_proxy_shift,
            "signal_blend_structure_oap_weight": blend_weight,
            "signal_blend_structure_shift": blend_shift,
            "signal_blend_proxy_oap_weight": proxy_blend_weight,
            "signal_blend_proxy_shift": proxy_blend_shift,
            "trend_override_structure_mode": trend_structure_mode,
            "trend_override_structure_weight": trend_structure_weight,
            "trend_override_structure_shift": trend_structure_shift,
            "trend_override_proxy_mode": trend_proxy_mode,
            "trend_override_proxy_weight": trend_proxy_weight,
            "trend_override_proxy_shift": trend_proxy_shift,
            "committee_method": committee_method,
            "committee_shift": committee_shift,
            "committee_target_vol": committee_target_vol,
            "medium_horizon_target": medium_target,
            "medium_horizon_months": medium_horizon,
            "medium_horizon_shift": medium_shift,
        },
        "prediction": {
            name: {
                "calibration": predictive_metrics(factor.loc[:CAL_END]),
                "locked": predictive_metrics(factor, TEST_START),
            }
            for name, factor in factors.items()
        },
        "medium_horizon_prediction": {
            "calibration": predictive_metrics(medium_factor.loc[:CAL_END]),
            "locked": predictive_metrics(medium_factor, TEST_START),
        },
        "bootstrap_vs_existing_structure_vol15": bootstrap,
        "notes": [
            "All selection and tuning use data through 2017 only.",
            "2018-2026 is never used to select features, blend weights, or shift sizes.",
            "OAP signals are translated to KRX sector-index analogues; they are not exact stock-level replications.",
            "OAP Placebo/Dropped signals were excluded before testing.",
        ],
    }
    (RESULTS / "openassetpricing_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        RESULTS / "openassetpricing_calibration.csv", index=False
    )
    blend_grid.assign(existing="structure").to_csv(
        RESULTS / "openassetpricing_signal_blend_calibration.csv", index=False
    )
    proxy_blend_grid.assign(existing="structure_index_volume").to_csv(
        RESULTS / "openassetpricing_signal_blend_proxy_calibration.csv", index=False
    )
    trend_structure_grid.assign(existing="structure").to_csv(
        RESULTS / "openassetpricing_trend_override_calibration.csv", index=False
    )
    trend_proxy_grid.assign(existing="structure_index_volume").to_csv(
        RESULTS / "openassetpricing_trend_override_proxy_calibration.csv", index=False
    )
    committee_grid.to_csv(
        RESULTS / "openassetpricing_committee_calibration.csv", index=False
    )
    medium_grid.to_csv(
        RESULTS / "openassetpricing_medium_horizon_calibration.csv", index=False
    )

    print("\n=== OAP FEATURE COVERAGE ===")
    print(features.notna().mean().to_string(float_format=lambda value: f"{value:.3f}"))
    print("\n=== SELECTED SETTINGS ===")
    print(json.dumps(validation["selection"], ensure_ascii=False, indent=2))
    print("\n=== KEY COMPARISON ===")
    key_names = [
        "FinalBlend",
        str(oap_winner),
        "structure_plus_oap_domestic",
        "structure_indexvolume_plus_oap_domestic",
        "ExistingStructureVol15",
        "ExistingStructureIndexVolumeVol15",
        "JointStructureOAPVol15",
        "JointStructureIndexVolumeOAPVol15",
        "SignalBlendStructureOAPVol15",
        "SignalBlendStructureIndexVolumeOAPVol15",
        "TrendOverrideStructureOAPVol15",
        "TrendOverrideStructureIndexVolumeOAPVol15",
        "CalibratedOAPRiskCommittee",
        "SelectedMediumHorizonOAPVol15",
    ]
    print(
        comparison[
            comparison["Period"].isin(
                ["calibration_2007_2017", "locked_2018_2026", "full_2007_2026"]
            )
            & comparison["Strategy"].isin(key_names)
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
