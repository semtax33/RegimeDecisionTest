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

from strategies.core.regime_research import load_monthly_asset_returns
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    build_daily_stress_features,
    build_monthly_stress_signals,
)
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20,
)
from strategies.stage30_abnormal_surface_erp import (
    abnormal_surface_erp_slsqp as stage30,
)
from strategies.stage31_long_iv_state_dependence import (
    long_iv_state_dependence_slsqp as stage31,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
HORIZONS = (1, 3, 6)
DECILES = 10
CRISIS_YEARS = (1998, 2008, 2020)
RESEARCH_END = pd.Period("2026-07", freq="M")
CONTROL_COLUMNS = (
    "vix6_stress_score",
    "recent_1m_return",
    "realized_vol_21d",
    "macro_fragility",
)

FROZEN_STRATEGY_FILES = (
    stage20.__file__,
    stage30.__file__,
    stage31.__file__,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def source_manifest() -> dict[str, dict[str, Any]]:
    return _file_manifest([stage31.LONG_IV_XLSX, stage31.K200_FUTURES_XLSX])


def frozen_strategy_manifest() -> dict[str, dict[str, Any]]:
    return _file_manifest([Path(path) for path in FROZEN_STRATEGY_FILES])


def _forward_compound_returns(
    monthly_returns: pd.Series, horizon: int
) -> pd.Series:
    legs = [monthly_returns.shift(-offset) for offset in range(horizon)]
    frame = pd.concat(legs, axis=1)
    valid = frame.notna().all(axis=1)
    output = frame.add(1.0).prod(axis=1).sub(1.0)
    output = output.where(valid)
    return output.rename(f"future_{horizon}m_return")


def _realized_volatility_signal(
    futures_close: pd.Series,
) -> pd.Series:
    log_return = np.log(futures_close).diff()
    daily_vol = log_return.rolling(21, min_periods=15).std(ddof=1) * math.sqrt(
        252.0
    )
    monthly = daily_vol.groupby(daily_vol.index.to_period("M")).last()
    monthly.index = monthly.index + 1
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    return monthly.rename("realized_vol_21d")


def build_research_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_iv, iv_audit = stage31.load_long_iv_daily()
    bucket = stage31.build_monthly_bucket_signals(daily_iv)
    futures_close, futures_audit = stage31.load_k200_futures_close()
    monthly_return = stage31.build_monthly_futures_returns(futures_close)

    frame = bucket.copy()
    frame["fear_term_slope"] = (
        frame["wing_asym_near"] - frame["wing_asym_next"]
    )
    for horizon in HORIZONS:
        frame[f"future_{horizon}m_return"] = _forward_compound_returns(
            monthly_return, horizon
        )
    frame["recent_1m_return"] = monthly_return.shift(1)
    frame = frame.join(_realized_volatility_signal(futures_close), how="left")

    project_returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(project_returns)
    frame["macro_fragility"] = (
        probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    )
    daily_stress = build_daily_stress_features()
    stress = build_monthly_stress_signals(frame.index, daily_stress)
    frame["vix6_stress_score"] = stress["stress_score"]
    frame = frame.loc[:RESEARCH_END].replace([np.inf, -np.inf], np.nan)

    audit = {
        "iv": iv_audit,
        "futures": futures_audit,
        "first_target_month": str(frame.index.min()),
        "last_target_month": str(frame.index.max()),
        "monthly_rows": int(len(frame)),
        "primary_signal": "wing_asym_near = near_put_otm2 - near_call_otm2",
        "only_secondary_signal": (
            "fear_term_slope = wing_asym_near - wing_asym_next"
        ),
        "horizons_months": list(HORIZONS),
        "deciles": DECILES,
        "realized_volatility": (
            "annualized standard deviation of the prior 21 daily log returns; "
            "minimum 15 observations"
        ),
        "vix6_control": "frozen Stage20 VIX6/VKOSPI stress_score",
        "crisis_years": list(CRISIS_YEARS),
        "winsorization": False,
        "parameter_or_bucket_search": False,
    }
    return frame, audit


def _stable_decile(series: pd.Series) -> pd.Series:
    valid = series.dropna().sort_index()
    ranks = valid.rank(method="first")
    assigned = pd.qcut(ranks, DECILES, labels=range(1, DECILES + 1)).astype(
        int
    )
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[assigned.index] = assigned.astype(float)
    return output


def _hac_univariate(
    signal: pd.Series,
    target: pd.Series,
    horizon: int,
    label: str,
    sample: str,
) -> dict[str, Any]:
    data = pd.concat(
        [signal.rename("signal"), target.rename("target")], axis=1
    ).dropna()
    x = (data["signal"] - data["signal"].mean()) / data["signal"].std(
        ddof=0
    )
    fit = sm.OLS(data["target"], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )
    return {
        "Signal": label,
        "Sample": sample,
        "HorizonMonths": horizon,
        "Observations": int(len(data)),
        "SpearmanIC": float(
            data["signal"].corr(data["target"], method="spearman")
        ),
        "StandardizedBeta": float(fit.params["signal"]),
        "HACBetaSE": float(fit.bse["signal"]),
        "HACBetaT": float(fit.tvalues["signal"]),
        "HACBetaPValue": float(fit.pvalues["signal"]),
        "RSquared": float(fit.rsquared),
    }


def decile_analysis(
    frame: pd.DataFrame, signal_column: str, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decile = _stable_decile(frame[signal_column])
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        target_column = f"future_{horizon}m_return"
        complete = pd.concat(
            [
                frame[signal_column].rename("signal"),
                decile.rename("decile"),
                frame[target_column].rename("target"),
            ],
            axis=1,
        ).dropna()
        for number in range(1, DECILES + 1):
            group = complete.loc[complete["decile"].eq(number), "target"]
            mean_return = float(group.mean())
            annualized_equivalent = (
                (1.0 + mean_return) ** (12.0 / horizon) - 1.0
                if mean_return > -1.0
                else float("nan")
            )
            detail_rows.append(
                {
                    "Signal": label,
                    "HorizonMonths": horizon,
                    "Decile": number,
                    "Observations": int(len(group)),
                    "MeanForwardReturn": mean_return,
                    "MedianForwardReturn": float(group.median()),
                    "PositiveReturnShare": float((group > 0.0).mean()),
                    "AnnualizedEquivalentOfMean": annualized_equivalent,
                    "MeanSignal": float(
                        complete.loc[
                            complete["decile"].eq(number), "signal"
                        ].mean()
                    ),
                }
            )
        horizon_rows = [
            row
            for row in detail_rows
            if row["Signal"] == label and row["HorizonMonths"] == horizon
        ]
        means = np.asarray(
            [row["MeanForwardReturn"] for row in horizon_rows], dtype=float
        )
        rank_result = spearmanr(np.arange(1, DECILES + 1), means)
        tails = complete.loc[complete["decile"].isin([1.0, 10.0])].copy()
        tails["top_decile"] = tails["decile"].eq(10.0).astype(float)
        tail_fit = sm.OLS(
            tails["target"], sm.add_constant(tails["top_decile"])
        ).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
        summary_rows.append(
            {
                "Signal": label,
                "HorizonMonths": horizon,
                "Observations": int(len(complete)),
                "DecileMeanSpearman": float(rank_result.statistic),
                "DecileMeanSpearmanPValue": float(rank_result.pvalue),
                "StrictlyMonotoneIncreasing": bool(
                    np.all(np.diff(means) > 0.0)
                ),
                "PositiveStepShare": float(np.mean(np.diff(means) > 0.0)),
                "Q10MinusQ1MeanReturn": float(means[-1] - means[0]),
                "Q10MinusQ1HACPValue": float(
                    tail_fit.pvalues["top_decile"]
                ),
                "Q10MeanReturn": float(means[-1]),
                "Q1MeanReturn": float(means[0]),
                "Q10MinusQ1To9MeanReturn": float(
                    means[-1] - np.mean(means[:-1])
                ),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def univariate_horizon_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column in [
        ("WingAsym_Near_OTM2", "wing_asym_near"),
        ("FearTermSlope", "fear_term_slope"),
    ]:
        for horizon in HORIZONS:
            rows.append(
                _hac_univariate(
                    frame[column],
                    frame[f"future_{horizon}m_return"],
                    horizon,
                    label,
                    "full_available",
                )
            )
    return pd.DataFrame(rows)


def control_availability_era_tests(frame: pd.DataFrame) -> pd.DataFrame:
    periods = [
        (
            "pre_frozen_controls_1997_08_2006_03",
            pd.Period("1997-08", "M"),
            pd.Period("2006-03", "M"),
        ),
        (
            "frozen_controls_available_2006_04_2026_07",
            pd.Period("2006-04", "M"),
            RESEARCH_END,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for sample, start, end in periods:
        for label, column in [
            ("WingAsym_Near_OTM2", "wing_asym_near"),
            ("FearTermSlope", "fear_term_slope"),
        ]:
            for horizon in HORIZONS:
                rows.append(
                    _hac_univariate(
                        frame.loc[start:end, column],
                        frame.loc[start:end, f"future_{horizon}m_return"],
                        horizon,
                        label,
                        sample,
                    )
                )
    return pd.DataFrame(rows)


def controlled_regression_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, signal_column in [
        ("WingAsym_Near_OTM2", "wing_asym_near"),
        ("FearTermSlope", "fear_term_slope"),
    ]:
        for horizon in HORIZONS:
            target_column = f"future_{horizon}m_return"
            columns = [signal_column, *CONTROL_COLUMNS, target_column]
            data = frame[columns].dropna().copy()
            standardized = data.copy()
            predictor_columns = [signal_column, *CONTROL_COLUMNS]
            for column in predictor_columns:
                scale = float(data[column].std(ddof=0))
                standardized[column] = (
                    data[column] - data[column].mean()
                ) / scale
            full_design = sm.add_constant(standardized[predictor_columns])
            control_design = sm.add_constant(
                standardized[list(CONTROL_COLUMNS)]
            )
            full_fit = sm.OLS(data[target_column], full_design).fit(
                cov_type="HAC", cov_kwds={"maxlags": horizon}
            )
            control_fit = sm.OLS(data[target_column], control_design).fit(
                cov_type="HAC", cov_kwds={"maxlags": horizon}
            )

            signal_residual = sm.OLS(
                standardized[signal_column], control_design
            ).fit().resid
            target_residual = sm.OLS(
                data[target_column], control_design
            ).fit().resid
            rows.append(
                {
                    "Signal": label,
                    "HorizonMonths": horizon,
                    "Observations": int(len(data)),
                    "FirstMonth": str(data.index.min()),
                    "LastMonth": str(data.index.max()),
                    "ControlledStandardizedBeta": float(
                        full_fit.params[signal_column]
                    ),
                    "HACBetaSE": float(full_fit.bse[signal_column]),
                    "HACBetaT": float(full_fit.tvalues[signal_column]),
                    "HACBetaPValue": float(full_fit.pvalues[signal_column]),
                    "PartialPearsonCorrelation": float(
                        signal_residual.corr(target_residual)
                    ),
                    "FullRSquared": float(full_fit.rsquared),
                    "ControlOnlyRSquared": float(control_fit.rsquared),
                    "IncrementalRSquared": float(
                        full_fit.rsquared - control_fit.rsquared
                    ),
                    **{
                        f"Beta_{column}": float(full_fit.params[column])
                        for column in CONTROL_COLUMNS
                    },
                }
            )
    return pd.DataFrame(rows)


def sequential_control_ladder(frame: pd.DataFrame) -> pd.DataFrame:
    """Show exactly which economically ordered control absorbs the signal."""

    steps = [
        ("WingOnly_CommonSample", []),
        ("Plus_VIX6", ["vix6_stress_score"]),
        (
            "Plus_RecentReturn",
            ["vix6_stress_score", "recent_1m_return"],
        ),
        (
            "Plus_RealizedVol",
            [
                "vix6_stress_score",
                "recent_1m_return",
                "realized_vol_21d",
            ],
        ),
        ("Plus_MacroFragility_AllControls", list(CONTROL_COLUMNS)),
    ]
    rows: list[dict[str, Any]] = []
    for label, signal_column in [
        ("WingAsym_Near_OTM2", "wing_asym_near"),
        ("FearTermSlope", "fear_term_slope"),
    ]:
        for horizon in HORIZONS:
            target_column = f"future_{horizon}m_return"
            # Keep the sample identical across ladder steps.
            columns = [signal_column, *CONTROL_COLUMNS, target_column]
            data = frame[columns].dropna().copy()
            standardized = data.copy()
            for column in [signal_column, *CONTROL_COLUMNS]:
                standardized[column] = (
                    data[column] - data[column].mean()
                ) / data[column].std(ddof=0)
            for step, controls in steps:
                predictors = [signal_column, *controls]
                design = sm.add_constant(standardized[predictors])
                fit = sm.OLS(data[target_column], design).fit(
                    cov_type="HAC", cov_kwds={"maxlags": horizon}
                )
                rows.append(
                    {
                        "Signal": label,
                        "HorizonMonths": horizon,
                        "ControlStep": step,
                        "Controls": ",".join(controls) if controls else "None",
                        "Observations": int(len(data)),
                        "StandardizedSignalBeta": float(
                            fit.params[signal_column]
                        ),
                        "HACBetaSE": float(fit.bse[signal_column]),
                        "HACBetaT": float(fit.tvalues[signal_column]),
                        "HACBetaPValue": float(fit.pvalues[signal_column]),
                        "RSquared": float(fit.rsquared),
                    }
                )
    return pd.DataFrame(rows)


def control_correlation_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["wing_asym_near", "fear_term_slope", *CONTROL_COLUMNS]
    data = frame[columns].dropna()
    correlation = data.corr(method="spearman")
    rows: list[dict[str, Any]] = []
    for signal in ["wing_asym_near", "fear_term_slope"]:
        for control in CONTROL_COLUMNS:
            rows.append(
                {
                    "Signal": signal,
                    "Control": control,
                    "Observations": int(len(data)),
                    "SpearmanCorrelation": float(
                        correlation.loc[signal, control]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _holding_window_overlaps_year(
    month: pd.Period, horizon: int, years: tuple[int, ...]
) -> bool:
    return any((month + offset).year in years for offset in range(horizon))


def crisis_exclusion_tests(frame: pd.DataFrame) -> pd.DataFrame:
    decile = _stable_decile(frame["wing_asym_near"])
    scenarios: list[tuple[str, tuple[int, ...]]] = [
        ("None", ()),
        ("Exclude1998", (1998,)),
        ("Exclude2008", (2008,)),
        ("Exclude2020", (2020,)),
        ("ExcludeAll_1998_2008_2020", CRISIS_YEARS),
    ]
    rows: list[dict[str, Any]] = []
    for scenario, excluded_years in scenarios:
        for horizon in HORIZONS:
            target = frame[f"future_{horizon}m_return"]
            base = pd.concat(
                [
                    frame["wing_asym_near"].rename("signal"),
                    decile.rename("decile"),
                    target.rename("target"),
                ],
                axis=1,
            ).dropna()
            keep = pd.Series(
                [
                    not _holding_window_overlaps_year(
                        month, horizon, excluded_years
                    )
                    for month in base.index
                ],
                index=base.index,
            )
            data = base.loc[keep]
            metrics = _hac_univariate(
                data["signal"],
                data["target"],
                horizon,
                "WingAsym_Near_OTM2",
                scenario,
            )
            q1 = data.loc[data["decile"].eq(1.0), "target"]
            q10 = data.loc[data["decile"].eq(10.0), "target"]
            metrics.update(
                {
                    "ExcludedYears": ",".join(map(str, excluded_years))
                    if excluded_years
                    else "None",
                    "RemovedObservations": int(len(base) - len(data)),
                    "Q1Observations": int(len(q1)),
                    "Q10Observations": int(len(q10)),
                    "Q1MeanReturn": float(q1.mean()),
                    "Q10MeanReturn": float(q10.mean()),
                    "Q10MinusQ1MeanReturn": float(q10.mean() - q1.mean()),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def causal_expanding_decile(signal: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=signal.index, dtype=float)
    for month in signal.index:
        current = signal.loc[month]
        history = signal.loc[signal.index < month].dropna()
        if pd.isna(current) or len(history) < DECILES:
            continue
        percentile = float(
            ((history < current).sum() + 0.5 * (history == current).sum())
            / len(history)
        )
        output.loc[month] = float(
            np.clip(math.ceil(percentile * DECILES), 1, DECILES)
        )
    return output


def _return_path_metrics(returns: pd.Series, label: str) -> dict[str, Any]:
    values = returns.dropna().astype(float)
    nav = (1.0 + values).cumprod()
    years = len(values) / 12.0
    cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(values.std(ddof=1) * math.sqrt(12.0))
    sharpe = (
        float(values.mean() / values.std(ddof=1) * math.sqrt(12.0))
        if values.std(ddof=1) > 0.0
        else float("nan")
    )
    drawdown = nav / nav.cummax() - 1.0
    invested = values.loc[values.ne(0.0)]
    return {
        "DiagnosticPath": label,
        "Start": str(values.index.min()),
        "End": str(values.index.max()),
        "Months": int(len(values)),
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "MDD": float(drawdown.min()),
        "FinalMultiple": float(nav.iloc[-1]),
        "PositiveMonths": float((values > 0.0).mean()),
        "InvestedMonths": int((values != 0.0).sum()),
        "ExposureShare": float((values != 0.0).mean()),
        "ConditionalMeanInvestedMonthReturn": float(invested.mean())
        if len(invested)
        else float("nan"),
        "ConditionalPositiveInvestedMonthShare": float((invested > 0.0).mean())
        if len(invested)
        else float("nan"),
        "TransactionCostsIncluded": False,
        "Interpretation": "mechanism diagnostic, not a promoted strategy",
    }


def causal_decile_timing_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    decile = causal_expanding_decile(frame["wing_asym_near"])
    returns = frame["future_1m_return"]
    common = decile.dropna().index.intersection(returns.dropna().index)
    decile = decile.loc[common]
    returns = returns.loc[common]
    paths = {
        "K200_buy_and_hold_same_sample": returns,
        "WingAsym_causal_Q10_only_else_cash": returns.where(
            decile.eq(10.0), 0.0
        ),
        "WingAsym_causal_Q1_only_else_cash": returns.where(
            decile.eq(1.0), 0.0
        ),
        "WingAsym_causal_Q10_long_Q1_short": pd.Series(
            np.select(
                [decile.eq(10.0), decile.eq(1.0)],
                [returns, -returns],
                default=0.0,
            ),
            index=common,
        ),
    }
    return pd.DataFrame(
        [_return_path_metrics(path, name) for name, path in paths.items()]
    )


def gate_decision(
    wing_decile_summary: pd.DataFrame,
    univariate: pd.DataFrame,
    controlled: pd.DataFrame,
    crisis: pd.DataFrame,
) -> tuple[dict[str, bool], bool]:
    decile_1m = wing_decile_summary.loc[
        wing_decile_summary["HorizonMonths"].eq(1)
    ].iloc[0]
    wing_univariate = univariate.loc[
        univariate["Signal"].eq("WingAsym_Near_OTM2")
    ].set_index("HorizonMonths")
    wing_controlled = controlled.loc[
        controlled["Signal"].eq("WingAsym_Near_OTM2")
    ].set_index("HorizonMonths")
    crisis_all_1m = crisis.loc[
        crisis["Sample"].eq("ExcludeAll_1998_2008_2020")
        & crisis["HorizonMonths"].eq(1)
    ].iloc[0]
    gates = {
        "one_month_decile_monotonicity": bool(
            decile_1m["DecileMeanSpearman"] > 0.0
            and decile_1m["DecileMeanSpearmanPValue"] < 0.10
            and decile_1m["Q10MinusQ1MeanReturn"] > 0.0
        ),
        "positive_ic_at_1_3_6m_and_two_significant_horizons": bool(
            (wing_univariate["SpearmanIC"] > 0.0).all()
            and (
                (wing_univariate["StandardizedBeta"] > 0.0)
                & (wing_univariate["HACBetaPValue"] < 0.10)
            ).sum()
            >= 2
        ),
        "one_month_survives_all_controls": bool(
            wing_controlled.loc[1, "ControlledStandardizedBeta"] > 0.0
            and wing_controlled.loc[1, "HACBetaPValue"] < 0.10
        ),
        "one_month_survives_all_crisis_exclusions": bool(
            crisis_all_1m["SpearmanIC"] > 0.0
            and crisis_all_1m["Q10MinusQ1MeanReturn"] > 0.0
        ),
    }
    return gates, bool(all(gates.values()))


def _plot_decile_curves(
    wing_detail: pd.DataFrame,
    term_detail: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for axis, data, title in [
        (axes[0], wing_detail, "WingAsym decile forward returns"),
        (axes[1], term_detail, "FearTermSlope decile forward returns"),
    ]:
        for horizon in HORIZONS:
            subset = data.loc[data["HorizonMonths"].eq(horizon)]
            axis.plot(
                subset["Decile"],
                subset["MeanForwardReturn"] * 100.0,
                marker="o",
                label=f"{horizon}m",
            )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_xticks(range(1, DECILES + 1))
        axis.set_xlabel("Full-sample decile")
        axis.set_ylabel("Mean forward return (%)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_robustness(
    controlled: pd.DataFrame,
    crisis: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    wing = controlled.loc[controlled["Signal"].eq("WingAsym_Near_OTM2")]
    axes[0].errorbar(
        wing["HorizonMonths"],
        wing["ControlledStandardizedBeta"] * 100.0,
        yerr=1.645 * wing["HACBetaSE"] * 100.0,
        marker="o",
        capsize=4,
    )
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(HORIZONS)
    axes[0].set_title("Controlled WingAsym beta (90% HAC interval)")
    axes[0].set_xlabel("Horizon months")
    axes[0].set_ylabel("Return per 1σ WingAsym (%)")
    axes[0].grid(alpha=0.2)

    crisis_1m = crisis.loc[crisis["HorizonMonths"].eq(1)]
    axes[1].bar(
        crisis_1m["Sample"], crisis_1m["SpearmanIC"], color="#2563a6"
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("1m WingAsym IC after crisis exclusions")
    axes[1].set_ylabel("Spearman IC")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_strategy_manifest()
    frame, audit = build_research_frame()

    wing_detail, wing_summary = decile_analysis(
        frame, "wing_asym_near", "WingAsym_Near_OTM2"
    )
    term_detail, term_summary = decile_analysis(
        frame, "fear_term_slope", "FearTermSlope"
    )
    univariate = univariate_horizon_tests(frame)
    era_tests = control_availability_era_tests(frame)
    controlled = controlled_regression_tests(frame)
    control_ladder = sequential_control_ladder(frame)
    correlations = control_correlation_diagnostics(frame)
    crisis = crisis_exclusion_tests(frame)
    timing = causal_decile_timing_diagnostic(frame)
    gates, overlay_eligible = gate_decision(
        wing_summary, univariate, controlled, crisis
    )

    stage20_performance = pd.read_csv(
        stage31.OUTPUT_DIR / "performance_comparison.csv"
    )
    stage20_performance = stage20_performance.loc[
        stage20_performance["Strategy"].eq("Stage20_VIX6")
    ].copy()

    source_after = source_manifest()
    frozen_after = frozen_strategy_manifest()
    source_unchanged = source_before == source_after
    frozen_unchanged = frozen_before == frozen_after
    checks = {
        "source_xlsx_files_unchanged": source_unchanged,
        "stage20_stage30_stage31_files_unchanged": frozen_unchanged,
        "only_otm2_primary_bucket_used": True,
        "only_fixed_1_3_6m_horizons_used": True,
        "exactly_ten_deciles_used": DECILES == 10,
        "only_one_secondary_term_signal_used": True,
        "no_winsorization": True,
        "no_parameter_bucket_or_window_search": True,
        "signals_precede_target_month": bool(
            (frame["bucket_signal_month"] < frame.index).all()
        ),
        "controls_are_observable_by_target_start": True,
        "crisis_years_are_exactly_feedback_years": CRISIS_YEARS
        == (1998, 2008, 2020),
        "stage20_overlay_not_required_because_gate_failed": bool(
            not overlay_eligible
        ),
    }
    decision = (
        "fear_premium_passes_stage33_overlay_research_gate"
        if overlay_eligible
        else "fear_premium_fails_overlay_gate_keep_stage20_frozen"
    )
    report = {
        "study": "Stage32_FearPremiumValidation",
        "decision": decision,
        "overlay_eligible": overlay_eligible,
        "scope": (
            "Mechanism validation only. Stage20, Stage30, and Stage31 are "
            "frozen. No expected-return alpha or risk overlay is added."
        ),
        "fixed_design": {
            "primary_signal": "near PUT OTM2 IV - near CALL OTM2 IV",
            "secondary_signal": "near WingAsym - next WingAsym",
            "deciles": DECILES,
            "horizons_months": list(HORIZONS),
            "controls": list(CONTROL_COLUMNS),
            "crisis_years": list(CRISIS_YEARS),
            "hac_lags": "equal to forward-return horizon in months",
            "significance_level": 0.10,
            "searched_parameters": None,
        },
        "overlay_gate_rules": {
            "decile": (
                "1m decile-mean Spearman > 0 with p<10%, and Q10-Q1 > 0"
            ),
            "horizon": (
                "IC positive at 1/3/6m and positive HAC beta with p<10% "
                "at two or more horizons"
            ),
            "controls": (
                "1m controlled WingAsym beta positive with p<10%"
            ),
            "crisis": (
                "1m IC and Q10-Q1 remain positive after excluding holding "
                "windows overlapping 1998, 2008, or 2020"
            ),
        },
        "gate_results": gates,
        "source_audit": audit,
        "wing_decile_summary": json.loads(
            wing_summary.to_json(orient="records", force_ascii=False)
        ),
        "term_slope_decile_summary": json.loads(
            term_summary.to_json(orient="records", force_ascii=False)
        ),
        "univariate_horizon_tests": json.loads(
            univariate.to_json(orient="records", force_ascii=False)
        ),
        "control_availability_era_tests": json.loads(
            era_tests.to_json(orient="records", force_ascii=False)
        ),
        "controlled_regression_tests": json.loads(
            controlled.to_json(orient="records", force_ascii=False)
        ),
        "sequential_control_ladder": json.loads(
            control_ladder.to_json(orient="records", force_ascii=False)
        ),
        "control_correlations": json.loads(
            correlations.to_json(orient="records", force_ascii=False)
        ),
        "crisis_exclusion_tests": json.loads(
            crisis.to_json(orient="records", force_ascii=False)
        ),
        "causal_decile_timing_diagnostic": json.loads(
            timing.to_json(orient="records", force_ascii=False)
        ),
        "frozen_stage20_performance": json.loads(
            stage20_performance.to_json(orient="records", force_ascii=False)
        ),
        "checks": checks,
        "source_files_unchanged": source_unchanged,
        "frozen_strategies_unchanged": frozen_unchanged,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_strategy_manifest_before": frozen_before,
        "frozen_strategy_manifest_after": frozen_after,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUTPUT_DIR / "monthly_fear_premium_research_frame.csv")
        pd.concat([wing_detail, term_detail], ignore_index=True).to_csv(
            OUTPUT_DIR / "decile_forward_returns.csv", index=False
        )
        pd.concat([wing_summary, term_summary], ignore_index=True).to_csv(
            OUTPUT_DIR / "decile_monotonicity_summary.csv", index=False
        )
        univariate.to_csv(
            OUTPUT_DIR / "horizon_univariate_hac.csv", index=False
        )
        era_tests.to_csv(
            OUTPUT_DIR / "control_availability_era_tests.csv", index=False
        )
        controlled.to_csv(
            OUTPUT_DIR / "controlled_predictive_regressions.csv", index=False
        )
        control_ladder.to_csv(
            OUTPUT_DIR / "sequential_control_ladder.csv", index=False
        )
        correlations.to_csv(
            OUTPUT_DIR / "signal_control_correlations.csv", index=False
        )
        crisis.to_csv(
            OUTPUT_DIR / "crisis_exclusion_robustness.csv", index=False
        )
        timing.to_csv(
            OUTPUT_DIR / "causal_decile_timing_diagnostic.csv", index=False
        )
        stage20_performance.to_csv(
            OUTPUT_DIR / "frozen_stage20_performance.csv", index=False
        )
        _plot_decile_curves(
            wing_detail,
            term_detail,
            OUTPUT_DIR / "decile_forward_return_curves.png",
        )
        _plot_robustness(
            controlled,
            crisis,
            OUTPUT_DIR / "control_and_crisis_robustness.png",
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return {
        "frame": frame,
        "wing_decile_detail": wing_detail,
        "wing_decile_summary": wing_summary,
        "term_decile_detail": term_detail,
        "term_decile_summary": term_summary,
        "univariate": univariate,
        "era_tests": era_tests,
        "controlled": controlled,
        "control_ladder": control_ladder,
        "correlations": correlations,
        "crisis": crisis,
        "timing": timing,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["wing_decile_summary"].to_string(index=False))
    print(result["term_decile_summary"].to_string(index=False))
    print(result["univariate"].to_string(index=False))
    print(result["era_tests"].to_string(index=False))
    print(result["controlled"].to_string(index=False))
    print(result["control_ladder"].to_string(index=False))
    print(result["crisis"].to_string(index=False))
    print(result["timing"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
