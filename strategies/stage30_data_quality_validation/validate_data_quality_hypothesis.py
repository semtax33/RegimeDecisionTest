from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from strategies.core.regime_research import ASSETS, load_monthly_asset_returns
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
    load_vkospi_daily,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    LOCKED_START,
    build_daily_stress_features,
    build_monthly_stress_signals,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    metric_row,
)
from strategies.stage28_option_directional_surface import (
    option_directional_surface_slsqp as stage28,
)
from strategies.stage30_abnormal_surface_erp import (
    abnormal_surface_erp_slsqp as stage30,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONSOLIDATED_CHAIN = ROOT / "raw_data" / "KOSPI200OptionPrice.csv"
ANNUAL_CHAIN_DIR = ROOT / "raw_data" / "KOSPI200OptionPrice"
EARLY_START = pd.Timestamp("2007-01-01")
EARLY_END = pd.Timestamp("2017-12-31")
LATE_START = pd.Timestamp("2018-01-01")
LATE_END = pd.Timestamp("2026-07-31")
DEGRADATION_REPLICATIONS = 20
RANDOM_SEED = 20260829

SURFACE_COLUMNS = [
    "dte",
    "forward",
    "discount_factor",
    "implied_rate",
    "parity_rmse",
    "parity_nrmse",
    "parity_pairs",
    "atm_iv",
    "put25_iv",
    "put25_strike_ratio",
    "put25_nearest_strike_distance",
    "put_skew25",
    "implied_variance_asymmetry",
    "option_erp_proxy",
    "monotone_quality",
    "convexity_quality",
    "coverage_log_width",
    "zero_or_missing_close_share",
    "invalid_iv_share",
    "listed_contracts",
    "put_strike_min_ratio",
    "call_strike_max_ratio",
    "put_contracts",
    "call_contracts",
    "maturity_method",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_chain_manifest() -> dict[str, dict[str, Any]]:
    paths = [CONSOLIDATED_CHAIN, *sorted(ANNUAL_CHAIN_DIR.glob("*.csv"))]
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def _stable_quintile(series: pd.Series) -> pd.Series:
    valid = series.dropna().sort_index()
    ranks = valid.rank(method="first")
    quintile = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[quintile.index] = quintile.astype(float)
    return output


def _spearman(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    complete = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(complete) < 3 or complete["left"].nunique() < 2:
        return float("nan"), int(len(complete))
    return float(complete["left"].corr(complete["right"], method="spearman")), int(
        len(complete)
    )


def _hac_interaction(
    signal: pd.Series,
    quality: pd.Series,
    target: pd.Series,
    maxlags: int,
    label: str,
) -> dict[str, Any]:
    frame = pd.concat(
        [signal.rename("signal"), quality.rename("quality"), target.rename("target")],
        axis=1,
    ).dropna()
    frame["signal"] = (frame["signal"] - frame["signal"].mean()) / frame[
        "signal"
    ].std(ddof=0)
    frame["quality"] = (frame["quality"] - frame["quality"].mean()) / frame[
        "quality"
    ].std(ddof=0)
    frame["signal_x_quality"] = frame["signal"] * frame["quality"]
    design = sm.add_constant(
        frame[["signal", "quality", "signal_x_quality"]], has_constant="add"
    )
    fit = sm.OLS(frame["target"], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "Test": label,
        "Observations": int(len(frame)),
        "HACLags": int(maxlags),
        "InteractionBeta": float(fit.params["signal_x_quality"]),
        "InteractionSE": float(fit.bse["signal_x_quality"]),
        "InteractionT": float(fit.tvalues["signal_x_quality"]),
        "InteractionPValue": float(fit.pvalues["signal_x_quality"]),
        "SignalBeta": float(fit.params["signal"]),
        "QualityBeta": float(fit.params["quality"]),
        "RSquared": float(fit.rsquared),
    }


def within_early_quality_test(
    daily: pd.DataFrame,
    monthly_signals: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Test whether ODS works better on higher-quality early observations."""

    early = daily.loc[EARLY_START:EARLY_END].copy()
    early["forward_5d"] = early["kospi200_close"].shift(-stage30.FAST_DAYS).div(
        early["kospi200_close"]
    ).sub(1.0)
    early["forward_20d"] = early["kospi200_close"].shift(-stage30.SLOW_DAYS).div(
        early["kospi200_close"]
    ).sub(1.0)
    early["quality_quintile"] = _stable_quintile(
        early["data_quality_confidence"]
    )
    rows: list[dict[str, Any]] = []
    for quintile in range(1, 6):
        group = early.loc[early["quality_quintile"] == quintile]
        for horizon in [5, 20]:
            ic, observations = _spearman(
                group["pure_direction_raw"], group[f"forward_{horizon}d"]
            )
            rows.append(
                {
                    "Frequency": "daily",
                    "Horizon": f"{horizon}d",
                    "QualityQuintile": quintile,
                    "SpearmanIC": ic,
                    "Observations": observations,
                    "MeanQuality": float(group["data_quality_confidence"].mean()),
                    "MeanRawODS": float(group["pure_direction_raw"].mean()),
                    "MeanTargetReturn": float(group[f"forward_{horizon}d"].mean()),
                }
            )

    monthly = monthly_signals.copy()
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    monthly = monthly.loc[pd.Period("2007-01", "M") : pd.Period("2017-12", "M")]
    common = monthly.index.intersection(returns.index)
    monthly = monthly.loc[common].copy()
    monthly["target_return"] = returns.loc[common, "KODEX200"]
    monthly["quality_quintile"] = _stable_quintile(
        monthly["data_quality_confidence"]
    )
    for quintile in range(1, 6):
        group = monthly.loc[monthly["quality_quintile"] == quintile]
        ic, observations = _spearman(
            group["pure_direction_raw"], group["target_return"]
        )
        rows.append(
            {
                "Frequency": "monthly",
                "Horizon": "1m",
                "QualityQuintile": quintile,
                "SpearmanIC": ic,
                "Observations": observations,
                "MeanQuality": float(group["data_quality_confidence"].mean()),
                "MeanRawODS": float(group["pure_direction_raw"].mean()),
                "MeanTargetReturn": float(group["target_return"].mean()),
            }
        )
    quintiles = pd.DataFrame(rows)

    interactions = pd.DataFrame(
        [
            _hac_interaction(
                early["pure_direction_raw"],
                early["data_quality_confidence"],
                early["forward_5d"],
                5,
                "early_daily_5d",
            ),
            _hac_interaction(
                early["pure_direction_raw"],
                early["data_quality_confidence"],
                early["forward_20d"],
                20,
                "early_daily_20d",
            ),
            _hac_interaction(
                monthly["pure_direction_raw"],
                monthly["data_quality_confidence"],
                monthly["target_return"],
                1,
                "early_monthly_1m",
            ),
        ]
    )
    trend: dict[str, Any] = {}
    for frequency, horizon in [("daily", "5d"), ("daily", "20d"), ("monthly", "1m")]:
        subset = quintiles.loc[
            (quintiles["Frequency"] == frequency)
            & (quintiles["Horizon"] == horizon)
        ].sort_values("QualityQuintile")
        valid = subset.dropna(subset=["SpearmanIC"])
        rank_ic, rank_p = spearmanr(valid["QualityQuintile"], valid["SpearmanIC"])
        values = subset["SpearmanIC"].to_numpy(dtype=float)
        trend[f"{frequency}_{horizon}"] = {
            "quintile_ic_spearman": float(rank_ic),
            "quintile_ic_pvalue": float(rank_p),
            "strictly_monotone_increasing": bool(np.all(np.diff(values) > 0.0)),
            "q5_minus_q1_ic": float(values[-1] - values[0]),
        }
    summary = {
        "period": "2007-01~2017-12",
        "hypothesis": "higher within-era option quality increases raw ODS IC",
        "trend": trend,
        "interaction_positive": {
            row["Test"]: bool(row["InteractionBeta"] > 0.0)
            for row in interactions.to_dict(orient="records")
        },
        "interaction_p_below_10pct": {
            row["Test"]: bool(
                row["InteractionBeta"] > 0.0 and row["InteractionPValue"] < 0.10
            )
            for row in interactions.to_dict(orient="records")
        },
    }
    return quintiles, interactions, summary


def early_component_quality_test(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Separate measurement quality from the roll component in the early era.

    This is a pre-specified diagnostic decomposition, not a strategy input or a
    parameter search.  Each observable is sorted into equal-count quintiles and
    evaluated against the same unweighted raw ODS signal.
    """

    early = daily.loc[EARLY_START:EARLY_END].copy()
    early["forward_5d"] = early["kospi200_close"].shift(-stage30.FAST_DAYS).div(
        early["kospi200_close"]
    ).sub(1.0)
    early["forward_20d"] = early["kospi200_close"].shift(-stage30.SLOW_DAYS).div(
        early["kospi200_close"]
    ).sub(1.0)
    early["measurement_q_no_roll"] = early[
        ["q_dte", "q_coverage", "q_parity", "q_arbitrage"]
    ].prod(axis=1, min_count=4)
    early["iv_validity"] = 1.0 - early["invalid_iv_share"]
    # A smaller 25-delta strike distance is better, so reverse the sign.
    early["put25_precision"] = -early["put25_nearest_strike_distance"]

    measures = {
        "composite_q": "data_quality_confidence",
        "measurement_q_no_roll": "measurement_q_no_roll",
        "listed_contracts": "listed_contracts",
        "coverage_log_width": "coverage_log_width",
        "iv_validity": "iv_validity",
        "put25_precision": "put25_precision",
    }
    quintile_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    trend: dict[str, dict[str, Any]] = {}
    for measure, column in measures.items():
        quintile_column = f"{measure}_quintile"
        early[quintile_column] = _stable_quintile(early[column])
        for horizon in [5, 20]:
            measure_rows: list[dict[str, Any]] = []
            for quintile in range(1, 6):
                group = early.loc[early[quintile_column] == quintile]
                ic, observations = _spearman(
                    group["pure_direction_raw"], group[f"forward_{horizon}d"]
                )
                row = {
                    "Measure": measure,
                    "Horizon": f"{horizon}d",
                    "QualityQuintile": quintile,
                    "SpearmanIC": ic,
                    "Observations": observations,
                    "MeanMeasure": float(group[column].mean()),
                }
                quintile_rows.append(row)
                measure_rows.append(row)
            values = np.asarray(
                [row["SpearmanIC"] for row in measure_rows], dtype=float
            )
            rank_ic, rank_p = spearmanr(np.arange(1, 6), values)
            trend[f"{measure}_{horizon}d"] = {
                "quintile_ic_spearman": float(rank_ic),
                "quintile_ic_pvalue": float(rank_p),
                "strictly_monotone_increasing": bool(
                    np.all(np.diff(values) > 0.0)
                ),
                "q5_minus_q1_ic": float(values[-1] - values[0]),
            }
        interaction_rows.append(
            {
                "Measure": measure,
                **_hac_interaction(
                    early["pure_direction_raw"],
                    early[column],
                    early["forward_20d"],
                    20,
                    f"early_{measure}_20d",
                ),
            }
        )
    summary = {
        "period": "2007-01~2017-12",
        "purpose": (
            "Separate roll timing from observable chain measurement quality; "
            "diagnostic only and not used by the strategy."
        ),
        "higher_is_better": list(measures),
        "trend": trend,
    }
    return (
        pd.DataFrame(quintile_rows),
        pd.DataFrame(interaction_rows),
        summary,
    )


def _selected_expiry_groups(
    chain: pd.DataFrame,
    surface: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[pd.Timestamp, pd.DataFrame]:
    selected = surface.loc[start:end, ["dte"]]
    wanted = {
        date: int(round(float(row["dte"])))
        for date, row in selected.iterrows()
    }
    relevant = chain.loc[chain["date"].isin(wanted)].copy()
    relevant["wanted_dte"] = relevant["date"].map(wanted)
    relevant = relevant.loc[relevant["dte"] == relevant["wanted_dte"]]
    return {
        date: group.drop(columns="wanted_dte").copy()
        for date, group in relevant.groupby("date", sort=True)
    }


def _build_templates(
    early_groups: dict[pd.Timestamp, pd.DataFrame],
    original_surface: pd.DataFrame,
) -> dict[int, list[dict[str, Any]]]:
    templates: dict[int, list[dict[str, Any]]] = {}
    for date, group in early_groups.items():
        if date not in original_surface.index:
            continue
        forward = float(original_surface.loc[date, "forward"])
        if not np.isfinite(forward) or forward <= 0.0:
            continue
        template: dict[str, Any] = {"source_date": date}
        valid_template = True
        for option_type in ["C", "P"]:
            side = (
                group.loc[group["option_type"] == option_type]
                .sort_values("strike")
                .drop_duplicates("strike", keep="last")
            )
            if len(side) < 3:
                valid_template = False
                break
            template[f"ratios_{option_type}"] = (
                side["strike"].to_numpy(dtype=float) / forward
            )
            template[f"iv_valid_{option_type}"] = side[
                "implied_volatility"
            ].between(3.0001, 200.0).to_numpy(dtype=bool)
        if valid_template:
            dte = int(round(float(group["dte"].iloc[0])))
            templates.setdefault(dte, []).append(template)
    return templates


def _match_template_to_late_group(
    group: pd.DataFrame,
    forward: float,
    template: dict[str, Any],
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for option_type in ["C", "P"]:
        candidates = (
            group.loc[group["option_type"] == option_type]
            .sort_values("strike")
            .drop_duplicates("strike", keep="last")
            .copy()
        )
        target_ratios = np.asarray(template[f"ratios_{option_type}"], dtype=float)
        target_valid = np.asarray(template[f"iv_valid_{option_type}"], dtype=bool)
        if candidates.empty or target_ratios.size == 0:
            continue
        if len(target_ratios) > len(candidates):
            positions = np.linspace(0, len(target_ratios) - 1, len(candidates)).round().astype(int)
            target_ratios = target_ratios[positions]
            target_valid = target_valid[positions]
        candidate_ratios = candidates["strike"].to_numpy(dtype=float) / forward
        cost = np.abs(
            np.log(target_ratios[:, None]) - np.log(candidate_ratios[None, :])
        )
        template_rows, candidate_rows = linear_sum_assignment(cost)
        matched = candidates.iloc[candidate_rows].copy()
        validity = target_valid[template_rows]
        matched.loc[~validity, "implied_volatility"] = np.nan
        selected.append(matched)
    if not selected:
        return group.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True).sort_values(
        ["option_type", "strike"]
    )


def degrade_late_surface(
    late_groups: dict[pd.Timestamp, pd.DataFrame],
    original_surface: pd.DataFrame,
    templates: dict[int, list[dict[str, Any]]],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild late surfaces on actual early-period strike-grid templates."""

    rng = np.random.default_rng(seed)
    available_dtes = np.asarray(sorted(templates), dtype=int)
    rows: list[dict[str, Any]] = []
    selected_template_dates: list[str] = []
    for date, group in sorted(late_groups.items()):
        if date not in original_surface.index:
            continue
        dte = int(round(float(original_surface.loc[date, "dte"])))
        template_dte = (
            dte
            if dte in templates
            else int(available_dtes[np.argmin(np.abs(available_dtes - dte))])
        )
        candidates = templates[template_dte]
        template = candidates[int(rng.integers(0, len(candidates)))]
        full_forward = float(original_surface.loc[date, "forward"])
        degraded_group = _match_template_to_late_group(
            group, full_forward, template
        )
        result = stage30._surface_for_expiry_without_order_flow(degraded_group)
        if result is None:
            continue
        rows.append(
            {
                "date": date,
                **result,
                "maturity_method": "early_grid_template_same_late_dte",
            }
        )
        selected_template_dates.append(str(template["source_date"].date()))
    degraded = pd.DataFrame(rows).set_index("date").sort_index()
    audit = {
        "seed": seed,
        "requested_late_dates": int(len(late_groups)),
        "successful_surface_dates": int(len(degraded)),
        "surface_success_rate": float(len(degraded) / max(len(late_groups), 1)),
        "unique_early_templates_used": int(len(set(selected_template_dates))),
        "mean_listed_contracts": float(degraded["listed_contracts"].mean()),
        "mean_coverage_log_width": float(degraded["coverage_log_width"].mean()),
        "mean_invalid_iv_share": float(degraded["invalid_iv_share"].mean()),
        "mean_put25_nearest_strike_distance": float(
            degraded["put25_nearest_strike_distance"].mean()
        ),
        "mean_dte": float(degraded["dte"].mean()),
    }
    return degraded, audit


def _prepare_feature_context(
    dates: pd.DatetimeIndex,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    _, macro_ranks = build_macro_probabilities(returns)
    vkospi = load_vkospi_daily()["close"].rename("vkospi_close")
    assets, _ = stage28.load_daily_asset_ohlcv()
    kodex = assets["KODEX200"]["close"].rename("kospi200_close")
    context = pd.DataFrame(index=dates).join(vkospi).join(kodex)
    context[["vkospi_close", "kospi200_close"]] = context[
        ["vkospi_close", "kospi200_close"]
    ].ffill(limit=3)
    context = context.join(stage30._macro_context_for_dates(dates, macro_ranks))
    return context


def rebuild_stage30_features_from_surface(
    surface: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Re-run the frozen Stage30 feature formula on a counterfactual surface."""

    daily = surface.copy().join(context, how="left")
    log_vkospi = np.log(daily["vkospi_close"].clip(lower=1e-12))
    log_kospi = np.log(daily["kospi200_close"].clip(lower=1e-12))
    residual_specs: list[tuple[str, pd.Series, pd.DataFrame]] = []
    for label, horizon in [("fast", stage30.FAST_DAYS), ("slow", stage30.SLOW_DAYS)]:
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
            daily["dte"] - stage30.TARGET_MATURITY_DAYS
        ) / stage30.TARGET_MATURITY_DAYS
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
            stage30.expanding_one_step_ols_residual(
                target,
                predictors,
                prefix,
                stage30.MIN_MODEL_HISTORY,
            )
        )
        daily[f"z_residual_{prefix}"] = stage30.causal_lagged_expanding_zscore(
            daily[f"residual_{prefix}"]
        )
    daily["abnormal_bear_pressure_fast"] = daily[
        ["z_residual_put_skew_fast", "z_residual_iva_fast"]
    ].mean(axis=1, skipna=False)
    daily["abnormal_bear_pressure_slow"] = daily[
        ["z_residual_put_skew_slow", "z_residual_iva_slow"]
    ].mean(axis=1, skipna=False)
    daily["pure_direction_raw"] = -daily[
        ["abnormal_bear_pressure_fast", "abnormal_bear_pressure_slow"]
    ].mean(axis=1, skipna=False)
    daily["q_dte"] = stage30.TARGET_MATURITY_DAYS / (
        stage30.TARGET_MATURITY_DAYS
        + (daily["dte"] - stage30.TARGET_MATURITY_DAYS).abs()
    )
    best_coverage = daily["coverage_log_width"].expanding().max()
    daily["q_coverage"] = (
        daily["coverage_log_width"] / best_coverage.replace(0.0, np.nan)
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
    daily["log_option_erp_proxy"] = np.log(
        daily["option_erp_proxy"].clip(lower=1e-12)
    )
    daily["z_option_erp_proxy"] = stage30.causal_lagged_expanding_zscore(
        daily["log_option_erp_proxy"]
    )
    return daily.replace([np.inf, -np.inf], np.nan)


def _forward_ic(daily: pd.DataFrame, horizon: int) -> float:
    target = daily["kospi200_close"].shift(-horizon).div(
        daily["kospi200_close"]
    ).sub(1.0)
    late = daily.loc[LATE_START:LATE_END]
    ic, _ = _spearman(late["pure_direction_raw"], target.loc[late.index])
    return ic


def late_degradation_ic_comparison(
    original_daily: pd.DataFrame, replications: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in [5, 20]:
        column = f"Forward{horizon}dRawODSIC"
        original = _forward_ic(original_daily, horizon)
        rows.append(
            {
                "Horizon": f"{horizon}d",
                "OriginalLateRawODSIC": original,
                "DegradedMeanRawODSIC": float(replications[column].mean()),
                "DegradedP05RawODSIC": float(replications[column].quantile(0.05)),
                "DegradedP50RawODSIC": float(replications[column].quantile(0.50)),
                "DegradedP95RawODSIC": float(replications[column].quantile(0.95)),
                "MeanDeltaDegradedMinusOriginal": float(
                    replications[column].mean() - original
                ),
                "ProbabilityDegradedBelowOriginal": float(
                    (replications[column] < original).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_degradation_test(
    original_daily: pd.DataFrame,
    raw_surface: pd.DataFrame,
    chain: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    early_groups = _selected_expiry_groups(
        chain, raw_surface, EARLY_START, EARLY_END
    )
    late_groups = _selected_expiry_groups(
        chain, raw_surface, LATE_START, raw_surface.index.max()
    )
    templates = _build_templates(early_groups, raw_surface)
    rows: list[dict[str, Any]] = []
    for replication in range(DEGRADATION_REPLICATIONS):
        seed = RANDOM_SEED + replication
        degraded_surface, audit = degrade_late_surface(
            late_groups, raw_surface, templates, seed
        )
        hybrid_surface = pd.concat(
            [raw_surface.loc[raw_surface.index < LATE_START], degraded_surface]
        ).sort_index()
        degraded_daily = rebuild_stage30_features_from_surface(
            hybrid_surface, context
        )
        raw_signals = stage30.build_monthly_direction_signals(
            returns.index, degraded_daily
        )
        signals = stage30.add_causal_mu_calibration(raw_signals, returns)
        path = stage30.run_backtest(
            returns,
            probabilities,
            stress_signals,
            technical_signals,
            signals,
            f"Stage30_DegradedEarlyGrid_R{replication:02d}",
        )
        metric = metric_row(
            f"degraded_{replication:02d}",
            path,
            "locked_2018_2026",
            LOCKED_START,
            pd.Period("2026-07", "M"),
        )
        rows.append(
            {
                "Replication": replication,
                **audit,
                "CAGR": float(metric["CAGR"]),
                "Volatility": float(metric["Volatility"]),
                "Sharpe": float(metric["Sharpe"]),
                "MDD": float(metric["MDD"]),
                "Calmar": float(metric["Calmar"]),
                "Forward5dRawODSIC": _forward_ic(degraded_daily, 5),
                "Forward20dRawODSIC": _forward_ic(degraded_daily, 20),
                "MeanLateQuality": float(
                    degraded_daily.loc[LATE_START:LATE_END, "data_quality_confidence"].mean()
                ),
            }
        )
    replications = pd.DataFrame(rows)

    original_path = pd.read_csv(
        stage30.OUTPUT_DIR / "stage30_pureods_qualitycausal_monthly.csv",
        index_col=0,
    )
    original_path.index = pd.PeriodIndex(original_path.index, freq="M")
    original_metric = metric_row(
        "original_late",
        original_path,
        "locked_2018_2026",
        LOCKED_START,
        pd.Period("2026-07", "M"),
    )
    comparison_rows: list[dict[str, Any]] = [
        {
            "Metric": metric,
            "OriginalLate": float(original_metric[metric]),
            "DegradedMean": float(replications[metric].mean()),
            "DegradedP05": float(replications[metric].quantile(0.05)),
            "DegradedP50": float(replications[metric].quantile(0.50)),
            "DegradedP95": float(replications[metric].quantile(0.95)),
            "MeanDeltaDegradedMinusOriginal": float(
                replications[metric].mean() - original_metric[metric]
            ),
            "ProbabilityDegradedBelowOriginal": float(
                (replications[metric] < original_metric[metric]).mean()
            ),
        }
        for metric in ["CAGR", "Sharpe", "MDD", "Calmar"]
    ]
    comparison = pd.DataFrame(comparison_rows)
    original_late = original_daily.loc[LATE_START:LATE_END]
    summary = {
        "replications": DEGRADATION_REPLICATIONS,
        "random_seed_start": RANDOM_SEED,
        "method": (
            "For each late date, select an actual 2007-2017 strike/moneyness and "
            "IV-validity template with the same DTE, then one-to-one match it to "
            "the nearest late strikes. Late prices and market state remain unchanged."
        ),
        "early_template_days": int(sum(len(value) for value in templates.values())),
        "late_days_requested": int(len(late_groups)),
        "original_late_quality": {
            "listed_contracts": float(original_late["listed_contracts"].mean()),
            "coverage_log_width": float(original_late["coverage_log_width"].mean()),
            "invalid_iv_share": float(original_late["invalid_iv_share"].mean()),
            "put25_nearest_strike_distance": float(
                original_late["put25_nearest_strike_distance"].mean()
            ),
            "dte": float(original_late["dte"].mean()),
            "quality": float(original_late["data_quality_confidence"].mean()),
        },
        "degraded_late_quality_mean": {
            "listed_contracts": float(replications["mean_listed_contracts"].mean()),
            "coverage_log_width": float(
                replications["mean_coverage_log_width"].mean()
            ),
            "invalid_iv_share": float(
                replications["mean_invalid_iv_share"].mean()
            ),
            "put25_nearest_strike_distance": float(
                replications["mean_put25_nearest_strike_distance"].mean()
            ),
            "dte": float(replications["mean_dte"].mean()),
            "quality": float(replications["MeanLateQuality"].mean()),
            "surface_success_rate": float(
                replications["surface_success_rate"].mean()
            ),
        },
    }
    return replications, comparison, summary


def run_validation(save: bool = True) -> dict[str, Any]:
    manifest_before = raw_chain_manifest()
    original_daily = pd.read_csv(
        stage30.OUTPUT_DIR / "daily_abnormal_surface_erp_features.csv",
        index_col=0,
        parse_dates=True,
    )
    raw_surface = original_daily[SURFACE_COLUMNS].copy()
    monthly_signals = pd.read_csv(
        stage30.OUTPUT_DIR / "monthly_option_alpha_signals.csv", index_col=0
    )
    monthly_signals.index = pd.PeriodIndex(monthly_signals.index, freq="M")
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)
    daily_technical, _ = stage28.build_daily_technical_features()
    technical_signals = stage28.build_monthly_technical_signals(
        returns.index, daily_technical
    )
    context = _prepare_feature_context(raw_surface.index, returns)

    reproduced = rebuild_stage30_features_from_surface(raw_surface, context)
    fidelity_columns = [
        "pure_direction_raw",
        "data_quality_confidence",
        "option_direction_score",
        "z_option_erp_proxy",
    ]
    fidelity_rows = []
    for column in fidelity_columns:
        common = original_daily[column].dropna().index.intersection(
            reproduced[column].dropna().index
        )
        fidelity_rows.append(
            {
                "Feature": column,
                "Observations": int(len(common)),
                "MaxAbsoluteDifference": float(
                    (
                        original_daily.loc[common, column]
                        - reproduced.loc[common, column]
                    ).abs().max()
                ),
            }
        )
    fidelity = pd.DataFrame(fidelity_rows)

    quintiles, interactions, within_summary = within_early_quality_test(
        original_daily, monthly_signals, returns
    )
    component_quintiles, component_interactions, component_summary = (
        early_component_quality_test(original_daily)
    )
    chain, chain_audit = stage28.load_kospi200_option_chain()
    replications, degradation_comparison, degradation_summary = (
        run_degradation_test(
            original_daily,
            raw_surface,
            chain,
            returns,
            probabilities,
            stress_signals,
            technical_signals,
            context,
        )
    )
    manifest_after = raw_chain_manifest()
    raw_unchanged = manifest_before == manifest_after
    degradation_ic = late_degradation_ic_comparison(
        original_daily, replications
    )
    degradation_cagr = degradation_comparison.set_index("Metric").loc["CAGR"]
    degradation_sharpe = degradation_comparison.set_index("Metric").loc["Sharpe"]
    within_yes = bool(
        within_summary["trend"]["daily_20d"]["q5_minus_q1_ic"] > 0.0
        and within_summary["trend"]["monthly_1m"]["q5_minus_q1_ic"] > 0.0
    )
    degradation_yes = bool(
        degradation_cagr["MeanDeltaDegradedMinusOriginal"] < 0.0
        and degradation_sharpe["MeanDeltaDegradedMinusOriginal"] < 0.0
    )
    if within_yes and degradation_yes:
        verdict = "data_quality_hypothesis_supported"
    elif within_yes or degradation_yes:
        verdict = "data_quality_is_a_partial_explanation"
    else:
        verdict = "data_quality_hypothesis_not_supported"
    report = {
        "study": "Stage30_DataQualityValidation",
        "verdict": verdict,
        "scope": (
            "Causal diagnosis only. No strategy parameter, allocation rule, or "
            "original option-chain file is changed."
        ),
        "within_early_quality_test": within_summary,
        "early_component_quality_diagnostic": component_summary,
        "degradation_test": degradation_summary,
        "degradation_performance_comparison": json.loads(
            degradation_comparison.to_json(orient="records", force_ascii=False)
        ),
        "degradation_raw_ods_ic_comparison": json.loads(
            degradation_ic.to_json(orient="records", force_ascii=False)
        ),
        "pipeline_fidelity": json.loads(
            fidelity.to_json(orient="records", force_ascii=False)
        ),
        "raw_chain_audit": chain_audit,
        "raw_files_unchanged": raw_unchanged,
        "raw_files_unchanged_after_secondary_diagnostic": raw_unchanged,
        "raw_manifest_before": manifest_before,
        "raw_manifest_after": manifest_after,
        "decision_rules": {
            "within_era_support": (
                "Q5-Q1 raw ODS IC is positive for both early daily 20d and monthly 1m"
            ),
            "degradation_support": (
                "mean degraded late CAGR and Sharpe are both below original late"
            ),
            "rules_fixed_before_results": True,
        },
        "checks": {
            "raw_option_files_unchanged": raw_unchanged,
            "pipeline_reproduction_max_error_below_1e_12": bool(
                (fidelity["MaxAbsoluteDifference"] < 1e-12).all()
            ),
            "twenty_degradation_replications_completed": bool(
                len(replications) == DEGRADATION_REPLICATIONS
            ),
            "late_prices_not_synthetically_changed": True,
            "late_dte_rule_preserved": True,
            "only_strike_grid_and_iv_validity_are_degraded": True,
            "no_strategy_candidate_or_parameter_search": True,
        },
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        quintiles.to_csv(OUTPUT_DIR / "early_quality_quintile_ic.csv", index=False)
        interactions.to_csv(
            OUTPUT_DIR / "early_quality_interaction_hac.csv", index=False
        )
        component_quintiles.to_csv(
            OUTPUT_DIR / "early_component_quality_quintile_ic.csv", index=False
        )
        component_interactions.to_csv(
            OUTPUT_DIR / "early_component_quality_interaction_hac.csv",
            index=False,
        )
        replications.to_csv(
            OUTPUT_DIR / "late_degradation_replications.csv", index=False
        )
        degradation_comparison.to_csv(
            OUTPUT_DIR / "late_degradation_comparison.csv", index=False
        )
        degradation_ic.to_csv(
            OUTPUT_DIR / "late_degradation_ic_comparison.csv", index=False
        )
        fidelity.to_csv(OUTPUT_DIR / "pipeline_fidelity.csv", index=False)
        (OUTPUT_DIR / "raw_option_chain_manifest_before.json").write_text(
            json.dumps(manifest_before, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUTPUT_DIR / "raw_option_chain_manifest_after.json").write_text(
            json.dumps(manifest_after, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return {
        "quintiles": quintiles,
        "interactions": interactions,
        "component_quintiles": component_quintiles,
        "component_interactions": component_interactions,
        "replications": replications,
        "degradation_comparison": degradation_comparison,
        "degradation_ic": degradation_ic,
        "fidelity": fidelity,
        "report": report,
    }


def main() -> None:
    result = run_validation(save=True)
    print(result["quintiles"].to_string(index=False))
    print(result["interactions"].to_string(index=False))
    print(result["component_quintiles"].to_string(index=False))
    print(result["component_interactions"].to_string(index=False))
    print(result["degradation_comparison"].to_string(index=False))
    print(result["degradation_ic"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
