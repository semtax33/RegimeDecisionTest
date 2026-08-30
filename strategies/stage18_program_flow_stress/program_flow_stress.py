from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    load_monthly_asset_returns,
    performance_summary,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    FULL_START,
    LOCKED_START,
    ONE_WEEK,
    build_daily_stress_features,
    build_monthly_stress_signals,
    causal_expanding_midrank,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    DYNAMIC_RISK_POLICY,
    STATIC_RISK_POLICY,
    concentration_summary,
    metric_row,
    run_backtest,
    solver_summary,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PROGRAM_TRADING_PATH = (
    ROOT / "raw_data" / "krx_program_trading_20000101_20260829.csv"
)

PROGRAM_CATEGORIES = ("차익", "비차익")
PROGRAM_FEATURE_COLUMNS = [
    "arbitrage_sell_pressure_5d",
    "non_arbitrage_sell_pressure_5d",
    "arbitrage_stress_rank",
    "non_arbitrage_stress_rank",
    "program_stress_component",
]


def load_program_trading_volume(
    path: Path = PROGRAM_TRADING_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and validate the normalized daily KRX program-trading file.

    Value fields are deliberately excluded from the signal. The user requested
    daily volume, and mixing value with volume would count the same orders twice.
    Likewise, the reported total is not an independent category because it is
    exactly arbitrage plus non-arbitrage in this file.
    """

    required = {
        "date",
        "market",
        "category",
        "sell_volume",
        "buy_volume",
        "net_buy_volume",
        "sell_value_krw",
        "buy_value_krw",
        "net_buy_value_krw",
    }
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Program-trading file is missing columns: {missing}")
    if frame.duplicated(["date", "market", "category"]).any():
        raise ValueError("Program-trading file contains duplicate keys.")
    if frame[list(required - {"date", "market", "category"})].isna().any().any():
        raise ValueError("Program-trading file contains missing numeric values.")
    if not (frame["market"] == "ALL").all():
        raise ValueError("Expected only the normalized ALL market aggregate.")
    category_counts = frame.groupby("date")["category"].nunique()
    if not (category_counts == 3).all():
        raise ValueError("Every date must contain arbitrage, non-arbitrage and total.")
    if not set(PROGRAM_CATEGORIES).issubset(set(frame["category"])):
        raise ValueError("Arbitrage/non-arbitrage categories are missing.")

    net_volume_mismatch = int(
        ((frame["buy_volume"] - frame["sell_volume"]) != frame["net_buy_volume"]).sum()
    )
    net_value_mismatch = int(
        (
            (frame["buy_value_krw"] - frame["sell_value_krw"])
            != frame["net_buy_value_krw"]
        ).sum()
    )
    if net_volume_mismatch or net_value_mismatch:
        raise ValueError("Net-buy arithmetic does not match buy minus sell.")

    numeric = [
        "sell_volume",
        "buy_volume",
        "net_buy_volume",
        "sell_value_krw",
        "buy_value_krw",
        "net_buy_value_krw",
    ]
    pivot = frame.pivot(index="date", columns="category", values=numeric)
    total_mismatches = {
        column: int(
            (
                pivot[(column, "전체")]
                - pivot[(column, "차익")]
                - pivot[(column, "비차익")]
            ).ne(0).sum()
        )
        for column in numeric
    }
    if any(total_mismatches.values()):
        raise ValueError("Reported total does not equal category components.")

    selected = frame.loc[frame["category"].isin(PROGRAM_CATEGORIES)].copy()
    selected = selected.sort_values(["date", "category"])
    audit = {
        "path": str(path),
        "rows": int(len(frame)),
        "unique_dates": int(frame["date"].nunique()),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "categories": sorted(frame["category"].unique().tolist()),
        "duplicate_keys": 0,
        "days_without_three_categories": 0,
        "missing_numeric_values": 0,
        "net_volume_mismatches": net_volume_mismatch,
        "net_value_mismatches": net_value_mismatch,
        "total_vs_parts_mismatches": total_mismatches,
    }
    return selected, audit


def build_program_volume_features(
    program: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one causal, scale-free program-selling stress block.

    For each independent KRX category, weekly sell pressure is

        sum(sell volume - buy volume) / sum(sell volume + buy volume).

    A causal expanding percentile puts changing market volume on a stable 0..1
    scale without a fitted lookback or standard-deviation cutoff. The two ranks
    receive equal weight. No total/value feature or fitted coefficient is used.
    """

    if program is None:
        program, audit = load_program_trading_volume()
    else:
        audit = {"provided_frame": True}
    pivot = program.pivot(
        index="date",
        columns="category",
        values=["sell_volume", "buy_volume"],
    ).sort_index()
    output = pd.DataFrame(index=pivot.index)
    definitions = {
        "차익": ("arbitrage_sell_pressure_5d", "arbitrage_stress_rank"),
        "비차익": (
            "non_arbitrage_sell_pressure_5d",
            "non_arbitrage_stress_rank",
        ),
    }
    for category, (pressure_column, rank_column) in definitions.items():
        net_selling = (
            pivot[("sell_volume", category)]
            - pivot[("buy_volume", category)]
        )
        gross_volume = (
            pivot[("sell_volume", category)]
            + pivot[("buy_volume", category)]
        )
        weekly_net_selling = net_selling.rolling(
            ONE_WEEK, min_periods=ONE_WEEK
        ).sum()
        weekly_gross_volume = gross_volume.rolling(
            ONE_WEEK, min_periods=ONE_WEEK
        ).sum()
        output[pressure_column] = weekly_net_selling / weekly_gross_volume
        output[rank_column] = causal_expanding_midrank(output[pressure_column])

    output["program_stress_component"] = output[
        ["arbitrage_stress_rank", "non_arbitrage_stress_rank"]
    ].mean(axis=1)
    return output.replace([np.inf, -np.inf], np.nan), audit


def build_program_augmented_daily_stress() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add program flow as a fifth equal-weight Stage 14 stress block."""

    original = build_daily_stress_features().copy()
    program, audit = build_program_volume_features()
    # Reindexing with forward fill is an as-of join: a VKOSPI/VIX6 date may use
    # only the latest KRX program observation at or before that date.
    aligned_program = program.reindex(original.index, method="ffill")
    daily = original.join(aligned_program, how="left")
    daily["original_stress_raw"] = original["stress_raw"]
    daily["original_stress_score"] = original["stress_score"]
    daily["original_recovery_score"] = original["recovery_score"]
    five_blocks = [
        "level_component",
        "shock_component",
        "tail_component",
        "persistence_component",
        "program_stress_component",
    ]
    daily["stress_raw"] = daily[five_blocks].mean(axis=1)
    one_week_mean = daily["stress_raw"].rolling(
        ONE_WEEK, min_periods=1
    ).mean()
    daily["stress_score"] = pd.concat(
        [daily["stress_raw"], one_week_mean], axis=1
    ).max(axis=1).clip(0.0, 1.0)
    recovery_intensity = (one_week_mean - daily["stress_raw"]).clip(lower=0.0)
    daily["recovery_score"] = causal_expanding_midrank(
        recovery_intensity.where(recovery_intensity > 0.0)
    ).fillna(0.0)
    audit["aligned_daily_start"] = str(daily.dropna(subset=["program_stress_component"]).index.min().date())
    audit["aligned_daily_end"] = str(daily.dropna(subset=["program_stress_component"]).index.max().date())
    audit["aligned_missing_program_rows"] = int(daily["program_stress_component"].isna().sum())
    return daily.replace([np.inf, -np.inf], np.nan), audit


def build_program_monthly_stress_signals(
    target_months: pd.PeriodIndex,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Map lagged daily stress to target months and preserve program fields."""

    monthly = build_monthly_stress_signals(target_months, daily)
    extra_columns = PROGRAM_FEATURE_COLUMNS + [
        "original_stress_raw",
        "original_stress_score",
        "original_recovery_score",
    ]
    for column in extra_columns:
        monthly[column] = [
            float(daily.loc[pd.Timestamp(signal_date), column])
            for signal_date in monthly["stress_signal_date"]
        ]
    return monthly


def _attach_program_fields(
    path: pd.DataFrame,
    monthly_signals: pd.DataFrame,
) -> pd.DataFrame:
    output = path.copy()
    fields = PROGRAM_FEATURE_COLUMNS + [
        "original_stress_score",
        "original_recovery_score",
    ]
    for column in fields:
        output[column] = monthly_signals.loc[output.index, column]
    return output


def _auc(labels: pd.Series, scores: pd.Series) -> float:
    """Mann-Whitney AUC with average ranks for ties."""

    valid = pd.concat([labels.rename("label"), scores.rename("score")], axis=1).dropna()
    positives = valid["label"].astype(bool)
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = valid["score"].rank(method="average")
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def predictive_diagnostics(
    returns: pd.DataFrame,
    original_signals: pd.DataFrame,
    program_signals: pd.DataFrame,
    start: pd.Period,
    end: pd.Period,
) -> list[dict[str, Any]]:
    """Report, but never optimize on, next-month Korean equity direction."""

    target_return = returns.loc[start:end, "KODEX200"]
    diagnostics: list[dict[str, Any]] = []
    scores = {
        "Original_VKOSPI_VIX6_Stress": original_signals.loc[
            start:end, "stress_score"
        ],
        "Program_Volume_Component": program_signals.loc[
            start:end, "program_stress_component"
        ],
        "Augmented_Five_Block_Stress": program_signals.loc[
            start:end, "stress_score"
        ],
    }
    for name, score in scores.items():
        common = target_return.index.intersection(score.dropna().index)
        y = target_return.loc[common]
        s = score.loc[common].clip(0.0, 1.0)
        negative_month = (y < 0.0).astype(float)
        diagnostics.append(
            {
                "Signal": name,
                "Start": str(common.min()),
                "End": str(common.max()),
                "Observations": int(len(common)),
                "DownsideIC": float(s.corr(-y, method="spearman")),
                "NegativeMonthAUC": _auc(negative_month, s),
                # This is a diagnostic probability proxy, not a calibrated
                # classifier probability; it is therefore labelled explicitly.
                "UncalibratedBrier": float(np.mean((s - negative_month) ** 2)),
                "MeanKODEXReturn": float(y.mean()),
            }
        )
    return diagnostics


def _paired_effect(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, Any]:
    common = baseline.loc[start:end].index.intersection(candidate.loc[start:end].index)
    difference = candidate.loc[common, "return"] - baseline.loc[common, "return"]
    unchanged = np.isclose(difference, 0.0, atol=1e-12)
    base_metrics = performance_summary(baseline.loc[common, "return"])
    candidate_metrics = performance_summary(candidate.loc[common, "return"])
    return {
        "start": str(common.min()),
        "end": str(common.max()),
        "months": int(len(common)),
        "annualized_arithmetic_alpha": float(difference.mean() * 12.0),
        "positive_alpha_months": int(((difference > 0.0) & ~unchanged).sum()),
        "negative_alpha_months": int(((difference < 0.0) & ~unchanged).sum()),
        "unchanged_months": int(unchanged.sum()),
        "cagr_change": float(candidate_metrics["CAGR"] - base_metrics["CAGR"]),
        "sharpe_change": float(candidate_metrics["Sharpe"] - base_metrics["Sharpe"]),
        "mdd_change": float(candidate_metrics["MDD"] - base_metrics["MDD"]),
    }


def run_research(save: bool = True) -> dict[str, Any]:
    """Run one predeclared Stage 14 augmentation with no parameter search."""

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    original_daily = build_daily_stress_features()
    original_signals = build_monthly_stress_signals(returns.index, original_daily)
    program_daily, data_audit = build_program_augmented_daily_stress()
    program_signals = build_program_monthly_stress_signals(
        returns.index, program_daily
    )

    baseline_static = run_backtest(
        returns, probabilities, original_signals, STATIC_RISK_POLICY
    )
    baseline_dynamic = run_backtest(
        returns, probabilities, original_signals, DYNAMIC_RISK_POLICY
    )
    program_static = _attach_program_fields(
        run_backtest(returns, probabilities, program_signals, STATIC_RISK_POLICY),
        program_signals,
    )
    program_dynamic = _attach_program_fields(
        run_backtest(returns, probabilities, program_signals, DYNAMIC_RISK_POLICY),
        program_signals,
    )
    paths = {
        "Stage14_Original_StaticLambda": baseline_static,
        "Stage14_Original_DynamicLambda": baseline_dynamic,
        "Stage18_ProgramStress_StaticLambda": program_static,
        "Stage18_ProgramStress_DynamicLambda": program_dynamic,
    }
    common_end = min(path.index.max() for path in paths.values())
    comparisons: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            comparisons.append(metric_row(name, path, period, start, common_end))
    comparison = pd.DataFrame(comparisons)

    diagnostics = pd.DataFrame(
        predictive_diagnostics(
            returns, original_signals, program_signals, FULL_START, common_end
        )
        + predictive_diagnostics(
            returns, original_signals, program_signals, LOCKED_START, common_end
        )
    )
    weights = program_dynamic[[f"w_{asset}" for asset in ASSETS]]
    checks = {
        "program_data_precedes_backtest": bool(
            pd.Timestamp(data_audit["aligned_daily_start"])
            < FULL_START.to_timestamp("M")
        ),
        "macro_signal_precedes_target": bool(
            (program_dynamic["macro_signal_month"] < program_dynamic.index).all()
        ),
        "stress_signal_precedes_target": bool(
            (program_dynamic["stress_signal_month"] < program_dynamic.index).all()
        ),
        "program_signal_is_lagged": bool(
            (
                pd.to_datetime(program_dynamic["stress_signal_date"]).dt.to_period("M")
                < program_dynamic.index
            ).all()
        ),
        "weights_sum_to_one": bool(np.allclose(weights.sum(axis=1), 1.0)),
        "weights_are_long_only": bool((weights >= -1e-10).all().all()),
        "no_leverage": bool(np.allclose(weights.sum(axis=1), 1.0)),
        "single_asset_majority_rule_removed": True,
        "no_hyperparameter_or_candidate_search": True,
        "one_predeclared_formula": True,
        "program_total_excluded_as_duplicate": True,
        "program_value_excluded_to_avoid_double_counting": True,
    }
    report: dict[str, Any] = {
        "strategy": "Stage18_ProgramFlowStress_DynamicLambda",
        "base_strategy": "Stage14_NoAssetCap_DynamicLambda",
        "program_feature_formula": {
            "category_weekly_sell_pressure": (
                "rolling_5d_sum(sell_volume-buy_volume) / "
                "rolling_5d_sum(sell_volume+buy_volume)"
            ),
            "normalization": "causal expanding empirical percentile",
            "program_stress_component": (
                "equal mean(arbitrage rank, non-arbitrage rank)"
            ),
            "augmented_raw_stress": (
                "equal mean(VKOSPI level, volatility shock, VIX6 tail, "
                "existing persistence, program volume stress)"
            ),
            "economic_horizon": "5 observations = one trading week",
            "searched_parameters": None,
        },
        "data_audit": data_audit,
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "predictive_diagnostics": json.loads(
            diagnostics.to_json(orient="records", force_ascii=False)
        ),
        "program_effect_vs_stage14_dynamic": {
            "full_2007_2026": _paired_effect(
                baseline_dynamic, program_dynamic, FULL_START, common_end
            ),
            "locked_2018_2026": _paired_effect(
                baseline_dynamic, program_dynamic, LOCKED_START, common_end
            ),
        },
        "program_moment_effect_static_lambda": {
            "full_2007_2026": _paired_effect(
                baseline_static, program_static, FULL_START, common_end
            ),
            "locked_2018_2026": _paired_effect(
                baseline_static, program_static, LOCKED_START, common_end
            ),
        },
        "program_static_vs_dynamic_lambda": {
            "full_2007_2026": _paired_effect(
                program_static, program_dynamic, FULL_START, common_end
            ),
            "locked_2018_2026": _paired_effect(
                program_static, program_dynamic, LOCKED_START, common_end
            ),
        },
        "program_component_correlation_with_original_stress": float(
            program_signals.loc[FULL_START:common_end, "program_stress_component"].corr(
                original_signals.loc[FULL_START:common_end, "stress_score"],
                method="spearman",
            )
        ),
        "program_dynamic_concentration": concentration_summary(program_dynamic),
        "program_dynamic_solver": solver_summary(program_dynamic),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        program_daily.to_csv(OUTPUT_DIR / "program_augmented_daily_stress.csv")
        program_signals.to_csv(OUTPUT_DIR / "program_augmented_monthly_signals.csv")
        baseline_static.to_csv(OUTPUT_DIR / "stage14_recomputed_static_monthly.csv")
        baseline_dynamic.to_csv(OUTPUT_DIR / "stage14_recomputed_dynamic_monthly.csv")
        program_static.to_csv(OUTPUT_DIR / "program_stress_static_monthly.csv")
        program_dynamic.to_csv(OUTPUT_DIR / "program_stress_dynamic_monthly.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        diagnostics.to_csv(OUTPUT_DIR / "predictive_diagnostics.csv", index=False)
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "program_daily": program_daily,
        "program_signals": program_signals,
        "baseline_static": baseline_static,
        "baseline_dynamic": baseline_dynamic,
        "program_static": program_static,
        "program_dynamic": program_dynamic,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(result["diagnostics"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
