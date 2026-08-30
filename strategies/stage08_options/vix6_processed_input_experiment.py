from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import performance_summary
from strategies.stage08_options.vix6_case1_model_comparison import (
    BASE_FEATURES,
    C_VALUES,
    apply_existing_vkospi_overlay,
    build_overlay_context,
    build_research_data,
    fit_expanding_logistic,
)
from strategies.stage08_options.vix6_case1_strategy import build_vix6_features
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import CAL_END, TEST_START
from strategies.stage06_vkospi.vkospi_model_robustness import make_tail_factor, run_factor_vol_target


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MONTHLY_FEATURE_PATH = RESULTS / "vix6_processed_input_features_monthly.csv"
COMPARISON_PATH = RESULTS / "vix6_processed_input_comparison_2007_2026.csv"
REPORT_PATH = RESULTS / "vix6_processed_input_report_2007_2026.json"
BEST_IMPORTANCE_PATH = RESULTS / "vix6_processed_input_best_importance.csv"
BEST_FACTOR_PATH = RESULTS / "vix6_processed_input_best_factor.csv"
BEST_MEDIUM_PATH = RESULTS / "vix6_processed_input_best_medium.csv"
BEST_FINAL_PATH = RESULTS / "vix6_processed_input_best_reconciled.csv"

COMPONENTS = (
    "sticky_strike",
    "parallel_shift",
    "put_skew",
    "call_skew",
    "downside_convexity",
    "upside_convexity",
)
FULL_TEST_START = pd.Period("2007-01", freq="M")
FULL_TEST_END = pd.Period("2026-12", freq="M")
TRANSFORMS = (
    "raw",
    "ma5",
    "ma21",
    "delta5",
    "delta21",
    "z63",
    "z126",
)


def _feature_names(*transforms: str) -> list[str]:
    return [
        f"{component}_{transform}_last"
        for transform in transforms
        for component in COMPONENTS
    ]


CANDIDATE_SETS: dict[str, list[str]] = {
    "raw6_control": _feature_names("raw"),
    "ma5_6": _feature_names("ma5"),
    "ma21_6": _feature_names("ma21"),
    "delta5_6": _feature_names("delta5"),
    "delta21_6": _feature_names("delta21"),
    "z63_6": _feature_names("z63"),
    "z126_6": _feature_names("z126"),
    "short_ma_delta_z18": _feature_names("ma5", "delta5", "z63"),
    "medium_ma_delta_z18": _feature_names("ma21", "delta21", "z126"),
    "multiscale_ma_z24": _feature_names("ma5", "ma21", "z63", "z126"),
}


def trailing_zscore(
    series: pd.Series,
    window: int,
    minimum: int,
) -> pd.Series:
    """Trailing z-score available on the observation date; no future rows are used."""
    mean = series.rolling(window, min_periods=minimum).mean()
    standard_deviation = series.rolling(window, min_periods=minimum).std(ddof=1)
    return ((series - mean) / standard_deviation.replace(0.0, np.nan)).clip(-6, 6)


def build_processed_monthly_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Transform daily VIX6 components and lag the monthly snapshot by one month."""
    missing = sorted(set(COMPONENTS).difference(features.columns))
    if missing:
        raise ValueError(f"Missing VIX6 decomposition components: {missing}")

    daily = pd.DataFrame(index=pd.DatetimeIndex(features.index).sort_values())
    source = features.reindex(daily.index)
    for component in COMPONENTS:
        values = pd.to_numeric(source[component], errors="coerce")
        daily[f"{component}_raw_last"] = values
        daily[f"{component}_ma5_last"] = values.rolling(5, min_periods=3).mean()
        daily[f"{component}_ma21_last"] = values.rolling(21, min_periods=10).mean()
        daily[f"{component}_delta5_last"] = values.diff(5)
        daily[f"{component}_delta21_last"] = values.diff(21)
        daily[f"{component}_z63_last"] = trailing_zscore(values, 63, 21)
        daily[f"{component}_z126_last"] = trailing_zscore(values, 126, 42)

    signal_month = daily.index.to_period("M")
    monthly = daily.groupby(signal_month).last()
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    monthly["signal_month"] = monthly.index.astype(str)
    monthly.index = monthly.index + 1
    monthly.index.name = "target_month"
    lag_ok = pd.PeriodIndex(monthly["signal_month"], freq="M") < monthly.index
    if not bool(lag_ok.all()):
        raise AssertionError("Every processed VIX6 input must precede its target month")
    return monthly.replace([np.inf, -np.inf], np.nan)


def _period_metrics(
    path: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> pd.Series:
    view = path
    if start is not None:
        view = view.loc[start:]
    if end is not None:
        view = view.loc[:end]
    return performance_summary(view["return"])


def _metric_columns(
    path: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float]:
    common = path.index.intersection(baseline.index)
    candidate = path.loc[common]
    reference = baseline.loc[common]
    output: dict[str, float] = {}
    periods = (
        ("full", FULL_TEST_START, FULL_TEST_END),
        ("pre2018", None, CAL_END),
        ("post2018", TEST_START, None),
    )
    for prefix, start, end in periods:
        metrics = _period_metrics(candidate, start, end)
        baseline_metrics = _period_metrics(reference, start, end)
        for metric in ("CAGR", "Sharpe", "MDD"):
            output[f"{prefix}_{metric}"] = float(metrics[metric])
            output[f"{prefix}_{metric}_delta"] = float(
                metrics[metric] - baseline_metrics[metric]
            )
    output["full_score"] = (
        output["full_CAGR_delta"] / 0.01
        + output["full_Sharpe_delta"] / 0.05
        + output["full_MDD_delta"] / 0.01
    )
    output["full_all_three_improve"] = bool(
        output["full_CAGR_delta"] > 0
        and output["full_Sharpe_delta"] > 0
        and output["full_MDD_delta"] >= -1e-12
    )
    return output


def _run_input_candidate(
    feature_names: list[str],
    c_value: float,
    data: pd.DataFrame,
    monthly_inputs: pd.DataFrame,
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    context: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    candidate_data = data.join(monthly_inputs[feature_names], how="left")
    probability, importance, fit_stats = fit_expanding_logistic(
        candidate_data,
        BASE_FEATURES + feature_names,
        c_value,
    )
    factor = make_tail_factor(probability, candidate_data["tail_event"])
    medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    _, final = apply_existing_vkospi_overlay(medium, context)
    return importance, factor, medium, final, fit_stats


def run_experiment(force_features: bool = False) -> dict[str, object]:
    option_features = build_vix6_features(force_features)
    monthly_inputs = build_processed_monthly_inputs(option_features)
    MONTHLY_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_inputs.to_csv(MONTHLY_FEATURE_PATH)

    returns, signals, defensive, _, data = build_research_data()
    context = build_overlay_context()
    base_probability, _, base_fit_stats = fit_expanding_logistic(
        data,
        BASE_FEATURES,
        0.10,
    )
    base_factor = make_tail_factor(base_probability, data["tail_event"])
    base_medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        base_factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    _, baseline = apply_existing_vkospi_overlay(base_medium, context)

    stored_baseline = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv",
        index_col=0,
    )
    stored_baseline.index = pd.PeriodIndex(stored_baseline.index, freq="M")
    common = baseline.index.intersection(stored_baseline.index)
    common = common[(common >= FULL_TEST_START) & (common <= FULL_TEST_END)]
    reproduction_error = float(
        (
            baseline.loc[common, "return"]
            - stored_baseline.loc[common, "return"]
        )
        .abs()
        .max()
    )
    if reproduction_error >= 1e-12:
        raise AssertionError(
            f"Existing strategy reproduction error is {reproduction_error:.3e}"
        )

    baseline_metrics = _period_metrics(
        baseline.loc[common],
        FULL_TEST_START,
        FULL_TEST_END,
    )
    rows: list[dict[str, object]] = [
        {
            "candidate": "Existing_Final_RobustVKOSPI",
            "transform_family": "baseline",
            "C": 0.10,
            "feature_count": len(BASE_FEATURES),
            "fit_count": int(base_fit_stats["fit_count"]),
            "convergence_warning_count": int(
                base_fit_stats["convergence_warning_count"]
            ),
            "full_CAGR": float(baseline_metrics["CAGR"]),
            "full_CAGR_delta": 0.0,
            "full_Sharpe": float(baseline_metrics["Sharpe"]),
            "full_Sharpe_delta": 0.0,
            "full_MDD": float(baseline_metrics["MDD"]),
            "full_MDD_delta": 0.0,
            "full_score": 0.0,
            "full_all_three_improve": False,
        }
    ]

    candidate_total = len(CANDIDATE_SETS) * len(C_VALUES)
    completed = 0
    for set_name, feature_names in CANDIDATE_SETS.items():
        transform_family = "raw_control" if set_name == "raw6_control" else "processed"
        for c_value in C_VALUES:
            _, _, _, final, fit_stats = _run_input_candidate(
                feature_names,
                c_value,
                data,
                monthly_inputs,
                returns,
                signals,
                defensive,
                context,
            )
            metrics = _metric_columns(final, baseline)
            completed += 1
            row: dict[str, object] = {
                "candidate": set_name,
                "transform_family": transform_family,
                "C": c_value,
                "feature_count": len(BASE_FEATURES) + len(feature_names),
                "added_feature_count": len(feature_names),
                "fit_count": int(fit_stats["fit_count"]),
                "convergence_warning_count": int(
                    fit_stats["convergence_warning_count"]
                ),
                **metrics,
            }
            rows.append(row)
            print(
                f"[{completed:02d}/{candidate_total}] {set_name} C={c_value:.2f} "
                f"CAGR={metrics['full_CAGR']:.4f} "
                f"Sharpe={metrics['full_Sharpe']:.4f} "
                f"MDD={metrics['full_MDD']:.4f}",
                flush=True,
            )

    comparison = pd.DataFrame(rows)
    candidate_mask = comparison["transform_family"].isin(["raw_control", "processed"])
    comparison.loc[candidate_mask, "rank_full_score"] = comparison.loc[
        candidate_mask, "full_score"
    ].rank(ascending=False, method="min")
    comparison = comparison.sort_values(
        ["transform_family", "full_score"],
        ascending=[True, False],
    )
    comparison.to_csv(COMPARISON_PATH, index=False)

    processed = comparison.loc[comparison["transform_family"].eq("processed")]
    best_processed = processed.sort_values("full_score", ascending=False).iloc[0]
    eligible = processed.loc[processed["full_all_three_improve"].astype(bool)]
    if eligible.empty:
        deployed_strategy = "Existing_Final_RobustVKOSPI"
        deployment_reason = (
            "No processed VIX6 logistic-input candidate improved full-period CAGR, "
            "Sharpe and MDD simultaneously; keep the existing Robust VKOSPI strategy."
        )
    else:
        deployed_strategy = str(
            eligible.sort_values("full_score", ascending=False).iloc[0]["candidate"]
        )
        deployment_reason = (
            "At least one processed VIX6 logistic-input candidate improved full-period "
            "CAGR, Sharpe and MDD simultaneously. This is an exploratory in-sample "
            "selection and requires a new untouched forward period."
        )

    best_name = str(best_processed["candidate"])
    best_c = float(best_processed["C"])
    best_features = CANDIDATE_SETS[best_name]
    importance, factor, medium, best_final, best_fit_stats = _run_input_candidate(
        best_features,
        best_c,
        data,
        monthly_inputs,
        returns,
        signals,
        defensive,
        context,
    )
    importance.to_csv(BEST_IMPORTANCE_PATH, index=False)
    factor.to_csv(BEST_FACTOR_PATH)
    medium.to_csv(BEST_MEDIUM_PATH)
    best_final.to_csv(BEST_FINAL_PATH)

    actual_start = str(common.min())
    actual_end = str(common.max())
    report: dict[str, object] = {
        "requested_test_period": "2007-2026",
        "actual_common_test_period": {
            "start": actual_start,
            "end": actual_end,
            "months": int(len(common)),
        },
        "evaluation_status": (
            "Full-period exploratory walk-forward backtest. Each monthly prediction "
            "uses only prior training rows with a two-month label embargo, but the "
            "2007-2026 period is used to compare candidates and is not an untouched holdout."
        ),
        "existing_strategy_reproduction_max_return_error": reproduction_error,
        "existing_strategy": {
            "CAGR": float(baseline_metrics["CAGR"]),
            "Sharpe": float(baseline_metrics["Sharpe"]),
            "MDD": float(baseline_metrics["MDD"]),
            "logistic_first_prediction_month": str(base_probability.first_valid_index()),
            "fit_stats": base_fit_stats,
        },
        "decomposition_components": list(COMPONENTS),
        "tested_transforms": {
            "raw": "unprocessed component level/control",
            "ma5": "5-observation trailing moving average",
            "ma21": "21-observation trailing moving average",
            "delta5": "5-observation change",
            "delta21": "21-observation change",
            "z63": "63-observation trailing z-score, clipped to [-6, 6]",
            "z126": "126-observation trailing z-score, clipped to [-6, 6]",
        },
        "candidate_sets": CANDIDATE_SETS,
        "candidate_count_excluding_baseline": candidate_total,
        "monthly_signal_lag": {
            "months": 1,
            "all_signal_months_strictly_before_target": bool(
                (
                    pd.PeriodIndex(monthly_inputs["signal_month"], freq="M")
                    < monthly_inputs.index
                ).all()
            ),
        },
        "training_label_embargo_months": 2,
        "best_processed_candidate": {
            "candidate": best_name,
            "C": best_c,
            "added_features": best_features,
            "fit_stats": best_fit_stats,
            "CAGR": float(best_processed["full_CAGR"]),
            "Sharpe": float(best_processed["full_Sharpe"]),
            "MDD": float(best_processed["full_MDD"]),
            "CAGR_delta": float(best_processed["full_CAGR_delta"]),
            "Sharpe_delta": float(best_processed["full_Sharpe_delta"]),
            "MDD_delta": float(best_processed["full_MDD_delta"]),
            "all_three_improve": bool(best_processed["full_all_three_improve"]),
        },
        "deployment_decision": deployed_strategy,
        "deployment_reason": deployment_reason,
        "artifacts": {
            "monthly_features": str(MONTHLY_FEATURE_PATH.relative_to(ROOT)),
            "comparison": str(COMPARISON_PATH.relative_to(ROOT)),
            "best_importance": str(BEST_IMPORTANCE_PATH.relative_to(ROOT)),
            "best_factor": str(BEST_FACTOR_PATH.relative_to(ROOT)),
            "best_medium": str(BEST_MEDIUM_PATH.relative_to(ROOT)),
            "best_final": str(BEST_FINAL_PATH.relative_to(ROOT)),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    run_experiment(force_features=args.force_features)


if __name__ == "__main__":
    main()
