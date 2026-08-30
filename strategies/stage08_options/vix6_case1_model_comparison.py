from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy import (
    build_domestic_features,
    build_no_sjm_signals,
    forward_path_loss,
    run_neutral_factor_blend,
)
from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage08_options.vix6_case1_strategy import (
    CALIBRATION_END,
    VALIDATION_START,
    Case1Config,
    build_aligned_case1_inputs,
    build_vix6_features,
    case1_stress,
    simulate_case1,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    CAL_END,
    TEST_START,
    build_daily_vkospi_signals,
    load_daily_open_levels,
    prepare_arrays,
    reconcile_to_monthly_reference,
    simulate,
)
from strategies.stage06_vkospi.vkospi_extended_diagnostics import DOMESTIC_FEATURES, OAP_COMPOSITES
from strategies.stage06_vkospi.vkospi_model_robustness import make_tail_factor, run_factor_vol_target
from strategies.stage06_vkospi.vkospi_robust_dynamic_experiment import (
    RobustStressConfig,
    align_features_to_arrays,
    build_robust_daily_features,
    stress_from_features,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MONTHLY_INPUT_PATH = RESULTS / "vix6_case1_input_features_monthly.csv"
SEARCH_PATH = RESULTS / "vix6_case1_input_feature_search.csv"
IMPORTANCE_PATH = RESULTS / "vix6_case1_input_feature_importance.csv"
INPUT_FACTOR_PATH = RESULTS / "vix6_case1_input_best_factor.csv"
INPUT_MEDIUM_PATH = RESULTS / "vix6_case1_input_best_medium.csv"
INPUT_DAILY_PATH = RESULTS / "vix6_case1_input_best_daily.csv"
INPUT_FINAL_PATH = RESULTS / "vix6_case1_input_best_reconciled.csv"
STANDALONE_DAILY_PATH = RESULTS / "vix6_case1_standalone_daily.csv"
STANDALONE_FINAL_PATH = RESULTS / "vix6_case1_standalone_reconciled.csv"
FINAL_COMPARISON_PATH = RESULTS / "vix6_case1_final_model_comparison.csv"
FINAL_SELECTION_PATH = RESULTS / "vix6_case1_final_selection.json"
SELECTED_PATH = RESULTS / "vix6_case1_selected_final_reconciled.csv"

MDD_TOLERANCE = -1e-12
BASE_FEATURES = DOMESTIC_FEATURES + OAP_COMPOSITES
VIX6_INPUT_SETS: dict[str, list[str]] = {
    "meta3": ["asymmetry_last", "breadth_z_last", "reaction_z_last"],
    "meta5": [
        "left_tail_last",
        "right_tail_last",
        "asymmetry_last",
        "breadth_z_last",
        "reaction_z_last",
    ],
    "impulse5": [
        "asymmetry_last",
        "breadth_z_last",
        "reaction_z_last",
        "left_impulse_z_last",
        "left_change_5_last",
    ],
    "monthly6": [
        "asymmetry_mean",
        "asymmetry_max",
        "breadth_z_mean",
        "breadth_z_max",
        "reaction_z_mean",
        "left_impulse_z_max",
    ],
    "state6": [
        "left_tail_last",
        "asymmetry_last",
        "breadth_z_last",
        "left_impulse_z_last",
        "left_fading_frac",
        "left_alert_frac",
    ],
    "components6": [
        "sticky_strike_last",
        "parallel_shift_last",
        "put_skew_last",
        "downside_convexity_last",
        "asymmetry_last",
        "breadth_z_last",
    ],
    "meta_state8": [
        "left_tail_last",
        "right_tail_last",
        "asymmetry_mean",
        "asymmetry_max",
        "breadth_z_max",
        "reaction_z_max",
        "left_fading_frac",
        "left_alert_frac",
    ],
}
C_VALUES = (0.01, 0.03, 0.10, 0.30)


@dataclass
class OverlayContext:
    levels: pd.DataFrame
    vkospi_signals: pd.DataFrame
    robust_daily: pd.DataFrame
    config: RobustStressConfig


def build_monthly_vix6_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only signal-month observations and assign them to the next month."""
    daily = features.copy()
    daily["signal_month"] = daily.index.to_period("M")
    daily["left_fading"] = (
        (daily["left_change_5"] < 0) & (daily["left_tail"] > 0)
    ).astype(float)
    daily["left_alert"] = (
        (daily["asymmetry"] > 0.50) | (daily["left_impulse_z"] > 0.50)
    ).astype(float)
    grouped = daily.groupby("signal_month")
    monthly = pd.DataFrame(index=pd.PeriodIndex(sorted(daily["signal_month"].unique()), freq="M"))
    for column in (
        "left_tail",
        "right_tail",
        "asymmetry",
        "breadth_z",
        "reaction_z",
        "left_impulse_z",
        "left_change_5",
        "parallel_shift",
        "sticky_strike",
        "put_skew",
        "downside_convexity",
    ):
        monthly[f"{column}_last"] = grouped[column].last()
    for column in (
        "left_tail",
        "right_tail",
        "asymmetry",
        "breadth_z",
        "reaction_z",
        "left_impulse_z",
    ):
        monthly[f"{column}_mean"] = grouped[column].mean()
        monthly[f"{column}_max"] = grouped[column].max()
    monthly["left_fading_frac"] = grouped["left_fading"].mean()
    monthly["left_alert_frac"] = grouped["left_alert"].mean()
    monthly["signal_month"] = monthly.index.astype(str)
    monthly.index = monthly.index + 1
    monthly.index.name = "target_month"
    assert (
        pd.PeriodIndex(monthly["signal_month"], freq="M") < monthly.index
    ).all()
    return monthly


def make_balanced_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def fit_expanding_logistic(
    data: pd.DataFrame,
    feature_names: list[str],
    c_value: float,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    """Walk-forward fit with the deployed two-month label embargo and guards."""
    probability = pd.Series(np.nan, index=data.index, dtype=float)
    coefficient_rows: list[dict[str, float | str]] = []
    convergence_warnings = 0
    for number, month in enumerate(data.index):
        train_end = number - 2
        if train_end < 36:
            continue
        train = data.iloc[:train_end].dropna(subset=["tail_event"])
        target = train["tail_event"].astype(int)
        if target.sum() < 4 or (len(target) - target.sum()) < 12:
            continue
        model = make_balanced_model(c_value)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train[feature_names], target)
        convergence_warnings += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        probability.loc[month] = float(
            model.predict_proba(data.loc[[month], feature_names])[:, 1][0]
        )
        coefficients = model.named_steps["model"].coef_[0]
        coefficient_rows.extend(
            {
                "prediction_month": str(month),
                "feature": feature,
                "standardized_coefficient": float(value),
            }
            for feature, value in zip(feature_names, coefficients)
        )
    coefficients = pd.DataFrame(coefficient_rows)
    if coefficients.empty:
        importance = pd.DataFrame(
            columns=["feature", "mean_coefficient", "mean_absolute_coefficient"]
        )
    else:
        importance = (
            coefficients.groupby("feature")["standardized_coefficient"]
            .agg(mean_coefficient="mean", mean_absolute_coefficient=lambda x: x.abs().mean())
            .reset_index()
            .sort_values("mean_absolute_coefficient", ascending=False)
        )
    stats = {
        "fit_count": float(probability.notna().sum()),
        "convergence_warning_count": float(convergence_warnings),
    }
    return probability, importance, stats


def build_research_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
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
    return returns, signals, defensive, neutral, data


def build_overlay_context() -> OverlayContext:
    report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(encoding="utf-8")
    )
    return OverlayContext(
        levels=load_daily_open_levels(),
        vkospi_signals=build_daily_vkospi_signals(),
        robust_daily=build_robust_daily_features(),
        config=RobustStressConfig(**report["winner"]),
    )


def apply_existing_vkospi_overlay(
    medium: pd.DataFrame,
    context: OverlayContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = prepare_arrays(context.levels, medium, context.vkospi_signals)
    robust = align_features_to_arrays(context.robust_daily, arrays)
    stress = stress_from_features(
        robust,
        context.config.mode,
        context.config.level_threshold,
        context.config.shock_threshold,
    )
    _, neutral_monthly = simulate(arrays, None, keep_daily=False)
    daily, overlay_monthly = simulate(
        arrays,
        context.config.dynamic_config(),
        keep_daily=True,
        stress_override=stress,
    )
    reconciled = reconcile_to_monthly_reference(medium, neutral_monthly, overlay_monthly)
    return daily, reconciled


def _metrics(
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


def _candidate_row(
    name: str,
    c_value: float,
    final: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float | str | bool]:
    row: dict[str, float | str | bool] = {"set": name, "C": c_value}
    windows = (
        ("cal", None, CALIBRATION_END),
        ("val", VALIDATION_START, CAL_END),
        ("locked", TEST_START, None),
    )
    for prefix, start, end in windows:
        candidate_metrics = _metrics(final, start, end)
        baseline_metrics = _metrics(baseline, start, end)
        for metric in ("CAGR", "Sharpe", "MDD"):
            row[f"{prefix}_{metric}"] = float(candidate_metrics[metric])
            row[f"{prefix}_{metric}d"] = float(
                candidate_metrics[metric] - baseline_metrics[metric]
            )
    row["prelock_gate"] = bool(
        row["cal_CAGRd"] > 0
        and row["cal_Sharped"] > 0
        and row["cal_MDDd"] >= MDD_TOLERANCE
        and row["val_CAGRd"] > 0
        and row["val_Sharped"] > 0
        and row["val_MDDd"] >= MDD_TOLERANCE
    )
    cal_score = (
        float(row["cal_CAGRd"]) / 0.01
        + float(row["cal_Sharped"]) / 0.05
        + float(row["cal_MDDd"]) / 0.01
    )
    validation_score = (
        float(row["val_CAGRd"]) / 0.01
        + float(row["val_Sharped"]) / 0.05
        + float(row["val_MDDd"]) / 0.01
    )
    row["prelock_score"] = min(cal_score, validation_score)
    return row


def search_input_candidates(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    data: pd.DataFrame,
    monthly_inputs: pd.DataFrame,
    baseline: pd.DataFrame,
    context: OverlayContext,
    force: bool,
) -> pd.DataFrame:
    expected = len(VIX6_INPUT_SETS) * len(C_VALUES)
    if SEARCH_PATH.exists() and not force:
        cached = pd.read_csv(SEARCH_PATH)
        if len(cached) == expected and {"set", "C"}.issubset(cached.columns):
            # Recompute the gate with a numerical MDD tolerance. An older cache
            # may have treated a -1e-16 reconciliation residual as a failure.
            cached["prelock_gate"] = (
                (cached["cal_CAGRd"] > 0)
                & (cached["cal_Sharped"] > 0)
                & (cached["cal_MDDd"] >= MDD_TOLERANCE)
                & (cached["val_CAGRd"] > 0)
                & (cached["val_Sharped"] > 0)
                & (cached["val_MDDd"] >= MDD_TOLERANCE)
            )
            if "prelock_score" not in cached:
                cached["prelock_score"] = cached.get("score", np.nan)
            cached.to_csv(SEARCH_PATH, index=False)
            return cached

    rows: list[dict[str, float | str | bool]] = []
    for set_name, extra_features in VIX6_INPUT_SETS.items():
        candidate_data = data.join(monthly_inputs[extra_features], how="left")
        feature_names = BASE_FEATURES + extra_features
        for c_value in C_VALUES:
            probability, _, _ = fit_expanding_logistic(
                candidate_data, feature_names, c_value
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
            rows.append(_candidate_row(set_name, c_value, final, baseline))
    search = pd.DataFrame(rows).sort_values(
        ["prelock_gate", "prelock_score"], ascending=False
    )
    search.to_csv(SEARCH_PATH, index=False)
    return search


def select_input_candidate(search: pd.DataFrame) -> pd.Series:
    eligible = search.loc[search["prelock_gate"].astype(bool)]
    pool = eligible if not eligible.empty else search
    return pool.sort_values("prelock_score", ascending=False).iloc[0]


def build_standalone_vix6_overlay(
    medium: pd.DataFrame,
    signals: pd.DataFrame,
    option_features: pd.DataFrame,
    context: OverlayContext,
) -> tuple[pd.DataFrame, pd.DataFrame, Case1Config]:
    """Replace VKOSPI stress with the best small-grid VIX6-only candidate."""
    config = Case1Config(
        tail_threshold=1.0,
        breadth_threshold=0.5,
        early_scale=0.10,
        confirmed_scale=0.80,
        recovery_relief=0.50,
        right_tail_relief=0.25,
        panic_relief=0.0,
        max_risk_transfer=0.25,
        rebalance_band=0.20,
        oil_cut_deflation=1.0,
        oil_cut_inflation=1.0,
        bond_share_deflation=0.0,
        bond_share_inflation=0.0,
    )
    arrays = prepare_arrays(context.levels, medium, context.vkospi_signals)
    aligned = build_aligned_case1_inputs(
        arrays, context.levels, signals, option_features
    )
    aligned["base_vkospi_stress"] = 0.0
    diagnostics = case1_stress(aligned, config)
    daily, monthly = simulate_case1(
        arrays, aligned, config, diagnostics=diagnostics, keep_daily=True
    )
    _, neutral_monthly = simulate(arrays, None, keep_daily=False)
    reconciled = reconcile_to_monthly_reference(medium, neutral_monthly, monthly)
    return daily, reconciled, config


def metric_row(
    period: str,
    strategy: str,
    path: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, float | int | str]:
    metrics = _metrics(path, start, end)
    return {
        "Period": period,
        "Strategy": strategy,
        **{key: float(value) for key, value in metrics.items()},
        "AvgTurnover": float(_view(path, start, end)["turnover"].mean()),
    }


def _view(
    path: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> pd.DataFrame:
    output = path
    if start is not None:
        output = output.loc[start:]
    if end is not None:
        output = output.loc[:end]
    return output


def run_comparison(force_search: bool = False) -> dict[str, object]:
    option_features = build_vix6_features(False)
    monthly_inputs = build_monthly_vix6_inputs(option_features)
    monthly_inputs.to_csv(MONTHLY_INPUT_PATH)
    returns, signals, defensive, neutral, data = build_research_data()
    context = build_overlay_context()

    base_probability, _, _ = fit_expanding_logistic(data, BASE_FEATURES, 0.10)
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
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv", index_col=0
    )
    stored_baseline.index = pd.PeriodIndex(stored_baseline.index, freq="M")
    common = baseline.index.intersection(stored_baseline.index)
    reproduction_error = float(
        (baseline.loc[common, "return"] - stored_baseline.loc[common, "return"])
        .abs()
        .max()
    )
    assert reproduction_error < 1e-12

    search = search_input_candidates(
        returns,
        signals,
        defensive,
        data,
        monthly_inputs,
        baseline,
        context,
        force_search,
    )
    selected_input = select_input_candidate(search)
    input_set = str(selected_input["set"])
    input_c = float(selected_input["C"])
    input_features = VIX6_INPUT_SETS[input_set]
    input_data = data.join(monthly_inputs[input_features], how="left")
    input_probability, importance, fit_stats = fit_expanding_logistic(
        input_data, BASE_FEATURES + input_features, input_c
    )
    input_factor = make_tail_factor(input_probability, input_data["tail_event"])
    input_medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        input_factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    input_daily, input_final = apply_existing_vkospi_overlay(input_medium, context)

    standalone_daily, standalone_final, standalone_config = (
        build_standalone_vix6_overlay(
            base_medium, signals, option_features, context
        )
    )
    hybrid = pd.read_csv(RESULTS / "vix6_case1_reconciled.csv", index_col=0)
    hybrid.index = pd.PeriodIndex(hybrid.index, freq="M")

    paths = {
        "Existing_Final_RobustVKOSPI": baseline,
        "VIX6_Hybrid_Overlay": hybrid,
        "VIX6_Logistic_Input": input_final,
        "VIX6_Standalone_Overlay": standalone_final,
    }
    periods = (
        ("calibration_2007_2012", None, CALIBRATION_END),
        ("validation_2013_2017", VALIDATION_START, CAL_END),
        ("prelock_2007_2017", None, CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", None, None),
    )
    comparison = pd.DataFrame(
        [
            metric_row(period, strategy, path, start, end)
            for period, start, end in periods
            for strategy, path in paths.items()
        ]
    )
    locked = comparison.loc[comparison["Period"].eq("locked_2018_2026")].set_index(
        "Strategy"
    )
    baseline_locked = locked.loc["Existing_Final_RobustVKOSPI"]
    alternatives: dict[str, dict[str, object]] = {}
    for strategy in paths:
        if strategy == "Existing_Final_RobustVKOSPI":
            continue
        row = locked.loc[strategy]
        deltas = {
            metric: float(row[metric] - baseline_locked[metric])
            for metric in ("CAGR", "Sharpe", "MDD")
        }
        alternatives[strategy] = {
            "locked_metrics": {
                metric: float(row[metric]) for metric in ("CAGR", "Sharpe", "MDD")
            },
            "delta_minus_existing": deltas,
            "all_three_improve": bool(
                deltas["CAGR"] > 0
                and deltas["Sharpe"] > 0
                and deltas["MDD"] >= MDD_TOLERANCE
            ),
        }
    winners = [
        strategy
        for strategy, result in alternatives.items()
        if result["all_three_improve"]
    ]
    if winners:
        selected_strategy = max(
            winners,
            key=lambda strategy: (
                alternatives[strategy]["delta_minus_existing"]["CAGR"] / 0.01
                + alternatives[strategy]["delta_minus_existing"]["Sharpe"] / 0.05
                + alternatives[strategy]["delta_minus_existing"]["MDD"] / 0.01
            ),
        )
        decision = "An alternative improved locked CAGR, Sharpe and MDD simultaneously."
    else:
        selected_strategy = "Existing_Final_RobustVKOSPI"
        decision = (
            "Keep the existing final strategy: no VIX6 alternative improved locked "
            "CAGR, Sharpe and MDD simultaneously."
        )

    valid_input_signal = input_daily["signal_date"].notna()
    input_lag_ok = bool(
        (
            input_daily.index[valid_input_signal]
            > pd.DatetimeIndex(input_daily.loc[valid_input_signal, "signal_date"])
        ).all()
    )
    valid_standalone_signal = standalone_daily["signal_date"].notna()
    standalone_lag_ok = bool(
        (
            standalone_daily.index[valid_standalone_signal]
            > pd.DatetimeIndex(
                standalone_daily.loc[valid_standalone_signal, "signal_date"]
            )
        ).all()
    )

    INPUT_FACTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    input_factor.to_csv(INPUT_FACTOR_PATH)
    input_medium.to_csv(INPUT_MEDIUM_PATH)
    input_daily.to_csv(INPUT_DAILY_PATH)
    input_final.to_csv(INPUT_FINAL_PATH)
    standalone_daily.to_csv(STANDALONE_DAILY_PATH)
    standalone_final.to_csv(STANDALONE_FINAL_PATH)
    comparison.to_csv(FINAL_COMPARISON_PATH, index=False)
    paths[selected_strategy].to_csv(SELECTED_PATH)

    report: dict[str, object] = {
        "selected_strategy": selected_strategy,
        "decision": decision,
        "development_status": (
            "Post-lock exploratory comparison requested by the user; locked metrics "
            "were observed during iteration and are used here for the explicit final "
            "keep-or-replace decision."
        ),
        "existing_reproduction_max_return_error": reproduction_error,
        "tradable_assets": list(ASSETS),
        "option_is_allocated_asset": False,
        "vix6_six_components": [
            "sticky_strike",
            "parallel_shift",
            "put_skew",
            "call_skew",
            "downside_convexity",
            "upside_convexity",
        ],
        "input_candidate": {
            "selected_set": input_set,
            "C": input_c,
            "base_features": BASE_FEATURES,
            "added_vix6_features": input_features,
            "all_features": BASE_FEATURES + input_features,
            "fit_stats": fit_stats,
            "prelock_gate": bool(selected_input["prelock_gate"]),
            "candidate_count": int(len(search)),
            "selection_uses_locked_metrics": False,
        },
        "standalone_overlay": {
            "config": asdict(standalone_config),
            "vkospi_stress_forced_to_zero": True,
        },
        "alternatives": alternatives,
        "lookahead_audit": {
            "monthly_input_signal_month_strictly_before_target": bool(
                (
                    pd.PeriodIndex(monthly_inputs["signal_month"], freq="M")
                    < monthly_inputs.index
                ).all()
            ),
            "input_overlay_signal_strictly_before_action_open": input_lag_ok,
            "standalone_signal_strictly_before_action_open": standalone_lag_ok,
            "training_label_embargo_months": 2,
        },
        "locked_existing": {
            metric: float(baseline_locked[metric])
            for metric in ("CAGR", "Sharpe", "MDD")
        },
        "comparison": json.loads(comparison.to_json(orient="records")),
    }
    FINAL_SELECTION_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected_strategy": selected_strategy,
        "decision": decision,
        "locked_existing": report["locked_existing"],
        "alternatives": alternatives,
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-search", action="store_true")
    args = parser.parse_args()
    run_comparison(force_search=args.force_search)


if __name__ == "__main__":
    main()
