from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy import (
    ROOT,
    RESULTS,
    VALIDATION_START,
    PERIODS,
    balanced_logistic_spec,
    build_domestic_features,
    build_no_sjm_components,
    build_no_sjm_signals,
    fixed_robust_overlay,
    forward_path_loss,
    metric_record,
    run_neutral_factor_blend,
)
from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import CAL_END, TEST_START, paired_multiobjective_bootstrap
from strategies.stage06_vkospi.vkospi_extended_diagnostics import DOMESTIC_FEATURES, OAP_COMPOSITES
from strategies.stage06_vkospi.vkospi_model_robustness import (
    fit_logistic_candidate,
    make_tail_factor,
    run_factor_vol_target,
)


UPPER_THRESHOLD = 0.20
LOWER_THRESHOLD = -0.20
HARD_WEIGHT = 0.40
LEVERAGE_CAPS = (1.00, 1.10, 1.20, 1.30)
LEVERAGE_FLOOR = 0.50
TARGET_VOL = 0.15

SIGNALS_PATH = RESULTS / "hysteresis_hard40_signals.csv"
FEATURES_PATH = RESULTS / "hysteresis_hard40_features.csv"
FACTOR_PATH = RESULTS / "hysteresis_hard40_factor.csv"
CALIBRATION_PATH = RESULTS / "hysteresis_hard40_leverage_calibration.csv"
COMPARISON_PATH = RESULTS / "hysteresis_hard40_leverage_comparison.csv"
SELECTED_MEDIUM_PATH = RESULTS / "hysteresis_hard40_leverage_selected_medium.csv"
SELECTED_FINAL_PATH = RESULTS / "hysteresis_hard40_leverage_selected_reconciled.csv"
REPORT_PATH = RESULTS / "hysteresis_hard40_leverage_validation.json"


def hysteresis_state(
    values: pd.Series,
    upper: float = UPPER_THRESHOLD,
    lower: float = LOWER_THRESHOLD,
) -> pd.Series:
    """Notebook-equivalent two-threshold state with memory inside the dead band."""
    if not lower < upper:
        raise ValueError("lower threshold must be below upper threshold")
    output = pd.Series(np.nan, index=values.index, dtype=float)
    current = np.nan
    for index, value in values.items():
        if pd.isna(value):
            continue
        if value > upper:
            current = 1.0
        elif value < lower:
            current = -1.0
        elif pd.isna(current):
            current = 1.0 if value >= 0 else -1.0
        output.loc[index] = current
    return output


def _full_macro_hysteresis_states(
    upper: float = UPPER_THRESHOLD,
    lower: float = LOWER_THRESHOLD,
) -> pd.DataFrame:
    """Run state memory on all available history before selecting trade months."""
    macro, _ = load_macro_data()
    growth = macro["growth"][["GDP_level", "Export_level", "BSI_level"]].mean(axis=1)
    inflation = macro["inflation"][["CPI_level", "PPI_level", "ImportPrice_level"]].mean(axis=1)
    states = pd.DataFrame(
        {
            "growth_score": growth,
            "inflation_score": inflation,
            "growth_state": hysteresis_state(growth, upper, lower),
            "inflation_state": hysteresis_state(inflation, upper, lower),
        }
    )
    states.index = states.index.to_period("M")
    return states[~states.index.duplicated(keep="last")]


def build_hysteresis_signals(
    returns: pd.DataFrame,
    upper: float = UPPER_THRESHOLD,
    lower: float = LOWER_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use hysteresis only for the representative regime consumed by Hard 40%."""
    signals, probabilities = build_no_sjm_signals(returns)
    components = build_no_sjm_components(returns)
    state_history = _full_macro_hysteresis_states(upper, lower)

    signal_months = pd.PeriodIndex(components["signal_month"], freq="M")
    aligned = state_history.reindex(signal_months).copy()
    aligned.index = components.index
    if aligned[["growth_state", "inflation_state"]].isna().any().any():
        raise ValueError("hysteresis state is unavailable for one or more signal months")

    regimes = np.select(
        [
            aligned["growth_state"].eq(1) & aligned["inflation_state"].eq(1),
            aligned["growth_state"].eq(-1) & aligned["inflation_state"].eq(1),
            aligned["growth_state"].eq(-1) & aligned["inflation_state"].eq(-1),
            aligned["growth_state"].eq(1) & aligned["inflation_state"].eq(-1),
        ],
        ["Overheating", "Stagflation", "Slowdown", "Goldilocks"],
        default="Unknown",
    )
    if "Unknown" in regimes:
        raise ValueError("unknown hysteresis regime")

    output = signals.copy()
    output["probability_argmax_regime"] = output["regime"]
    output["regime"] = regimes
    output["growth_score"] = aligned["growth_score"]
    output["inflation_score"] = aligned["inflation_score"]
    output["growth_state"] = aligned["growth_state"].astype(int)
    output["inflation_state"] = aligned["inflation_state"].astype(int)
    output["hysteresis_changed_argmax"] = (
        output["regime"] != output["probability_argmax_regime"]
    )
    assert (output["signal_month"] < output.index).all()
    return output, probabilities, components


def _cap_name(cap: float) -> str:
    return f"Hysteresis_Hard40_LevCap{cap:.1f}"


def _safe_cap_token(cap: float) -> str:
    return f"{cap:.1f}".replace(".", "p")


def _fit_factor(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    domestic = build_domestic_features(signals, returns)
    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    composites.index = pd.PeriodIndex(composites.index, freq="M")
    neutral = run_neutral_factor_blend(returns, signals, defensive)
    data = domestic[DOMESTIC_FEATURES].join(composites[OAP_COMPOSITES], how="left")
    data = data.loc[data.index.intersection(neutral.index)].copy()
    path_loss = forward_path_loss(neutral.loc[data.index, "return"], horizon=2)
    data["tail_event"] = (path_loss < -0.05).where(path_loss.notna()).astype(float)
    probability, fit_stats = fit_logistic_candidate(data, balanced_logistic_spec())
    return domestic, make_tail_factor(probability, data["tail_event"]), fit_stats


def _rank_and_select(
    calibration: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, str, int]:
    output = calibration.copy()
    metric_columns = [
        f"{prefix}_{metric}"
        for prefix in ("Cal", "Validation")
        for metric in ("CAGR", "Sharpe", "MDD")
    ]
    for column in metric_columns:
        output[f"Rank_{column}"] = output[column].rank(pct=True, method="average")
    output["MultiObjectiveScore"] = output[
        [f"Rank_{column}" for column in metric_columns]
    ].mean(axis=1)

    baseline_cal = performance_summary(baseline.loc[:CAL_END, "return"])
    baseline_validation = performance_summary(
        baseline.loc[VALIDATION_START:CAL_END, "return"]
    )
    for prefix, metrics in (("Cal", baseline_cal), ("Validation", baseline_validation)):
        for metric in ("CAGR", "Sharpe", "MDD"):
            output[f"{prefix}_{metric}DeltaVsCurrent"] = output[f"{prefix}_{metric}"] - float(
                metrics[metric]
            )

    strict_mask = np.logical_and.reduce(
        [
            output[f"{prefix}_CAGRDeltaVsCurrent"].gt(0)
            & output[f"{prefix}_SharpeDeltaVsCurrent"].gt(0)
            & output[f"{prefix}_MDDDeltaVsCurrent"].ge(0)
            for prefix in ("Cal", "Validation")
        ]
    )
    strict = output.loc[strict_mask]
    if len(strict):
        eligible = strict
        selection_rule = (
            "CAGR, Sharpe and MDD all improve versus the current strategy in both "
            "pre-2018 windows; rank breaks ties"
        )
    else:
        eligible = output
        selection_rule = (
            "No all-three pre-2018 improvement; select the highest equal-weight rank "
            "of CAGR, Sharpe and MDD in 2007-2017 and 2013-2017 as a research candidate"
        )
    winner = eligible.sort_values(
        ["MultiObjectiveScore", "Validation_Sharpe", "Validation_MDD", "Validation_CAGR"],
        ascending=False,
    ).iloc[0]
    output["Selected"] = output["Candidate"].eq(winner["Candidate"])
    output["StrictPrelockPass"] = strict_mask
    selected = output.loc[output["Selected"]].iloc[0]
    return output, selected, selection_rule, int(len(strict))


def main() -> None:
    returns, _ = load_monthly_asset_returns(False)
    signals, probabilities, components = build_hysteresis_signals(returns)
    defensive = run_backtest(returns, signals, StrategyConfig(), mode="proposed")
    domestic, factor, fit_stats = _fit_factor(returns, signals, defensive)

    medium_paths: dict[float, pd.DataFrame] = {}
    final_paths: dict[float, pd.DataFrame] = {}
    for cap in LEVERAGE_CAPS:
        medium = run_factor_vol_target(
            returns,
            signals,
            defensive,
            factor,
            max_shift=0.20,
            target_vol=TARGET_VOL,
            leverage_min=LEVERAGE_FLOOR,
            leverage_max=cap,
            initial_leverage=min(1.20, cap),
        )
        _, _, final = fixed_robust_overlay(medium)
        medium_paths[cap] = medium
        final_paths[cap] = final
        token = _safe_cap_token(cap)
        medium.to_csv(RESULTS / f"hysteresis_hard40_leverage_cap_{token}_medium.csv")
        final.to_csv(RESULTS / f"hysteresis_hard40_leverage_cap_{token}_reconciled.csv")

    current = pd.read_csv(
        RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv", index_col=0
    )
    current.index = pd.PeriodIndex(current.index, freq="M")

    calibration_rows: list[dict[str, object]] = []
    for cap, path in final_paths.items():
        row: dict[str, object] = {"Candidate": _cap_name(cap), "LeverageCap": cap}
        for prefix, start, end in (
            ("Cal", None, CAL_END),
            ("Validation", VALIDATION_START, CAL_END),
        ):
            view = path
            if start is not None:
                view = view.loc[start:]
            if end is not None:
                view = view.loc[:end]
            metrics = performance_summary(view["return"])
            row.update({f"{prefix}_{key}": float(value) for key, value in metrics.items()})
            row[f"{prefix}_AvgLeverage"] = float(
                medium_paths[cap].loc[view.index, "leverage"].mean()
            )
        calibration_rows.append(row)
    calibration, winner, selection_rule, strict_count = _rank_and_select(
        pd.DataFrame(calibration_rows), current
    )
    selected_cap = float(winner["LeverageCap"])
    selected_medium = medium_paths[selected_cap]
    selected_final = final_paths[selected_cap]

    paths: dict[str, pd.DataFrame] = {
        "Current_NoHysteresis_LevCap1.5": current,
        **{_cap_name(cap): path for cap, path in final_paths.items()},
    }
    comparison = pd.DataFrame(
        [
            metric_record(period, strategy, path, start, end)
            for period, start, end in PERIODS
            for strategy, path in paths.items()
        ]
    )

    locked_current = current.loc[TEST_START:, "return"]
    locked_selected = selected_final.loc[TEST_START:, "return"]
    common = locked_current.index.intersection(locked_selected.index)
    locked_current_metrics = performance_summary(locked_current.loc[common])
    locked_selected_metrics = performance_summary(locked_selected.loc[common])
    locked_deltas = {
        metric: float(locked_selected_metrics[metric] - locked_current_metrics[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    full_current_metrics = performance_summary(current["return"])
    full_selected_metrics = performance_summary(selected_final["return"])
    full_deltas = {
        metric: float(full_selected_metrics[metric] - full_current_metrics[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    selected_strict_pass = bool(winner["StrictPrelockPass"])

    regime_switches = int(signals["regime"].ne(signals["regime"].shift()).sum() - 1)
    argmax_switches = int(
        signals["probability_argmax_regime"]
        .ne(signals["probability_argmax_regime"].shift())
        .sum()
        - 1
    )
    report = {
        "objective": "Notebook-style hysteresis + current Hard 40% + leverage cap 1.0-1.3",
        "implementation": {
            "macro_model": "no-SJM causal macro probability model unchanged",
            "hysteresis": {
                "upper": UPPER_THRESHOLD,
                "lower": LOWER_THRESHOLD,
                "inputs": ["mean growth z-score", "mean inflation z-score"],
                "scope": "representative regime used by Hard 40% only",
                "state_rule": "switch high above +0.2, switch low below -0.2, otherwise retain prior state",
                "full_history_initialization": True,
            },
            "allocation": {
                "hard_weight": HARD_WEIGHT,
                "defensive_slsqp_weight": 1 - HARD_WEIGHT,
                "logistic_max_shift": 0.20,
                "target_vol": TARGET_VOL,
                "leverage_floor": LEVERAGE_FLOOR,
                "leverage_caps_tested": list(LEVERAGE_CAPS),
                "vkospi_overlay": "existing Robust VKOSPI overlay unchanged",
            },
            "logistic_fit_stats": fit_stats,
        },
        "selection": {
            "uses_locked_period_for_selection": False,
            "calibration_end": str(CAL_END),
            "validation_window": f"{VALIDATION_START} to {CAL_END}",
            "rule": selection_rule,
            "strict_eligible_count": strict_count,
            "selected_candidate": str(winner["Candidate"]),
            "selected_leverage_cap": selected_cap,
            "selected_strict_prelock_pass": selected_strict_pass,
            "promotion_status": (
                "eligible_to_replace_current_on_prelock_gate"
                if selected_strict_pass
                else "research_candidate_only_current_strategy_preserved"
            ),
        },
        "regime_audit": {
            "months": int(len(signals)),
            "hysteresis_switches": regime_switches,
            "probability_argmax_switches": argmax_switches,
            "months_changed_vs_probability_argmax": int(
                signals["hysteresis_changed_argmax"].sum()
            ),
            "signal_is_strictly_prior_to_target": bool(
                (signals["signal_month"] < signals.index).all()
            ),
        },
        "locked_audit_2018_2026": {
            "current": {key: float(value) for key, value in locked_current_metrics.items()},
            "selected": {key: float(value) for key, value in locked_selected_metrics.items()},
            "delta_selected_minus_current": locked_deltas,
            "passes_all_three": bool(
                locked_deltas["CAGR"] > 0
                and locked_deltas["Sharpe"] > 0
                and locked_deltas["MDD"] >= 0
            ),
            "bootstrap": paired_multiobjective_bootstrap(
                locked_current.loc[common], locked_selected.loc[common]
            ),
        },
        "full_audit_2007_2026": {
            "current": {key: float(value) for key, value in full_current_metrics.items()},
            "selected": {key: float(value) for key, value in full_selected_metrics.items()},
            "delta_selected_minus_current": full_deltas,
            "passes_all_three": bool(
                full_deltas["CAGR"] > 0
                and full_deltas["Sharpe"] > 0
                and full_deltas["MDD"] >= 0
            ),
        },
        "deployment_note": (
            "Candidate outputs are isolated. The current no-hysteresis result files are not overwritten."
        ),
    }

    signals.to_csv(SIGNALS_PATH)
    domestic.to_csv(FEATURES_PATH)
    factor.to_csv(FACTOR_PATH)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    selected_medium.to_csv(SELECTED_MEDIUM_PATH)
    selected_final.to_csv(SELECTED_FINAL_PATH)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== HYSTERESIS + HARD 40% + LEVERAGE CAP ===")
    print(calibration.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== SELECTED / CURRENT ===")
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))
    print(json.dumps(report["full_audit_2007_2026"], ensure_ascii=False, indent=2))
    print(json.dumps(report["locked_audit_2018_2026"], ensure_ascii=False, indent=2))
    print("saved", REPORT_PATH)


if __name__ == "__main__":
    main()
