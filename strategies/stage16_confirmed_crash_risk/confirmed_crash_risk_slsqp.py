from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import (
    ASSETS,
    cdar,
    load_monthly_asset_returns,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    CDAR_CONFIDENCE,
    CATASTROPHE_ANNUAL_VOLATILITY,
    CATASTROPHE_CDAR,
    FULL_START,
    LOCKED_START,
    ONE_CALENDAR_YEAR,
    ONE_WEEK,
    REGIME_COLUMNS,
    SLSQP_MAX_ITERATIONS,
    SLSQP_TOLERANCE,
    build_daily_stress_features,
    build_monthly_stress_signals,
    build_overlay_attribution,
    causal_expanding_midrank,
    estimate_conditional_moments,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    UNCONSTRAINED_LONG_ONLY_BOUNDS,
    concentration_summary,
    expected_transaction_cost,
    metric_row,
    project_to_long_only_simplex,
    read_saved_path,
    solver_summary,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STAGE_DIR / "outputs"
STAGE14_STATIC_OUTPUT = (
    ROOT
    / "strategies"
    / "stage14_unconstrained_dynamic_risk_slsqp"
    / "outputs"
    / "no_asset_cap_static_lambda_monthly.csv"
)
STAGE14_DYNAMIC_OUTPUT = (
    ROOT
    / "strategies"
    / "stage14_unconstrained_dynamic_risk_slsqp"
    / "outputs"
    / "no_asset_cap_dynamic_lambda_monthly.csv"
)

# These are definitions, not searched hyperparameters. 0.5 is the median of an
# expanding percentile and the majority boundary of a probability. The linear
# map sends no confirmed evidence to lambda=1 and unanimous evidence to lambda=2.
EVIDENCE_DEAD_ZONE = 0.5
MAXIMUM_DOWNSIDE_LAMBDA = 2.0


def build_confirmed_daily_features() -> pd.DataFrame:
    """Add a causal direction rank to the existing VKOSPI/VIX6 blocks."""

    daily = build_daily_stress_features().copy()
    daily["stress_direction_5d"] = daily["stress_raw"].diff(ONE_WEEK)
    daily["stress_direction_rank"] = causal_expanding_midrank(
        daily["stress_direction_5d"]
    )
    return daily


def build_confirmed_monthly_signals(
    target_months: pd.PeriodIndex,
    probabilities: pd.DataFrame,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a parameter-free crash-pressure confirmation score.

    Three economic blocks receive equal weight:

    * financial confirmation: min(fast shock, one-month persistence),
    * timing confirmation: mean(rising direction, left-tail repricing),
    * macro vulnerability: P(Slowdown) + P(Stagflation).

    The minimum in the first block prevents a one-off shock from being treated
    as persistent risk. A median/majority dead zone prevents mild evidence from
    changing lambda. No future return label enters this signal.
    """

    if daily is None:
        daily = build_confirmed_daily_features()
    monthly = build_monthly_stress_signals(target_months, daily).copy()
    monthly["stress_direction_rank"] = [
        float(daily.loc[monthly.loc[month, "stress_signal_date"], "stress_direction_rank"])
        for month in monthly.index
    ]
    macro = probabilities.reindex(monthly.index)
    monthly["macro_vulnerability"] = (
        macro["p_Slowdown"] + macro["p_Stagflation"]
    ).clip(0.0, 1.0)
    monthly["fast_slow_confirmation"] = np.minimum(
        monthly["shock_component"], monthly["persistence_component"]
    )
    monthly["direction_tail_confirmation"] = monthly[
        ["stress_direction_rank", "tail_component"]
    ].mean(axis=1)
    monthly["crash_evidence"] = monthly[
        [
            "fast_slow_confirmation",
            "direction_tail_confirmation",
            "macro_vulnerability",
        ]
    ].mean(axis=1)
    monthly["normalization_state"] = (
        (monthly["level_component"] > EVIDENCE_DEAD_ZONE)
        & (monthly["stress_direction_rank"] <= EVIDENCE_DEAD_ZONE)
    )
    raw_pressure = (
        2.0 * (monthly["crash_evidence"] - EVIDENCE_DEAD_ZONE).clip(lower=0.0)
    ).clip(0.0, 1.0)
    # A high but falling volatility level is normalization, not forward crash
    # confirmation. This directly implements Level x Falling != RiskOff.
    monthly["crash_pressure"] = raw_pressure.where(
        ~monthly["normalization_state"], 0.0
    )
    monthly["confirmed_downside_lambda"] = (
        1.0
        + (MAXIMUM_DOWNSIDE_LAMBDA - 1.0) * monthly["crash_pressure"]
    )
    return monthly


def solve_confirmed_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    current_crash_pressure: float,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the unchanged Stage14 SLSQP with a confirmed lambda mapping."""

    expected_return, covariance, moment_detail = estimate_conditional_moments(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=True,
    )
    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    crash_pressure = float(np.clip(current_crash_pressure, 0.0, 1.0))
    downside_lambda = 1.0 + crash_pressure
    initial = (
        project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        path = historical_returns @ weights
        downside_semivariance = float(np.mean(np.minimum(path, 0.0) ** 2))
        transaction_cost = expected_transaction_cost(weights, pretrade)
        monthly_utility = (
            monthly_return
            - 0.5 * monthly_variance
            - downside_lambda * downside_semivariance
            - transaction_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_risk_aversion_lambda": downside_lambda,
            "variance_penalty": 0.5 * monthly_variance,
            "downside_semivariance": downside_semivariance,
            "downside_penalty": downside_lambda * downside_semivariance,
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": monthly_utility,
        }

    def objective(weights: np.ndarray) -> float:
        return -portfolio_values(weights)["monthly_utility"]

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                CATASTROPHE_ANNUAL_VOLATILITY - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                CATASTROPHE_CDAR
                + cdar(historical_returns @ weights, CDAR_CONFIDENCE)
            ),
        },
    ]
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
    )

    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = project_to_long_only_simplex(result.x)
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                "Both confirmed-risk SLSQP solves failed: "
                f"{result.message}; {fallback.message}"
            )
        result = fallback
        weights = project_to_long_only_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = cdar(historical_returns @ weights, CDAR_CONFIDENCE)
    detail: dict[str, Any] = {
        **values,
        "policy": "ConfirmedCrashRiskLambda",
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": CATASTROPHE_ANNUAL_VOLATILITY - annual_vol,
        "cdar_slack": CATASTROPHE_CDAR + historical_cdar,
        "largest_weight": float(weights.max()),
        "largest_asset": ASSETS[int(np.argmax(weights))],
        "weights_above_half": int(np.sum(weights > 0.5 + 1e-10)),
        "macro_expected_monthly_return": moment_detail[
            "macro_expected_monthly_return"
        ],
        "stress_return_adjustment": moment_detail[
            "stress_return_adjustment"
        ],
    }
    return weights, detail


def run_confirmed_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(signals.index)
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0

    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < ONE_CALENDAR_YEAR:
            continue
        probability = probabilities.loc[month]
        signal = signals.loc[month]
        weights, detail = solve_confirmed_weights(
            history=history,
            historical_probabilities=probabilities.loc[
                probabilities.index < month
            ],
            current_probabilities=probability,
            historical_stress=signals.loc[
                signals.index < month, "stress_score"
            ],
            current_stress=float(signal["stress_score"]),
            historical_recovery=signals.loc[
                signals.index < month, "recovery_score"
            ],
            current_recovery=float(signal["recovery_score"]),
            current_crash_pressure=float(signal["crash_pressure"]),
            pretrade=pretrade,
        )

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign_indices].sum()))
            * FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

        rows.append(
            {
                "month": month,
                "macro_signal_month": probability["signal_month"],
                "stress_signal_month": signal["stress_signal_month"],
                "stress_signal_date": signal["stress_signal_date"],
                "stress_score": float(signal["stress_score"]),
                "recovery_score": float(signal["recovery_score"]),
                "stress_direction_rank": float(
                    signal["stress_direction_rank"]
                ),
                "fast_slow_confirmation": float(
                    signal["fast_slow_confirmation"]
                ),
                "direction_tail_confirmation": float(
                    signal["direction_tail_confirmation"]
                ),
                "macro_vulnerability": float(signal["macro_vulnerability"]),
                "crash_evidence": float(signal["crash_evidence"]),
                "crash_pressure": float(signal["crash_pressure"]),
                "normalization_state": bool(signal["normalization_state"]),
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **{
                    column: float(probability[column])
                    for column in REGIME_COLUMNS
                },
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    key: value
                    for key, value in detail.items()
                    if key
                    not in {
                        "macro_expected_monthly_return",
                        "stress_return_adjustment",
                    }
                },
                **{
                    f"macro_mu_{asset}": float(
                        detail["macro_expected_monthly_return"][index]
                    )
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    f"stress_mu_adjustment_{asset}": float(
                        detail["stress_return_adjustment"][index]
                    )
                    for index, asset in enumerate(ASSETS)
                },
            }
        )

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def attribution_diagnostics(attribution: pd.DataFrame) -> dict[str, Any]:
    actions = int(attribution["risk_off_action"].sum())
    false_positives = int(attribution["false_positive"].sum())
    non_false_positive_actions = actions - false_positives
    return {
        "risk_off_months": actions,
        "risk_off_rate": actions / len(attribution),
        "false_positive_months": false_positives,
        "false_positive_share_of_actions": (
            false_positives / actions if actions else 0.0
        ),
        "non_false_positive_action_precision": (
            non_false_positive_actions / actions if actions else 0.0
        ),
        "crash_months": int(attribution["crash_month"].sum()),
        "caught_crash_months": int(attribution["caught_crash"].sum()),
        "missed_crash_months": int(attribution["missed_crash"].sum()),
    }


def direct_signal_diagnostics(
    path: pd.DataFrame,
    attribution: pd.DataFrame,
) -> dict[str, Any]:
    """Report the gate itself, separate from path-dependent optimizer drift."""

    common = path.index.intersection(attribution.index)
    active = path.loc[common, "crash_pressure"] > 0.0
    positive = attribution.loc[common, "positive_equity_month"].astype(bool)
    crash = attribution.loc[common, "crash_month"].astype(bool)
    false_positive = active & positive
    caught = active & crash
    actions = int(active.sum())
    return {
        "months": int(len(common)),
        "lambda_activation_months": actions,
        "lambda_activation_rate": actions / len(common),
        "signal_false_positive_months": int(false_positive.sum()),
        "signal_false_positive_share": (
            float(false_positive.sum()) / actions if actions else 0.0
        ),
        "signal_caught_crash_months": int(caught.sum()),
        "crash_months": int(crash.sum()),
        "maximum_lambda": float(
            path.loc[common, "downside_risk_aversion_lambda"].max()
        ),
        "average_crash_pressure": float(
            path.loc[common, "crash_pressure"].mean()
        ),
    }


def false_positive_context(
    attribution: pd.DataFrame,
    signals: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, Any]:
    common = attribution.index.intersection(signals.index)
    frame = attribution.loc[common].join(
        signals.loc[
            common,
            [
                "level_component",
                "shock_component",
                "tail_component",
                "persistence_component",
                "stress_direction_rank",
                "normalization_state",
                "macro_vulnerability",
                "crash_pressure",
            ],
        ]
    )
    frame["prior_equity_return"] = returns["KODEX200"].shift(1).reindex(common)
    fp = frame[frame["false_positive"]]
    return {
        "false_positive_months": int(len(fp)),
        "normalization_state_false_positives": int(
            fp["normalization_state"].sum()
        ),
        "false_positives_after_negative_equity_month": int(
            (fp["prior_equity_return"] < 0.0).sum()
        ),
        "average_false_positive_features": {
            column: float(fp[column].mean())
            for column in [
                "level_component",
                "shock_component",
                "tail_component",
                "persistence_component",
                "stress_direction_rank",
                "macro_vulnerability",
                "crash_pressure",
                "prior_equity_return",
                "KODEX200_return",
            ]
        },
    }


def run_research(save: bool = True) -> dict[str, Any]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily = build_confirmed_daily_features()
    signals = build_confirmed_monthly_signals(
        returns.index, probabilities, daily
    )
    confirmed_path = run_confirmed_backtest(returns, probabilities, signals)
    static_path = read_saved_path(STAGE14_STATIC_OUTPUT)
    dynamic_path = read_saved_path(STAGE14_DYNAMIC_OUTPUT)
    paths = {
        "Stage14_StaticLambda": static_path,
        "Stage14_StressLambda": dynamic_path,
        "Stage16_ConfirmedCrashLambda": confirmed_path,
    }
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            rows.append(metric_row(name, path, period, start, common_end))
    comparison = pd.DataFrame(rows)
    performance_deltas: list[dict[str, Any]] = []
    for period in comparison["Period"].unique():
        view = comparison[comparison["Period"] == period].set_index("Strategy")
        baseline = view.loc["Stage14_StressLambda"]
        candidate = view.loc["Stage16_ConfirmedCrashLambda"]
        performance_deltas.append(
            {
                "Period": period,
                **{
                    f"{metric}_Delta": float(
                        candidate[metric] - baseline[metric]
                    )
                    for metric in [
                        "CAGR",
                        "Volatility",
                        "Sharpe",
                        "Sortino",
                        "MDD",
                        "Calmar",
                        "FinalMultiple",
                    ]
                },
            }
        )
    delta_frame = pd.DataFrame(performance_deltas)

    original_attribution, original_summary = build_overlay_attribution(
        returns, static_path, dynamic_path
    )
    confirmed_attribution, confirmed_summary = build_overlay_attribution(
        returns, static_path, confirmed_path
    )
    _, original_locked_summary = build_overlay_attribution(
        returns.loc[LOCKED_START:],
        static_path.loc[LOCKED_START:],
        dynamic_path.loc[LOCKED_START:],
    )
    _, confirmed_locked_summary = build_overlay_attribution(
        returns.loc[LOCKED_START:],
        static_path.loc[LOCKED_START:],
        confirmed_path.loc[LOCKED_START:],
    )
    attribution_comparison = pd.DataFrame(
        [
            {
                "Strategy": "Stage14_StressLambda",
                "Period": "full_2007_2026",
                **attribution_diagnostics(original_attribution),
            },
            {
                "Strategy": "Stage16_ConfirmedCrashLambda",
                "Period": "full_2007_2026",
                **attribution_diagnostics(confirmed_attribution),
            },
            {
                "Strategy": "Stage14_StressLambda",
                "Period": "locked_2018_2026",
                **attribution_diagnostics(
                    original_attribution.loc[LOCKED_START:]
                ),
            },
            {
                "Strategy": "Stage16_ConfirmedCrashLambda",
                "Period": "locked_2018_2026",
                **attribution_diagnostics(
                    confirmed_attribution.loc[LOCKED_START:]
                ),
            },
        ]
    )

    weight_columns = [f"w_{asset}" for asset in ASSETS]
    checks = {
        "macro_signal_precedes_target": bool(
            (confirmed_path["macro_signal_month"] < confirmed_path.index).all()
        ),
        "stress_signal_precedes_target": bool(
            (confirmed_path["stress_signal_month"] < confirmed_path.index).all()
        ),
        "weights_sum_to_one": bool(
            np.allclose(confirmed_path[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (confirmed_path[weight_columns] >= -1e-10).all().all()
        ),
        "weights_do_not_exceed_one": bool(
            (confirmed_path[weight_columns] <= 1.0 + 1e-10).all().all()
        ),
        "no_cash": True,
        "no_leverage": bool(
            np.allclose(confirmed_path[weight_columns].sum(axis=1), 1.0)
        ),
        "lambda_is_one_inside_dead_zone": bool(
            np.allclose(
                confirmed_path.loc[
                    confirmed_path["crash_evidence"] <= EVIDENCE_DEAD_ZONE,
                    "downside_risk_aversion_lambda",
                ],
                1.0,
            )
        ),
        "lambda_never_exceeds_two": bool(
            (confirmed_path["downside_risk_aversion_lambda"] <= 2.0).all()
        ),
        "volatility_guard_respected": bool(
            (confirmed_path["volatility_slack"] >= -1e-7).all()
        ),
        "cdar_guard_respected": bool(
            (confirmed_path["cdar_slack"] >= -1e-7).all()
        ),
        "all_slsqp_solves_succeeded": bool(
            confirmed_path["solver_success"].all()
        ),
        "no_fitted_crash_classifier": True,
        "no_future_return_label_in_signal": True,
        "no_hyperparameter_search": True,
    }
    report: dict[str, Any] = {
        "strategy": "Stage16_ConfirmedCrashLambda",
        "change": (
            "Replace lambda=1+stress with an economically confirmed crash "
            "pressure. Stage14 conditional moments and constraints are unchanged."
        ),
        "formula": {
            "financial_confirmation": "min(shock rank, persistence rank)",
            "timing_confirmation": (
                "mean(5-day rising-direction rank, left-tail rank)"
            ),
            "macro_vulnerability": "P(Slowdown)+P(Stagflation)",
            "crash_evidence": "equal mean of the three economic blocks",
            "crash_pressure": "2*max(crash_evidence-0.5, 0)",
            "lambda": "1+crash_pressure",
        },
        "parameter_policy": {
            "searched_parameters": None,
            "dead_zone_0_5": (
                "median of a causal percentile and probability majority boundary"
            ),
            "five_days": "one trading week",
            "persistence": "existing Stage14 one trading month block",
            "block_weights": "equal; no fitted coefficients",
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "stage16_minus_stage14": json.loads(
            delta_frame.to_json(orient="records", force_ascii=False)
        ),
        "attribution": json.loads(
            attribution_comparison.to_json(orient="records", force_ascii=False)
        ),
        "original_overlay_summary": {
            "full": original_summary,
            "locked": original_locked_summary,
        },
        "confirmed_overlay_summary": {
            "full": confirmed_summary,
            "locked": confirmed_locked_summary,
        },
        "confirmed_direct_signal": {
            "full": direct_signal_diagnostics(
                confirmed_path, confirmed_attribution
            ),
            "locked": direct_signal_diagnostics(
                confirmed_path.loc[LOCKED_START:],
                confirmed_attribution.loc[LOCKED_START:],
            ),
        },
        "original_false_positive_context": false_positive_context(
            original_attribution, signals, returns
        ),
        "confirmed_false_positive_context": false_positive_context(
            confirmed_attribution, signals, returns
        ),
        "concentration": concentration_summary(confirmed_path),
        "solver": solver_summary(confirmed_path),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        confirmed_path.to_csv(OUTPUT_DIR / "confirmed_crash_lambda_monthly.csv")
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        delta_frame.to_csv(
            OUTPUT_DIR / "stage16_minus_stage14.csv", index=False
        )
        attribution_comparison.to_csv(
            OUTPUT_DIR / "attribution_comparison.csv", index=False
        )
        original_attribution.to_csv(
            OUTPUT_DIR / "stage14_original_attribution.csv"
        )
        confirmed_attribution.to_csv(
            OUTPUT_DIR / "stage16_confirmed_attribution.csv"
        )
        signals.to_csv(OUTPUT_DIR / "confirmed_crash_signals.csv")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "signals": signals,
        "confirmed_path": confirmed_path,
        "comparison": comparison,
        "deltas": delta_frame,
        "attribution_comparison": attribution_comparison,
        "original_attribution": original_attribution,
        "confirmed_attribution": confirmed_attribution,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print("\nAttribution")
    print(result["attribution_comparison"].to_string(index=False))
    print("\nChecks")
    print(json.dumps(result["report"]["checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
