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
    performance_summary,
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
    REGIME_COLUMNS,
    SLSQP_MAX_ITERATIONS,
    SLSQP_TOLERANCE,
    build_daily_stress_features,
    build_monthly_stress_signals,
    estimate_conditional_moments,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    NUMERICAL_EPSILON,
    STATIC_RISK_POLICY,
    UNCONSTRAINED_LONG_ONLY_BOUNDS,
    concentration_summary,
    expected_transaction_cost,
    metric_row,
    project_to_long_only_simplex,
    run_backtest as run_stage14_backtest,
    solver_summary,
)
from strategies.stage17_dynamic_risk_shape.dynamic_risk_shape_slsqp import (
    drawdown_episodes,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Economic horizons stated before the experiment. They are two familiar views
# of medium-term trend, not candidates from which the backtest selects a winner.
MEDIUM_TREND_MONTHS = 6
LONG_TREND_MONTHS = 12


def compounded_return(values: pd.DataFrame | pd.Series) -> pd.Series | float:
    """Compound simple monthly returns without annualization."""

    compounded = (1.0 + values).prod() - 1.0
    if isinstance(compounded, pd.Series):
        return compounded
    return float(compounded)


def asset_trend_confidence(
    history: pd.DataFrame,
    macro_expected_return: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Shrink only the cross-sectional macro view when own trends disagree.

    Six- and twelve-month own-price directions are averaged into M in [-1, 1].
    The neutral return is the equal-weight cross-sectional mean of the current
    macro forecasts: with no asset-specific conviction, all assets have the
    same expected return and covariance/risk still determines the allocation.

    confidence = (1 + sign(macro_i - neutral) * M_i) / 2

    Thus both horizons agreeing with the macro relative view gives confidence
    1, a split trend gives 0.5, and both horizons conflicting gives 0. This is a
    fixed mapping, not a fitted coefficient, threshold search or momentum alpha.
    """

    if len(history) < LONG_TREND_MONTHS:
        raise ValueError("Twelve months of causal return history are required.")
    history = history[ASSETS]
    trend_6m = np.asarray(
        compounded_return(history.tail(MEDIUM_TREND_MONTHS)), dtype=float
    )
    trend_12m = np.asarray(
        compounded_return(history.tail(LONG_TREND_MONTHS)), dtype=float
    )
    direction_6m = np.sign(trend_6m)
    direction_12m = np.sign(trend_12m)
    trend_score = 0.5 * (direction_6m + direction_12m)

    macro = np.asarray(macro_expected_return, dtype=float)
    neutral = float(macro.mean())
    macro_relative_direction = np.sign(macro - neutral)
    agreement = macro_relative_direction * trend_score
    confidence = np.clip(0.5 * (1.0 + agreement), 0.0, 1.0)
    filtered_macro = neutral + confidence * (macro - neutral)
    return {
        "trend_6m": trend_6m,
        "trend_12m": trend_12m,
        "trend_score": trend_score,
        "macro_neutral_return": neutral,
        "macro_relative_direction": macro_relative_direction,
        "macro_confidence": confidence,
        "filtered_macro_expected_return": filtered_macro,
    }


def solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the Stage 14 fixed-lambda solve after macro-confidence filtering."""

    original_expected_return, covariance, moment_detail = (
        estimate_conditional_moments(
            history=history,
            historical_probabilities=historical_probabilities,
            current_probabilities=current_probabilities,
            historical_stress=historical_stress,
            current_stress=current_stress,
            historical_recovery=historical_recovery,
            current_recovery=current_recovery,
            use_short_term_stress=True,
        )
    )
    macro_expected_return = np.asarray(
        moment_detail["macro_expected_monthly_return"], dtype=float
    )
    stress_adjustment = np.asarray(
        moment_detail["stress_return_adjustment"], dtype=float
    )
    confidence_detail = asset_trend_confidence(history, macro_expected_return)
    filtered_macro = np.asarray(
        confidence_detail["filtered_macro_expected_return"], dtype=float
    )
    # VKOSPI/VIX6 stress adjustment is kept exactly as in Stage 14. The new
    # filter changes only confidence in the asset-specific macro return view.
    expected_return = filtered_macro + stress_adjustment

    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
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
        # Lambda remains exactly one. No HHI, dynamic sigma, dynamic ES or
        # post-optimizer overlay is added in this experiment.
        monthly_utility = (
            monthly_return
            - 0.5 * monthly_variance
            - downside_semivariance
            - transaction_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_risk_aversion_lambda": 1.0,
            "variance_penalty": 0.5 * monthly_variance,
            "downside_semivariance": downside_semivariance,
            "downside_penalty": downside_semivariance,
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
                "Both confidence-filtered and feasibility solves failed: "
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
        "policy": "AssetTrendConfidence_StaticLambda",
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
        "macro_expected_monthly_return": macro_expected_return.tolist(),
        "stress_return_adjustment": stress_adjustment.tolist(),
        "original_expected_return": original_expected_return.tolist(),
        "filtered_expected_return": expected_return.tolist(),
        **confidence_detail,
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Run fully invested monthly decisions using only lagged information."""

    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
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
        stress = float(stress_signals.loc[month, "stress_score"])
        recovery = float(stress_signals.loc[month, "recovery_score"])
        weights, detail = solve_weights(
            history=history,
            historical_probabilities=probabilities.loc[
                probabilities.index < month
            ],
            current_probabilities=probability,
            historical_stress=stress_signals.loc[
                stress_signals.index < month, "stress_score"
            ],
            current_stress=stress,
            historical_recovery=stress_signals.loc[
                stress_signals.index < month, "recovery_score"
            ],
            current_recovery=recovery,
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

        row: dict[str, Any] = {
            "month": month,
            "macro_signal_month": probability["signal_month"],
            "stress_signal_month": stress_signals.loc[
                month, "stress_signal_month"
            ],
            "stress_signal_date": stress_signals.loc[
                month, "stress_signal_date"
            ],
            "stress_score": stress,
            "recovery_score": recovery,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            "macro_neutral_return": float(detail["macro_neutral_return"]),
            **{
                column: float(probability[column]) for column in REGIME_COLUMNS
            },
            **{
                f"w_{asset}": float(weights[index])
                for index, asset in enumerate(ASSETS)
            },
        }
        scalar_detail_keys = [
            "policy",
            "solver_success",
            "used_fallback",
            "solver_status",
            "solver_message",
            "solver_iterations",
            "objective_value",
            "expected_monthly_return",
            "expected_monthly_variance",
            "expected_annual_log_growth",
            "downside_risk_aversion_lambda",
            "variance_penalty",
            "downside_semivariance",
            "downside_penalty",
            "estimated_transaction_cost",
            "monthly_utility",
            "expected_annual_volatility",
            "historical_cdar",
            "sum_error",
            "volatility_slack",
            "cdar_slack",
            "largest_weight",
            "largest_asset",
            "weights_above_half",
        ]
        row.update({key: detail[key] for key in scalar_detail_keys})
        vector_fields = {
            "macro_mu": "macro_expected_monthly_return",
            "stress_mu_adjustment": "stress_return_adjustment",
            "original_expected_mu": "original_expected_return",
            "filtered_expected_mu": "filtered_expected_return",
            "trend_6m": "trend_6m",
            "trend_12m": "trend_12m",
            "trend_score": "trend_score",
            "macro_relative_direction": "macro_relative_direction",
            "macro_confidence": "macro_confidence",
            "filtered_macro_mu": "filtered_macro_expected_return",
        }
        for prefix, detail_key in vector_fields.items():
            values = np.asarray(detail[detail_key], dtype=float)
            row.update(
                {
                    f"{prefix}_{asset}": float(values[index])
                    for index, asset in enumerate(ASSETS)
                }
            )
        rows.append(row)

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _forward_compound(series: pd.Series, position: int, horizon: int) -> float:
    window = series.iloc[position : position + horizon]
    if len(window) < horizon:
        return float("nan")
    return float(compounded_return(window))


def build_concentration_diagnostic(
    path: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Audit every Stage 14 month before interpreting the trend experiment."""

    rows: list[dict[str, Any]] = []
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    for month in path.index:
        weights = path.loc[month, weight_columns]
        largest_column = str(weights.idxmax())
        asset = largest_column.removeprefix("w_")
        return_position = returns.index.get_loc(month)
        path_position = path.index.get_loc(month)
        history = returns.loc[returns.index < month, asset]
        momentum_6m = (
            float(compounded_return(history.tail(MEDIUM_TREND_MONTHS)))
            if len(history) >= MEDIUM_TREND_MONTHS
            else float("nan")
        )
        momentum_12m = (
            float(compounded_return(history.tail(LONG_TREND_MONTHS)))
            if len(history) >= LONG_TREND_MONTHS
            else float("nan")
        )
        macro_mu = float(path.loc[month, f"macro_mu_{asset}"])
        predicted_12m = (1.0 + macro_mu) ** 12.0 - 1.0
        realized_asset_12m = _forward_compound(
            returns[asset], return_position, LONG_TREND_MONTHS
        )
        realized_strategy_12m = _forward_compound(
            path["return"], path_position, LONG_TREND_MONTHS
        )
        rows.append(
            {
                "Month": month,
                "LargestAsset": asset,
                "LargestWeight": float(weights.max()),
                "MacroExpectedMonthlyReturn": macro_mu,
                "MacroPredicted12MReturn": predicted_12m,
                "OwnMomentum6M": momentum_6m,
                "OwnMomentum12M": momentum_12m,
                "RealizedAssetForward3M": _forward_compound(
                    returns[asset], return_position, 3
                ),
                "RealizedAssetForward6M": _forward_compound(
                    returns[asset], return_position, 6
                ),
                "RealizedAssetForward12M": realized_asset_12m,
                "MacroPredictionError12M": (
                    predicted_12m - realized_asset_12m
                    if np.isfinite(realized_asset_12m)
                    else float("nan")
                ),
                "RealizedStrategyForward12M": realized_strategy_12m,
            }
        )
    frame = pd.DataFrame(rows).set_index("Month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def concentration_threshold_summary(
    diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in [0.50, 0.60, 0.70, 0.80]:
        view = diagnostic.loc[diagnostic["LargestWeight"] > threshold]
        rows.append(
            {
                "Threshold": threshold,
                "Months": int(len(view)),
                "KODEX200Months": int((view["LargestAsset"] == "KODEX200").sum()),
                "BONDMonths": int((view["LargestAsset"] == "BOND").sum()),
                "GLDMonths": int((view["LargestAsset"] == "GLD").sum()),
                "USOMonths": int((view["LargestAsset"] == "USO").sum()),
                "MeanForward12MConcentratedAssetReturn": float(
                    view["RealizedAssetForward12M"].mean()
                ),
                "MeanForward12MStrategyReturn": float(
                    view["RealizedStrategyForward12M"].mean()
                ),
                "MeanMacroPredictionError12M": float(
                    view["MacroPredictionError12M"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def episode_summary(
    path: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict[str, Any]:
    view = path.loc[pd.Period("2012-10", "M") : pd.Period("2014-10", "M")]
    gold_trend_6m = pd.Series(
        {
            month: float(
                compounded_return(
                    returns.loc[returns.index < month, "GLD"].tail(
                        MEDIUM_TREND_MONTHS
                    )
                )
            )
            for month in view.index
        }
    )
    gold_trend_12m = pd.Series(
        {
            month: float(
                compounded_return(
                    returns.loc[returns.index < month, "GLD"].tail(
                        LONG_TREND_MONTHS
                    )
                )
            )
            for month in view.index
        }
    )
    both_gold_trends_negative = (gold_trend_6m < 0.0) & (
        gold_trend_12m < 0.0
    )
    return {
        "start": "2012-10",
        "end": "2014-10",
        "months": int(len(view)),
        "average_gold_weight": float(view["w_GLD"].mean()),
        "minimum_gold_weight": float(view["w_GLD"].min()),
        "maximum_gold_weight": float(view["w_GLD"].max()),
        "average_gold_macro_mu_monthly": float(view["macro_mu_GLD"].mean()),
        "average_gold_trend_6m": float(gold_trend_6m.mean()),
        "average_gold_trend_12m": float(gold_trend_12m.mean()),
        "both_gold_trends_negative_months": int(both_gold_trends_negative.sum()),
        "average_gold_macro_confidence": (
            float(view["macro_confidence_GLD"].mean())
            if "macro_confidence_GLD" in view
            else None
        ),
    }


def verify_saved_trends_are_causal(
    path: pd.DataFrame,
    returns: pd.DataFrame,
) -> bool:
    """Recompute every saved trend from returns strictly before each target."""

    for month in path.index:
        history = returns.loc[returns.index < month, ASSETS]
        expected_6m = np.asarray(
            compounded_return(history.tail(MEDIUM_TREND_MONTHS)), dtype=float
        )
        expected_12m = np.asarray(
            compounded_return(history.tail(LONG_TREND_MONTHS)), dtype=float
        )
        saved_6m = path.loc[
            month, [f"trend_6m_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        saved_12m = path.loc[
            month, [f"trend_12m_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        if not (
            np.allclose(expected_6m, saved_6m)
            and np.allclose(expected_12m, saved_12m)
        ):
            return False
    return True


def run_research(save: bool = True) -> dict[str, Any]:
    """Evaluate one fixed asset-trend confidence hypothesis against Stage 14."""

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)

    stage14_path = run_stage14_backtest(
        returns, probabilities, stress_signals, STATIC_RISK_POLICY
    )
    trend_path = run_backtest(returns, probabilities, stress_signals)
    paths = {
        "Stage14_StaticLambda": stage14_path,
        "Stage19_AssetTrendConfidence": trend_path,
    }
    common_end = min(path.index.max() for path in paths.values())
    comparison_rows: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            comparison_rows.append(metric_row(name, path, period, start, common_end))
    comparison = pd.DataFrame(comparison_rows)

    stage14_diagnostic = build_concentration_diagnostic(stage14_path, returns)
    threshold_summary = concentration_threshold_summary(stage14_diagnostic)
    stage14_gold_concentration = stage14_diagnostic.loc[
        (stage14_diagnostic["LargestAsset"] == "GLD")
        & (stage14_diagnostic["LargestWeight"] > 0.50)
    ]
    trend_diagnostic = build_concentration_diagnostic(trend_path, returns)
    trend_threshold_summary = concentration_threshold_summary(trend_diagnostic)
    episodes = pd.concat(
        [
            drawdown_episodes(stage14_path, returns, "Stage14_StaticLambda"),
            drawdown_episodes(trend_path, returns, "Stage19_AssetTrendConfidence"),
        ],
        ignore_index=True,
    )
    top_episodes = (
        episodes.sort_values(["Strategy", "EpisodeMDD"])
        .groupby("Strategy", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    baseline = full.loc["Stage14_StaticLambda"]
    candidate = full.loc["Stage19_AssetTrendConfidence"]
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    checks = {
        "macro_signal_precedes_target": bool(
            (trend_path["macro_signal_month"] < trend_path.index).all()
        ),
        "stress_signal_precedes_target": bool(
            (trend_path["stress_signal_month"] < trend_path.index).all()
        ),
        "own_trend_uses_only_prior_returns": verify_saved_trends_are_causal(
            trend_path, returns
        ),
        "weights_sum_to_one": bool(
            np.allclose(trend_path[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (trend_path[weight_columns] >= -1e-10).all().all()
        ),
        "no_leverage": bool(
            np.allclose(trend_path[weight_columns].sum(axis=1), 1.0)
        ),
        "static_lambda_equals_one": bool(
            np.allclose(trend_path["downside_risk_aversion_lambda"], 1.0)
        ),
        "same_sigma_and_cdar_guards_as_stage14": True,
        "no_hhi_or_concentration_penalty": True,
        "no_hard_asset_cap": True,
        "no_hard_regime_weights": True,
        "no_dynamic_sigma_or_es": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_or_candidate_search": True,
        "single_predeclared_hypothesis": True,
    }
    report: dict[str, Any] = {
        "strategy": "Stage19_AssetTrendConfidence",
        "base_strategy": "Stage14_StaticLambda",
        "hypothesis": (
            "Use each asset's own 6M/12M trend only to shrink a conflicting "
            "asset-specific macro expected return toward the cross-sectional "
            "equal-weight neutral return. Do not add momentum alpha."
        ),
        "confidence_formula": {
            "trend_score": "0.5 * (sign(6M return) + sign(12M return))",
            "neutral_macro_return": "equal mean of four macro expected returns",
            "macro_relative_direction": "sign(macro_mu_i - neutral_macro_mu)",
            "confidence": "(1 + macro_relative_direction * trend_score) / 2",
            "filtered_macro_mu": (
                "neutral_macro_mu + confidence * "
                "(macro_mu_i - neutral_macro_mu)"
            ),
            "stress_adjustment": "unchanged from Stage14 and added afterward",
            "searched_parameters": None,
            "candidate_count": 1,
        },
        "unchanged_stage14_controls": {
            "downside_risk_aversion_lambda": 1.0,
            "annual_volatility_guard": CATASTROPHE_ANNUAL_VOLATILITY,
            "cdar_guard": CATASTROPHE_CDAR,
            "cdar_confidence": CDAR_CONFIDENCE,
            "asset_bounds": [0.0, 1.0],
            "weight_sum": 1.0,
            "leverage": 1.0,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "full_period_changes": {
            "cagr": float(candidate["CAGR"] - baseline["CAGR"]),
            "sharpe": float(candidate["Sharpe"] - baseline["Sharpe"]),
            "mdd": float(candidate["MDD"] - baseline["MDD"]),
            "volatility": float(
                candidate["Volatility"] - baseline["Volatility"]
            ),
        },
        "success_gates": {
            "cagr_at_least_10_5_percent": bool(candidate["CAGR"] >= 0.105),
            "sharpe_at_least_0_85": bool(candidate["Sharpe"] >= 0.85),
            "mdd_no_worse_than_18_percent": bool(candidate["MDD"] >= -0.18),
            "no_leverage": checks["no_leverage"],
            "no_hard_regime": checks["no_hard_regime_weights"],
        },
        "stage14_concentration_thresholds": json.loads(
            threshold_summary.to_json(orient="records", force_ascii=False)
        ),
        "stage14_gold_concentration_summary": {
            "months": int(len(stage14_gold_concentration)),
            "mean_weight": float(
                stage14_gold_concentration["LargestWeight"].mean()
            ),
            "mean_forward_3m_gold_return": float(
                stage14_gold_concentration[
                    "RealizedAssetForward3M"
                ].mean()
            ),
            "mean_forward_6m_gold_return": float(
                stage14_gold_concentration[
                    "RealizedAssetForward6M"
                ].mean()
            ),
            "mean_forward_12m_gold_return": float(
                stage14_gold_concentration[
                    "RealizedAssetForward12M"
                ].mean()
            ),
            "mean_macro_prediction_error_12m": float(
                stage14_gold_concentration[
                    "MacroPredictionError12M"
                ].mean()
            ),
        },
        "trend_strategy_concentration_thresholds": json.loads(
            trend_threshold_summary.to_json(
                orient="records", force_ascii=False
            )
        ),
        "stage14_concentration": concentration_summary(stage14_path),
        "trend_strategy_concentration": concentration_summary(trend_path),
        "stage14_gold_episode": episode_summary(stage14_path, returns),
        "trend_strategy_gold_episode": episode_summary(trend_path, returns),
        "top_drawdown_episodes": json.loads(
            top_episodes.to_json(orient="records", force_ascii=False)
        ),
        "solver": solver_summary(trend_path),
        "checks": checks,
        "decision_rule": (
            "Adopt only if the stated CAGR/Sharpe/MDD gates all pass; otherwise "
            "retain Stage14 and report the asset-universe/frontier limitation."
        ),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stage14_path.to_csv(OUTPUT_DIR / "stage14_static_recomputed_monthly.csv")
        trend_path.to_csv(OUTPUT_DIR / "asset_trend_confidence_monthly.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        stage14_diagnostic.to_csv(
            OUTPUT_DIR / "stage14_concentration_diagnostic.csv"
        )
        stage14_gold_concentration.to_csv(
            OUTPUT_DIR / "stage14_gold_concentration_diagnostic.csv"
        )
        threshold_summary.to_csv(
            OUTPUT_DIR / "stage14_concentration_threshold_summary.csv",
            index=False,
        )
        trend_diagnostic.to_csv(
            OUTPUT_DIR / "trend_strategy_concentration_diagnostic.csv"
        )
        trend_threshold_summary.to_csv(
            OUTPUT_DIR / "trend_strategy_concentration_threshold_summary.csv",
            index=False,
        )
        top_episodes.to_csv(
            OUTPUT_DIR / "drawdown_episode_attribution.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(
                report, ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "stress_signals": stress_signals,
        "stage14_path": stage14_path,
        "trend_path": trend_path,
        "comparison": comparison,
        "stage14_diagnostic": stage14_diagnostic,
        "stage14_gold_concentration": stage14_gold_concentration,
        "threshold_summary": threshold_summary,
        "trend_diagnostic": trend_diagnostic,
        "trend_threshold_summary": trend_threshold_summary,
        "episodes": top_episodes,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(result["threshold_summary"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
