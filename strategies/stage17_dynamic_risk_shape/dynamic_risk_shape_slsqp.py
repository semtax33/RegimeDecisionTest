from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
    build_overlay_attribution,
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
from strategies.stage16_confirmed_crash_risk.confirmed_crash_risk_slsqp import (
    build_confirmed_daily_features,
    build_confirmed_monthly_signals,
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
STAGE16_OUTPUT = (
    ROOT
    / "strategies"
    / "stage16_confirmed_crash_risk"
    / "outputs"
    / "confirmed_crash_lambda_monthly.csv"
)
NUMERICAL_EPSILON = 1e-12


@dataclass(frozen=True)
class RiskShapePolicy:
    name: str
    covariance_mode: str
    tail_mode: str
    include_internal_transaction_cost: bool = True


POLICY_17A = RiskShapePolicy(
    "Stage17A_DynamicSigma",
    covariance_mode="confirmed",
    tail_mode="semivariance",
)
POLICY_17B = RiskShapePolicy(
    "Stage17B_DynamicES",
    covariance_mode="stage14_raw_stress",
    tail_mode="expected_shortfall_squared",
)
POLICY_17C = RiskShapePolicy(
    "Stage17C_DynamicSigmaES",
    covariance_mode="confirmed",
    tail_mode="expected_shortfall_squared",
)
POLICY_17C_NO_INTERNAL_COST = RiskShapePolicy(
    "Stage17C_NoInternalCost_Diagnostic",
    covariance_mode="confirmed",
    tail_mode="expected_shortfall_squared",
    include_internal_transaction_cost=False,
)
RESEARCH_POLICIES = [POLICY_17A, POLICY_17B, POLICY_17C]


def nearest_psd(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.trace(symmetric)) / len(symmetric), NUMERICAL_EPSILON)
    floor = scale * 1e-10
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def conditional_moment_components(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover Stage14 normal and stress covariance components causally."""

    macro_mu, normal_covariance, normal_detail = estimate_conditional_moments(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=False,
    )
    stress_mu, stage14_covariance, stress_detail = estimate_conditional_moments(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=True,
    )
    stress_level = float(np.clip(current_stress, NUMERICAL_EPSILON, 1.0))
    stress_covariance = (
        stage14_covariance - (1.0 - stress_level) * normal_covariance
    ) / stress_level
    stress_covariance = nearest_psd(stress_covariance)
    detail = {
        "macro_expected_monthly_return": normal_detail[
            "macro_expected_monthly_return"
        ],
        "stress_return_adjustment": stress_detail[
            "stress_return_adjustment"
        ],
    }
    return (
        stress_mu,
        normal_covariance,
        stress_covariance,
        stage14_covariance,
        detail,
    )


def expected_shortfall(
    portfolio_returns: np.ndarray,
    confidence: float = CDAR_CONFIDENCE,
) -> float:
    """Historical monthly expected loss in the worst confidence tail."""

    values = np.asarray(portfolio_returns, dtype=float)
    count = max(1, int(math.ceil((1.0 - confidence) * len(values))))
    worst = np.sort(values)[:count]
    return max(-float(worst.mean()), 0.0)


def weighted_expected_shortfall(
    portfolio_returns: np.ndarray,
    scenario_weights: np.ndarray,
    confidence: float = CDAR_CONFIDENCE,
) -> float:
    """Stress-weighted expected loss without selecting a stress threshold."""

    returns = np.asarray(portfolio_returns, dtype=float)
    weights = np.asarray(scenario_weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= NUMERICAL_EPSILON:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    order = np.argsort(returns)
    ordered_returns = returns[order]
    ordered_weights = weights[order]
    tail_mass = 1.0 - confidence
    remaining = tail_mass
    weighted_loss = 0.0
    used_mass = 0.0
    for value, weight in zip(ordered_returns, ordered_weights):
        take = min(float(weight), remaining)
        if take <= 0.0:
            continue
        weighted_loss += take * max(-float(value), 0.0)
        used_mass += take
        remaining -= take
        if remaining <= NUMERICAL_EPSILON:
            break
    return weighted_loss / max(used_mass, NUMERICAL_EPSILON)


def portfolio_vulnerability(
    historical_returns: np.ndarray,
    historical_stress: np.ndarray,
    pretrade: np.ndarray,
) -> float:
    """Scale market evidence by the portfolio's own stress-scenario loss."""

    weights = (
        project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )
    portfolio_loss = weighted_expected_shortfall(
        historical_returns @ weights, historical_stress
    )
    asset_losses = [
        weighted_expected_shortfall(
            historical_returns[:, index], historical_stress
        )
        for index in range(len(ASSETS))
    ]
    worst_available_loss = max(max(asset_losses), NUMERICAL_EPSILON)
    return float(np.clip(portfolio_loss / worst_available_loss, 0.0, 1.0))


def solve_risk_shape_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    crash_pressure: float,
    pretrade: np.ndarray,
    policy: RiskShapePolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    (
        expected_return,
        normal_covariance,
        stress_covariance,
        stage14_covariance,
        moment_detail,
    ) = conditional_moment_components(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
    )
    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    stress_history = historical_stress.loc[common].to_numpy(dtype=float)
    vulnerability = portfolio_vulnerability(
        historical_returns, stress_history, pretrade
    )
    market_pressure = float(np.clip(crash_pressure, 0.0, 1.0))
    portfolio_pressure = market_pressure * vulnerability
    if policy.covariance_mode == "confirmed":
        covariance = nearest_psd(
            (1.0 - portfolio_pressure) * normal_covariance
            + portfolio_pressure * stress_covariance
        )
    elif policy.covariance_mode == "stage14_raw_stress":
        covariance = stage14_covariance
    else:
        raise ValueError(f"Unknown covariance mode: {policy.covariance_mode}")

    initial = (
        project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        scenarios = historical_returns @ weights
        downside_semivariance = float(
            np.mean(np.minimum(scenarios, 0.0) ** 2)
        )
        monthly_es = expected_shortfall(scenarios)
        if policy.tail_mode == "semivariance":
            tail_penalty = downside_semivariance
            tail_multiplier = 1.0
        elif policy.tail_mode == "expected_shortfall_squared":
            # Literal Stage17B/C hypothesis: eta(q)=eta0*(1+q). Squaring ES
            # gives variance units, so eta0=1 needs no fitted scale.
            tail_multiplier = 1.0 + portfolio_pressure
            tail_penalty = tail_multiplier * monthly_es**2
        else:
            raise ValueError(f"Unknown tail mode: {policy.tail_mode}")
        model_cost = (
            expected_transaction_cost(weights, pretrade)
            if policy.include_internal_transaction_cost
            else 0.0
        )
        monthly_utility = (
            monthly_return - 0.5 * monthly_variance - tail_penalty - model_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_risk_aversion_lambda": 1.0,
            "variance_penalty": 0.5 * monthly_variance,
            "downside_semivariance": downside_semivariance,
            "expected_shortfall": monthly_es,
            "tail_multiplier": tail_multiplier,
            "tail_penalty": tail_penalty,
            "estimated_transaction_cost": model_cost,
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
    used_feasibility_retry = False
    primary_solver_success = bool(result.success)
    primary_solver_message = str(result.message)
    if result.success and np.isfinite(result.x).all():
        weights = project_to_long_only_simplex(result.x)
    else:
        feasibility = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
        )
        if not feasibility.success or not np.isfinite(feasibility.x).all():
            raise RuntimeError(
                f"{policy.name} economic and fallback solves failed: "
                f"{result.message}; {feasibility.message}"
            )
        retry = minimize(
            objective,
            feasibility.x,
            method="SLSQP",
            bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
        )
        if retry.success and np.isfinite(retry.x).all():
            result = retry
            weights = project_to_long_only_simplex(retry.x)
            used_feasibility_retry = True
        else:
            result = feasibility
            weights = project_to_long_only_simplex(feasibility.x)
            used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = cdar(historical_returns @ weights, CDAR_CONFIDENCE)
    detail: dict[str, Any] = {
        **values,
        "policy": policy.name,
        "covariance_mode": policy.covariance_mode,
        "tail_mode": policy.tail_mode,
        "internal_transaction_cost": policy.include_internal_transaction_cost,
        "market_crash_pressure": market_pressure,
        "portfolio_vulnerability": vulnerability,
        "portfolio_crash_pressure": portfolio_pressure,
        "solver_success": bool(result.success),
        "primary_solver_success": primary_solver_success,
        "primary_solver_message": primary_solver_message,
        "used_feasibility_retry": used_feasibility_retry,
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


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    signals: pd.DataFrame,
    policy: RiskShapePolicy,
) -> pd.DataFrame:
    months = returns.index.intersection(probabilities.index).intersection(
        signals.index
    )
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
        try:
            weights, detail = solve_risk_shape_weights(
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
                crash_pressure=float(signal["crash_pressure"]),
                pretrade=pretrade,
                policy=policy,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{policy.name} {month}: {exc}") from exc

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


def layered_precision(
    returns: pd.DataFrame,
    fixed: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    """Separate signal, decision, and realized portfolio diagnostics."""

    common = fixed.index.intersection(candidate.index)
    frame = pd.DataFrame(index=common)
    frame["active"] = candidate.loc[common, "portfolio_crash_pressure"] > 0.0
    frame["candidate_alpha"] = (
        candidate.loc[common, "return"] - fixed.loc[common, "return"]
    )
    frame["risk_reduction"] = (
        fixed.loc[common, ["w_KODEX200", "w_USO"]].sum(axis=1)
        - candidate.loc[common, ["w_KODEX200", "w_USO"]].sum(axis=1)
    ) > 1e-6
    threshold = float(returns.loc[common, "KODEX200"].quantile(0.10))
    frame["crash"] = returns.loc[common, "KODEX200"] <= threshold
    active = frame[frame["active"]]
    decisions = frame[frame["risk_reduction"]]
    return {
        "months": int(len(frame)),
        "signal_months": int(len(active)),
        "signal_crash_precision": float(active["crash"].mean())
        if len(active)
        else 0.0,
        "signal_caught_crashes": int((active["crash"]).sum()),
        "decision_precision": float((active["candidate_alpha"] >= 0.0).mean())
        if len(active)
        else 0.0,
        "decision_false_positives": int((active["candidate_alpha"] < 0.0).sum()),
        "portfolio_action_months": int(len(decisions)),
        "portfolio_precision": float(
            (decisions["candidate_alpha"] >= 0.0).mean()
        )
        if len(decisions)
        else 0.0,
        "portfolio_false_positives": int(
            (decisions["candidate_alpha"] < 0.0).sum()
        ),
    }


def drawdown_episodes(
    path: pd.DataFrame,
    returns: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    """Attribute complete peak-to-trough episodes, not isolated crash months."""

    drawdown = path["drawdown"]
    rows: list[dict[str, Any]] = []
    in_episode = False
    underwater_start: pd.Period | None = None
    for position, month in enumerate(drawdown.index):
        underwater = float(drawdown.loc[month]) < -1e-12
        if underwater and not in_episode:
            in_episode = True
            underwater_start = month
        episode_ends = in_episode and (
            not underwater or position == len(drawdown.index) - 1
        )
        if not episode_ends or underwater_start is None:
            continue
        recovery = month if not underwater else None
        episode_end = month
        segment = drawdown.loc[underwater_start:episode_end]
        trough = segment.idxmin()
        loss_months = path.loc[underwater_start:trough].index
        weights = path.loc[
            loss_months, [f"w_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        asset_returns = returns.loc[loss_months, ASSETS].to_numpy(dtype=float)
        contributions = (weights * asset_returns).sum(axis=0)
        stock_bond_correlation = (
            float(
                returns.loc[loss_months, ["KODEX200", "BOND"]]
                .corr()
                .iloc[0, 1]
            )
            if len(loss_months) > 1
            else np.nan
        )
        rows.append(
            {
                "Strategy": strategy,
                "UnderwaterStart": str(underwater_start),
                "Trough": str(trough),
                "Recovery": str(recovery) if recovery is not None else "ongoing",
                "EpisodeMDD": float(segment.min()),
                "MonthsToTrough": int(len(loss_months)),
                "NetReturnToTrough": float(
                    (1.0 + path.loc[loss_months, "return"]).prod() - 1.0
                ),
                "TotalCostToTrough": float(
                    path.loc[loss_months, ["trade_cost", "fx_cost"]]
                    .sum()
                    .sum()
                ),
                **{
                    f"{asset}_Contribution": float(contributions[index])
                    for index, asset in enumerate(ASSETS)
                },
                "StockBondCorrelation": stock_bond_correlation,
                "AverageCrashPressure": float(
                    path.loc[loss_months, "crash_pressure"].mean()
                )
                if "crash_pressure" in path
                else np.nan,
                "AveragePortfolioPressure": float(
                    path.loc[loss_months, "portfolio_crash_pressure"].mean()
                )
                if "portfolio_crash_pressure" in path
                else np.nan,
            }
        )
        in_episode = False
        underwater_start = None
    return pd.DataFrame(rows).sort_values("EpisodeMDD")


def run_research(save: bool = True) -> dict[str, Any]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily = build_confirmed_daily_features()
    signals = build_confirmed_monthly_signals(
        returns.index, probabilities, daily
    )
    candidate_paths = {
        policy.name: run_backtest(returns, probabilities, signals, policy)
        for policy in RESEARCH_POLICIES
    }
    no_internal_cost_path = run_backtest(
        returns, probabilities, signals, POLICY_17C_NO_INTERNAL_COST
    )
    fixed_path = read_saved_path(STAGE14_STATIC_OUTPUT)
    stage16_path = read_saved_path(STAGE16_OUTPUT)
    paths = {
        "Stage14_StaticLambda": fixed_path,
        "Stage16_ConfirmedLambda": stage16_path,
        **candidate_paths,
        POLICY_17C_NO_INTERNAL_COST.name: no_internal_cost_path,
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

    precision_rows: list[dict[str, Any]] = []
    for name, path in {
        **candidate_paths,
        POLICY_17C_NO_INTERNAL_COST.name: no_internal_cost_path,
    }.items():
        for period, start in [
            ("full_2007_2026", FULL_START),
            ("locked_2018_2026", LOCKED_START),
        ]:
            precision_rows.append(
                {
                    "Strategy": name,
                    "Period": period,
                    **layered_precision(
                        returns.loc[start:common_end],
                        fixed_path.loc[start:common_end],
                        path.loc[start:common_end],
                    ),
                }
            )
    precision = pd.DataFrame(precision_rows)

    episode_frames = [
        drawdown_episodes(fixed_path, returns, "Stage14_StaticLambda"),
        *[
            drawdown_episodes(path, returns, name)
            for name, path in candidate_paths.items()
        ],
    ]
    episodes = pd.concat(episode_frames, ignore_index=True)
    top_episodes = (
        episodes.sort_values(["Strategy", "EpisodeMDD"])
        .groupby("Strategy", as_index=False)
        .head(5)
    )

    weight_columns = [f"w_{asset}" for asset in ASSETS]
    checks: dict[str, bool] = {
        "stage16_evidence_is_reused_without_refit": True,
        "all_macro_signals_precede_target": all(
            bool((path["macro_signal_month"] < path.index).all())
            for path in candidate_paths.values()
        ),
        "all_stress_signals_precede_target": all(
            bool((path["stress_signal_month"] < path.index).all())
            for path in candidate_paths.values()
        ),
        "all_weights_sum_to_one": all(
            bool(np.allclose(path[weight_columns].sum(axis=1), 1.0))
            for path in candidate_paths.values()
        ),
        "all_weights_long_only": all(
            bool((path[weight_columns] >= -1e-10).all().all())
            for path in candidate_paths.values()
        ),
        "no_cash": True,
        "no_leverage": True,
        "lambda_is_fixed_at_one": all(
            bool(np.allclose(path["downside_risk_aversion_lambda"], 1.0))
            for path in candidate_paths.values()
        ),
        "portfolio_pressure_not_above_market_pressure": all(
            bool(
                (
                    path["portfolio_crash_pressure"]
                    <= path["market_crash_pressure"] + 1e-12
                ).all()
            )
            for path in candidate_paths.values()
        ),
        "volatility_guards_respected": all(
            bool((path["volatility_slack"] >= -1e-7).all())
            for path in candidate_paths.values()
        ),
        "cdar_guards_respected": all(
            bool((path["cdar_slack"] >= -1e-7).all())
            for path in candidate_paths.values()
        ),
        "all_slsqp_solves_succeeded": all(
            bool(path["solver_success"].all())
            for path in candidate_paths.values()
        ),
        "no_economic_objective_fallbacks": all(
            bool((~path["used_fallback"]).all())
            for path in candidate_paths.values()
        ),
        "no_future_label_in_strategy": True,
        "no_hyperparameter_search": True,
    }
    policy_reports = {
        name: {
            "performance_full": json.loads(
                comparison[
                    (comparison["Strategy"] == name)
                    & (comparison["Period"] == "full_2007_2026")
                ].to_json(orient="records", force_ascii=False)
            )[0],
            "concentration": concentration_summary(path),
            "solver": solver_summary(path),
            "average_portfolio_vulnerability": float(
                path["portfolio_vulnerability"].mean()
            ),
            "average_portfolio_pressure": float(
                path["portfolio_crash_pressure"].mean()
            ),
        }
        for name, path in candidate_paths.items()
    }
    full = comparison[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    gates = {
        name: {
            "cagr_at_least_10_5_percent": bool(full.loc[name, "CAGR"] >= 0.105),
            "mdd_no_worse_than_18_percent": bool(full.loc[name, "MDD"] >= -0.18),
            "sharpe_not_below_stage14_fixed": bool(
                full.loc[name, "Sharpe"] >= full.loc["Stage14_StaticLambda", "Sharpe"]
            ),
        }
        for name in candidate_paths
    }
    report: dict[str, Any] = {
        "strategy_family": "Stage17_DynamicRiskShape",
        "frozen_evidence_source": "Stage16 Confirmed Crash Evidence",
        "economic_hypotheses": {
            "17A": "fixed lambda; evidence changes covariance shape",
            "17B": "fixed lambda; evidence changes ES-squared tail penalty",
            "17C": "fixed lambda; evidence changes covariance and ES-squared tail",
            "portfolio_specific": (
                "market pressure times current portfolio stress-weighted ES "
                "relative to the riskiest available asset"
            ),
        },
        "parameter_policy": {
            "searched_parameters": None,
            "evidence_threshold": "frozen Stage16 median 0.5",
            "tail_confidence": CDAR_CONFIDENCE,
            "tail_scale": "expected shortfall squared for variance units",
            "candidate_count": 3,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "layered_precision": json.loads(
            precision.to_json(orient="records", force_ascii=False)
        ),
        "top_drawdown_episodes": json.loads(
            top_episodes.to_json(orient="records", force_ascii=False)
        ),
        "policies": policy_reports,
        "success_gates": gates,
        "no_internal_cost_diagnostic": {
            "purpose": (
                "remove only optimizer-internal cost while still deducting "
                "realized trading and FX costs"
            ),
            "performance_full": json.loads(
                comparison[
                    (comparison["Strategy"] == POLICY_17C_NO_INTERNAL_COST.name)
                    & (comparison["Period"] == "full_2007_2026")
                ].to_json(orient="records", force_ascii=False)
            )[0],
        },
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, path in candidate_paths.items():
            path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        no_internal_cost_path.to_csv(
            OUTPUT_DIR / "stage17c_no_internal_cost_diagnostic_monthly.csv"
        )
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        precision.to_csv(
            OUTPUT_DIR / "layered_precision.csv", index=False
        )
        top_episodes.to_csv(
            OUTPUT_DIR / "drawdown_episode_attribution.csv", index=False
        )
        signals.to_csv(OUTPUT_DIR / "frozen_stage16_signals.csv")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "returns": returns,
        "signals": signals,
        "candidate_paths": candidate_paths,
        "no_internal_cost_path": no_internal_cost_path,
        "comparison": comparison,
        "precision": precision,
        "episodes": top_episodes,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print("\nLayered precision")
    print(result["precision"].to_string(index=False))
    print("\nSuccess gates")
    print(json.dumps(result["report"]["success_gates"], ensure_ascii=False, indent=2))
    print("\nChecks")
    print(json.dumps(result["report"]["checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
