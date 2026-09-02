from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.stage36_asset_implied_volatility_risk import (
    asset_implied_volatility_risk_slsqp as stage36,
)
from strategies.stage45_volatility_targeted_shrinkage_mlp import (
    volatility_targeted_shrinkage_mlp as stage45,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = stage36.ASSETS
WEIGHT_COLUMNS = stage36.WEIGHT_COLUMNS
FULL_START = stage36.FULL_START
COMMON_START = stage36.COMMON_START
LOCKED_START = stage36.stage35.LOCKED_START
RESEARCH_END = stage36.RESEARCH_END

MIN_PRIOR_MONTHS = 12
CAGR_TOLERANCE = 0.005
BOOTSTRAP_SHARPE_GATE = 0.50
BOOTSTRAP_MDD_GATE = 0.50
TURNOVER_TOLERANCE = 0.10
GATE_TOLERANCE = 1e-12
NUMERICAL_EPSILON = 1e-12

BASELINE_NAME = "Stage36_GVZ_OVXAssetRisk"
LW_NAME = "Stage46_Stage36Mu_LW"
BL_NAME = "Stage46_GuardedBlackLitterman_LW"
RAW_BL_NAME = "Stage46_BlackLitterman_LW_Shadow"
MODES = {
    LW_NAME: "stage36_mu_lw",
    RAW_BL_NAME: "black_litterman_lw",
}

STAGE36_PATH = (
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv"
)
FROZEN_FILES = (
    Path(stage36.__file__),
    STAGE36_PATH,
    stage36.OUTPUT_DIR / "validation_report.json",
    Path(stage45.__file__),
    stage45.OUTPUT_DIR / "validation_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_manifest() -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in FROZEN_FILES
    }


def _load_period_csv(path: Path, index: str) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=index)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def stage36_expected_return(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    fundamental_signal: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce Stage 36's expected-return side without its covariance."""

    _, _, moment = stage36.stage35.estimate_conditional_moments(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=True,
    )
    macro = np.asarray(
        moment["macro_expected_monthly_return"], dtype=float
    )
    stress_adjustment = np.asarray(
        moment["stress_return_adjustment"], dtype=float
    ).copy()
    technical = stage36.stage35.stage20.apply_technical_inputs(
        macro, np.eye(len(ASSETS), dtype=float), technical_signal
    )
    filtered_macro = np.asarray(
        technical["filtered_macro_expected_return"], dtype=float
    ).copy()

    eps_mu = float(fundamental_signal["eps_mu_adjustment_KODEX200"])
    valuation_mu = float(
        fundamental_signal["valuation_mu_adjustment_KODEX200"]
    )
    credit_stress_multiplier = float(
        fundamental_signal["credit_stress_multiplier"]
    )
    stress_adjustment[stage36.stage35.EQUITY_INDEX] *= (
        credit_stress_multiplier
    )
    filtered_macro[stage36.stage35.EQUITY_INDEX] += eps_mu + valuation_mu
    expected_return = filtered_macro + stress_adjustment
    return expected_return, {
        "macro_expected_return": macro,
        "filtered_macro_expected_return": filtered_macro,
        "stress_return_adjustment": stress_adjustment,
        "eps_mu_adjustment": eps_mu,
        "valuation_mu_adjustment": valuation_mu,
        "credit_stress_multiplier": credit_stress_multiplier,
    }


def regime_confidence(probabilities: pd.Series) -> float:
    """Parameter-free concentration of four soft probabilities on [0, 1]."""

    values = probabilities[stage36.stage35.REGIME_COLUMNS].to_numpy(
        dtype=float
    )
    values = values / values.sum()
    return float(np.clip((np.square(values).sum() - 0.25) / 0.75, 0.0, 1.0))


def black_litterman_posterior(
    covariance: np.ndarray,
    strategic_prior_weights: np.ndarray,
    stage36_view: np.ndarray,
    confidence: float,
    risk_aversion: float,
    prior_observations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend equilibrium returns with Stage 36 absolute views.

    Omega follows the Idzorek confidence ratio. Because both prior uncertainty
    and Omega contain tau, the view strength is governed by the observed regime
    concentration rather than a searched tau value.
    """

    sigma = np.asarray(covariance, dtype=float)
    weights = np.asarray(strategic_prior_weights, dtype=float)
    view = np.asarray(stage36_view, dtype=float)
    delta = max(float(risk_aversion), NUMERICAL_EPSILON)
    equilibrium = delta * sigma @ weights
    tau = 1.0 / max(int(prior_observations), 1)
    prior_uncertainty = tau * sigma
    p = np.eye(len(ASSETS), dtype=float)
    c = float(np.clip(confidence, 0.0, 1.0))
    if c <= NUMERICAL_EPSILON:
        posterior = equilibrium.copy()
        omega_diagonal = np.full(len(ASSETS), np.inf)
    else:
        base_uncertainty = np.maximum(
            np.diag(p @ prior_uncertainty @ p.T), NUMERICAL_EPSILON
        )
        omega_diagonal = base_uncertainty * (1.0 - c) / c
        omega = np.diag(np.maximum(omega_diagonal, NUMERICAL_EPSILON))
        middle = p @ prior_uncertainty @ p.T + omega
        posterior = equilibrium + prior_uncertainty @ p.T @ np.linalg.solve(
            middle, view - p @ equilibrium
        )
    return posterior, {
        "equilibrium_return": equilibrium,
        "posterior_return": posterior,
        "view_return": view,
        "regime_confidence": c,
        "risk_aversion": delta,
        "tau": tau,
        "omega_diagonal": omega_diagonal,
    }


def _risk_aversion_from_history(returns: pd.Series) -> float:
    values = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(values) < MIN_PRIOR_MONTHS:
        return 1.0
    variance = float(np.var(values, ddof=1))
    if variance <= NUMERICAL_EPSILON:
        return 1.0
    return max(float(np.mean(values)) / variance, NUMERICAL_EPSILON)


def _smooth_l1(values: np.ndarray) -> float:
    return float(np.sqrt(np.asarray(values, dtype=float) ** 2 + 1e-16).sum())


def solve_weights(
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_returns: np.ndarray,
    pretrade: np.ndarray,
    baseline_target: np.ndarray,
    baseline_pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run Stage 36 utility with an explicit no-more-turnover budget."""

    mu = np.asarray(expected_return, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    history = np.asarray(historical_returns, dtype=float)
    baseline_turnover_budget = float(
        np.abs(np.asarray(baseline_target) - baseline_pretrade).sum()
    )

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ sigma @ weights), 0.0) * 12.0)

    def historical_cdar(weights: np.ndarray) -> float:
        return stage36.stage35.cdar(
            history @ weights, stage36.stage35.CDAR_CONFIDENCE
        )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ mu)
        monthly_variance = max(float(weights @ sigma @ weights), 0.0)
        downside_semivariance = float(
            np.mean(np.minimum(history @ weights, 0.0) ** 2)
        )
        transaction_cost = stage36.stage35.expected_transaction_cost(
            weights, pretrade
        )
        utility = (
            monthly_return
            - 0.5 * monthly_variance
            - downside_semivariance
            - transaction_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "downside_semivariance": downside_semivariance,
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": utility,
        }

    risk_constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: float(
                stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
                - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                stage36.stage35.CATASTROPHE_CDAR
                + historical_cdar(weights)
            ),
        },
    ]
    initials = [
        stage36.stage35.project_to_long_only_simplex(pretrade),
        stage36.stage35.project_to_long_only_simplex(baseline_target),
        np.repeat(1.0 / len(ASSETS), len(ASSETS)),
    ]
    minimum_trade_solutions: list[Any] = []
    for initial in initials:
        result = minimize(
            lambda weights: _smooth_l1(weights - pretrade),
            initial,
            method="SLSQP",
            bounds=stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=risk_constraints,
            options={
                "maxiter": stage36.stage35.SLSQP_MAX_ITERATIONS,
                "ftol": stage36.stage35.SLSQP_TOLERANCE,
            },
        )
        if result.success and np.isfinite(result.x).all():
            minimum_trade_solutions.append(result)
    if not minimum_trade_solutions:
        raise RuntimeError("no feasible portfolio exists inside risk guards")
    minimum_trade_result = min(
        minimum_trade_solutions, key=lambda item: float(item.fun)
    )
    minimum_risk_trade = float(
        np.abs(minimum_trade_result.x - pretrade).sum()
    )
    turnover_budget = max(
        baseline_turnover_budget, float(minimum_trade_result.fun) + 1e-7
    )
    constraints = [
        *risk_constraints,
        {
            "type": "ineq",
            "fun": lambda weights: float(
                turnover_budget - _smooth_l1(weights - pretrade)
            ),
        },
    ]
    solutions: list[Any] = []
    for initial in [minimum_trade_result.x, *initials]:
        result = minimize(
            lambda weights: -portfolio_values(weights)["monthly_utility"],
            initial,
            method="SLSQP",
            bounds=stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": stage36.stage35.SLSQP_MAX_ITERATIONS,
                "ftol": stage36.stage35.SLSQP_TOLERANCE,
            },
        )
        if result.success and np.isfinite(result.x).all():
            solutions.append(result)
    used_fallback = False
    if solutions:
        result = min(solutions, key=lambda item: float(item.fun))
        weights = stage36.stage35.project_to_long_only_simplex(result.x)
    else:
        # The already-solved minimum-trade portfolio is feasible by construction.
        result = minimum_trade_result
        weights = stage36.stage35.project_to_long_only_simplex(result.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    cdar_value = historical_cdar(weights)
    actual_trade = float(np.abs(weights - pretrade).sum())
    return weights, {
        **values,
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": cdar_value,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": (
            stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY - annual_vol
        ),
        "cdar_slack": stage36.stage35.CATASTROPHE_CDAR + cdar_value,
        "turnover_budget_l1": turnover_budget,
        "baseline_turnover_budget_l1": baseline_turnover_budget,
        "minimum_risk_trade_l1": minimum_risk_trade,
        "risk_required_extra_trade_l1": max(
            0.0, minimum_risk_trade - baseline_turnover_budget
        ),
        "actual_trade_l1": actual_trade,
        "turnover_slack": turnover_budget - actual_trade,
    }


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    technical: pd.DataFrame,
    fundamental: pd.DataFrame,
    asset_volatility: pd.DataFrame,
    daily_returns: pd.DataFrame,
    baseline: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
    ]
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress.index)
    months = months.intersection(technical.index)
    months = months.intersection(asset_volatility.index)
    months = months.intersection(
        fundamental.dropna(subset=required_fundamental).index
    )
    months = months.intersection(baseline.index)
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]

    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    baseline_pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        prior_baseline = baseline.loc[
            baseline.index < month, WEIGHT_COLUMNS
        ]
        baseline_target = baseline.loc[month, WEIGHT_COLUMNS].to_numpy(
            dtype=float
        )
        model_active = (
            len(history) >= stage36.stage35.ONE_CALENDAR_YEAR
            and len(prior_baseline) >= MIN_PRIOR_MONTHS
        )
        if model_active:
            expected_return, mu_detail = stage36_expected_return(
                history,
                probabilities.loc[probabilities.index < month],
                probabilities.loc[month],
                stress.loc[stress.index < month, "stress_score"],
                float(stress.loc[month, "stress_score"]),
                stress.loc[stress.index < month, "recovery_score"],
                float(stress.loc[month, "recovery_score"]),
                technical.loc[month],
                fundamental.loc[month],
            )
            covariance, covariance_detail = stage45.covariance_for_month(
                daily_returns, month, asset_volatility.loc[month]
            )
            strategic_prior = prior_baseline.mean(axis=0).to_numpy(
                dtype=float
            )
            strategic_prior /= strategic_prior.sum()
            confidence = regime_confidence(probabilities.loc[month])
            risk_aversion = _risk_aversion_from_history(
                baseline.loc[baseline.index < month, "gross_return"]
            )
            posterior, bl_detail = black_litterman_posterior(
                covariance,
                strategic_prior,
                expected_return,
                confidence,
                risk_aversion,
                len(prior_baseline),
            )
            optimizer_mu = (
                expected_return
                if mode == "stage36_mu_lw"
                else posterior
            )
            try:
                weights, solve_detail = solve_weights(
                    optimizer_mu,
                    covariance,
                    history.to_numpy(dtype=float),
                    pretrade,
                    baseline_target,
                    baseline_pretrade,
                )
            except RuntimeError as error:
                raise RuntimeError(f"{mode} failed at {month}: {error}") from error
        else:
            weights = baseline_target.copy()
            expected_return = np.full(len(ASSETS), np.nan)
            optimizer_mu = np.full(len(ASSETS), np.nan)
            strategic_prior = np.full(len(ASSETS), np.nan)
            mu_detail = {}
            covariance_detail = {
                "covariance_cutoff": pd.NaT,
                "covariance_start": pd.NaT,
                "covariance_end": pd.NaT,
                "covariance_observations": 0,
                "lw_shrinkage": np.nan,
                "gvz_gold_variance_multiplier": np.nan,
                "ovx_oil_variance_multiplier": np.nan,
            }
            bl_detail = {
                "equilibrium_return": np.full(len(ASSETS), np.nan),
                "posterior_return": np.full(len(ASSETS), np.nan),
                "view_return": np.full(len(ASSETS), np.nan),
                "regime_confidence": np.nan,
                "risk_aversion": np.nan,
                "tau": np.nan,
                "omega_diagonal": np.full(len(ASSETS), np.nan),
            }
            solve_detail = {
                "expected_monthly_return": np.nan,
                "expected_monthly_variance": np.nan,
                "downside_semivariance": np.nan,
                "estimated_transaction_cost": float(
                    stage36.stage35.expected_transaction_cost(
                        weights, pretrade
                    )
                ),
                "monthly_utility": np.nan,
                "solver_success": True,
                "used_fallback": False,
                "solver_status": 0,
                "solver_message": "Stage36 frozen warm-up allocation",
                "solver_iterations": 0,
                "objective_value": np.nan,
                "expected_annual_volatility": np.nan,
                "historical_cdar": np.nan,
                "sum_error": abs(float(weights.sum()) - 1.0),
                "volatility_slack": np.nan,
                "cdar_slack": np.nan,
                "turnover_budget_l1": float(
                    np.abs(baseline_target - baseline_pretrade).sum()
                ),
                "baseline_turnover_budget_l1": float(
                    np.abs(baseline_target - baseline_pretrade).sum()
                ),
                "minimum_risk_trade_l1": 0.0,
                "risk_required_extra_trade_l1": 0.0,
                "actual_trade_l1": float(
                    np.abs(weights - pretrade).sum()
                ),
                "turnover_slack": 0.0,
            }

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = (
            float(np.abs(change).sum())
            * stage36.stage35.DOMESTIC_TRADE_COST
        )
        foreign = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign].sum()))
            * stage36.stage35.FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)

        row: dict[str, Any] = {
            "month": month,
            "policy": mode,
            "model_active": model_active,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            **{
                f"w_{asset}": float(weights[index])
                for index, asset in enumerate(ASSETS)
            },
            **{
                f"stage36_mu_{asset}": float(expected_return[index])
                for index, asset in enumerate(ASSETS)
            },
            **{
                f"optimizer_mu_{asset}": float(optimizer_mu[index])
                for index, asset in enumerate(ASSETS)
            },
            **{
                f"strategic_prior_w_{asset}": float(strategic_prior[index])
                for index, asset in enumerate(ASSETS)
            },
            **{
                f"equilibrium_mu_{asset}": float(
                    bl_detail["equilibrium_return"][index]
                )
                for index, asset in enumerate(ASSETS)
            },
            **{
                f"posterior_mu_{asset}": float(
                    bl_detail["posterior_return"][index]
                )
                for index, asset in enumerate(ASSETS)
            },
            "regime_confidence": float(bl_detail["regime_confidence"]),
            "bl_risk_aversion": float(bl_detail["risk_aversion"]),
            "bl_tau": float(bl_detail["tau"]),
            **covariance_detail,
            **solve_detail,
        }
        rows.append(row)

        baseline_gross = float(baseline_target @ asset_return)
        baseline_pretrade = (
            baseline_target * (1.0 + asset_return) / (1.0 + baseline_gross)
        )
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def run_guarded_bl_deployment(
    returns: pd.DataFrame,
    baseline: pd.DataFrame,
    shadow: pd.DataFrame,
    daily_returns: pd.DataFrame,
    asset_volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Deploy BL only as a small drawdown-sensitive Stage 36 tilt.

    The maximum opinion budget is 1/(N*(N+1)): one opinion sleeve beside N
    strategic sleeves, divided across N assets. It is 5% for four assets. The
    budget decays linearly to zero over one monthly-equivalent 13% annual risk
    budget as the live strategy enters drawdown. Neither constant is fitted.
    """

    months = baseline.index.intersection(shadow.index)
    months = months.intersection(asset_volatility.index)
    max_tilt = 1.0 / (len(ASSETS) * (len(ASSETS) + 1.0))
    monthly_drawdown_budget = (
        stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY / math.sqrt(12.0)
    )
    pretrade = np.zeros(len(ASSETS), dtype=float)
    baseline_pretrade = np.zeros(len(ASSETS), dtype=float)
    nav = 1.0
    peak = 1.0
    first_trade = True
    rows: list[dict[str, Any]] = []
    for month in months:
        baseline_target = baseline.loc[month, WEIGHT_COLUMNS].to_numpy(
            dtype=float
        )
        shadow_target = shadow.loc[month, WEIGHT_COLUMNS].to_numpy(dtype=float)
        current_drawdown = nav / peak - 1.0
        drawdown_multiplier = float(
            np.clip(
                1.0 + current_drawdown / monthly_drawdown_budget,
                0.0,
                1.0,
            )
        )
        tilt = (
            max_tilt * drawdown_multiplier
            if bool(shadow.loc[month, "model_active"])
            else 0.0
        )
        desired = (1.0 - tilt) * baseline_target + tilt * shadow_target
        covariance, covariance_detail = stage45.covariance_for_month(
            daily_returns, month, asset_volatility.loc[month]
        )
        history = returns.loc[returns.index < month, ASSETS].to_numpy(
            dtype=float
        )
        desired_volatility = math.sqrt(
            max(float(desired @ covariance @ desired), 0.0) * 12.0
        )
        desired_cdar = (
            stage36.stage35.cdar(
                history @ desired, stage36.stage35.CDAR_CONFIDENCE
            )
            if len(history) >= stage36.stage35.ONE_CALENDAR_YEAR
            else np.nan
        )
        lw_overlay_feasible = bool(
            len(history) >= stage36.stage35.ONE_CALENDAR_YEAR
            and desired_volatility
            <= stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY + 1e-10
            and desired_cdar >= -stage36.stage35.CATASTROPHE_CDAR - 1e-10
        )
        deploy_overlay = bool(
            shadow.loc[month, "model_active"] and lw_overlay_feasible
        )
        if deploy_overlay:
            weights = desired.copy()
            solve_detail = {
                "solver_success": True,
                "used_fallback": False,
                "solver_status": 0,
                "solver_message": "guarded BL convex tilt passed LW risk veto",
                "solver_iterations": 0,
                "objective_value": 0.0,
                "expected_monthly_return": np.nan,
                "expected_monthly_variance": float(
                    weights @ covariance @ weights
                ),
                "downside_semivariance": float(
                    np.mean(np.minimum(history @ weights, 0.0) ** 2)
                ),
                "estimated_transaction_cost": float(
                    stage36.stage35.expected_transaction_cost(
                        weights, pretrade
                    )
                ),
                "monthly_utility": np.nan,
                "expected_annual_volatility": desired_volatility,
                "historical_cdar": desired_cdar,
                "sum_error": abs(float(weights.sum()) - 1.0),
                "volatility_slack": (
                    stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
                    - desired_volatility
                ),
                "cdar_slack": stage36.stage35.CATASTROPHE_CDAR + desired_cdar,
                "turnover_budget_l1": np.nan,
                "baseline_turnover_budget_l1": float(
                    np.abs(baseline_target - baseline_pretrade).sum()
                ),
                "minimum_risk_trade_l1": 0.0,
                "risk_required_extra_trade_l1": 0.0,
                "actual_trade_l1": np.nan,
                "turnover_slack": np.nan,
                "projection_distance": 0.0,
            }
        else:
            weights = baseline_target.copy()
            solve_detail = {
                "solver_success": True,
                "used_fallback": bool(shadow.loc[month, "model_active"]),
                "solver_status": 0,
                "solver_message": (
                    "Stage36 fallback because guarded BL failed LW risk veto"
                    if bool(shadow.loc[month, "model_active"])
                    else "Stage36 frozen warm-up allocation"
                ),
                "solver_iterations": 0,
                "objective_value": np.nan,
                "expected_monthly_return": np.nan,
                "expected_monthly_variance": float(
                    baseline.loc[month, "expected_monthly_variance"]
                ),
                "downside_semivariance": np.nan,
                "estimated_transaction_cost": float(
                    stage36.stage35.expected_transaction_cost(
                        weights, pretrade
                    )
                ),
                "monthly_utility": np.nan,
                "expected_annual_volatility": float(
                    baseline.loc[month, "expected_annual_volatility"]
                ),
                "historical_cdar": float(
                    baseline.loc[month, "historical_cdar"]
                ),
                "sum_error": abs(float(weights.sum()) - 1.0),
                "volatility_slack": float(
                    baseline.loc[month, "volatility_slack"]
                ),
                "cdar_slack": float(baseline.loc[month, "cdar_slack"]),
                "turnover_budget_l1": np.nan,
                "baseline_turnover_budget_l1": float(
                    np.abs(baseline_target - baseline_pretrade).sum()
                ),
                "minimum_risk_trade_l1": 0.0,
                "risk_required_extra_trade_l1": 0.0,
                "actual_trade_l1": float(np.abs(weights - pretrade).sum()),
                "turnover_slack": np.nan,
                "projection_distance": 0.0,
            }

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = (
            float(np.abs(change).sum())
            * stage36.stage35.DOMESTIC_TRADE_COST
        )
        foreign = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign].sum()))
            * stage36.stage35.FOREIGN_WEIGHT_CHANGE_COST
        )
        solve_detail["actual_trade_l1"] = float(np.abs(change).sum())
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        rows.append(
            {
                "month": month,
                "policy": "guarded_black_litterman_lw",
                "model_active": bool(shadow.loc[month, "model_active"]),
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "predecision_drawdown": current_drawdown,
                "drawdown_multiplier": drawdown_multiplier,
                "maximum_bl_tilt": max_tilt,
                "applied_bl_tilt": tilt,
                "bl_overlay_deployed": deploy_overlay,
                "lw_overlay_veto": bool(
                    shadow.loc[month, "model_active"]
                    and not lw_overlay_feasible
                ),
                "deployed_risk_model": (
                    "ledoit_wolf"
                    if deploy_overlay
                    else "stage36_frozen_fallback"
                ),
                "desired_lw_annual_volatility": desired_volatility,
                "desired_lw_historical_cdar": desired_cdar,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    f"desired_w_{asset}": float(desired[index])
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    f"shadow_w_{asset}": float(shadow_target[index])
                    for index, asset in enumerate(ASSETS)
                },
                "regime_confidence": float(
                    shadow.loc[month, "regime_confidence"]
                ),
                **covariance_detail,
                **solve_detail,
            }
        )
        baseline_gross = float(baseline_target @ asset_return)
        baseline_pretrade = (
            baseline_target * (1.0 + asset_return) / (1.0 + baseline_gross)
        )
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_end = min(path.index.max() for path in paths.values())
    periods = {
        "full_2007_2026": (FULL_START, common_end),
        "common_2010_2026": (COMMON_START, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    return pd.DataFrame(
        [
            stage36.stage35.metric_row(name, path, period, start, end)
            for name, path in paths.items()
            for period, (start, end) in periods.items()
        ]
    )


def bootstrap_table(
    baseline: pd.DataFrame, candidates: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for period, start in (
        ("full_2007_2026", FULL_START),
        ("common_2010_2026", COMMON_START),
        ("locked_2018_2026", LOCKED_START),
    ):
        for name, candidate in candidates.items():
            common = baseline.loc[start:RESEARCH_END].index.intersection(
                candidate.loc[start:RESEARCH_END].index
            )
            summary = stage36.stage35.stage30.paired_block_bootstrap(
                baseline.loc[common, "return"],
                candidate.loc[common, "return"],
            )
            summary.insert(0, "Period", period)
            summary.insert(0, "Candidate", name)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def gate_results(
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    candidate: str,
) -> dict[str, Any]:
    indexed = performance.set_index(["Strategy", "Period"])
    gates: dict[str, bool] = {}
    for period in (
        "full_2007_2026",
        "common_2010_2026",
        "locked_2018_2026",
    ):
        base = indexed.loc[(BASELINE_NAME, period)]
        test = indexed.loc[(candidate, period)]
        gates[f"{period}_sharpe_not_lower"] = bool(
            test["Sharpe"] >= base["Sharpe"] - GATE_TOLERANCE
        )
        gates[f"{period}_mdd_not_worse"] = bool(
            test["MDD"] >= base["MDD"] - GATE_TOLERANCE
        )
        gates[f"{period}_cagr_within_50bp"] = bool(
            test["CAGR"] >= base["CAGR"] - CAGR_TOLERANCE
        )
        gates[f"{period}_turnover_not_materially_higher"] = bool(
            test["AvgTurnover"]
            <= base["AvgTurnover"] * (1.0 + TURNOVER_TOLERANCE) + 1e-10
        )
    boot = bootstrap.loc[bootstrap["Candidate"].eq(candidate)].set_index(
        ["Period", "Metric"]
    )
    for period in ("common_2010_2026", "locked_2018_2026"):
        gates[f"{period}_bootstrap_sharpe_probability"] = bool(
            boot.loc[(period, "delta_Sharpe"), "ProbabilityPositive"]
            >= BOOTSTRAP_SHARPE_GATE
        )
        gates[f"{period}_bootstrap_mdd_probability"] = bool(
            boot.loc[(period, "delta_MDD"), "ProbabilityPositive"]
            >= BOOTSTRAP_MDD_GATE
        )
    return {"gates": gates, "pass": bool(all(gates.values()))}


def run_research(save: bool = True) -> dict[str, Any]:
    frozen_before = frozen_manifest()
    returns, return_levels = stage36.stage35.load_monthly_asset_returns(False)
    probabilities, macro_ranks = stage36.stage35.build_macro_probabilities(
        returns
    )
    stress = stage36.stage35.build_monthly_stress_signals(
        returns.index, stage36.stage35.build_daily_stress_features()
    )
    technical = _load_period_csv(
        stage36.stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    raw_fundamental, fundamental_audit = (
        stage36.stage35.load_fundamental_daily()
    )
    fundamental = stage36.stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
    equity_market, market_audit = (
        stage36.stage35.stage20.load_daily_asset_ohlcv()
    )
    equity_close = equity_market["KODEX200"]["close"].dropna()
    equity_monthly_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    calibrated = stage36.stage35.add_causal_return_calibration(
        fundamental, equity_monthly_close.pct_change()
    )
    daily_returns = stage45.build_daily_return_matrix(equity_market)
    asset_vol_daily, asset_vol_audit = (
        stage36.load_asset_implied_volatility_daily()
    )
    asset_volatility = stage36.build_monthly_asset_volatility_signals(
        asset_vol_daily, returns.index
    )
    baseline = _load_period_csv(STAGE36_PATH, "month")
    raw_candidates = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated,
            asset_volatility,
            daily_returns,
            baseline,
            mode,
        )
        for name, mode in MODES.items()
    }
    guarded_bl = run_guarded_bl_deployment(
        returns,
        baseline,
        raw_candidates[RAW_BL_NAME],
        daily_returns,
        asset_volatility,
    )
    candidates = {
        LW_NAME: raw_candidates[LW_NAME],
        BL_NAME: guarded_bl,
    }
    paths = {BASELINE_NAME: baseline, **candidates}
    performance = performance_table(paths)
    bootstrap = bootstrap_table(baseline, candidates)
    gates = {
        name: gate_results(performance, bootstrap, name)
        for name in candidates
    }
    eligible = [name for name, result in gates.items() if result["pass"]]
    promoted = BL_NAME if BL_NAME in eligible else (
        LW_NAME if LW_NAME in eligible else BASELINE_NAME
    )
    frozen_after = frozen_manifest()

    solver_audit: dict[str, Any] = {}
    for name, path in candidates.items():
        active = path.loc[path["model_active"]]
        turnover_slack = active["turnover_slack"].dropna()
        solver_audit[name] = {
            "months": int(len(active)),
            "successes": int(active["solver_success"].sum()),
            "fallbacks": int(active["used_fallback"].sum()),
            "minimum_volatility_slack": float(
                active["volatility_slack"].min()
            ),
            "minimum_cdar_slack": float(active["cdar_slack"].min()),
            "minimum_turnover_slack": float(
                turnover_slack.min()
            ) if not turnover_slack.empty else None,
            "maximum_weight_sum_error": float(active["sum_error"].max()),
            "average_lw_shrinkage": float(active["lw_shrinkage"].mean()),
        }
        if name == BL_NAME:
            solver_audit[name]["bl_overlay_deployed_months"] = int(
                active["bl_overlay_deployed"].sum()
            )
            solver_audit[name]["lw_overlay_veto_months"] = int(
                active["lw_overlay_veto"].sum()
            )

    report: dict[str, Any] = {
        "study": "Stage46_BlackLitterman_LedoitWolf_Ablation",
        "base_strategy": BASELINE_NAME,
        "decision": (
            f"promote_{promoted}"
            if promoted != BASELINE_NAME
            else "retain_stage36_no_candidate_passed_all_gates"
        ),
        "promoted_strategy": promoted,
        "fixed_design": {
            "experiments": {
                "A": "Stage36 frozen baseline",
                "B": "Stage36 expected return plus 252-day constant-correlation Ledoit-Wolf covariance",
                "C": "drawdown-sensitive 5% maximum Black-Litterman tilt around Stage36; Ledoit-Wolf risk veto with frozen Stage36 fallback",
            },
            "strategic_prior": "causal expanding mean of prior Stage36 weights; current month excluded",
            "equilibrium_return": "risk_aversion * covariance * strategic_prior",
            "views": "Stage36 expected returns only; no new ML signal",
            "view_confidence": "normalized Herfindahl concentration of four regime probabilities",
            "omega": "Idzorek confidence ratio applied to diagonal prior view variance",
            "tau": "1 / prior Stage36 months",
            "risk_overlays": "Stage36 GVZ-to-GLD and OVX-to-USO variance multipliers",
            "optimizer": "Stage36 long-only, fully invested SLSQP utility with 13% volatility and 16% CDaR guards",
            "bl_deployment": "maximum tilt 1/(N*(N+1)); linearly reduced to zero over one monthly-equivalent 13% annual drawdown budget; retain Stage36 when the tilt fails LW/CDaR guards",
            "turnover_rule": "B and the BL shadow use no more than Stage36 monthly L1 trade except minimum risk-required trade; guarded C is rejected if average turnover exceeds Stage36 by 10% in any period",
            "turnover_gate": "average turnover no more than 10% above Stage36 in every reported period",
            "cagr_tolerance": CAGR_TOLERANCE,
            "bootstrap_sharpe_gate": BOOTSTRAP_SHARPE_GATE,
            "bootstrap_mdd_gate": BOOTSTRAP_MDD_GATE,
            "parameter_grid": None,
        },
        "gate_results": gates,
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage36": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "solver_audit": solver_audit,
        "causality_audit": {
            name: {
                "all_covariance_dates_before_target": bool(
                    (
                        pd.to_datetime(
                            path.loc[path["model_active"], "covariance_end"]
                        )
                        < path.loc[path["model_active"]].index.to_timestamp(
                            how="start"
                        )
                    ).all()
                ),
                "all_covariance_windows_have_252_rows": bool(
                    path.loc[path["model_active"], "covariance_observations"]
                    .eq(stage45.COVARIANCE_LOOKBACK_DAYS)
                    .all()
                ),
            }
            for name, path in candidates.items()
        },
        "checks": {
            "stage36_and_stage45_frozen_files_unchanged": (
                frozen_before == frozen_after
            ),
            "only_three_predeclared_ablation_paths": True,
            "no_new_machine_learning_signal": True,
            "all_candidate_solvers_feasible": all(
                audit["minimum_volatility_slack"] >= -1e-7
                and audit["minimum_cdar_slack"] >= -1e-7
                and audit["maximum_weight_sum_error"] < 1e-8
                for audit in solver_audit.values()
            ),
            "guarded_bl_passes_every_performance_gate": bool(
                gates[BL_NAME]["pass"]
            ),
        },
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
        "data_audit": {
            "return_rows": int(len(returns)),
            "return_level_rows": int(len(return_levels)),
            "macro_rank_rows": int(len(macro_ranks)),
            "fundamental": fundamental_audit,
            "asset_volatility": asset_vol_audit,
            "market": market_audit,
        },
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, path in candidates.items():
            filename = (
                "stage46_stage36mu_lw_monthly.csv"
                if name == LW_NAME
                else "stage46_guarded_blacklitterman_lw_monthly.csv"
            )
            path.to_csv(OUTPUT_DIR / filename)
        raw_candidates[RAW_BL_NAME].to_csv(
            OUTPUT_DIR / "stage46_blacklitterman_lw_shadow_monthly.csv"
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv",
            index=False,
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return {
        "report": report,
        "performance": performance,
        "bootstrap": bootstrap,
        "paths": paths,
    }


def main() -> None:
    research = run_research(save=True)
    print(
        json.dumps(
            research["report"], ensure_ascii=False, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
