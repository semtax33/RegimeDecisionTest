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
from strategies.stage46_black_litterman_shrinkage import (
    black_litterman_shrinkage_slsqp as stage46,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = stage36.ASSETS
WEIGHT_COLUMNS = stage36.WEIGHT_COLUMNS
FULL_START = stage36.FULL_START
COMMON_START = stage36.COMMON_START
LOCKED_START = stage36.stage35.LOCKED_START
RESEARCH_END = stage36.RESEARCH_END

BASELINE_NAME = stage46.BASELINE_NAME
B_NAME = "Stage47_B_GuardedStage36Objective_LW"
C_NAME = "Stage47_C_GuardedExAnteSharpe_LW"
D_NAME = "Stage47_D_GuardedDownsidePenalty_LW"
E_NAME = "Stage47_E_GuardedDirectSortino_LW"
OBJECTIVE_MODES = {
    C_NAME: "ex_ante_sharpe",
    D_NAME: "downside_penalty",
    E_NAME: "direct_sortino",
}
PROMOTION_PRIORITY = (D_NAME, C_NAME, B_NAME, E_NAME)

MIN_HISTORY_MONTHS = 12
CAGR_TOLERANCE = stage46.CAGR_TOLERANCE
TURNOVER_TOLERANCE = stage46.TURNOVER_TOLERANCE
NUMERICAL_EPSILON = 1e-12

STAGE36_PATH = stage46.STAGE36_PATH
FROZEN_FILES = (
    Path(stage36.__file__),
    STAGE36_PATH,
    stage36.OUTPUT_DIR / "validation_report.json",
    Path(stage45.__file__),
    stage45.OUTPUT_DIR / "validation_report.json",
    Path(stage46.__file__),
    stage46.OUTPUT_DIR / "validation_report.json",
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


def objective_statistics(
    weights: np.ndarray,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_returns: np.ndarray,
    historical_cash_returns: np.ndarray,
    current_cash_return: float,
    pretrade: np.ndarray,
) -> dict[str, float]:
    """Compute all three objective families from the same causal inputs."""

    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_return, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    history = np.asarray(historical_returns, dtype=float)
    cash_history = np.asarray(historical_cash_returns, dtype=float)
    transaction_cost = stage36.stage35.expected_transaction_cost(w, pretrade)
    expected_total_return = float(w @ mu)
    expected_excess_return = (
        expected_total_return - float(current_cash_return) - transaction_cost
    )
    monthly_variance = max(float(w @ sigma @ w), 0.0)
    historical_excess = history @ w - cash_history
    downside_semivariance = float(
        np.mean(np.minimum(historical_excess, 0.0) ** 2)
    )
    ex_ante_sharpe = expected_excess_return / math.sqrt(
        max(monthly_variance, NUMERICAL_EPSILON)
    )
    direct_sortino = expected_excess_return / math.sqrt(
        max(downside_semivariance, NUMERICAL_EPSILON)
    )
    downside_utility = expected_excess_return - downside_semivariance
    return {
        "expected_total_return": expected_total_return,
        "expected_excess_return_after_cost": expected_excess_return,
        "expected_monthly_variance": monthly_variance,
        "historical_downside_semivariance": downside_semivariance,
        "estimated_transaction_cost": transaction_cost,
        "ex_ante_sharpe_objective": ex_ante_sharpe,
        "downside_penalty_objective": downside_utility,
        "direct_sortino_objective": direct_sortino,
    }


def _objective_value(statistics: dict[str, float], mode: str) -> float:
    mapping = {
        "ex_ante_sharpe": "ex_ante_sharpe_objective",
        "downside_penalty": "downside_penalty_objective",
        "direct_sortino": "direct_sortino_objective",
    }
    if mode not in mapping:
        raise ValueError(f"unknown objective mode: {mode}")
    return float(statistics[mapping[mode]])


def solve_objective_weights(
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_returns: np.ndarray,
    historical_cash_returns: np.ndarray,
    current_cash_return: float,
    pretrade: np.ndarray,
    baseline_target: np.ndarray,
    baseline_pretrade: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one objective with Stage 36 risk and turnover constraints."""

    mu = np.asarray(expected_return, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    history = np.asarray(historical_returns, dtype=float)
    cash_history = np.asarray(historical_cash_returns, dtype=float)
    baseline_turnover_budget = float(
        np.abs(np.asarray(baseline_target) - baseline_pretrade).sum()
    )

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ sigma @ weights), 0.0) * 12.0)

    def historical_cdar(weights: np.ndarray) -> float:
        return stage36.stage35.cdar(
            history @ weights, stage36.stage35.CDAR_CONFIDENCE
        )

    def score(weights: np.ndarray) -> float:
        statistics = objective_statistics(
            weights,
            mu,
            sigma,
            history,
            cash_history,
            current_cash_return,
            pretrade,
        )
        return _objective_value(statistics, mode)

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
    initial_pretrade = stage36.stage35.project_to_long_only_simplex(pretrade)
    initial_baseline = stage36.stage35.project_to_long_only_simplex(
        baseline_target
    )
    initial_equal = np.repeat(1.0 / len(ASSETS), len(ASSETS))
    feasibility_initials = [initial_pretrade, initial_baseline, initial_equal]
    minimum_trade_solutions: list[Any] = []
    for initial in feasibility_initials:
        result = minimize(
            lambda weights: stage46._smooth_l1(weights - pretrade),
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
        raise RuntimeError(f"{mode}: no feasible portfolio inside risk guards")
    minimum_trade = min(
        minimum_trade_solutions, key=lambda item: float(item.fun)
    )
    minimum_risk_trade = float(
        np.abs(minimum_trade.x - pretrade).sum()
    )
    turnover_budget = max(
        baseline_turnover_budget, float(minimum_trade.fun) + 1e-7
    )
    constraints = [
        *risk_constraints,
        {
            "type": "ineq",
            "fun": lambda weights: float(
                turnover_budget - stage46._smooth_l1(weights - pretrade)
            ),
        },
    ]
    corner_initials = [
        np.eye(len(ASSETS), dtype=float)[index]
        for index in range(len(ASSETS))
    ]
    objective_initials = [
        minimum_trade.x,
        *feasibility_initials,
        *corner_initials,
    ]
    solutions: list[Any] = []
    for initial in objective_initials:
        result = minimize(
            lambda weights: -score(weights),
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
    used_fallback = not bool(solutions)
    result = (
        min(solutions, key=lambda item: float(item.fun))
        if solutions
        else minimum_trade
    )
    weights = stage36.stage35.project_to_long_only_simplex(result.x)
    statistics = objective_statistics(
        weights,
        mu,
        sigma,
        history,
        cash_history,
        current_cash_return,
        pretrade,
    )
    annual_vol = annual_volatility(weights)
    cdar_value = historical_cdar(weights)
    actual_trade = float(np.abs(weights - pretrade).sum())
    return weights, {
        **statistics,
        "objective_mode": mode,
        "selected_objective_value": _objective_value(statistics, mode),
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


def run_objective_shadow(
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
        "ktb_3y_pct",
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
    cash_returns = fundamental["ktb_3y_pct"] / 100.0 / 12.0

    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    baseline_pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history_months = returns.index[
            (returns.index < month) & returns.index.isin(cash_returns.index)
        ]
        history = returns.loc[history_months, ASSETS]
        baseline_target = baseline.loc[month, WEIGHT_COLUMNS].to_numpy(
            dtype=float
        )
        model_active = len(history) >= MIN_HISTORY_MONTHS
        if model_active:
            expected_return, _ = stage46.stage36_expected_return(
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
            weights, solve_detail = solve_objective_weights(
                expected_return,
                covariance,
                history.to_numpy(dtype=float),
                cash_returns.loc[history_months].to_numpy(dtype=float),
                float(cash_returns.loc[month]),
                pretrade,
                baseline_target,
                baseline_pretrade,
                mode,
            )
        else:
            weights = baseline_target.copy()
            expected_return = np.full(len(ASSETS), np.nan)
            covariance_detail = {
                "covariance_cutoff": pd.NaT,
                "covariance_start": pd.NaT,
                "covariance_end": pd.NaT,
                "covariance_observations": 0,
                "lw_shrinkage": np.nan,
                "gvz_gold_variance_multiplier": np.nan,
                "ovx_oil_variance_multiplier": np.nan,
            }
            solve_detail = {
                "objective_mode": mode,
                "selected_objective_value": np.nan,
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
                "actual_trade_l1": float(np.abs(weights - pretrade).sum()),
                "turnover_slack": 0.0,
                "expected_total_return": np.nan,
                "expected_excess_return_after_cost": np.nan,
                "expected_monthly_variance": np.nan,
                "historical_downside_semivariance": np.nan,
                "estimated_transaction_cost": np.nan,
                "ex_ante_sharpe_objective": np.nan,
                "downside_penalty_objective": np.nan,
                "direct_sortino_objective": np.nan,
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
        rows.append(
            {
                "month": month,
                "policy": f"shadow_{mode}",
                "model_active": model_active,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "cash_return": float(cash_returns.loc[month]),
                "regime_confidence": stage46.regime_confidence(
                    probabilities.loc[month]
                ),
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    f"stage36_mu_{asset}": float(expected_return[index])
                    for index, asset in enumerate(ASSETS)
                },
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


def guard_shadow(
    name: str,
    returns: pd.DataFrame,
    baseline: pd.DataFrame,
    shadow: pd.DataFrame,
    daily_returns: pd.DataFrame,
    asset_volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the identical Stage 46 5% drawdown and LW risk veto."""

    guarded = stage46.run_guarded_bl_deployment(
        returns, baseline, shadow, daily_returns, asset_volatility
    ).copy()
    guarded["policy"] = name
    guarded = guarded.rename(
        columns={
            "maximum_bl_tilt": "maximum_objective_tilt",
            "applied_bl_tilt": "applied_objective_tilt",
            "bl_overlay_deployed": "objective_overlay_deployed",
            "lw_overlay_veto": "objective_lw_risk_veto",
        }
    )
    return guarded


def performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return stage46.performance_table(paths)


def bootstrap_table(
    baseline: pd.DataFrame, candidates: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    return stage46.bootstrap_table(baseline, candidates)


def gate_results(
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    candidate: str,
) -> dict[str, Any]:
    return stage46.gate_results(performance, bootstrap, candidate)


def run_research(save: bool = True) -> dict[str, Any]:
    frozen_before = frozen_manifest()
    returns, return_levels = stage36.stage35.load_monthly_asset_returns(False)
    probabilities, macro_ranks = stage36.stage35.build_macro_probabilities(
        returns
    )
    stress = stage36.stage35.build_monthly_stress_signals(
        returns.index, stage36.stage35.build_daily_stress_features()
    )
    technical = stage46._load_period_csv(
        stage36.stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    raw_fundamental, fundamental_audit = (
        stage36.stage35.load_fundamental_daily()
    )
    fundamental = stage36.stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
    market, market_audit = stage36.stage35.stage20.load_daily_asset_ohlcv()
    equity_close = market["KODEX200"]["close"].dropna()
    monthly_close = equity_close.groupby(equity_close.index.to_period("M")).last()
    calibrated = stage36.stage35.add_causal_return_calibration(
        fundamental, monthly_close.pct_change()
    )
    daily_returns = stage45.build_daily_return_matrix(market)
    asset_vol_daily, asset_vol_audit = (
        stage36.load_asset_implied_volatility_daily()
    )
    asset_volatility = stage36.build_monthly_asset_volatility_signals(
        asset_vol_daily, returns.index
    )
    baseline = stage46._load_period_csv(STAGE36_PATH, "month")

    b_shadow = stage46.run_backtest(
        returns,
        probabilities,
        stress,
        technical,
        calibrated,
        asset_volatility,
        daily_returns,
        baseline,
        "stage36_mu_lw",
    )
    shadows = {B_NAME: b_shadow}
    for name, mode in OBJECTIVE_MODES.items():
        shadows[name] = run_objective_shadow(
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
    candidates = {
        name: guard_shadow(
            name,
            returns,
            baseline,
            shadow,
            daily_returns,
            asset_volatility,
        )
        for name, shadow in shadows.items()
    }
    paths = {BASELINE_NAME: baseline, **candidates}
    performance = performance_table(paths)
    bootstrap = bootstrap_table(baseline, candidates)
    gates = {
        name: gate_results(performance, bootstrap, name)
        for name in candidates
    }
    eligible = [
        name for name in PROMOTION_PRIORITY if gates[name]["pass"]
    ]
    promoted = eligible[0] if eligible else BASELINE_NAME
    frozen_after = frozen_manifest()

    shadow_audit: dict[str, Any] = {}
    deployment_audit: dict[str, Any] = {}
    for name, shadow in shadows.items():
        active = shadow.loc[shadow["model_active"]]
        shadow_audit[name] = {
            "active_months": int(len(active)),
            "solver_successes": int(active["solver_success"].sum()),
            "fallbacks": int(active["used_fallback"].sum()),
            "minimum_volatility_slack": float(
                active["volatility_slack"].min()
            ),
            "minimum_cdar_slack": float(active["cdar_slack"].min()),
            "minimum_turnover_slack": float(
                active["turnover_slack"].min()
            ),
            "maximum_weight_sum_error": float(active["sum_error"].max()),
        }
        deployed = candidates[name]
        deployment_audit[name] = {
            "maximum_objective_tilt": float(
                deployed["applied_objective_tilt"].max()
            ),
            "overlay_deployed_months": int(
                deployed["objective_overlay_deployed"].sum()
            ),
            "lw_risk_veto_months": int(
                deployed["objective_lw_risk_veto"].sum()
            ),
            "long_only": bool(
                (deployed[WEIGHT_COLUMNS].to_numpy() >= -1e-10).all()
            ),
            "fully_invested": bool(
                np.allclose(
                    deployed[WEIGHT_COLUMNS].sum(axis=1), 1.0, atol=1e-8
                )
            ),
        }

    report: dict[str, Any] = {
        "study": "Stage47_Guarded_Ratio_Downside_Objectives",
        "base_strategy": BASELINE_NAME,
        "decision": (
            f"promote_{promoted}"
            if promoted != BASELINE_NAME
            else "retain_stage36_no_objective_passed_all_gates"
        ),
        "promoted_strategy": promoted,
        "promotion_priority": list(PROMOTION_PRIORITY),
        "fixed_design": {
            "experiments": {
                "A": "Stage36 frozen baseline",
                "B": "Stage36 expected return plus LW covariance plus original Stage36 utility",
                "C": "Stage36 expected return plus LW covariance; directly maximize ex-ante Sharpe",
                "D": "Stage36 expected return plus LW guard; maximize excess return after cost minus historical excess downside semivariance",
                "E": "Stage36 expected return plus LW guard; directly maximize ex-ante Sortino",
            },
            "cash_proxy": "prior-known KTB 3Y annual yield divided by 12",
            "covariance": "252 complete prior daily rows; constant-correlation Ledoit-Wolf; Stage36 GVZ/OVX variance scaling",
            "shadow_constraints": "long-only, fully invested, 13% annual volatility, -16% CDaR, Stage36 turnover budget except minimum risk-required trade",
            "common_deployment_guard": "Stage46 rule: maximum 1/(N*(N+1))=5% tilt, drawdown attenuation, LW/CDaR risk veto, Stage36 fallback",
            "downside_penalty_lambda": 1.0,
            "parameter_grid": None,
            "promotion_rule": "predeclared D then C then B then E; first candidate passing every Stage46 performance and majority-positive bootstrap gate",
        },
        "gate_results": gates,
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage36": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "shadow_solver_audit": shadow_audit,
        "deployment_audit": deployment_audit,
        "causality_audit": {
            name: {
                "all_covariance_dates_before_target": bool(
                    (
                        pd.to_datetime(
                            shadow.loc[
                                shadow["model_active"], "covariance_end"
                            ]
                        )
                        < shadow.loc[shadow["model_active"]]
                        .index.to_timestamp(how="start")
                    ).all()
                ),
                "all_covariance_windows_have_252_rows": bool(
                    shadow.loc[
                        shadow["model_active"], "covariance_observations"
                    ]
                    .eq(stage45.COVARIANCE_LOOKBACK_DAYS)
                    .all()
                ),
            }
            for name, shadow in shadows.items()
        },
        "checks": {
            "stage36_stage45_stage46_frozen_files_unchanged": (
                frozen_before == frozen_after
            ),
            "exactly_five_predeclared_performance_paths": bool(
                performance["Strategy"].nunique() == 5
            ),
            "no_new_return_forecast_model": True,
            "all_shadow_solvers_feasible": all(
                audit["minimum_volatility_slack"] >= -1e-7
                and audit["minimum_cdar_slack"] >= -1e-7
                and audit["minimum_turnover_slack"] >= -1e-7
                and audit["maximum_weight_sum_error"] < 1e-8
                for audit in shadow_audit.values()
            ),
            "all_deployments_long_only_fully_invested": all(
                audit["long_only"] and audit["fully_invested"]
                for audit in deployment_audit.values()
            ),
            "promoted_strategy_passes_every_gate": bool(
                promoted != BASELINE_NAME and gates[promoted]["pass"]
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
        filenames = {
            B_NAME: "stage47_b_guarded_stage36objective_lw_monthly.csv",
            C_NAME: "stage47_c_guarded_exante_sharpe_lw_monthly.csv",
            D_NAME: "stage47_d_guarded_downside_penalty_lw_monthly.csv",
            E_NAME: "stage47_e_guarded_direct_sortino_lw_monthly.csv",
        }
        shadow_filenames = {
            B_NAME: "stage47_b_stage36objective_lw_shadow.csv",
            C_NAME: "stage47_c_exante_sharpe_lw_shadow.csv",
            D_NAME: "stage47_d_downside_penalty_lw_shadow.csv",
            E_NAME: "stage47_e_direct_sortino_lw_shadow.csv",
        }
        for name, path in candidates.items():
            path.to_csv(OUTPUT_DIR / filenames[name])
        for name, path in shadows.items():
            path.to_csv(OUTPUT_DIR / shadow_filenames[name])
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
        "shadows": shadows,
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
