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


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = stage36.ASSETS
WEIGHT_COLUMNS = stage36.WEIGHT_COLUMNS
FULL_START = stage36.FULL_START
COMMON_START = stage36.COMMON_START
LOCKED_START = stage36.stage35.LOCKED_START
RESEARCH_END = stage36.RESEARCH_END

STRATEGY_NAME = "Stage43_DynamicDD12_ExAnteSharpe11"
STAGE36_NAME = "Stage36_Frozen"
STAGE36_MODE = "gvz_ovx_asset_risk"

NAV_FLOOR_RATIO = 0.88
MDD_FLOOR = NAV_FLOOR_RATIO - 1.0
EX_ANTE_SHARPE_FLOOR = 1.10
TAIL_CONFIDENCE = stage36.stage35.CDAR_CONFIDENCE
CONSTRAINT_TOLERANCE = 1e-7

FROZEN_STAGE36_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
)


class InfeasiblePortfolioError(RuntimeError):
    """No fully invested long-only portfolio satisfies the strict constraints."""


class DrawdownFloorAlreadyBreached(RuntimeError):
    """The realized strategy NAV is already below 88% of its running peak."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_stage36_manifest() -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in FROZEN_STAGE36_FILES}


def remaining_loss_budget(current_drawdown: float) -> float:
    """Return the additional loss that would place NAV exactly at 88% of peak.

    If current NAV is ``(1+d) * peak``, the next-period return at the hard floor
    solves ``(1+d) * (1+L) = 0.88``.  No fitted risk multiplier is involved.
    """

    drawdown = float(current_drawdown)
    if not np.isfinite(drawdown) or drawdown <= -1.0:
        raise ValueError("Current drawdown must be finite and greater than -100%.")
    return float(NAV_FLOOR_RATIO / (1.0 + drawdown) - 1.0)


def historical_cdar(portfolio_returns: np.ndarray) -> float:
    """Use Stage36's exact drawdown-at-risk definition and 90% confidence."""

    return stage36.stage35.cdar(
        np.asarray(portfolio_returns, dtype=float), TAIL_CONFIDENCE
    )


def build_stage36_forecast(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    fundamental_signal: pd.Series,
    asset_vol_signal: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rebuild Stage36's current monthly mu and covariance without its utility.

    This mirrors Stage36's macro conditional moments, daily technical filter,
    earnings/valuation adjustments, credit covariance scaling and GVZ/OVX
    variance-only scaling.  Stage43 changes the optimization layer, not the
    forecast information set.
    """

    _, base_covariance, moment_detail = stage36.stage35.estimate_conditional_moments(
        history=history,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=True,
    )
    macro_expected_return = np.asarray(
        moment_detail["macro_expected_monthly_return"], dtype=float
    )
    stress_adjustment = np.asarray(
        moment_detail["stress_return_adjustment"], dtype=float
    ).copy()
    technical = stage36.stage35.stage20.apply_technical_inputs(
        macro_expected_return, base_covariance, technical_signal
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
    credit_variance_multiplier = 1.0 + float(
        fundamental_signal["credit_stress_rank"]
    )
    stress_adjustment[stage36.stage35.EQUITY_INDEX] *= credit_stress_multiplier
    filtered_macro[stage36.stage35.EQUITY_INDEX] += eps_mu + valuation_mu
    expected_return = filtered_macro + stress_adjustment

    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)
    credit_scaling = np.eye(len(ASSETS), dtype=float)
    credit_scaling[
        stage36.stage35.EQUITY_INDEX, stage36.stage35.EQUITY_INDEX
    ] = math.sqrt(credit_variance_multiplier)
    covariance = credit_scaling @ covariance @ credit_scaling

    gvz_multiplier = float(asset_vol_signal["gvz_gld_variance_multiplier"])
    ovx_multiplier = float(asset_vol_signal["ovx_uso_variance_multiplier"])
    asset_scaling = np.eye(len(ASSETS), dtype=float)
    asset_scaling[stage36.GOLD_INDEX, stage36.GOLD_INDEX] = math.sqrt(
        gvz_multiplier
    )
    asset_scaling[stage36.OIL_INDEX, stage36.OIL_INDEX] = math.sqrt(
        ovx_multiplier
    )
    covariance = asset_scaling @ covariance @ asset_scaling

    detail = {
        "eps_mu_adjustment_KODEX200": eps_mu,
        "valuation_mu_adjustment_KODEX200": valuation_mu,
        "credit_stress_confirmation_multiplier": credit_stress_multiplier,
        "credit_equity_variance_multiplier": credit_variance_multiplier,
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
    }
    return expected_return, covariance, detail


def _unique_starts(
    pretrade: np.ndarray, stage36_reference: np.ndarray
) -> list[np.ndarray]:
    """Deterministic numerical starts, not a strategy-parameter search."""

    raw_starts = [
        pretrade,
        stage36_reference,
        np.repeat(1.0 / len(ASSETS), len(ASSETS)),
        *np.eye(len(ASSETS), dtype=float),
    ]
    starts: list[np.ndarray] = []
    for raw in raw_starts:
        raw = np.asarray(raw, dtype=float)
        if not np.isfinite(raw).all() or raw.sum() <= 0.0:
            continue
        projected = stage36.stage35.project_to_long_only_simplex(raw)
        if not any(np.max(np.abs(projected - old)) <= 1e-12 for old in starts):
            starts.append(projected)
    return starts


def portfolio_forecast_statistics(
    weights: np.ndarray,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_asset_returns: np.ndarray,
    pretrade: np.ndarray,
) -> dict[str, float]:
    """Compute the Stage43 objective and both strict constraints."""

    weights = np.asarray(weights, dtype=float)
    monthly_return = float(weights @ expected_return)
    monthly_variance = max(float(weights @ covariance @ weights), 0.0)
    monthly_volatility = math.sqrt(monthly_variance)
    transaction_cost = stage36.stage35.expected_transaction_cost(
        weights, pretrade
    )
    expected_monthly_log_growth = (
        monthly_return - 0.5 * monthly_variance - transaction_cost
    )
    ex_ante_sharpe = (
        math.sqrt(12.0)
        * (monthly_return - transaction_cost)
        / monthly_volatility
        if monthly_volatility > 1e-12
        else (-math.inf if monthly_return - transaction_cost <= 0.0 else math.inf)
    )
    scenario_returns = np.asarray(historical_asset_returns, dtype=float) @ weights
    return {
        "expected_monthly_return": monthly_return,
        "expected_monthly_variance": monthly_variance,
        "expected_annual_volatility": monthly_volatility * math.sqrt(12.0),
        "estimated_transaction_cost": transaction_cost,
        "expected_monthly_log_growth_net": expected_monthly_log_growth,
        "expected_annual_log_growth_net": 12.0 * expected_monthly_log_growth,
        "expected_annual_geometric_growth_net": float(
            np.expm1(np.clip(12.0 * expected_monthly_log_growth, -50.0, 50.0))
        ),
        "ex_ante_sharpe": ex_ante_sharpe,
        "historical_cdar90": historical_cdar(scenario_returns),
    }


def _is_feasible(
    weights: np.ndarray,
    values: dict[str, float],
    tail_budget: float,
) -> bool:
    weights = np.asarray(weights, dtype=float)
    return bool(
        np.isfinite(weights).all()
        and abs(float(weights.sum()) - 1.0) <= 1e-6
        and weights.min() >= -CONSTRAINT_TOLERANCE
        and weights.max() <= 1.0 + CONSTRAINT_TOLERANCE
        and values["ex_ante_sharpe"]
        >= EX_ANTE_SHARPE_FLOOR - CONSTRAINT_TOLERANCE
        and values["historical_cdar90"] >= tail_budget - CONSTRAINT_TOLERANCE
    )


def solve_weights(
    historical_asset_returns: np.ndarray,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    pretrade: np.ndarray,
    stage36_reference: np.ndarray,
    current_drawdown: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize expected log growth under ex-ante Sharpe and dynamic DD risk."""

    if current_drawdown <= MDD_FLOOR - CONSTRAINT_TOLERANCE:
        raise DrawdownFloorAlreadyBreached(
            f"Current drawdown {current_drawdown:.6%} is already below -12%."
        )
    tail_budget = remaining_loss_budget(current_drawdown)

    def values(weights: np.ndarray) -> dict[str, float]:
        return portfolio_forecast_statistics(
            weights,
            expected_return,
            covariance,
            historical_asset_returns,
            pretrade,
        )

    def objective(weights: np.ndarray) -> float:
        growth = values(weights)["expected_monthly_log_growth_net"]
        return -growth if np.isfinite(growth) else 1e12

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                values(weights)["ex_ante_sharpe"] - EX_ANTE_SHARPE_FLOOR
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                values(weights)["historical_cdar90"] - tail_budget
            ),
        },
    ]

    attempts: list[dict[str, Any]] = []
    feasible: list[tuple[float, np.ndarray, Any, dict[str, float]]] = []
    for start_number, start in enumerate(
        _unique_starts(pretrade, stage36_reference), start=1
    ):
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": stage36.stage35.SLSQP_MAX_ITERATIONS,
                "ftol": stage36.stage35.SLSQP_TOLERANCE,
            },
        )
        candidate = stage36.stage35.project_to_long_only_simplex(result.x)
        statistics = values(candidate)
        candidate_feasible = bool(
            result.success
            and _is_feasible(candidate, statistics, tail_budget)
        )
        attempts.append(
            {
                "start": start_number,
                "success": bool(result.success),
                "feasible": candidate_feasible,
                "message": str(result.message),
                "iterations": int(result.nit),
                "ex_ante_sharpe": statistics["ex_ante_sharpe"],
                "historical_cdar90": statistics["historical_cdar90"],
            }
        )
        if candidate_feasible:
            feasible.append(
                (-float(result.fun), candidate, result, statistics)
            )

    if not feasible:
        best_sharpe = max(item["ex_ante_sharpe"] for item in attempts)
        best_cdar = max(item["historical_cdar90"] for item in attempts)
        raise InfeasiblePortfolioError(
            "No feasible portfolio without relaxing Stage43 constraints: "
            f"drawdown={current_drawdown:.6%}, tail_budget={tail_budget:.6%}, "
            f"best attempted Sharpe={best_sharpe:.6f}, "
            f"best attempted CDaR={best_cdar:.6%}."
        )

    _, weights, result, statistics = max(feasible, key=lambda item: item[0])
    detail: dict[str, Any] = {
        **statistics,
        "policy": STRATEGY_NAME,
        "objective": "maximize_expected_geometric_growth_net",
        "current_drawdown_before_trade": float(current_drawdown),
        "remaining_loss_budget": tail_budget,
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "solver_start_count": len(attempts),
        "feasible_solution_count": len(feasible),
        "infeasible_portfolio": False,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "ex_ante_sharpe_slack": statistics["ex_ante_sharpe"]
        - EX_ANTE_SHARPE_FLOOR,
        "dynamic_tail_budget_slack": statistics["historical_cdar90"]
        - tail_budget,
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    stage36_reference_path: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the strict strategy; stop rather than relax after infeasibility."""

    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
        "credit_stress_rank",
    ]
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(asset_vol_signals.index)
    months = months.intersection(stage36_reference_path.index)
    months = months.intersection(
        fundamental_signals.dropna(subset=required_fundamental).index
    )
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]

    rows: list[dict[str, Any]] = []
    infeasible_events: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < stage36.stage35.ONE_CALENDAR_YEAR:
            continue
        current_drawdown = nav / peak - 1.0
        probability = probabilities.loc[month]
        stress = float(stress_signals.loc[month, "stress_score"])
        recovery = float(stress_signals.loc[month, "recovery_score"])
        technical_signal = technical_signals.loc[month]
        fundamental_signal = fundamental_signals.loc[month]
        asset_vol_signal = asset_vol_signals.loc[month]
        expected_return, covariance, forecast_detail = build_stage36_forecast(
            history,
            probabilities.loc[probabilities.index < month],
            probability,
            stress_signals.loc[
                stress_signals.index < month, "stress_score"
            ],
            stress,
            stress_signals.loc[
                stress_signals.index < month, "recovery_score"
            ],
            recovery,
            technical_signal,
            fundamental_signal,
            asset_vol_signal,
        )
        common = history.index.intersection(
            stress_signals.loc[
                stress_signals.index < month, "stress_score"
            ].dropna().index
        )
        historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
        stage36_reference = stage36_reference_path.loc[
            month, WEIGHT_COLUMNS
        ].to_numpy(dtype=float)
        try:
            weights, detail = solve_weights(
                historical_returns,
                expected_return,
                covariance,
                pretrade,
                stage36_reference,
                current_drawdown,
            )
        except (InfeasiblePortfolioError, DrawdownFloorAlreadyBreached) as error:
            infeasible_events.append(
                {
                    "month": str(month),
                    "current_drawdown": current_drawdown,
                    "remaining_loss_budget": remaining_loss_budget(
                        current_drawdown
                    ),
                    "reason": type(error).__name__,
                    "message": str(error),
                }
            )
            break

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = (
            float(np.abs(change).sum()) * stage36.stage35.DOMESTIC_TRADE_COST
        )
        foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign_indices].sum()))
            * stage36.stage35.FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        realized_drawdown = nav / peak - 1.0
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

        row: dict[str, Any] = {
            "month": month,
            "signal_cutoff_month": month - 1,
            "history_end_month": history.index.max(),
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": realized_drawdown,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            **{
                f"w_{asset}": float(weights[index])
                for index, asset in enumerate(ASSETS)
            },
            **forecast_detail,
            **detail,
        }
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=["month"]).set_index("month")
        output.index = pd.PeriodIndex(output.index, freq="M")
    else:
        output = output.set_index("month")
        output.index = pd.PeriodIndex(output.index, freq="M")
    return output, infeasible_events


def _performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "full_2007_2026": (FULL_START, RESEARCH_END),
        "common_2010_2026": (COMMON_START, RESEARCH_END),
        "locked_2018_2026": (LOCKED_START, RESEARCH_END),
    }
    for name, path in paths.items():
        for period_name, (start, end) in periods.items():
            view = path.loc[start:end]
            complete = bool(
                not view.empty
                and view.index.min() == start
                and view.index.max() == end
                and len(view) == len(pd.period_range(start, end, freq="M"))
            )
            if complete:
                row = stage36.stage35.metric_row(
                    name, path, period_name, start, end
                )
            else:
                row = {
                    "Strategy": name,
                    "Period": period_name,
                    "Start": str(view.index.min()) if not view.empty else None,
                    "End": str(view.index.max()) if not view.empty else None,
                    "Months": int(len(view)),
                    **{
                        key: np.nan
                        for key in (
                            "CAGR",
                            "Volatility",
                            "Sharpe",
                            "Sortino",
                            "MDD",
                            "Calmar",
                            "FinalMultiple",
                            "PositiveMonths",
                            "AvgTurnover",
                            "TotalCost",
                        )
                    },
                }
            row["CompletePeriod"] = complete
            rows.append(row)
    return pd.DataFrame(rows)


def _failure_window_performance(
    stage36_path: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    """Compare the candidate only over its genuinely executable path.

    These figures are diagnostic and are never presented as 2007--2026
    performance.  They make an early strict-strategy failure measurable without
    filling later months with a relaxed rule or a hindsight fallback.
    """

    if candidate.empty:
        return pd.DataFrame()
    start = candidate.index.min()
    end = candidate.index.max()
    rows = [
        stage36.stage35.metric_row(
            STAGE36_NAME, stage36_path, "executable_failure_window", start, end
        ),
        stage36.stage35.metric_row(
            STRATEGY_NAME, candidate, "executable_failure_window", start, end
        ),
    ]
    return pd.DataFrame(rows)


def run_research(save: bool = True) -> dict[str, Any]:
    stage36_before = frozen_stage36_manifest()

    daily, data_audit = stage36.load_asset_implied_volatility_daily()
    returns, return_audit = stage36.stage35.load_monthly_asset_returns(False)
    probabilities, macro_audit = stage36.stage35.build_macro_probabilities(returns)
    stress = stage36.stage35.build_monthly_stress_signals(
        returns.index, stage36.stage35.build_daily_stress_features()
    )
    market, market_audit = stage36.stage35.stage20.load_daily_asset_ohlcv()
    raw_fundamental, _ = stage36.stage35.load_fundamental_daily()
    fundamental = stage36.stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
    equity_close = market["KODEX200"]["close"].dropna()
    equity_monthly_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    calibrated = stage36.stage35.add_causal_return_calibration(
        fundamental, equity_monthly_close.pct_change()
    )
    technical = stage36.stage35.stage34._load_period_csv(
        stage36.stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    asset_vol_signals = stage36.build_monthly_asset_volatility_signals(
        daily, returns.index
    )
    stage36_path = stage36.stage35.stage34._load_period_csv(
        stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
        "month",
    )

    candidate, infeasible_events = run_backtest(
        returns,
        probabilities,
        stress,
        technical,
        calibrated,
        asset_vol_signals,
        stage36_path,
    )
    paths = {STAGE36_NAME: stage36_path, STRATEGY_NAME: candidate}
    performance = _performance_table(paths)
    failure_window_performance = _failure_window_performance(
        stage36_path, candidate
    )
    full = performance.loc[performance["Period"].eq("full_2007_2026")].set_index(
        "Strategy"
    )

    candidate_complete = bool(full.loc[STRATEGY_NAME, "CompletePeriod"])
    validation = {
        name: {
            "complete_full_period": bool(row["CompletePeriod"]),
            "CAGR": float(row["CAGR"]) if pd.notna(row["CAGR"]) else None,
            "Sharpe": float(row["Sharpe"]) if pd.notna(row["Sharpe"]) else None,
            "MDD": float(row["MDD"]) if pd.notna(row["MDD"]) else None,
            "mdd_at_least_minus_12pct": bool(
                pd.notna(row["MDD"]) and float(row["MDD"]) >= MDD_FLOOR
            ),
            "sharpe_at_least_1_1": bool(
                pd.notna(row["Sharpe"])
                and float(row["Sharpe"]) >= EX_ANTE_SHARPE_FLOOR
            ),
            "full_validation_pass": bool(
                row["CompletePeriod"]
                and pd.notna(row["MDD"])
                and float(row["MDD"]) >= MDD_FLOOR
                and pd.notna(row["Sharpe"])
                and float(row["Sharpe"]) >= EX_ANTE_SHARPE_FLOOR
            ),
        }
        for name, row in full.iterrows()
    }
    eligible = [
        name for name, values in validation.items() if values["full_validation_pass"]
    ]
    selected = (
        max(eligible, key=lambda name: validation[name]["CAGR"])
        if eligible
        else None
    )
    stage36_after = frozen_stage36_manifest()

    monthly_checks: dict[str, Any]
    if candidate.empty:
        monthly_checks = {
            "months": 0,
            "all_solver_success": None,
            "minimum_ex_ante_sharpe_slack": None,
            "minimum_dynamic_tail_budget_slack": None,
            "all_history_ends_before_target": None,
            "max_weight_sum_error": None,
        }
    else:
        monthly_checks = {
            "months": int(len(candidate)),
            "all_solver_success": bool(candidate["solver_success"].all()),
            "minimum_ex_ante_sharpe_slack": float(
                candidate["ex_ante_sharpe_slack"].min()
            ),
            "minimum_dynamic_tail_budget_slack": float(
                candidate["dynamic_tail_budget_slack"].min()
            ),
            "all_history_ends_before_target": bool(
                (candidate["history_end_month"] < candidate.index).all()
            ),
            "max_weight_sum_error": float(candidate["sum_error"].max()),
            "minimum_weight": float(candidate[WEIGHT_COLUMNS].min().min()),
            "maximum_weight": float(candidate[WEIGHT_COLUMNS].max().max()),
            "minimum_realized_drawdown": float(candidate["drawdown"].min()),
        }

    report = {
        "stage": 43,
        "base": "Stage36_GVZ_OVXAssetRisk",
        "strategy": STRATEGY_NAME,
        "objective": (
            "maximize w'mu - 0.5*w'Sigma*w - current transaction cost"
        ),
        "constraints": {
            "ex_ante_sharpe": ">= 1.10",
            "dynamic_tail_risk": (
                "historical CDaR(90%) >= 0.88/(1+current realized DD)-1"
            ),
            "weights": "sum=1 and each in [0,1]",
            "final_validation": "realized MDD >= -12% and Sharpe >= 1.10",
        },
        "preserved": {
            "stage36_mu_and_sigma_information_set": True,
            "stage36_transaction_cost_and_execution": True,
            "stage36_cdar_confidence": TAIL_CONFIDENCE,
            "no_leverage_long_only_sum_one": True,
            "single_asset_majority_cap": False,
        },
        "anti_overfit": {
            "candidate_count": 1,
            "threshold_search": False,
            "lookback_search": False,
            "objective_weight_search": False,
            "constraint_relaxation": False,
            "future_return_in_optimizer": False,
        },
        "candidate_complete_full_period": candidate_complete,
        "infeasible_events": infeasible_events,
        "monthly_checks": monthly_checks,
        "validation_gates": validation,
        "eligible_strategies": eligible,
        "selected_highest_cagr_strategy": selected,
        "stage36_frozen_files_unchanged": stage36_before == stage36_after,
        "performance": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "partial_failure_window_performance": json.loads(
            failure_window_performance.to_json(
                orient="records", force_ascii=False
            )
        ),
        "data_audit": data_audit,
        "return_audit": {
            "rows": int(len(return_audit)),
            "first_month": str(return_audit.index.min()),
            "last_month": str(return_audit.index.max()),
        },
        "macro_audit": {
            "rows": int(len(macro_audit)),
            "first_date": str(macro_audit.index.min()),
            "last_date": str(macro_audit.index.max()),
        },
        "market_audit": market_audit,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(OUTPUT_DIR / "stage43_dynamic_dd12_budget_monthly.csv")
        performance.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        failure_window_performance.to_csv(
            OUTPUT_DIR / "partial_failure_window_performance.csv", index=False
        )
        pd.DataFrame(infeasible_events).to_csv(
            OUTPUT_DIR / "infeasible_events.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
