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
RESEARCH_END = stage36.RESEARCH_END
LOCKED_START = stage36.stage35.LOCKED_START

MDD_FLOOR = -0.14
SHARPE_FLOOR = 1.0
CDAR_FLOOR = -stage36.stage35.CATASTROPHE_CDAR
CONSTRAINT_TOLERANCE = 1e-7

STRATEGIES = {
    "Stage41_CAGR_HardMDD14_Sharpe1": "hard_mdd",
    "Stage41_CAGR_CDaR16_Sharpe1": "cdar",
}

FROZEN_STAGE36_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
)


class InfeasiblePortfolioError(RuntimeError):
    """Raised when no long-only portfolio satisfies the declared guardrails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_stage36_manifest() -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in FROZEN_STAGE36_FILES}


def historical_max_drawdown(monthly_returns: np.ndarray) -> float:
    returns = np.asarray(monthly_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[-len(wealth) :]
    return float(np.min(wealth / peaks - 1.0))


def historical_statistics(
    weights: np.ndarray, historical_asset_returns: np.ndarray
) -> dict[str, float]:
    """Compute the objective and guardrails from strictly prior monthly returns."""

    portfolio_returns = np.asarray(historical_asset_returns, dtype=float) @ np.asarray(
        weights, dtype=float
    )
    if portfolio_returns.size == 0 or np.any(portfolio_returns <= -1.0):
        return {
            "historical_cagr": -1.0,
            "historical_mdd": -1.0,
            "historical_sharpe": -math.inf,
            "historical_cdar": -1.0,
        }
    annual_log_growth = 12.0 * float(np.mean(np.log1p(portfolio_returns)))
    cagr = float(np.expm1(np.clip(annual_log_growth, -50.0, 50.0)))
    volatility = float(np.std(portfolio_returns, ddof=1))
    sharpe = (
        math.sqrt(12.0) * float(np.mean(portfolio_returns)) / volatility
        if volatility > 1e-12
        else (-math.inf if float(np.mean(portfolio_returns)) <= 0.0 else math.inf)
    )
    return {
        "historical_cagr": cagr,
        "historical_mdd": historical_max_drawdown(portfolio_returns),
        "historical_sharpe": sharpe,
        "historical_cdar": stage36.stage35.cdar(
            portfolio_returns, stage36.stage35.CDAR_CONFIDENCE
        ),
    }


def _is_feasible(weights: np.ndarray, values: dict[str, float], mode: str) -> bool:
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or abs(float(weights.sum()) - 1.0) > 1e-6:
        return False
    if weights.min() < -CONSTRAINT_TOLERANCE or weights.max() > 1.0 + CONSTRAINT_TOLERANCE:
        return False
    if values["historical_sharpe"] < SHARPE_FLOOR - CONSTRAINT_TOLERANCE:
        return False
    if mode == "hard_mdd":
        return values["historical_mdd"] >= MDD_FLOOR - CONSTRAINT_TOLERANCE
    if mode == "cdar":
        return values["historical_cdar"] >= CDAR_FLOOR - CONSTRAINT_TOLERANCE
    raise ValueError(f"Unknown guardrail mode: {mode}")


def _unique_starts(
    pretrade: np.ndarray, stage36_reference: np.ndarray
) -> list[np.ndarray]:
    """Deterministic starts; these are not a hyperparameter grid."""

    stage35 = stage36.stage35
    starts: list[np.ndarray] = []
    raw_starts = [
        pretrade,
        stage36_reference,
        np.repeat(1.0 / len(ASSETS), len(ASSETS)),
        *np.eye(len(ASSETS), dtype=float),
    ]
    for raw in raw_starts:
        raw = np.asarray(raw, dtype=float)
        if not np.isfinite(raw).all() or raw.sum() <= 0.0:
            continue
        projected = stage35.project_to_long_only_simplex(raw)
        if not any(np.max(np.abs(projected - existing)) <= 1e-12 for existing in starts):
            starts.append(projected)
    return starts


def solve_cagr_guardrail_weights(
    history: pd.DataFrame,
    pretrade: np.ndarray,
    stage36_reference: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize causal historical CAGR under the requested absolute guardrails."""

    if mode not in {"hard_mdd", "cdar"}:
        raise ValueError(f"Unknown guardrail mode: {mode}")
    historical_returns = history[ASSETS].to_numpy(dtype=float)

    def values(weights: np.ndarray) -> dict[str, float]:
        return historical_statistics(weights, historical_returns)

    def objective(weights: np.ndarray) -> float:
        cagr = values(weights)["historical_cagr"]
        return -cagr if np.isfinite(cagr) else 1e12

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: values(weights)["historical_sharpe"]
            - SHARPE_FLOOR,
        },
    ]
    if mode == "hard_mdd":
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: values(weights)["historical_mdd"] - MDD_FLOOR,
            }
        )
    else:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: values(weights)["historical_cdar"]
                - CDAR_FLOOR,
            }
        )

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
            result.success and _is_feasible(candidate, statistics, mode)
        )
        attempts.append(
            {
                "start": start_number,
                "success": bool(result.success),
                "feasible": candidate_feasible,
                "message": str(result.message),
                "iterations": int(result.nit),
                "historical_cagr": statistics["historical_cagr"],
            }
        )
        if candidate_feasible:
            feasible.append(
                (-float(result.fun), candidate, result, statistics)
            )

    if not feasible:
        messages = "; ".join(
            f"start {item['start']}: {item['message']}" for item in attempts
        )
        raise InfeasiblePortfolioError(
            f"No feasible {mode} portfolio with {len(history)} prior months. {messages}"
        )

    _, weights, result, statistics = max(feasible, key=lambda item: item[0])
    estimated_cost = stage36.stage35.expected_transaction_cost(weights, pretrade)
    detail: dict[str, Any] = {
        **statistics,
        "policy": f"Stage41_{mode}",
        "guardrail_mode": mode,
        "objective": "maximize_causal_historical_cagr",
        "objective_value": float(result.fun),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "solver_start_count": len(attempts),
        "feasible_solution_count": len(feasible),
        "infeasible_portfolio": False,
        "estimated_transaction_cost": estimated_cost,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "sharpe_slack": statistics["historical_sharpe"] - SHARPE_FLOOR,
        "mdd_slack": statistics["historical_mdd"] - MDD_FLOOR,
        "cdar_slack": statistics["historical_cdar"] - CDAR_FLOOR,
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    stage36_reference_path: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """Run Stage36's monthly execution and costs with the Stage41 optimizer."""

    months = returns.index.intersection(stage36_reference_path.index)
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < stage36.stage35.ONE_CALENDAR_YEAR:
            continue
        reference = stage36_reference_path.loc[month, WEIGHT_COLUMNS].to_numpy(
            dtype=float
        )
        weights, detail = solve_cagr_guardrail_weights(
            history, pretrade, reference, mode
        )
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
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
                "signal_cutoff_month": month - 1,
                "history_end_month": history.index.max(),
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
                **detail,
            }
        )
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
    for period_name, start in (
        ("full_2007_2026", FULL_START),
        ("common_2010_2026", COMMON_START),
    ):
        for name, candidate in candidates.items():
            summary = stage36.stage35.stage30.paired_block_bootstrap(
                baseline.loc[start:RESEARCH_END, "return"],
                candidate.loc[start:RESEARCH_END, "return"],
            )
            summary.insert(0, "Period", period_name)
            summary.insert(0, "Candidate", name)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _path_checks(path: pd.DataFrame, mode: str) -> dict[str, Any]:
    weights = path[WEIGHT_COLUMNS].to_numpy(dtype=float)
    selected_guardrail_slack = "mdd_slack" if mode == "hard_mdd" else "cdar_slack"
    return {
        "months": int(len(path)),
        "all_solver_success": bool(path["solver_success"].all()),
        "infeasible_months": int(path["infeasible_portfolio"].sum()),
        "max_sum_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "minimum_historical_sharpe_slack": float(path["sharpe_slack"].min()),
        "minimum_selected_guardrail_slack": float(
            path[selected_guardrail_slack].min()
        ),
        "all_history_ends_before_target": bool(
            (pd.PeriodIndex(path["history_end_month"], freq="M") < path.index).all()
        ),
    }


def run_research(save: bool = True) -> dict[str, Any]:
    before = frozen_stage36_manifest()
    returns, return_audit = stage36.stage35.load_monthly_asset_returns(False)
    baseline = stage36.stage35.stage34._load_period_csv(
        stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv", "month"
    )
    candidates = {
        name: run_backtest(returns, baseline, mode)
        for name, mode in STRATEGIES.items()
    }
    paths = {"Stage36_Frozen": baseline, **candidates}
    performance = performance_table(paths)
    bootstrap = bootstrap_table(baseline, candidates)
    checks = {
        name: _path_checks(path, STRATEGIES[name])
        for name, path in candidates.items()
    }

    indexed = performance.set_index(["Strategy", "Period"])
    gates: dict[str, Any] = {}
    for name in paths:
        full = indexed.loc[(name, "full_2007_2026")]
        period_metrics = {
            period: {
                "CAGR": float(indexed.loc[(name, period), "CAGR"]),
                "Sharpe": float(indexed.loc[(name, period), "Sharpe"]),
                "MDD": float(indexed.loc[(name, period), "MDD"]),
            }
            for period in (
                "full_2007_2026",
                "common_2010_2026",
                "locked_2018_2026",
            )
        }
        gates[name] = {
            "period_metrics": period_metrics,
            "full_mdd_at_least_minus_14pct": bool(full["MDD"] >= MDD_FLOOR),
            "full_sharpe_at_least_one": bool(full["Sharpe"] >= SHARPE_FLOOR),
            "full_validation_pass": bool(
                full["MDD"] >= MDD_FLOOR and full["Sharpe"] >= SHARPE_FLOOR
            ),
        }
        if name in checks:
            gates[name]["all_monthly_optimization_constraints_pass"] = bool(
                checks[name]["all_solver_success"]
                and checks[name]["infeasible_months"] == 0
                and checks[name]["minimum_historical_sharpe_slack"]
                >= -CONSTRAINT_TOLERANCE
                and checks[name]["minimum_selected_guardrail_slack"]
                >= -CONSTRAINT_TOLERANCE
            )
        else:
            gates[name]["all_monthly_optimization_constraints_pass"] = None

    eligible = [
        name
        for name, gate in gates.items()
        if gate["full_validation_pass"]
        and (
            gate["all_monthly_optimization_constraints_pass"] is not False
        )
    ]
    selected = max(
        eligible,
        key=lambda name: float(indexed.loc[(name, "full_2007_2026"), "CAGR"]),
    ) if eligible else None

    after = frozen_stage36_manifest()
    report = {
        "stage": 41,
        "base": "Stage36_GVZ_OVXAssetRisk",
        "objective": "maximize causal historical CAGR",
        "candidate_definitions": {
            "A": "maximize CAGR subject to historical MDD >= -14% and Sharpe >= 1",
            "B": "maximize CAGR subject to historical CDaR(90%) >= -16% and Sharpe >= 1; realized MDD >= -14% is a validation gate",
        },
        "preserved": {
            "monthly_asset_returns": "Stage36 input loader",
            "monthly_execution_and_costs": "Stage36 definitions",
            "long_only_unlevered_sum_one": True,
            "single_asset_majority_cap": False,
        },
        "deliberately_not_used": {
            "future_realized_returns_in_optimization": True,
            "calmar_or_sortino_ratio_objective": True,
            "objective_weighting": True,
            "guardrail_relaxation_when_infeasible": True,
            "stage36_13pct_volatility_constraint": True,
        },
        "important_scope_note": (
            "The supplied feedback defines CAGR, MDD and Sharpe from strictly prior "
            "historical asset returns. Therefore Stage36 macro/technical/GVZ/OVX "
            "forecasts do not enter the new objective; Stage36 is preserved as the "
            "data, execution, cost and frozen comparison framework."
        ),
        "thresholds": {
            "mdd": MDD_FLOOR,
            "sharpe": SHARPE_FLOOR,
            "cdar": CDAR_FLOOR,
            "cdar_confidence": stage36.stage35.CDAR_CONFIDENCE,
        },
        "anti_overfit": {
            "candidate_count": len(STRATEGIES),
            "threshold_grid_search": False,
            "lookback_grid_search": False,
            "infeasible_guardrail_relaxation": False,
            "realized_validation_metrics_used_in_optimizer": False,
        },
        "stage36_frozen_files_unchanged": before == after,
        "return_audit": return_audit,
        "checks": checks,
        "validation_gates": gates,
        "eligible_strategies": eligible,
        "selected_highest_cagr_strategy": selected,
        "performance": json.loads(performance.to_json(orient="records")),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        candidates["Stage41_CAGR_HardMDD14_Sharpe1"].to_csv(
            OUTPUT_DIR / "stage41_hard_mdd_monthly.csv"
        )
        candidates["Stage41_CAGR_CDaR16_Sharpe1"].to_csv(
            OUTPUT_DIR / "stage41_cdar_monthly.csv"
        )
        performance.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return {
        "paths": paths,
        "performance": performance,
        "bootstrap": bootstrap,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print("=== STAGE41 MAXIMUM CAGR WITH ABSOLUTE GUARDRAILS ===")
    print(
        result["performance"][
            [
                "Strategy",
                "Period",
                "CAGR",
                "Volatility",
                "Sharpe",
                "Sortino",
                "MDD",
                "Calmar",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\n=== VALIDATION GATES ===")
    print(
        json.dumps(
            result["report"]["validation_gates"], ensure_ascii=False, indent=2
        )
    )
    print("selected", result["report"]["selected_highest_cagr_strategy"])


if __name__ == "__main__":
    main()
