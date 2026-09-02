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
STAGE36_MODE = "gvz_ovx_asset_risk"
SHARPE_FLOOR = 1.0
RATIO_EPSILON = 1e-10

OBJECTIVE_MODES = {
    "Stage40_CausalCalmar": "calmar",
    "Stage40_CausalSortino": "sortino",
}

FROZEN_STAGE36_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_stage36_manifest() -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in FROZEN_STAGE36_FILES}


def historical_max_drawdown(monthly_returns: np.ndarray) -> float:
    """Maximum drawdown of a causal historical fixed-weight return path."""

    returns = np.asarray(monthly_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[-len(wealth) :]
    return float(np.min(wealth / peaks - 1.0))


def ratio_statistics(
    weights: np.ndarray,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_returns: np.ndarray,
    estimated_transaction_cost: float,
) -> dict[str, float]:
    """Causal ex-ante numerators and historical downside denominators.

    The expected return and covariance are the Stage36 estimates available at
    the decision month.  Historical downside and drawdown use only asset
    returns strictly before that month.  No realized future return enters an
    optimizer objective.
    """

    weights = np.asarray(weights, dtype=float)
    monthly_return = float(weights @ expected_return)
    monthly_variance = max(float(weights @ covariance @ weights), 0.0)
    net_monthly_return = monthly_return - float(estimated_transaction_cost)
    annual_expected_return = 12.0 * net_monthly_return
    annual_log_growth = 12.0 * (net_monthly_return - 0.5 * monthly_variance)
    expected_cagr = float(np.expm1(np.clip(annual_log_growth, -50.0, 50.0)))

    scenario_returns = np.asarray(historical_returns, dtype=float) @ weights
    downside_deviation = float(
        np.sqrt(np.mean(np.minimum(scenario_returns, 0.0) ** 2))
        * math.sqrt(12.0)
    )
    historical_mdd = historical_max_drawdown(scenario_returns)
    annual_volatility = math.sqrt(monthly_variance * 12.0)

    sortino = annual_expected_return / max(downside_deviation, RATIO_EPSILON)
    calmar = expected_cagr / max(abs(historical_mdd), RATIO_EPSILON)
    expected_sharpe = annual_expected_return / max(
        annual_volatility, RATIO_EPSILON
    )
    return {
        "expected_monthly_return": monthly_return,
        "net_expected_monthly_return": net_monthly_return,
        "expected_monthly_variance": monthly_variance,
        "expected_annual_return": annual_expected_return,
        "expected_annual_log_growth": annual_log_growth,
        "expected_cagr": expected_cagr,
        "historical_downside_deviation": downside_deviation,
        "historical_max_drawdown": historical_mdd,
        "causal_sortino_objective": sortino,
        "causal_calmar_objective": calmar,
        "expected_sharpe": expected_sharpe,
        "expected_annual_volatility": annual_volatility,
        "estimated_transaction_cost": float(estimated_transaction_cost),
    }


def _optimizer_inputs(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Reproduce Stage36 mu and Sigma, including the combined GVZ/OVX overlay."""

    stage35 = stage36.stage35
    _, base_covariance, moment_detail = stage35.estimate_conditional_moments(
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
    technical = stage35.stage20.apply_technical_inputs(
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
    stress_adjustment[stage35.EQUITY_INDEX] *= credit_stress_multiplier
    filtered_macro[stage35.EQUITY_INDEX] += eps_mu + valuation_mu
    expected_return = filtered_macro + stress_adjustment

    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)
    credit_scaling = np.eye(len(ASSETS), dtype=float)
    credit_scaling[stage35.EQUITY_INDEX, stage35.EQUITY_INDEX] = math.sqrt(
        credit_variance_multiplier
    )
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

    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    detail = {
        "eps_mu_adjustment_KODEX200": eps_mu,
        "valuation_mu_adjustment_KODEX200": valuation_mu,
        "credit_stress_confirmation_multiplier": credit_stress_multiplier,
        "credit_equity_variance_multiplier": credit_variance_multiplier,
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
        "gvz_mu_adjustment_GLD": 0.0,
        "ovx_mu_adjustment_USO": 0.0,
        "expected_mu_GLD": float(expected_return[stage36.GOLD_INDEX]),
        "expected_mu_USO": float(expected_return[stage36.OIL_INDEX]),
    }
    return expected_return, covariance, historical_returns, detail


def solve_ratio_weights(
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
    pretrade: np.ndarray,
    objective_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize causal Calmar or Sortino under the frozen Stage36 constraints."""

    if objective_mode not in {"calmar", "sortino"}:
        raise ValueError(f"Unknown objective mode: {objective_mode}")
    stage35 = stage36.stage35
    expected_return, covariance, historical_returns, input_detail = _optimizer_inputs(
        history,
        historical_probabilities,
        current_probabilities,
        historical_stress,
        current_stress,
        historical_recovery,
        current_recovery,
        technical_signal,
        fundamental_signal,
        asset_vol_signal,
    )
    initial = (
        stage35.project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def values(weights: np.ndarray) -> dict[str, float]:
        transaction_cost = stage35.expected_transaction_cost(weights, pretrade)
        return ratio_statistics(
            weights,
            expected_return,
            covariance,
            historical_returns,
            transaction_cost,
        )

    objective_key = f"causal_{objective_mode}_objective"

    def objective(weights: np.ndarray) -> float:
        ratio = values(weights)[objective_key]
        return -float(ratio) if np.isfinite(ratio) else 1e12

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(
            max(float(weights @ covariance @ weights), 0.0) * 12.0
        )

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage35.CATASTROPHE_ANNUAL_VOLATILITY
                - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage35.CATASTROPHE_CDAR
                + stage35.cdar(
                    historical_returns @ weights, stage35.CDAR_CONFIDENCE
                )
            ),
        },
    ]
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={
            "maxiter": stage35.SLSQP_MAX_ITERATIONS,
            "ftol": stage35.SLSQP_TOLERANCE,
        },
    )
    primary_solver_message = str(result.message)
    used_restart = False
    restart_attempts = 0
    if not (result.success and np.isfinite(result.x).all()):
        inverse_variance = 1.0 / np.maximum(np.diag(covariance), 1e-12)
        inverse_variance /= inverse_variance.sum()
        restart_points = (
            np.repeat(1.0 / len(ASSETS), len(ASSETS)),
            inverse_variance,
        )
        for restart in restart_points:
            restart_attempts += 1
            restarted = minimize(
                objective,
                restart,
                method="SLSQP",
                bounds=stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
                constraints=constraints,
                options={
                    "maxiter": stage35.SLSQP_MAX_ITERATIONS,
                    "ftol": stage35.SLSQP_TOLERANCE,
                },
            )
            if restarted.success and np.isfinite(restarted.x).all():
                result = restarted
                used_restart = True
                break
    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = stage35.project_to_long_only_simplex(result.x)
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": stage35.SLSQP_MAX_ITERATIONS,
                "ftol": stage35.SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                f"Stage40 and fallback solves failed: {result.message}; "
                f"{fallback.message}"
            )
        result = fallback
        weights = stage35.project_to_long_only_simplex(fallback.x)
        used_fallback = True

    statistics = values(weights)
    historical_cdar = stage35.cdar(
        historical_returns @ weights, stage35.CDAR_CONFIDENCE
    )
    detail: dict[str, Any] = {
        **statistics,
        **input_detail,
        "policy": f"Stage40_{objective_mode}",
        "objective_mode": objective_mode,
        "objective_value": float(result.fun),
        "optimized_ratio": float(statistics[objective_key]),
        "solver_success": bool(result.success),
        "primary_solver_message": primary_solver_message,
        "used_objective_restart": used_restart,
        "objective_restart_attempts": restart_attempts,
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": (
            stage35.CATASTROPHE_ANNUAL_VOLATILITY
            - statistics["expected_annual_volatility"]
        ),
        "cdar_slack": stage35.CATASTROPHE_CDAR + historical_cdar,
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    objective_mode: str,
) -> pd.DataFrame:
    """Stage36 walk-forward backtest with only the optimizer objective changed."""

    stage35 = stage36.stage35
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
    months = months.intersection(
        fundamental_signals.dropna(subset=required_fundamental).index
    )
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]

    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < stage35.ONE_CALENDAR_YEAR:
            continue
        probability = probabilities.loc[month]
        stress = float(stress_signals.loc[month, "stress_score"])
        recovery = float(stress_signals.loc[month, "recovery_score"])
        technical_signal = technical_signals.loc[month]
        fundamental_signal = fundamental_signals.loc[month]
        asset_vol_signal = asset_vol_signals.loc[month]
        weights, detail = solve_ratio_weights(
            history,
            probabilities.loc[probabilities.index < month],
            probability,
            stress_signals.loc[stress_signals.index < month, "stress_score"],
            stress,
            stress_signals.loc[stress_signals.index < month, "recovery_score"],
            recovery,
            technical_signal,
            fundamental_signal,
            asset_vol_signal,
            pretrade,
            objective_mode,
        )
        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = (
            float(np.abs(change).sum()) * stage35.DOMESTIC_TRADE_COST
        )
        foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign_indices].sum()))
            * stage35.FOREIGN_WEIGHT_CHANGE_COST
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
                "asset_vol_signal_month": asset_vol_signal[
                    "asset_vol_signal_month"
                ],
                "gvz_signal_date": asset_vol_signal["gvz_signal_date"],
                "ovx_signal_date": asset_vol_signal["ovx_signal_date"],
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


def load_stage36_inputs() -> dict[str, Any]:
    """Build exactly the causal input panels used by Stage36."""

    stage35 = stage36.stage35
    daily, data_audit = stage36.load_asset_implied_volatility_daily()
    returns, return_audit = stage35.load_monthly_asset_returns(False)
    probabilities, macro_audit = stage35.build_macro_probabilities(returns)
    stress = stage35.build_monthly_stress_signals(
        returns.index, stage35.build_daily_stress_features()
    )
    market, market_audit = stage35.stage20.load_daily_asset_ohlcv()
    raw_fundamental, _ = stage35.load_fundamental_daily()
    fundamental = stage35.build_monthly_fundamental_signals(raw_fundamental)
    equity_close = market["KODEX200"]["close"].dropna()
    equity_monthly_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    calibrated = stage35.add_causal_return_calibration(
        fundamental, equity_monthly_close.pct_change()
    )
    technical = stage35.stage34._load_period_csv(
        stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    asset_vol_signals = stage36.build_monthly_asset_volatility_signals(
        daily, returns.index
    )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "stress": stress,
        "technical": technical,
        "fundamental": calibrated,
        "asset_vol": asset_vol_signals,
        "audits": {
            "asset_vol": data_audit,
            "returns": return_audit,
            "macro": macro_audit,
            "market": market_audit,
        },
    }


def performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_end = min(path.index.max() for path in paths.values())
    periods = {
        "full_2007_2026": (FULL_START, common_end),
        "common_2010_2026": (COMMON_START, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    rows = [
        stage36.stage35.metric_row(name, path, period, start, end)
        for name, path in paths.items()
        for period, (start, end) in periods.items()
    ]
    return pd.DataFrame(rows)


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


def _strategy_checks(path: pd.DataFrame) -> dict[str, Any]:
    weights = path[WEIGHT_COLUMNS].to_numpy(dtype=float)
    return {
        "months": int(len(path)),
        "all_solver_success": bool(path["solver_success"].all()),
        "objective_restart_months": int(path["used_objective_restart"].sum()),
        "fallback_months": int(path["used_fallback"].sum()),
        "max_sum_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "minimum_volatility_slack": float(path["volatility_slack"].min()),
        "minimum_cdar_slack": float(path["cdar_slack"].min()),
        "all_signal_months_precede_target": bool(
            (
                pd.PeriodIndex(path["asset_vol_signal_month"], freq="M")
                < path.index
            ).all()
        ),
    }


def run_research(save: bool = True) -> dict[str, Any]:
    before = frozen_stage36_manifest()
    inputs = load_stage36_inputs()
    candidates = {
        name: run_backtest(
            inputs["returns"],
            inputs["probabilities"],
            inputs["stress"],
            inputs["technical"],
            inputs["fundamental"],
            inputs["asset_vol"],
            objective_mode,
        )
        for name, objective_mode in OBJECTIVE_MODES.items()
    }
    stage36_path = stage36.stage35.stage34._load_period_csv(
        stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv", "month"
    )
    paths = {"Stage36_Frozen": stage36_path, **candidates}
    performance = performance_table(paths)
    bootstrap = bootstrap_table(stage36_path, candidates)

    indexed = performance.set_index(["Strategy", "Period"])
    gates: dict[str, Any] = {}
    checks = {name: _strategy_checks(path) for name, path in candidates.items()}
    for name in candidates:
        period_sharpes = {
            period: float(indexed.loc[(name, period), "Sharpe"])
            for period in (
                "full_2007_2026",
                "common_2010_2026",
                "locked_2018_2026",
            )
        }
        target_metric = "Calmar" if OBJECTIVE_MODES[name] == "calmar" else "Sortino"
        candidate_target = float(
            indexed.loc[(name, "full_2007_2026"), target_metric]
        )
        baseline_target = float(
            indexed.loc[("Stage36_Frozen", "full_2007_2026"), target_metric]
        )
        gates[name] = {
            "sharpe_floor": SHARPE_FLOOR,
            "period_sharpes": period_sharpes,
            "full_sharpe_at_least_one": bool(
                period_sharpes["full_2007_2026"] >= SHARPE_FLOOR
            ),
            "all_reported_periods_sharpe_at_least_one": bool(
                min(period_sharpes.values()) >= SHARPE_FLOOR
            ),
            "constraints_and_solver_pass": bool(
                checks[name]["all_solver_success"]
                and checks[name]["minimum_weight"] >= -1e-9
                and checks[name]["maximum_weight"] <= 1.0 + 1e-9
                and checks[name]["max_sum_error"] <= 1e-8
                and checks[name]["minimum_volatility_slack"] >= -1e-7
                and checks[name]["minimum_cdar_slack"] >= -1e-7
                and checks[name]["all_signal_months_precede_target"]
            ),
            "optimized_realized_metric": target_metric,
            "full_realized_metric": candidate_target,
            "stage36_full_realized_metric": baseline_target,
            "optimized_realized_metric_improves_vs_stage36": bool(
                candidate_target > baseline_target
            ),
        }
        gates[name]["meets_user_full_sharpe_requirement"] = bool(
            gates[name]["full_sharpe_at_least_one"]
            and gates[name]["constraints_and_solver_pass"]
        )
        gates[name]["promote_over_stage36"] = bool(
            gates[name]["meets_user_full_sharpe_requirement"]
            and gates[name]["optimized_realized_metric_improves_vs_stage36"]
        )

    eligible = [
        name
        for name, gate in gates.items()
        if gate["meets_user_full_sharpe_requirement"]
    ]
    promoted = [
        name for name, gate in gates.items() if gate["promote_over_stage36"]
    ]
    if promoted:
        selected = max(
            promoted,
            key=lambda name: float(indexed.loc[(name, "full_2007_2026"), "Sharpe"]),
        )
    else:
        selected = "Stage36_Frozen"

    after = frozen_stage36_manifest()
    report = {
        "stage": 40,
        "base": "Stage36_GVZ_OVXAssetRisk",
        "change_scope": "SLSQP objective only",
        "objective_definitions": {
            "calmar": (
                "expected causal CAGR after estimated transaction cost divided "
                "by absolute MDD of the strictly prior fixed-weight return path"
            ),
            "sortino": (
                "expected annual arithmetic return after estimated transaction "
                "cost divided by annualized downside deviation of the strictly "
                "prior fixed-weight return path"
            ),
        },
        "preserved": {
            "stage36_mu_sigma_and_gvz_ovx": True,
            "long_only_unlevered_sum_one": True,
            "annual_volatility_guard": stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY,
            "historical_cdar_guard": stage36.stage35.CATASTROPHE_CDAR,
            "cdar_confidence": stage36.stage35.CDAR_CONFIDENCE,
            "cost_model": "Stage36 unchanged",
            "single_asset_majority_cap": False,
        },
        "anti_overfit": {
            "objective_candidates_prespecified": list(OBJECTIVE_MODES.values()),
            "ratio_hyperparameter_grid_search": False,
            "future_returns_in_objective": False,
            "realized_sharpe_used_only_as_post_backtest_gate": True,
        },
        "stage36_frozen_files_unchanged": before == after,
        "checks": checks,
        "sharpe_gates": gates,
        "full_sharpe_requirement_eligible_strategies": eligible,
        "promoted_strategies": promoted,
        "selected_strategy": selected,
        "performance": json.loads(performance.to_json(orient="records")),
        "input_audits": inputs["audits"],
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, path in candidates.items():
            filename = (
                "stage40_calmar_monthly.csv"
                if OBJECTIVE_MODES[name] == "calmar"
                else "stage40_sortino_monthly.csv"
            )
            path.to_csv(OUTPUT_DIR / filename)
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
    print("=== STAGE40 CALMAR / SORTINO OBJECTIVES ===")
    print(
        result["performance"][
            ["Strategy", "Period", "CAGR", "Volatility", "Sharpe", "Sortino", "MDD", "Calmar"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\n=== SHARPE >= 1 GATES ===")
    print(json.dumps(result["report"]["sharpe_gates"], ensure_ascii=False, indent=2))
    print("selected", result["report"]["selected_strategy"])


if __name__ == "__main__":
    main()
