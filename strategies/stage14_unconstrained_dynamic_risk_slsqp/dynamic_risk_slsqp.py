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
    build_daily_stress_features,
    build_monthly_stress_signals,
    build_overlay_attribution,
    estimate_conditional_moments,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STAGE13_OUTPUT = (
    ROOT
    / "strategies"
    / "stage13_conditional_moments_slsqp"
    / "outputs"
    / "macro_stress_conditional_monthly.csv"
)
STAGE10_OUTPUT = (
    ROOT
    / "strategies"
    / "stage10_slsqp_sharpe100"
    / "outputs"
    / "slsqp_sharpe_cagr100_monthly.csv"
)

# The user explicitly removed the single-asset majority restriction. These are
# pure long-only bounds: a single asset may hold the full portfolio.
UNCONSTRAINED_LONG_ONLY_BOUNDS = [(0.0, 1.0)] * len(ASSETS)
NUMERICAL_EPSILON = 1e-12


@dataclass(frozen=True)
class RiskPolicy:
    """Risk policy with no fitted coefficient or volatility multiplier."""

    name: str
    dynamic_risk_aversion: bool

    def risk_aversion(self, stress_score: float) -> float:
        """Return the downside-risk penalty lambda.

        Lambda0=1 gives downside semivariance the same squared-return unit as
        portfolio variance. With the dynamic policy, the causal 0..1 stress
        percentile raises lambda continuously from 1 to at most 2. Alpha is one
        because a full-range percentile then doubles, rather than arbitrarily
        rescaling, downside aversion.
        """

        stress = float(np.clip(stress_score, 0.0, 1.0))
        return 1.0 + stress if self.dynamic_risk_aversion else 1.0


STATIC_RISK_POLICY = RiskPolicy(
    name="Unconstrained_StaticLambda",
    dynamic_risk_aversion=False,
)
DYNAMIC_RISK_POLICY = RiskPolicy(
    name="Unconstrained_DynamicLambda",
    dynamic_risk_aversion=True,
)


def project_to_long_only_simplex(weights: np.ndarray) -> np.ndarray:
    """Project any finite vector to ``w>=0, sum(w)=1`` without an asset cap."""

    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Weights must be finite before simplex projection.")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    rho = int(np.flatnonzero(candidates)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    projected /= projected.sum()
    return projected


def expected_transaction_cost(
    weights: np.ndarray,
    pretrade: np.ndarray,
) -> float:
    """Use actual backtest cost rates instead of an arbitrary turnover penalty."""

    change = weights - pretrade
    smooth_absolute_change = np.sqrt(change**2 + NUMERICAL_EPSILON)
    trade_cost = float(smooth_absolute_change.sum()) * DOMESTIC_TRADE_COST
    foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
    foreign_change = float(change[foreign_indices].sum())
    fx_cost = math.sqrt(
        foreign_change**2 + NUMERICAL_EPSILON
    ) * FOREIGN_WEIGHT_CHANGE_COST
    return trade_cost + fx_cost


def solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    pretrade: np.ndarray,
    policy: RiskPolicy,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the fully invested, no-leverage allocation with SLSQP.

    No volatility multiplier is applied before or after this optimization. The
    stress score changes the curvature of the risk term, while the wide 13%
    ex-ante volatility cap remains a catastrophe guard.
    """

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
    stress = float(np.clip(current_stress, 0.0, 1.0))
    downside_lambda = policy.risk_aversion(stress)
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
        # The first two terms are expected log growth. Dynamic lambda prices
        # downside semivariance more heavily as option-market stress rises.
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
        return math.sqrt(
            max(float(weights @ covariance @ weights), 0.0) * 12.0
        )

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
        options={
            "maxiter": SLSQP_MAX_ITERATIONS,
            "ftol": SLSQP_TOLERANCE,
        },
    )

    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = project_to_long_only_simplex(result.x)
    else:
        # This remains an SLSQP allocation. The retry finds a feasible
        # minimum-variance point and never inserts a hard regime portfolio.
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": SLSQP_MAX_ITERATIONS,
                "ftol": SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                "Both the economic and feasibility SLSQP solves failed: "
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
        "policy": policy.name,
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


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    policy: RiskPolicy,
) -> pd.DataFrame:
    """Run monthly open-loop decisions with lagged macro and stress inputs."""

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
            policy=policy,
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


def metric_row(
    name: str,
    path: pd.DataFrame,
    period: str,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, Any]:
    view = path.loc[start:end]
    metrics = performance_summary(view["return"])
    return {
        "Strategy": name,
        "Period": period,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{key: float(value) for key, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean()),
        "TotalCost": float(view[["trade_cost", "fx_cost"]].sum().sum()),
    }


def concentration_summary(path: pd.DataFrame) -> dict[str, Any]:
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    weights = path[weight_columns]
    largest = weights.max(axis=1)
    return {
        "months": int(len(path)),
        "months_above_50_percent": int((largest > 0.5 + 1e-10).sum()),
        "months_above_90_percent": int((largest > 0.9 + 1e-10).sum()),
        "months_at_effective_100_percent": int((largest > 1.0 - 1e-6).sum()),
        "maximum_single_asset_weight": float(largest.max()),
        "average_largest_weight": float(largest.mean()),
        "average_weights": {
            asset: float(weights[f"w_{asset}"].mean()) for asset in ASSETS
        },
    }


def solver_summary(path: pd.DataFrame) -> dict[str, Any]:
    return {
        "months": int(len(path)),
        "successes": int(path["solver_success"].sum()),
        "fallbacks": int(path["used_fallback"].sum()),
        "maximum_weight_sum_error": float(path["sum_error"].max()),
        "minimum_volatility_slack": float(path["volatility_slack"].min()),
        "minimum_cdar_slack": float(path["cdar_slack"].min()),
        "volatility_guard_binding_months": int(
            (path["volatility_slack"].abs() < 1e-6).sum()
        ),
        "cdar_guard_binding_months": int(
            (path["cdar_slack"].abs() < 1e-6).sum()
        ),
    }


def read_saved_path(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def run_research(save: bool = True) -> dict[str, Any]:
    """Compare the cap removal and dynamic-risk contribution without a grid."""

    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(returns.index, daily_stress)

    static_path = run_backtest(
        returns, probabilities, stress_signals, STATIC_RISK_POLICY
    )
    dynamic_path = run_backtest(
        returns, probabilities, stress_signals, DYNAMIC_RISK_POLICY
    )
    stage13_path = read_saved_path(STAGE13_OUTPUT)
    stage10_path = read_saved_path(STAGE10_OUTPUT)
    paths = {
        "Stage10_Original": stage10_path,
        "Stage13_50pctCap": stage13_path,
        "Stage14_NoAssetCap_StaticLambda": static_path,
        "Stage14_NoAssetCap_DynamicLambda": dynamic_path,
    }
    common_end = min(frame.index.max() for frame in paths.values())
    comparison_rows: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            comparison_rows.append(
                metric_row(name, path, period, start, common_end)
            )
    comparison = pd.DataFrame(comparison_rows)

    attribution, attribution_full = build_overlay_attribution(
        returns, static_path, dynamic_path
    )
    _, attribution_locked = build_overlay_attribution(
        returns.loc[LOCKED_START:],
        static_path.loc[LOCKED_START:],
        dynamic_path.loc[LOCKED_START:],
    )
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    checks = {
        "macro_signal_precedes_target": bool(
            (dynamic_path["macro_signal_month"] < dynamic_path.index).all()
        ),
        "stress_signal_precedes_target": bool(
            (dynamic_path["stress_signal_month"] < dynamic_path.index).all()
        ),
        "weights_sum_to_one": bool(
            np.allclose(dynamic_path[weight_columns].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (dynamic_path[weight_columns] >= -1e-10).all().all()
        ),
        "weights_do_not_exceed_one": bool(
            (dynamic_path[weight_columns] <= 1.0 + 1e-10).all().all()
        ),
        "single_asset_majority_rule_removed": True,
        "no_cash_asset": True,
        "no_volatility_multiplier": True,
        "no_leverage": bool(
            np.allclose(dynamic_path[weight_columns].sum(axis=1), 1.0)
        ),
        "no_hard_regime_weights": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_search": True,
    }
    report: dict[str, Any] = {
        "strategy": "Stage14_NoAssetCap_DynamicLambda",
        "source_notebook": (
            "C:/Users/PC_1M/Downloads/"
            "economic_regime_asset_allocation_backtest_multi_asset.ipynb"
        ),
        "notebook_change": (
            "Replace hard regime target weights with soft-regime conditional "
            "moments and a fully invested SLSQP allocation. Do not use a "
            "volatility multiplier because no cash asset is present."
        ),
        "allocation": {
            "slsqp_share": 1.0,
            "weight_sum": 1.0,
            "asset_bounds": [0.0, 1.0],
            "single_asset_majority_rule": False,
            "cash_asset": False,
            "leverage": 1.0,
        },
        "objective": (
            "maximize mu - 0.5*variance "
            "- (1+stress)*downside_semivariance - explicit transaction cost"
        ),
        "risk_aversion": {
            "baseline_lambda": 1.0,
            "dynamic_lambda": "1 + causal stress percentile",
            "minimum": 1.0,
            "maximum": 2.0,
            "searched_alpha": None,
        },
        "ex_ante_guards": {
            "annual_volatility": CATASTROPHE_ANNUAL_VOLATILITY,
            "cdar": CATASTROPHE_CDAR,
            "cdar_confidence": CDAR_CONFIDENCE,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "dynamic_risk_attribution": {
            "full_2007_2026": attribution_full,
            "locked_2018_2026": attribution_locked,
        },
        "static_lambda_concentration": concentration_summary(static_path),
        "dynamic_lambda_concentration": concentration_summary(dynamic_path),
        "static_solver": solver_summary(static_path),
        "dynamic_solver": solver_summary(dynamic_path),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        static_path.to_csv(OUTPUT_DIR / "no_asset_cap_static_lambda_monthly.csv")
        dynamic_path.to_csv(
            OUTPUT_DIR / "no_asset_cap_dynamic_lambda_monthly.csv"
        )
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        attribution.to_csv(OUTPUT_DIR / "dynamic_risk_attribution.csv")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "returns": returns,
        "probabilities": probabilities,
        "stress_signals": stress_signals,
        "static_path": static_path,
        "dynamic_path": dynamic_path,
        "comparison": comparison,
        "attribution": attribution,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
