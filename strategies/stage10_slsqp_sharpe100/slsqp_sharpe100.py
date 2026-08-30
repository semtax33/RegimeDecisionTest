from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    cdar,
    ewma_cov,
    load_monthly_asset_returns,
    performance_summary,
    soft_anchor,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
)
from strategies.stage08_vkospi_factorial.factorial_bridge import (
    zero_macro_signals,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FULL_START = pd.Period("2007-04", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")
SHARPE_SCORE_WEIGHT = 0.50
CAGR_SCORE_WEIGHT = 0.50


def project_to_bounded_simplex(
    weights: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Project weights onto sum-one box constraints without bound drift."""
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    if lower.sum() > 1.0 or upper.sum() < 1.0:
        raise ValueError("The supplied bounds cannot contain a unit simplex.")
    left = float(np.min(weights - upper))
    right = float(np.max(weights - lower))
    for _ in range(100):
        shift = 0.5 * (left + right)
        projected = np.clip(weights - shift, lower, upper)
        if projected.sum() > 1.0:
            left = shift
        else:
            right = shift
    projected = np.clip(weights - 0.5 * (left + right), lower, upper)
    residual = float(1.0 - projected.sum())
    if residual > 0:
        room = upper - projected
        projected[int(np.argmax(room))] += residual
    elif residual < 0:
        room = projected - lower
        projected[int(np.argmax(room))] += residual
    return projected


def expected_return_and_covariance(
    history: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the existing SLSQP estimator while changing only its objective."""
    covariance = ewma_cov(
        history.tail(84), config.half_life, leverage=1.0
    )
    recent = history.tail(84)[ASSETS]
    long_mean = history[ASSETS].mean().to_numpy(dtype=float)
    recent_mean = (
        recent.ewm(halflife=24, adjust=False)
        .mean()
        .iloc[-1]
        .to_numpy(dtype=float)
    )
    expected_monthly_return = np.clip(
        0.80 * long_mean + 0.20 * recent_mean,
        -0.006,
        0.015,
    )
    return expected_monthly_return, covariance, recent.to_numpy(dtype=float)


def initial_weights(
    signal: pd.Series,
    covariance: np.ndarray,
    config: StrategyConfig,
) -> np.ndarray:
    """Use the current regime/volatility prior only as the numerical start point."""
    anchor = soft_anchor(signal)
    anchor = (
        config.regime_strength * anchor
        + (1 - config.regime_strength)
        * np.array([0.20, 0.45, 0.30, 0.05])
    )
    volatility = np.sqrt(np.diag(covariance)).clip(0.005, None)
    tilted = anchor * (
        np.median(volatility) / volatility
    ) ** config.invvol_tilt
    tilted /= tilted.sum()
    prior = 0.55 * anchor + 0.45 * tilted
    return prior / prior.sum()


def maximize_expected_sharpe(
    signal: pd.Series,
    history: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Maximize expected annualized Sharpe subject to existing SLSQP constraints."""
    if len(history) < 12:
        raise ValueError("At least 12 prior monthly observations are required.")
    expected_return, covariance, recent_returns = (
        expected_return_and_covariance(history, config)
    )
    prior = initial_weights(signal, covariance, config)
    target_volatility = config.target_vol * (
        0.86 + 0.20 * float(signal["p_growth_high"])
    )

    def annualized_values(weights: np.ndarray) -> tuple[float, float, float]:
        annual_return = 12 * float(weights @ expected_return)
        annual_volatility = math.sqrt(
            max(float(weights @ covariance @ weights), 0.0) * 12
        )
        expected_sharpe = (
            annual_return / annual_volatility
            if annual_volatility > 1e-12
            else -1e9
        )
        return annual_return, annual_volatility, expected_sharpe

    def objective(weights: np.ndarray) -> float:
        return -annualized_values(weights)[2]

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: (
                target_volatility
                - math.sqrt(
                    max(
                        float(weights @ covariance @ weights),
                        0.0,
                    )
                    * 12
                )
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.max_cdar
                + cdar(recent_returns @ weights, 0.90)
            ),
        },
    ]
    bounds = [
        (0.02, 0.68),
        (0.05, 0.88),
        (0.02, 0.62),
        (0.00, 0.38),
    ]
    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )
    if result.success and np.isfinite(result.x).all():
        weights = project_to_bounded_simplex(result.x, bounds)
        used_fallback = False
    else:
        weights = prior
        used_fallback = True
    annual_return, annual_volatility, expected_sharpe = annualized_values(
        weights
    )
    constraint_values = {
        "sum_error": float(abs(weights.sum() - 1.0)),
        "volatility_slack": float(
            target_volatility - annual_volatility
        ),
        "cdar_slack": float(
            config.max_cdar + cdar(recent_returns @ weights, 0.90)
        ),
    }
    detail: dict[str, object] = {
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun) if np.isfinite(result.fun) else np.nan,
        "expected_annual_return": annual_return,
        "expected_annual_volatility": annual_volatility,
        "expected_sharpe": expected_sharpe,
        "target_volatility": target_volatility,
        **constraint_values,
    }
    return weights, detail


def maximize_expected_sharpe_and_cagr(
    signal: pd.Series,
    history: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Maximize an equal-weight standardized Sharpe and expected-CAGR score."""
    if len(history) < 12:
        raise ValueError("At least 12 prior monthly observations are required.")
    expected_return, covariance, recent_returns = (
        expected_return_and_covariance(history, config)
    )
    prior = initial_weights(signal, covariance, config)
    target_volatility = config.target_vol * (
        0.86 + 0.20 * float(signal["p_growth_high"])
    )

    def annualized_values(
        weights: np.ndarray,
    ) -> tuple[float, float, float, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(
            float(weights @ covariance @ weights), 0.0
        )
        annual_return = 12 * monthly_return
        annual_volatility = math.sqrt(monthly_variance * 12)
        expected_sharpe = (
            annual_return / annual_volatility
            if annual_volatility > 1e-12
            else -1e9
        )
        expected_cagr = math.expm1(
            float(
                np.clip(
                    12 * (monthly_return - 0.5 * monthly_variance),
                    -20,
                    20,
                )
            )
        )
        return (
            annual_return,
            annual_volatility,
            expected_sharpe,
            expected_cagr,
        )

    asset_values = np.array(
        [
            annualized_values(np.eye(len(ASSETS))[index])[2:]
            for index in range(len(ASSETS))
        ],
        dtype=float,
    )
    metric_center = asset_values.mean(axis=0)
    metric_scale = asset_values.std(axis=0, ddof=0)
    metric_scale = np.where(metric_scale > 1e-8, metric_scale, 1.0)

    def standardized_score(
        weights: np.ndarray,
    ) -> tuple[float, float, float]:
        _, _, expected_sharpe, expected_cagr = annualized_values(weights)
        standardized = (
            np.array([expected_sharpe, expected_cagr]) - metric_center
        ) / metric_scale
        combined = (
            SHARPE_SCORE_WEIGHT * standardized[0]
            + CAGR_SCORE_WEIGHT * standardized[1]
        )
        return float(combined), float(standardized[0]), float(standardized[1])

    def objective(weights: np.ndarray) -> float:
        return -standardized_score(weights)[0]

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: (
                target_volatility
                - math.sqrt(
                    max(
                        float(weights @ covariance @ weights),
                        0.0,
                    )
                    * 12
                )
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.max_cdar
                + cdar(recent_returns @ weights, 0.90)
            ),
        },
    ]
    bounds = [
        (0.02, 0.68),
        (0.05, 0.88),
        (0.02, 0.62),
        (0.00, 0.38),
    ]
    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        # 1e-9 avoids false line-search failures while keeping constraints
        # accurate to far below any economically meaningful allocation unit.
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if result.success and np.isfinite(result.x).all():
        weights = project_to_bounded_simplex(result.x, bounds)
        used_fallback = False
    else:
        weights = prior
        used_fallback = True
    (
        annual_return,
        annual_volatility,
        expected_sharpe,
        expected_cagr,
    ) = annualized_values(weights)
    combined_score, standardized_sharpe, standardized_cagr = (
        standardized_score(weights)
    )
    detail: dict[str, object] = {
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun) if np.isfinite(result.fun) else np.nan,
        "expected_annual_return": annual_return,
        "expected_annual_volatility": annual_volatility,
        "expected_sharpe": expected_sharpe,
        "expected_cagr": expected_cagr,
        "standardized_sharpe": standardized_sharpe,
        "standardized_cagr": standardized_cagr,
        "combined_score": combined_score,
        "target_volatility": target_volatility,
        "sum_error": float(abs(weights.sum() - 1.0)),
        "volatility_slack": float(
            target_volatility - annual_volatility
        ),
        "cdar_slack": float(
            config.max_cdar + cdar(recent_returns @ weights, 0.90)
        ),
    }
    return weights, detail


def run_optimizer_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    optimizer,
) -> pd.DataFrame:
    """Use the optimizer output as 100% of the portfolio, with no Hard blend."""
    config = StrategyConfig()
    months = returns.index.intersection(signals.index)
    rows: list[dict[str, object]] = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < 12:
            continue
        weights, detail = optimizer(signals.loc[month], history, config)
        delta = weights - pretrade
        turnover = (
            float(np.abs(delta).sum())
            if first_trade
            else 0.5 * float(np.abs(delta).sum())
        )
        trade_cost = float(np.abs(delta).sum()) * DOMESTIC_TRADE_COST
        fx_cost = (
            abs(
                (weights[2] + weights[3])
                - (pretrade[2] + pretrade[3])
            )
            * FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1 + asset_return) / (1 + gross_return)
        rows.append(
            {
                "month": month,
                "signal_month": signals.loc[month, "signal_month"],
                "regime": signals.loc[month, "regime"],
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **detail,
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
        first_trade = False
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def run_sharpe_slsqp_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    return run_optimizer_backtest(
        returns, signals, maximize_expected_sharpe
    )


def run_sharpe_cagr_slsqp_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    return run_optimizer_backtest(
        returns, signals, maximize_expected_sharpe_and_cagr
    )


def load_comparison_paths() -> dict[str, pd.DataFrame]:
    paths: dict[str, pd.DataFrame] = {}
    for name, relative in {
        "Previous_StrictHard40_SLSQP60": (
            "strategies/stage09_strict_hard_slsqp/outputs/"
            "strict_hard40_slsqp60_monthly.csv"
        ),
        "OriginalMultiObjective_SLSQP100": (
            "strategies/stage09_strict_hard_slsqp/outputs/slsqp_path.csv"
        ),
        "ZeroTune_VKOSPI": (
            "strategies/stage07_zero_tune_vkospi/outputs/zero_tune_monthly.csv"
        ),
        "Current_Robust_VKOSPI": (
            "results/balanced_logistic_no_sjm_final_reconciled.csv"
        ),
    }.items():
        frame = pd.read_csv(ROOT / relative, index_col=0)
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        paths[name] = frame
    return paths


def metric_record(
    strategy: str,
    path: pd.DataFrame,
    period: str,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, object]:
    view = path.loc[start:end]
    metrics = performance_summary(view["return"])
    return {
        "Strategy": strategy,
        "Period": period,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{name: float(value) for name, value in metrics.items()},
        "AvgTurnover": float(view["turnover"].mean()),
    }


def run_research(save: bool = True) -> dict[str, object]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    signals = zero_macro_signals(probabilities)
    sharpe_cagr_path = run_sharpe_cagr_slsqp_backtest(returns, signals)
    sharpe_only_path = run_sharpe_slsqp_backtest(returns, signals)

    paths = {
        "SharpeCAGRObjective_SLSQP100": sharpe_cagr_path,
        "SharpeOnlyObjective_SLSQP100": sharpe_only_path,
        **load_comparison_paths(),
    }
    common_end = min(path.index.max() for path in paths.values())
    comparison_rows: list[dict[str, object]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            comparison_rows.append(
                metric_record(name, path, period, start, common_end)
            )
    comparison = pd.DataFrame(comparison_rows)
    full = comparison.loc[
        comparison["Period"].eq("full_2007_2026")
        & comparison["Strategy"].eq("SharpeCAGRObjective_SLSQP100")
    ].iloc[0]
    locked = comparison.loc[
        comparison["Period"].eq("locked_2018_2026")
        & comparison["Strategy"].eq("SharpeCAGRObjective_SLSQP100")
    ].iloc[0]

    def solver_summary(path: pd.DataFrame) -> dict[str, object]:
        return {
            "months": int(len(path)),
            "successes": int(path["solver_success"].sum()),
            "failures": int((~path["solver_success"]).sum()),
            "fallbacks": int(path["used_fallback"].sum()),
            "maximum_weight_sum_error": float(path["sum_error"].max()),
            "minimum_volatility_slack": float(
                path["volatility_slack"].min()
            ),
            "minimum_cdar_slack": float(path["cdar_slack"].min()),
        }

    report = {
        "strategy": "SharpeCAGRObjective_SLSQP100",
        "allocation": {
            "hard_share": 0.0,
            "slsqp_share": 1.0,
            "leverage": 1.0,
        },
        "objective": (
            f"maximize {SHARPE_SCORE_WEIGHT:.2f} times cross-sectionally "
            "standardized expected Sharpe plus "
            f"{CAGR_SCORE_WEIGHT:.2f} times cross-sectionally standardized "
            "expected CAGR"
        ),
        "expected_cagr_definition": (
            "exp(12 * (expected monthly return - 0.5 * expected monthly "
            "variance)) - 1"
        ),
        "standardization": (
            "At each month, center and scale each metric using the four "
            "standalone asset values estimated from the same causal history."
        ),
        "preserved_from_existing_slsqp": [
            "expected return estimator",
            "EWMA covariance estimator",
            "long-only asset bounds",
            "weights sum to one",
            "state-dependent target volatility constraint",
            "CDaR constraint",
        ],
        "removed_from_objective": [
            "volatility penalty",
            "CDaR penalty",
            "turnover penalty",
            "tracking penalty",
        ],
        "full_period": {
            name: float(full[name])
            for name in [
                "Months", "CAGR", "Sharpe", "MDD",
                "Calmar", "AvgTurnover",
            ]
        },
        "locked_period": {
            name: float(locked[name])
            for name in [
                "Months", "CAGR", "Sharpe", "MDD",
                "Calmar", "AvgTurnover",
            ]
        },
        "solver": solver_summary(sharpe_cagr_path),
        "sharpe_only_solver": solver_summary(sharpe_only_path),
        "checks": {
            "macro_signal_precedes_target": bool(
                (
                    sharpe_cagr_path["signal_month"]
                    < sharpe_cagr_path.index
                ).all()
            ),
            "weight_sum_is_one": bool(
                np.allclose(
                    sharpe_cagr_path[
                        [f"w_{asset}" for asset in ASSETS]
                    ].sum(axis=1),
                    1.0,
                )
            ),
            "constraints_within_numerical_tolerance": bool(
                sharpe_cagr_path["volatility_slack"].min() >= -1e-8
                and sharpe_cagr_path["cdar_slack"].min() >= -1e-8
            ),
            "no_hard_blend": True,
            "no_leverage": True,
        },
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        signals.to_csv(OUTPUT_DIR / "zero_tune_macro_signals.csv")
        sharpe_cagr_path.to_csv(
            OUTPUT_DIR / "slsqp_sharpe_cagr100_monthly.csv"
        )
        sharpe_only_path.to_csv(
            OUTPUT_DIR / "slsqp_sharpe100_monthly.csv"
        )
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        (OUTPUT_DIR / "slsqp_sharpe_cagr100_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "signals": signals,
        "path": sharpe_cagr_path,
        "sharpe_only_path": sharpe_only_path,
        "comparison": comparison,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print("saved", OUTPUT_DIR)


if __name__ == "__main__":
    main()
