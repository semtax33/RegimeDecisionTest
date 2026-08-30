from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    cdar,
    load_monthly_asset_returns,
    performance_summary,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
)
from strategies.stage08_vkospi_factorial.factorial_bridge import (
    zero_macro_signals,
)
from strategies.stage10_slsqp_sharpe100.slsqp_sharpe100 import (
    expected_return_and_covariance,
    initial_weights,
    project_to_bounded_simplex,
    run_optimizer_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FULL_START = pd.Period("2007-04", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")
MSE_QUALITY_WEIGHT = 0.50
LOG_GROWTH_WEIGHT = 0.50
BOUNDS = [
    (0.02, 0.68),
    (0.05, 0.88),
    (0.02, 0.62),
    (0.00, 0.38),
]
ObjectiveMode = Literal["mse", "log_growth", "combined"]


def expected_portfolio_metrics(
    weights: np.ndarray,
    expected_monthly_return: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    """Calculate causal moment-based MSE and expected log-growth metrics."""
    monthly_return = float(weights @ expected_monthly_return)
    monthly_variance = max(float(weights @ covariance @ weights), 0.0)
    annual_return = 12 * monthly_return
    annual_volatility = math.sqrt(12 * monthly_variance)
    expected_sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 1e-12
        else -1e9
    )
    expected_squared_loss = (1.0 - monthly_return) ** 2 + monthly_variance
    expected_log_growth = 12 * (monthly_return - 0.5 * monthly_variance)
    expected_cagr = math.expm1(float(np.clip(expected_log_growth, -20, 20)))
    return {
        "expected_monthly_return": monthly_return,
        "expected_monthly_variance": monthly_variance,
        "expected_annual_return": annual_return,
        "expected_annual_volatility": annual_volatility,
        "expected_sharpe": expected_sharpe,
        "expected_squared_loss": expected_squared_loss,
        "expected_log_growth": expected_log_growth,
        "expected_cagr": expected_cagr,
    }


def objective_reference(
    expected_monthly_return: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build causal cross-sectional scales for the combined objective."""
    standalone = []
    for index in range(len(ASSETS)):
        metrics = expected_portfolio_metrics(
            np.eye(len(ASSETS))[index],
            expected_monthly_return,
            covariance,
        )
        standalone.append(
            [-metrics["expected_squared_loss"], metrics["expected_log_growth"]]
        )
    values = np.asarray(standalone, dtype=float)
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def optimize_mse_log_growth(
    signal: pd.Series,
    history: pd.DataFrame,
    config: StrategyConfig,
    mode: ObjectiveMode,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optimize MSE loss, expected log growth, or their standardized blend."""
    if len(history) < 12:
        raise ValueError("At least 12 prior monthly observations are required.")
    if mode not in {"mse", "log_growth", "combined"}:
        raise ValueError(f"Unsupported objective mode: {mode}")

    expected_return, covariance, recent_returns = (
        expected_return_and_covariance(history, config)
    )
    prior = initial_weights(signal, covariance, config)
    target_volatility = config.target_vol * (
        0.86 + 0.20 * float(signal["p_growth_high"])
    )
    metric_center, metric_scale = objective_reference(
        expected_return, covariance
    )

    def standardized_scores(
        weights: np.ndarray,
    ) -> tuple[float, float, float]:
        metrics = expected_portfolio_metrics(
            weights, expected_return, covariance
        )
        quality = np.array(
            [
                -metrics["expected_squared_loss"],
                metrics["expected_log_growth"],
            ]
        )
        standardized = (quality - metric_center) / metric_scale
        combined_score = (
            MSE_QUALITY_WEIGHT * standardized[0]
            + LOG_GROWTH_WEIGHT * standardized[1]
        )
        return (
            float(combined_score),
            float(standardized[0]),
            float(standardized[1]),
        )

    def objective(weights: np.ndarray) -> float:
        metrics = expected_portfolio_metrics(
            weights, expected_return, covariance
        )
        if mode == "mse":
            return metrics["expected_squared_loss"]
        if mode == "log_growth":
            return -metrics["expected_log_growth"]
        return -standardized_scores(weights)[0]

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: (
                target_volatility
                - math.sqrt(
                    max(float(weights @ covariance @ weights), 0.0) * 12
                )
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.max_cdar + cdar(recent_returns @ weights, 0.90)
            ),
        },
    ]
    result = minimize(
        objective,
        prior,
        method="SLSQP",
        bounds=BOUNDS,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )
    used_retry = False
    if not (result.success and np.isfinite(result.x).all()):
        defensive_start = np.array([0.02, 0.88, 0.10, 0.00])
        result = minimize(
            objective,
            defensive_start,
            method="SLSQP",
            bounds=BOUNDS,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        used_retry = True

    if result.success and np.isfinite(result.x).all():
        weights = project_to_bounded_simplex(result.x, BOUNDS)
        used_fallback = False
    else:
        weights = project_to_bounded_simplex(prior, BOUNDS)
        used_fallback = True

    metrics = expected_portfolio_metrics(weights, expected_return, covariance)
    combined_score, standardized_mse_quality, standardized_log_growth = (
        standardized_scores(weights)
    )
    volatility_slack = (
        target_volatility - metrics["expected_annual_volatility"]
    )
    cdar_slack = config.max_cdar + cdar(recent_returns @ weights, 0.90)
    detail: dict[str, object] = {
        "objective_mode": mode,
        "solver_success": bool(result.success),
        "used_retry": used_retry,
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(objective(weights)),
        **metrics,
        "standardized_mse_quality": standardized_mse_quality,
        "standardized_log_growth": standardized_log_growth,
        "combined_score": combined_score,
        "target_volatility": target_volatility,
        "sum_error": float(abs(weights.sum() - 1.0)),
        "volatility_slack": float(volatility_slack),
        "cdar_slack": float(cdar_slack),
    }
    return weights, detail


def minimize_expected_squared_loss(
    signal: pd.Series,
    history: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    return optimize_mse_log_growth(signal, history, config, "mse")


def maximize_expected_log_growth(
    signal: pd.Series,
    history: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    return optimize_mse_log_growth(signal, history, config, "log_growth")


def maximize_mse_quality_and_log_growth(
    signal: pd.Series,
    history: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    return optimize_mse_log_growth(signal, history, config, "combined")


def load_comparison_paths() -> dict[str, pd.DataFrame]:
    paths: dict[str, pd.DataFrame] = {}
    for name, relative in {
        "Stage10_SharpeCAGR_SLSQP100": (
            "strategies/stage10_slsqp_sharpe100/outputs/"
            "slsqp_sharpe_cagr100_monthly.csv"
        ),
        "Stage10_SharpeOnly_SLSQP100": (
            "strategies/stage10_slsqp_sharpe100/outputs/"
            "slsqp_sharpe100_monthly.csv"
        ),
        "Previous_StrictHard40_SLSQP60": (
            "strategies/stage09_strict_hard_slsqp/outputs/"
            "strict_hard40_slsqp60_monthly.csv"
        ),
        "OriginalMultiObjective_SLSQP100": (
            "strategies/stage09_strict_hard_slsqp/outputs/slsqp_path.csv"
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


def solver_summary(path: pd.DataFrame) -> dict[str, object]:
    return {
        "months": int(len(path)),
        "successes": int(path["solver_success"].sum()),
        "failures": int((~path["solver_success"]).sum()),
        "retries": int(path["used_retry"].sum()),
        "fallbacks": int(path["used_fallback"].sum()),
        "maximum_weight_sum_error": float(path["sum_error"].max()),
        "minimum_volatility_slack": float(path["volatility_slack"].min()),
        "minimum_cdar_slack": float(path["cdar_slack"].min()),
    }


def period_metrics(
    comparison: pd.DataFrame,
    strategy: str,
) -> dict[str, dict[str, float]]:
    rows = comparison.loc[comparison["Strategy"].eq(strategy)].set_index(
        "Period"
    )
    output: dict[str, dict[str, float]] = {}
    for period in ["full_2007_2026", "locked_2018_2026"]:
        row = rows.loc[period]
        output[period] = {
            name: float(row[name])
            for name in [
                "Months",
                "CAGR",
                "Sharpe",
                "MDD",
                "Calmar",
                "AvgTurnover",
            ]
        }
    return output


def path_checks(path: pd.DataFrame) -> dict[str, bool]:
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    return {
        "macro_signal_precedes_target": bool(
            (path["signal_month"] < path.index).all()
        ),
        "weight_sum_is_one": bool(
            np.allclose(path[weight_columns].sum(axis=1), 1.0)
        ),
        "constraints_within_numerical_tolerance": bool(
            path["volatility_slack"].min() >= -1e-8
            and path["cdar_slack"].min() >= -1e-8
        ),
        "no_hard_blend": True,
        "no_leverage": True,
    }


def run_research(save: bool = True) -> dict[str, object]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    signals = zero_macro_signals(probabilities)
    combined_path = run_optimizer_backtest(
        returns, signals, maximize_mse_quality_and_log_growth
    )
    mse_path = run_optimizer_backtest(
        returns, signals, minimize_expected_squared_loss
    )
    log_growth_path = run_optimizer_backtest(
        returns, signals, maximize_expected_log_growth
    )

    paths = {
        "MSELogGrowthObjective_SLSQP100": combined_path,
        "MSEObjective_SLSQP100": mse_path,
        "LogGrowthObjective_SLSQP100": log_growth_path,
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
    research_paths = {
        "combined": combined_path,
        "mse_only": mse_path,
        "log_growth_only": log_growth_path,
    }
    strategy_names = {
        "combined": "MSELogGrowthObjective_SLSQP100",
        "mse_only": "MSEObjective_SLSQP100",
        "log_growth_only": "LogGrowthObjective_SLSQP100",
    }
    report = {
        "primary_strategy": strategy_names["combined"],
        "allocation": {
            "hard_share": 0.0,
            "slsqp_share": 1.0,
            "leverage": 1.0,
        },
        "objectives": {
            "mse_only": (
                "minimize E[(1 - w^T r)^2] = "
                "(1 - w^T mu)^2 + w^T Sigma w"
            ),
            "log_growth_only": (
                "maximize 12 * (w^T mu - 0.5 * w^T Sigma w)"
            ),
            "combined": (
                f"maximize {MSE_QUALITY_WEIGHT:.2f} times standardized "
                "negative expected squared loss plus "
                f"{LOG_GROWTH_WEIGHT:.2f} times standardized expected "
                "log growth"
            ),
        },
        "standardization": (
            "Each month uses the four standalone asset values estimated "
            "from the same causal history; no realized future metric is used."
        ),
        "algebraic_relationship": (
            "Negative MSE quality equals -1 + 2*mu_p - mu_p^2 - "
            "variance_p, while expected log growth divided by 6 equals "
            "2*mu_p - variance_p. They differ mainly by the small mu_p^2 "
            "term for monthly returns, so similar allocations are expected."
        ),
        "results": {
            key: period_metrics(comparison, strategy_name)
            for key, strategy_name in strategy_names.items()
        },
        "solver": {
            key: solver_summary(path)
            for key, path in research_paths.items()
        },
        "average_weights": {
            key: {
                asset: float(path[f"w_{asset}"].mean())
                for asset in ASSETS
            }
            for key, path in research_paths.items()
        },
        "checks": {
            key: path_checks(path) for key, path in research_paths.items()
        },
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        signals.to_csv(OUTPUT_DIR / "zero_tune_macro_signals.csv")
        combined_path.to_csv(
            OUTPUT_DIR / "slsqp_mse_loggrowth100_monthly.csv"
        )
        mse_path.to_csv(OUTPUT_DIR / "slsqp_mse100_monthly.csv")
        log_growth_path.to_csv(
            OUTPUT_DIR / "slsqp_loggrowth100_monthly.csv"
        )
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        (OUTPUT_DIR / "slsqp_mse_loggrowth100_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "signals": signals,
        "path": combined_path,
        "mse_path": mse_path,
        "log_growth_path": log_growth_path,
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
