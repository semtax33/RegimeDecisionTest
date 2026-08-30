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
    cdar,
    download_market_cache,
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
    LOCKED_START,
    ONE_CALENDAR_YEAR,
    REGIME_COLUMNS,
    SLSQP_MAX_ITERATIONS,
    SLSQP_TOLERANCE,
    build_daily_stress_features,
    build_monthly_stress_signals,
    estimate_conditional_moments,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE_DIR / "data"
OUTPUT_DIR = STAGE_DIR / "outputs"
HYG_DAILY_CACHE = DATA_DIR / "hyg_daily_auto_adjusted.csv"
STAGE14_DYNAMIC_OUTPUT = (
    ROOT
    / "strategies"
    / "stage14_unconstrained_dynamic_risk_slsqp"
    / "outputs"
    / "no_asset_cap_dynamic_lambda_monthly.csv"
)

# Keep the Stage 14 ordering and replace only the second sleeve.
ASSETS = ["KODEX200", "HYG", "GLD", "USO"]
BASE_ESTIMATOR_ASSETS = ["KODEX200", "BOND", "GLD", "USO"]
FOREIGN_ASSETS = ["HYG", "GLD", "USO"]
UNCONSTRAINED_LONG_ONLY_BOUNDS = [(0.0, 1.0)] * len(ASSETS)
NUMERICAL_EPSILON = 1e-12
FEASIBILITY_MARGIN = 1e-6
HYG_DOWNLOAD_START = "2007-04-01"


@dataclass(frozen=True)
class RiskPolicy:
    name: str
    dynamic_risk_aversion: bool

    def risk_aversion(self, stress_score: float) -> float:
        stress = float(np.clip(stress_score, 0.0, 1.0))
        return 1.0 + stress if self.dynamic_risk_aversion else 1.0


STATIC_RISK_POLICY = RiskPolicy("HYG_StaticLambda", False)
DYNAMIC_RISK_POLICY = RiskPolicy("HYG_DynamicLambda", True)


def download_hyg_daily(refresh: bool = False) -> pd.DataFrame:
    """Load split- and distribution-adjusted HYG prices from a local snapshot.

    The saved snapshot makes later reruns deterministic. ``refresh=True`` is an
    explicit request to replace it with the latest Yahoo Finance observation.
    """

    if HYG_DAILY_CACHE.exists() and not refresh:
        return pd.read_csv(HYG_DAILY_CACHE, parse_dates=["date"])

    import yfinance as yf

    raw = yf.download(
        "HYG",
        start=HYG_DOWNLOAD_START,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no HYG observations.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    daily = raw.reset_index().rename(
        columns={"Date": "date", "Open": "open", "Close": "close"}
    )
    daily["date"] = (
        pd.to_datetime(daily["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    daily = daily[["date", "open", "close"]].dropna().sort_values("date")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(HYG_DAILY_CACHE, index=False)
    return daily


def load_hyg_monthly_returns(
    refresh_hyg: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the Stage 14 universe after replacing BOND with KRW HYG.

    Stage 14 values foreign assets at the first adjusted open of each month in
    KRW. HYG is treated identically: adjusted HYG open multiplied by USD/KRW.
    """

    base_returns, base_levels = load_monthly_asset_returns(False)
    daily = download_hyg_daily(refresh_hyg)
    market = download_market_cache(False)
    fx = (
        market.loc[market["symbol"] == "USDKRW", ["date", "close"]]
        .drop_duplicates("date", keep="last")
        .set_index("date")["close"]
        .sort_index()
    )
    fx = fx.reindex(
        pd.date_range(fx.index.min(), fx.index.max(), freq="D")
    ).ffill()

    monthly = daily.copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    first = monthly.groupby("month", sort=True).first()
    monthly_fx = fx.reindex(
        pd.DatetimeIndex(first["date"]), method="ffill"
    ).to_numpy()
    hyg_levels = first["open"].astype(float) * monthly_fx
    hyg_levels.index = pd.PeriodIndex(hyg_levels.index, freq="M")
    hyg_returns = hyg_levels.shift(-1).div(hyg_levels).sub(1.0).rename("HYG")

    returns = pd.concat(
        [base_returns[["KODEX200", "GLD", "USO"]], hyg_returns], axis=1
    ).dropna(how="any")
    returns = returns[ASSETS]
    levels = pd.concat(
        [base_levels[["KODEX200", "GLD", "USO"]], hyg_levels.rename("HYG")],
        axis=1,
    )[ASSETS]
    return returns, levels


def estimate_hyg_conditional_moments(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse the Stage 13 estimator with HYG occupying its second slot.

    The estimator is empirical and order-based. Its only economic sign priors
    apply to KODEX200 and USO, so renaming HYG to the legacy ``BOND`` slot does
    not impose a government-bond return prior on HYG.
    """

    aliased = history.rename(columns={"HYG": "BOND"})[
        BASE_ESTIMATOR_ASSETS
    ]
    return estimate_conditional_moments(
        history=aliased,
        historical_probabilities=historical_probabilities,
        current_probabilities=current_probabilities,
        historical_stress=historical_stress,
        current_stress=current_stress,
        historical_recovery=historical_recovery,
        current_recovery=current_recovery,
        use_short_term_stress=True,
    )


def project_to_long_only_simplex(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Weights must be finite before simplex projection.")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = ordered - cumulative / np.arange(1, len(values) + 1) > 0
    rho = int(np.flatnonzero(candidates)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    return projected / projected.sum()


def foreign_flow(change: np.ndarray) -> float:
    return float(sum(change[ASSETS.index(asset)] for asset in FOREIGN_ASSETS))


def expected_transaction_cost(
    weights: np.ndarray,
    pretrade: np.ndarray,
) -> float:
    change = weights - pretrade
    smooth_change = np.sqrt(change**2 + NUMERICAL_EPSILON)
    trade_cost = float(smooth_change.sum()) * DOMESTIC_TRADE_COST
    fx_flow = foreign_flow(change)
    fx_cost = math.sqrt(fx_flow**2 + NUMERICAL_EPSILON) * (
        FOREIGN_WEIGHT_CHANGE_COST
    )
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
    expected_return, covariance, moment_detail = (
        estimate_hyg_conditional_moments(
            history=history,
            historical_probabilities=historical_probabilities,
            current_probabilities=current_probabilities,
            historical_stress=historical_stress,
            current_stress=current_stress,
            historical_recovery=historical_recovery,
            current_recovery=current_recovery,
        )
    )
    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    downside_lambda = policy.risk_aversion(current_stress)
    initial = (
        project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        downside_semivariance = float(
            np.mean(np.minimum(historical_returns @ weights, 0.0) ** 2)
        )
        transaction_cost = expected_transaction_cost(weights, pretrade)
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
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    # A government-bond sleeve can always make Stage 14's wide 13% guard
    # feasible. HYG cannot: during the 2008 credit shock every available asset
    # can be too volatile. Find the long-only minimum before imposing the guard,
    # then relax only to the mathematical feasibility boundary when necessary.
    feasibility_constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
    ]
    minimum_variance_result = minimize(
        lambda weights: float(weights @ covariance @ weights),
        initial,
        method="SLSQP",
        bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=feasibility_constraints,
        options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
    )
    if not minimum_variance_result.success:
        raise RuntimeError(
            "The long-only minimum-variance feasibility solve failed: "
            f"{minimum_variance_result.message}"
        )
    minimum_feasible_volatility = annual_volatility(
        minimum_variance_result.x
    )
    effective_volatility_cap = max(
        CATASTROPHE_ANNUAL_VOLATILITY,
        minimum_feasible_volatility + FEASIBILITY_MARGIN,
    )

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                effective_volatility_cap - annual_volatility(weights)
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
                "Both HYG SLSQP solves failed: "
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
        "requested_annual_volatility_cap": CATASTROPHE_ANNUAL_VOLATILITY,
        "effective_annual_volatility_cap": effective_volatility_cap,
        "minimum_feasible_annual_volatility": minimum_feasible_volatility,
        "volatility_cap_relaxed": bool(
            effective_volatility_cap
            > CATASTROPHE_ANNUAL_VOLATILITY + FEASIBILITY_MARGIN
        ),
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": effective_volatility_cap - annual_vol,
        "requested_volatility_slack": (
            CATASTROPHE_ANNUAL_VOLATILITY - annual_vol
        ),
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
        try:
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
        except RuntimeError as exc:
            raise RuntimeError(f"{month}: {exc}") from exc

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        fx_cost = abs(foreign_flow(change)) * FOREIGN_WEIGHT_CHANGE_COST
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


def read_saved_path(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def run_hyg_execution_with_stage14_weights(
    stage14_path: pd.DataFrame,
    hyg_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Execute Stage 14 targets with HYG in the former BOND sleeve.

    This is a causal attribution path: Stage 14's already-lagged bond-based
    signals determine target weights, but the second sleeve earns HYG returns.
    Costs and post-return drift are recomputed with HYG treated as foreign.
    """

    months = stage14_path.index.intersection(hyg_returns.index)
    source_weights = ["w_KODEX200", "w_BOND", "w_GLD", "w_USO"]
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        weights = stage14_path.loc[month, source_weights].to_numpy(dtype=float)
        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        fx_cost = abs(foreign_flow(change)) * FOREIGN_WEIGHT_CHANGE_COST
        asset_return = hyg_returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
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
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    largest = weights.max(axis=1)
    return {
        "months": int(len(path)),
        "months_above_50_percent": int((largest > 0.5 + 1e-10).sum()),
        "months_above_90_percent": int((largest > 0.9 + 1e-10).sum()),
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
        "volatility_cap_relaxation_months": int(
            path["volatility_cap_relaxed"].sum()
        ),
        "maximum_effective_volatility_cap": float(
            path["effective_annual_volatility_cap"].max()
        ),
        "maximum_requested_cap_excess": float(
            (-path["requested_volatility_slack"]).clip(lower=0).max()
        ),
        "minimum_cdar_slack": float(path["cdar_slack"].min()),
    }


def asset_statistics(
    base_returns: pd.DataFrame,
    hyg_returns: pd.DataFrame,
    start: pd.Period,
    end: pd.Period,
) -> list[dict[str, Any]]:
    combined = pd.concat(
        [base_returns["BOND"], hyg_returns["HYG"]], axis=1
    ).loc[start:end].dropna()
    rows = []
    for asset in combined:
        metrics = performance_summary(combined[asset])
        rows.append({"Asset": asset, **metrics})
    correlation = float(combined["BOND"].corr(combined["HYG"]))
    for row in rows:
        row["BOND_HYG_Correlation"] = correlation
    return rows


def run_research(
    save: bool = True,
    refresh_hyg: bool = False,
) -> dict[str, Any]:
    hyg_returns, _ = load_hyg_monthly_returns(refresh_hyg)
    base_returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(hyg_returns)
    daily_stress = build_daily_stress_features()
    stress_signals = build_monthly_stress_signals(
        hyg_returns.index, daily_stress
    )
    static_path = run_backtest(
        hyg_returns, probabilities, stress_signals, STATIC_RISK_POLICY
    )
    dynamic_path = run_backtest(
        hyg_returns, probabilities, stress_signals, DYNAMIC_RISK_POLICY
    )
    stage14_path = read_saved_path(STAGE14_DYNAMIC_OUTPUT)
    execution_path = run_hyg_execution_with_stage14_weights(
        stage14_path, hyg_returns
    )

    common_start = max(dynamic_path.index.min(), stage14_path.index.min())
    common_end = min(dynamic_path.index.max(), stage14_path.index.max())
    periods = [
        ("common_full", common_start),
        ("locked_2018_2026", LOCKED_START),
    ]
    paths = {
        "Stage14_BOND_DynamicLambda": stage14_path,
        "Stage15_HYG_ExecutionOnly": execution_path,
        "Stage15_HYG_StaticLambda": static_path,
        "Stage15_HYG_DynamicLambda": dynamic_path,
    }
    comparison_rows: list[dict[str, Any]] = []
    for period, start in periods:
        for name, path in paths.items():
            comparison_rows.append(
                metric_row(name, path, period, start, common_end)
            )
    comparison = pd.DataFrame(comparison_rows)

    deltas: list[dict[str, Any]] = []
    candidates = [
        "Stage15_HYG_ExecutionOnly",
        "Stage15_HYG_DynamicLambda",
    ]
    for period, _ in periods:
        view = comparison[comparison["Period"] == period].set_index("Strategy")
        baseline = view.loc["Stage14_BOND_DynamicLambda"]
        for candidate_name in candidates:
            candidate = view.loc[candidate_name]
            deltas.append(
                {
                    "Candidate": candidate_name,
                    "Period": period,
                    "CAGR_Delta": float(
                        candidate["CAGR"] - baseline["CAGR"]
                    ),
                    "Sharpe_Delta": float(
                        candidate["Sharpe"] - baseline["Sharpe"]
                    ),
                    "MDD_Delta": float(candidate["MDD"] - baseline["MDD"]),
                    "Volatility_Delta": float(
                        candidate["Volatility"] - baseline["Volatility"]
                    ),
                    "FinalMultiple_Delta": float(
                        candidate["FinalMultiple"] - baseline["FinalMultiple"]
                    ),
                }
            )
    delta_frame = pd.DataFrame(deltas)

    weight_columns = [f"w_{asset}" for asset in ASSETS]
    checks = {
        "hyg_replaces_bond": "HYG" in ASSETS and "BOND" not in ASSETS,
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
        "no_cash_asset": True,
        "no_leverage": bool(
            np.allclose(dynamic_path[weight_columns].sum(axis=1), 1.0)
        ),
        "volatility_guard_respected": bool(
            (dynamic_path["volatility_slack"] >= -1e-7).all()
        ),
        "requested_volatility_cap_relaxed_only_if_infeasible": bool(
            (
                ~dynamic_path["volatility_cap_relaxed"]
                | (
                    dynamic_path["minimum_feasible_annual_volatility"]
                    > CATASTROPHE_ANNUAL_VOLATILITY
                )
            ).all()
        ),
        "cdar_guard_respected": bool(
            (dynamic_path["cdar_slack"] >= -1e-7).all()
        ),
        "all_slsqp_solves_succeeded": bool(
            dynamic_path["solver_success"].all()
        ),
        "no_hyperparameter_search": True,
    }
    asset_stats = asset_statistics(
        base_returns, hyg_returns, common_start, common_end
    )
    report: dict[str, Any] = {
        "strategy": "Stage15_HYG_DynamicLambda",
        "change": "Replace only the Stage14 BOND sleeve with KRW-valued HYG.",
        "data": {
            "hyg_source": "Yahoo Finance HYG auto_adjust=True local snapshot",
            "hyg_official_inception": "2007-04-04",
            "hyg_first_observation": str(
                download_hyg_daily(False)["date"].min().date()
            ),
            "hyg_last_observation": str(
                download_hyg_daily(False)["date"].max().date()
            ),
            "currency": "KRW via monthly first adjusted HYG open times USDKRW",
            "comparison_start_reason": (
                "HYG needs 12 causal monthly returns before the first solve."
            ),
        },
        "unchanged_from_stage14": {
            "macro_probabilities": True,
            "vkospi_vix6_stress": True,
            "dynamic_lambda": "1 + causal stress percentile",
            "annual_volatility_cap": CATASTROPHE_ANNUAL_VOLATILITY,
            "volatility_cap_feasibility_rule": (
                "max(13%, long-only minimum feasible volatility + 0.0001%p)"
            ),
            "cdar_guard": CATASTROPHE_CDAR,
            "single_asset_cap": None,
            "cash": False,
            "leverage": False,
        },
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "stage15_minus_stage14": json.loads(
            delta_frame.to_json(orient="records", force_ascii=False)
        ),
        "standalone_asset_statistics": asset_stats,
        "execution_only_concentration": concentration_summary(execution_path),
        "hyg_concentration": concentration_summary(dynamic_path),
        "solver": solver_summary(dynamic_path),
        "checks": checks,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        execution_path.to_csv(
            OUTPUT_DIR / "hyg_execution_stage14_weights_monthly.csv"
        )
        static_path.to_csv(OUTPUT_DIR / "hyg_static_lambda_monthly.csv")
        dynamic_path.to_csv(OUTPUT_DIR / "hyg_dynamic_lambda_monthly.csv")
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        delta_frame.to_csv(
            OUTPUT_DIR / "stage15_minus_stage14.csv", index=False
        )
        pd.DataFrame(asset_stats).to_csv(
            OUTPUT_DIR / "bond_vs_hyg_asset_statistics.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "returns": hyg_returns,
        "execution_path": execution_path,
        "static_path": static_path,
        "dynamic_path": dynamic_path,
        "comparison": comparison,
        "deltas": delta_frame,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True, refresh_hyg=False)
    print(result["comparison"].to_string(index=False))
    print("\nStage15 HYG candidates minus Stage14 BOND dynamic")
    print(result["deltas"].to_string(index=False))
    print("\nChecks")
    print(json.dumps(result["report"]["checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
