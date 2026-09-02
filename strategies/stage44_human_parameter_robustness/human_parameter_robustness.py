from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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

BASELINE_NAME = "Stage36_Frozen"
CORE_BOTH = "Stage44_PF_BothGuards"
CORE_CDAR_ONLY = "Stage44_PF_CDaROnly"
CORE_VOL_ONLY = "Stage44_PF_VolOnly"
CORE_NO_GUARDS = "Stage44_PF_NoGuards"

ORIGINAL_VOL_CAP = stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
ORIGINAL_CDAR_LIMIT = stage36.stage35.CATASTROPHE_CDAR
ORIGINAL_CDAR_CONFIDENCE = stage36.stage35.CDAR_CONFIDENCE
CONSTRAINT_TOLERANCE = 1e-7

# These are reporting grids, declared before results and never used to select a
# promoted strategy.  Round neighboring policy values test local smoothness.
VOL_SENSITIVITY = (0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
CDAR_LIMIT_SENSITIVITY = (0.12, 0.14, 0.16, 0.18, 0.20)
CDAR_CONFIDENCE_SENSITIVITY = (0.80, 0.85, 0.90, 0.95)

FROZEN_STAGE36_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
)


@dataclass(frozen=True)
class PortfolioSpecification:
    name: str
    volatility_cap: float | None
    cdar_limit: float | None
    cdar_confidence: float = ORIGINAL_CDAR_CONFIDENCE
    family: str = "core_ablation"
    displayed_value: float | None = None


CORE_SPECIFICATIONS = (
    PortfolioSpecification(CORE_BOTH, ORIGINAL_VOL_CAP, ORIGINAL_CDAR_LIMIT),
    PortfolioSpecification(CORE_CDAR_ONLY, None, ORIGINAL_CDAR_LIMIT),
    PortfolioSpecification(CORE_VOL_ONLY, ORIGINAL_VOL_CAP, None),
    PortfolioSpecification(CORE_NO_GUARDS, None, None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_stage36_manifest() -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in FROZEN_STAGE36_FILES}


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
    """Reproduce Stage36's mu and Sigma without changing its information set."""

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

    return expected_return, covariance, {
        "eps_mu_adjustment_KODEX200": eps_mu,
        "valuation_mu_adjustment_KODEX200": valuation_mu,
        "credit_stress_confirmation_multiplier": credit_stress_multiplier,
        "credit_equity_variance_multiplier": credit_variance_multiplier,
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
    }


def prepare_month_contexts(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build each causal Stage36 forecast once for all robustness paths."""

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

    contexts: list[dict[str, Any]] = []
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < stage36.stage35.ONE_CALENDAR_YEAR:
            continue
        historical_stress = stress_signals.loc[
            stress_signals.index < month, "stress_score"
        ]
        historical_recovery = stress_signals.loc[
            stress_signals.index < month, "recovery_score"
        ]
        expected_return, covariance, detail = build_stage36_forecast(
            history,
            probabilities.loc[probabilities.index < month],
            probabilities.loc[month],
            historical_stress,
            float(stress_signals.loc[month, "stress_score"]),
            historical_recovery,
            float(stress_signals.loc[month, "recovery_score"]),
            technical_signals.loc[month],
            fundamental_signals.loc[month],
            asset_vol_signals.loc[month],
        )
        common = history.index.intersection(historical_stress.dropna().index)
        contexts.append(
            {
                "month": month,
                "history_end_month": history.index.max(),
                "expected_return": expected_return,
                "covariance": covariance,
                "historical_returns": history.loc[
                    common, ASSETS
                ].to_numpy(dtype=float),
                "asset_return": returns.loc[month, ASSETS].to_numpy(dtype=float),
                "forecast_detail": detail,
            }
        )
    return contexts


def portfolio_values(
    weights: np.ndarray,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_returns: np.ndarray,
    pretrade: np.ndarray,
    cdar_confidence: float,
) -> dict[str, float]:
    """Parameter-free objective plus diagnostics, all in monthly units."""

    monthly_return = float(weights @ expected_return)
    monthly_variance = max(float(weights @ covariance @ weights), 0.0)
    scenario_returns = historical_returns @ weights
    downside_semivariance = float(
        np.mean(np.minimum(scenario_returns, 0.0) ** 2)
    )
    transaction_cost = stage36.stage35.expected_transaction_cost(
        weights, pretrade
    )
    log_growth = monthly_return - 0.5 * monthly_variance - transaction_cost
    return {
        "expected_monthly_return": monthly_return,
        "expected_monthly_variance": monthly_variance,
        "expected_annual_volatility": math.sqrt(monthly_variance * 12.0),
        "expected_monthly_log_growth_net": log_growth,
        "expected_annual_log_growth_net": 12.0 * log_growth,
        "estimated_transaction_cost": transaction_cost,
        "downside_semivariance_diagnostic": downside_semivariance,
        "downside_semivariance_coefficient_in_objective": 0.0,
        "historical_cdar": stage36.stage35.cdar(
            scenario_returns, cdar_confidence
        ),
    }


def solve_weights(
    context: dict[str, Any],
    pretrade: np.ndarray,
    specification: PortfolioSpecification,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use Stage36's SLSQP mechanics with only its semivariance term removed."""

    expected_return = context["expected_return"]
    covariance = context["covariance"]
    historical_returns = context["historical_returns"]
    initial = (
        stage36.stage35.project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def values(weights: np.ndarray) -> dict[str, float]:
        return portfolio_values(
            weights,
            expected_return,
            covariance,
            historical_returns,
            pretrade,
            specification.cdar_confidence,
        )

    def objective(weights: np.ndarray) -> float:
        return -values(weights)["expected_monthly_log_growth_net"]

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
    ]
    if specification.volatility_cap is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: (
                    specification.volatility_cap
                    - math.sqrt(
                        max(float(weights @ covariance @ weights), 0.0) * 12.0
                    )
                ),
            }
        )
    if specification.cdar_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: (
                    specification.cdar_limit
                    + stage36.stage35.cdar(
                        historical_returns @ weights,
                        specification.cdar_confidence,
                    )
                ),
            }
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={
            "maxiter": stage36.stage35.SLSQP_MAX_ITERATIONS,
            "ftol": stage36.stage35.SLSQP_TOLERANCE,
        },
    )
    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = stage36.stage35.project_to_long_only_simplex(result.x)
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": stage36.stage35.SLSQP_MAX_ITERATIONS,
                "ftol": stage36.stage35.SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                f"Stage44 and minimum-variance fallback failed for "
                f"{specification.name}: {result.message}; {fallback.message}"
            )
        result = fallback
        weights = stage36.stage35.project_to_long_only_simplex(fallback.x)
        used_fallback = True

    statistics = values(weights)
    volatility_slack = (
        specification.volatility_cap - statistics["expected_annual_volatility"]
        if specification.volatility_cap is not None
        else np.nan
    )
    cdar_slack = (
        specification.cdar_limit + statistics["historical_cdar"]
        if specification.cdar_limit is not None
        else np.nan
    )
    if (
        specification.volatility_cap is not None
        and volatility_slack < -CONSTRAINT_TOLERANCE
    ) or (
        specification.cdar_limit is not None
        and cdar_slack < -CONSTRAINT_TOLERANCE
    ):
        raise RuntimeError(
            f"Solver returned an infeasible solution for {specification.name}."
        )

    return weights, {
        **statistics,
        "policy": specification.name,
        "family": specification.family,
        "displayed_value": specification.displayed_value,
        "objective": "parameter_free_expected_log_growth",
        "volatility_cap": specification.volatility_cap,
        "cdar_limit": specification.cdar_limit,
        "cdar_confidence": specification.cdar_confidence,
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": volatility_slack,
        "cdar_slack": cdar_slack,
    }


def run_backtest(
    contexts: list[dict[str, Any]], specification: PortfolioSpecification
) -> pd.DataFrame:
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    rows: list[dict[str, Any]] = []
    for context in contexts:
        weights, detail = solve_weights(context, pretrade, specification)
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
        asset_return = context["asset_return"]
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": context["month"],
                "signal_cutoff_month": context["month"] - 1,
                "history_end_month": context["history_end_month"],
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
                **context["forecast_detail"],
                **detail,
            }
        )
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "full_2007_2026": (FULL_START, RESEARCH_END),
        "common_2010_2026": (COMMON_START, RESEARCH_END),
        "locked_2018_2026": (LOCKED_START, RESEARCH_END),
    }
    for name, path in paths.items():
        for period, (start, end) in periods.items():
            rows.append(stage36.stage35.metric_row(name, path, period, start, end))
    return pd.DataFrame(rows)


def constraint_summary(
    specifications: dict[str, PortfolioSpecification],
    paths: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, specification in specifications.items():
        path = paths[name]
        rows.append(
            {
                "Strategy": name,
                "Family": specification.family,
                "DisplayedValue": specification.displayed_value,
                "VolatilityCap": specification.volatility_cap,
                "CDaRLimit": specification.cdar_limit,
                "CDaRConfidence": specification.cdar_confidence,
                "Months": len(path),
                "SolverSuccesses": int(path["solver_success"].sum()),
                "Fallbacks": int(path["used_fallback"].sum()),
                "VolatilityBindingMonths": (
                    int(path["volatility_slack"].abs().lt(1e-6).sum())
                    if specification.volatility_cap is not None
                    else 0
                ),
                "CDaRBindingMonths": (
                    int(path["cdar_slack"].abs().lt(1e-6).sum())
                    if specification.cdar_limit is not None
                    else 0
                ),
                "MinimumVolatilitySlack": (
                    float(path["volatility_slack"].min())
                    if specification.volatility_cap is not None
                    else np.nan
                ),
                "MinimumCDaRSlack": (
                    float(path["cdar_slack"].min())
                    if specification.cdar_limit is not None
                    else np.nan
                ),
                "MaximumWeightSumError": float(path["sum_error"].max()),
                "AllHistoryBeforeTarget": bool(
                    (path["history_end_month"] < path.index).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def weight_path_comparison(
    baseline: pd.DataFrame,
    core_paths: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    references = {BASELINE_NAME: baseline, CORE_BOTH: core_paths[CORE_BOTH]}
    for name, candidate in core_paths.items():
        reference_name = BASELINE_NAME if name == CORE_BOTH else CORE_BOTH
        reference = references[reference_name]
        common = reference.index.intersection(candidate.index)
        difference = (
            candidate.loc[common, WEIGHT_COLUMNS]
            - reference.loc[common, WEIGHT_COLUMNS]
        ).abs()
        rows.append(
            {
                "Candidate": name,
                "Reference": reference_name,
                "MeanAbsoluteWeightDifference": float(
                    difference.to_numpy().mean()
                ),
                "MaximumAbsoluteWeightDifference": float(
                    difference.to_numpy().max()
                ),
                "ReturnCorrelation": float(
                    candidate.loc[common, "return"].corr(
                        reference.loc[common, "return"]
                    )
                ),
                "MeanTurnover": float(candidate.loc[common, "turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_table(
    baseline: pd.DataFrame, core_paths: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for period, start in (
        ("full_2007_2026", FULL_START),
        ("common_2010_2026", COMMON_START),
    ):
        for name, path in core_paths.items():
            summary = stage36.stage35.stage30.paired_block_bootstrap(
                baseline.loc[start:RESEARCH_END, "return"],
                path.loc[start:RESEARCH_END, "return"],
            )
            summary.insert(0, "Period", period)
            summary.insert(0, "Candidate", name)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def sensitivity_summary(performance: pd.DataFrame) -> pd.DataFrame:
    full = performance.loc[performance["Period"].eq("full_2007_2026")].copy()
    rows: list[dict[str, Any]] = []
    for family, group in full.groupby("Family", sort=False):
        if not str(family).endswith("sensitivity"):
            continue
        row: dict[str, Any] = {
            "Family": family,
            "Paths": len(group),
            "MinimumTestedValue": float(group["DisplayedValue"].min()),
            "MaximumTestedValue": float(group["DisplayedValue"].max()),
            "SelectionUse": "report-only; no best path adopted",
        }
        for metric in ("CAGR", "Sharpe", "MDD", "AvgTurnover"):
            row[f"{metric}Min"] = float(group[metric].min())
            row[f"{metric}Max"] = float(group[metric].max())
            row[f"{metric}Range"] = float(group[metric].max() - group[metric].min())
        rows.append(row)
    return pd.DataFrame(rows)


def allocation_summary(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        weights = path[WEIGHT_COLUMNS]
        largest = weights.max(axis=1)
        row: dict[str, Any] = {
            "Strategy": name,
            "MaximumSingleAssetWeight": float(largest.max()),
            "MonthsAbove50Percent": int((largest > 0.50 + 1e-10).sum()),
            "MonthsAbove90Percent": int((largest > 0.90 + 1e-10).sum()),
        }
        row.update(
            {
                f"AverageWeight_{asset}": float(weights[f"w_{asset}"].mean())
                for asset in ASSETS
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_reproduction_audit(
    contexts: list[dict[str, Any]], baseline: pd.DataFrame
) -> dict[str, Any]:
    """Prove the copied forecast layer matches saved Stage36 portfolio moments."""

    return_errors: list[float] = []
    variance_errors: list[float] = []
    audited_months: list[str] = []
    for context in contexts:
        month = context["month"]
        if month not in baseline.index:
            continue
        weights = baseline.loc[month, WEIGHT_COLUMNS].to_numpy(dtype=float)
        reproduced_return = float(weights @ context["expected_return"])
        reproduced_variance = float(
            weights @ context["covariance"] @ weights
        )
        return_errors.append(
            abs(
                reproduced_return
                - float(baseline.loc[month, "expected_monthly_return"])
            )
        )
        variance_errors.append(
            abs(
                reproduced_variance
                - float(baseline.loc[month, "expected_monthly_variance"])
            )
        )
        audited_months.append(str(month))
    return {
        "months": len(audited_months),
        "first_month": min(audited_months),
        "last_month": max(audited_months),
        "maximum_absolute_expected_return_error": max(return_errors),
        "maximum_absolute_expected_variance_error": max(variance_errors),
    }


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Element": "variance coefficient 0.5",
                "Classification": "mathematically derived",
                "Test": "retained",
                "Reason": "second-order expansion of expected log growth",
            },
            {
                "Element": "downside semivariance coefficient 1",
                "Classification": "human risk preference",
                "Test": "removed in every Stage44 path",
                "Reason": "clean objective ablation against frozen Stage36",
            },
            {
                "Element": "annual volatility cap 13%",
                "Classification": "risk-governance choice",
                "Test": "remove and 10%-15% sensitivity",
                "Reason": "not used for model selection",
            },
            {
                "Element": "CDaR limit -16%",
                "Classification": "risk-governance choice",
                "Test": "remove and -12% to -20% sensitivity",
                "Reason": "not used for model selection",
            },
            {
                "Element": "CDaR confidence 90%",
                "Classification": "tail-definition choice",
                "Test": "80%, 85%, 90%, 95% sensitivity",
                "Reason": "not used for model selection",
            },
            {
                "Element": "transaction costs 15bp + 5bp",
                "Classification": "implementation assumption",
                "Test": "retained",
                "Reason": "actual return deduction, not a fitted utility weight",
            },
        ]
    )


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
    fundamental = stage36.stage35.build_monthly_fundamental_signals(raw_fundamental)
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
    baseline = stage36.stage35.stage34._load_period_csv(
        stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv", "month"
    )
    contexts = prepare_month_contexts(
        returns, probabilities, stress, technical, calibrated, asset_vol_signals
    )

    specifications = {item.name: item for item in CORE_SPECIFICATIONS}
    paths = {name: run_backtest(contexts, spec) for name, spec in specifications.items()}
    core_paths = dict(paths)

    sensitivity_groups: dict[str, list[str]] = {
        "volatility_cap_sensitivity": [],
        "cdar_limit_sensitivity": [],
        "cdar_confidence_sensitivity": [],
    }

    def add_sensitivity(specification: PortfolioSpecification) -> None:
        specifications[specification.name] = specification
        if (
            specification.volatility_cap == ORIGINAL_VOL_CAP
            and specification.cdar_limit == ORIGINAL_CDAR_LIMIT
            and specification.cdar_confidence == ORIGINAL_CDAR_CONFIDENCE
        ):
            alias = paths[CORE_BOTH].copy()
            alias["policy"] = specification.name
            alias["family"] = specification.family
            alias["displayed_value"] = specification.displayed_value
            paths[specification.name] = alias
        else:
            paths[specification.name] = run_backtest(contexts, specification)
        sensitivity_groups[specification.family].append(specification.name)

    for value in VOL_SENSITIVITY:
        add_sensitivity(
            PortfolioSpecification(
                f"Stage44_VolCap_{int(round(value * 100)):02d}pct",
                value,
                ORIGINAL_CDAR_LIMIT,
                family="volatility_cap_sensitivity",
                displayed_value=value,
            )
        )
    for value in CDAR_LIMIT_SENSITIVITY:
        add_sensitivity(
            PortfolioSpecification(
                f"Stage44_CDaRLimit_{int(round(value * 100)):02d}pct",
                ORIGINAL_VOL_CAP,
                value,
                family="cdar_limit_sensitivity",
                displayed_value=value,
            )
        )
    for value in CDAR_CONFIDENCE_SENSITIVITY:
        add_sensitivity(
            PortfolioSpecification(
                f"Stage44_CDaRConfidence_{int(round(value * 100)):02d}pct",
                ORIGINAL_VOL_CAP,
                ORIGINAL_CDAR_LIMIT,
                cdar_confidence=value,
                family="cdar_confidence_sensitivity",
                displayed_value=value,
            )
        )

    all_performance = performance_table({BASELINE_NAME: baseline, **paths})
    core_performance = all_performance.loc[
        all_performance["Strategy"].isin([BASELINE_NAME, *core_paths])
    ].copy()
    sensitivity_performance = all_performance.loc[
        ~all_performance["Strategy"].isin([BASELINE_NAME, *core_paths])
    ].copy()
    for name, specification in specifications.items():
        mask = all_performance["Strategy"].eq(name)
        all_performance.loc[mask, "Family"] = specification.family
        all_performance.loc[mask, "DisplayedValue"] = specification.displayed_value
        mask_core = core_performance["Strategy"].eq(name)
        core_performance.loc[mask_core, "Family"] = specification.family
        mask_sensitivity = sensitivity_performance["Strategy"].eq(name)
        sensitivity_performance.loc[
            mask_sensitivity, "Family"
        ] = specification.family
        sensitivity_performance.loc[
            mask_sensitivity, "DisplayedValue"
        ] = specification.displayed_value

    bindings = constraint_summary(specifications, paths)
    weights = weight_path_comparison(baseline, core_paths)
    bootstrap = bootstrap_table(baseline, core_paths)
    sensitivity = sensitivity_summary(sensitivity_performance)
    allocations = allocation_summary({BASELINE_NAME: baseline, **core_paths})
    forecast_audit = forecast_reproduction_audit(contexts, baseline)
    inventory = _inventory()
    stage36_after = frozen_stage36_manifest()

    full = core_performance.loc[
        core_performance["Period"].eq("full_2007_2026")
    ].set_index("Strategy")
    report = {
        "stage": 44,
        "study": "Stage36 human-parameter robustness",
        "decision": "robustness evidence only; no alternative path promoted",
        "frozen_baseline": BASELINE_NAME,
        "objective_change": (
            "removed downside semivariance coefficient 1; retained "
            "w'mu - 0.5*w'Sigma*w - actual transaction cost"
        ),
        "core_ablation_design": {
            CORE_BOTH: "13% volatility cap and -16% CDaR guard",
            CORE_CDAR_ONLY: "remove volatility cap only",
            CORE_VOL_ONLY: "remove CDaR guard only",
            CORE_NO_GUARDS: "remove both human risk-policy guards",
        },
        "sensitivity_design": {
            "volatility_caps": VOL_SENSITIVITY,
            "cdar_limits": CDAR_LIMIT_SENSITIVITY,
            "cdar_confidences": CDAR_CONFIDENCE_SENSITIVITY,
            "selection_use": "none; original Stage36 values remain frozen",
        },
        "anti_overfit": {
            "future_return_in_optimizer": False,
            "best_sensitivity_path_promoted": False,
            "input_edge_changed": False,
            "mu_or_sigma_model_changed": False,
            "transaction_cost_changed": False,
            "threshold_results_used_for_selection": False,
        },
        "headline_full_period": {
            name: {
                metric: float(full.loc[name, metric])
                for metric in ("CAGR", "Sharpe", "MDD", "AvgTurnover")
            }
            for name in full.index
        },
        "stage36_frozen_files_unchanged": stage36_before == stage36_after,
        "stage36_forecast_reproduction_audit": forecast_audit,
        "all_paths_causal": bool(bindings["AllHistoryBeforeTarget"].all()),
        "all_solvers_successful": bool(
            (bindings["SolverSuccesses"] == bindings["Months"]).all()
        ),
        "total_fallbacks": int(bindings["Fallbacks"].sum()),
        "performance": json.loads(
            all_performance.to_json(orient="records", force_ascii=False)
        ),
        "constraint_summary": json.loads(
            bindings.to_json(orient="records", force_ascii=False)
        ),
        "weight_path_comparison": json.loads(
            weights.to_json(orient="records", force_ascii=False)
        ),
        "sensitivity_summary": json.loads(
            sensitivity.to_json(orient="records", force_ascii=False)
        ),
        "allocation_summary": json.loads(
            allocations.to_json(orient="records", force_ascii=False)
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
        for name, path in core_paths.items():
            path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        for family, names in sensitivity_groups.items():
            for name in names:
                paths[name].to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        core_performance.to_csv(
            OUTPUT_DIR / "core_ablation_performance.csv", index=False
        )
        sensitivity_performance.to_csv(
            OUTPUT_DIR / "sensitivity_performance.csv", index=False
        )
        all_performance.to_csv(
            OUTPUT_DIR / "all_performance.csv", index=False
        )
        bindings.to_csv(OUTPUT_DIR / "constraint_binding_summary.csv", index=False)
        weights.to_csv(OUTPUT_DIR / "weight_path_comparison.csv", index=False)
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv", index=False
        )
        sensitivity.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)
        allocations.to_csv(OUTPUT_DIR / "allocation_summary.csv", index=False)
        inventory.to_csv(OUTPUT_DIR / "human_parameter_inventory.csv", index=False)
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
