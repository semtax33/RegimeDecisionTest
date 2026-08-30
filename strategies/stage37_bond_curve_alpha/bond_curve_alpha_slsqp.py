from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import spearmanr

from strategies.stage36_asset_implied_volatility_risk import (
    asset_implied_volatility_risk_slsqp as stage36,
)


stage35 = stage36.stage35
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CURVE_XLSX = ROOT / "raw_data" / "260829_국고채.회사채.xlsx"

ASSETS = stage36.ASSETS
FULL_START = stage36.FULL_START
COMMON_START = stage36.COMMON_START
RESEARCH_END = stage36.RESEARCH_END
BOND_INDEX = ASSETS.index("BOND")
FIVE_YEAR_DURATION = 5.0
ROLL_MATURITY_MONTHS = 24.0
RATE_MOMENTUM_DAYS = 60
RATE_MOMENTUM_MONTHS = 3.0
MIN_CAUSAL_MONTHS = 60
SIGNIFICANCE_LEVEL = 0.10
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]

SOURCE_FILES = (CURVE_XLSX,)
FROZEN_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
)

MODES = {
    "Stage37_NoBondAlphaReproduction": "baseline_reproduction",
    "Stage37_CarryRollOnly": "carry_roll_only",
    "Stage37_RateMomentumOnly": "rate_momentum_only",
    "Stage37_BondCurveAlpha": "bond_curve_alpha",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def source_manifest() -> dict[str, dict[str, Any]]:
    return _manifest(SOURCE_FILES)


def frozen_manifest() -> dict[str, dict[str, Any]]:
    return _manifest(FROZEN_FILES)


def _causal_center(series: pd.Series) -> pd.Series:
    prior_mean = (
        series.shift(1).expanding(min_periods=MIN_CAUSAL_MONTHS).mean()
    )
    return series - prior_mean


def _causal_zscore(series: pd.Series) -> pd.Series:
    prior = series.shift(1)
    mean = prior.expanding(min_periods=MIN_CAUSAL_MONTHS).mean()
    std = prior.expanding(min_periods=MIN_CAUSAL_MONTHS).std(ddof=1)
    return ((series - mean) / std.where(std > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )


def load_treasury_curve_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_excel(
        CURVE_XLSX,
        header=None,
        skiprows=14,
        usecols=range(9),
        engine="openpyxl",
    )
    raw.columns = [
        "date",
        "ktb_1y_pct",
        "ktb_2y_pct",
        "ktb_3y_pct",
        "ktb_5y_pct",
        "ktb_10y_pct",
        "ktb_20y_pct",
        "ktb_30y_pct",
        "ktb_50y_pct",
    ]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    daily = raw.dropna(subset=["date"]).set_index("date").sort_index()
    daily = daily.apply(pd.to_numeric, errors="coerce")
    daily = daily.loc[~daily.index.duplicated(keep="last")]

    daily["carry_5y_monthly"] = daily["ktb_5y_pct"] / 1200.0
    daily["roll_down_5y_to_3y_monthly"] = (
        FIVE_YEAR_DURATION
        * (daily["ktb_5y_pct"] - daily["ktb_3y_pct"])
        / (ROLL_MATURITY_MONTHS * 100.0)
    )
    daily["rate_momentum_5y_monthly"] = (
        FIVE_YEAR_DURATION
        * (
            daily["ktb_5y_pct"].shift(RATE_MOMENTUM_DAYS)
            - daily["ktb_5y_pct"]
        )
        / (RATE_MOMENTUM_MONTHS * 100.0)
    )
    daily["curve_slope_10y_minus_3y_pctpt"] = (
        daily["ktb_10y_pct"] - daily["ktb_3y_pct"]
    )
    daily["carry_roll_proxy"] = (
        daily["carry_5y_monthly"]
        + daily["roll_down_5y_to_3y_monthly"]
    )
    daily["curve_total_return_proxy"] = (
        daily["carry_roll_proxy"] + daily["rate_momentum_5y_monthly"]
    )

    audit = {
        "curve_source": str(CURVE_XLSX.resolve()),
        "daily_rows": int(len(daily)),
        "first_date": str(daily.index.min().date()),
        "last_date": str(daily.index.max().date()),
        "ktb_3y_first_valid": str(
            daily["ktb_3y_pct"].dropna().index.min().date()
        ),
        "ktb_5y_first_valid": str(
            daily["ktb_5y_pct"].dropna().index.min().date()
        ),
        "ktb_10y_first_valid": str(
            daily["ktb_10y_pct"].dropna().index.min().date()
        ),
        "ktb_2y_first_valid": str(
            daily["ktb_2y_pct"].dropna().index.min().date()
        ),
        "two_year_excluded_reason": (
            "KTB 2Y begins in 2021; no backfill or synthetic 2Y history"
        ),
        "anchor_maturity_years": 5,
        "roll_down_shorter_maturity_years": 3,
        "duration_assumption_years": FIVE_YEAR_DURATION,
        "rate_momentum_observations": RATE_MOMENTUM_DAYS,
        "winsorization": False,
        "parameter_grid": False,
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit


def build_monthly_bond_curve_signals(daily: pd.DataFrame) -> pd.DataFrame:
    required = [
        "ktb_3y_pct",
        "ktb_5y_pct",
        "ktb_10y_pct",
        "carry_5y_monthly",
        "roll_down_5y_to_3y_monthly",
        "rate_momentum_5y_monthly",
        "curve_slope_10y_minus_3y_pctpt",
        "carry_roll_proxy",
        "curve_total_return_proxy",
    ]
    rows: list[dict[str, Any]] = []
    for signal_month, group in daily.groupby(daily.index.to_period("M")):
        complete = group.dropna(subset=required)
        if complete.empty:
            continue
        current = complete.iloc[-1]
        rows.append(
            {
                "target_month": signal_month + 1,
                "bond_curve_signal_month": signal_month,
                "bond_curve_signal_date": complete.index[-1],
                **{column: float(current[column]) for column in required},
            }
        )
    signals = pd.DataFrame(rows).set_index("target_month").sort_index()
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    signals["carry_roll_alpha_adjustment"] = _causal_center(
        signals["carry_roll_proxy"]
    )
    signals["rate_momentum_alpha_adjustment"] = _causal_center(
        signals["rate_momentum_5y_monthly"]
    )
    signals["bond_curve_alpha_adjustment"] = _causal_center(
        signals["curve_total_return_proxy"]
    )
    for column in (
        "carry_5y_monthly",
        "roll_down_5y_to_3y_monthly",
        "rate_momentum_5y_monthly",
        "curve_slope_10y_minus_3y_pctpt",
        "curve_total_return_proxy",
    ):
        signals[f"{column}_z"] = _causal_zscore(signals[column])
    return signals.replace([np.inf, -np.inf], np.nan)


def build_bond_research_frame(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    bond_close: pd.Series,
) -> pd.DataFrame:
    frame = signals.copy()
    for horizon in (1, 3, 6):
        frame[f"future_{horizon}m_bond_return"] = (
            stage35.stage34._forward_compound(returns["BOND"], horizon)
        )
    frame["recent_1m_bond_return"] = returns["BOND"].shift(1)
    frame["bond_realized_vol_21d"] = (
        stage35.stage34._realized_volatility_signal(bond_close)
    )
    frame["macro_fragility"] = (
        probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    )
    frame["vix6_stress_score"] = stress["stress_score"]
    return frame.loc[FULL_START:RESEARCH_END].replace(
        [np.inf, -np.inf], np.nan
    )


def _standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        std = float(frame[column].std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"No usable variation in {column}")
        output[column] = (frame[column] - frame[column].mean()) / std
    return output


def bond_predictive_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    features = {
        "Carry": "carry_5y_monthly_z",
        "RollDown": "roll_down_5y_to_3y_monthly_z",
        "RateMomentum": "rate_momentum_5y_monthly_z",
        "CurveComposite": "curve_total_return_proxy_z",
    }
    controls = [
        "recent_1m_bond_return",
        "bond_realized_vol_21d",
        "macro_fragility",
        "vix6_stress_score",
    ]
    periods = {
        "full_2007_2026": (FULL_START, RESEARCH_END),
        "common_2010_2026": (COMMON_START, RESEARCH_END),
        "locked_2018_2026": (stage35.LOCKED_START, RESEARCH_END),
    }
    rows: list[dict[str, Any]] = []
    for period_name, (start, end) in periods.items():
        view = frame.loc[start:end]
        for horizon in (1, 3, 6):
            target = f"future_{horizon}m_bond_return"
            for feature_name, feature in features.items():
                for model, predictors in (
                    ("FeatureOnly", [feature]),
                    ("FullControls", [feature, *controls]),
                ):
                    complete = view[[target, *predictors]].dropna()
                    if len(complete) < 36:
                        continue
                    standardized = _standardize(complete, predictors)
                    fit = sm.OLS(
                        complete[target], sm.add_constant(standardized)
                    ).fit(
                        cov_type="HAC",
                        cov_kwds={"maxlags": horizon},
                    )
                    ic, ic_p = spearmanr(
                        complete[feature],
                        complete[target],
                        nan_policy="omit",
                    )
                    rows.append(
                        {
                            "Feature": feature_name,
                            "Period": period_name,
                            "HorizonMonths": horizon,
                            "Model": model,
                            "Observations": int(len(complete)),
                            "StandardizedBeta": float(fit.params[feature]),
                            "HACPValue": float(fit.pvalues[feature]),
                            "SpearmanIC": float(ic),
                            "ICPValue": float(ic_p),
                            "AdjustedR2": float(fit.rsquared_adj),
                        }
                    )
    return pd.DataFrame(rows)


def _bond_mu_adjustment(signal: pd.Series, mode: str) -> float:
    if mode == "carry_roll_only":
        return float(signal["carry_roll_alpha_adjustment"])
    if mode == "rate_momentum_only":
        return float(signal["rate_momentum_alpha_adjustment"])
    if mode == "bond_curve_alpha":
        return float(signal["bond_curve_alpha_adjustment"])
    return 0.0


def _solve_weights(
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
    bond_curve_signal: pd.Series,
    pretrade: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen Stage36 combined path plus one BOND expected-return adjustment."""

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
    stress_adjustment[stage35.EQUITY_INDEX] *= credit_stress_multiplier
    filtered_macro[stage35.EQUITY_INDEX] += eps_mu + valuation_mu

    bond_mu = _bond_mu_adjustment(bond_curve_signal, mode)
    filtered_macro[BOND_INDEX] += bond_mu
    expected_return = filtered_macro + stress_adjustment

    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)
    credit_variance_multiplier = 1.0 + float(
        fundamental_signal["credit_stress_rank"]
    )
    credit_scaling = np.eye(len(ASSETS), dtype=float)
    credit_scaling[stage35.EQUITY_INDEX, stage35.EQUITY_INDEX] = math.sqrt(
        credit_variance_multiplier
    )
    covariance = credit_scaling @ covariance @ credit_scaling

    gvz_multiplier = float(
        asset_vol_signal["gvz_gld_variance_multiplier"]
    )
    ovx_multiplier = float(
        asset_vol_signal["ovx_uso_variance_multiplier"]
    )
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
    initial = (
        stage35.project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        realized_history = historical_returns @ weights
        downside_semivariance = float(
            np.mean(np.minimum(realized_history, 0.0) ** 2)
        )
        transaction_cost = stage35.expected_transaction_cost(
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
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_risk_aversion_lambda": 1.0,
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": utility,
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
                stage35.CATASTROPHE_ANNUAL_VOLATILITY
                - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage35.CATASTROPHE_CDAR
                + stage35.cdar(
                    historical_returns @ weights,
                    stage35.CDAR_CONFIDENCE,
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
                f"Stage37 and fallback solves failed: {result.message}; "
                f"{fallback.message}"
            )
        result = fallback
        weights = stage35.project_to_long_only_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = stage35.cdar(
        historical_returns @ weights, stage35.CDAR_CONFIDENCE
    )
    detail = {
        **values,
        "policy": f"Stage37_{mode}",
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": (
            stage35.CATASTROPHE_ANNUAL_VOLATILITY - annual_vol
        ),
        "cdar_slack": stage35.CATASTROPHE_CDAR + historical_cdar,
        "bond_mu_adjustment": bond_mu,
        "carry_roll_alpha_adjustment": float(
            bond_curve_signal["carry_roll_alpha_adjustment"]
        ),
        "rate_momentum_alpha_adjustment": float(
            bond_curve_signal["rate_momentum_alpha_adjustment"]
        ),
        "bond_curve_alpha_adjustment": float(
            bond_curve_signal["bond_curve_alpha_adjustment"]
        ),
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
        "gvz_mu_adjustment_GLD": 0.0,
        "ovx_mu_adjustment_USO": 0.0,
        "expected_mu_BOND": float(expected_return[BOND_INDEX]),
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    bond_curve_signals: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
        "credit_stress_rank",
    ]
    required_bond = [
        "carry_roll_alpha_adjustment",
        "rate_momentum_alpha_adjustment",
        "bond_curve_alpha_adjustment",
    ]
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(asset_vol_signals.index)
    months = months.intersection(
        fundamental_signals.dropna(subset=required_fundamental).index
    )
    months = months.intersection(
        bond_curve_signals.dropna(subset=required_bond).index
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
        bond_curve_signal = bond_curve_signals.loc[month]
        weights, detail = _solve_weights(
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
            bond_curve_signal,
            pretrade,
            mode,
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
                "bond_curve_signal_month": bond_curve_signal[
                    "bond_curve_signal_month"
                ],
                "bond_curve_signal_date": bond_curve_signal[
                    "bond_curve_signal_date"
                ],
                "ktb_3y_pct": float(bond_curve_signal["ktb_3y_pct"]),
                "ktb_5y_pct": float(bond_curve_signal["ktb_5y_pct"]),
                "ktb_10y_pct": float(bond_curve_signal["ktb_10y_pct"]),
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


def _performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_end = min(path.index.max() for path in paths.values())
    periods = {
        "full_2007_2026": (FULL_START, common_end),
        "common_2010_2026": (COMMON_START, common_end),
        "locked_2018_2026": (stage35.LOCKED_START, common_end),
    }
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        for period_name, (start, end) in periods.items():
            rows.append(
                stage35.metric_row(name, path, period_name, start, end)
            )
    return pd.DataFrame(rows)


def _bootstrap_table(
    baseline: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for period_name, start in (
        ("full_2007_2026", FULL_START),
        ("common_2010_2026", COMMON_START),
    ):
        for name, candidate in candidates.items():
            summary = stage35.stage30.paired_block_bootstrap(
                baseline.loc[start:RESEARCH_END, "return"],
                candidate.loc[start:RESEARCH_END, "return"],
            )
            summary.insert(0, "Period", period_name)
            summary.insert(0, "Candidate", name)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def gate_decision(
    performance: pd.DataFrame,
    regressions: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    perf = performance.set_index(["Strategy", "Period"])
    baseline_name = "Stage36_Frozen"
    candidate_name = "Stage37_BondCurveAlpha"
    full_base = perf.loc[(baseline_name, "full_2007_2026")]
    full_test = perf.loc[(candidate_name, "full_2007_2026")]
    common_base = perf.loc[(baseline_name, "common_2010_2026")]
    common_test = perf.loc[(candidate_name, "common_2010_2026")]
    locked_base = perf.loc[(baseline_name, "locked_2018_2026")]
    locked_test = perf.loc[(candidate_name, "locked_2018_2026")]
    performance_gates = {
        "full_cagr_higher": bool(full_test["CAGR"] > full_base["CAGR"]),
        "full_sharpe_not_lower": bool(
            full_test["Sharpe"] >= full_base["Sharpe"]
        ),
        "full_mdd_not_worse_by_more_than_50bp": bool(
            full_test["MDD"] >= full_base["MDD"] - 0.005
        ),
        "common_cagr_higher": bool(
            common_test["CAGR"] > common_base["CAGR"]
        ),
        "common_sharpe_not_lower": bool(
            common_test["Sharpe"] >= common_base["Sharpe"]
        ),
        "locked_cagr_higher": bool(
            locked_test["CAGR"] > locked_base["CAGR"]
        ),
    }
    mechanism = regressions.loc[
        regressions["Feature"].eq("CurveComposite")
        & regressions["Period"].eq("common_2010_2026")
        & regressions["Model"].eq("FullControls")
        & regressions["HorizonMonths"].isin([1, 3])
    ]
    mechanism_gates = {
        "one_and_three_month_betas_positive": bool(
            mechanism["StandardizedBeta"].gt(0.0).all()
        ),
        "at_least_one_hac_p_below_10pct": bool(
            (
                mechanism["StandardizedBeta"].gt(0.0)
                & mechanism["HACPValue"].lt(SIGNIFICANCE_LEVEL)
            ).any()
        ),
    }
    boot = bootstrap.loc[
        bootstrap["Candidate"].eq(candidate_name)
        & bootstrap["Period"].eq("common_2010_2026")
    ].set_index("Metric")
    bootstrap_gates = {
        "cagr_improvement_probability_at_least_60pct": bool(
            boot.loc["delta_CAGR", "ProbabilityPositive"] >= 0.60
        ),
        "sharpe_improvement_probability_at_least_50pct": bool(
            boot.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.50
        ),
    }
    promote = bool(
        all(performance_gates.values())
        and all(mechanism_gates.values())
        and all(bootstrap_gates.values())
    )
    return {
        "performance_gates": performance_gates,
        "mechanism_gates": mechanism_gates,
        "bootstrap_gates": bootstrap_gates,
        "promote": promote,
        "decision": (
            "promote_stage37_bond_curve_alpha"
            if promote
            else "retain_stage36_and_treat_bond_curve_alpha_as_research_only"
        ),
        "promoted_strategy": (
            candidate_name if promote else baseline_name
        ),
    }


def _plot_performance(performance: pd.DataFrame, path: Path) -> None:
    strategies = [
        "Stage36_Frozen",
        "Stage37_CarryRollOnly",
        "Stage37_RateMomentumOnly",
        "Stage37_BondCurveAlpha",
    ]
    labels = ["Stage36", "Carry+Roll", "Rate trend", "Composite"]
    metrics = [
        ("CAGR", 100.0, "CAGR (%)"),
        ("Sharpe", 1.0, "Sharpe"),
        ("MDD", 100.0, "MDD (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, period in enumerate(("full_2007_2026", "common_2010_2026")):
        view = performance.loc[
            performance["Period"].eq(period)
        ].set_index("Strategy")
        for column, (metric, scale, title) in enumerate(metrics):
            values = [
                float(view.loc[name, metric]) * scale
                for name in strategies
            ]
            axes[row, column].bar(
                labels,
                values,
                color=["#62788d", "#a9822d", "#b45b43", "#18786b"],
            )
            axes[row, column].axhline(0.0, color="#333", linewidth=0.7)
            axes[row, column].set_title(
                f"{'2007-2026' if row == 0 else '2010-2026'} · {title}"
            )
            axes[row, column].tick_params(axis="x", rotation=18)
            axes[row, column].grid(axis="y", alpha=0.22)
    fig.suptitle("Stage37 Korean Treasury Curve Alpha")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_nav(paths: dict[str, pd.DataFrame], path: Path) -> None:
    styles = {
        "Stage36_Frozen": ("#526b80", 2.2),
        "Stage37_CarryRollOnly": ("#af862e", 1.4),
        "Stage37_RateMomentumOnly": ("#b65b43", 1.4),
        "Stage37_BondCurveAlpha": ("#12786a", 2.5),
    }
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    for name, (color, width) in styles.items():
        series = paths[name]["return"].loc[FULL_START:RESEARCH_END]
        nav = (1.0 + series).cumprod()
        ax.plot(
            nav.index.to_timestamp(),
            nav,
            label=name,
            color=color,
            linewidth=width,
        )
    ax.set_yscale("log")
    ax.set_title("Stage36 vs Stage37 Net NAV")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()

    curve_daily, data_audit = load_treasury_curve_daily()
    bond_signals = build_monthly_bond_curve_signals(curve_daily)
    returns, _ = stage35.load_monthly_asset_returns(False)
    probabilities, _ = stage35.build_macro_probabilities(returns)
    stress = stage35.build_monthly_stress_signals(
        returns.index, stage35.build_daily_stress_features()
    )
    market, market_audit = stage35.stage20.load_daily_asset_ohlcv()

    raw_fundamental, _ = stage35.load_fundamental_daily()
    fundamental = stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
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
    asset_vol_daily, _ = stage36.load_asset_implied_volatility_daily()
    asset_vol_signals = stage36.build_monthly_asset_volatility_signals(
        asset_vol_daily, returns.index
    )

    research = build_bond_research_frame(
        bond_signals,
        returns,
        probabilities,
        stress,
        market["BOND"]["close"].dropna(),
    )
    regressions = bond_predictive_regressions(research)
    candidate_paths = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated,
            asset_vol_signals,
            bond_signals,
            mode,
        )
        for name, mode in MODES.items()
    }
    stage36_path = stage35.stage34._load_period_csv(
        stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
        "month",
    )
    paths = {"Stage36_Frozen": stage36_path, **candidate_paths}
    performance = _performance_table(paths)
    bootstrap = _bootstrap_table(stage36_path, candidate_paths)
    gates = gate_decision(performance, regressions, bootstrap)

    reproduction = candidate_paths["Stage37_NoBondAlphaReproduction"]
    common = stage36_path.index.intersection(reproduction.index)
    max_return_error = float(
        (
            stage36_path.loc[common, "return"]
            - reproduction.loc[common, "return"]
        )
        .abs()
        .max()
    )
    max_weight_error = float(
        (
            stage36_path.loc[common, WEIGHT_COLUMNS]
            - reproduction.loc[common, WEIGHT_COLUMNS]
        )
        .abs()
        .to_numpy()
        .max()
    )

    source_after = source_manifest()
    frozen_after = frozen_manifest()
    checks = {
        "source_file_unchanged": source_before == source_after,
        "stage36_frozen_files_unchanged": frozen_before == frozen_after,
        "bond_signal_precedes_target": bool(
            (
                bond_signals["bond_curve_signal_month"]
                < bond_signals.index
            ).all()
        ),
        "first_2007_signal_has_60_month_presample": bool(
            pd.notna(
                bond_signals.loc[
                    FULL_START, "bond_curve_alpha_adjustment"
                ]
            )
        ),
        "no_2y_backfill": bool(
            data_audit["ktb_2y_first_valid"] == "2021-03-10"
        ),
        "fixed_physical_curve_formula": bool(
            FIVE_YEAR_DURATION == 5.0
            and ROLL_MATURITY_MONTHS == 24.0
            and RATE_MOMENTUM_DAYS == 60
            and RATE_MOMENTUM_MONTHS == 3.0
        ),
        "no_parameter_grid_or_winsorization": bool(
            not data_audit["winsorization"]
            and not data_audit["parameter_grid"]
        ),
        "no_change_reproduces_stage36_returns": bool(
            max_return_error < 5e-7
        ),
        "no_change_reproduces_stage36_weights": bool(
            max_weight_error < 5e-6
        ),
        "no_leverage_long_only_sum_to_one": bool(
            all(
                np.allclose(path[WEIGHT_COLUMNS].sum(axis=1), 1.0)
                and (path[WEIGHT_COLUMNS] >= -1e-10).all().all()
                and (path[WEIGHT_COLUMNS] <= 1.0 + 1e-10).all().all()
                for path in candidate_paths.values()
            )
        ),
        "all_candidate_solvers_feasible": bool(
            all(
                path["solver_success"].all()
                and not path["used_fallback"].any()
                and path["volatility_slack"].min() >= -1e-7
                and path["cdar_slack"].min() >= -1e-7
                for path in candidate_paths.values()
            )
        ),
    }

    report = {
        "study": "Stage37_KoreanTreasuryCurveAlpha",
        "decision": gates["decision"],
        "promoted_strategy": gates["promoted_strategy"],
        "scope": (
            "Stage36 is frozen. One causal 5Y Treasury total-return proxy "
            "adjusts only BOND mu; no covariance or other asset mu changes."
        ),
        "fixed_design": {
            "carry": "KTB 5Y yield / 12",
            "roll_down": (
                "5Y duration × (KTB5Y-KTB3Y) / 24 maturity months"
            ),
            "rate_momentum": (
                "5Y duration × prior-quarter 5Y yield decline / 3 months"
            ),
            "composite": "carry + roll-down + rate momentum",
            "alpha_adjustment": (
                "current composite minus expanding prior mean"
            ),
            "duration_years": FIVE_YEAR_DURATION,
            "causal_center_min_months": MIN_CAUSAL_MONTHS,
            "searched_parameters": None,
            "candidate_selection": (
                "composite path predeclared; component paths attribution only"
            ),
        },
        "data_audit": data_audit,
        "gate_results": gates,
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "bond_predictive_regressions": json.loads(
            regressions.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage36": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "reproduction_audit": {
            "months": int(len(common)),
            "max_absolute_return_error": max_return_error,
            "max_absolute_weight_error": max_weight_error,
        },
        "solver_audit": {
            name: stage35.solver_summary(path)
            for name, path in candidate_paths.items()
        },
        "checks": checks,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
        "market_audit": market_audit,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        curve_daily.to_csv(OUTPUT_DIR / "normalized_treasury_curve_daily.csv")
        bond_signals.to_csv(OUTPUT_DIR / "monthly_bond_curve_signals.csv")
        research.to_csv(OUTPUT_DIR / "monthly_bond_research_frame.csv")
        regressions.to_csv(
            OUTPUT_DIR / "bond_predictive_regressions.csv", index=False
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv",
            index=False,
        )
        for name, candidate_path in candidate_paths.items():
            candidate_path.to_csv(
                OUTPUT_DIR / f"{name.lower()}_monthly.csv"
            )
        _plot_performance(
            performance, OUTPUT_DIR / "performance_comparison.png"
        )
        _plot_nav(paths, OUTPUT_DIR / "nav_comparison.png")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

