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

from strategies.stage07_zero_tune_vkospi import zero_tune_strategy as stage07
from strategies.stage36_asset_implied_volatility_risk import (
    asset_implied_volatility_risk_slsqp as stage36,
)
from strategies.stage37_bond_curve_alpha import bond_curve_alpha_slsqp as stage37


stage35 = stage36.stage35
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CURVE_XLSX = stage37.CURVE_XLSX
CPI_XLSX = stage07.get_path(stage07.RAW_DIR, "소비자물가 상승률.xlsx")
OHLCV_CACHE = stage35.stage20.OHLCV_CACHE

ASSETS = stage36.ASSETS
FULL_START = stage36.FULL_START
COMMON_START = stage36.COMMON_START
RESEARCH_END = stage36.RESEARCH_END
GOLD_INDEX = ASSETS.index("GLD")
MIN_CAUSAL_MONTHS = 60
FX_MOMENTUM_DAYS = 63
GOLD_TREND_DAYS = 252
REAL_YIELD_CHANGE_MONTHS = 3
SIGNIFICANCE_LEVEL = 0.10
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]

SOURCE_FILES = (CURVE_XLSX, CPI_XLSX, OHLCV_CACHE)
FROZEN_FILES = (
    Path(stage36.__file__),
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv",
    stage36.OUTPUT_DIR / "validation_report.json",
)

MODES = {
    "Stage38_NoGoldStateReproduction": "baseline_reproduction",
    "Stage38_RealYieldState": "real_yield_state",
    "Stage38_FXState": "fx_state",
    "Stage38_GoldTrendState": "gold_trend_state",
    "Stage38_GoldCompositeState": "gold_composite_state",
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


def _causal_rank_after_history(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    rank = stage35.causal_expanding_midrank(series)
    prior_count = (
        series.notna().shift(1).fillna(False).astype(int).cumsum().astype(int)
    )
    return rank.where(prior_count >= MIN_CAUSAL_MONTHS), prior_count


def _load_cpi_release_series() -> pd.Series:
    cpi = pd.read_excel(CPI_XLSX, index_col=0, skiprows=6)
    cpi.columns = ["CPI_QoQ", "CPI_YoY"]
    cpi.index = (
        pd.to_datetime(cpi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)
    )
    return pd.to_numeric(cpi["CPI_YoY"], errors="coerce").sort_index()


def load_gold_state_inputs() -> tuple[dict[str, pd.Series], dict[str, Any]]:
    curve_daily, curve_audit = stage37.load_treasury_curve_daily()
    cpi = _load_cpi_release_series().dropna()

    market = pd.read_csv(OHLCV_CACHE, parse_dates=["date"])
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    gld_usd = (
        market.loc[market["symbol"].eq("GLD"), ["date", "close"]]
        .dropna()
        .set_index("date")["close"]
        .sort_index()
    )
    usdkrw = (
        market.loc[market["symbol"].eq("USDKRW"), ["date", "close"]]
        .dropna()
        .set_index("date")["close"]
        .sort_index()
    )
    fx_momentum = np.log(usdkrw.where(usdkrw > 0.0)).diff(
        FX_MOMENTUM_DAYS
    )
    gold_trend = np.log(gld_usd.where(gld_usd > 0.0)).diff(
        GOLD_TREND_DAYS
    )
    inputs = {
        "ktb_10y_pct": curve_daily["ktb_10y_pct"].dropna(),
        "cpi_yoy_pct": cpi,
        "usdkrw": usdkrw,
        "fx_momentum_63d": fx_momentum,
        "gld_usd": gld_usd,
        "gold_trend_252d": gold_trend,
    }
    audit = {
        "curve_source": curve_audit["curve_source"],
        "cpi_source": str(CPI_XLSX.resolve()),
        "ohlcv_source": str(OHLCV_CACHE.resolve()),
        "ktb_10y_first_valid": str(
            inputs["ktb_10y_pct"].index.min().date()
        ),
        "cpi_first_release_date": str(cpi.index.min().date()),
        "gld_usd_first_valid": str(gld_usd.index.min().date()),
        "usdkrw_first_valid": str(usdkrw.index.min().date()),
        "fx_momentum_days": FX_MOMENTUM_DAYS,
        "gold_trend_days": GOLD_TREND_DAYS,
        "real_yield_change_months": REAL_YIELD_CHANGE_MONTHS,
        "minimum_prior_months": MIN_CAUSAL_MONTHS,
        "winsorization": False,
        "parameter_grid": False,
    }
    return inputs, audit


def _latest(series: pd.Series, month_end: pd.Timestamp) -> tuple[Any, float]:
    known = series.loc[:month_end].dropna()
    if known.empty:
        return pd.NaT, np.nan
    return known.index[-1], float(known.iloc[-1])


def build_monthly_gold_state_signals(
    inputs: dict[str, pd.Series],
    target_months: pd.PeriodIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        signal_month = target_month - 1
        month_end = signal_month.to_timestamp("M")
        ktb_date, ktb10 = _latest(inputs["ktb_10y_pct"], month_end)
        cpi_date, cpi = _latest(inputs["cpi_yoy_pct"], month_end)
        fx_date, fx_momentum = _latest(
            inputs["fx_momentum_63d"], month_end
        )
        gold_date, gold_trend = _latest(
            inputs["gold_trend_252d"], month_end
        )
        rows.append(
            {
                "target_month": target_month,
                "gold_state_signal_month": signal_month,
                "ktb_signal_date": ktb_date,
                "cpi_release_date": cpi_date,
                "fx_signal_date": fx_date,
                "gold_trend_signal_date": gold_date,
                "ktb_10y_pct": ktb10,
                "cpi_yoy_pct": cpi,
                "real_yield_proxy_pct": ktb10 - cpi,
                "fx_momentum_63d": fx_momentum,
                "gold_trend_252d": gold_trend,
            }
        )
    signals = pd.DataFrame(rows).set_index("target_month")
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    signals["real_yield_change_3m_pctpt"] = signals[
        "real_yield_proxy_pct"
    ].diff(REAL_YIELD_CHANGE_MONTHS)

    rank_columns = [
        "real_yield_proxy_pct",
        "real_yield_change_3m_pctpt",
        "fx_momentum_63d",
        "gold_trend_252d",
    ]
    for column in rank_columns:
        rank, count = _causal_rank_after_history(signals[column])
        signals[f"{column}_rank"] = rank
        signals[f"{column}_prior_count"] = count

    signals["real_yield_support"] = 0.5 * (
        1.0 - signals["real_yield_proxy_pct_rank"]
        + 1.0 - signals["real_yield_change_3m_pctpt_rank"]
    )
    signals["fx_support"] = signals["fx_momentum_63d_rank"]
    signals["gold_trend_support"] = signals["gold_trend_252d_rank"]
    signals["gold_composite_state"] = (
        signals[
            ["real_yield_support", "fx_support", "gold_trend_support"]
        ].mean(axis=1, skipna=False)
    )
    signals["real_yield_active"] = signals[
        "real_yield_support"
    ].notna()
    signals["fx_active"] = signals["fx_support"].notna()
    signals["gold_trend_active"] = signals[
        "gold_trend_support"
    ].notna()
    signals["gold_composite_active"] = signals[
        "gold_composite_state"
    ].notna()

    for state in (
        "real_yield_support",
        "fx_support",
        "gold_trend_support",
        "gold_composite_state",
    ):
        signals[state] = signals[state].fillna(0.5)
    return signals.replace([np.inf, -np.inf], np.nan)


def build_gold_research_frame(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    gold_close_krw: pd.Series,
) -> pd.DataFrame:
    frame = signals.copy()
    for horizon in (1, 3, 6):
        frame[f"future_{horizon}m_gold_return"] = (
            stage35.stage34._forward_compound(returns["GLD"], horizon)
        )
    frame["recent_1m_gold_return"] = returns["GLD"].shift(1)
    frame["gold_realized_vol_21d"] = (
        stage35.stage34._realized_volatility_signal(gold_close_krw)
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


def gold_predictive_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    features = {
        "RealYieldState": "real_yield_support",
        "FXState": "fx_support",
        "GoldTrendState": "gold_trend_support",
        "GoldCompositeState": "gold_composite_state",
    }
    controls = [
        "recent_1m_gold_return",
        "gold_realized_vol_21d",
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
            target = f"future_{horizon}m_gold_return"
            for feature_name, feature in features.items():
                for model, predictors in (
                    ("FeatureOnly", [feature]),
                    ("FullControls", [feature, *controls]),
                ):
                    active_column = {
                        "real_yield_support": "real_yield_active",
                        "fx_support": "fx_active",
                        "gold_trend_support": "gold_trend_active",
                        "gold_composite_state": "gold_composite_active",
                    }[feature]
                    active_view = view.loc[view[active_column]]
                    complete = active_view[[target, *predictors]].dropna()
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


def _state_score(signal: pd.Series, mode: str) -> float:
    if mode == "real_yield_state":
        return float(signal["real_yield_support"])
    if mode == "fx_state":
        return float(signal["fx_support"])
    if mode == "gold_trend_state":
        return float(signal["gold_trend_support"])
    if mode == "gold_composite_state":
        return float(signal["gold_composite_state"])
    return 0.5


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
    gold_state_signal: pd.Series,
    pretrade: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen Stage36 plus an economically signed GLD macro-confidence state."""

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

    gold_state = _state_score(gold_state_signal, mode)
    macro_neutral = float(macro_expected_return.mean())
    gold_relative_mu = float(filtered_macro[GOLD_INDEX] - macro_neutral)
    gold_state_signed = 2.0 * gold_state - 1.0
    if mode == "calibrated_composite_mu":
        gold_mu_adjustment = float(
            gold_state_signal["calibrated_gold_mu_adjustment"]
        )
    else:
        gold_mu_adjustment = abs(gold_relative_mu) * gold_state_signed
    filtered_macro[GOLD_INDEX] += gold_mu_adjustment
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
                f"Stage38 and fallback solves failed: {result.message}; "
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
        "policy": f"Stage38_{mode}",
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
        "gold_state_score": gold_state,
        "gold_state_signed": gold_state_signed,
        "gold_relative_mu_before_state": gold_relative_mu,
        "gold_mu_adjustment": gold_mu_adjustment,
        "real_yield_support": float(
            gold_state_signal["real_yield_support"]
        ),
        "fx_support": float(gold_state_signal["fx_support"]),
        "gold_trend_support": float(
            gold_state_signal["gold_trend_support"]
        ),
        "gold_composite_state": float(
            gold_state_signal["gold_composite_state"]
        ),
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
        "expected_mu_GLD": float(expected_return[GOLD_INDEX]),
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    gold_state_signals: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
        "credit_stress_rank",
    ]
    required_gold = [
        "real_yield_support",
        "fx_support",
        "gold_trend_support",
        "gold_composite_state",
    ]
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(asset_vol_signals.index)
    months = months.intersection(gold_state_signals.index)
    months = months.intersection(
        fundamental_signals.dropna(subset=required_fundamental).index
    )
    months = months.intersection(
        gold_state_signals.dropna(subset=required_gold).index
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
        gold_signal = gold_state_signals.loc[month]
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
            technical_signals.loc[month],
            fundamental_signals.loc[month],
            asset_vol_signals.loc[month],
            gold_signal,
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
                "gold_state_signal_month": gold_signal[
                    "gold_state_signal_month"
                ],
                "ktb_signal_date": gold_signal["ktb_signal_date"],
                "cpi_release_date": gold_signal["cpi_release_date"],
                "fx_signal_date": gold_signal["fx_signal_date"],
                "gold_trend_signal_date": gold_signal[
                    "gold_trend_signal_date"
                ],
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
    base_name = "Stage36_Frozen"
    test_name = "Stage38_GoldCompositeState"
    full_base = perf.loc[(base_name, "full_2007_2026")]
    full_test = perf.loc[(test_name, "full_2007_2026")]
    common_base = perf.loc[(base_name, "common_2010_2026")]
    common_test = perf.loc[(test_name, "common_2010_2026")]
    locked_base = perf.loc[(base_name, "locked_2018_2026")]
    locked_test = perf.loc[(test_name, "locked_2018_2026")]
    performance_gates = {
        "full_cagr_not_lower": bool(
            full_test["CAGR"] >= full_base["CAGR"]
        ),
        "full_sharpe_higher": bool(
            full_test["Sharpe"] > full_base["Sharpe"]
        ),
        "full_mdd_better": bool(full_test["MDD"] > full_base["MDD"]),
        "common_sharpe_not_lower": bool(
            common_test["Sharpe"] >= common_base["Sharpe"]
        ),
        "common_mdd_not_worse": bool(
            common_test["MDD"] >= common_base["MDD"]
        ),
        "locked_cagr_not_lower": bool(
            locked_test["CAGR"] >= locked_base["CAGR"]
        ),
    }
    mechanism = regressions.loc[
        regressions["Feature"].eq("GoldCompositeState")
        & regressions["Period"].eq("common_2010_2026")
        & regressions["Model"].eq("FullControls")
        & regressions["HorizonMonths"].isin([1, 3, 6])
    ]
    mechanism_gates = {
        "all_horizon_betas_positive": bool(
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
        bootstrap["Candidate"].eq(test_name)
        & bootstrap["Period"].eq("common_2010_2026")
    ].set_index("Metric")
    bootstrap_gates = {
        "sharpe_improvement_probability_at_least_60pct": bool(
            boot.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.60
        ),
        "mdd_improvement_probability_at_least_50pct": bool(
            boot.loc["delta_MDD", "ProbabilityPositive"] >= 0.50
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
            "promote_stage38_gold_composite_state"
            if promote
            else "retain_stage36_and_treat_gold_state_as_research_only"
        ),
        "promoted_strategy": test_name if promote else base_name,
    }


def _plot_performance(performance: pd.DataFrame, path: Path) -> None:
    strategies = [
        "Stage36_Frozen",
        "Stage38_RealYieldState",
        "Stage38_FXState",
        "Stage38_GoldTrendState",
        "Stage38_GoldCompositeState",
    ]
    labels = ["Stage36", "Real yield", "FX", "Gold trend", "Composite"]
    metrics = [
        ("CAGR", 100.0, "CAGR (%)"),
        ("Sharpe", 1.0, "Sharpe"),
        ("MDD", 100.0, "MDD (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8))
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
                color=["#60788c", "#9b7c35", "#ad6046", "#4e7598", "#147869"],
            )
            axes[row, column].axhline(0.0, color="#333", linewidth=0.7)
            axes[row, column].set_title(
                f"{'2007-2026' if row == 0 else '2010-2026'} · {title}"
            )
            axes[row, column].tick_params(axis="x", rotation=18)
            axes[row, column].grid(axis="y", alpha=0.22)
    fig.suptitle("Stage38 Gold State Engine")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_nav(paths: dict[str, pd.DataFrame], path: Path) -> None:
    styles = {
        "Stage36_Frozen": ("#506b80", 2.2),
        "Stage38_RealYieldState": ("#9d7c30", 1.3),
        "Stage38_FXState": ("#b25d45", 1.3),
        "Stage38_GoldTrendState": ("#4d7597", 1.3),
        "Stage38_GoldCompositeState": ("#11786a", 2.5),
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
    ax.set_title("Stage36 vs Stage38 Net NAV")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()

    inputs, data_audit = load_gold_state_inputs()
    returns, _ = stage35.load_monthly_asset_returns(False)
    probabilities, _ = stage35.build_macro_probabilities(returns)
    stress = stage35.build_monthly_stress_signals(
        returns.index, stage35.build_daily_stress_features()
    )
    market, market_audit = stage35.stage20.load_daily_asset_ohlcv()
    gold_signals = build_monthly_gold_state_signals(
        inputs, returns.index
    )

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

    research = build_gold_research_frame(
        gold_signals,
        returns,
        probabilities,
        stress,
        market["GLD"]["close"].dropna(),
    )
    regressions = gold_predictive_regressions(research)
    candidate_paths = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated,
            asset_vol_signals,
            gold_signals,
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

    reproduction = candidate_paths["Stage38_NoGoldStateReproduction"]
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
    active = gold_signals.loc[gold_signals["gold_composite_active"]]

    source_after = source_manifest()
    frozen_after = frozen_manifest()
    checks = {
        "source_files_unchanged": source_before == source_after,
        "stage36_frozen_files_unchanged": frozen_before == frozen_after,
        "signal_month_precedes_target": bool(
            (
                gold_signals["gold_state_signal_month"]
                < gold_signals.index
            ).all()
        ),
        "release_dates_not_after_signal_month_end": bool(
            all(
                (
                    gold_signals.loc[
                        gold_signals[column].notna(), column
                    ]
                    <= pd.Series(
                        [
                            period.to_timestamp("M")
                            for period in gold_signals.loc[
                                gold_signals[column].notna()
                            ].index
                            - 1
                        ],
                        index=gold_signals.loc[
                            gold_signals[column].notna()
                        ].index,
                    )
                ).all()
                for column in (
                    "ktb_signal_date",
                    "cpi_release_date",
                    "fx_signal_date",
                    "gold_trend_signal_date",
                )
            )
        ),
        "minimum_60_prior_months_before_composite_active": bool(
            active[
                [
                    "real_yield_proxy_pct_prior_count",
                    "real_yield_change_3m_pctpt_prior_count",
                    "fx_momentum_63d_prior_count",
                    "gold_trend_252d_prior_count",
                ]
            ].min().min()
            >= MIN_CAUSAL_MONTHS
        ),
        "preactivation_is_neutral": bool(
            gold_signals.loc[
                ~gold_signals["gold_composite_active"],
                "gold_composite_state",
            ].eq(0.5).all()
        ),
        "state_scores_bounded_zero_one": bool(
            gold_signals[
                [
                    "real_yield_support",
                    "fx_support",
                    "gold_trend_support",
                    "gold_composite_state",
                ]
            ].ge(0.0).all().all()
            and gold_signals[
                [
                    "real_yield_support",
                    "fx_support",
                    "gold_trend_support",
                    "gold_composite_state",
                ]
            ].le(1.0).all().all()
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
        "study": "Stage38_GoldStateAlpha",
        "decision": gates["decision"],
        "promoted_strategy": gates["promoted_strategy"],
        "scope": (
            "Stage36 is frozen. Korean real-yield pressure, USDKRW support, "
            "and USD gold trend modify only GLD macro-relative mu confidence."
        ),
        "fixed_design": {
            "real_yield_proxy": "KTB10Y minus released CPI YoY",
            "real_yield_support": (
                "mean(1-rank(real yield level), "
                "1-rank(3M real yield change))"
            ),
            "fx_support": "rank(63-trading-day USDKRW log return)",
            "gold_trend_support": (
                "rank(252-trading-day USD GLD log return)"
            ),
            "composite": (
                "equal mean(real-yield support, FX support, gold trend support)"
            ),
            "mu_mapping": (
                "abs(Stage36 filtered GLD macro-relative mu) "
                "times (2*state-1)"
            ),
            "minimum_prior_months": MIN_CAUSAL_MONTHS,
            "searched_parameters": None,
            "candidate_selection": (
                "composite predeclared; three component paths attribution only"
            ),
        },
        "data_audit": data_audit,
        "activation_audit": {
            "first_composite_active_target": (
                str(active.index.min()) if not active.empty else None
            ),
            "neutral_backtest_months": int(
                (
                    ~gold_signals.loc[
                        FULL_START:RESEARCH_END,
                        "gold_composite_active",
                    ]
                ).sum()
            ),
        },
        "gate_results": gates,
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "gold_predictive_regressions": json.loads(
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
        gold_signals.to_csv(OUTPUT_DIR / "monthly_gold_state_signals.csv")
        research.to_csv(OUTPUT_DIR / "monthly_gold_research_frame.csv")
        regressions.to_csv(
            OUTPUT_DIR / "gold_predictive_regressions.csv", index=False
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
    # Windows cp949 consoles cannot encode every decomposed Hangul filename.
    print(json.dumps(run_research(save=True), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
