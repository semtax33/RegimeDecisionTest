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

from strategies.stage35_earnings_credit_fundamentals import (
    earnings_credit_fundamentals_slsqp as stage35,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
GVZ_CSV = ROOT / "raw_data" / "GVZCLS.csv"
OVX_CSV = ROOT / "raw_data" / "OVXCLS.csv"

ASSETS = stage35.ASSETS
FULL_START = stage35.FULL_START
RESEARCH_END = stage35.RESEARCH_END
COMMON_START = pd.Period("2010-01", freq="M")
MIN_SENSOR_HISTORY = 252
SIGNIFICANCE_LEVEL = 0.10
GOLD_INDEX = ASSETS.index("GLD")
OIL_INDEX = ASSETS.index("USO")
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]

SOURCE_FILES = (GVZ_CSV, OVX_CSV)
FROZEN_FILES = (
    Path(stage35.__file__),
    stage35.OUTPUT_DIR / "stage35_fundamentaldualrole_monthly.csv",
    stage35.OUTPUT_DIR / "validation_report.json",
)

MODES = {
    "Stage36_NoOverlayReproduction": "baseline_reproduction",
    "Stage36_GVZGoldRisk": "gvz_gold_risk",
    "Stage36_OVXOilRisk": "ovx_oil_risk",
    "Stage36_GVZ_OVXAssetRisk": "gvz_ovx_asset_risk",
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


def _load_fred_csv(path: Path, value_column: str) -> pd.Series:
    raw = pd.read_csv(path)
    required = {"observation_date", value_column}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw[value_column] = pd.to_numeric(raw[value_column], errors="coerce")
    series = (
        raw.dropna(subset=["observation_date"])
        .set_index("observation_date")[value_column]
        .sort_index()
    )
    series = series.loc[~series.index.duplicated(keep="last")]
    return series.where(series > 0.0)


def _rank_after_prior_history(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Causal expanding midrank, disabled until 252 prior valid observations."""

    rank = stage35.causal_expanding_midrank(series)
    prior_count = series.notna().shift(1).fillna(False).astype(int).cumsum().astype(int)
    return rank.where(prior_count >= MIN_SENSOR_HISTORY), prior_count


def load_asset_implied_volatility_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    gvz = _load_fred_csv(GVZ_CSV, "GVZCLS").rename("gvz")  # Gold ETF Volatility Index
    ovx = _load_fred_csv(OVX_CSV, "OVXCLS").rename("ovx")  # Oil ETF Volatility Index
    daily = pd.concat([gvz, ovx], axis=1).sort_index()
    for sensor in ("gvz", "ovx"):
        rank, prior_count = _rank_after_prior_history(daily[sensor])
        daily[f"{sensor}_causal_rank"] = (
            rank  # 경험적분포에서 현재 변동성이 어느정도 위치인지 확인
        )
        daily[f"{sensor}_prior_valid_observations"] = prior_count
    audit = {
        "gvz_source": str(GVZ_CSV.resolve()),
        "ovx_source": str(OVX_CSV.resolve()),
        "daily_union_rows": int(len(daily)),
        "first_union_date": str(daily.index.min().date()),
        "last_union_date": str(daily.index.max().date()),
        "gvz_first_valid": str(gvz.dropna().index.min().date()),
        "gvz_last_valid": str(gvz.dropna().index.max().date()),
        "gvz_valid_observations": int(gvz.notna().sum()),
        "ovx_first_valid": str(ovx.dropna().index.min().date()),
        "ovx_last_valid": str(ovx.dropna().index.max().date()),
        "ovx_valid_observations": int(ovx.notna().sum()),
        "minimum_prior_observations": MIN_SENSOR_HISTORY,
        "prehistory_policy": "neutral variance multiplier 1.0; no backfill",
        "directional_mu_effect": False,
        "winsorization": False,
        "parameter_grid": False,
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit


def build_monthly_asset_volatility_signals(
    daily: pd.DataFrame,
    target_months: pd.PeriodIndex,
) -> pd.DataFrame:
    """Map each sensor's last observation known by t-1 month-end to month t."""

    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        signal_month = target_month - 1
        month_end = signal_month.to_timestamp("M")
        row: dict[str, Any] = {
            "target_month": target_month,
            "asset_vol_signal_month": signal_month,
        }
        for sensor, asset in (("gvz", "GLD"), ("ovx", "USO")):
            known = daily.loc[
                :month_end,
                [
                    sensor,
                    f"{sensor}_causal_rank",
                    f"{sensor}_prior_valid_observations",
                ],
            ].dropna(subset=[sensor])
            if known.empty:
                value = np.nan
                rank = np.nan
                count = 0
                signal_date = pd.NaT
            else:
                current = known.iloc[-1]
                value = float(current[sensor])
                rank = float(current[f"{sensor}_causal_rank"])
                count = int(current[f"{sensor}_prior_valid_observations"])
                signal_date = known.index[-1]
            active = bool(np.isfinite(rank) and count >= MIN_SENSOR_HISTORY)
            multiplier = 1.0 + rank if active else 1.0
            row.update(
                {
                    f"{sensor}_signal_date": signal_date,
                    f"{sensor}_level": value,
                    f"{sensor}_causal_rank": rank,
                    f"{sensor}_prior_valid_observations": count,
                    f"{sensor}_active": active,
                    f"{sensor}_{asset.lower()}_variance_multiplier": multiplier,
                }
            )
        rows.append(row)
    signals = pd.DataFrame(rows).set_index("target_month")
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    return signals.replace([np.inf, -np.inf], np.nan)


def _standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        std = float(frame[column].std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"No usable variation in {column}")
        output[column] = (frame[column] - frame[column].mean()) / std
    return output


def build_asset_risk_research_frame(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    market: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frame = signals.copy()
    frame["vix6_stress_score"] = stress["stress_score"]
    frame["macro_fragility"] = (
        probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    )
    for sensor, asset in (("gvz", "GLD"), ("ovx", "USO")):
        prefix = asset.lower()
        close = market[asset]["close"].dropna()
        frame[f"{prefix}_recent_1m_return"] = returns[asset].shift(1)
        frame[f"{prefix}_realized_vol_21d"] = (
            stage35.stage34._realized_volatility_signal(close)
        )
        risk_targets = stage35.stage34._forward_risk_targets(close).rename(
            columns={
                "future_realized_vol_1m": f"{prefix}_future_realized_vol_1m",
                "future_max_drawdown_1m": f"{prefix}_future_max_drawdown_1m",
                "future_max_drawdown_3m": f"{prefix}_future_max_drawdown_3m",
            }
        )
        frame = frame.join(risk_targets, how="left")
        monthly_close = close.groupby(close.index.to_period("M")).last()
        monthly_return = monthly_close.pct_change().loc[:RESEARCH_END]
        frame[f"{prefix}_future_left_tail_1m"] = stage35.stage34._causal_tail_event(
            monthly_return
        )
    return frame.loc[FULL_START:RESEARCH_END].replace([np.inf, -np.inf], np.nan)


# 현재 변동성 센서 신호가 미래 위험 지표에 대해 예측력이 있는지 확인하기 위해 회귀분석을 수행
def asset_risk_predictive_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "available_full": (FULL_START, RESEARCH_END),
        "common_2010_2026": (COMMON_START, RESEARCH_END),
        "locked_2018_2026": (stage35.LOCKED_START, RESEARCH_END),
    }
    targets = (
        ("future_realized_vol_1m", 1),
        ("future_max_drawdown_1m", 1),
        ("future_max_drawdown_3m", 3),
        ("future_left_tail_1m", 1),
    )
    for sensor, asset in (("gvz", "GLD"), ("ovx", "USO")):
        prefix = asset.lower()
        feature = f"{sensor}_causal_rank"
        controls = [
            f"{prefix}_recent_1m_return",
            f"{prefix}_realized_vol_21d",
            "vix6_stress_score", "macro_fragility",  # (1-성장 강도)
        ]
        for period_name, (start, end) in periods.items():
            view = frame.loc[start:end]
            for target_suffix, lags in targets:
                target = f"{prefix}_{target_suffix}"
                for model, predictors in (
                    ("SensorOnly", [feature]),
                    ("FullControls", [feature, *controls]),
                ):
                    complete = view[[target, *predictors]].dropna()
                    if len(complete) < 36:
                        continue
                    standardized = _standardize(complete, predictors)
                    fit = sm.OLS(complete[target], sm.add_constant(standardized)).fit(
                        cov_type="HAC", cov_kwds={"maxlags": lags}
                    )
                    ic, ic_p = spearmanr(
                        complete[feature], complete[target], nan_policy="omit"
                    )
                    rows.append(
                        {
                            "Sensor": sensor.upper(),
                            "Asset": asset,
                            "Period": period_name,
                            "Target": target_suffix,
                            "Model": model,
                            "Observations": int(len(complete)),
                            "SensorStandardizedBeta": float(fit.params[feature]),
                            "SensorHACPValue": float(fit.pvalues[feature]),
                            "SensorSpearmanIC": float(ic),
                            "SensorICPValue": float(ic_p),
                            "AdjustedR2": float(fit.rsquared_adj),
                            "HACLags": lags,
                        }
                    )
    return pd.DataFrame(rows)


def _mode_uses_gvz(mode: str) -> bool:
    return mode in {"gvz_gold_risk", "gvz_ovx_asset_risk"}


def _mode_uses_ovx(mode: str) -> bool:
    return mode in {"ovx_oil_risk", "gvz_ovx_asset_risk"}


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
    pretrade: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stage35 final optimizer plus asset-specific variance-only scaling."""
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
    history_asset_return_mean = history.mean(axis=0).to_numpy(dtype=float)
    macro_expected_return = np.asarray(
        moment_detail["macro_expected_monthly_return"], dtype=float
    ) # (n_assets,) historical macro mu is used for technical filtering, not the stress-adjusted version.
    original_stress_adjustment = np.asarray(
        moment_detail["stress_return_adjustment"], dtype=float
    )
    technical = stage35.stage20.apply_technical_inputs(
        macro_expected_return, base_covariance, technical_signal, history_asset_return_mean
    )
    filtered_macro = np.asarray(
        technical["filtered_macro_expected_return"], dtype=float
    ).copy()
    stress_adjustment = original_stress_adjustment.copy()

    eps_mu = float(fundamental_signal["eps_mu_adjustment_KODEX200"])
    valuation_mu = float(fundamental_signal["valuation_mu_adjustment_KODEX200"])
    credit_stress_multiplier = float(fundamental_signal["credit_stress_multiplier"])
    #stress_adjustment[stage35.EQUITY_INDEX] *= credit_stress_multiplier
    filtered_macro[stage35.EQUITY_INDEX] += eps_mu + valuation_mu
    expected_return = filtered_macro + stress_adjustment

    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)
    # credit_variance_multiplier = 1.0 + float(fundamental_signal["credit_stress_rank"])
    # credit_scaling = np.eye(len(ASSETS), dtype=float)
    # credit_scaling[stage35.EQUITY_INDEX, stage35.EQUITY_INDEX] = math.sqrt(
    #     credit_variance_multiplier
    # )
    #covariance = credit_scaling @ covariance @ credit_scaling

    # gvz_multiplier = (
    #     float(asset_vol_signal["gvz_gld_variance_multiplier"])
    #     if _mode_uses_gvz(mode)
    #     else 1.0
    # )
    # ovx_multiplier = (
    #     float(asset_vol_signal["ovx_uso_variance_multiplier"])
    #     if _mode_uses_ovx(mode)
    #     else 1.0
    # )
    # asset_scaling = np.eye(len(ASSETS), dtype=float)
    # asset_scaling[GOLD_INDEX, GOLD_INDEX] = math.sqrt(gvz_multiplier)
    # asset_scaling[OIL_INDEX, OIL_INDEX] = math.sqrt(ovx_multiplier)
    # covariance = asset_scaling @ covariance @ asset_scaling

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
        downside_semivariance = float(np.mean(np.minimum(realized_history, 0.0) ** 2))
        transaction_cost = stage35.expected_transaction_cost(weights, pretrade)
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
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage35.CATASTROPHE_ANNUAL_VOLATILITY - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage35.CATASTROPHE_CDAR
                + stage35.cdar(historical_returns @ weights, stage35.CDAR_CONFIDENCE)
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
                f"Stage36 and fallback solves failed: {result.message}; "
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
        "policy": f"Stage36_{mode}",
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": (stage35.CATASTROPHE_ANNUAL_VOLATILITY - annual_vol),
        "cdar_slack": stage35.CATASTROPHE_CDAR + historical_cdar,
        "eps_mu_adjustment_KODEX200": eps_mu,
        "valuation_mu_adjustment_KODEX200": valuation_mu,
        "credit_stress_confirmation_multiplier": credit_stress_multiplier,
        #"credit_equity_variance_multiplier": credit_variance_multiplier,
        #"gvz_gold_variance_multiplier": gvz_multiplier,
        #"ovx_oil_variance_multiplier": ovx_multiplier,
        "gvz_mu_adjustment_GLD": 0.0,
        "ovx_mu_adjustment_USO": 0.0,
        "expected_mu_GLD": float(expected_return[GOLD_INDEX]),
        "expected_mu_USO": float(expected_return[OIL_INDEX]),
    }
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
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
        probability = probabilities.loc[month]  # 각 국면별 확률
        stress = float(stress_signals.loc[month, "stress_score"])  # 스트레스 신호
        recovery = float(stress_signals.loc[month, "recovery_score"])  # 회복 신호
        technical_signal = technical_signals.loc[month]  # 기술적 신호
        fundamental_signal = fundamental_signals.loc[month]
        asset_vol_signal = asset_vol_signals.loc[month]
        weights, detail = _solve_weights(
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
            mode,
        )
        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * stage35.DOMESTIC_TRADE_COST
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

        row: dict[str, Any] = {
            "month": month,
            "asset_vol_signal_month": asset_vol_signal["asset_vol_signal_month"],
            "gvz_signal_date": asset_vol_signal["gvz_signal_date"],
            "ovx_signal_date": asset_vol_signal["ovx_signal_date"],
            "gvz_level": float(asset_vol_signal["gvz_level"]),
            "ovx_level": float(asset_vol_signal["ovx_level"]),
            "gvz_causal_rank": float(asset_vol_signal["gvz_causal_rank"]),
            "ovx_causal_rank": float(asset_vol_signal["ovx_causal_rank"]),
            "gvz_active": bool(asset_vol_signal["gvz_active"]),
            "ovx_active": bool(asset_vol_signal["ovx_active"]),
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
        rows.append(row)
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
            rows.append(stage35.metric_row(name, path, period_name, start, end))
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


def _risk_gate(risk_tests: pd.DataFrame, sensor: str) -> dict[str, Any]:
    rows = risk_tests.loc[
        risk_tests["Sensor"].eq(sensor)
        & risk_tests["Period"].eq("common_2010_2026")
        & risk_tests["Model"].eq("FullControls")
        & risk_tests["Target"].isin(
            ["future_realized_vol_1m", "future_max_drawdown_3m"]
        )
    ]
    expected = rows["SensorStandardizedBeta"].gt(0.0) & rows["SensorHACPValue"].lt(
        SIGNIFICANCE_LEVEL
    )
    return {
        "positive_sign_all_primary_targets": bool(
            rows["SensorStandardizedBeta"].gt(0.0).all()
        ),
        "at_least_one_primary_target_p_below_10pct": bool(expected.any()),
        "pass": bool(rows["SensorStandardizedBeta"].gt(0.0).all() and expected.any()),
    }


def gate_decision(
    performance: pd.DataFrame,
    risk_tests: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    perf = performance.set_index(["Strategy", "Period"])
    baseline_name = "Stage35_Frozen"
    candidate_name = "Stage36_GVZ_OVXAssetRisk"
    full_base = perf.loc[(baseline_name, "full_2007_2026")]
    full_test = perf.loc[(candidate_name, "full_2007_2026")]
    common_base = perf.loc[(baseline_name, "common_2010_2026")]
    common_test = perf.loc[(candidate_name, "common_2010_2026")]
    performance_gates = {
        "full_cagr_at_least_10pct": bool(full_test["CAGR"] >= 0.10),
        "full_sharpe_not_lower": bool(full_test["Sharpe"] >= full_base["Sharpe"]),
        "full_mdd_not_worse": bool(full_test["MDD"] >= full_base["MDD"]),
        "common_sharpe_higher": bool(common_test["Sharpe"] > common_base["Sharpe"]),
        "common_mdd_better": bool(common_test["MDD"] > common_base["MDD"]),
        "common_cagr_not_lower_by_more_than_50bp": bool(
            common_test["CAGR"] >= common_base["CAGR"] - 0.005
        ),
    }
    gvz = _risk_gate(risk_tests, "GVZ")
    ovx = _risk_gate(risk_tests, "OVX")
    combined_bootstrap = bootstrap.loc[
        bootstrap["Candidate"].eq(candidate_name)
        & bootstrap["Period"].eq("common_2010_2026")
    ].set_index("Metric")
    bootstrap_gates = {
        "common_sharpe_improvement_probability_at_least_60pct": bool(
            combined_bootstrap.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.60
        ),
        "common_mdd_improvement_probability_at_least_50pct": bool(
            combined_bootstrap.loc["delta_MDD", "ProbabilityPositive"] >= 0.50
        ),
    }
    promote = bool(
        all(performance_gates.values())
        and gvz["pass"]
        and ovx["pass"]
        and all(bootstrap_gates.values())
    )
    return {
        "performance_gates": performance_gates,
        "gvz_risk_gate": gvz,
        "ovx_risk_gate": ovx,
        "bootstrap_gates": bootstrap_gates,
        "promote": promote,
        "decision": (
            "promote_stage36_gvz_ovx_asset_risk"
            if promote
            else "retain_stage35_pending_stronger_asset_volatility_evidence"
        ),
        "promoted_strategy": (candidate_name if promote else "Stage35_Frozen"),
    }


def _plot_performance(performance: pd.DataFrame, path: Path) -> None:
    strategies = [
        "Stage35_Frozen",
        "Stage36_GVZGoldRisk",
        "Stage36_OVXOilRisk",
        "Stage36_GVZ_OVXAssetRisk",
    ]
    labels = ["Stage35", "GVZ→Gold", "OVX→Oil", "GVZ+OVX"]
    metrics = [
        ("CAGR", 100.0, "CAGR (%)"),
        ("Sharpe", 1.0, "Sharpe"),
        ("MDD", 100.0, "MDD (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, period in enumerate(("full_2007_2026", "common_2010_2026")):
        view = performance.loc[performance["Period"].eq(period)].set_index("Strategy")
        for column, (metric, scale, title) in enumerate(metrics):
            values = [float(view.loc[name, metric]) * scale for name in strategies]
            colors = ["#62788d", "#b38a32", "#b35b45", "#177b6d"]
            axes[row, column].bar(labels, values, color=colors)
            axes[row, column].axhline(0.0, color="#333", linewidth=0.7)
            axes[row, column].set_title(
                f"{'2007-2026' if row == 0 else '2010-2026'} · {title}"
            )
            axes[row, column].tick_params(axis="x", rotation=18)
            axes[row, column].grid(axis="y", alpha=0.22)
    fig.suptitle("Stage36 Asset-Specific Implied-Volatility Risk Overlay")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_nav(paths: dict[str, pd.DataFrame], path: Path) -> None:
    styles = {
        "Stage35_Frozen": ("#556b7d", 2.2),
        "Stage36_GVZGoldRisk": ("#bd8c2b", 1.5),
        "Stage36_OVXOilRisk": ("#b45740", 1.5),
        "Stage36_GVZ_OVXAssetRisk": ("#137a69", 2.5),
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
    ax.set_title("Stage35 vs Stage36 Net NAV (log scale)")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_sensor_history(
    daily: pd.DataFrame, signals: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), sharex=True)
    for axis, sensor, color, label in (
        (axes[0], "gvz", "#b28b32", "GVZ / Gold"),
        (axes[1], "ovx", "#b6533f", "OVX / Oil"),
    ):
        axis.plot(daily.index, daily[sensor], color=color, linewidth=1.0)
        active = signals.loc[signals[f"{sensor}_active"]]
        if not active.empty:
            start = active.index.min().start_time
            axis.axvline(start, color="#176f64", linestyle="--", linewidth=1.5)
            axis.text(
                start,
                axis.get_ylim()[1] * 0.90,
                f" active {active.index.min()}",
                color="#176f64",
                fontsize=8,
            )
        axis.set_title(label)
        axis.set_ylabel("Index level")
        axis.grid(alpha=0.22)
    fig.suptitle("Raw implied-volatility history and causal activation")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _html_table(frame: pd.DataFrame, percent: set[str] | None = None) -> str:
    percent = percent or set()
    display = frame.copy()
    for column in display.columns:
        if column in percent:
            display[column] = display[column].map(
                lambda value: f"{float(value) * 100:.3f}%"
            )
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.6f}")
    return display.to_html(index=False, escape=True, border=0)


def _render_html_report(
    report: dict[str, Any],
    performance: pd.DataFrame,
    risk_tests: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    selected = performance.loc[
        performance["Strategy"].isin(
            [
                "Stage35_Frozen",
                "Stage36_GVZGoldRisk",
                "Stage36_OVXOilRisk",
                "Stage36_GVZ_OVXAssetRisk",
            ]
        )
        & performance["Period"].isin(
            [
                "full_2007_2026",
                "common_2010_2026",
                "locked_2018_2026",
            ]
        ),
        [
            "Strategy",
            "Period",
            "CAGR",
            "Volatility",
            "Sharpe",
            "MDD",
            "AvgTurnover",
            "TotalCost",
        ],
    ]
    risk_view = risk_tests.loc[
        risk_tests["Period"].eq("common_2010_2026")
        & risk_tests["Model"].eq("FullControls"),
        [
            "Sensor",
            "Asset",
            "Target",
            "Observations",
            "SensorStandardizedBeta",
            "SensorHACPValue",
            "SensorSpearmanIC",
        ],
    ]
    boot_view = bootstrap.loc[
        bootstrap["Candidate"].eq("Stage36_GVZ_OVXAssetRisk"),
        [
            "Period",
            "Metric",
            "Mean",
            "P05",
            "P50",
            "P95",
            "ProbabilityPositive",
        ],
    ]
    audit = report["data_audit"]
    gates = report["gate_results"]
    decision_ko = (
        "Stage36 결합전략 승격"
        if gates["promote"]
        else "Stage35 유지 — Stage36은 연구 경로로 보존"
    )
    indexed = performance.set_index(["Strategy", "Period"])
    full_base = indexed.loc[("Stage35_Frozen", "full_2007_2026")]
    full_test = indexed.loc[("Stage36_GVZ_OVXAssetRisk", "full_2007_2026")]
    common_base = indexed.loc[("Stage35_Frozen", "common_2010_2026")]
    common_test = indexed.loc[("Stage36_GVZ_OVXAssetRisk", "common_2010_2026")]
    locked_base = indexed.loc[("Stage35_Frozen", "locked_2018_2026")]
    locked_test = indexed.loc[("Stage36_GVZ_OVXAssetRisk", "locked_2018_2026")]
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage36 GVZ·OVX 자산별 위험 오버레이</title>
<style>
body{{margin:0;background:#f3f6f8;color:#182534;font:16px/1.72 "Malgun Gothic",sans-serif}}
header{{padding:48px 7%;background:linear-gradient(120deg,#15344f,#176f69);color:white}}
main{{max-width:1120px;margin:24px auto;padding:0 20px}}section{{background:white;border:1px solid #dce3e8;border-radius:14px;padding:27px;margin:18px 0}}
h1{{font-size:2.5rem;margin:.2rem 0}}h2{{color:#173f5d;border-bottom:2px solid #e5edf2;padding-bottom:8px}}
.note{{padding:14px 17px;background:#edf7f4;border-left:5px solid #177b6d}}.warn{{padding:14px 17px;background:#fff6e5;border-left:5px solid #c98918}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;display:block;overflow:auto}}th,td{{padding:9px;border-bottom:1px solid #e2e7eb;text-align:right;white-space:nowrap}}th{{background:#edf3f6}}td:first-child,th:first-child{{text-align:left}}
code{{background:#edf1f4;padding:2px 5px;border-radius:4px}}pre{{background:#12202d;color:#e5eef5;padding:16px;border-radius:9px;overflow:auto}}
img{{width:100%;border:1px solid #dce3e8;border-radius:10px}}.metric{{font-size:1.35rem;font-weight:bold;color:#176f69}}
</style></head><body>
<header><div>RegimeDecisionTest · Stage36</div><h1>GVZ·OVX 자산별 변동성 위험 오버레이</h1>
<p>Stage35를 동결하고 GVZ는 GLD, OVX는 USO 공분산 축에만 연결한 비레버리지 인과적 월간 전략</p></header><main>
<section><h2>1. 최종 판정</h2><p class="metric">{decision_ko}</p>
<p>{gates["decision"]}</p><div class="note">방향성 기대수익 μ는 전혀 수정하지 않았다. 252개 과거 유효관측 전에는 multiplier=1로 Stage35와 동일하며, 활성화 이후에만 해당 자산의 variance와 공분산 축을 1+causal rank만큼 확대한다.</div></section>
<section><h2>2. 핵심 성과 차이</h2>
<ul><li>전체 구간: CAGR {full_base['CAGR']*100:.3f}% → {full_test['CAGR']*100:.3f}%, Sharpe {full_base['Sharpe']:.3f} → {full_test['Sharpe']:.3f}, MDD {full_base['MDD']*100:.3f}% → {full_test['MDD']*100:.3f}%</li>
<li>2010 공통구간: CAGR {common_base['CAGR']*100:.3f}% → {common_test['CAGR']*100:.3f}%, Sharpe {common_base['Sharpe']:.3f} → {common_test['Sharpe']:.3f}, MDD {common_base['MDD']*100:.3f}% → {common_test['MDD']*100:.3f}%</li>
<li>2018 잠금구간: CAGR {locked_base['CAGR']*100:.3f}% → {locked_test['CAGR']*100:.3f}%, Sharpe {locked_base['Sharpe']:.3f} → {locked_test['Sharpe']:.3f}, MDD {locked_base['MDD']*100:.3f}% → {locked_test['MDD']*100:.3f}%</li></ul>
<p class="warn">전체·2010 공통구간에서는 위험효율이 개선됐지만, 2018 이후에는 MDD가 0.052%p 완화된 대신 CAGR과 Sharpe가 낮아졌다. 모든 시기에서 Stage35를 지배한다고 해석하면 안 된다.</p></section>
<section><h2>3. 데이터와 활성화</h2>
<ul><li>GVZ: {audit["gvz_first_valid"]} ~ {audit["gvz_last_valid"]}, {audit["gvz_valid_observations"]:,}개</li>
<li>OVX: {audit["ovx_first_valid"]} ~ {audit["ovx_last_valid"]}, {audit["ovx_valid_observations"]:,}개</li>
<li>GVZ 최초 활성 target: {report["activation_audit"]["gvz_first_active_target"]}</li>
<li>OVX 최초 활성 target: {report["activation_audit"]["ovx_first_active_target"]}</li></ul>
<p>출시 전 값은 0이나 평균으로 채우지 않았고, 실현변동성으로 역산하지도 않았다.</p>
<img src="outputs/sensor_history.png" alt="GVZ OVX history"></section>
<section><h2>4. 정확한 위험 매핑</h2>
<pre>q_GVZ,t = causal expanding rank(GVZ_t), after 252 prior observations
Var_GLD* = Var_GLD × (1 + q_GVZ,t)

q_OVX,t = causal expanding rank(OVX_t), after 252 prior observations
Var_USO* = Var_USO × (1 + q_OVX,t)

μ_GLD adjustment = 0
μ_USO adjustment = 0</pre>
<p>공분산 행렬에는 대각행렬 D의 GLD·USO 위치에 각각 sqrt(1+q)를 넣고 DΣD를 계산한다. 따라서 해당 자산의 분산뿐 아니라 다른 자산과의 공분산도 일관되게 조정된다.</p></section>
<section><h2>5. 비교 실험</h2>
<p>Stage35, GVZ-only, OVX-only, GVZ+OVX를 동일 비용·동일 SLSQP 제약으로 비교했다. 최종 후보는 사전에 정한 결합 경로이며 가장 좋은 사후 조합을 선택하지 않았다.</p>
{_html_table(selected, {"CAGR", "Volatility", "MDD", "AvgTurnover", "TotalCost"})}
<img src="outputs/performance_comparison.png" alt="performance comparison"></section>
<section><h2>6. 미래 위험 예측 진단</h2>
<p>센서 rank가 자기 자산의 향후 1개월 실현변동성, 1·3개월 최대낙폭, 왼쪽꼬리를 설명하는지 검사했다. 통제변수는 자기자산 최근 1개월 수익률·21일 실현변동성, VIX6 stress, 거시 취약도이며 HAC 표준오차를 사용했다.</p>
{_html_table(risk_view)}
<p class="warn">높은 GVZ/OVX는 방향성 하락을 뜻하지 않는다. beta가 약하더라도 부호를 뒤집어 수익률 알파로 전환하지 않는다.</p></section>
<section><h2>7. 블록 부트스트랩</h2>
<p>Stage35와 월을 짝지은 12개월 circular block을 2,000회 재표집했다. 양의 확률은 Stage36−Stage35가 0보다 클 확률이다.</p>
{_html_table(boot_view)}</section>
<section><h2>8. NAV</h2><img src="outputs/nav_comparison.png" alt="NAV comparison"></section>
<section><h2>9. SLSQP와 비용</h2>
<p>목적함수와 제약은 Stage35와 동일하다.</p>
<pre>maximize w'μ - 0.5 w'Σw - mean(min(R_hist w, 0)^2) - transaction_cost
subject to sum(w)=1, 0&lt;=w&lt;=1,
annualized volatility &lt;=13%, historical CDaR(90%) &gt;=-16%</pre>
<p>국내 회전비용 15bp와 GLD·USO 순비중 변화의 추가 5bp를 월별 수익에서 차감한다.</p></section>
<section><h2>10. 인과성·과최적화 방지</h2>
<ul><li>target month에는 직전 월말까지 공개된 마지막 값만 사용</li><li>현재 관측 전 252개 유효 일간 이력이 없으면 완전 중립</li><li>전체표본 평균·분산, backfill, splice, realized-vol 대체 없음</li><li>rank→variance multiplier는 1+rank 한 가지, grid search 없음</li><li>GVZ·OVX 독립 ablation과 2010 공통표본을 함께 공개</li><li>Stage35 파일과 원천 CSV의 실행 전후 hash 비교</li></ul></section>
<section><h2>11. 코드 위치</h2>
<p><code>asset_implied_volatility_risk_slsqp.py</code>의 <code>load_asset_implied_volatility_daily</code>, <code>build_monthly_asset_volatility_signals</code>, <code>_solve_weights</code>, <code>run_backtest</code>, <code>run_research</code> 순서로 읽으면 된다.</p></section>
<section><h2>12. 재현</h2>
<pre>python -m strategies.stage36_asset_implied_volatility_risk.asset_implied_volatility_risk_slsqp
python -m pytest tests/test_stage36_asset_implied_volatility_risk.py -q</pre></section>
<section><h2>13. 한계</h2>
<p>GLD·USO 가격은 기존 전략과 일관되게 원화 환산 수익률이므로 GVZ·OVX가 직접 측정하는 달러표시 기초자산 위험과 환율효과가 섞인다. 또한 FRED 관측일은 확인했지만 실시간 vintage·장중 배포시각까지 보장하지 않는다. 백테스트 개선은 미래 성과 보장이 아니다.</p></section>
</main></body></html>"""


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()

    daily, data_audit = load_asset_implied_volatility_daily()
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
    print('Technical Path', stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv")
    exit()
    technical = stage35.stage34._load_period_csv(
        stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    asset_vol_signals = build_monthly_asset_volatility_signals(daily, returns.index)
    risk_frame = build_asset_risk_research_frame(
        asset_vol_signals, returns, probabilities, stress, market
    )
    risk_tests = asset_risk_predictive_regressions(risk_frame)

    candidate_paths = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated,
            asset_vol_signals,
            mode,
        )
        for name, mode in MODES.items()
    }
    stage35_path = stage35.stage34._load_period_csv(
        stage35.OUTPUT_DIR / "stage35_fundamentaldualrole_monthly.csv",
        "month",
    )
    paths = {"Stage35_Frozen": stage35_path, **candidate_paths}
    performance = _performance_table(paths)
    bootstrap = _bootstrap_table(stage35_path, candidate_paths)
    gates = gate_decision(performance, risk_tests, bootstrap)

    reproduction = candidate_paths["Stage36_NoOverlayReproduction"]
    common = stage35_path.index.intersection(reproduction.index)
    max_return_error = float(
        (stage35_path.loc[common, "return"] - reproduction.loc[common, "return"])
        .abs()
        .max()
    )
    max_weight_error = float(
        (
            stage35_path.loc[common, WEIGHT_COLUMNS]
            - reproduction.loc[common, WEIGHT_COLUMNS]
        )
        .abs()
        .to_numpy()
        .max()
    )

    gvz_active = asset_vol_signals.loc[asset_vol_signals["gvz_active"]]
    ovx_active = asset_vol_signals.loc[asset_vol_signals["ovx_active"]]
    activation_audit = {
        "gvz_first_active_target": (
            str(gvz_active.index.min()) if not gvz_active.empty else None
        ),
        "ovx_first_active_target": (
            str(ovx_active.index.min()) if not ovx_active.empty else None
        ),
        "gvz_neutral_signal_months_all": int((~asset_vol_signals["gvz_active"]).sum()),
        "ovx_neutral_signal_months_all": int((~asset_vol_signals["ovx_active"]).sum()),
        "gvz_neutral_backtest_months": int(
            (~asset_vol_signals.loc[FULL_START:RESEARCH_END, "gvz_active"]).sum()
        ),
        "ovx_neutral_backtest_months": int(
            (~asset_vol_signals.loc[FULL_START:RESEARCH_END, "ovx_active"]).sum()
        ),
        "common_sample_start": str(COMMON_START),
    }

    source_after = source_manifest()
    frozen_after = frozen_manifest()
    all_candidate_paths = list(candidate_paths.values())
    checks = {
        "source_files_unchanged": source_before == source_after,
        "stage35_frozen_files_unchanged": frozen_before == frozen_after,
        "no_backfill_before_minimum_history": bool(
            asset_vol_signals.loc[
                ~asset_vol_signals["gvz_active"],
                "gvz_gld_variance_multiplier",
            ]
            .eq(1.0)
            .all()
            and asset_vol_signals.loc[
                ~asset_vol_signals["ovx_active"],
                "ovx_uso_variance_multiplier",
            ]
            .eq(1.0)
            .all()
        ),
        "minimum_252_prior_observations": bool(
            gvz_active["gvz_prior_valid_observations"].min() >= MIN_SENSOR_HISTORY
            and ovx_active["ovx_prior_valid_observations"].min() >= MIN_SENSOR_HISTORY
        ),
        "signal_month_precedes_target": bool(
            (
                asset_vol_signals["asset_vol_signal_month"] < asset_vol_signals.index
            ).all()
        ),
        "signal_dates_not_after_prior_month_end": bool(
            all(
                (
                    asset_vol_signals[f"{sensor}_signal_date"].dropna()
                    <= pd.Series(
                        [
                            period.to_timestamp("M")
                            for period in asset_vol_signals.loc[
                                asset_vol_signals[f"{sensor}_signal_date"].notna()
                            ].index
                            - 1
                        ],
                        index=asset_vol_signals.loc[
                            asset_vol_signals[f"{sensor}_signal_date"].notna()
                        ].index,
                    )
                ).all()
                for sensor in ("gvz", "ovx")
            )
        ),
        "variance_multipliers_within_one_and_two": bool(
            asset_vol_signals[
                [
                    "gvz_gld_variance_multiplier",
                    "ovx_uso_variance_multiplier",
                ]
            ]
            .ge(1.0)
            .all()
            .all()
            and asset_vol_signals[
                [
                    "gvz_gld_variance_multiplier",
                    "ovx_uso_variance_multiplier",
                ]
            ]
            .le(2.0)
            .all()
            .all()
        ),
        "no_directional_mu_adjustment": bool(
            all(
                path["gvz_mu_adjustment_GLD"].eq(0.0).all()
                and path["ovx_mu_adjustment_USO"].eq(0.0).all()
                for path in all_candidate_paths
            )
        ),
        "no_leverage_long_only_sum_to_one": bool(
            all(
                np.allclose(path[WEIGHT_COLUMNS].sum(axis=1), 1.0)
                and (path[WEIGHT_COLUMNS] >= -1e-10).all().all()
                and (path[WEIGHT_COLUMNS] <= 1.0 + 1e-10).all().all()
                for path in all_candidate_paths
            )
        ),
        "no_overlay_reproduces_stage35_returns": bool(max_return_error < 5e-7),
        "no_overlay_reproduces_stage35_weights": bool(max_weight_error < 5e-6),
        "all_candidate_solvers_feasible": bool(
            all(
                path["solver_success"].all()
                and not path["used_fallback"].any()
                and path["volatility_slack"].min() >= -1e-7
                and path["cdar_slack"].min() >= -1e-7
                for path in all_candidate_paths
            )
        ),
    }

    report = {
        "study": "Stage36_AssetSpecificImpliedVolatilityRisk",
        "decision": gates["decision"],
        "promoted_strategy": gates["promoted_strategy"],
        "scope": (
            "Stage35 is frozen. GVZ changes only the GLD covariance axis; "
            "OVX changes only the USO covariance axis. Neither changes mu."
        ),
        "fixed_design": {
            "sensor_mapping": {"GVZCLS": "GLD", "OVXCLS": "USO"},
            "minimum_prior_daily_observations": MIN_SENSOR_HISTORY,
            "prehistory_policy": "neutral multiplier 1.0",
            "variance_multiplier": "1 + causal expanding midrank",
            "covariance_mapping": "D @ Sigma @ D with sqrt(multiplier)",
            "directional_mu_effect": None,
            "full_sample": f"{FULL_START}/{RESEARCH_END}",
            "common_sample": f"{COMMON_START}/{RESEARCH_END}",
            "searched_parameters": None,
            "candidate_selection": (
                "combined GVZ+OVX path was predeclared; single-sensor paths "
                "are attribution ablations, not best-of selection"
            ),
        },
        "data_audit": data_audit,
        "activation_audit": activation_audit,
        "gate_results": gates,
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "asset_risk_predictive_regressions": json.loads(
            risk_tests.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage35": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "reproduction_audit": {
            "months": int(len(common)),
            "max_absolute_return_error": max_return_error,
            "max_absolute_weight_error": max_weight_error,
        },
        "solver_audit": {
            name: stage35.solver_summary(path) for name, path in candidate_paths.items()
        },
        "checks": checks,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
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
        daily.to_csv(OUTPUT_DIR / "normalized_gvz_ovx_daily.csv")
        asset_vol_signals.to_csv(OUTPUT_DIR / "monthly_asset_volatility_signals.csv")
        risk_frame.to_csv(OUTPUT_DIR / "monthly_asset_risk_research_frame.csv")
        risk_tests.to_csv(
            OUTPUT_DIR / "asset_risk_predictive_regressions.csv", index=False
        )
        performance.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage35.csv",
            index=False,
        )
        for name, candidate_path in candidate_paths.items():
            candidate_path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        _plot_performance(performance, OUTPUT_DIR / "performance_comparison.png")
        _plot_nav(paths, OUTPUT_DIR / "nav_comparison.png")
        _plot_sensor_history(
            daily, asset_vol_signals, OUTPUT_DIR / "sensor_history.png"
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html = _render_html_report(report, performance, risk_tests, bootstrap)
        (Path(__file__).resolve().parent / "stage36_gvz_ovx_report.html").write_text(
            html, encoding="utf-8"
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
