from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from strategies.stage36_asset_implied_volatility_risk import (
    asset_implied_volatility_risk_slsqp as stage36,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = stage36.ASSETS
WEIGHT_COLUMNS = [f"w_{asset}" for asset in ASSETS]
FULL_START = stage36.FULL_START
RESEARCH_END = stage36.RESEARCH_END
LOCKED_START = stage36.stage35.LOCKED_START
REGIME_COLUMNS = stage36.stage35.REGIME_COLUMNS
MIN_CALIBRATION_MONTHS = 60
TRADING_DAYS_PER_MONTH = 21
NUMERICAL_EPSILON = 1e-12

BASELINE_PATH = (
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv"
)
FROZEN_FILES = (
    Path(stage36.__file__),
    BASELINE_PATH,
    stage36.OUTPUT_DIR / "validation_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_manifest() -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in FROZEN_FILES
    }


def _nearest_psd(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(
        float(np.trace(symmetric)) / len(symmetric), NUMERICAL_EPSILON
    )
    floor = scale * 1e-10
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _weighted_mean_covariance(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= NUMERICAL_EPSILON:
        weights = np.ones(len(values), dtype=float)
        total = float(len(values))
    normalized = weights / total
    mean = normalized @ values
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    return mean, _nearest_psd(covariance)


def positive_part_signal_reliability(
    coefficient: float, standard_error: float
) -> float:
    """Estimate the share of coefficient magnitude exceeding sampling noise.

    If beta_hat = beta + noise with estimated noise variance se^2, then
    max(beta_hat^2-se^2, 0)/beta_hat^2 is a positive-part estimate of the
    signal share. Unlike beta*R^2, the shrinkage uses the uncertainty of the
    coefficient being shrunk and has a sampling-error interpretation.
    """

    coefficient = float(coefficient)
    standard_error = float(standard_error)
    if (
        not np.isfinite(coefficient)
        or not np.isfinite(standard_error)
        or abs(coefficient) <= NUMERICAL_EPSILON
    ):
        return 0.0
    return float(
        np.clip(
            1.0 - standard_error**2 / coefficient**2,
            0.0,
            1.0,
        )
    )


def _fit_hac(
    target: np.ndarray,
    predictors: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = sm.add_constant(np.asarray(predictors, dtype=float), has_constant="add")
    y = np.asarray(target, dtype=float)
    try:
        if weights is None:
            fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        else:
            fit = sm.WLS(y, x, weights=np.asarray(weights, dtype=float)).fit(
                cov_type="HAC", cov_kwds={"maxlags": 1}
            )
        coefficients = np.asarray(fit.params, dtype=float)
        errors = np.asarray(fit.bse, dtype=float)
    except (ValueError, np.linalg.LinAlgError):
        coefficients = np.zeros(x.shape[1], dtype=float)
        errors = np.full(x.shape[1], np.inf, dtype=float)
    return coefficients, errors


def estimate_evidence_moments(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Stage 36 macro/stress moments with HAC uncertainty shrinkage.

    Stress and recovery enter one weighted regression per regime, so each
    coefficient is incremental to the other. No sign is hard-coded and no R2
    is multiplied into a slope. The covariance remains Stage 36's continuous
    stress-conditioned covariance, preserving the useful VKOSPI mechanism.
    """

    common = history.index.intersection(historical_probabilities.index)
    common = common.intersection(historical_stress.dropna().index)
    common = common.intersection(historical_recovery.dropna().index)
    if len(common) < stage36.stage35.ONE_CALENDAR_YEAR:
        raise ValueError("At least 12 causal months are required.")

    values = history.loc[common, ASSETS].to_numpy(dtype=float)
    probabilities = historical_probabilities.loc[common, REGIME_COLUMNS]
    stress = historical_stress.loc[common].to_numpy(dtype=float)
    recovery = historical_recovery.loc[common].to_numpy(dtype=float)
    current_p = current_probabilities[REGIME_COLUMNS].to_numpy(dtype=float)
    current_p = current_p / current_p.sum()
    current_s = float(np.clip(current_stress, 0.0, 1.0))
    current_r = float(np.clip(current_recovery, 0.0, 1.0))

    unconditional_mean, unconditional_covariance = _weighted_mean_covariance(
        values, np.ones(len(values), dtype=float)
    )
    macro_mean = np.zeros(len(ASSETS), dtype=float)
    macro_covariance = np.zeros((len(ASSETS), len(ASSETS)), dtype=float)
    stress_covariance = np.zeros_like(macro_covariance)
    return_adjustment = np.zeros(len(ASSETS), dtype=float)
    regime_details: dict[str, Any] = {}

    for regime_index, regime_column in enumerate(REGIME_COLUMNS):
        regime_weights = probabilities[regime_column].to_numpy(dtype=float)
        raw_mean, raw_covariance = _weighted_mean_covariance(
            values, regime_weights
        )
        effective_sample = float(
            regime_weights.sum() ** 2
            / max(float(np.square(regime_weights).sum()), NUMERICAL_EPSILON)
        )
        credibility = effective_sample / (
            effective_sample + stage36.stage35.ONE_CALENDAR_YEAR
        )
        regime_mean = (
            credibility * raw_mean + (1.0 - credibility) * unconditional_mean
        )
        regime_covariance = (
            credibility * raw_covariance
            + (1.0 - credibility) * unconditional_covariance
        )

        stress_mean = float(np.average(stress, weights=regime_weights))
        recovery_mean = float(np.average(recovery, weights=regime_weights))
        predictors = np.column_stack(
            [stress - stress_mean, recovery - recovery_mean]
        )
        beta = np.zeros((2, len(ASSETS)), dtype=float)
        raw_beta = np.zeros_like(beta)
        standard_error = np.full_like(beta, np.inf)
        reliability = np.zeros_like(beta)
        for asset_index in range(len(ASSETS)):
            coefficients, errors = _fit_hac(
                values[:, asset_index], predictors, regime_weights
            )
            raw_beta[:, asset_index] = coefficients[1:]
            standard_error[:, asset_index] = errors[1:]
            for feature_index in range(2):
                reliability[feature_index, asset_index] = (
                    positive_part_signal_reliability(
                        raw_beta[feature_index, asset_index],
                        standard_error[feature_index, asset_index],
                    )
                )
            beta[:, asset_index] = raw_beta[:, asset_index] * reliability[:, asset_index]

        stress_weights = regime_weights * np.clip(stress, 0.0, 1.0)
        _, raw_stress_covariance = _weighted_mean_covariance(
            values, stress_weights
        )
        stress_effective_sample = float(
            stress_weights.sum() ** 2
            / max(float(np.square(stress_weights).sum()), NUMERICAL_EPSILON)
        )
        stress_credibility = stress_effective_sample / (
            stress_effective_sample + stage36.stage35.ONE_CALENDAR_YEAR
        )
        regime_stress_covariance = (
            stress_credibility * raw_stress_covariance
            + (1.0 - stress_credibility) * regime_covariance
        )

        probability = float(current_p[regime_index])
        macro_mean += probability * regime_mean
        macro_covariance += probability * regime_covariance
        stress_covariance += probability * regime_stress_covariance
        return_adjustment += probability * (
            beta[0] * (current_s - stress_mean)
            + beta[1] * (current_r - recovery_mean)
        )
        regime_details[regime_column.removeprefix("p_")] = {
            "effective_sample": effective_sample,
            "credibility": credibility,
            "stress_beta_raw": raw_beta[0].tolist(),
            "stress_beta_hac_se": standard_error[0].tolist(),
            "stress_beta_reliability": reliability[0].tolist(),
            "stress_beta_shrunk": beta[0].tolist(),
            "recovery_beta_raw": raw_beta[1].tolist(),
            "recovery_beta_hac_se": standard_error[1].tolist(),
            "recovery_beta_reliability": reliability[1].tolist(),
            "recovery_beta_shrunk": beta[1].tolist(),
        }

    covariance = (
        (1.0 - current_s) * macro_covariance
        + current_s * stress_covariance
    )
    detail = {
        "macro_expected_return": macro_mean,
        "stress_return_adjustment": return_adjustment,
        "regime_models": regime_details,
    }
    return macro_mean + return_adjustment, _nearest_psd(covariance), detail


def conflict_only_technical_filter(
    economic_expected_return: np.ndarray,
    technical_direction: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Veto only conflicting active views; neutral/aligned signals do nothing.

    The cross-sectional mean is not a cash-return forecast. Under sum(w)=1, a
    common shift to every expected return does not change the optimizer, so it
    is the exact zero point for cross-sectional active views.
    """

    expected = np.asarray(economic_expected_return, dtype=float)
    technical = np.clip(np.asarray(technical_direction, dtype=float), -1.0, 1.0)
    neutral = float(expected.mean())
    active = expected - neutral
    relative_direction = np.sign(active)
    conflict = np.clip(-relative_direction * technical, 0.0, 1.0)
    confidence = 1.0 - conflict
    filtered = neutral + confidence * active
    return filtered, {
        "cross_sectional_neutral": neutral,
        "active_expected_return": active,
        "technical_direction": technical,
        "technical_conflict": conflict,
        "technical_confidence": confidence,
    }


def _standardize_from_history(
    history: pd.DataFrame, current: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    means = history.mean()
    standard_deviations = history.std(ddof=1)
    valid = standard_deviations.gt(NUMERICAL_EPSILON)
    standardized_history = (
        history.loc[:, valid] - means.loc[valid]
    ) / standard_deviations.loc[valid]
    standardized_current = (
        current.loc[valid] - means.loc[valid]
    ) / standard_deviations.loc[valid]
    return standardized_history, standardized_current.to_numpy(dtype=float)


def estimate_credit_adjustment(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    month: pd.Period,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate credit's incremental return effect for every asset.

    Each target-month return is regressed on the credit signal already known
    before that month, controlling for VKOSPI stress/recovery, macro fragility,
    and the asset's lagged return. Only the uncertainty-shrunk credit term is
    added, avoiding a second macro/stress forecast.
    """

    common = returns.index[returns.index < month]
    common = common.intersection(probabilities.index)
    common = common.intersection(stress_signals.index)
    common = common.intersection(fundamental_signals.index)
    base = pd.DataFrame(index=common)
    base["credit"] = fundamental_signals.loc[common, "credit_widening_z"]
    base["stress"] = stress_signals.loc[common, "stress_score"]
    base["recovery"] = stress_signals.loc[common, "recovery_score"]
    base["macro_fragility"] = probabilities.loc[
        common, ["p_Slowdown", "p_Stagflation"]
    ].sum(axis=1)

    current = pd.Series(
        {
            "credit": fundamental_signals.loc[month, "credit_widening_z"],
            "stress": stress_signals.loc[month, "stress_score"],
            "recovery": stress_signals.loc[month, "recovery_score"],
            "macro_fragility": probabilities.loc[
                month, ["p_Slowdown", "p_Stagflation"]
            ].sum(),
        },
        dtype=float,
    )
    adjustments = np.zeros(len(ASSETS), dtype=float)
    details: dict[str, Any] = {}
    for asset_index, asset in enumerate(ASSETS):
        frame = base.copy()
        frame["recent_return"] = returns[asset].shift(1).reindex(common)
        frame["target"] = returns.loc[common, asset]
        complete = frame.dropna()
        asset_current = current.copy()
        asset_current["recent_return"] = (
            float(returns.loc[returns.index < month, asset].iloc[-1])
            if len(returns.loc[returns.index < month])
            else np.nan
        )
        if len(complete) < MIN_CALIBRATION_MONTHS or not np.isfinite(asset_current).all():
            details[asset] = {
                "observations": int(len(complete)),
                "last_training_month": (
                    complete.index.max() if len(complete) else None
                ),
                "credit_beta": 0.0,
                "credit_hac_se": None,
                "credit_reliability": 0.0,
                "adjustment": 0.0,
            }
            continue
        predictors = complete.drop(columns="target")
        # Credit is already a causal z-score whose zero has an economic meaning:
        # no widening relative to the information set. Keep that unit and
        # standardize only nuisance controls from the historical fit sample.
        standardized = predictors[["credit"]].copy()
        standardized_current_values: dict[str, float] = {
            "credit": float(asset_current["credit"])
        }
        controls, current_controls = _standardize_from_history(
            predictors.drop(columns="credit"),
            asset_current[predictors.drop(columns="credit").columns],
        )
        for control_index, control in enumerate(controls.columns):
            standardized[control] = controls[control]
            standardized_current_values[control] = float(
                current_controls[control_index]
            )
        standardized_current = np.array(
            [standardized_current_values[column] for column in standardized],
            dtype=float,
        )
        coefficients, errors = _fit_hac(
            complete.loc[standardized.index, "target"].to_numpy(dtype=float),
            standardized.to_numpy(dtype=float),
        )
        credit_index = list(standardized.columns).index("credit") + 1
        beta = float(coefficients[credit_index])
        standard_error = float(errors[credit_index])
        reliability = positive_part_signal_reliability(beta, standard_error)
        shrunk_beta = beta * reliability
        current_credit = float(asset_current["credit"])
        adjustment = shrunk_beta * current_credit
        adjustments[asset_index] = adjustment
        details[asset] = {
            "observations": int(len(complete)),
            "last_training_month": complete.index.max(),
            "credit_beta": beta,
            "credit_hac_se": standard_error,
            "credit_reliability": reliability,
            "credit_current_standardized": current_credit,
            "adjustment": adjustment,
        }
    return adjustments, details


def build_asset_volatility_signals(
    daily: pd.DataFrame, target_months: pd.PeriodIndex
) -> pd.DataFrame:
    """Keep only the lagged ranks needed by Stage 48; no 1+rank multiplier."""

    rows: list[dict[str, Any]] = []
    for target_month in target_months:
        cutoff = (target_month - 1).to_timestamp("M")
        row: dict[str, Any] = {
            "target_month": target_month,
            "asset_vol_signal_month": target_month - 1,
        }
        for sensor in ("gvz", "ovx"):
            columns = [
                sensor,
                f"{sensor}_causal_rank",
                f"{sensor}_prior_valid_observations",
            ]
            known = daily.loc[:cutoff, columns].dropna(subset=[sensor])
            if known.empty:
                level, rank, count, date = np.nan, np.nan, 0, pd.NaT
            else:
                current = known.iloc[-1]
                level = float(current[sensor])
                rank = float(current[f"{sensor}_causal_rank"])
                count = int(current[f"{sensor}_prior_valid_observations"])
                date = known.index[-1]
            row.update(
                {
                    f"{sensor}_signal_date": date,
                    f"{sensor}_level": level,
                    f"{sensor}_causal_rank": rank,
                    f"{sensor}_active": bool(
                        np.isfinite(rank) and count >= stage36.MIN_SENSOR_HISTORY
                    ),
                }
            )
        rows.append(row)
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def build_daily_return_matrix(
    market: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    close = pd.concat(
        {
            asset: market[asset]["close"].dropna().astype(float)
            for asset in ASSETS
        },
        axis=1,
    ).sort_index()
    calendar = pd.date_range(close.index.min(), close.index.max(), freq="B")
    levels = close.reindex(calendar).ffill(limit=5)
    return levels.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )[ASSETS]


def monthly_realized_variance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    grouped = daily_returns.groupby(daily_returns.index.to_period("M"))
    variance = grouped.var(ddof=1) * TRADING_DAYS_PER_MONTH
    counts = grouped.count()
    return variance.where(counts >= 15)


def current_variance_features(
    technical_signal: pd.Series,
    fundamental_signal: pd.Series,
    asset_vol_signal: pd.Series,
) -> dict[str, dict[str, float]]:
    """Return bounded risk-state features, centered at their median rank."""

    credit_rank = float(fundamental_signal["credit_stress_rank"])
    credit_state = (
        float(np.clip(credit_rank, 0.0, 1.0) - 0.5)
        if np.isfinite(credit_rank)
        else 0.0
    )
    output: dict[str, dict[str, float]] = {}
    for asset in ASSETS:
        atr_rank = float(technical_signal[f"atr_percentile_{asset}"])
        atr_state = (
            float(np.clip(atr_rank, 0.0, 1.0) - 0.5)
            if np.isfinite(atr_rank)
            else 0.0
        )
        iv_state = 0.0
        if asset == "GLD" and bool(asset_vol_signal["gvz_active"]):
            iv_state = float(
                np.clip(asset_vol_signal["gvz_causal_rank"], 0.0, 1.0)
                - 0.5
            )
        elif asset == "USO" and bool(asset_vol_signal["ovx_active"]):
            iv_state = float(
                np.clip(asset_vol_signal["ovx_causal_rank"], 0.0, 1.0)
                - 0.5
            )
        output[asset] = {
            "atr_state": atr_state,
            "credit_state": credit_state,
            "iv_state": iv_state,
        }
    return output


def calibrated_variance_multipliers(
    records: list[dict[str, Any]],
    current_features: dict[str, dict[str, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict one residual variance ratio per asset from prior months only.

    The response is log(realized variance / base predicted variance). ATR and
    credit ranks are bounded state variables; GVZ/OVX rank is included only for
    its economically matched asset. Coefficients are HAC-shrunk, the forecast
    is kept inside the historically observed target support, and exactly one
    D Sigma D correction is returned.
    """

    frame = pd.DataFrame(records)
    multipliers = np.ones(len(ASSETS), dtype=float)
    details: dict[str, Any] = {}
    for asset_index, asset in enumerate(ASSETS):
        columns = ["atr_state", "credit_state"]
        if asset in {"GLD", "USO"}:
            columns.append("iv_state")
        if frame.empty:
            asset_history = pd.DataFrame()
        else:
            asset_history = frame.loc[frame["asset"].eq(asset)].dropna(
                subset=["target_log_ratio", *columns]
            )
        if len(asset_history) < MIN_CALIBRATION_MONTHS:
            details[asset] = {
                "observations": int(len(asset_history)),
                "last_training_month": (
                    asset_history["decision_month"].max()
                    if len(asset_history)
                    else None
                ),
                "active": False,
                "multiplier": 1.0,
                "prediction_log_ratio": 0.0,
                "coefficients": {},
                "reliability": {},
            }
            continue
        coefficients, errors = _fit_hac(
            asset_history["target_log_ratio"].to_numpy(dtype=float),
            asset_history[columns].to_numpy(dtype=float),
        )
        names = ["intercept", *columns]
        reliability = np.array(
            [
                positive_part_signal_reliability(coefficient, error)
                for coefficient, error in zip(coefficients, errors)
            ],
            dtype=float,
        )
        shrunk = coefficients * reliability
        current = np.array(
            [1.0, *[current_features[asset][column] for column in columns]],
            dtype=float,
        )
        raw_prediction = float(current @ shrunk)
        lower = float(asset_history["target_log_ratio"].min())
        upper = float(asset_history["target_log_ratio"].max())
        prediction = float(np.clip(raw_prediction, lower, upper))
        multiplier = float(math.exp(prediction))
        multipliers[asset_index] = multiplier
        details[asset] = {
            "observations": int(len(asset_history)),
            "last_training_month": asset_history["decision_month"].max(),
            "active": True,
            "multiplier": multiplier,
            "prediction_log_ratio": prediction,
            "raw_prediction_log_ratio": raw_prediction,
            "support_lower": lower,
            "support_upper": upper,
            "coefficients": {
                name: float(value) for name, value in zip(names, coefficients)
            },
            "hac_standard_errors": {
                name: float(value) for name, value in zip(names, errors)
            },
            "reliability": {
                name: float(value) for name, value in zip(names, reliability)
            },
            "shrunk_coefficients": {
                name: float(value) for name, value in zip(names, shrunk)
            },
        }
    return multipliers, details


def solve_weights(
    history: pd.DataFrame,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the unchanged Stage 36 long-only economic objective."""

    historical_returns = history[ASSETS].to_numpy(dtype=float)
    initial = (
        stage36.stage35.project_to_long_only_simplex(pretrade)
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
        transaction_cost = stage36.stage35.expected_transaction_cost(
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
            "downside_semivariance": downside_semivariance,
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": utility,
        }

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
                - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage36.stage35.CATASTROPHE_CDAR
                + stage36.stage35.cdar(
                    historical_returns @ weights,
                    stage36.stage35.CDAR_CONFIDENCE,
                )
            ),
        },
    ]
    result = minimize(
        lambda weights: -portfolio_values(weights)["monthly_utility"],
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
                f"Economic and fallback solves failed: {result.message}; "
                f"{fallback.message}"
            )
        result = fallback
        weights = stage36.stage35.project_to_long_only_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_volatility_value = annual_volatility(weights)
    historical_cdar = stage36.stage35.cdar(
        historical_returns @ weights, stage36.stage35.CDAR_CONFIDENCE
    )
    return weights, {
        **values,
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_volatility_value,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": (
            stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
            - annual_volatility_value
        ),
        "cdar_slack": stage36.stage35.CATASTROPHE_CDAR + historical_cdar,
    }


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    realized_variance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_widening_z",
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
    variance_records: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < stage36.stage35.ONE_CALENDAR_YEAR:
            continue
        probability = probabilities.loc[month]
        stress = float(stress_signals.loc[month, "stress_score"])
        recovery = float(stress_signals.loc[month, "recovery_score"])
        technical_signal = technical_signals.loc[month]
        fundamental_signal = fundamental_signals.loc[month]
        asset_vol_signal = asset_vol_signals.loc[month]

        base_expected, base_covariance, moment_detail = estimate_evidence_moments(
            history,
            probabilities.loc[probabilities.index < month],
            probability,
            stress_signals.loc[stress_signals.index < month, "stress_score"],
            stress,
            stress_signals.loc[
                stress_signals.index < month, "recovery_score"
            ],
            recovery,
        )
        credit_adjustment, credit_detail = estimate_credit_adjustment(
            returns, probabilities, stress_signals, fundamental_signals, month
        )
        fundamental_adjustment = np.zeros(len(ASSETS), dtype=float)
        equity_index = ASSETS.index("KODEX200")
        fundamental_adjustment[equity_index] = float(
            fundamental_signal["eps_mu_adjustment_KODEX200"]
            + fundamental_signal["valuation_mu_adjustment_KODEX200"]
        )
        economic_expected = (
            base_expected + fundamental_adjustment + credit_adjustment
        )
        technical_direction = np.array(
            [
                float(technical_signal[f"technical_direction_{asset}"])
                for asset in ASSETS
            ],
            dtype=float,
        )
        expected_return, technical_detail = conflict_only_technical_filter(
            economic_expected, technical_direction
        )

        variance_features = current_variance_features(
            technical_signal, fundamental_signal, asset_vol_signal
        )
        variance_multipliers, variance_detail = (
            calibrated_variance_multipliers(
                variance_records, variance_features
            )
        )
        scaling = np.diag(np.sqrt(variance_multipliers))
        covariance = _nearest_psd(scaling @ base_covariance @ scaling)
        weights, solve_detail = solve_weights(
            history, expected_return, covariance, pretrade
        )

        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = (
            float(np.abs(change).sum())
            * stage36.stage35.DOMESTIC_TRADE_COST
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

        row: dict[str, Any] = {
            "month": month,
            "macro_signal_month": probability["signal_month"],
            "stress_signal_month": stress_signals.loc[
                month, "stress_signal_month"
            ],
            "technical_signal_month": technical_signal[
                "technical_signal_month"
            ],
            "fundamental_signal_month": fundamental_signal[
                "fundamental_signal_month"
            ],
            "asset_vol_signal_month": asset_vol_signal[
                "asset_vol_signal_month"
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
            "base_covariance_min_eigenvalue": float(
                np.linalg.eigvalsh(base_covariance).min()
            ),
            "adjusted_covariance_min_eigenvalue": float(
                np.linalg.eigvalsh(covariance).min()
            ),
            "variance_calibration_prior_records": int(len(variance_records)),
            **{
                column: float(probability[column])
                for column in REGIME_COLUMNS
            },
            **{
                f"w_{asset}": float(weights[index])
                for index, asset in enumerate(ASSETS)
            },
            **solve_detail,
        }
        for asset_index, asset in enumerate(ASSETS):
            row[f"macro_mu_{asset}"] = float(
                moment_detail["macro_expected_return"][asset_index]
            )
            row[f"stress_mu_adjustment_{asset}"] = float(
                moment_detail["stress_return_adjustment"][asset_index]
            )
            row[f"fundamental_mu_adjustment_{asset}"] = float(
                fundamental_adjustment[asset_index]
            )
            row[f"credit_mu_adjustment_{asset}"] = float(
                credit_adjustment[asset_index]
            )
            row[f"economic_mu_before_technical_{asset}"] = float(
                economic_expected[asset_index]
            )
            row[f"technical_confidence_{asset}"] = float(
                technical_detail["technical_confidence"][asset_index]
            )
            row[f"expected_mu_{asset}"] = float(expected_return[asset_index])
            row[f"variance_multiplier_{asset}"] = float(
                variance_multipliers[asset_index]
            )
            row[f"variance_calibration_observations_{asset}"] = int(
                variance_detail[asset]["observations"]
            )
            row[f"variance_calibration_last_training_month_{asset}"] = (
                variance_detail[asset]["last_training_month"]
            )
            row[f"credit_calibration_observations_{asset}"] = int(
                credit_detail.get(asset, {}).get("observations", 0)
            )
            row[f"credit_calibration_last_training_month_{asset}"] = (
                credit_detail.get(asset, {}).get("last_training_month")
            )
            row[f"credit_beta_reliability_{asset}"] = float(
                credit_detail.get(asset, {}).get("credit_reliability", 0.0)
            )
            row[f"atr_state_{asset}"] = variance_features[asset]["atr_state"]
            row[f"credit_state_{asset}"] = variance_features[asset][
                "credit_state"
            ]
            row[f"iv_state_{asset}"] = variance_features[asset]["iv_state"]
        row["cross_sectional_neutral"] = float(
            technical_detail["cross_sectional_neutral"]
        )
        row["eps_mu_adjustment_KODEX200"] = float(
            fundamental_signal["eps_mu_adjustment_KODEX200"]
        )
        row["valuation_mu_adjustment_KODEX200"] = float(
            fundamental_signal["valuation_mu_adjustment_KODEX200"]
        )
        rows.append(row)

        if month in realized_variance.index:
            for asset_index, asset in enumerate(ASSETS):
                realized = float(realized_variance.loc[month, asset])
                base_variance = float(base_covariance[asset_index, asset_index])
                if (
                    np.isfinite(realized)
                    and realized > NUMERICAL_EPSILON
                    and base_variance > NUMERICAL_EPSILON
                ):
                    variance_records.append(
                        {
                            "decision_month": month,
                            "asset": asset,
                            "target_log_ratio": math.log(
                                realized / base_variance
                            ),
                            "realized_variance": realized,
                            "base_predicted_variance": base_variance,
                            **variance_features[asset],
                        }
                    )

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    records = pd.DataFrame(variance_records)
    return output, records


def _performance_table(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    common_start = max(baseline.index.min(), candidate.index.min())
    common_end = min(baseline.index.max(), candidate.index.max())
    active = candidate.index[
        candidate["variance_calibration_observations_KODEX200"].ge(
            MIN_CALIBRATION_MONTHS
        )
    ]
    periods: dict[str, tuple[pd.Period, pd.Period]] = {
        "full_common": (common_start, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    if len(active):
        periods["variance_calibration_active"] = (
            max(active.min(), common_start),
            common_end,
        )
    rows: list[dict[str, Any]] = []
    for name, path in (
        ("Stage36_GVZ_OVXAssetRisk", baseline),
        ("Stage48_EvidenceCalibrated", candidate),
    ):
        for period, (start, end) in periods.items():
            rows.append(
                stage36.stage35.metric_row(name, path, period, start, end)
            )
    return pd.DataFrame(rows)


def _bootstrap_table(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    common_start = max(baseline.index.min(), candidate.index.min())
    common_end = min(baseline.index.max(), candidate.index.max())
    periods = {
        "full_common": (common_start, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    rows: list[pd.DataFrame] = []
    for period, (start, end) in periods.items():
        result = stage36.stage35.stage30.paired_block_bootstrap(
            baseline.loc[start:end, "return"],
            candidate.loc[start:end, "return"],
            replications=2000,
            block_months=12,
        )
        result.insert(0, "Period", period)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _comparison_value(
    performance: pd.DataFrame,
    strategy: str,
    period: str,
    metric: str,
) -> float:
    value = performance.loc[
        performance["Strategy"].eq(strategy)
        & performance["Period"].eq(period),
        metric,
    ]
    return float(value.iloc[0])


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "rows": int(len(value)),
            "columns": [str(column) for column in value.columns],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "rows": int(len(value)),
            "name": str(value.name),
        }
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Period, pd.Timestamp, Path)):
        return str(value)
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return value


def run_research(save: bool = True) -> dict[str, Any]:
    frozen_before = frozen_manifest()

    implied_daily, implied_audit = stage36.load_asset_implied_volatility_daily()
    returns, return_audit = stage36.stage35.load_monthly_asset_returns(False)
    probabilities, macro_audit = stage36.stage35.build_macro_probabilities(
        returns
    )
    stress_signals = stage36.stage35.build_monthly_stress_signals(
        returns.index, stage36.stage35.build_daily_stress_features()
    )
    market, market_audit = stage36.stage35.stage20.load_daily_asset_ohlcv()

    raw_fundamental, fundamental_audit = (
        stage36.stage35.load_fundamental_daily()
    )
    fundamental_signals = (
        stage36.stage35.build_monthly_fundamental_signals(raw_fundamental)
    )
    equity_monthly_close = (
        market["KODEX200"]["close"]
        .dropna()
        .groupby(market["KODEX200"]["close"].dropna().index.to_period("M"))
        .last()
    )
    fundamental_signals = stage36.stage35.add_causal_return_calibration(
        fundamental_signals, equity_monthly_close.pct_change()
    )
    technical_signals = stage36.stage35.stage34._load_period_csv(
        stage36.stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    asset_vol_signals = build_asset_volatility_signals(
        implied_daily, returns.index
    )
    daily_returns = build_daily_return_matrix(market)
    realized_variance = monthly_realized_variance(daily_returns)

    candidate, variance_records = run_backtest(
        returns,
        probabilities,
        stress_signals,
        technical_signals,
        fundamental_signals,
        asset_vol_signals,
        realized_variance,
    )
    baseline = stage36.stage35.stage34._load_period_csv(
        BASELINE_PATH, "month"
    )
    performance = _performance_table(baseline, candidate)
    bootstrap = _bootstrap_table(baseline, candidate)

    weights = candidate[WEIGHT_COLUMNS]
    multiplier_columns = [
        f"variance_multiplier_{asset}" for asset in ASSETS
    ]
    signal_columns = [
        "macro_signal_month",
        "stress_signal_month",
        "technical_signal_month",
        "fundamental_signal_month",
        "asset_vol_signal_month",
    ]
    signal_lag_checks = {
        column: bool(
            (
                pd.PeriodIndex(candidate[column], freq="M")
                < candidate.index
            ).all()
        )
        for column in signal_columns
    }
    model_lag_checks: dict[str, bool] = {}
    for model in ("variance", "credit"):
        for asset in ASSETS:
            column = f"{model}_calibration_last_training_month_{asset}"
            available = candidate[column].notna()
            model_lag_checks[f"{model}_{asset}"] = bool(
                (
                    pd.PeriodIndex(candidate.loc[available, column], freq="M")
                    < candidate.index[available]
                ).all()
            )
    checks = {
        "stage36_frozen_files_unchanged": frozen_before == frozen_manifest(),
        "weights_finite": bool(np.isfinite(weights.to_numpy()).all()),
        "weights_long_only": bool(weights.min().min() >= -1e-8),
        "weights_fully_invested": bool(
            np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        ),
        "volatility_guard_satisfied": bool(
            candidate["volatility_slack"].min() >= -1e-7
        ),
        "cdar_guard_satisfied": bool(
            candidate["cdar_slack"].min() >= -1e-7
        ),
        "base_covariance_psd": bool(
            candidate["base_covariance_min_eigenvalue"].min() > 0.0
        ),
        "adjusted_covariance_psd": bool(
            candidate["adjusted_covariance_min_eigenvalue"].min() > 0.0
        ),
        "variance_multipliers_positive_finite": bool(
            np.isfinite(candidate[multiplier_columns].to_numpy()).all()
            and candidate[multiplier_columns].min().min() > 0.0
        ),
        "signals_strictly_lagged": bool(all(signal_lag_checks.values())),
        "calibration_training_strictly_precedes_decision": bool(
            all(model_lag_checks.values())
        ),
        "credit_model_covers_all_assets": bool(
            all(
                f"credit_mu_adjustment_{asset}" in candidate
                for asset in ASSETS
            )
        ),
        "variance_calibration_waits_60_months": bool(
            all(
                candidate.loc[
                    candidate[
                        f"variance_calibration_observations_{asset}"
                    ].lt(MIN_CALIBRATION_MONTHS),
                    f"variance_multiplier_{asset}",
                ].eq(1.0).all()
                for asset in ASSETS
            )
        ),
        "returns_finite": bool(np.isfinite(candidate["return"]).all()),
    }
    stage36_full_sharpe = _comparison_value(
        performance,
        "Stage36_GVZ_OVXAssetRisk",
        "full_common",
        "Sharpe",
    )
    stage48_full_sharpe = _comparison_value(
        performance,
        "Stage48_EvidenceCalibrated",
        "full_common",
        "Sharpe",
    )
    stage36_locked_mdd = _comparison_value(
        performance,
        "Stage36_GVZ_OVXAssetRisk",
        "locked_2018_2026",
        "MDD",
    )
    stage48_locked_mdd = _comparison_value(
        performance,
        "Stage48_EvidenceCalibrated",
        "locked_2018_2026",
        "MDD",
    )
    gate = {
        "causal_and_feasible": bool(all(checks.values())),
        "full_sharpe_not_below_stage36": bool(
            stage48_full_sharpe >= stage36_full_sharpe
        ),
        "locked_mdd_not_worse_than_stage36": bool(
            stage48_locked_mdd >= stage36_locked_mdd
        ),
    }
    gate["promotion_pass"] = bool(all(gate.values()))

    diagnostics = {
        "months": int(len(candidate)),
        "start": str(candidate.index.min()),
        "end": str(candidate.index.max()),
        "variance_records": int(len(variance_records)),
        "variance_multiplier_summary": {
            asset: {
                "mean": float(candidate[f"variance_multiplier_{asset}"].mean()),
                "median": float(
                    candidate[f"variance_multiplier_{asset}"].median()
                ),
                "minimum": float(candidate[f"variance_multiplier_{asset}"].min()),
                "maximum": float(candidate[f"variance_multiplier_{asset}"].max()),
                "first_active": (
                    str(
                        candidate.index[
                            candidate[
                                f"variance_calibration_observations_{asset}"
                            ].ge(MIN_CALIBRATION_MONTHS)
                        ].min()
                    )
                    if candidate[
                        f"variance_calibration_observations_{asset}"
                    ].ge(MIN_CALIBRATION_MONTHS).any()
                    else None
                ),
            }
            for asset in ASSETS
        },
        "credit_adjustment_nonzero_months": {
            asset: int(
                candidate[f"credit_mu_adjustment_{asset}"].abs().gt(1e-15).sum()
            )
            for asset in ASSETS
        },
        "credit_mean_reliability": {
            asset: float(candidate[f"credit_beta_reliability_{asset}"].mean())
            for asset in ASSETS
        },
        "average_weights": {
            asset: float(candidate[f"w_{asset}"].mean()) for asset in ASSETS
        },
        "fallback_months": int(candidate["used_fallback"].sum()),
        "signal_lag_checks": signal_lag_checks,
        "model_training_lag_checks": model_lag_checks,
    }
    report = {
        "strategy": "Stage48_EvidenceCalibrated",
        "design": {
            "stress_mu": "joint WLS by soft regime, HAC(1) SE, positive-part signal-to-noise shrinkage",
            "credit_mu": "incremental all-asset HAC model after stress/recovery/macro/recent-return controls",
            "technical_mu": "conflict-only veto around cross-sectional active-return zero point",
            "variance": "one causal log realized/base variance calibration followed by one D Sigma D map",
            "rank_vs_zscore": "bounded causal ranks for monotone risk states; signed causal z-score for return alpha",
            "removed": [
                "beta times R-squared slope scaling",
                "hard stress/recovery sign restrictions",
                "credit multiplication of VKOSPI return adjustment",
                "KODEX200-only credit variance scaling",
                "sequential ATR/credit/GVZ/OVX 1+rank variance multipliers",
                "technical neutral-signal half-shrink",
                "unused Stage36 regression reports, candidate modes, plots, and notebook paths",
            ],
        },
        "checks": checks,
        "gate": gate,
        "diagnostics": diagnostics,
        "data_audit": {
            "returns": return_audit,
            "macro": macro_audit,
            "market": market_audit,
            "fundamental": fundamental_audit,
            "implied_volatility": implied_audit,
        },
        "frozen_manifest": frozen_manifest(),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(OUTPUT_DIR / "stage48_monthly.csv")
        variance_records.to_csv(
            OUTPUT_DIR / "variance_calibration_history.csv", index=False
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv", index=False
        )
        with (OUTPUT_DIR / "validation_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                _json_ready(report),
                handle,
                ensure_ascii=False,
                indent=2,
            )
    return {
        "path": candidate,
        "variance_records": variance_records,
        "performance": performance,
        "bootstrap": bootstrap,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["performance"].to_string(index=False))
    print(json.dumps(_json_ready(result["report"]["gate"]), indent=2))


if __name__ == "__main__":
    main()
