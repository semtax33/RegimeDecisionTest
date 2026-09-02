from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import BayesianRidge

from strategies.stage45_volatility_targeted_shrinkage_mlp import (
    volatility_targeted_shrinkage_mlp as stage45,
)
from strategies.stage48_evidence_calibrated_stage36 import (
    evidence_calibrated_stage36 as stage48,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = stage48.ASSETS
WEIGHT_COLUMNS = stage48.WEIGHT_COLUMNS
FULL_START = stage48.FULL_START
LOCKED_START = stage48.LOCKED_START
RESEARCH_END = stage48.RESEARCH_END
REGIME_COLUMNS = stage48.REGIME_COLUMNS
TRADING_DAYS_PER_MONTH = 21.0
COVARIANCE_LOOKBACK_DAYS = 252
OBSERVATIONS_PER_PARAMETER = 5
NUMERICAL_EPSILON = 1e-12

STAGE36_PATH = stage48.BASELINE_PATH
STAGE48_PATH = stage48.OUTPUT_DIR / "stage48_monthly.csv"
FROZEN_FILES = (
    Path(stage48.__file__),
    STAGE48_PATH,
    stage48.OUTPUT_DIR / "validation_report.json",
    Path(stage48.stage36.__file__),
    STAGE36_PATH,
    stage48.stage36.OUTPUT_DIR / "validation_report.json",
)

MODE_MACRO = "Stage49_MacroBayes"
MODE_RETURN = "Stage49_ReturnEvidence"
MODE_FULL = "Stage49_FullPrequential"
MODES = (MODE_MACRO, MODE_RETURN, MODE_FULL)
RETURN_BLOCKS = (
    "stress_recovery",
    "credit",
    "technical",
    "fundamental",
)

BASE_RETURN_FEATURES = (
    "p_Overheating",
    "p_Slowdown",
    "p_Stagflation",
)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


@dataclass
class ForecastFit:
    prediction: float
    prediction_std: float
    observations: int
    last_training_month: pd.Period | None


def _fit_bayesian_forecast(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: tuple[str, ...],
    current: pd.Series,
) -> ForecastFit | None:
    """Fit a causal empirical-Bayes ridge forecast without a tuned penalty.

    BayesianRidge estimates coefficient and noise precision by marginal
    likelihood. The five-observations-per-parameter rule is a declared design
    sufficiency condition, replacing Stage 48's fixed 60 months regardless of
    model dimension.
    """

    complete = frame[[target_column, *feature_columns]].dropna()
    minimum = OBSERVATIONS_PER_PARAMETER * (len(feature_columns) + 1)
    if len(complete) < minimum or current[list(feature_columns)].isna().any():
        return None
    predictors = complete.loc[:, feature_columns].astype(float)
    means = predictors.mean()
    standard_deviations = predictors.std(ddof=1)
    usable = standard_deviations.gt(NUMERICAL_EPSILON)
    if not usable.any():
        return None
    columns = tuple(standard_deviations.index[usable])
    x = (
        (predictors.loc[:, columns] - means.loc[list(columns)])
        / standard_deviations.loc[list(columns)]
    )
    current_x = (
        (current.loc[list(columns)] - means.loc[list(columns)])
        / standard_deviations.loc[list(columns)]
    )
    model = BayesianRidge(
        fit_intercept=True,
        compute_score=True,
        tol=1e-7,
        max_iter=1_000,
    )
    model.fit(
        x.to_numpy(dtype=float),
        complete.loc[x.index, target_column].to_numpy(dtype=float),
    )
    prediction, prediction_std = model.predict(
        current_x.to_numpy(dtype=float).reshape(1, -1),
        return_std=True,
    )
    return ForecastFit(
        prediction=float(prediction[0]),
        prediction_std=float(prediction_std[0]),
        observations=int(len(complete)),
        last_training_month=complete.index.max(),
    )


def prequential_return_stacking_weight(
    records: pd.DataFrame,
) -> dict[str, float | int]:
    """Find the convex base/full mixture minimizing prior OOS squared error."""

    if records.empty:
        return {
            "weight": 0.0,
            "observations": 0,
            "mean_loss_improvement": 0.0,
        }
    complete = records[
        ["actual_return", "base_forecast", "full_forecast"]
    ].dropna()
    observations = int(len(complete))
    if observations == 0:
        return {
            "weight": 0.0,
            "observations": 0,
            "mean_loss_improvement": 0.0,
        }
    actual = complete["actual_return"].to_numpy(dtype=float)
    base = complete["base_forecast"].to_numpy(dtype=float)
    full = complete["full_forecast"].to_numpy(dtype=float)
    difference = full - base
    denominator = float(difference @ difference)
    weight = (
        float(
            np.clip(
                difference @ (actual - base) / denominator,
                0.0,
                1.0,
            )
        )
        if denominator > NUMERICAL_EPSILON
        else 0.0
    )
    base_loss = np.square(actual - base)
    full_loss = np.square(actual - full)
    return {
        "weight": weight,
        "observations": observations,
        "mean_loss_improvement": float(
            np.mean(base_loss - full_loss)
        ),
    }


def prequential_variance_stacking_weight(
    records: pd.DataFrame,
) -> dict[str, float | int]:
    """Find the convex variance mixture minimizing prior OOS QLIKE."""

    if records.empty:
        return {
            "weight": 0.0,
            "observations": 0,
            "mean_loss_improvement": 0.0,
        }
    complete = records[
        [
            "realized_variance",
            "base_variance_forecast",
            "full_variance_forecast",
            "base_qlike",
            "full_qlike",
        ]
    ].dropna()
    observations = int(len(complete))
    if observations == 0:
        return {
            "weight": 0.0,
            "observations": 0,
            "mean_loss_improvement": 0.0,
        }
    realized = complete["realized_variance"].to_numpy(dtype=float)
    base = complete["base_variance_forecast"].to_numpy(dtype=float)
    full = complete["full_variance_forecast"].to_numpy(dtype=float)

    def objective(weight: float) -> float:
        forecast = (1.0 - weight) * base + weight * full
        ratio = realized / forecast
        return float(np.mean(ratio - np.log(ratio) - 1.0))

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    weight = float(np.clip(result.x, 0.0, 1.0)) if result.success else 0.0
    return {
        "weight": weight,
        "observations": observations,
        "mean_loss_improvement": float(
            (complete["base_qlike"] - complete["full_qlike"]).mean()
        ),
    }


def qlike_loss(realized_variance: float, forecast_variance: float) -> float:
    """Patton's scale-free QLIKE loss, normalized to zero at a perfect forecast."""

    realized = float(realized_variance)
    forecast = float(forecast_variance)
    if realized <= 0.0 or forecast <= 0.0:
        return math.inf
    ratio = realized / forecast
    return float(ratio - math.log(ratio) - 1.0)


def build_return_feature_frame(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
) -> pd.DataFrame:
    frame = probabilities.loc[:, REGIME_COLUMNS].copy()
    frame["stress_score"] = stress_signals["stress_score"]
    frame["recovery_score"] = stress_signals["recovery_score"]
    frame["credit_widening_z"] = fundamental_signals["credit_widening_z"]
    frame["eps_revision_z"] = fundamental_signals["eps_revision_z"]
    frame["valuation_gap_z"] = fundamental_signals["valuation_gap_z"]
    for asset in ASSETS:
        frame[f"technical_direction_{asset}"] = technical_signals[
            f"technical_direction_{asset}"
        ]
        frame[f"return_{asset}"] = returns[asset]
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index().replace([np.inf, -np.inf], np.nan)


def _return_model_specs(asset: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return an economically ordered nested forecast hierarchy."""

    macro = BASE_RETURN_FEATURES
    stress = (*macro, "stress_score", "recovery_score")
    credit = (*stress, "credit_widening_z")
    technical = (*credit, f"technical_direction_{asset}")
    specs = [
        ("macro", macro),
        ("stress_recovery", stress),
        ("credit", credit),
        ("technical", technical),
    ]
    if asset == "KODEX200":
        specs.append(
            (
                "fundamental",
                (*technical, "eps_revision_z", "valuation_gap_z"),
            )
        )
    return specs


def return_forecasts_for_month(
    feature_frame: pd.DataFrame,
    month: pd.Period,
    forecast_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build nested forecasts and validate every incremental signal block.

    Credit is compared with macro+stress/recovery, not with an unrelated null;
    technical is compared with the otherwise identical model including credit.
    Consequently a different signal block cannot provide the evidence weight
    that activates credit or technical.
    """

    base_mu = np.zeros(len(ASSETS), dtype=float)
    evidence_mu = np.zeros(len(ASSETS), dtype=float)
    details: dict[str, Any] = {}
    history = feature_frame.loc[feature_frame.index < month]
    current = feature_frame.loc[month]
    records = pd.DataFrame(forecast_records)
    for asset_index, asset in enumerate(ASSETS):
        target = f"return_{asset}"
        specs = _return_model_specs(asset)
        macro_fit = _fit_bayesian_forecast(
            history, target, specs[0][1], current
        )
        if macro_fit is None:
            available = history[target].dropna()
            macro_prediction = float(available.mean())
            macro_observations = int(len(available))
            macro_last_month = available.index.max()
            macro_std = float(available.std(ddof=1))
        else:
            macro_prediction = macro_fit.prediction
            macro_observations = macro_fit.observations
            macro_last_month = macro_fit.last_training_month
            macro_std = macro_fit.prediction_std

        prior_prediction = macro_prediction
        final_prediction = macro_prediction
        blocks: dict[str, Any] = {}
        for block, features in specs[1:]:
            model_fit = _fit_bayesian_forecast(
                history, target, features, current
            )
            if model_fit is None:
                model_prediction = prior_prediction
                prediction_std = math.nan
                training_observations = 0
                last_training_month = None
            else:
                model_prediction = model_fit.prediction
                prediction_std = model_fit.prediction_std
                training_observations = model_fit.observations
                last_training_month = model_fit.last_training_month
            if records.empty:
                block_records = pd.DataFrame()
            else:
                block_records = records.loc[
                    records["asset"].eq(asset)
                    & records["block"].eq(block)
                    & records["full_forecast_available"].eq(True)
                ]
            evidence = prequential_return_stacking_weight(block_records)
            weight = float(evidence["weight"]) if model_fit is not None else 0.0
            raw_increment = model_prediction - prior_prediction
            weighted_increment = weight * raw_increment
            final_prediction += weighted_increment
            blocks[block] = {
                "base_prediction": prior_prediction,
                "model_prediction": model_prediction,
                "prediction_std": prediction_std,
                "training_observations": training_observations,
                "last_training_month": last_training_month,
                "full_forecast_available": model_fit is not None,
                "evidence": evidence,
                "raw_increment": raw_increment,
                "weighted_increment": weighted_increment,
            }
            prior_prediction = model_prediction

        base_mu[asset_index] = macro_prediction
        evidence_mu[asset_index] = final_prediction
        details[asset] = {
            "base_prediction": macro_prediction,
            "base_prediction_std": macro_std,
            "base_observations": macro_observations,
            "base_last_training_month": macro_last_month,
            "blocks": blocks,
            "final_prediction": final_prediction,
        }
    return base_mu, evidence_mu, details


def covariance_for_month(
    daily_returns: pd.DataFrame, target_month: pd.Period
) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff = (target_month - 1).to_timestamp("M")
    window = daily_returns.loc[:cutoff, ASSETS].dropna(how="any").tail(
        COVARIANCE_LOOKBACK_DAYS
    )
    if len(window) < COVARIANCE_LOOKBACK_DAYS:
        raise ValueError(
            f"{target_month} has {len(window)} prior complete daily rows; "
            f"{COVARIANCE_LOOKBACK_DAYS} required"
        )
    daily_covariance, shrinkage = stage45.ledoit_wolf_constant_correlation(
        window.to_numpy(dtype=float)
    )
    covariance = stage48._nearest_psd(
        daily_covariance * TRADING_DAYS_PER_MONTH
    )
    return covariance, {
        "covariance_cutoff": cutoff,
        "covariance_start": window.index.min(),
        "covariance_end": window.index.max(),
        "covariance_observations": int(len(window)),
        "ledoit_wolf_shrinkage": float(shrinkage),
    }


def current_variance_features(
    stress_signal: pd.Series,
    technical_signal: pd.Series,
    fundamental_signal: pd.Series,
    asset_vol_signal: pd.Series,
) -> dict[str, pd.Series]:
    credit_rank = float(fundamental_signal["credit_stress_rank"])
    credit_state = (
        float(np.clip(credit_rank, 0.0, 1.0) - 0.5)
        if np.isfinite(credit_rank)
        else 0.0
    )
    stress_state = float(
        np.clip(float(stress_signal["stress_score"]), 0.0, 1.0) - 0.5
    )
    output: dict[str, pd.Series] = {}
    for asset in ASSETS:
        atr_rank = float(technical_signal[f"atr_percentile_{asset}"])
        atr_state = (
            float(np.clip(atr_rank, 0.0, 1.0) - 0.5)
            if np.isfinite(atr_rank)
            else 0.0
        )
        iv_state = 0.0
        iv_available = 0.0
        if asset == "GLD" and bool(asset_vol_signal["gvz_active"]):
            iv_state = float(
                np.clip(asset_vol_signal["gvz_causal_rank"], 0.0, 1.0)
                - 0.5
            )
            iv_available = 1.0
        elif asset == "USO" and bool(asset_vol_signal["ovx_active"]):
            iv_state = float(
                np.clip(asset_vol_signal["ovx_causal_rank"], 0.0, 1.0)
                - 0.5
            )
            iv_available = 1.0
        output[asset] = pd.Series(
            {
                "stress_state": stress_state,
                "atr_state": atr_state,
                "credit_state": credit_state,
                "iv_state": iv_state,
                "iv_available": iv_available,
            },
            dtype=float,
        )
    return output


def _variance_feature_columns(asset: str) -> tuple[str, ...]:
    common = ("stress_state", "atr_state", "credit_state")
    return (*common, "iv_state", "iv_available") if asset in {"GLD", "USO"} else common


def variance_forecasts_for_month(
    records: list[dict[str, Any]],
    current_features: dict[str, pd.Series],
    base_covariance: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.DataFrame(records)
    multipliers = np.ones(len(ASSETS), dtype=float)
    details: dict[str, Any] = {}
    for asset_index, asset in enumerate(ASSETS):
        base_variance = float(base_covariance[asset_index, asset_index])
        columns = _variance_feature_columns(asset)
        if frame.empty:
            asset_records = pd.DataFrame()
            training = pd.DataFrame()
        else:
            asset_records = frame.loc[frame["asset"].eq(asset)].copy()
            training = asset_records.set_index("decision_month")
            training.index = pd.PeriodIndex(training.index, freq="M")
        fit = (
            _fit_bayesian_forecast(
                training,
                "target_log_ratio",
                columns,
                current_features[asset],
            )
            if len(training)
            else None
        )
        if len(asset_records):
            scored = asset_records.loc[
                asset_records["full_forecast_available"].eq(True)
            ]
        else:
            scored = pd.DataFrame()
        evidence = prequential_variance_stacking_weight(scored)
        if fit is None:
            predicted_log_ratio = 0.0
            log_mean_variance_ratio = 0.0
            full_variance = base_variance
            evidence_weight = 0.0
            observations = 0
            last_training_month = None
            prediction_std = math.nan
        else:
            predicted_log_ratio = fit.prediction
            # If log(V_realized / V_base) is Gaussian under the Bayesian
            # predictive distribution, the conditional mean variance ratio is
            # exp(mean + 0.5 variance), not exp(mean). This Jensen correction
            # prevents systematic risk understatement from log retransformation.
            log_mean_variance_ratio = (
                fit.prediction + 0.5 * fit.prediction_std**2
            )
            try:
                full_variance = base_variance * math.exp(
                    log_mean_variance_ratio
                )
            except OverflowError:
                full_variance = math.inf
            if not np.isfinite(full_variance) or full_variance <= 0.0:
                full_variance = base_variance
                predicted_log_ratio = 0.0
                log_mean_variance_ratio = 0.0
            evidence_weight = float(evidence["weight"])
            observations = fit.observations
            last_training_month = fit.last_training_month
            prediction_std = fit.prediction_std
        full_ratio = full_variance / base_variance
        multiplier = float(
            (1.0 - evidence_weight) + evidence_weight * full_ratio
        )
        multipliers[asset_index] = multiplier
        details[asset] = {
            "base_variance_forecast": base_variance,
            "full_variance_forecast": full_variance,
            "full_forecast_available": fit is not None,
            "predicted_log_ratio": predicted_log_ratio,
            "log_mean_variance_ratio": log_mean_variance_ratio,
            "prediction_std": prediction_std,
            "training_observations": observations,
            "last_training_month": last_training_month,
            "evidence": evidence,
            "final_multiplier": multiplier,
            "final_variance_forecast": base_variance * multiplier,
        }
    return multipliers, details


def solve_weights(
    history: pd.DataFrame,
    expected_return: np.ndarray,
    covariance: np.ndarray,
    pretrade: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize a log-utility certainty equivalent with actual trading costs.

    Stage 48's extra downside-semivariance term is removed: variance already
    prices second-moment risk, while the inherited CDaR constraint separately
    enforces the declared tail-risk mandate. This avoids an uncalibrated double
    penalty without changing the Stage 36 governance limits.
    """

    historical_returns = history[ASSETS].to_numpy(dtype=float)
    initial = (
        stage48.stage36.stage35.project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        transaction_cost = stage48.stage36.stage35.expected_transaction_cost(
            weights, pretrade
        )
        certainty_equivalent = (
            monthly_return - 0.5 * monthly_variance - transaction_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "estimated_transaction_cost": transaction_cost,
            "monthly_certainty_equivalent": certainty_equivalent,
        }

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage48.stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
                - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                stage48.stage36.stage35.CATASTROPHE_CDAR
                + stage48.stage36.stage35.cdar(
                    historical_returns @ weights,
                    stage48.stage36.stage35.CDAR_CONFIDENCE,
                )
            ),
        },
    ]
    result = minimize(
        lambda weights: -portfolio_values(weights)[
            "monthly_certainty_equivalent"
        ],
        initial,
        method="SLSQP",
        bounds=stage48.stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={
            "maxiter": stage48.stage36.stage35.SLSQP_MAX_ITERATIONS,
            "ftol": stage48.stage36.stage35.SLSQP_TOLERANCE,
        },
    )
    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = stage48.stage36.stage35.project_to_long_only_simplex(
            result.x
        )
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=stage48.stage36.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": stage48.stage36.stage35.SLSQP_MAX_ITERATIONS,
                "ftol": stage48.stage36.stage35.SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                f"Economic and fallback solves failed: {result.message}; "
                f"{fallback.message}"
            )
        result = fallback
        weights = stage48.stage36.stage35.project_to_long_only_simplex(
            fallback.x
        )
        used_fallback = True

    values = portfolio_values(weights)
    annual_volatility_value = annual_volatility(weights)
    historical_cdar = stage48.stage36.stage35.cdar(
        historical_returns @ weights,
        stage48.stage36.stage35.CDAR_CONFIDENCE,
    )
    return weights, {
        **values,
        "policy": policy,
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
            stage48.stage36.stage35.CATASTROPHE_ANNUAL_VOLATILITY
            - annual_volatility_value
        ),
        "cdar_slack": stage48.stage36.stage35.CATASTROPHE_CDAR
        + historical_cdar,
    }


def run_backtests(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    fundamental_signals: pd.DataFrame,
    asset_vol_signals: pd.DataFrame,
    daily_returns: pd.DataFrame,
    realized_variance: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    feature_frame = build_return_feature_frame(
        returns,
        probabilities,
        stress_signals,
        technical_signals,
        fundamental_signals,
    )
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(fundamental_signals.index)
    months = months.intersection(asset_vol_signals.index)
    months = months.intersection(feature_frame.index)
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]

    states: dict[str, dict[str, Any]] = {
        mode: {
            "pretrade": np.zeros(len(ASSETS), dtype=float),
            "first_trade": True,
            "nav": 1.0,
            "peak": 1.0,
            "rows": [],
        }
        for mode in MODES
    }
    return_records: list[dict[str, Any]] = []
    variance_records: list[dict[str, Any]] = []

    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < 12:
            continue
        try:
            base_covariance, covariance_detail = covariance_for_month(
                daily_returns, month
            )
        except ValueError:
            continue
        base_mu, evidence_mu, return_detail = return_forecasts_for_month(
            feature_frame, month, return_records
        )
        variance_features = current_variance_features(
            stress_signals.loc[month],
            technical_signals.loc[month],
            fundamental_signals.loc[month],
            asset_vol_signals.loc[month],
        )
        variance_multipliers, variance_detail = (
            variance_forecasts_for_month(
                variance_records, variance_features, base_covariance
            )
        )
        scaling = np.diag(np.sqrt(variance_multipliers))
        adjusted_covariance = stage48._nearest_psd(
            scaling @ base_covariance @ scaling
        )
        model_inputs = {
            MODE_MACRO: (base_mu, base_covariance),
            MODE_RETURN: (evidence_mu, base_covariance),
            MODE_FULL: (evidence_mu, adjusted_covariance),
        }
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)

        for mode, (expected_return, covariance) in model_inputs.items():
            state = states[mode]
            pretrade = np.asarray(state["pretrade"], dtype=float)
            weights, solve_detail = solve_weights(
                history, expected_return, covariance, pretrade, mode
            )
            change = weights - pretrade
            turnover = (
                float(np.abs(change).sum())
                if state["first_trade"]
                else 0.5 * float(np.abs(change).sum())
            )
            trade_cost = (
                float(np.abs(change).sum())
                * stage48.stage36.stage35.DOMESTIC_TRADE_COST
            )
            foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
            fx_cost = (
                abs(float(change[foreign_indices].sum()))
                * stage48.stage36.stage35.FOREIGN_WEIGHT_CHANGE_COST
            )
            gross_return = float(weights @ asset_return)
            net_return = gross_return - trade_cost - fx_cost
            state["nav"] *= 1.0 + net_return
            state["peak"] = max(state["peak"], state["nav"])
            state["pretrade"] = (
                weights * (1.0 + asset_return) / (1.0 + gross_return)
            )
            state["first_trade"] = False

            row: dict[str, Any] = {
                "month": month,
                "macro_signal_month": probabilities.loc[
                    month, "signal_month"
                ],
                "stress_signal_month": stress_signals.loc[
                    month, "stress_signal_month"
                ],
                "technical_signal_month": technical_signals.loc[
                    month, "technical_signal_month"
                ],
                "fundamental_signal_month": fundamental_signals.loc[
                    month, "fundamental_signal_month"
                ],
                "asset_vol_signal_month": asset_vol_signals.loc[
                    month, "asset_vol_signal_month"
                ],
                "covariance_cutoff": covariance_detail[
                    "covariance_cutoff"
                ],
                "covariance_start": covariance_detail[
                    "covariance_start"
                ],
                "covariance_end": covariance_detail["covariance_end"],
                "covariance_observations": covariance_detail[
                    "covariance_observations"
                ],
                "ledoit_wolf_shrinkage": covariance_detail[
                    "ledoit_wolf_shrinkage"
                ],
                "stress_score": float(
                    stress_signals.loc[month, "stress_score"]
                ),
                "recovery_score": float(
                    stress_signals.loc[month, "recovery_score"]
                ),
                "return": net_return,
                "gross_return": gross_return,
                "nav": state["nav"],
                "drawdown": state["nav"] / state["peak"] - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "base_covariance_min_eigenvalue": float(
                    np.linalg.eigvalsh(base_covariance).min()
                ),
                "adjusted_covariance_min_eigenvalue": float(
                    np.linalg.eigvalsh(adjusted_covariance).min()
                ),
                **{
                    column: float(probabilities.loc[month, column])
                    for column in REGIME_COLUMNS
                },
                **{
                    f"w_{asset}": float(weights[asset_index])
                    for asset_index, asset in enumerate(ASSETS)
                },
                **solve_detail,
            }
            for asset_index, asset in enumerate(ASSETS):
                return_asset_detail = return_detail[asset]
                variance_asset_detail = variance_detail[asset]
                row[f"macro_base_mu_{asset}"] = float(base_mu[asset_index])
                row[f"macro_prediction_std_{asset}"] = float(
                    return_asset_detail["base_prediction_std"]
                )
                row[f"macro_training_observations_{asset}"] = int(
                    return_asset_detail["base_observations"]
                )
                row[f"macro_last_training_month_{asset}"] = (
                    return_asset_detail["base_last_training_month"]
                )
                row[f"evidence_model_mu_{asset}"] = float(
                    return_asset_detail["final_prediction"]
                )
                for block in RETURN_BLOCKS:
                    block_detail = return_asset_detail["blocks"].get(block)
                    prefix = f"return_{block}_{asset}"
                    row[f"{prefix}_evidence_weight"] = float(
                        block_detail["evidence"]["weight"]
                        if block_detail is not None
                        else 0.0
                    )
                    row[f"{prefix}_evidence_observations"] = int(
                        block_detail["evidence"]["observations"]
                        if block_detail is not None
                        else 0
                    )
                    row[f"{prefix}_mean_loss_improvement"] = float(
                        block_detail["evidence"][
                            "mean_loss_improvement"
                        ]
                        if block_detail is not None
                        else 0.0
                    )
                    row[f"{prefix}_last_training_month"] = (
                        block_detail["last_training_month"]
                        if block_detail is not None
                        else None
                    )
                    row[f"{prefix}_training_observations"] = int(
                        block_detail["training_observations"]
                        if block_detail is not None
                        else 0
                    )
                    row[f"{prefix}_prediction_std"] = float(
                        block_detail["prediction_std"]
                        if block_detail is not None
                        else math.nan
                    )
                    row[f"{prefix}_raw_increment"] = float(
                        block_detail["raw_increment"]
                        if block_detail is not None
                        else 0.0
                    )
                    row[f"{prefix}_weighted_increment"] = float(
                        block_detail["weighted_increment"]
                        if block_detail is not None
                        else 0.0
                    )
                row[f"credit_direct_contribution_{asset}"] = float(
                    return_asset_detail["blocks"]["credit"][
                        "weighted_increment"
                    ]
                )
                row[f"technical_direct_contribution_{asset}"] = float(
                    return_asset_detail["blocks"]["technical"][
                        "weighted_increment"
                    ]
                )
                row[f"expected_mu_{asset}"] = float(
                    expected_return[asset_index]
                )
                row[f"sensor_variance_multiplier_{asset}"] = float(
                    variance_multipliers[asset_index]
                )
                row[f"applied_variance_multiplier_{asset}"] = float(
                    variance_multipliers[asset_index]
                    if mode == MODE_FULL
                    else 1.0
                )
                row[f"variance_evidence_weight_{asset}"] = float(
                    variance_asset_detail["evidence"]["weight"]
                )
                row[f"variance_evidence_observations_{asset}"] = int(
                    variance_asset_detail["evidence"]["observations"]
                )
                row[f"variance_mean_loss_improvement_{asset}"] = float(
                    variance_asset_detail["evidence"][
                        "mean_loss_improvement"
                    ]
                )
                row[f"variance_model_last_training_month_{asset}"] = (
                    variance_asset_detail["last_training_month"]
                )
                row[f"variance_training_observations_{asset}"] = int(
                    variance_asset_detail["training_observations"]
                )
                row[f"variance_prediction_std_{asset}"] = float(
                    variance_asset_detail["prediction_std"]
                )
                row[f"variance_log_mean_ratio_{asset}"] = float(
                    variance_asset_detail["log_mean_variance_ratio"]
                )
                for feature, value in variance_features[asset].items():
                    row[f"{feature}_{asset}"] = float(value)
            state["rows"].append(row)

        realized_asset_return = returns.loc[month, ASSETS]
        for asset in ASSETS:
            detail = return_detail[asset]
            actual = float(realized_asset_return[asset])
            for block, block_detail in detail["blocks"].items():
                return_records.append(
                    {
                        "decision_month": month,
                        "asset": asset,
                        "block": block,
                        "actual_return": actual,
                        "base_forecast": block_detail[
                            "base_prediction"
                        ],
                        "full_forecast": block_detail[
                            "model_prediction"
                        ],
                        "full_forecast_available": block_detail[
                            "full_forecast_available"
                        ],
                        "base_squared_error": (
                            actual - block_detail["base_prediction"]
                        )
                        ** 2,
                        "full_squared_error": (
                            actual - block_detail["model_prediction"]
                        )
                        ** 2,
                        "evidence_weight_used": block_detail[
                            "evidence"
                        ]["weight"],
                    }
                )

        if month in realized_variance.index:
            for asset_index, asset in enumerate(ASSETS):
                realized = float(realized_variance.loc[month, asset])
                detail = variance_detail[asset]
                base_forecast = float(detail["base_variance_forecast"])
                full_forecast = float(detail["full_variance_forecast"])
                if (
                    np.isfinite(realized)
                    and realized > NUMERICAL_EPSILON
                    and base_forecast > NUMERICAL_EPSILON
                ):
                    variance_records.append(
                        {
                            "decision_month": month,
                            "asset": asset,
                            "target_log_ratio": math.log(
                                realized / base_forecast
                            ),
                            "realized_variance": realized,
                            "base_variance_forecast": base_forecast,
                            "full_variance_forecast": full_forecast,
                            "full_forecast_available": detail[
                                "full_forecast_available"
                            ],
                            "base_qlike": qlike_loss(
                                realized, base_forecast
                            ),
                            "full_qlike": qlike_loss(
                                realized, full_forecast
                            ),
                            "evidence_weight_used": detail["evidence"][
                                "weight"
                            ],
                            **variance_features[asset].to_dict(),
                        }
                    )

    paths: dict[str, pd.DataFrame] = {}
    for mode, state in states.items():
        path = pd.DataFrame(state["rows"]).set_index("month")
        path.index = pd.PeriodIndex(path.index, freq="M")
        paths[mode] = path
    return paths, pd.DataFrame(return_records), pd.DataFrame(variance_records)


def _performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_start = max(path.index.min() for path in paths.values())
    common_end = min(path.index.max() for path in paths.values())
    periods = {
        "full_common": (common_start, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        for period, (start, end) in periods.items():
            rows.append(
                stage48.stage36.stage35.metric_row(
                    name, path, period, start, end
                )
            )
    return pd.DataFrame(rows)


def _bootstrap_table(
    references: dict[str, pd.DataFrame],
    candidates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for candidate_name, candidate in candidates.items():
        for reference_name, reference in references.items():
            common_start = max(reference.index.min(), candidate.index.min())
            common_end = min(reference.index.max(), candidate.index.max())
            for period, (start, end) in {
                "full_common": (common_start, common_end),
                "locked_2018_2026": (LOCKED_START, common_end),
            }.items():
                result = (
                    stage48.stage36.stage35.stage30.paired_block_bootstrap(
                        reference.loc[start:end, "return"],
                        candidate.loc[start:end, "return"],
                        replications=2000,
                        block_months=12,
                    )
                )
                result.insert(0, "Reference", reference_name)
                result.insert(1, "Candidate", candidate_name)
                result.insert(2, "Period", period)
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _metric(
    performance: pd.DataFrame,
    strategy: str,
    period: str,
    column: str,
) -> float:
    value = performance.loc[
        performance["Strategy"].eq(strategy)
        & performance["Period"].eq(period),
        column,
    ]
    return float(value.iloc[0])


def _return_training_lag_check(path: pd.DataFrame) -> bool:
    for asset in ASSETS:
        for block, _ in _return_model_specs(asset)[1:]:
            column = f"return_{block}_{asset}_last_training_month"
            available = path[column].notna()
            if not (
                pd.PeriodIndex(path.loc[available, column], freq="M")
                < path.index[available]
            ).all():
                return False
    return True


def _variance_training_lag_check(path: pd.DataFrame) -> bool:
    for asset in ASSETS:
        column = f"variance_model_last_training_month_{asset}"
        available = path[column].notna()
        if not (
            pd.PeriodIndex(path.loc[available, column], freq="M")
            < path.index[available]
        ).all():
            return False
    return True


def _return_evidence_count_check(
    path: pd.DataFrame,
    records: pd.DataFrame,
) -> bool:
    for month, row in path.iterrows():
        for asset in ASSETS:
            for block, _ in _return_model_specs(asset)[1:]:
                expected = int(
                    (
                        records["asset"].eq(asset)
                        & records["block"].eq(block)
                        & records["full_forecast_available"].eq(True)
                        & (
                            pd.PeriodIndex(
                                records["decision_month"], freq="M"
                            )
                            < month
                        )
                    ).sum()
                )
                column = (
                    f"return_{block}_{asset}_evidence_observations"
                )
                if int(row[column]) != expected:
                    return False
    return True


def _variance_evidence_count_check(
    path: pd.DataFrame,
    records: pd.DataFrame,
) -> bool:
    record_months = pd.PeriodIndex(records["decision_month"], freq="M")
    for month, row in path.iterrows():
        for asset in ASSETS:
            expected = int(
                (
                    records["asset"].eq(asset)
                    & records["full_forecast_available"].eq(True)
                    & (record_months < month)
                ).sum()
            )
            if int(row[f"variance_evidence_observations_{asset}"]) != expected:
                return False
    return True


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Period, pd.Timestamp, Path)):
        return str(value)
    if np.isscalar(value) and pd.isna(value):
        return None
    return value


def run_research(save: bool = True) -> dict[str, Any]:
    frozen_before = frozen_manifest()
    returns, return_audit = stage48.stage36.stage35.load_monthly_asset_returns(
        False
    )
    probabilities, macro_audit = (
        stage48.stage36.stage35.build_macro_probabilities(returns)
    )
    stress_signals = stage48.stage36.stage35.build_monthly_stress_signals(
        returns.index,
        stage48.stage36.stage35.build_daily_stress_features(),
    )
    market, market_audit = (
        stage48.stage36.stage35.stage20.load_daily_asset_ohlcv()
    )
    raw_fundamental, fundamental_audit = (
        stage48.stage36.stage35.load_fundamental_daily()
    )
    fundamental_signals = (
        stage48.stage36.stage35.build_monthly_fundamental_signals(
            raw_fundamental
        )
    )
    technical_signals = stage48.stage36.stage35.stage34._load_period_csv(
        stage48.stage36.stage35.stage20.OUTPUT_DIR
        / "monthly_technical_signals.csv",
        "target_month",
    )
    implied_daily, implied_audit = (
        stage48.stage36.load_asset_implied_volatility_daily()
    )
    asset_vol_signals = stage48.build_asset_volatility_signals(
        implied_daily, returns.index
    )
    daily_returns = stage48.build_daily_return_matrix(market)
    realized_variance = stage48.monthly_realized_variance(daily_returns)

    candidates, return_records, variance_records = run_backtests(
        returns,
        probabilities,
        stress_signals,
        technical_signals,
        fundamental_signals,
        asset_vol_signals,
        daily_returns,
        realized_variance,
    )
    stage36_path = stage48.stage36.stage35.stage34._load_period_csv(
        STAGE36_PATH, "month"
    )
    stage48_path = stage48.stage36.stage35.stage34._load_period_csv(
        STAGE48_PATH, "month"
    )
    all_paths = {
        "Stage36_GVZ_OVXAssetRisk": stage36_path,
        "Stage48_EvidenceCalibrated": stage48_path,
        **candidates,
    }
    performance = _performance_table(all_paths)
    bootstrap = _bootstrap_table(
        {
            "Stage36_GVZ_OVXAssetRisk": stage36_path,
            "Stage48_EvidenceCalibrated": stage48_path,
        },
        candidates,
    )

    full = candidates[MODE_FULL]
    signal_columns = [
        "macro_signal_month",
        "stress_signal_month",
        "technical_signal_month",
        "fundamental_signal_month",
        "asset_vol_signal_month",
    ]
    signal_lags = {
        column: bool(
            (pd.PeriodIndex(full[column], freq="M") < full.index).all()
        )
        for column in signal_columns
    }
    weight_checks: dict[str, bool] = {}
    for mode, path in candidates.items():
        weights = path[WEIGHT_COLUMNS]
        weight_checks[f"{mode}_finite"] = bool(
            np.isfinite(weights.to_numpy()).all()
        )
        weight_checks[f"{mode}_long_only"] = bool(
            weights.min().min() >= -1e-8
        )
        weight_checks[f"{mode}_fully_invested"] = bool(
            np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        )
        weight_checks[f"{mode}_volatility_guard"] = bool(
            path["volatility_slack"].min() >= -1e-7
        )
        weight_checks[f"{mode}_cdar_guard"] = bool(
            path["cdar_slack"].min() >= -1e-7
        )

    return_weight_columns = [
        f"return_{block}_{asset}_evidence_weight"
        for asset in ASSETS
        for block, _ in _return_model_specs(asset)[1:]
    ]
    variance_weight_columns = [
        f"variance_evidence_weight_{asset}" for asset in ASSETS
    ]
    multiplier_columns = [
        f"sensor_variance_multiplier_{asset}" for asset in ASSETS
    ]
    checks = {
        "stage36_and_stage48_frozen_files_unchanged": (
            frozen_before == frozen_manifest()
        ),
        **weight_checks,
        "signals_strictly_lagged": bool(all(signal_lags.values())),
        "return_training_strictly_precedes_decision": (
            _return_training_lag_check(full)
        ),
        "variance_training_strictly_precedes_decision": (
            _variance_training_lag_check(full)
        ),
        "return_evidence_uses_only_prior_forecasts": (
            _return_evidence_count_check(full, return_records)
        ),
        "variance_evidence_uses_only_prior_forecasts": (
            _variance_evidence_count_check(full, variance_records)
        ),
        "covariance_uses_only_prior_daily_data": bool(
            (
                pd.to_datetime(full["covariance_cutoff"])
                <= pd.DatetimeIndex(
                    [(month - 1).to_timestamp("M") for month in full.index]
                )
            ).all()
        ),
        "base_covariance_psd": bool(
            full["base_covariance_min_eigenvalue"].min() > 0.0
        ),
        "adjusted_covariance_psd": bool(
            full["adjusted_covariance_min_eigenvalue"].min() > 0.0
        ),
        "return_evidence_scores_bounded_zero_one": bool(
            full[return_weight_columns].min().min() >= 0.0
            and full[return_weight_columns].max().max() <= 1.0
        ),
        "variance_evidence_scores_bounded_zero_one": bool(
            full[variance_weight_columns].min().min() >= 0.0
            and full[variance_weight_columns].max().max() <= 1.0
        ),
        "variance_multipliers_positive_finite": bool(
            np.isfinite(full[multiplier_columns].to_numpy()).all()
            and full[multiplier_columns].min().min() > 0.0
        ),
        "returns_finite": bool(
            all(np.isfinite(path["return"]).all() for path in candidates.values())
        ),
    }

    gate = {
        "causal_feasible_and_numerically_valid": bool(all(checks.values())),
        "full_sharpe_not_below_stage48": bool(
            _metric(performance, MODE_FULL, "full_common", "Sharpe")
            >= _metric(
                performance,
                "Stage48_EvidenceCalibrated",
                "full_common",
                "Sharpe",
            )
        ),
        "locked_mdd_not_worse_than_stage48": bool(
            _metric(performance, MODE_FULL, "locked_2018_2026", "MDD")
            >= _metric(
                performance,
                "Stage48_EvidenceCalibrated",
                "locked_2018_2026",
                "MDD",
            )
        ),
        "full_turnover_not_above_stage48": bool(
            _metric(performance, MODE_FULL, "full_common", "AvgTurnover")
            <= _metric(
                performance,
                "Stage48_EvidenceCalibrated",
                "full_common",
                "AvgTurnover",
            )
        ),
    }
    gate["promotion_pass"] = bool(all(gate.values()))
    macro_gate = {
        "full_sharpe_not_below_stage48": bool(
            _metric(performance, MODE_MACRO, "full_common", "Sharpe")
            >= _metric(
                performance,
                "Stage48_EvidenceCalibrated",
                "full_common",
                "Sharpe",
            )
        ),
        "locked_mdd_not_worse_than_stage48": bool(
            _metric(performance, MODE_MACRO, "locked_2018_2026", "MDD")
            >= _metric(
                performance,
                "Stage48_EvidenceCalibrated",
                "locked_2018_2026",
                "MDD",
            )
        ),
        "full_turnover_not_above_stage48": bool(
            _metric(performance, MODE_MACRO, "full_common", "AvgTurnover")
            <= _metric(
                performance,
                "Stage48_EvidenceCalibrated",
                "full_common",
                "AvgTurnover",
            )
        ),
    }
    macro_gate["comparison_pass"] = bool(all(macro_gate.values()))

    diagnostics = {
        "months": int(len(full)),
        "start": str(full.index.min()),
        "end": str(full.index.max()),
        "return_forecast_records": int(len(return_records)),
        "variance_forecast_records": int(len(variance_records)),
        "return_evidence": {
            asset: {
                block: {
                    "mean_weight": float(
                        full[
                            f"return_{block}_{asset}_evidence_weight"
                        ].mean()
                    ),
                    "nonzero_months": int(
                        full[
                            f"return_{block}_{asset}_evidence_weight"
                        ].gt(0.0).sum()
                    ),
                    "last_weight": float(
                        full[
                            f"return_{block}_{asset}_evidence_weight"
                        ].iloc[-1]
                    ),
                    "last_mean_loss_improvement": float(
                        full[
                            f"return_{block}_{asset}_mean_loss_improvement"
                        ].iloc[-1]
                    ),
                }
                for block, _ in _return_model_specs(asset)[1:]
            }
            for asset in ASSETS
        },
        "variance_evidence": {
            asset: {
                "mean_weight": float(
                    full[f"variance_evidence_weight_{asset}"].mean()
                ),
                "nonzero_months": int(
                    full[f"variance_evidence_weight_{asset}"].gt(0.0).sum()
                ),
                "last_weight": float(
                    full[f"variance_evidence_weight_{asset}"].iloc[-1]
                ),
                "last_mean_loss_improvement": float(
                    full[f"variance_mean_loss_improvement_{asset}"].iloc[-1]
                ),
                "multiplier_mean": float(
                    full[f"sensor_variance_multiplier_{asset}"].mean()
                ),
                "multiplier_min": float(
                    full[f"sensor_variance_multiplier_{asset}"].min()
                ),
                "multiplier_max": float(
                    full[f"sensor_variance_multiplier_{asset}"].max()
                ),
            }
            for asset in ASSETS
        },
        "average_weights": {
            asset: float(full[f"w_{asset}"].mean()) for asset in ASSETS
        },
        "fallback_months": {
            mode: int(path["used_fallback"].sum())
            for mode, path in candidates.items()
        },
        "signal_lag_checks": signal_lags,
    }
    report = {
        "strategy": MODE_FULL,
        "design": {
            "return_model": "hierarchical Bayesian ridge macro -> stress/recovery -> credit -> technical -> KODEX fundamentals; each increment gets the convex weight minimizing prior one-step OOS MSE",
            "credit_scope": "signed causal credit widening z-score enters every asset after macro and stress/recovery; only credit-specific incremental OOS evidence can activate it",
            "technical_scope": "direct incremental return predictor with its own OOS evidence; no neutral target and no cross-sectional shrink rule",
            "covariance": "prior 252 complete trading days, Ledoit-Wolf constant-correlation shrinkage",
            "variance_model": "Bayesian log realized/base variance model with lognormal Jensen retransformation; convex prior-OOS QLIKE stacking; one D Sigma D map",
            "objective": "monthly log-utility certainty equivalent mu - 0.5 variance - actual expected transaction cost",
            "retained_policy_not_statistical_inference": "Stage36 13% annual-volatility and 16% 90%-CDaR governance guards",
            "removed_from_stage48": [
                "same-sample beta/SE positive-part reliability",
                "four separate soft-regime WLS stress/recovery coefficient systems",
                "cross-sectional-mean technical veto",
                "one-sided non-HAC EPS and valuation slope inheritance",
                "fixed 60-month threshold independent of model dimension",
                "historical min/max clipping of log variance forecasts",
                "monthly covariance plus stress-weighted covariance interpolation",
                "variance plus arbitrary unit-weight downside-semivariance double penalty",
            ],
        },
        "checks": checks,
        "gate": gate,
        "macro_ablation_gate": macro_gate,
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
        filenames = {
            MODE_MACRO: "stage49_macro_bayes_monthly.csv",
            MODE_RETURN: "stage49_return_evidence_monthly.csv",
            MODE_FULL: "stage49_full_prequential_monthly.csv",
        }
        for mode, path in candidates.items():
            path.to_csv(OUTPUT_DIR / filenames[mode])
        return_records.to_csv(
            OUTPUT_DIR / "return_prequential_history.csv", index=False
        )
        variance_records.to_csv(
            OUTPUT_DIR / "variance_prequential_history.csv", index=False
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap.csv", index=False
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
        "paths": candidates,
        "return_records": return_records,
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
