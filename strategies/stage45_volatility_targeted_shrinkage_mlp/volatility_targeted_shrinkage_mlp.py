from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from strategies.stage36_asset_implied_volatility_risk import (
    asset_implied_volatility_risk_slsqp as stage36,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ORIGINAL_SOURCE = Path(
    r"C:\Users\PC_1M\Downloads\volatility-targeted-shrinkage-mlp.py"
)

ASSETS = stage36.ASSETS
WEIGHT_COLUMNS = stage36.WEIGHT_COLUMNS
FULL_START = stage36.FULL_START
LOCKED_START = stage36.stage35.LOCKED_START
RESEARCH_END = stage36.RESEARCH_END

# The economic design is fixed before the backtest. These are not searched.
TARGET_ANNUAL_VOLATILITY = 0.10
COVARIANCE_LOOKBACK_DAYS = 252
TRADING_DAYS_PER_MONTH = 21.0
MIN_TRAIN_MONTHS = 36
RETRAIN_EVERY_YEARS = 1
HIDDEN_UNITS = 4
MLP_ALPHA = 0.01
MLP_MAX_ITERATIONS = 1_000
ENSEMBLE_SEEDS = (20260730, 20260731, 20260732)
NUMERICAL_EPSILON = 1e-12

REGIME_FEATURES = tuple(stage36.stage35.REGIME_COLUMNS)
GLOBAL_FEATURES = (
    *REGIME_FEATURES,
    "stress_score",
    "recovery_score",
    "ktb_3y_pct",
    "yield_curve_10y_minus_3y_pctpt",
    "credit_easing_z",
)
ASSET_FEATURES = (
    "trailing_return_1m",
    "trailing_return_3m",
    "trailing_return_6m",
    "trailing_return_12m",
    "trailing_volatility_12m",
    "technical_direction",
    "atr_percentile",
    "k_score",
    "implied_volatility_rank",
)
MODEL_FEATURES = (
    *GLOBAL_FEATURES,
    *ASSET_FEATURES,
    *(f"asset_{asset}" for asset in ASSETS),
)

STAGE36_PATH = (
    stage36.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv"
)
FROZEN_FILES = (
    Path(stage36.__file__),
    STAGE36_PATH,
    stage36.OUTPUT_DIR / "validation_report.json",
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
        }
        for path in FROZEN_FILES
    }


def ledoit_wolf_constant_correlation(
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the source strategy's constant-correlation shrinkage estimate.

    ``values`` is T by N. The estimator retains each sample variance and
    shrinks every off-diagonal correlation toward their common average.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("values must be a T by N matrix with T >= 2")
    if not np.isfinite(x).all():
        raise ValueError("values must be finite")

    observations, assets = x.shape
    centered = x - x.mean(axis=0, keepdims=True)
    sample = (centered.T @ centered) / observations

    variances = np.maximum(np.diag(sample).copy(), 1e-16)
    standard_deviations = np.sqrt(variances)
    outer_standard_deviation = np.outer(
        standard_deviations, standard_deviations
    )
    off_diagonal = ~np.eye(assets, dtype=bool)
    average_correlation = (
        float((sample / outer_standard_deviation)[off_diagonal].mean())
        if assets > 1
        else 0.0
    )

    target = average_correlation * outer_standard_deviation
    np.fill_diagonal(target, variances)

    squared = centered * centered
    pi_matrix = (squared.T @ squared) / observations - sample * sample
    pi_hat = float(pi_matrix.sum())

    cross = ((squared * centered).T @ centered) / observations
    theta_ii = cross - variances[:, None] * sample
    theta_jj = cross.T - variances[None, :] * sample
    ratio = np.outer(1.0 / standard_deviations, standard_deviations)
    rho_off = (average_correlation / 2.0) * (
        ratio * theta_ii + ratio.T * theta_jj
    )
    rho_hat = float(
        np.diag(pi_matrix).sum() + rho_off[off_diagonal].sum()
    )

    distance = target - sample
    gamma_hat = float((distance * distance).sum())
    shrinkage = (
        0.0
        if gamma_hat <= 1e-30
        else float(
            np.clip(
                (pi_hat - rho_hat) / gamma_hat / observations,
                0.0,
                1.0,
            )
        )
    )
    covariance = shrinkage * target + (1.0 - shrinkage) * sample
    covariance.flat[:: assets + 1] += NUMERICAL_EPSILON
    return covariance, shrinkage


def build_daily_return_matrix(
    market: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align Stage 36's KRW close series without using future observations."""

    close = pd.concat(
        {
            asset: market[asset]["close"].dropna().astype(float)
            for asset in ASSETS
        },
        axis=1,
    ).sort_index()
    calendar = pd.date_range(close.index.min(), close.index.max(), freq="B")
    levels = close.reindex(calendar).ffill(limit=5)
    daily = levels.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    return daily.dropna(how="any")[ASSETS]


def covariance_for_month(
    daily_returns: pd.DataFrame,
    target_month: pd.Period,
    asset_volatility_signal: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate covariance using only observations through the prior month."""

    cutoff = (target_month - 1).to_timestamp("M")
    window = daily_returns.loc[:cutoff, ASSETS].tail(
        COVARIANCE_LOOKBACK_DAYS
    )
    if len(window) < COVARIANCE_LOOKBACK_DAYS:
        raise ValueError(
            f"{target_month} has only {len(window)} prior complete daily rows"
        )
    daily_covariance, shrinkage = ledoit_wolf_constant_correlation(
        window.to_numpy(dtype=float)
    )
    covariance = daily_covariance * TRADING_DAYS_PER_MONTH

    # Preserve Stage 36's promoted variance-only use of GVZ and OVX.
    gvz_multiplier = float(
        asset_volatility_signal["gvz_gld_variance_multiplier"]
    )
    ovx_multiplier = float(
        asset_volatility_signal["ovx_uso_variance_multiplier"]
    )
    scaling = np.eye(len(ASSETS), dtype=float)
    scaling[ASSETS.index("GLD"), ASSETS.index("GLD")] = math.sqrt(
        gvz_multiplier
    )
    scaling[ASSETS.index("USO"), ASSETS.index("USO")] = math.sqrt(
        ovx_multiplier
    )
    covariance = scaling @ covariance @ scaling
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "covariance_cutoff": cutoff,
        "covariance_start": window.index.min(),
        "covariance_end": window.index.max(),
        "covariance_observations": int(len(window)),
        "lw_shrinkage": shrinkage,
        "gvz_gold_variance_multiplier": gvz_multiplier,
        "ovx_oil_variance_multiplier": ovx_multiplier,
    }


def _compound_lagged_returns(
    returns: pd.DataFrame, months: int
) -> pd.DataFrame:
    return (
        (1.0 + returns.shift(1))
        .rolling(months, min_periods=months)
        .apply(np.prod, raw=True)
        .sub(1.0)
    )


def build_asset_month_panel(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    technical: pd.DataFrame,
    fundamental: pd.DataFrame,
    asset_volatility: pd.DataFrame,
) -> pd.DataFrame:
    """Build a causal four-asset panel with an actual cash-excess target.

    There is no cross-sectional ranking, demeaning, pooled monthly context, or
    minimum-universe filter. Every time-varying input for target month t was
    observable by the end of t-1. The KTB 3-year yield is used only as a
    transparent investable-cash return proxy, divided by twelve.
    """

    lagged = {
        horizon: _compound_lagged_returns(returns, horizon)
        for horizon in (1, 3, 6, 12)
    }
    trailing_volatility = (
        returns.shift(1).rolling(12, min_periods=12).std(ddof=1)
        * math.sqrt(12.0)
    )
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress.index)
    months = months.intersection(technical.index)
    months = months.intersection(fundamental.index)
    months = months.intersection(asset_volatility.index)
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]

    rows: list[dict[str, Any]] = []
    for month in months:
        global_values = {
            **{
                feature: float(probabilities.loc[month, feature])
                for feature in REGIME_FEATURES
            },
            "stress_score": float(stress.loc[month, "stress_score"]),
            "recovery_score": float(stress.loc[month, "recovery_score"]),
            "ktb_3y_pct": float(fundamental.loc[month, "ktb_3y_pct"]),
            "yield_curve_10y_minus_3y_pctpt": float(
                fundamental.loc[month, "yield_curve_10y_minus_3y_pctpt"]
            ),
            "credit_easing_z": float(
                fundamental.loc[month, "credit_easing_z"]
            ),
        }
        cash_return = global_values["ktb_3y_pct"] / 100.0 / 12.0
        for asset in ASSETS:
            implied_rank = 0.5
            if asset == "GLD" and bool(
                asset_volatility.loc[month, "gvz_active"]
            ):
                implied_rank = float(
                    asset_volatility.loc[month, "gvz_causal_rank"]
                )
            elif asset == "USO" and bool(
                asset_volatility.loc[month, "ovx_active"]
            ):
                implied_rank = float(
                    asset_volatility.loc[month, "ovx_causal_rank"]
                )
            row: dict[str, Any] = {
                "month": month,
                "asset": asset,
                **global_values,
                "trailing_return_1m": float(lagged[1].loc[month, asset]),
                "trailing_return_3m": float(lagged[3].loc[month, asset]),
                "trailing_return_6m": float(lagged[6].loc[month, asset]),
                "trailing_return_12m": float(
                    lagged[12].loc[month, asset]
                ),
                "trailing_volatility_12m": float(
                    trailing_volatility.loc[month, asset]
                ),
                "technical_direction": float(
                    technical.loc[month, f"technical_direction_{asset}"]
                ),
                "atr_percentile": float(
                    technical.loc[month, f"atr_percentile_{asset}"]
                ),
                "k_score": float(
                    technical.loc[month, f"k_score_{asset}"]
                ),
                "implied_volatility_rank": implied_rank,
                "cash_return": cash_return,
                "target_excess_return": float(
                    returns.loc[month, asset] - cash_return
                ),
                "feature_signal_month": month - 1,
            }
            for identifier in ASSETS:
                row[f"asset_{identifier}"] = float(asset == identifier)
            rows.append(row)
    panel = pd.DataFrame(rows).set_index(["month", "asset"]).sort_index()
    return panel.replace([np.inf, -np.inf], np.nan)


class TinyMLPEnsemble:
    """A deterministic, yearly refit, tiny pooled asset-return MLP."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.models: list[MLPRegressor] = []
        self.asset_target_mean: dict[str, float] = {}
        self.asset_target_scale: dict[str, float] = {}
        self.training_months = 0
        self.training_rows = 0
        self.cutoff: pd.Period | None = None

    def fit(self, training: pd.DataFrame, cutoff: pd.Period) -> "TinyMLPEnsemble":
        usable = training.dropna(subset=["target_excess_return"])
        month_values = usable.index.get_level_values("month")
        if int(month_values.nunique()) < MIN_TRAIN_MONTHS:
            raise ValueError("not enough completed training months")

        x = usable.loc[:, MODEL_FEATURES].to_numpy(dtype=float)
        x = self.imputer.fit_transform(x)
        x = self.scaler.fit_transform(x)
        y = usable["target_excess_return"].to_numpy(dtype=float).copy()
        assets = usable.index.get_level_values("asset")
        for asset in ASSETS:
            mask = np.asarray(assets == asset)
            asset_y = y[mask]
            center = float(asset_y.mean())
            scale = float(asset_y.std(ddof=0))
            if not np.isfinite(scale) or scale <= NUMERICAL_EPSILON:
                scale = 1.0
            self.asset_target_mean[asset] = center
            self.asset_target_scale[asset] = scale
            y[mask] = (asset_y - center) / scale

        self.models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for seed in ENSEMBLE_SEEDS:
                model = MLPRegressor(
                    hidden_layer_sizes=(HIDDEN_UNITS,),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=MLP_ALPHA,
                    max_iter=MLP_MAX_ITERATIONS,
                    tol=1e-7,
                    random_state=seed,
                )
                model.fit(x, y)
                self.models.append(model)
        self.training_months = int(month_values.nunique())
        self.training_rows = int(len(usable))
        self.cutoff = cutoff
        return self

    def predict(self, current: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("model has not been fitted")
        x = current.loc[:, MODEL_FEATURES].to_numpy(dtype=float)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        normalized = np.mean(
            [model.predict(x) for model in self.models], axis=0
        )
        assets = current.index.get_level_values("asset")
        output = np.empty(len(current), dtype=float)
        for position, asset in enumerate(assets):
            output[position] = (
                self.asset_target_mean[str(asset)]
                + normalized[position] * self.asset_target_scale[str(asset)]
            )
        return output


def fit_model_for_month(
    panel: pd.DataFrame, target_month: pd.Period
) -> TinyMLPEnsemble | None:
    completed = panel.loc[
        panel.index.get_level_values("month") < target_month
    ]
    training_months = completed.index.get_level_values("month").nunique()
    if int(training_months) < MIN_TRAIN_MONTHS:
        return None
    return TinyMLPEnsemble().fit(completed, target_month - 1)


def _portfolio_cdar(
    weights: np.ndarray,
    historical_assets: np.ndarray,
    historical_cash: np.ndarray,
) -> float:
    cash_weight = 1.0 - float(weights.sum())
    portfolio_returns = (
        historical_assets @ weights + historical_cash * cash_weight
    )
    return stage36.stage35.cdar(
        portfolio_returns, stage36.stage35.CDAR_CONFIDENCE
    )


def solve_volatility_targeted_weights(
    expected_excess_return: np.ndarray,
    covariance: np.ndarray,
    historical_assets: np.ndarray,
    historical_cash: np.ndarray,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize forecast excess return under Stage 36-compatible guards."""

    mu = np.asarray(expected_excess_return, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    initial = np.clip(np.asarray(pretrade, dtype=float), 0.0, 1.0)
    if initial.sum() > 1.0:
        initial /= initial.sum()

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ sigma @ weights), 0.0) * 12.0)

    def utility(weights: np.ndarray) -> float:
        return float(weights @ mu) - stage36.stage35.expected_transaction_cost(
            weights, pretrade
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda weights: float(1.0 - weights.sum()),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                TARGET_ANNUAL_VOLATILITY - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                stage36.stage35.CATASTROPHE_CDAR
                + _portfolio_cdar(
                    weights, historical_assets, historical_cash
                )
            ),
        },
    ]
    result = minimize(
        lambda weights: -utility(weights),
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
        weights = np.clip(result.x, 0.0, 1.0)
    else:
        fallback = minimize(
            lambda weights: float(weights @ sigma @ weights),
            np.zeros(len(ASSETS), dtype=float),
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
                "MLP allocation and minimum-variance fallback failed: "
                f"{result.message}; {fallback.message}"
            )
        result = fallback
        weights = np.clip(fallback.x, 0.0, 1.0)
        used_fallback = True

    annual_vol = annual_volatility(weights)
    historical_cdar = _portfolio_cdar(
        weights, historical_assets, historical_cash
    )
    cash_weight = max(0.0, 1.0 - float(weights.sum()))
    detail = {
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_monthly_excess_return": float(weights @ mu),
        "estimated_transaction_cost": float(
            stage36.stage35.expected_transaction_cost(weights, pretrade)
        ),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "risky_weight_sum": float(weights.sum()),
        "cash_weight": cash_weight,
        "budget_slack": cash_weight,
        "volatility_slack": TARGET_ANNUAL_VOLATILITY - annual_vol,
        "cdar_slack": stage36.stage35.CATASTROPHE_CDAR + historical_cdar,
    }
    return weights, detail


def _load_stage36_path() -> pd.DataFrame:
    path = pd.read_csv(STAGE36_PATH, index_col="month")
    path.index = pd.PeriodIndex(path.index, freq="M")
    return path


def run_backtest(
    returns: pd.DataFrame,
    panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    asset_volatility: pd.DataFrame,
    stage36_path: pd.DataFrame,
) -> pd.DataFrame:
    months = returns.index.intersection(asset_volatility.index)
    months = months.intersection(
        panel.index.get_level_values("month").unique()
    )
    months = months.intersection(stage36_path.index)
    months = months[(months >= FULL_START) & (months <= RESEARCH_END)]

    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    fitted_year: int | None = None
    model: TinyMLPEnsemble | None = None

    cash_by_month = (
        panel.reset_index()
        .groupby("month")["cash_return"]
        .first()
        .sort_index()
    )
    for month in months:
        if fitted_year != month.year:
            model = fit_model_for_month(panel, month)
            fitted_year = month.year

        covariance, covariance_detail = covariance_for_month(
            daily_returns, month, asset_volatility.loc[month]
        )
        current_panel = panel.loc[[month]].reindex(ASSETS, level="asset")
        model_active = model is not None
        if model_active:
            expected_excess = model.predict(current_panel)
            historical_months = returns.index[
                (returns.index < month) & (returns.index.isin(cash_by_month.index))
            ]
            historical_assets = returns.loc[
                historical_months, ASSETS
            ].to_numpy(dtype=float)
            historical_cash = cash_by_month.loc[
                historical_months
            ].to_numpy(dtype=float)
            weights, solve_detail = solve_volatility_targeted_weights(
                expected_excess,
                covariance,
                historical_assets,
                historical_cash,
                pretrade,
            )
        else:
            expected_excess = np.full(len(ASSETS), np.nan, dtype=float)
            weights = stage36_path.loc[month, WEIGHT_COLUMNS].to_numpy(
                dtype=float
            )
            solve_detail = {
                "solver_success": True,
                "used_fallback": False,
                "solver_status": 0,
                "solver_message": "Stage36 frozen warm-up allocation",
                "solver_iterations": 0,
                "objective_value": np.nan,
                "expected_monthly_excess_return": np.nan,
                "estimated_transaction_cost": float(
                    stage36.stage35.expected_transaction_cost(
                        weights, pretrade
                    )
                ),
                "expected_annual_volatility": math.sqrt(
                    max(float(weights @ covariance @ weights), 0.0) * 12.0
                ),
                "historical_cdar": np.nan,
                "risky_weight_sum": float(weights.sum()),
                "cash_weight": max(0.0, 1.0 - float(weights.sum())),
                "budget_slack": max(0.0, 1.0 - float(weights.sum())),
                "volatility_slack": np.nan,
                "cdar_slack": np.nan,
            }

        cash_weight = float(solve_detail["cash_weight"])
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
        foreign = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign].sum()))
            * stage36.stage35.FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        cash_return = float(cash_by_month.loc[month])
        gross_return = float(
            weights @ asset_return + cash_weight * cash_return
        )
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

        row: dict[str, Any] = {
            "month": month,
            "policy": (
                "Stage45_TinyMLP_LW_VolTarget"
                if model_active
                else "Stage36_Frozen_Warmup"
            ),
            "model_active": model_active,
            "model_fit_cutoff": (
                model.cutoff if model_active else pd.NaT
            ),
            "model_training_months": (
                model.training_months if model_active else 0
            ),
            "model_training_rows": (
                model.training_rows if model_active else 0
            ),
            "cash_return": cash_return,
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
            **{
                f"predicted_excess_return_{asset}": float(
                    expected_excess[index]
                )
                for index, asset in enumerate(ASSETS)
            },
            **{
                f"realized_excess_return_{asset}": float(
                    asset_return[index] - cash_return
                )
                for index, asset in enumerate(ASSETS)
            },
            **covariance_detail,
            **solve_detail,
        }
        rows.append(row)
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _performance_table(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    active_start = candidate.index[candidate["model_active"]][0]
    common_end = min(baseline.index.max(), candidate.index.max())
    periods = {
        "full_with_stage36_warmup": (FULL_START, common_end),
        "mlp_active": (active_start, common_end),
        "locked_2018_2026": (LOCKED_START, common_end),
    }
    rows: list[dict[str, Any]] = []
    for name, path in {
        "Stage36_GVZ_OVXAssetRisk": baseline,
        "Stage45_TinyMLP_LW_VolTarget": candidate,
    }.items():
        for period_name, (start, end) in periods.items():
            rows.append(
                stage36.stage35.metric_row(
                    name, path, period_name, start, end
                )
            )
    return pd.DataFrame(rows)


def _bootstrap_table(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> pd.DataFrame:
    active_start = candidate.index[candidate["model_active"]][0]
    rows: list[pd.DataFrame] = []
    for period_name, start in (
        ("mlp_active", active_start),
        ("locked_2018_2026", LOCKED_START),
    ):
        common = baseline.loc[start:RESEARCH_END].index.intersection(
            candidate.loc[start:RESEARCH_END].index
        )
        summary = stage36.stage35.stage30.paired_block_bootstrap(
            baseline.loc[common, "return"],
            candidate.loc[common, "return"],
        )
        summary.insert(0, "Period", period_name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _forecast_diagnostics(candidate: pd.DataFrame) -> dict[str, Any]:
    active = candidate.loc[candidate["model_active"]]
    predicted = active[
        [f"predicted_excess_return_{asset}" for asset in ASSETS]
    ].to_numpy(dtype=float)
    realized = active[
        [f"realized_excess_return_{asset}" for asset in ASSETS]
    ].to_numpy(dtype=float)
    rows: dict[str, Any] = {
        "active_months": int(len(active)),
        "asset_month_predictions": int(predicted.size),
        "pooled_rmse": float(np.sqrt(np.mean((predicted - realized) ** 2))),
        "pooled_mae": float(np.mean(np.abs(predicted - realized))),
        "pooled_directional_accuracy": float(
            np.mean(np.sign(predicted) == np.sign(realized))
        ),
        "by_asset": {},
    }
    for index, asset in enumerate(ASSETS):
        p = predicted[:, index]
        r = realized[:, index]
        rows["by_asset"][asset] = {
            "rmse": float(np.sqrt(np.mean((p - r) ** 2))),
            "mae": float(np.mean(np.abs(p - r))),
            "directional_accuracy": float(
                np.mean(np.sign(p) == np.sign(r))
            ),
            "correlation": float(np.corrcoef(p, r)[0, 1]),
        }
    return rows


def run_research(save: bool = True) -> dict[str, Any]:
    frozen_before = frozen_manifest()
    returns, return_levels = stage36.stage35.load_monthly_asset_returns(False)
    probabilities, macro_ranks = stage36.stage35.build_macro_probabilities(
        returns
    )
    stress = stage36.stage35.build_monthly_stress_signals(
        returns.index, stage36.stage35.build_daily_stress_features()
    )
    technical = stage36.stage35.stage34._load_period_csv(
        stage36.stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    raw_fundamental, fundamental_audit = (
        stage36.stage35.load_fundamental_daily()
    )
    fundamental = stage36.stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
    asset_volatility_daily, asset_volatility_audit = (
        stage36.load_asset_implied_volatility_daily()
    )
    asset_volatility = stage36.build_monthly_asset_volatility_signals(
        asset_volatility_daily, returns.index
    )
    market, market_audit = stage36.stage35.stage20.load_daily_asset_ohlcv()
    daily_returns = build_daily_return_matrix(market)
    panel = build_asset_month_panel(
        returns,
        probabilities,
        stress,
        technical,
        fundamental,
        asset_volatility,
    )
    baseline = _load_stage36_path()
    candidate = run_backtest(
        returns, panel, daily_returns, asset_volatility, baseline
    )
    performance = _performance_table(baseline, candidate)
    bootstrap = _bootstrap_table(baseline, candidate)
    forecast = _forecast_diagnostics(candidate)

    active = candidate.loc[candidate["model_active"]]
    warmup = candidate.loc[~candidate["model_active"]]
    warmup_common = warmup.index.intersection(baseline.index)
    warmup_return_error = float(
        (
            warmup.loc[warmup_common, "return"]
            - baseline.loc[warmup_common, "return"]
        )
        .abs()
        .max()
    )
    warmup_weight_error = float(
        (
            warmup.loc[warmup_common, WEIGHT_COLUMNS]
            - baseline.loc[warmup_common, WEIGHT_COLUMNS]
        )
        .abs()
        .to_numpy()
        .max()
    )
    frozen_after = frozen_manifest()

    indexed = performance.set_index(["Strategy", "Period"])
    baseline_active = indexed.loc[
        ("Stage36_GVZ_OVXAssetRisk", "mlp_active")
    ]
    candidate_active = indexed.loc[
        ("Stage45_TinyMLP_LW_VolTarget", "mlp_active")
    ]
    baseline_locked = indexed.loc[
        ("Stage36_GVZ_OVXAssetRisk", "locked_2018_2026")
    ]
    candidate_locked = indexed.loc[
        ("Stage45_TinyMLP_LW_VolTarget", "locked_2018_2026")
    ]
    gates = {
        "active_sharpe_not_lower": bool(
            candidate_active["Sharpe"] >= baseline_active["Sharpe"]
        ),
        "active_mdd_not_worse": bool(
            candidate_active["MDD"] >= baseline_active["MDD"]
        ),
        "active_cagr_not_lower_by_more_than_50bp": bool(
            candidate_active["CAGR"] >= baseline_active["CAGR"] - 0.005
        ),
        "locked_sharpe_not_lower": bool(
            candidate_locked["Sharpe"] >= baseline_locked["Sharpe"]
        ),
        "locked_mdd_not_worse": bool(
            candidate_locked["MDD"] >= baseline_locked["MDD"]
        ),
    }
    promote = bool(all(gates.values()))
    report: dict[str, Any] = {
        "study": "Stage45_VolatilityTargetedShrinkageTinyMLP",
        "base_strategy": "Stage36_GVZ_OVXAssetRisk",
        "decision": (
            "promote_stage45_tiny_mlp_lw_vol_target"
            if promote
            else "retain_stage36_and_keep_stage45_as_research_only"
        ),
        "promote": promote,
        "fixed_design": {
            "assets": list(ASSETS),
            "target": "next-month realized asset return minus prior-known KTB 3Y yield / 12",
            "normalization": "training-window time-series StandardScaler; no cross-sectional ranks",
            "model": "pooled 4-asset MLP, one tanh hidden layer",
            "hidden_units": HIDDEN_UNITS,
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "minimum_training_months": MIN_TRAIN_MONTHS,
            "refit_frequency": "annual",
            "covariance": "252 complete prior daily rows; constant-correlation Ledoit-Wolf; monthly scale",
            "stage36_risk_overlay": "GVZ only scales GLD variance; OVX only scales USO variance",
            "target_annual_volatility": TARGET_ANNUAL_VOLATILITY,
            "portfolio": "long-only risky assets; sum risky <= 1; remainder in cash; no leverage",
            "cdar_guard": stage36.stage35.CATASTROPHE_CDAR,
            "transaction_costs": "same domestic and foreign-weight-change rates as Stage36",
            "hyperparameter_search": None,
        },
        "model_activation_month": str(active.index.min()),
        "performance_gates": gates,
        "headline_deltas_vs_stage36": {
            "mlp_active_cagr": float(
                candidate_active["CAGR"] - baseline_active["CAGR"]
            ),
            "mlp_active_sharpe": float(
                candidate_active["Sharpe"] - baseline_active["Sharpe"]
            ),
            "mlp_active_mdd": float(
                candidate_active["MDD"] - baseline_active["MDD"]
            ),
            "locked_cagr": float(
                candidate_locked["CAGR"] - baseline_locked["CAGR"]
            ),
            "locked_sharpe": float(
                candidate_locked["Sharpe"] - baseline_locked["Sharpe"]
            ),
            "locked_mdd": float(
                candidate_locked["MDD"] - baseline_locked["MDD"]
            ),
        },
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage36": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "forecast_diagnostics": forecast,
        "solver_audit": {
            "active_months": int(len(active)),
            "successes": int(active["solver_success"].sum()),
            "fallbacks": int(active["used_fallback"].sum()),
            "minimum_budget_slack": float(active["budget_slack"].min()),
            "minimum_volatility_slack": float(
                active["volatility_slack"].min()
            ),
            "minimum_cdar_slack": float(active["cdar_slack"].min()),
            "maximum_risky_weight_sum": float(
                active["risky_weight_sum"].max()
            ),
            "average_cash_weight": float(active["cash_weight"].mean()),
            "average_lw_shrinkage": float(active["lw_shrinkage"].mean()),
        },
        "causality_audit": {
            "all_model_cutoffs_before_target": bool(
                (
                    pd.PeriodIndex(active["model_fit_cutoff"], freq="M")
                    < active.index
                ).all()
            ),
            "all_covariance_dates_before_target": bool(
                (
                    pd.to_datetime(active["covariance_end"])
                    < active.index.to_timestamp(how="start")
                ).all()
            ),
            "all_covariance_windows_have_252_rows": bool(
                active["covariance_observations"].eq(
                    COVARIANCE_LOOKBACK_DAYS
                ).all()
            ),
            "feature_signal_month_precedes_target": bool(
                (
                    panel["feature_signal_month"]
                    < panel.index.get_level_values("month")
                ).all()
            ),
        },
        "reproduction_audit": {
            "warmup_months": int(len(warmup)),
            "maximum_absolute_return_error": warmup_return_error,
            "maximum_absolute_weight_error": warmup_weight_error,
        },
        "checks": {
            "stage36_frozen_files_unchanged": frozen_before == frozen_after,
            "no_cross_sectional_rank": True,
            "no_cross_sectional_target_demeaning": True,
            "cash_excess_target": True,
            "long_only_no_leverage": bool(
                (active[WEIGHT_COLUMNS].to_numpy() >= -1e-10).all()
                and (active["risky_weight_sum"] <= 1.0 + 1e-8).all()
            ),
            "all_active_solvers_feasible": bool(
                active["solver_success"].all()
                and (active["budget_slack"] >= -1e-8).all()
                and (active["volatility_slack"] >= -1e-7).all()
                and (active["cdar_slack"] >= -1e-7).all()
            ),
            "warmup_reproduces_stage36": bool(
                warmup_return_error < 1e-12 and warmup_weight_error < 1e-12
            ),
        },
        "source_code": {
            "path": str(ORIGINAL_SOURCE),
            "sha256": _sha256(ORIGINAL_SOURCE)
            if ORIGINAL_SOURCE.exists()
            else None,
        },
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
        "return_audit": {
            "rows": int(len(returns)),
            "first_month": str(returns.index.min()),
            "last_month": str(returns.index.max()),
            "level_rows": int(len(return_levels)),
        },
        "macro_audit": {
            "probability_rows": int(len(probabilities)),
            "first_target_month": str(probabilities.index.min()),
            "last_target_month": str(probabilities.index.max()),
            "rank_rows": int(len(macro_ranks)),
        },
        "fundamental_audit": fundamental_audit,
        "asset_volatility_audit": asset_volatility_audit,
        "market_audit": market_audit,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(OUTPUT_DIR / "stage45_monthly.csv")
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage36.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "month": month,
                    "asset": asset,
                    "predicted_excess_return": candidate.loc[
                        month, f"predicted_excess_return_{asset}"
                    ],
                    "realized_excess_return": candidate.loc[
                        month, f"realized_excess_return_{asset}"
                    ],
                }
                for month in active.index
                for asset in ASSETS
            ]
        ).to_csv(OUTPUT_DIR / "oos_forecasts.csv", index=False)
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return {
        "report": report,
        "performance": performance,
        "bootstrap": bootstrap,
        "candidate": candidate,
        "panel": panel,
    }


def main() -> None:
    research = run_research(save=True)
    print(
        json.dumps(
            research["report"], ensure_ascii=False, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
