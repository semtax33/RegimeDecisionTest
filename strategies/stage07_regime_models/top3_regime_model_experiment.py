from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from jumpmodels.jump import JumpModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from strategies.core.regime_research import ASSETS, load_macro_data, load_monthly_asset_returns, performance_summary


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", message="Workbook contains no default style")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")
HORIZONS = (1, 3, 6)
MODELS = ("SJM", "CJM", "TVTP-HMM", "CJM+LightGBM")
RANDOM_STATE = 20260827


@dataclass(frozen=True)
class OverlayConfig:
    threshold: float
    max_shift: float
    bond_share: float
    horizon_weights: tuple[float, float, float]

    @property
    def name(self) -> str:
        weights = "-".join(f"{value:.2f}" for value in self.horizon_weights)
        return (
            f"th{self.threshold:.2f}_shift{self.max_shift:.2f}"
            f"_bond{self.bond_share:.2f}_hw{weights}"
        )


def _period_index(values: pd.Series | pd.Index) -> pd.PeriodIndex:
    return pd.PeriodIndex(values.astype(str), freq="M")


def _read_period_csv(path: Path, index_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[index_column] = _period_index(frame[index_column])
    return frame.set_index(index_column).sort_index()


def _rolling_compound(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1


def build_master_features() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build month-end features that were observable by each signal month."""
    macro, _ = load_macro_data()
    macro = macro.copy()
    macro.columns = [f"macro_{group}_{name}" for group, name in macro.columns]
    macro.index = macro.index.to_period("M")

    asset_returns, _ = load_monthly_asset_returns(refresh=False)
    asset_returns.index = pd.PeriodIndex(asset_returns.index, freq="M")
    market = pd.DataFrame(index=asset_returns.index)
    for asset in ASSETS:
        series = asset_returns[asset]
        market[f"market_{asset}_ret1"] = series
        market[f"market_{asset}_ret3"] = _rolling_compound(series, 3)
        market[f"market_{asset}_ret6"] = _rolling_compound(series, 6)
    market["market_KODEX200_vol3"] = asset_returns["KODEX200"].rolling(3).std(ddof=1) * math.sqrt(12)
    market["market_KODEX200_vol6"] = asset_returns["KODEX200"].rolling(6).std(ddof=1) * math.sqrt(12)
    wealth = (1 + asset_returns["KODEX200"]).cumprod()
    market["market_KODEX200_drawdown"] = wealth / wealth.cummax() - 1
    market["market_KODEX200_drawdown12"] = market["market_KODEX200_drawdown"].rolling(12).min()

    stress = _read_period_csv(CACHE / "stress_monthly.csv", "date")
    stress_columns = [
        column
        for column in stress.columns
        if column
        in {
            "VIX_last",
            "VIX_last_d1",
            "VIX_last_d3",
            "VIX_last_z60",
            "BAA_SPREAD_last",
            "BAA_SPREAD_last_d1",
            "BAA_SPREAD_last_d3",
            "BAA_SPREAD_last_z60",
            "NFCI_last",
            "NFCI_last_d1",
            "NFCI_last_d3",
            "NFCI_last_z60",
            "STLFSI_last",
            "STLFSI_last_d1",
            "STLFSI_last_d3",
            "STLFSI_last_z60",
        }
    ]
    stress = stress[stress_columns].add_prefix("stress_")
    for base in ("VIX_last", "BAA_SPREAD_last", "NFCI_last", "STLFSI_last"):
        d1 = f"stress_{base}_d1"
        if d1 in stress:
            stress[f"stress_{base}_acceleration"] = stress[d1].diff()

    vkospi = _read_period_csv(RESULTS / "vkospi_features.csv", "month")
    vkospi_columns = [
        "vkospi_raw_close",
        "vkospi_return_5",
        "vkospi_return_21",
        "vkospi_return_63",
        "vkospi_mean_ratio_21",
        "vkospi_level_z_63",
        "vkospi_level_z_252",
        "vkospi_return_vol_21",
        "vkospi_positive_fraction_21",
        "vkospi_oap_high52",
        "vkospi_oap_realized_vol_21",
    ]
    vkospi = vkospi[[column for column in vkospi_columns if column in vkospi]]

    signals = _read_period_csv(RESULTS / "regime_signals.csv", "target_month")
    signals["signal_month"] = _period_index(signals["signal_month"])
    signals = signals.set_index("signal_month", drop=False)
    sjm = signals[
        [
            "p_growth_high",
            "p_inflation_high",
            "p_growth_sjm",
            "p_inflation_sjm",
            "p_Goldilocks",
            "p_Overheating",
            "p_Slowdown",
            "p_Stagflation",
            "growth_switches",
            "inflation_switches",
        ]
    ].copy()
    sjm.columns = [f"sjm_{column}" for column in sjm.columns]
    sjm["sjm_p_risk_off"] = 1 - sjm["sjm_p_growth_high"]
    sjm["sjm_p_risk_off_d1"] = sjm["sjm_p_risk_off"].diff()
    sjm["sjm_p_risk_off_d3"] = sjm["sjm_p_risk_off"].diff(3)

    features = pd.concat([macro, market, stress, vkospi, sjm], axis=1, sort=True)
    features = features.replace([np.inf, -np.inf], np.nan).sort_index()
    features = features.loc[pd.Period("2005-04", "M") :]
    useful = features.notna().mean() >= 0.35
    features = features.loc[:, useful]

    risk_off_label = (asset_returns["KODEX200"] < 0).astype(float)
    risk_off_label.name = "risk_off"
    baseline = _read_period_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", "month"
    )
    return features, risk_off_label, asset_returns, baseline


def _robust_frame(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, RobustScaler]:
    usable = history.notna().mean() >= 0.65
    selected = history.loc[:, usable]
    medians = selected.median()
    filled = selected.fillna(medians)
    scaler = RobustScaler(quantile_range=(25, 75)).fit(filled)
    scaled = pd.DataFrame(
        scaler.transform(filled), index=filled.index, columns=filled.columns
    ).clip(-6, 6)
    return scaled, medians, scaler


CJM_CANDIDATES = (
    "market_KODEX200_ret1",
    "market_KODEX200_ret3",
    "market_KODEX200_ret6",
    "market_KODEX200_vol3",
    "market_KODEX200_vol6",
    "market_KODEX200_drawdown12",
    "market_USDKRW_ret1",
    "stress_VIX_last_z60",
    "stress_BAA_SPREAD_last_z60",
    "stress_NFCI_last_z60",
    "stress_STLFSI_last_z60",
    "vkospi_return_21",
    "vkospi_level_z_63",
    "vkospi_level_z_252",
    "vkospi_return_vol_21",
)


def _smoothed_transition(matrix: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        return np.full((2, 2), 0.5)
    matrix = (1 - smoothing) * matrix + smoothing * 0.5
    return matrix / matrix.sum(axis=1, keepdims=True)


def _current_duration(states: np.ndarray) -> int:
    if len(states) == 0:
        return 0
    current = states[-1]
    duration = 1
    for value in states[-2::-1]:
        if value != current:
            break
        duration += 1
    return duration


def build_cjm_forecasts(
    features: pd.DataFrame,
    returns: pd.Series,
    signal_months: pd.PeriodIndex,
    jump_penalty: float = 3.0,
) -> pd.DataFrame:
    """Causal expanding-window Continuous Jump Model forecasts."""
    columns = [column for column in CJM_CANDIDATES if column in features]
    source = features[columns]
    rows: list[dict[str, object]] = []
    for number, signal_month in enumerate(signal_months):
        common = source.loc[:signal_month].index.intersection(returns.dropna().index)
        if len(common) < 36:
            continue
        common = common[-144:]
        raw_history = source.loc[common]
        scaled, _, _ = _robust_frame(raw_history)
        aligned_returns = returns.reindex(scaled.index).astype(float)
        if scaled.shape[1] < 5 or aligned_returns.notna().sum() < 36:
            continue
        try:
            model = JumpModel(
                n_components=2,
                jump_penalty=jump_penalty,
                cont=True,
                grid_size=0.10,
                mode_loss=True,
                random_state=RANDOM_STATE,
                max_iter=250,
                n_init=3,
                tol=1e-7,
            )
            model.fit(scaled, ret_ser=aligned_returns, sort_by="ret")
            online = model.predict_proba_online(scaled)
            risk_state = int(np.nanargmin(np.asarray(model.ret_, dtype=float)))
            current = online.iloc[-1].to_numpy(dtype=float)
            transition = _smoothed_transition(model.transmat_)
            historical_risk_rate = float((aligned_returns < 0).mean())
            forecast = {}
            for horizon in HORIZONS:
                raw_forecast = float(
                    (current @ np.linalg.matrix_power(transition, horizon))[risk_state]
                )
                forecast[horizon] = 0.90 * raw_forecast + 0.10 * historical_risk_rate
            risk_path = online.iloc[:, risk_state].astype(float)
            hard_path = (risk_path >= 0.5).astype(int).to_numpy()
            rows.append(
                {
                    "signal_month": signal_month,
                    "p_current": float(current[risk_state]),
                    **{f"p_h{horizon}": forecast[horizon] for horizon in HORIZONS},
                    "p_delta1": float(risk_path.diff().iloc[-1]) if len(risk_path) > 1 else 0.0,
                    "p_delta3": float(risk_path.diff(3).iloc[-1]) if len(risk_path) > 3 else 0.0,
                    "duration": _current_duration(hard_path),
                    "switches12": int(np.sum(hard_path[-12:][1:] != hard_path[-12:][:-1])),
                    "train_rows": len(scaled),
                    "feature_count": scaled.shape[1],
                    "risk_state_return": float(model.ret_[risk_state]),
                    "fit_success": True,
                }
            )
        except Exception as error:
            rows.append(
                {
                    "signal_month": signal_month,
                    "fit_success": False,
                    "error": type(error).__name__,
                }
            )
        if (number + 1) % 36 == 0:
            print(f"CJM forecasts: {number + 1}/{len(signal_months)}")
    frame = pd.DataFrame(rows).set_index("signal_month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index()


def build_sjm_forecasts(features: pd.DataFrame, signal_months: pd.PeriodIndex) -> pd.DataFrame:
    source = features["sjm_p_risk_off"].dropna().clip(0.01, 0.99)
    rows = []
    for signal_month in signal_months:
        history = source.loc[:signal_month]
        if len(history) < 12:
            continue
        states = (history >= 0.5).astype(int).to_numpy()
        counts = np.ones((2, 2), dtype=float)
        for previous, current_state in zip(states[:-1], states[1:]):
            counts[previous, current_state] += 1
        transition = counts / counts.sum(axis=1, keepdims=True)
        current = np.array([1 - history.iloc[-1], history.iloc[-1]], dtype=float)
        rows.append(
            {
                "signal_month": signal_month,
                "p_current": float(current[1]),
                **{
                    f"p_h{horizon}": float(
                        (current @ np.linalg.matrix_power(transition, horizon))[1]
                    )
                    for horizon in HORIZONS
                },
                "train_rows": len(history),
            }
        )
    frame = pd.DataFrame(rows).set_index("signal_month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index()


TVTP_CANDIDATES = (
    "market_KODEX200_drawdown12",
    "market_KODEX200_vol6",
    "stress_BAA_SPREAD_last_z60",
    "stress_NFCI_last_z60",
    "stress_VIX_last_z60",
    "vkospi_level_z_252",
)


def _fit_tvtp_fallback(
    x_scaled: pd.DataFrame,
    risk_labels: pd.Series,
    current_probability: float,
) -> tuple[dict[int, float], str]:
    x_transition = x_scaled.shift(1).iloc[1:]
    y_transition = risk_labels.reindex(x_scaled.index).iloc[1:].astype(int)
    previous_state = risk_labels.reindex(x_scaled.index).shift(1).iloc[1:].rename("previous")
    design = pd.concat([previous_state, x_transition], axis=1).dropna()
    target = y_transition.reindex(design.index)
    if len(design) < 30 or target.nunique() < 2:
        return {horizon: current_probability for horizon in HORIZONS}, "persistence"
    model = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(design, target)
    current_x = x_scaled.iloc[-1]
    probability = current_probability
    output = {}
    for step in range(1, max(HORIZONS) + 1):
        row0 = pd.DataFrame([[0.0, *current_x]], columns=design.columns)
        row1 = pd.DataFrame([[1.0, *current_x]], columns=design.columns)
        p0 = float(model.predict_proba(row0)[0, 1])
        p1 = float(model.predict_proba(row1)[0, 1])
        probability = (1 - probability) * p0 + probability * p1
        if step in HORIZONS:
            output[step] = float(probability)
    return output, "logistic_tvtp_fallback"


def build_tvtp_forecasts(
    features: pd.DataFrame,
    returns: pd.Series,
    labels: pd.Series,
    signal_months: pd.PeriodIndex,
) -> pd.DataFrame:
    """Fit a Gaussian two-state Markov regression with TVTP covariates."""
    columns = [column for column in TVTP_CANDIDATES if column in features]
    source = features[columns]
    rows: list[dict[str, object]] = []
    for number, signal_month in enumerate(signal_months):
        common = source.loc[:signal_month].index.intersection(returns.dropna().index)
        if len(common) < 60:
            continue
        common = common[-144:]
        raw_history = source.loc[common]
        scaled, _, _ = _robust_frame(raw_history)
        endog = returns.reindex(scaled.index).astype(float)
        if scaled.shape[1] < 3:
            continue
        transition_x = sm.add_constant(scaled.shift(1), has_constant="add").iloc[1:]
        endog_fit = endog.iloc[1:]
        try:
            model = MarkovRegression(
                endog_fit,
                k_regimes=2,
                trend="c",
                switching_variance=True,
                exog_tvtp=transition_x,
            )
            result = model.fit(
                method="bfgs",
                maxiter=120,
                disp=False,
                em_iter=8,
                search_reps=0,
            )
            filtered = result.filtered_marginal_probabilities
            weighted_returns = np.array(
                [float((filtered[state] * endog_fit).sum() / filtered[state].sum()) for state in range(2)]
            )
            weighted_volatility = np.array(
                [
                    math.sqrt(
                        float(
                            (
                                filtered[state]
                                * (endog_fit - weighted_returns[state]).pow(2)
                            ).sum()
                            / filtered[state].sum()
                        )
                    )
                    for state in range(2)
                ]
            )
            risk_state = int(np.nanargmin(weighted_returns - 0.25 * weighted_volatility))
            probability = filtered.iloc[-1].to_numpy(dtype=float)
            forecast_x = sm.add_constant(scaled.iloc[[-1]], has_constant="add")
            transition = model.regime_transition_matrix(
                result.params, exog_tvtp=forecast_x
            )[:, :, 0]
            forecasts = {}
            for step in range(1, max(HORIZONS) + 1):
                probability = transition @ probability
                if step in HORIZONS:
                    raw_forecast = float(probability[risk_state])
                    historical_risk_rate = float(labels.reindex(scaled.index).mean())
                    forecasts[step] = 0.80 * raw_forecast + 0.20 * historical_risk_rate
            method = "statsmodels_tvtp_markov_regression"
            current_probability = float(filtered.iloc[-1, risk_state])
            converged = bool(result.mle_retvals.get("converged", False))
        except Exception:
            current_probability = float(labels.reindex(scaled.index).tail(12).mean())
            forecasts, method = _fit_tvtp_fallback(
                scaled,
                labels,
                current_probability,
            )
            weighted_returns = np.array([np.nan, np.nan])
            risk_state = 1
            converged = False
        rows.append(
            {
                "signal_month": signal_month,
                "p_current": float(np.clip(current_probability, 0.01, 0.99)),
                **{
                    f"p_h{horizon}": float(np.clip(forecasts[horizon], 0.01, 0.99))
                    for horizon in HORIZONS
                },
                "train_rows": len(scaled),
                "feature_count": scaled.shape[1],
                "risk_state": risk_state,
                "risk_state_return": float(weighted_returns[risk_state]),
                "fit_method": method,
                "converged": converged,
            }
        )
        if (number + 1) % 24 == 0:
            print(f"TVTP-HMM forecasts: {number + 1}/{len(signal_months)}")
    frame = pd.DataFrame(rows).set_index("signal_month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.sort_index()


LGBM_PREFERRED = (
    "sjm_p_growth_high",
    "sjm_p_inflation_high",
    "sjm_p_risk_off",
    "sjm_p_risk_off_d1",
    "sjm_p_risk_off_d3",
    "market_KODEX200_ret1",
    "market_KODEX200_ret3",
    "market_KODEX200_ret6",
    "market_KODEX200_vol3",
    "market_KODEX200_vol6",
    "market_KODEX200_drawdown12",
    "market_BOND_ret3",
    "market_GLD_ret3",
    "market_USO_ret3",
    "macro_growth_GDP_level",
    "macro_growth_Export_level",
    "macro_growth_BSI_level",
    "macro_growth_GDP_level_d3",
    "macro_growth_Export_level_d3",
    "macro_growth_BSI_level_d3",
    "stress_VIX_last",
    "stress_VIX_last_d1",
    "stress_VIX_last_d3",
    "stress_VIX_last_z60",
    "stress_BAA_SPREAD_last",
    "stress_BAA_SPREAD_last_d1",
    "stress_BAA_SPREAD_last_d3",
    "stress_BAA_SPREAD_last_z60",
    "stress_NFCI_last",
    "stress_NFCI_last_d1",
    "stress_NFCI_last_d3",
    "stress_NFCI_last_z60",
    "stress_STLFSI_last_z60",
    "vkospi_raw_close",
    "vkospi_return_21",
    "vkospi_return_63",
    "vkospi_level_z_63",
    "vkospi_level_z_252",
    "vkospi_return_vol_21",
)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def build_lightgbm_forecasts(
    features: pd.DataFrame,
    labels: pd.Series,
    cjm: pd.DataFrame,
    signal_months: pd.PeriodIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = [column for column in LGBM_PREFERRED if column in features]
    x_source = features[selected].copy()
    cjm_features = cjm[
        [
            column
            for column in (
                "p_current",
                "p_h1",
                "p_h3",
                "p_h6",
                "p_delta1",
                "p_delta3",
                "duration",
                "switches12",
            )
            if column in cjm
        ]
    ].add_prefix("cjm_")
    x_source = x_source.join(cjm_features, how="left")
    importance = pd.Series(0.0, index=x_source.columns)
    importance_fits = 0
    rows: list[dict[str, object]] = []

    for number, signal_month in enumerate(signal_months):
        if signal_month not in x_source.index:
            continue
        for horizon in HORIZONS:
            target = labels.shift(-horizon)
            realization = pd.Series(
                [month + horizon for month in labels.index], index=labels.index
            )
            train_index = x_source.index.intersection(target.dropna().index)
            train_index = train_index[realization.reindex(train_index) <= signal_month]
            if len(train_index) < 60:
                continue
            train_index = train_index[-144:]
            if realization.loc[train_index].max() > signal_month:
                raise AssertionError("A LightGBM training label crosses the signal month")
            x_all = x_source.loc[train_index]
            y_all = target.loc[train_index].astype(int)
            usable = x_all.notna().mean() >= 0.60
            columns = list(x_all.columns[usable])
            if len(columns) < 12 or y_all.nunique() < 2:
                continue
            validation_size = min(24, max(15, len(x_all) // 5))
            split = len(x_all) - validation_size
            x_train, x_valid = x_all.iloc[:split][columns], x_all.iloc[split:][columns]
            y_train, y_valid = y_all.iloc[:split], y_all.iloc[split:]
            medians = x_train.median()
            x_train = x_train.fillna(medians)
            x_valid = x_valid.fillna(medians)
            x_test = x_source.loc[[signal_month], columns].fillna(medians)
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.025,
                max_depth=3,
                num_leaves=7,
                min_child_samples=8,
                subsample=0.80,
                colsample_bytree=0.70,
                reg_alpha=0.25,
                reg_lambda=0.50,
                class_weight="balanced",
                random_state=RANDOM_STATE + horizon,
                n_jobs=1,
                verbosity=-1,
            )
            callbacks = [lgb.early_stopping(35, verbose=False)]
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
            valid_raw = model.predict_proba(x_valid)[:, 1]
            raw_probability = float(model.predict_proba(x_test)[:, 1][0])
            calibration_method = "base_rate_shrinkage"
            calibrated = 0.75 * raw_probability + 0.25 * float(y_train.mean())
            if y_valid.nunique() == 2 and len(y_valid) >= 15:
                calibrator = LogisticRegression(
                    C=0.50,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )
                calibrator.fit(_logit(valid_raw).reshape(-1, 1), y_valid)
                calibrated = float(
                    calibrator.predict_proba(
                        _logit(np.array([raw_probability])).reshape(-1, 1)
                    )[0, 1]
                )
                calibration_method = "platt_recent_validation"
            gain = pd.Series(
                model.booster_.feature_importance(importance_type="gain"),
                index=columns,
            )
            if gain.sum() > 0:
                importance.loc[columns] += gain / gain.sum()
                importance_fits += 1
            rows.append(
                {
                    "signal_month": signal_month,
                    "horizon": horizon,
                    "probability": float(np.clip(calibrated, 0.01, 0.99)),
                    "raw_probability": float(np.clip(raw_probability, 0.01, 0.99)),
                    "calibration_method": calibration_method,
                    "train_rows": len(x_all),
                    "feature_count": len(columns),
                    "fit_end_month": train_index[-1],
                    "max_label_month": realization.loc[train_index].max(),
                    "best_iteration": int(model.best_iteration_ or model.n_estimators),
                }
            )
        if (number + 1) % 24 == 0:
            print(f"CJM+LightGBM forecasts: {number + 1}/{len(signal_months)}")
    long = pd.DataFrame(rows)
    probability_wide = long.pivot(index="signal_month", columns="horizon", values="probability")
    probability_wide.columns = [f"p_h{int(column)}" for column in probability_wide.columns]
    probability_wide.index = pd.PeriodIndex(probability_wide.index, freq="M")
    importance_frame = (
        (importance / max(importance_fits, 1))
        .sort_values(ascending=False)
        .rename("mean_gain_share")
        .to_frame()
        .reset_index(names="feature")
    )
    long["signal_month"] = pd.PeriodIndex(long["signal_month"], freq="M")
    long["fit_end_month"] = pd.PeriodIndex(long["fit_end_month"], freq="M")
    long["max_label_month"] = pd.PeriodIndex(long["max_label_month"], freq="M")
    return probability_wide, importance_frame, long


def expected_calibration_error(y: pd.Series, p: pd.Series, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    groups = pd.cut(p, bins=edges, include_lowest=True, duplicates="drop")
    total = len(y)
    error = 0.0
    for _, index in groups.groupby(groups, observed=False).groups.items():
        if len(index) == 0:
            continue
        error += len(index) / total * abs(float(y.loc[index].mean()) - float(p.loc[index].mean()))
    return float(error)


def probability_metric_rows(
    forecasts: dict[str, pd.DataFrame], labels: pd.Series
) -> pd.DataFrame:
    rows = []
    for model_name, frame in forecasts.items():
        for horizon in HORIZONS:
            column = f"p_h{horizon}"
            if column not in frame:
                continue
            prediction = frame[column].dropna().clip(1e-5, 1 - 1e-5)
            target = labels.shift(-horizon).reindex(prediction.index)
            outcome_month = pd.Series(
                [month + horizon for month in prediction.index], index=prediction.index
            )
            current_state = labels.reindex(prediction.index)
            for period_name, mask in (
                ("full_oos", pd.Series(True, index=prediction.index)),
                ("locked_2018_2026", outcome_month >= LOCKED_START),
            ):
                valid = mask & target.notna() & current_state.notna()
                y = target.loc[valid].astype(int)
                p = prediction.loc[valid]
                current = current_state.loc[valid].astype(int)
                if len(y) < 12 or y.nunique() < 2:
                    continue
                actual_transition = (y != current).astype(int)
                predicted_transition = ((p >= 0.5).astype(int) != current).astype(int)
                rows.append(
                    {
                        "Period": period_name,
                        "Model": model_name,
                        "Horizon": horizon,
                        "Observations": len(y),
                        "Brier": float(brier_score_loss(y, p)),
                        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
                        "ECE5": expected_calibration_error(y, p, bins=5),
                        "AUC": float(roc_auc_score(y, p)),
                        "BalancedAccuracy": float(
                            balanced_accuracy_score(y, p >= 0.5)
                        ),
                        "TransitionRecall": float(
                            recall_score(
                                actual_transition,
                                predicted_transition,
                                zero_division=0,
                            )
                        ),
                        "MeanProbability": float(p.mean()),
                        "RiskOffRate": float(y.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _forecast_long(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, frame in forecasts.items():
        for signal_month, values in frame.iterrows():
            rows.append(
                {
                    "signal_month": signal_month,
                    "Model": model_name,
                    **{
                        f"p_h{horizon}": values.get(f"p_h{horizon}", np.nan)
                        for horizon in HORIZONS
                    },
                    "p_current": values.get("p_current", np.nan),
                }
            )
    return pd.DataFrame(rows)


def run_probability_overlay(
    baseline: pd.DataFrame,
    asset_returns: pd.DataFrame,
    forecast: pd.DataFrame,
    config: OverlayConfig,
) -> pd.DataFrame:
    months = baseline.index.intersection(asset_returns.index)
    rows = []
    nav = 1.0
    peak = 1.0
    previous_shift = 0.0
    previous_gld = 0.0
    weights = np.array(config.horizon_weights, dtype=float)
    weights /= weights.sum()
    for month in months:
        signal_month = month - 1
        if signal_month in forecast.index:
            values = np.array(
                [forecast.loc[signal_month].get(f"p_h{horizon}", np.nan) for horizon in HORIZONS],
                dtype=float,
            )
            if np.isfinite(values).all():
                risk_probability = float(values @ weights)
            else:
                risk_probability = 0.5
        else:
            risk_probability = 0.5
        activation = float(
            np.clip(
                (risk_probability - config.threshold) / max(1 - config.threshold, 1e-6),
                0,
                1,
            )
        )
        shift = config.max_shift * activation
        bond_return = float(asset_returns.loc[month, "BOND"])
        gld_return = float(asset_returns.loc[month, "GLD"])
        defensive_return = config.bond_share * bond_return + (1 - config.bond_share) * gld_return
        base_return = float(baseline.loc[month, "return"])
        incremental_turnover = 2 * abs(shift - previous_shift)
        gld_exposure = (1 - config.bond_share) * shift
        incremental_trade_cost = incremental_turnover * 0.0015
        incremental_fx_cost = abs(gld_exposure - previous_gld) * 0.0005
        net_return = (
            (1 - shift) * base_return
            + shift * defensive_return
            - incremental_trade_cost
            - incremental_fx_cost
        )
        nav *= 1 + net_return
        peak = max(peak, nav)
        rows.append(
            {
                "month": month,
                "return": net_return,
                "baseline_return": base_return,
                "defensive_return": defensive_return,
                "risk_probability": risk_probability,
                "activation": activation,
                "defensive_shift": shift,
                "turnover": float(baseline.loc[month].get("turnover", 0.0))
                + incremental_turnover,
                "incremental_turnover": incremental_turnover,
                "incremental_trade_cost": incremental_trade_cost,
                "incremental_fx_cost": incremental_fx_cost,
                "nav": nav,
                "drawdown": nav / peak - 1,
            }
        )
        previous_shift = shift
        previous_gld = gld_exposure
    return pd.DataFrame(rows).set_index("month")


def _metric_record(period: str, strategy: str, frame: pd.DataFrame) -> dict[str, object]:
    summary = performance_summary(frame["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **summary.to_dict(),
        "AvgTurnover": float(frame["turnover"].mean()),
        "AvgDefensiveShift": float(frame.get("defensive_shift", pd.Series(0.0, index=frame.index)).mean()),
        "ActiveMonths": int((frame.get("defensive_shift", pd.Series(0.0, index=frame.index)) > 1e-6).sum()),
    }


def calibrate_overlays(
    baseline: pd.DataFrame,
    asset_returns: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, tuple[OverlayConfig, pd.DataFrame]]]:
    base_cal = baseline.loc[:CAL_END]
    base_metrics = performance_summary(base_cal["return"])
    rows = []
    paths: dict[tuple[str, str], pd.DataFrame] = {}
    weight_options = ((1.0, 0.0, 0.0), (0.5, 0.3, 0.2), (1 / 3, 1 / 3, 1 / 3))
    for model_name, forecast in forecasts.items():
        if model_name == "SJM":
            continue
        for threshold in (0.50, 0.55, 0.60, 0.65):
            for max_shift in (0.10, 0.20, 0.30, 0.40):
                for bond_share in (0.0, 0.5, 1.0):
                    for horizon_weights in weight_options:
                        config = OverlayConfig(
                            threshold=threshold,
                            max_shift=max_shift,
                            bond_share=bond_share,
                            horizon_weights=horizon_weights,
                        )
                        backtest = run_probability_overlay(
                            baseline, asset_returns, forecast, config
                        )
                        paths[(model_name, config.name)] = backtest
                        sample = backtest.loc[:CAL_END]
                        metrics = performance_summary(sample["return"])
                        rows.append(
                            {
                                "Model": model_name,
                                "Config": config.name,
                                **asdict(config),
                                **metrics.to_dict(),
                                "AvgTurnover": float(sample["turnover"].mean()),
                                "AvgDefensiveShift": float(sample["defensive_shift"].mean()),
                                "ActiveMonths": int((sample["defensive_shift"] > 1e-6).sum()),
                                "CAGRDelta": float(metrics["CAGR"] - base_metrics["CAGR"]),
                                "SharpeDelta": float(metrics["Sharpe"] - base_metrics["Sharpe"]),
                                "MDDDelta": float(metrics["MDD"] - base_metrics["MDD"]),
                                "CalmarDelta": float(metrics["Calmar"] - base_metrics["Calmar"]),
                            }
                        )
    calibration = pd.DataFrame(rows)
    winners: dict[str, tuple[OverlayConfig, pd.DataFrame]] = {}
    for model_name, group in calibration.groupby("Model", sort=False):
        group = group.copy()
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
            group[f"Rank_{metric}"] = group[metric].rank(pct=True)
        group["MultiObjectiveScore"] = group[
            ["Rank_CAGR", "Rank_Sharpe", "Rank_MDD", "Rank_Calmar"]
        ].mean(axis=1)
        eligible = group.loc[
            (group["CAGR"] >= 0.97 * base_metrics["CAGR"])
            & (group["MDD"] >= base_metrics["MDD"] - 0.01)
            & (group["ActiveMonths"] >= 6)
        ]
        if eligible.empty:
            eligible = group.loc[group["ActiveMonths"] >= 3]
        winner = eligible.sort_values(
            ["MultiObjectiveScore", "Sharpe", "Calmar", "CAGR"],
            ascending=False,
        ).iloc[0]
        config = OverlayConfig(
            threshold=float(winner["threshold"]),
            max_shift=float(winner["max_shift"]),
            bond_share=float(winner["bond_share"]),
            horizon_weights=tuple(winner["horizon_weights"]),
        )
        winners[model_name] = (
            config,
            paths[(model_name, str(winner["Config"]))],
        )
        calibration.loc[group.index, "MultiObjectiveScore"] = group[
            "MultiObjectiveScore"
        ]
        calibration.loc[group.index, "Selected"] = False
        calibration.loc[winner.name, "Selected"] = True
    return calibration, winners


def paired_multiobjective_bootstrap(
    reference: pd.Series,
    candidate: pd.Series,
    simulations: int = 5000,
    block_length: int = 6,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    joined = pd.concat([reference.rename("reference"), candidate.rename("candidate")], axis=1).dropna()
    values = joined.to_numpy(dtype=float)
    count = len(values)
    rng = np.random.default_rng(seed)
    deltas = np.zeros((simulations, 3), dtype=float)
    for simulation in range(simulations):
        starts = rng.integers(0, count, size=math.ceil(count / block_length))
        indices = np.concatenate(
            [(start + np.arange(block_length)) % count for start in starts]
        )[:count]
        reference_metrics = performance_summary(values[indices, 0])
        candidate_metrics = performance_summary(values[indices, 1])
        deltas[simulation] = [
            candidate_metrics["CAGR"] - reference_metrics["CAGR"],
            candidate_metrics["Sharpe"] - reference_metrics["Sharpe"],
            candidate_metrics["MDD"] - reference_metrics["MDD"],
        ]
    return {
        "block_length": block_length,
        "simulations": simulations,
        "probability_cagr_improves": float(np.mean(deltas[:, 0] > 0)),
        "probability_sharpe_improves": float(np.mean(deltas[:, 1] > 0)),
        "probability_mdd_improves": float(np.mean(deltas[:, 2] > 0)),
        "probability_all_three_improve": float(np.mean((deltas > 0).all(axis=1))),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    features, labels, asset_returns, baseline = build_master_features()
    signal_months = pd.period_range(
        max(features.index.min(), pd.Period("2007-01", "M")),
        min(features.index.max(), labels.index.max()),
        freq="M",
    )
    kodex_returns = asset_returns["KODEX200"]

    print("Building causal SJM and official continuous jump model forecasts...")
    sjm = build_sjm_forecasts(features, signal_months)
    cjm = build_cjm_forecasts(features, kodex_returns, signal_months)
    print("Building time-varying-transition Markov-regression forecasts...")
    tvtp = build_tvtp_forecasts(features, kodex_returns, labels, signal_months)
    print("Building horizon-specific LightGBM forecasts with Platt calibration...")
    cjm_lgbm, importance, lgbm_audit = build_lightgbm_forecasts(
        features, labels, cjm, signal_months
    )
    forecasts = {
        "SJM": sjm,
        "CJM": cjm,
        "TVTP-HMM": tvtp,
        "CJM+LightGBM": cjm_lgbm,
    }
    probability_metrics = probability_metric_rows(forecasts, labels)
    probability_long = _forecast_long(forecasts)

    print("Calibrating portfolio overlays through 2017 only...")
    calibration, winners = calibrate_overlays(baseline, asset_returns, forecasts)
    baseline_path = baseline.copy()
    if "turnover" not in baseline_path:
        baseline_path["turnover"] = 0.0
    comparison_rows = []
    periods = (
        ("calibration_2007_2017", baseline.index.min(), CAL_END),
        ("locked_2018_2026", LOCKED_START, baseline.index.max()),
        ("full_2007_2026", baseline.index.min(), baseline.index.max()),
    )
    for period_name, start, end in periods:
        comparison_rows.append(
            _metric_record(period_name, "Existing_VKOSPI_Dynamic", baseline_path.loc[start:end])
        )
        for model_name, (_, backtest) in winners.items():
            comparison_rows.append(
                _metric_record(period_name, model_name, backtest.loc[start:end])
            )
    comparison = pd.DataFrame(comparison_rows)

    validation: dict[str, object] = {}
    baseline_locked = baseline_path.loc[LOCKED_START:]
    baseline_locked_metrics = performance_summary(baseline_locked["return"])
    for model_name, (config, backtest) in winners.items():
        locked = backtest.loc[LOCKED_START:]
        metrics = performance_summary(locked["return"])
        deltas = {
            metric: float(metrics[metric] - baseline_locked_metrics[metric])
            for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
        }
        validation[model_name] = {
            "selected_config": asdict(config),
            "locked_metrics": metrics.to_dict(),
            "locked_deltas": deltas,
            "passes_all_three": bool(
                deltas["CAGR"] > 0 and deltas["Sharpe"] > 0 and deltas["MDD"] > 0
            ),
            "bootstrap": paired_multiobjective_bootstrap(
                baseline_locked["return"], locked["return"]
            ),
        }
        backtest.to_csv(
            RESULTS / f"top3_regime_model_backtest_{model_name.lower().replace('+', '_plus_').replace('-', '_')}.csv"
        )

    lgbm_audit_columns = [
        "signal_month",
        "horizon",
        "max_label_month",
        "fit_end_month",
        "train_rows",
        "feature_count",
        "calibration_method",
    ]
    # The matching per-fit records are saved to top3_regime_model_lgbm_audit.csv.
    report = {
        "objective": "Apply feedback-ranked CJM+LightGBM, TVTP-HMM, and CJM upgrades",
        "baseline": "Existing VKOSPI dynamic strategy (monthly reconciled authority)",
        "risk_off_definition": "KODEX200 monthly open-to-open return below zero",
        "timing": "features through signal month t; target return starts in month t+1",
        "horizons_months": list(HORIZONS),
        "calibration_end": str(CAL_END),
        "locked_start": str(LOCKED_START),
        "models": {
            "SJM": "existing sparse-jump probability with causal empirical transition propagation",
            "CJM": "official jumpmodels.JumpModel(cont=True), online probabilities, transition propagation",
            "TVTP-HMM": "statsmodels two-state Gaussian MarkovRegression with exog_tvtp",
            "CJM+LightGBM": "horizon-specific LightGBM plus recent holdout Platt calibration",
        },
        "feature_engineering": "Level + 1M/3M momentum + rolling z-score + acceleration for macro, credit, financial conditions, VIX/VKOSPI and market variables",
        "baseline_locked_metrics": baseline_locked_metrics.to_dict(),
        "validation": validation,
        "probability_metric_rows": int(len(probability_metrics)),
        "cjm_fit_success_rate": float(cjm["fit_success"].fillna(False).mean()),
        "tvtp_primary_fit_rate": float(
            (tvtp["fit_method"] == "statsmodels_tvtp_markov_regression").mean()
        ),
        "lgbm_causality_audit_fields": lgbm_audit_columns,
    }

    probability_long.to_csv(
        RESULTS / "top3_regime_model_probabilities.csv", index=False
    )
    probability_metrics.to_csv(
        RESULTS / "top3_regime_model_prediction_metrics.csv", index=False
    )
    calibration.to_csv(RESULTS / "top3_regime_model_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "top3_regime_model_comparison.csv", index=False)
    importance.to_csv(
        RESULTS / "top3_regime_model_feature_importance.csv", index=False
    )
    lgbm_audit.to_csv(RESULTS / "top3_regime_model_lgbm_audit.csv", index=False)
    cjm.to_csv(RESULTS / "top3_regime_model_cjm_diagnostics.csv")
    tvtp.to_csv(RESULTS / "top3_regime_model_tvtp_diagnostics.csv")
    (RESULTS / "top3_regime_model_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== PORTFOLIO COMPARISON ===")
    print(
        comparison[
            ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== PROBABILITY QUALITY ===")
    print(
        probability_metrics[
            ["Period", "Model", "Horizon", "Brier", "LogLoss", "ECE5", "TransitionRecall"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== LOCKED VALIDATION ===")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
