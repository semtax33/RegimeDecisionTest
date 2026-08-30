from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from strategies.core.regime_research import performance_summary
from strategies.stage07_regime_models.top3_regime_model_experiment import (
    CAL_END,
    LOCKED_START,
    RANDOM_STATE,
    build_master_features,
    expected_calibration_error,
    paired_multiobjective_bootstrap,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import load_vkospi_daily


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ReprocessOverlayConfig:
    threshold: float
    max_shift: float
    receiver: str

    @property
    def name(self) -> str:
        return f"th{self.threshold:.2f}_shift{self.max_shift:.2f}_{self.receiver.lower()}"


def _last_percentile(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.nan
    return float((np.sum(finite[:-1] <= finite[-1]) + 1) / len(finite))


def rolling_percentile(series: pd.Series, window: int, minimum: int) -> pd.Series:
    return series.rolling(window, min_periods=minimum).apply(_last_percentile, raw=True)


def robust_zscore(series: pd.Series, window: int, minimum: int) -> pd.Series:
    median = series.rolling(window, min_periods=minimum).median()
    mad = (series - median).abs().rolling(window, min_periods=minimum).median()
    return ((series - median) / (1.4826 * mad.replace(0, np.nan))).clip(-6, 6)


def _days_since_high(values: np.ndarray) -> float:
    finite = np.where(np.isfinite(values), values, -np.inf)
    if not np.isfinite(finite).any():
        return np.nan
    return float(len(values) - 1 - int(np.argmax(finite)))


def build_reprocessed_vkospi_features() -> pd.DataFrame:
    """Construct causal, month-end VKOSPI state features from raw daily data."""
    daily = load_vkospi_daily().copy()
    close = daily["close"].astype(float)
    log_close = np.log(close.where(close > 0))
    log_return = log_close.diff()
    frame = pd.DataFrame(index=daily.index)

    frame["vk_direct_close"] = close
    for window in (1, 5, 10, 21, 63):
        frame[f"vk_logret_{window}"] = log_close.diff(window)
    for window, minimum in ((63, 42), (126, 84), (252, 126), (756, 252)):
        frame[f"vk_pctile_{window}"] = rolling_percentile(close, window, minimum)
        frame[f"vk_robust_z_{window}"] = robust_zscore(log_close, window, minimum)

    for horizon in (5, 10, 21):
        scale = log_return.rolling(63, min_periods=42).std(ddof=1) * math.sqrt(horizon)
        frame[f"vk_shock_{horizon}"] = (log_close.diff(horizon) / scale.replace(0, np.nan)).clip(-8, 8)
    frame["vk_acceleration_5"] = log_close.diff(5) - log_close.diff(5).shift(5)
    frame["vk_acceleration_21"] = log_close.diff(21) - log_close.diff(21).shift(21)
    frame["vk_fast_slow"] = log_close.diff(5) - 5 / 21 * log_close.diff(21)

    positive = log_return.clip(lower=0)
    for window in (5, 21, 63):
        frame[f"vk_upside_semivol_{window}"] = np.sqrt(
            positive.pow(2).rolling(window, min_periods=max(3, window // 2)).mean()
        ) * math.sqrt(252)
        frame[f"vk_positive_fraction_{window}"] = (
            (log_return > 0).rolling(window, min_periods=max(3, window // 2)).mean()
        )

    high = daily["high"].astype(float).where(daily["high"].astype(float) > 0, close)
    low = daily["low"].astype(float).where(daily["low"].astype(float) > 0, close)
    range_log = np.log(high / low.replace(0, np.nan))
    frame["vk_intraday_range_5"] = range_log.rolling(5, min_periods=3).mean()
    frame["vk_intraday_range_21"] = range_log.rolling(21, min_periods=10).mean()
    high21 = high.rolling(21, min_periods=10).max()
    low21 = low.rolling(21, min_periods=10).min()
    high63 = high.rolling(63, min_periods=31).max()
    frame["vk_distance_high21"] = close / high21 - 1
    frame["vk_distance_high63"] = close / high63 - 1
    frame["vk_close_location21"] = (close - low21) / (high21 - low21).replace(0, np.nan)
    frame["vk_days_since_high63"] = close.rolling(63, min_periods=31).apply(
        _days_since_high, raw=True
    )

    level = frame["vk_pctile_252"].fillna(frame["vk_pctile_126"])
    rising = expit(frame["vk_shock_5"].fillna(0).clip(-6, 6))
    frame["vk_fear_rising"] = level * rising
    frame["vk_fear_falling"] = level * (1 - rising)
    frame["vk_panic_confirmation"] = (
        (level - 0.70).clip(lower=0) / 0.30
        * frame["vk_close_location21"].fillna(0.5)
        * rising
    ).clip(0, 1)
    frame["vk_panic_exhaustion"] = (
        (level - 0.70).clip(lower=0) / 0.30
        * (-frame["vk_distance_high21"]).clip(0, 0.5) / 0.5
        * (1 - rising)
    ).clip(0, 1)
    frame["vk_level_slope"] = frame["vk_robust_z_63"] - frame["vk_robust_z_252"]

    monthly_last = frame.resample("ME").last()
    monthly = pd.DataFrame(index=monthly_last.index)
    for column in frame.columns:
        monthly[column] = monthly_last[column]
    monthly["vk_month_mean"] = close.resample("ME").mean()
    monthly["vk_month_max"] = high.resample("ME").max()
    monthly["vk_month_min"] = low.resample("ME").min()
    monthly["vk_month_close_to_max"] = monthly["vk_direct_close"] / monthly["vk_month_max"] - 1
    monthly["vk_month_range"] = monthly["vk_month_max"] / monthly["vk_month_min"] - 1
    monthly["vk_month_mean_gap"] = monthly["vk_direct_close"] / monthly["vk_month_mean"] - 1
    monthly.index = monthly.index.to_period("M")
    return monthly.replace([np.inf, -np.inf], np.nan).sort_index()


def build_legacy_signal_month_features() -> pd.DataFrame:
    legacy = pd.read_csv(RESULTS / "vkospi_features.csv")
    legacy["signal_month"] = pd.to_datetime(legacy["vkospi_signal_date"]).dt.to_period("M")
    legacy = legacy.set_index("signal_month").sort_index()
    columns = [
        "vkospi_raw_close",
        "vkospi_return_5",
        "vkospi_return_21",
        "vkospi_return_63",
        "vkospi_level_z_63",
        "vkospi_level_z_252",
        "vkospi_return_vol_21",
        "vkospi_positive_fraction_21",
        "vkospi_oap_high52",
        "vkospi_oap_realized_vol_21",
    ]
    output = legacy[[column for column in columns if column in legacy]].copy()
    output.columns = [f"legacy_{column}" for column in output.columns]
    return output


BASE_FEATURES = (
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
    "macro_growth_GDP_level",
    "macro_growth_Export_level",
    "macro_growth_BSI_level",
    "macro_growth_GDP_level_d3",
    "macro_growth_Export_level_d3",
    "macro_growth_BSI_level_d3",
    "stress_VIX_last_z60",
    "stress_BAA_SPREAD_last_z60",
    "stress_NFCI_last_z60",
    "stress_STLFSI_last_z60",
)

ROBUST_COLUMNS = (
    "vk_direct_close",
    "vk_logret_5",
    "vk_logret_21",
    "vk_pctile_126",
    "vk_pctile_252",
    "vk_robust_z_63",
    "vk_robust_z_252",
    "vk_shock_5",
    "vk_shock_21",
    "vk_acceleration_5",
    "vk_fast_slow",
    "vk_distance_high21",
    "vk_close_location21",
    "vk_month_close_to_max",
    "vk_month_mean_gap",
)

INTERACTION_COLUMNS = ROBUST_COLUMNS + (
    "vk_logret_10",
    "vk_logret_63",
    "vk_pctile_63",
    "vk_pctile_756",
    "vk_robust_z_756",
    "vk_shock_10",
    "vk_acceleration_21",
    "vk_upside_semivol_5",
    "vk_upside_semivol_21",
    "vk_positive_fraction_5",
    "vk_positive_fraction_21",
    "vk_intraday_range_5",
    "vk_intraday_range_21",
    "vk_distance_high63",
    "vk_days_since_high63",
    "vk_fear_rising",
    "vk_fear_falling",
    "vk_panic_confirmation",
    "vk_panic_exhaustion",
    "vk_level_slope",
    "vk_month_range",
)


def build_feature_variants() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    master, _, asset_returns, baseline = build_master_features()
    base_columns = [column for column in BASE_FEATURES if column in master]
    base = master[base_columns]
    legacy = build_legacy_signal_month_features()
    reprocessed = build_reprocessed_vkospi_features()
    variants = {
        "LegacySignalMonth": base.join(legacy, how="outer"),
        "RobustMultiscale": base.join(
            reprocessed[[column for column in ROBUST_COLUMNS if column in reprocessed]],
            how="outer",
        ),
        "PanicInteraction": base.join(
            reprocessed[[column for column in INTERACTION_COLUMNS if column in reprocessed]],
            how="outer",
        ),
    }
    variants = {
        name: frame.replace([np.inf, -np.inf], np.nan).sort_index()
        for name, frame in variants.items()
    }
    return variants, asset_returns, baseline


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def walk_forward_defensive_model(
    features: pd.DataFrame,
    advantage: pd.Series,
    signal_months: pd.PeriodIndex,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_advantage = advantage.shift(-1)
    target = (target_advantage > 0).astype(float).where(target_advantage.notna())
    realization = pd.Series([month + 1 for month in advantage.index], index=advantage.index)
    rows = []
    audit_rows = []
    importance = pd.Series(0.0, index=features.columns)
    importance_fits = 0
    for number, signal_month in enumerate(signal_months):
        if signal_month not in features.index:
            continue
        train_index = features.index.intersection(target.dropna().index)
        train_index = train_index[realization.reindex(train_index) <= signal_month]
        if len(train_index) < 60:
            continue
        train_index = train_index[-144:]
        if realization.loc[train_index].max() > signal_month:
            raise AssertionError("Defensive model label crosses signal month")
        x_all = features.loc[train_index]
        y_all = target.loc[train_index].astype(int)
        advantage_all = target_advantage.loc[train_index].astype(float)
        usable = x_all.notna().mean() >= 0.60
        columns = list(x_all.columns[usable])
        if len(columns) < 12 or y_all.nunique() < 2:
            continue
        validation_size = min(24, max(15, len(x_all) // 5))
        split = len(x_all) - validation_size
        x_train = x_all.iloc[:split][columns]
        x_valid = x_all.iloc[split:][columns]
        y_train, y_valid = y_all.iloc[:split], y_all.iloc[split:]
        a_train, a_valid = advantage_all.iloc[:split], advantage_all.iloc[split:]
        medians = x_train.median()
        x_train = x_train.fillna(medians)
        x_valid = x_valid.fillna(medians)
        x_test = features.loc[[signal_month], columns].fillna(medians)

        classifier = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=320,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.80,
            colsample_bytree=0.72,
            reg_alpha=0.30,
            reg_lambda=0.60,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        )
        classifier.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        valid_raw = classifier.predict_proba(x_valid)[:, 1]
        raw_probability = float(classifier.predict_proba(x_test)[:, 1][0])
        probability = 0.75 * raw_probability + 0.25 * float(y_train.mean())
        calibration_method = "base_rate_shrinkage"
        if y_valid.nunique() == 2 and len(y_valid) >= 15:
            calibrator = LogisticRegression(
                C=0.50,
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
            calibrator.fit(_logit(valid_raw).reshape(-1, 1), y_valid)
            probability = float(
                calibrator.predict_proba(
                    _logit(np.array([raw_probability])).reshape(-1, 1)
                )[0, 1]
            )
            calibration_method = "platt_recent_validation"

        regressor = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=320,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.80,
            colsample_bytree=0.72,
            reg_alpha=0.30,
            reg_lambda=0.60,
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
            verbosity=-1,
        )
        regressor.fit(
            x_train,
            a_train,
            eval_set=[(x_valid, a_valid)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        expected_advantage = float(regressor.predict(x_test)[0])
        scale = max(float(a_train.std(ddof=1)), 0.01)
        regression_probability = float(expit(expected_advantage / scale))
        combined_probability = float(
            np.clip(0.70 * probability + 0.30 * regression_probability, 0.01, 0.99)
        )
        gain = pd.Series(
            classifier.booster_.feature_importance(importance_type="gain"),
            index=columns,
        )
        if gain.sum() > 0:
            importance.loc[columns] += gain / gain.sum()
            importance_fits += 1
        rows.append(
            {
                "signal_month": signal_month,
                "p_defensive": combined_probability,
                "p_classifier": float(np.clip(probability, 0.01, 0.99)),
                "p_regression": regression_probability,
                "expected_advantage": expected_advantage,
                "realized_advantage_next": float(target_advantage.loc[signal_month])
                if signal_month in target_advantage.index and pd.notna(target_advantage.loc[signal_month])
                else np.nan,
                "realized_defensive_next": float(target.loc[signal_month])
                if signal_month in target.index and pd.notna(target.loc[signal_month])
                else np.nan,
                "train_rows": len(x_all),
                "feature_count": len(columns),
                "calibration_method": calibration_method,
            }
        )
        audit_rows.append(
            {
                "Model": name,
                "signal_month": signal_month,
                "fit_end_month": train_index[-1],
                "max_label_month": realization.loc[train_index].max(),
                "train_rows": len(x_all),
                "feature_count": len(columns),
            }
        )
        if (number + 1) % 36 == 0:
            print(f"{name}: {number + 1}/{len(signal_months)}")
    forecasts = pd.DataFrame(rows).set_index("signal_month")
    forecasts.index = pd.PeriodIndex(forecasts.index, freq="M")
    audit = pd.DataFrame(audit_rows)
    importance_frame = (
        (importance / max(importance_fits, 1))
        .sort_values(ascending=False)
        .rename("mean_gain_share")
        .to_frame()
        .reset_index(names="feature")
    )
    importance_frame["Model"] = name
    return forecasts.sort_index(), audit, importance_frame


def run_overlay(
    baseline: pd.DataFrame,
    asset_returns: pd.DataFrame,
    forecast: pd.DataFrame,
    config: ReprocessOverlayConfig,
) -> pd.DataFrame:
    months = baseline.index.intersection(asset_returns.index)
    rows = []
    nav = 1.0
    peak = 1.0
    previous_shift = 0.0
    for month in months:
        signal_month = month - 1
        probability = (
            float(forecast.loc[signal_month, "p_defensive"])
            if signal_month in forecast.index
            else 0.5
        )
        activation = float(
            np.clip(
                (probability - config.threshold) / max(1 - config.threshold, 1e-6),
                0,
                1,
            )
        )
        shift = config.max_shift * activation
        defensive_return = float(asset_returns.loc[month, config.receiver])
        base_return = float(baseline.loc[month, "return"])
        incremental_turnover = 2 * abs(shift - previous_shift)
        trade_cost = incremental_turnover * 0.0015
        fx_cost = abs(shift - previous_shift) * 0.0005 if config.receiver == "GLD" else 0.0
        net_return = (
            (1 - shift) * base_return
            + shift * defensive_return
            - trade_cost
            - fx_cost
        )
        nav *= 1 + net_return
        peak = max(peak, nav)
        rows.append(
            {
                "month": month,
                "return": net_return,
                "baseline_return": base_return,
                "defensive_return": defensive_return,
                "p_defensive": probability,
                "activation": activation,
                "defensive_shift": shift,
                "turnover": float(baseline.loc[month].get("turnover", 0.0))
                + incremental_turnover,
                "incremental_turnover": incremental_turnover,
                "incremental_trade_cost": trade_cost,
                "incremental_fx_cost": fx_cost,
                "nav": nav,
                "drawdown": nav / peak - 1,
            }
        )
        previous_shift = shift
    return pd.DataFrame(rows).set_index("month")


def _metric_record(period: str, strategy: str, frame: pd.DataFrame) -> dict[str, object]:
    metrics = performance_summary(frame["return"])
    shift = frame.get("defensive_shift", pd.Series(0.0, index=frame.index))
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(frame["turnover"].mean()),
        "AvgDefensiveShift": float(shift.mean()),
        "ActiveMonths": int((shift > 1e-6).sum()),
    }


def probability_metrics(
    forecasts: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for (variant, receiver), forecast in forecasts.items():
        for period, view in (
            ("full_oos", forecast),
            (
                "locked_2018_2026",
                forecast.loc[pd.Period("2017-12", "M") :],
            ),
        ):
            sample = view.dropna(subset=["p_defensive", "realized_defensive_next"])
            if len(sample) < 12:
                continue
            y = sample["realized_defensive_next"].astype(int)
            p = sample["p_defensive"].clip(1e-5, 1 - 1e-5)
            rows.append(
                {
                    "Period": period,
                    "Variant": variant,
                    "Receiver": receiver,
                    "Observations": len(sample),
                    "Brier": float(brier_score_loss(y, p)),
                    "LogLoss": float(log_loss(y, p, labels=[0, 1])),
                    "ECE5": expected_calibration_error(y, p, bins=5),
                    "AUC": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
                    "MeanProbability": float(p.mean()),
                    "DefensiveWinRate": float(y.mean()),
                    "MeanExpectedAdvantage": float(sample["expected_advantage"].mean()),
                    "MeanRealizedAdvantage": float(sample["realized_advantage_next"].mean()),
                }
            )
    return pd.DataFrame(rows)


def calibrate_variants(
    baseline: pd.DataFrame,
    asset_returns: pd.DataFrame,
    forecasts: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, tuple[ReprocessOverlayConfig, pd.DataFrame]]]:
    base_metrics = performance_summary(baseline.loc[:CAL_END, "return"])
    rows = []
    paths = {}
    for (variant, receiver), forecast in forecasts.items():
        for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
            for max_shift in (0.10, 0.20, 0.30, 0.40):
                config = ReprocessOverlayConfig(threshold, max_shift, receiver)
                backtest = run_overlay(baseline, asset_returns, forecast, config)
                paths[(variant, config.name)] = backtest
                sample = backtest.loc[:CAL_END]
                metrics = performance_summary(sample["return"])
                rows.append(
                    {
                        "Variant": variant,
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
    winners = {}
    for variant, group in calibration.groupby("Variant", sort=False):
        ranked = group.copy()
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
            ranked[f"Rank_{metric}"] = ranked[metric].rank(pct=True)
        ranked["MultiObjectiveScore"] = ranked[
            ["Rank_CAGR", "Rank_Sharpe", "Rank_MDD", "Rank_Calmar"]
        ].mean(axis=1)
        eligible = ranked.loc[
            (ranked["CAGR"] >= 0.98 * base_metrics["CAGR"])
            & (ranked["MDD"] >= base_metrics["MDD"] - 0.005)
            & (ranked["ActiveMonths"] >= 5)
        ]
        if eligible.empty:
            eligible = ranked.loc[ranked["ActiveMonths"] >= 3]
        winner = eligible.sort_values(
            ["MultiObjectiveScore", "Sharpe", "Calmar", "CAGR"],
            ascending=False,
        ).iloc[0]
        config = ReprocessOverlayConfig(
            threshold=float(winner["threshold"]),
            max_shift=float(winner["max_shift"]),
            receiver=str(winner["receiver"]),
        )
        winners[variant] = (config, paths[(variant, str(winner["Config"]))])
        calibration.loc[group.index, "MultiObjectiveScore"] = ranked[
            "MultiObjectiveScore"
        ]
        calibration.loc[group.index, "SelectedWithinVariant"] = False
        calibration.loc[winner.name, "SelectedWithinVariant"] = True

    winner_rows = calibration.loc[calibration["SelectedWithinVariant"].fillna(False)].copy()
    for metric in ("CAGR", "Sharpe", "MDD", "Calmar"):
        winner_rows[f"AcrossRank_{metric}"] = winner_rows[metric].rank(pct=True)
    winner_rows["AcrossScore"] = winner_rows[
        ["AcrossRank_CAGR", "AcrossRank_Sharpe", "AcrossRank_MDD", "AcrossRank_Calmar"]
    ].mean(axis=1)
    overall = winner_rows.sort_values(
        ["AcrossScore", "Sharpe", "Calmar", "CAGR"], ascending=False
    ).iloc[0]
    calibration["SelectedOverall"] = False
    calibration.loc[overall.name, "SelectedOverall"] = True
    return calibration, winners


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    variants, asset_returns, baseline = build_feature_variants()
    baseline.index = pd.PeriodIndex(baseline.index, freq="M")
    signal_months = pd.period_range(
        pd.Period("2007-01", "M"), baseline.index.max() - 1, freq="M"
    )
    forecasts = {}
    audits = []
    importance = []
    print("Building defensive-policy models for three VKOSPI processing variants...")
    for variant, features in variants.items():
        for receiver in ("BOND", "GLD"):
            advantage = asset_returns[receiver].reindex(baseline.index) - baseline["return"]
            name = f"{variant}_{receiver}"
            forecast, audit, feature_importance = walk_forward_defensive_model(
                features, advantage, signal_months, name
            )
            forecasts[(variant, receiver)] = forecast
            audits.append(audit)
            importance.append(feature_importance)
            forecast.to_csv(
                RESULTS / f"vkospi_reprocess_forecast_{variant.lower()}_{receiver.lower()}.csv"
            )

    metrics = probability_metrics(forecasts)
    print("Selecting processing variant and overlay through 2017 only...")
    calibration, winners = calibrate_variants(
        baseline, asset_returns, forecasts
    )
    overall_row = calibration.loc[calibration["SelectedOverall"].fillna(False)].iloc[0]
    selected_variant = str(overall_row["Variant"])
    selected_config, selected_backtest = winners[selected_variant]

    previous = pd.read_csv(
        RESULTS / "top3_regime_model_backtest_cjm_plus_lightgbm.csv",
        index_col="month",
    )
    previous.index = pd.PeriodIndex(previous.index, freq="M")
    baseline_path = baseline.copy()
    comparison_rows = []
    periods = (
        ("calibration_2007_2017", baseline.index.min(), CAL_END),
        ("locked_2018_2026", LOCKED_START, baseline.index.max()),
        ("full_2007_2026", baseline.index.min(), baseline.index.max()),
    )
    for period, start, end in periods:
        comparison_rows.append(
            _metric_record(period, "Existing_VKOSPI_Dynamic", baseline_path.loc[start:end])
        )
        comparison_rows.append(
            _metric_record(period, "Previous_CJM_LightGBM", previous.loc[start:end])
        )
        for variant, (_, backtest) in winners.items():
            comparison_rows.append(
                _metric_record(period, variant, backtest.loc[start:end])
            )
    comparison = pd.DataFrame(comparison_rows)

    baseline_locked = baseline_path.loc[LOCKED_START:]
    selected_locked = selected_backtest.loc[LOCKED_START:]
    baseline_metrics = performance_summary(baseline_locked["return"])
    selected_metrics = performance_summary(selected_locked["return"])
    locked_deltas = {
        metric: float(selected_metrics[metric] - baseline_metrics[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    report = {
        "objective": "Reprocess VKOSPI to reduce false defensive activations",
        "feature_variants": list(variants),
        "processing": {
            "LegacySignalMonth": "legacy features realigned by actual signal date",
            "RobustMultiscale": "causal percentiles, rolling MAD z-scores, volatility-normalized shocks and month-end state",
            "PanicInteraction": "robust multiscale plus acceleration, high-distance, close-location, panic confirmation/exhaustion interactions",
        },
        "policy_target": "whether next-month BOND or GLD return exceeds the existing VKOSPI dynamic strategy return",
        "calibration_end": str(CAL_END),
        "locked_start": str(LOCKED_START),
        "selected_variant": selected_variant,
        "selected_config": asdict(selected_config),
        "baseline_locked_metrics": baseline_metrics.to_dict(),
        "selected_locked_metrics": selected_metrics.to_dict(),
        "locked_deltas": locked_deltas,
        "passes_all_three": bool(
            locked_deltas["CAGR"] > 0
            and locked_deltas["Sharpe"] > 0
            and locked_deltas["MDD"] > 0
        ),
        "bootstrap": paired_multiobjective_bootstrap(
            baseline_locked["return"], selected_locked["return"]
        ),
    }

    reprocessed = build_reprocessed_vkospi_features()
    reprocessed.to_csv(RESULTS / "vkospi_reprocessed_features.csv")
    pd.concat(audits, ignore_index=True).to_csv(
        RESULTS / "vkospi_reprocess_audit.csv", index=False
    )
    pd.concat(importance, ignore_index=True).to_csv(
        RESULTS / "vkospi_reprocess_feature_importance.csv", index=False
    )
    metrics.to_csv(RESULTS / "vkospi_reprocess_prediction_metrics.csv", index=False)
    calibration.to_csv(RESULTS / "vkospi_reprocess_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "vkospi_reprocess_comparison.csv", index=False)
    selected_backtest.to_csv(RESULTS / "vkospi_reprocess_selected_backtest.csv")
    for variant, (_, backtest) in winners.items():
        backtest.to_csv(
            RESULTS / f"vkospi_reprocess_backtest_{variant.lower()}.csv"
        )
    (RESULTS / "vkospi_reprocess_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SELECTED ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n=== COMPARISON ===")
    print(
        comparison[
            ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== PROBABILITY METRICS ===")
    print(
        metrics[
            ["Period", "Variant", "Receiver", "Brier", "LogLoss", "ECE5", "AUC"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
