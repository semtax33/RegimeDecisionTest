from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import (
    CAL_END,
    FACTOR_TEST_START,
    TEST_START,
    FactorBlendConfig,
    build_features,
    causal_hmm_features,
    download_ohlcv,
    feature_columns_for_variant,
    market_frames,
    paired_block_bootstrap,
    run_factor_blend,
    splice_kospi200_proxy,
)
from strategies.core.regime_research import (
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
RISK_THRESHOLD_SIGMA = 1.25
RISK_PERCENTILE_TRIGGER = 0.80
REFIT_EVERY_MONTHS = 3
TRAIN_DAYS = 1260


def tail_labels(close: pd.Series, horizon: int) -> pd.DataFrame:
    returns = close.pct_change()
    current_vol = returns.rolling(20, min_periods=15).std(ddof=1)
    paths = pd.concat(
        [close.shift(-day) / close - 1 for day in range(1, horizon + 1)],
        axis=1,
    )
    adverse_excursion = paths.min(axis=1)
    full_horizon = close.shift(-horizon).notna()
    standardized_adverse = -adverse_excursion / (current_vol * math.sqrt(horizon)).replace(0, np.nan)
    event = (standardized_adverse >= RISK_THRESHOLD_SIGMA).astype(float)
    event = event.where(full_horizon & standardized_adverse.notna())
    label_date = pd.Series(close.index, index=close.index).shift(-horizon)
    return pd.DataFrame(
        {
            "tail_event": event,
            "label_date": label_date,
            "adverse_excursion": adverse_excursion.where(full_horizon),
            "standardized_adverse": standardized_adverse.where(full_horizon),
        },
        index=close.index,
    )


def causal_percentile(values: pd.Series, lookback: int = 60) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    history: list[float] = []
    for index, value in values.items():
        if not np.isfinite(value):
            continue
        reference = np.asarray(history[-lookback:], dtype=float)
        if len(reference) >= 12:
            result.loc[index] = (float(np.sum(reference <= value)) + 1.0) / (len(reference) + 1.0)
        else:
            result.loc[index] = 0.5
        history.append(float(value))
    return result


def walk_forward_tail_model(
    features: pd.DataFrame,
    close: pd.Series,
    target_months: pd.PeriodIndex,
    horizon: int,
) -> pd.DataFrame:
    columns = feature_columns_for_variant(features, "kr_only")
    x_source = features[columns].copy()
    labels = tail_labels(close, horizon)
    rows: list[dict[str, object]] = []
    model: lgb.LGBMClassifier | None = None
    fitted_columns: list[str] = []
    medians = pd.Series(dtype=float)
    fit_count = 0

    for number, target_month in enumerate(target_months):
        signal_end = (target_month - 1).to_timestamp("M")
        candidates = x_source.index[x_source.index <= signal_end]
        if candidates.empty:
            continue
        signal_date = candidates[-1]
        should_refit = model is None or number % REFIT_EVERY_MONTHS == 0
        if should_refit:
            known = (labels["label_date"] <= signal_date) & labels["tail_event"].notna()
            train_index = x_source.index[known.fillna(False)]
            if len(train_index) < 504:
                continue
            train_index = train_index[-TRAIN_DAYS:]
            x_train_all = x_source.loc[train_index]
            y_train_all = labels.loc[train_index, "tail_event"].astype(int)
            usable = x_train_all.notna().mean() >= 0.70
            fitted_columns = list(x_train_all.columns[usable])
            if len(fitted_columns) < 20 or y_train_all.nunique() < 2:
                continue
            validation_size = min(252, max(126, len(train_index) // 5))
            split = len(train_index) - validation_size
            x_train = x_train_all.iloc[:split][fitted_columns]
            y_train = y_train_all.iloc[:split]
            x_valid = x_train_all.iloc[split:][fitted_columns]
            y_valid = y_train_all.iloc[split:]
            medians = x_train.median()
            positive = max(int(y_train.sum()), 1)
            negative = max(int((1 - y_train).sum()), 1)
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                subsample=0.80,
                colsample_bytree=0.70,
                reg_alpha=0.10,
                reg_lambda=0.20,
                min_child_samples=50,
                scale_pos_weight=negative / positive,
                random_state=20260826 + horizon,
                n_jobs=1,
                verbosity=-1,
            )
            fit_kwargs: dict[str, object] = {}
            if y_valid.nunique() > 1:
                fit_kwargs = {
                    "eval_set": [(x_valid.fillna(medians), y_valid)],
                    "eval_metric": "auc",
                    "callbacks": [lgb.early_stopping(40, verbose=False)],
                }
            model.fit(x_train.fillna(medians), y_train, **fit_kwargs)
            fit_count += 1

        if model is None:
            continue
        test = x_source.loc[[signal_date], fitted_columns].fillna(medians)
        probability = float(model.predict_proba(test)[:, 1][0])
        rows.append(
            {
                "target_month": target_month,
                "signal_date": signal_date,
                "p_tail_raw": probability,
                "tail_event": labels.loc[signal_date, "tail_event"],
                "adverse_excursion": labels.loc[signal_date, "adverse_excursion"],
                "standardized_adverse": labels.loc[signal_date, "standardized_adverse"],
                "horizon": horizon,
                "fit_count": fit_count,
                "train_rows": len(train_index),
                "feature_count": len(fitted_columns),
            }
        )
        if (number + 1) % 36 == 0:
            print(f"Tail-risk {horizon}D predictions: {number + 1}/{len(target_months)}")

    out = pd.DataFrame(rows).set_index("target_month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    out["risk_percentile"] = causal_percentile(out["p_tail_raw"])
    severity = ((out["risk_percentile"] - RISK_PERCENTILE_TRIGGER) / (1 - RISK_PERCENTILE_TRIGGER)).clip(0, 1)
    out["risk_severity"] = severity
    # run_factor_blend maps a 0.15 probability deviation to a full score.
    # Keeping p_up at or below 0.5 makes the overlay de-risk only.
    out["p_up"] = 0.5 - 0.15 * severity
    out["score"] = -severity
    return out


def ensemble_tail_models(factors: list[pd.DataFrame]) -> pd.DataFrame:
    common = factors[0].index
    for factor in factors[1:]:
        common = common.intersection(factor.index)
    out = pd.DataFrame(index=common)
    out["p_tail_raw"] = np.mean(
        np.column_stack([factor.loc[common, "p_tail_raw"] for factor in factors]), axis=1
    )
    out["risk_percentile"] = np.mean(
        np.column_stack([factor.loc[common, "risk_percentile"] for factor in factors]), axis=1
    )
    severity = ((out["risk_percentile"] - RISK_PERCENTILE_TRIGGER) / (1 - RISK_PERCENTILE_TRIGGER)).clip(0, 1)
    out["risk_severity"] = severity
    out["p_up"] = 0.5 - 0.15 * severity
    out["score"] = -severity
    # The ensemble is evaluated against either horizon event occurring.
    out["tail_event"] = np.max(
        np.column_stack([factor.loc[common, "tail_event"] for factor in factors]), axis=1
    )
    out["adverse_excursion"] = np.min(
        np.column_stack([factor.loc[common, "adverse_excursion"] for factor in factors]), axis=1
    )
    out["horizon"] = "10D_20D_ensemble"
    return out


def prediction_metrics(factor: pd.DataFrame, start: pd.Period | None = None) -> dict[str, float]:
    view = factor.loc[start:] if start is not None else factor
    view = view.dropna(subset=["p_tail_raw", "tail_event"])
    if view.empty:
        return {}
    y = view["tail_event"].astype(int)
    p = view["p_tail_raw"].clip(1e-6, 1 - 1e-6)
    cutoff = float(p.quantile(0.80))
    high_risk = p >= cutoff
    return {
        "observations": int(len(view)),
        "event_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "recall_at_top_20pct": float(y[high_risk].sum() / max(y.sum(), 1)),
        "precision_at_top_20pct": float(y[high_risk].mean()),
    }


def metric_record(period: str, strategy: str, backtest: pd.DataFrame) -> dict[str, float | str]:
    metrics = performance_summary(backtest["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(backtest["turnover"].mean()),
    }


def main(rebuild: bool = False) -> None:
    raw = download_ohlcv(refresh=False)
    frames = market_frames(raw)
    frames["KODEX200"] = splice_kospi200_proxy(frames["KODEX200"])
    kr_hmm_path = CACHE / "kr_hmm_diag756.csv"
    us_hmm_path = CACHE / "us_hmm_diag756.csv"
    if kr_hmm_path.exists() and us_hmm_path.exists():
        kr_hmm = pd.read_csv(kr_hmm_path, index_col=0, parse_dates=True)
        us_hmm = pd.read_csv(us_hmm_path, index_col=0, parse_dates=True)
    else:
        kr_hmm = causal_hmm_features(frames, market="kr")
        us_hmm = causal_hmm_features(frames, market="us")
        kr_hmm.to_csv(kr_hmm_path)
        us_hmm.to_csv(us_hmm_path)
    features, close = build_features(frames, kr_hmm, us_hmm)
    target_months = pd.period_range(FACTOR_TEST_START, close.index[-1].to_period("M"), freq="M")

    factors: dict[str, pd.DataFrame] = {}
    for horizon in (10, 20):
        path = RESULTS / f"short_tail_risk_{horizon}d_factor.csv"
        if path.exists() and not rebuild:
            factor = pd.read_csv(path)
            factor["target_month"] = pd.PeriodIndex(factor["target_month"], freq="M")
            factor = factor.set_index("target_month")
        else:
            factor = walk_forward_tail_model(features, close, target_months, horizon)
            factor.to_csv(path)
        factors[f"TailRisk{horizon}D"] = factor
    factors["TailRiskEnsemble"] = ensemble_tail_models(list(factors.values()))
    factors["TailRiskEnsemble"].to_csv(RESULTS / "short_tail_risk_ensemble_factor.csv")

    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")
    neutral_factor = next(iter(factors.values()))
    baseline = run_factor_blend(
        asset_returns,
        signals,
        defensive,
        neutral_factor,
        FactorBlendConfig(max_shift=0.0),
    )
    calibration_start = baseline.index.min()
    baseline_calibration = performance_summary(baseline.loc[calibration_start:CAL_END, "return"])
    shift_grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    calibration_rows: list[dict[str, object]] = []
    backtests: dict[tuple[str, float], pd.DataFrame] = {}
    winners: dict[str, tuple[float, pd.DataFrame]] = {}

    for name, factor in factors.items():
        for shift in shift_grid:
            backtest = run_factor_blend(
                asset_returns,
                signals,
                defensive,
                factor,
                FactorBlendConfig(max_shift=shift),
            )
            backtests[(name, shift)] = backtest
            metrics = performance_summary(backtest.loc[calibration_start:CAL_END, "return"])
            calibration_rows.append(
                {
                    "Strategy": name,
                    "max_shift": shift,
                    **metrics.to_dict(),
                    "AvgTurnover": float(backtest.loc[calibration_start:CAL_END, "turnover"].mean()),
                }
            )
        table = pd.DataFrame([row for row in calibration_rows if row["Strategy"] == name])
        eligible = table[
            (table["MDD"] >= -0.15)
            & (table["CAGR"] >= 0.95 * float(baseline_calibration["CAGR"]))
        ]
        selection_pool = eligible if not eligible.empty else table
        winner = selection_pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
        shift = float(winner["max_shift"])
        winners[name] = (shift, backtests[(name, shift)])

    comparison_rows: list[dict[str, object]] = []
    periods = (
        ("calibration_2007_2017", calibration_start, CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", calibration_start, None),
    )
    for label, start, end in periods:
        strategies = {"FinalBlend": baseline, **{name: value[1] for name, value in winners.items()}}
        for name, backtest in strategies.items():
            view = backtest.loc[start:end] if end is not None else backtest.loc[start:]
            comparison_rows.append(metric_record(label, name, view))
    comparison = pd.DataFrame(comparison_rows)

    baseline_locked = baseline.loc[TEST_START:]
    baseline_metrics = performance_summary(baseline_locked["return"])
    validation: dict[str, object] = {}
    for name, (shift, backtest) in winners.items():
        locked = backtest.loc[TEST_START:]
        metrics = performance_summary(locked["return"])
        validation[name] = {
            "selected_shift": shift,
            "locked_deltas": {
                key: float(metrics[key] - baseline_metrics[key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar")
            },
            "prediction_full": prediction_metrics(factors[name]),
            "prediction_locked": prediction_metrics(factors[name], TEST_START),
            "bootstrap": paired_block_bootstrap(baseline_locked["return"], locked["return"]),
        }
        backtest.to_csv(RESULTS / f"short_tail_risk_{name.lower()}_backtest.csv")

    report = {
        "method": {
            "target": "volatility-scaled maximum adverse excursion over the next 10/20 trading days",
            "event_threshold_sigma": RISK_THRESHOLD_SIGMA,
            "feature_set": "Korean-only technical + KOSPI200 + USDKRW + Korean HMM",
            "model": "quarterly-refit five-year rolling LightGBM",
            "overlay": "asymmetric de-risk only above causal trailing 80th risk percentile",
            "factor_start": str(FACTOR_TEST_START),
            "calibration_end": str(CAL_END),
            "locked_start": str(TEST_START),
        },
        "validation": validation,
    }
    pd.DataFrame(calibration_rows).to_csv(RESULTS / "short_tail_risk_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "short_tail_risk_comparison.csv", index=False)
    (RESULTS / "short_tail_risk_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== WINNERS ===")
    print(pd.DataFrame({name: {"max_shift": value[0]} for name, value in winners.items()}).T)
    print("\n=== COMPARISON ===")
    print(
        comparison[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== VALIDATION ===")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
