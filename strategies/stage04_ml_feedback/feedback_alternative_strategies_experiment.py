from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler

from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import (
    CAL_END,
    FACTOR_TEST_START,
    TEST_START,
    FactorBlendConfig,
    apply_factor_tilt,
    build_features,
    download_ohlcv,
    market_frames,
    paired_block_bootstrap,
    run_factor_blend,
    splice_kospi200_proxy,
)
from strategies.core.regime_research import (
    ASSETS,
    SparseJump2,
    StrategyConfig,
    compute_regime_signals,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
HORIZON = 10


def monthly_factor_from_daily(
    daily_score: pd.Series,
    target_months: pd.PeriodIndex,
    name: str,
) -> pd.DataFrame:
    daily_score = daily_score.dropna().sort_index()
    rows = []
    for target_month in target_months:
        signal_end = (target_month - 1).to_timestamp("M")
        available = daily_score.index[daily_score.index <= signal_end]
        if available.empty:
            continue
        signal_date = available[-1]
        score = float(np.clip(daily_score.loc[signal_date], -1, 1))
        rows.append(
            {
                "target_month": target_month,
                "signal_date": signal_date,
                "factor_name": name,
                "score": score,
                "p_up": (score + 1) / 2,
            }
        )
    out = pd.DataFrame(rows).set_index("target_month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out


def attach_realized(factor: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    out = factor.copy()
    forward_return = close.shift(-HORIZON) / close - 1
    out["realized_10d_return"] = [
        float(forward_return.loc[date]) if date in forward_return.index and pd.notna(forward_return.loc[date]) else np.nan
        for date in pd.to_datetime(out["signal_date"])
    ]
    out["realized_up"] = (out["realized_10d_return"] > 0).astype(float).where(out["realized_10d_return"].notna())
    return out


def factor_prediction_metrics(factor: pd.DataFrame, start: pd.Period) -> dict[str, float]:
    view = factor.loc[start:].dropna(subset=["p_up", "realized_up"])
    if view.empty:
        return {}
    y = view["realized_up"].astype(int)
    probability = view["p_up"].clip(1e-6, 1 - 1e-6)
    return {
        "observations": int(len(view)),
        "auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier_score": float(brier_score_loss(y, probability)),
        "mean_probability": float(probability.mean()),
        "positive_rate": float(y.mean()),
    }


def build_jump_factor(close: pd.Series, target_months: pd.PeriodIndex) -> pd.DataFrame:
    returns = close.pct_change()
    downside = returns.clip(upper=0)
    risk = pd.DataFrame(index=close.index)
    risk["downside_deviation_10"] = np.sqrt(downside.pow(2).rolling(10).mean()) * math.sqrt(252)
    for window in [20, 60]:
        downside_risk = np.sqrt(downside.pow(2).rolling(window).mean()) * math.sqrt(252)
        sortino = returns.rolling(window).mean() * 252 / downside_risk.replace(0, np.nan)
        risk[f"negative_sortino_{window}"] = -sortino
    risk = risk.replace([np.inf, -np.inf], np.nan).dropna()

    model = SparseJump2(jump_penalty=3.0, keep_features=3, max_iter=30)
    rows = []
    for number, target_month in enumerate(target_months):
        signal_end = (target_month - 1).to_timestamp("M")
        history = risk.loc[:signal_end].tail(756)
        if len(history) < 252:
            continue
        p_risk, detail = model.fit_predict_high(history)
        score = float(np.clip(1 - 2 * p_risk, -1, 1))
        rows.append(
            {
                "target_month": target_month,
                "signal_date": history.index[-1],
                "factor_name": "jump_downside_sortino",
                "p_up": (score + 1) / 2,
                "score": score,
                "p_risk": p_risk,
                "switches_in_window": detail["switches"],
            }
        )
        if (number + 1) % 60 == 0:
            print(f"Jump factor: {number + 1}/{len(target_months)}")
    out = pd.DataFrame(rows).set_index("target_month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out


def duration_aware_score(hmm: pd.DataFrame, minimum_duration: int = 20) -> pd.Series:
    probability_columns = ["kr_hmm_p_bear", "kr_hmm_p_sideways", "kr_hmm_p_bull"]
    probabilities = hmm[probability_columns].dropna().astype(float)
    if probabilities.empty:
        return pd.Series(dtype=float)
    current = int(np.argmax(probabilities.iloc[0].to_numpy()))
    duration = 1
    scores = []
    for date, row in probabilities.iterrows():
        values = row.to_numpy(dtype=float)
        candidate = int(np.argmax(values))
        if (
            candidate != current
            and duration >= minimum_duration
            and values[candidate] >= max(0.50, values[current] + 0.10)
        ):
            current = candidate
            duration = 1
        else:
            duration += 1
        discrete = [-1.0, 0.0, 1.0][current]
        posterior_score = float(values @ np.array([-1.0, 0.0, 1.0]))
        scores.append((date, 0.70 * discrete + 0.30 * posterior_score))
    return pd.Series(dict(scores)).sort_index().clip(-1, 1)


def engineered_technical_score(features: pd.DataFrame) -> pd.Series:
    trend = np.tanh(features["dist_sma_200"] / 0.10)
    short_reversal = -np.tanh(features["ret_5"] / 0.05)
    rsi_reversal = np.clip(-(features["rsi_14"] - 0.5) * 2, -1, 1)
    vol = features["realized_vol_20"]
    vol_median = vol.rolling(252, min_periods=126).median()
    vol_iqr = (vol.rolling(252, min_periods=126).quantile(0.75) - vol.rolling(252, min_periods=126).quantile(0.25)).replace(0, np.nan)
    low_vol = -np.tanh((vol - vol_median) / vol_iqr)
    score = 0.35 * trend + 0.25 * short_reversal + 0.25 * rsi_reversal + 0.15 * low_vol
    return score.clip(-1, 1)


def walk_forward_technical_ensemble(
    features: pd.DataFrame,
    close: pd.Series,
    target_months: pd.PeriodIndex,
) -> pd.DataFrame:
    domestic_prefixes = ("krw_", "kospi200_", "corr_krw_", "kr_hmm_")
    global_prefixes = (
        "corr_spy_",
        "beta_spy_",
        "relative_spy_",
        "spy_ret_",
        "sox_",
        "vix_",
        "vix3m_",
        "btc_",
        "corr_btc_",
        "credit_",
        "yield_",
        "gold_",
        "dollar_",
        "oil_",
        "us_hmm_",
    )
    columns = [
        col for col in features
        if not col.startswith(domestic_prefixes) and not col.startswith(global_prefixes)
    ]
    x_source = features[columns]
    label_date = pd.Series(x_source.index, index=x_source.index).shift(-HORIZON)
    forward_return = close.shift(-HORIZON) / close - 1
    target = (forward_return > 0).astype(float).where(forward_return.notna())
    rows = []
    # Quarterly refits keep the experiment causal while avoiding a needlessly
    # expensive monthly refit.  The most recent fitted model is held between
    # refits, which also matches a practical production retraining schedule.
    refit_months = target_months[::3]

    for number, target_month in enumerate(refit_months):
        signal_end = (target_month - 1).to_timestamp("M")
        candidates = x_source.index[x_source.index <= signal_end]
        if candidates.empty:
            continue
        signal_date = candidates[-1]
        known = (label_date <= signal_date) & target.notna()
        train_index = x_source.index[known.fillna(False)]
        if len(train_index) < 504:
            continue
        # A rolling five-year sample is more appropriate for a non-stationary
        # technical model and materially reduces fit time on a busy workstation.
        train_index = train_index[-1260:]
        x_train = x_source.loc[train_index]
        y_train = target.loc[train_index].astype(int)
        usable = x_train.notna().mean() >= 0.70
        selected = list(x_train.columns[usable])
        if len(selected) < 20:
            continue
        medians = x_train[selected].median()
        scaler = RobustScaler().fit(x_train[selected].fillna(medians))
        train_scaled = scaler.transform(x_train[selected].fillna(medians))
        test_scaled = scaler.transform(x_source.loc[[signal_date], selected].fillna(medians))

        models = [
            ExtraTreesClassifier(
                n_estimators=40,
                max_depth=8,
                min_samples_leaf=15,
                max_features=0.70,
                class_weight="balanced",
                random_state=20260825,
                n_jobs=1,
            ),
            HistGradientBoostingClassifier(
                max_iter=25,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=0.10,
                random_state=20260825,
            ),
            LogisticRegression(
                C=0.10,
                class_weight="balanced",
                max_iter=500,
                random_state=20260825,
            ),
        ]
        component_probabilities = []
        for model in models:
            model.fit(train_scaled, y_train)
            component_probabilities.append(float(model.predict_proba(test_scaled)[:, 1][0]))
        probability = float(np.mean(component_probabilities))
        rows.append(
            {
                "target_month": target_month,
                "signal_date": signal_date,
                "factor_name": "technical_ml_ensemble",
                "p_up": probability,
                "score": float(np.clip((probability - 0.5) / 0.15, -1, 1)),
                "p_extra_trees": component_probabilities[0],
                "p_hist_gradient_boosting": component_probabilities[1],
                "p_logistic": component_probabilities[2],
                "train_rows": len(train_index),
                "feature_count": len(selected),
            }
        )
        if (number + 1) % 12 == 0:
            print(f"Technical ML ensemble refits: {number + 1}/{len(refit_months)}")
    out = pd.DataFrame(rows).set_index("target_month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.reindex(target_months).ffill()


def run_vol_target(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive_path: pd.DataFrame,
    target_vol: float = 0.15,
    financing_rate: float = 0.04,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index).intersection(defensive_path.index)
    rows = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        defensive = defensive_path.loc[month, [f"w_{asset}" for asset in ASSETS]].to_numpy(dtype=float)
        base = 0.40 * hard + 0.60 * defensive
        history = returns.loc[returns.index < month, ASSETS].tail(24)
        if len(history) >= 12:
            conditional_path = history.to_numpy(dtype=float) @ base
            weights = np.exp(np.linspace(-2.0, 0.0, len(conditional_path)))
            weights /= weights.sum()
            mean = float(weights @ conditional_path)
            variance = float(weights @ (conditional_path - mean) ** 2)
            forecast_vol = math.sqrt(max(variance, 1e-8) * 12)
            leverage = float(np.clip(target_vol / forecast_vol, 0.50, 1.50))
        else:
            forecast_vol = np.nan
            leverage = 1.20
        asset_weights = leverage * base
        debt_weight = 1 - leverage
        delta = asset_weights - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = (
            abs((asset_weights[2] + asset_weights[3]) - (pretrade[2] + pretrade[3]))
            * 0.0005
            * cost_multiplier
        )
        financing = debt_weight * ((1 + financing_rate) ** (1 / 12) - 1)
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(asset_weights @ asset_return + financing)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = asset_weights * (1 + asset_return) / (1 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "financing_return": financing,
                "forecast_vol": forecast_vol,
                "leverage": leverage,
                **{f"w_{asset}": asset_weights[i] for i, asset in enumerate(ASSETS)},
            }
        )
    return pd.DataFrame(rows).set_index("month")


def metric_record(period: str, strategy: str, backtest: pd.DataFrame) -> dict[str, float | str]:
    metrics = performance_summary(backtest["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(backtest["turnover"].mean()),
    }


def main() -> None:
    raw = download_ohlcv()
    frames = market_frames(raw)
    frames["KODEX200"] = splice_kospi200_proxy(frames["KODEX200"])
    close = frames["KODEX200"]["close"].dropna().loc["2000-01-01":]
    kr_hmm = pd.read_csv(CACHE / "kr_hmm_diag756.csv", index_col=0, parse_dates=True)
    us_hmm = pd.read_csv(CACHE / "us_hmm_diag756.csv", index_col=0, parse_dates=True)

    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")
    target_months = pd.period_range(FACTOR_TEST_START, signals.index.max(), freq="M")

    print("Building Jump Model, HMM, HSMM-style, and engineered technical factors...")
    factors: dict[str, pd.DataFrame] = {}
    factors["JumpModel"] = build_jump_factor(close, target_months)
    hmm_score = kr_hmm["kr_hmm_score"].astype(float)
    factors["GeneralHMM"] = monthly_factor_from_daily(hmm_score, target_months, "general_hmm")
    factors["DurationAwareHSMM"] = monthly_factor_from_daily(
        duration_aware_score(kr_hmm), target_months, "duration_aware_hsmm"
    )
    features, feature_close = build_features(frames, kr_hmm, us_hmm)
    factors["EngineeredTechnical"] = monthly_factor_from_daily(
        engineered_technical_score(features), target_months, "engineered_technical"
    )

    ensemble_path = RESULTS / "technical_ml_ensemble_factor.csv"
    if ensemble_path.exists():
        ensemble = pd.read_csv(ensemble_path, index_col=0)
        ensemble.index = pd.PeriodIndex(ensemble.index, freq="M")
        ensemble["signal_date"] = pd.to_datetime(ensemble["signal_date"])
    else:
        print("Building non-LSTM technical ML ensemble...")
        ensemble = walk_forward_technical_ensemble(features, feature_close, target_months)
        ensemble.to_csv(ensemble_path)
    factors["TechnicalMLEnsemble"] = ensemble

    for name, factor in list(factors.items()):
        factors[name] = attach_realized(factor, close)
        factors[name].to_csv(RESULTS / f"alternative_factor_{name.lower()}.csv")

    baseline_factor = next(iter(factors.values()))
    baseline = run_factor_blend(
        asset_returns,
        signals,
        defensive,
        baseline_factor,
        FactorBlendConfig(max_shift=0.0),
    )
    calibration_start = signals.index.min()
    baseline_cal = performance_summary(baseline.loc[calibration_start:CAL_END, "return"])

    calibration_rows = []
    all_backtests: dict[tuple[str, float], pd.DataFrame] = {}
    winners: dict[str, tuple[FactorBlendConfig, pd.DataFrame]] = {}
    for strategy, factor in factors.items():
        for max_shift in [0.0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.200]:
            cfg = FactorBlendConfig(max_shift=max_shift)
            backtest = run_factor_blend(asset_returns, signals, defensive, factor, cfg)
            all_backtests[(strategy, max_shift)] = backtest
            sample = backtest.loc[calibration_start:CAL_END]
            metrics = performance_summary(sample["return"])
            calibration_rows.append(
                {
                    "Strategy": strategy,
                    **asdict(cfg),
                    **metrics.to_dict(),
                    "AvgTurnover": float(sample["turnover"].mean()),
                    "MDD15Pass": bool(metrics["MDD"] >= -0.15),
                    "CAGRRetention": float(metrics["CAGR"] / baseline_cal["CAGR"]),
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    winner_rows = []
    for strategy, group in calibration.groupby("Strategy", sort=False):
        eligible = group[(group["MDD15Pass"]) & (group["CAGRRetention"] >= 0.95)]
        if eligible.empty:
            eligible = group[group["max_shift"] == 0.0]
        row = eligible.sort_values(["Sharpe", "Calmar", "CAGR"], ascending=False).iloc[0]
        cfg = FactorBlendConfig(
            max_shift=float(row["max_shift"]),
            probability_scale=float(row["probability_scale"]),
            hard_fraction=float(row["hard_fraction"]),
            leverage=float(row["leverage"]),
            financing_rate=float(row["financing_rate"]),
        )
        winners[strategy] = (cfg, all_backtests[(strategy, cfg.max_shift)])
        winner_rows.append(row.to_dict())

    vol_target = run_vol_target(asset_returns, signals, defensive, target_vol=0.15)
    winners["VolTarget15"] = (FactorBlendConfig(max_shift=np.nan), vol_target)

    comparison_rows = []
    periods = [
        ("calibration_2007_2017", calibration_start, CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", pd.Period("2007-01", "M"), None),
    ]
    for period, start, end in periods:
        strategies = {"FinalBlend": baseline, **{name: value[1] for name, value in winners.items()}}
        for name, backtest in strategies.items():
            sample = backtest.loc[start:end] if end is not None else backtest.loc[start:]
            comparison_rows.append(metric_record(period, name, sample))
    comparison = pd.DataFrame(comparison_rows)

    validation: dict[str, dict] = {}
    baseline_locked = baseline.loc[TEST_START:]
    baseline_locked_metrics = performance_summary(baseline_locked["return"])
    for name, (cfg, backtest) in winners.items():
        locked = backtest.loc[TEST_START:]
        metrics = performance_summary(locked["return"])
        sharpe_delta = float(metrics["Sharpe"] - baseline_locked_metrics["Sharpe"])
        calmar_delta = float(metrics["Calmar"] - baseline_locked_metrics["Calmar"])
        nonzero = True if name == "VolTarget15" else bool(cfg.max_shift > 0)
        gates = {
            "nonzero_selected": nonzero,
            "locked_sharpe_delta_at_least_0_02": bool(sharpe_delta >= 0.02),
            "locked_calmar_delta_at_least_0_05": bool(calmar_delta >= 0.05),
            "locked_mdd_within_15": bool(metrics["MDD"] >= -0.15),
            "locked_cagr_retention_at_least_95pct": bool(
                metrics["CAGR"] >= 0.95 * baseline_locked_metrics["CAGR"]
            ),
        }
        gates["performance_improved"] = bool(
            nonzero
            and (gates["locked_sharpe_delta_at_least_0_02"] or gates["locked_calmar_delta_at_least_0_05"])
            and gates["locked_mdd_within_15"]
            and gates["locked_cagr_retention_at_least_95pct"]
        )
        validation[name] = {
            "config": asdict(cfg),
            "locked_deltas": {
                "CAGR": float(metrics["CAGR"] - baseline_locked_metrics["CAGR"]),
                "Sharpe": sharpe_delta,
                "MDD": float(metrics["MDD"] - baseline_locked_metrics["MDD"]),
                "Calmar": calmar_delta,
            },
            "gates": gates,
            "bootstrap": paired_block_bootstrap(baseline_locked["return"], locked["return"]),
        }
        backtest.to_csv(RESULTS / f"alternative_backtest_{name.lower()}.csv")

    report = {
        "factor_oos_start": str(FACTOR_TEST_START),
        "portfolio_start": str(signals.index.min()),
        "calibration_end": str(CAL_END),
        "locked_start": str(TEST_START),
        "prediction_metrics_2005_2026": {
            name: factor_prediction_metrics(factor, FACTOR_TEST_START)
            for name, factor in factors.items()
        },
        "prediction_metrics_locked": {
            name: factor_prediction_metrics(factor, TEST_START)
            for name, factor in factors.items()
        },
        "validation": validation,
        "successful_strategies": [
            name for name, detail in validation.items() if detail["gates"]["performance_improved"]
        ],
        "notes": {
            "DurationAwareHSMM": "Explicit-duration approximation over causal 3-state HMM posteriors; not a fitted parametric HSMM package.",
            "TechnicalMLEnsemble": "Equal-weight ExtraTrees, HistGradientBoosting, and LogisticRegression; LSTM excluded.",
            "VolTarget15": "15% annualized target with ex-ante 24-month EWMA portfolio volatility and leverage clipped to [0.5, 1.5].",
        },
    }

    calibration.to_csv(RESULTS / "feedback_alternatives_calibration.csv", index=False)
    pd.DataFrame(winner_rows).to_csv(RESULTS / "feedback_alternatives_winners.csv", index=False)
    comparison.to_csv(RESULTS / "feedback_alternatives_comparison.csv", index=False)
    (RESULTS / "feedback_alternatives_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== CALIBRATION WINNERS ===")
    print(
        pd.DataFrame(winner_rows)[["Strategy", "max_shift", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== SAME-SAMPLE COMPARISON ===")
    print(
        comparison[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== IMPROVEMENT GATES ===")
    print(json.dumps({name: detail["gates"] for name, detail in validation.items()}, ensure_ascii=False, indent=2))
    print("\n=== FACTOR PREDICTION METRICS 2005-2026 ===")
    print(json.dumps(report["prediction_metrics_2005_2026"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
