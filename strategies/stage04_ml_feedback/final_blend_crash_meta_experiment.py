from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import (
    CAL_END,
    TEST_START,
    FactorBlendConfig,
    paired_block_bootstrap,
    run_factor_blend,
)
from strategies.core.regime_research import (
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage04_ml_feedback.short_regime_tail_risk_experiment import causal_percentile


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DOMESTIC_FEATURES = [
    "base_USO",
    "base_GLD",
    "base_KODEX200",
    "p_inflation_high",
    "proxy_mom1",
    "proxy_mom6",
    "proxy_vol6",
    "daily_mom21",
    "daily_mom252",
    "daily_vol21",
    "daily_downvol21",
    "daily_mean_corr63",
]
STRESS_FEATURES = DOMESTIC_FEATURES + [
    "VIX_last",
    "VIX_last_d1",
    "VIX_last_z60",
    "BAA_SPREAD_last_d1",
    "NFCI_last_d1",
    "STLFSI_last_d1",
]
TRIGGER_PERCENTILE = 0.80


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def walk_forward_probability(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    min_train: int = 36,
) -> pd.Series:
    probability = pd.Series(np.nan, index=data.index, dtype=float)
    template = make_model()
    for number in range(len(data)):
        # Exclude the immediately prior row as a conservative one-month label embargo.
        train_end = number - 1
        if train_end < min_train:
            continue
        train = data.iloc[:train_end]
        y = train[target].astype(int)
        if y.sum() < 4 or (len(y) - y.sum()) < 12:
            continue
        model = clone(template)
        model.fit(train[features], y)
        probability.iloc[number] = float(model.predict_proba(data.iloc[[number]][features])[:, 1][0])
    return probability


def make_factor(probability: pd.Series, target: pd.Series, name: str) -> pd.DataFrame:
    factor = pd.DataFrame({"p_tail_raw": probability, "tail_event": target.astype(float)})
    factor["risk_percentile"] = causal_percentile(factor["p_tail_raw"])
    severity = ((factor["risk_percentile"] - TRIGGER_PERCENTILE) / (1 - TRIGGER_PERCENTILE)).clip(0, 1)
    factor["risk_severity"] = severity
    factor["p_up"] = 0.5 - 0.15 * severity
    factor["score"] = -severity
    factor["factor_name"] = name
    return factor


def predictive_metrics(factor: pd.DataFrame, start: pd.Period | None = None) -> dict[str, float]:
    view = factor.loc[start:] if start is not None else factor
    view = view.dropna(subset=["p_tail_raw", "tail_event"])
    y = view["tail_event"].astype(int)
    p = view["p_tail_raw"].clip(1e-6, 1 - 1e-6)
    cutoff = float(p.quantile(0.80))
    selected = p >= cutoff
    return {
        "observations": int(len(view)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if y.nunique() > 1 else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "recall_at_top_20pct": float(y[selected].sum() / max(y.sum(), 1)),
        "precision_at_top_20pct": float(y[selected].mean()),
    }


def metric_record(period: str, strategy: str, backtest: pd.DataFrame) -> dict[str, float | str]:
    metrics = performance_summary(backtest["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(backtest["turnover"].mean()),
    }


def main() -> None:
    feature_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    feature_data.index = pd.PeriodIndex(feature_data.index, freq="M")
    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")

    neutral = pd.DataFrame({"p_up": 0.5}, index=feature_data.index)
    baseline = run_factor_blend(
        asset_returns,
        signals,
        defensive,
        neutral,
        FactorBlendConfig(max_shift=0.0),
    )
    feature_data = feature_data.loc[feature_data.index.intersection(baseline.index)].copy()
    baseline_return = baseline.loc[feature_data.index, "return"]
    lagged_vol = baseline_return.rolling(24, min_periods=12).std(ddof=1).shift(1)
    feature_data["final_loss3"] = (baseline_return < -0.03).astype(int)
    feature_data["final_loss4"] = (baseline_return < -0.04).astype(int)
    feature_data["final_loss_1sigma"] = (baseline_return < -lagged_vol).astype(int)

    specifications = {
        "loss3_stress": ("final_loss3", STRESS_FEATURES),
        "loss4_domestic": ("final_loss4", DOMESTIC_FEATURES),
        "loss4_stress": ("final_loss4", STRESS_FEATURES),
        "loss_1sigma_stress": ("final_loss_1sigma", STRESS_FEATURES),
    }
    factors: dict[str, pd.DataFrame] = {}
    for name, (target, selected_features) in specifications.items():
        probability = walk_forward_probability(feature_data, target, selected_features)
        factor = make_factor(probability, feature_data[target], target)
        factors[name] = factor
        factor.to_csv(RESULTS / f"final_blend_crash_meta_{name}_factor.csv")

    baseline_calibration = performance_summary(baseline.loc[:CAL_END, "return"])
    shifts = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    calibration_rows: list[dict[str, object]] = []
    backtests: dict[tuple[str, float], pd.DataFrame] = {}
    winners: dict[str, tuple[float, pd.DataFrame]] = {}
    for name, factor in factors.items():
        for shift in shifts:
            backtest = run_factor_blend(
                asset_returns,
                signals,
                defensive,
                factor,
                FactorBlendConfig(max_shift=shift),
            )
            backtests[(name, shift)] = backtest
            metrics = performance_summary(backtest.loc[:CAL_END, "return"])
            calibration_rows.append(
                {
                    "Strategy": name,
                    "max_shift": shift,
                    **metrics.to_dict(),
                    "AvgTurnover": float(backtest.loc[:CAL_END, "turnover"].mean()),
                }
            )
        table = pd.DataFrame([row for row in calibration_rows if row["Strategy"] == name])
        eligible = table[
            (table["MDD"] >= -0.15)
            & (table["CAGR"] >= 0.95 * float(baseline_calibration["CAGR"]))
        ]
        pool = eligible if not eligible.empty else table
        winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
        shift = float(winner["max_shift"])
        winners[name] = (shift, backtests[(name, shift)])

    comparison_rows: list[dict[str, object]] = []
    periods = (
        ("calibration_2007_2017", baseline.index.min(), CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", baseline.index.min(), None),
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
            "prediction_calibration": predictive_metrics(factors[name].loc[:CAL_END]),
            "prediction_locked": predictive_metrics(factors[name], TEST_START),
            "locked_deltas": {
                key: float(metrics[key] - baseline_metrics[key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar")
            },
            "bootstrap": paired_block_bootstrap(baseline_locked["return"], locked["return"]),
        }
        backtest.to_csv(RESULTS / f"final_blend_crash_meta_{name}_backtest.csv")

    report = {
        "method": {
            "target": "next-month FinalBlend loss meta-label",
            "features": "domestic portfolio state plus lagged VIX/credit/financial-stress auxiliaries",
            "model": "expanding walk-forward balanced logistic regression with one-month label embargo",
            "position_mapping": "causal top-20% risk percentile, de-risk only",
            "calibration_end": str(CAL_END),
            "locked_start": str(TEST_START),
        },
        "validation": validation,
    }
    pd.DataFrame(calibration_rows).to_csv(RESULTS / "final_blend_crash_meta_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "final_blend_crash_meta_comparison.csv", index=False)
    (RESULTS / "final_blend_crash_meta_validation.json").write_text(
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
