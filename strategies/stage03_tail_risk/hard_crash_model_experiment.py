from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strategies.core.regime_research import (
    ASSETS,
    DEFENSIVE,
    StrategyConfig,
    compute_regime_signals,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", "M")
TEST_START = pd.Period("2018-01", "M")


FEATURE_SETS = {
    "market": [
        "base_USO", "base_GLD", "base_KODEX200", "p_inflation_high",
        "proxy_mom1", "proxy_mom6", "proxy_mom12", "proxy_vol6", "proxy_vol12",
        "daily_mom21", "daily_mom252", "daily_vol21", "daily_downvol21",
        "daily_skew21", "daily_mean_corr63",
    ],
    "stress": [
        "base_USO", "base_GLD", "base_KODEX200", "p_inflation_high",
        "proxy_mom1", "proxy_mom6", "proxy_vol6",
        "daily_mom21", "daily_mom252", "daily_vol21", "daily_downvol21",
        "daily_mean_corr63", "VIX_last", "VIX_last_d1", "VIX_last_z60",
        "BAA_SPREAD_last_d1", "NFCI_last_d1", "STLFSI_last_d1",
    ],
}


def make_model(kind: str, strength: float):
    if kind == "logit":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=strength, class_weight="balanced", max_iter=2000, solver="liblinear")),
        ])
    if kind == "gbdt":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.04,
                max_depth=int(strength),
                min_samples_leaf=8,
                subsample=0.80,
                random_state=20260824,
            )),
        ])
    raise ValueError(kind)


def walkforward_probability(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    model,
    min_train: int = 36,
) -> pd.Series:
    p = pd.Series(np.nan, index=data.index, dtype=float)
    for i in range(len(data)):
        # One extra month of training-label lag: at the target month's first
        # open, the immediately prior open-to-open return is not assumed to be
        # available early enough for model refitting and execution.
        train_end = i - 1
        if train_end < min_train:
            continue
        train = data.iloc[:train_end]
        y = train[target].astype(int)
        if y.sum() < 4 or (len(y) - y.sum()) < 12:
            continue
        fitted = clone(model)
        sample_weight = None
        if model.named_steps["model"].__class__.__name__ == "GradientBoostingClassifier":
            positive_weight = (len(y) - y.sum()) / max(y.sum(), 1)
            sample_weight = np.where(y.to_numpy() == 1, positive_weight, 1.0)
            fitted.fit(train[features], y, model__sample_weight=sample_weight)
        else:
            fitted.fit(train[features], y)
        p.iloc[i] = float(fitted.predict_proba(data.iloc[[i]][features])[:, 1][0])
    return p


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_set: str
    target: str
    kind: str
    strength: float


@dataclass(frozen=True)
class ProtectionConfig:
    model_name: str
    probability_threshold: float
    protection_fraction: float
    vix_threshold: float
    fsi_threshold: float
    vol_cap: float
    dd_start: float

    @property
    def name(self) -> str:
        return (
            f"{self.model_name}_pt{self.probability_threshold:.2f}_pf{self.protection_fraction:.2f}"
            f"_vx{self.vix_threshold:.0f}_fs{self.fsi_threshold:.0f}_vc{self.vol_cap:.2f}_dd{abs(self.dd_start):.2f}"
        )


def run_protected(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    feature_data: pd.DataFrame,
    probability: pd.Series,
    cfg: ProtectionConfig,
    start: str | None = None,
    end: str | None = None,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index).intersection(feature_data.index)
    if start:
        months = months[months >= pd.Period(start, "M")]
    if end:
        months = months[months <= pd.Period(end, "M")]

    rows = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        base = hard_regime_weights(signals.loc[month])
        f = feature_data.loc[month]
        p = float(probability.loc[month]) if pd.notna(probability.loc[month]) else 0.0

        protection = cfg.protection_fraction if p >= cfg.probability_threshold else 0.0
        crisis_rule = bool(f["VIX_last"] >= cfg.vix_threshold or f["STLFSI_last"] >= cfg.fsi_threshold)
        if crisis_rule:
            protection = 1.0
        forecast_vol = float(f["daily_vol21"])
        if np.isfinite(forecast_vol) and forecast_vol > cfg.vol_cap:
            protection = max(protection, float(np.clip(1 - cfg.vol_cap / forecast_vol, 0.0, 1.0)))

        current_dd = nav / peak - 1.0
        if current_dd < cfg.dd_start:
            severity = float(np.clip((cfg.dd_start - current_dd) / 0.05, 0.0, 1.0))
            protection = max(protection, 0.70 + 0.30 * severity)

        w = (1 - protection) * base + protection * DEFENSIVE
        w = np.clip(w, 0.0, None)
        w /= w.sum()
        delta = w - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((w[2] + w[3]) - (pretrade[2] + pretrade[3])) * 0.0005 * cost_multiplier
        asset_r = returns.loc[month, ASSETS].to_numpy()
        gross_return = float(w @ asset_r)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        end_w = w * (1 + asset_r) / (1 + gross_return)
        rows.append({
            "month": month,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            "regime": signals.loc[month, "regime"],
            "crash_probability": p,
            "protection": protection,
            "crisis_rule": crisis_rule,
            "forecast_vol": forecast_vol,
            **{f"w_{asset}": w[i] for i, asset in enumerate(ASSETS)},
            **{f"hard_w_{asset}": base[i] for i, asset in enumerate(ASSETS)},
        })
        pretrade = end_w
        first_trade = False
    return pd.DataFrame(rows).set_index("month")


feature_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
feature_data.index = pd.PeriodIndex(feature_data.index, freq="M")
features_macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features_macro, asset_returns)
base_cfg = StrategyConfig()

specs = [
    ModelSpec("none", "market", "loss5", "logit", 0.1),
    ModelSpec("logit_m_l5_c01", "market", "loss5", "logit", 0.1),
    ModelSpec("logit_s_l5_c01", "stress", "loss5", "logit", 0.1),
    ModelSpec("logit_s_l8_c01", "stress", "loss8", "logit", 0.1),
    ModelSpec("gbdt_s_l5_d1", "stress", "loss5", "gbdt", 1.0),
    ModelSpec("gbdt_s_l5_d2", "stress", "loss5", "gbdt", 2.0),
]

probabilities: dict[str, pd.Series] = {}
model_audit = []
for spec in specs:
    if spec.name == "none":
        p = pd.Series(0.0, index=feature_data.index)
    else:
        p = walkforward_probability(
            feature_data,
            FEATURE_SETS[spec.feature_set],
            spec.target,
            make_model(spec.kind, spec.strength),
        )
    probabilities[spec.name] = p
    for label, mask in [
        ("calibration", feature_data.index <= CAL_END),
        ("locked", feature_data.index >= TEST_START),
    ]:
        view = pd.DataFrame({"p": p[mask], "y": feature_data.loc[mask, spec.target]}).dropna()
        auc = roc_auc_score(view["y"], view["p"]) if spec.name != "none" and view["y"].nunique() == 2 else np.nan
        model_audit.append({"Model": spec.name, "Period": label, "N": len(view), "Events": int(view["y"].sum()), "AUC": auc})
print("=== WALK-FORWARD MODEL AUDIT ===")
print(pd.DataFrame(model_audit).round(4).to_string(index=False))

hard_cal = run_backtest(asset_returns, signals, base_cfg, mode="hard", end=str(CAL_END))
defensive_cal = run_backtest(asset_returns, signals, base_cfg, mode="proposed", end=str(CAL_END))
hard_cal_m = performance_summary(hard_cal["return"])
defensive_cal_m = performance_summary(defensive_cal["return"])

candidates = [
    ProtectionConfig(model, prob, fraction, vix, fsi, vol_cap, dd_start)
    for model, prob, fraction, vix, fsi, vol_cap, dd_start in itertools.product(
        [s.name for s in specs],
        [0.45, 0.55, 0.65],
        [0.75, 1.00],
        [25.0, 30.0, 35.0],
        [1.0, 2.0, 99.0],
        [0.18, 0.22, 0.26, 99.0],
        [-0.03, -0.05],
    )
]

rows = []
for i, candidate in enumerate(candidates, start=1):
    bt = run_protected(asset_returns, signals, feature_data, probabilities[candidate.model_name], candidate, end=str(CAL_END))
    m = performance_summary(bt["return"])
    rows.append({
        "name": candidate.name,
        **asdict(candidate),
        **m.to_dict(),
        "AvgTurnover": float(bt["turnover"].mean()),
        "AvgProtection": float(bt["protection"].mean()),
        "MDD9Pass": bool(m["MDD"] >= -0.09),
        "MDD10Pass": bool(m["MDD"] >= -0.10),
        "ReturnRetention": float(m["CAGR"] / hard_cal_m["CAGR"]),
    })
    if i % 500 == 0:
        print(f"tested {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "hard_crash_model_calibration.csv", index=False)
eligible = ranking[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)

print("\n=== CALIBRATION BASELINES ===")
print("Hard", hard_cal_m.round(4).to_dict())
print("Current defensive", defensive_cal_m.round(4).to_dict())
print("\n=== STRICT 9% MDD ELIGIBLE ===", len(eligible))
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection", "ReturnRetention"]].head(25).round(4).to_string(index=False))
print("\n=== BEST MDD FRONTIER ===")
print(ranking.sort_values(["MDD", "CAGR"], ascending=[False, False])[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgProtection", "ReturnRetention"]].head(20).round(4).to_string(index=False))

if eligible.empty:
    print("No candidate meets the strict calibration gate.")
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = ProtectionConfig(**{field: winner_row[field] for field in ProtectionConfig.__dataclass_fields__})
print("\n=== LOCKED WINNER ===")
print(winner)

comparison_rows = []
for label, start, end in [("calibration", None, str(CAL_END)), ("locked_test", str(TEST_START), None), ("full", None, None)]:
    tests = [
        ("Hard", run_backtest(asset_returns, signals, base_cfg, mode="hard", start=start, end=end)),
        ("CurrentDefensive", run_backtest(asset_returns, signals, base_cfg, mode="proposed", start=start, end=end)),
        ("CrashProtectedHard", run_protected(asset_returns, signals, feature_data, probabilities[winner.model_name], winner, start=start, end=end)),
    ]
    for strategy, bt in tests:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})

comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINES VS WINNER ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_full = run_protected(asset_returns, signals, feature_data, probabilities[winner.model_name], winner)
winner_full.to_csv(RESULTS / "hard_crash_model_backtest.csv")
comparison.to_csv(RESULTS / "hard_crash_model_comparison.csv", index=False)
pd.DataFrame(model_audit).to_csv(RESULTS / "hard_crash_model_auc.csv", index=False)
with (RESULTS / "hard_crash_model_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)

