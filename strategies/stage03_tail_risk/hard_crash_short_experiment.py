from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

source = (ROOT / "hard_crash_model_experiment.py").read_text(encoding="utf-8")
source = source.split("\nfeature_data = pd.read_csv")[0]
exec(compile(source, str(ROOT / "hard_crash_model_experiment.py"), "exec"), globals())


def expanding_rank(probability: pd.Series, min_history: int = 24) -> pd.Series:
    result = pd.Series(np.nan, index=probability.index, dtype=float)
    for i in range(len(probability)):
        current = probability.iloc[i]
        past = probability.iloc[:i].dropna()
        if pd.notna(current) and len(past) >= min_history:
            result.iloc[i] = float((past <= current).mean())
    return result


@dataclass(frozen=True)
class ShortConfig:
    model_name: str
    rank_threshold: float
    hedge_strength: float
    vix_strength: float
    vol_cap: float
    dd_start: float
    dd_strength: float
    borrow_rate: float = 0.03

    @property
    def name(self) -> str:
        return (
            f"{self.model_name}_rq{self.rank_threshold:.2f}_hs{self.hedge_strength:.2f}"
            f"_vs{self.vix_strength:.2f}_vc{self.vol_cap:.2f}_dd{abs(self.dd_start):.2f}"
            f"_ds{self.dd_strength:.2f}"
        )


def run_short_hedged(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    feature_data: pd.DataFrame,
    probability: pd.Series,
    rank: pd.Series,
    cfg: ShortConfig,
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
        q = float(rank.loc[month]) if pd.notna(rank.loc[month]) else 0.0

        hedge = cfg.hedge_strength if q >= cfg.rank_threshold else 0.0
        if float(f["VIX_last"]) >= 35.0:
            hedge = max(hedge, cfg.vix_strength)
        forecast_vol = float(f["daily_vol21"])
        if np.isfinite(forecast_vol) and forecast_vol > cfg.vol_cap:
            hedge = max(hedge, float(np.clip(1 - cfg.vol_cap / forecast_vol, 0.0, 1.0)))

        current_dd = nav / peak - 1.0
        if current_dd < cfg.dd_start:
            severity = float(np.clip((cfg.dd_start - current_dd) / 0.06, 0.0, 1.0))
            hedge = max(hedge, cfg.dd_strength * (0.50 + 0.50 * severity))

        # hedge > 1 creates a modest short in the currently selected hard asset
        # financed by an overweight in the existing defensive basket.
        w = (1 - hedge) * base + hedge * DEFENSIVE
        delta = w - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((w[2] + w[3]) - (pretrade[2] + pretrade[3])) * 0.0005 * cost_multiplier
        borrow_cost = float(np.abs(np.minimum(w, 0.0)).sum() * cfg.borrow_rate / 12)
        asset_r = returns.loc[month, ASSETS].to_numpy()
        gross_return = float(w @ asset_r)
        net_return = gross_return - trade_cost - fx_cost - borrow_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = w * (1 + asset_r) / (1 + gross_return)
        first_trade = False
        rows.append({
            "month": month, "return": net_return, "gross_return": gross_return,
            "nav": nav, "drawdown": nav / peak - 1, "turnover": turnover,
            "trade_cost": trade_cost, "fx_cost": fx_cost, "borrow_cost": borrow_cost,
            "regime": signals.loc[month, "regime"], "crash_probability": p,
            "risk_rank": q, "hedge_strength": hedge, "gross_exposure": float(np.abs(w).sum()),
            **{f"w_{a}": w[i] for i, a in enumerate(ASSETS)},
            **{f"hard_w_{a}": base[i] for i, a in enumerate(ASSETS)},
        })
    return pd.DataFrame(rows).set_index("month")


feature_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
feature_data.index = pd.PeriodIndex(feature_data.index, freq="M")
macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
base_cfg = StrategyConfig()

specs = [
    ModelSpec("logit_s_l5_c01", "stress", "loss5", "logit", 0.1),
    ModelSpec("logit_s_l8_c01", "stress", "loss8", "logit", 0.1),
    ModelSpec("gbdt_s_l5_d1", "stress", "loss5", "gbdt", 1.0),
]
probabilities = {}
ranks = {}
for spec in specs:
    p = walkforward_probability(feature_data, FEATURE_SETS[spec.feature_set], spec.target, make_model(spec.kind, spec.strength))
    probabilities[spec.name] = p
    ranks[spec.name] = expanding_rank(p)

hard = run_backtest(asset_returns, signals, base_cfg, mode="hard")
defensive = run_backtest(asset_returns, signals, base_cfg, mode="proposed")
hard_cal_m = performance_summary(hard.loc[:str(CAL_END), "return"])

candidates = [
    ShortConfig(model, rank_q, hedge, vix_hedge, vol_cap, dd_start, dd_hedge)
    for model, rank_q, hedge, vix_hedge, vol_cap, dd_start, dd_hedge in itertools.product(
        [s.name for s in specs],
        [0.90, 0.95],
        [1.00, 1.25, 1.50, 1.75],
        [1.00, 1.25],
        [0.22, 0.30, 99.0],
        [-0.03, -0.05],
        [0.75, 1.00, 1.25],
    )
]

rows = []
for i, candidate in enumerate(candidates, start=1):
    bt = run_short_hedged(
        asset_returns, signals, feature_data,
        probabilities[candidate.model_name], ranks[candidate.model_name], candidate, end=str(CAL_END),
    )
    m = performance_summary(bt["return"])
    rows.append({
        "name": candidate.name, **asdict(candidate), **m.to_dict(),
        "AvgTurnover": float(bt["turnover"].mean()), "AvgHedge": float(bt["hedge_strength"].mean()),
        "AvgGrossExposure": float(bt["gross_exposure"].mean()), "MDD9Pass": bool(m["MDD"] >= -0.09),
        "CAGRRetention": float(m["CAGR"] / hard_cal_m["CAGR"]),
    })
    if i % 300 == 0:
        print(f"tested {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "hard_crash_short_calibration.csv", index=False)
eligible = ranking[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
print("=== STRICT 9% MDD ELIGIBLE ===", len(eligible))
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgHedge", "AvgGrossExposure", "CAGRRetention"]].head(30).round(4).to_string(index=False))
if eligible.empty:
    print("No short-hedged candidate passed the strict calibration gate.")
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = ShortConfig(**{
    field: str(winner_row[field]) if field == "model_name" else float(winner_row[field])
    for field in ShortConfig.__dataclass_fields__
})
print("\n=== LOCKED WINNER ===")
print(winner)
winner_full = run_short_hedged(
    asset_returns, signals, feature_data,
    probabilities[winner.model_name], ranks[winner.model_name], winner,
)

comparison_rows = []
for label, start, end in [("calibration", None, "2017-12"), ("locked_test", "2018-01", None), ("full", None, None)]:
    for strategy, bt in [
        ("Hard", hard.loc[start:end] if start else hard.loc[:end] if end else hard),
        ("CurrentDefensive", defensive.loc[start:end] if start else defensive.loc[:end] if end else defensive),
        ("ShortHedgedHard", winner_full.loc[start:end] if start else winner_full.loc[:end] if end else winner_full),
    ]:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})
comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINES VS WINNER ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_full.to_csv(RESULTS / "hard_crash_short_backtest.csv")
comparison.to_csv(RESULTS / "hard_crash_short_comparison.csv", index=False)
with (RESULTS / "hard_crash_short_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)

