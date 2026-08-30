from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


features, _ = load_macro_data()
returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, returns)
cfg = StrategyConfig()

hard = run_backtest(returns, signals, cfg, mode="hard")
defensive = run_backtest(returns, signals, cfg, mode="proposed")


def cycle(bt: pd.DataFrame) -> dict[str, object]:
    wealth = (1 + bt["return"]).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovery = after[after >= wealth.loc[peak]].index.min() if (after >= wealth.loc[peak]).any() else None
    return {"peak": peak, "trough": trough, "recovery": recovery, "mdd": float(dd.min())}


print("=== PERFORMANCE ===")
for label, start, end in [
    ("calibration", None, "2017-12"),
    ("locked", "2018-01", None),
    ("full", None, None),
]:
    print(f"\n{label}")
    for name, mode in [("Hard", "hard"), ("Defensive", "proposed")]:
        bt = run_backtest(returns, signals, cfg, mode=mode, start=start, end=end)
        print(name, performance_summary(bt["return"])[["CAGR", "Volatility", "Sharpe", "MDD", "Calmar"]].round(4).to_dict())

print("\n=== MAX DRAWDOWN CYCLE ===")
print(cycle(hard))

weights = hard[[f"w_{a}" for a in ASSETS]].copy()
weights.columns = ASSETS
asset_r = returns.loc[hard.index, ASSETS]
contrib = weights * asset_r

print("\n=== REGIME CONTRIBUTION ===")
by_regime = hard.groupby("regime").agg(N=("return", "size"), Mean=("return", "mean"), Vol=("return", "std"))
by_regime["AnnMean"] = by_regime["Mean"] * 12
by_regime["AnnVol"] = by_regime["Vol"] * np.sqrt(12)
by_regime["HitRate"] = hard.groupby("regime")["return"].apply(lambda x: float((x > 0).mean()))
print(by_regime.round(4).to_string())

print("\n=== WORST 25 MONTHS ===")
worst = hard.nsmallest(25, "return")[["regime", "return", "drawdown", "turnover"]].join(asset_r, rsuffix="_asset")
print(worst.round(4).to_string())

print("\n=== ANNUAL RETURNS ===")
annual = pd.DataFrame({
    "Hard": hard["return"].groupby(hard.index.year).apply(lambda x: (1 + x).prod() - 1),
    "Defensive": defensive["return"].groupby(defensive.index.year).apply(lambda x: (1 + x).prod() - 1),
})
print(annual.round(4).to_string())

print("\n=== HARD REGIME TRANSITIONS ===")
changed = hard["regime"].ne(hard["regime"].shift())
print(hard.loc[changed, ["regime", "return", "drawdown"]].round(4).to_string())

print("\n=== WORST MONTH AVOIDANCE VALUE ===")
for n in [1, 3, 5, 10, 15, 20]:
    adjusted = hard["return"].copy()
    adjusted.loc[adjusted.nsmallest(n).index] = defensive.loc[adjusted.nsmallest(n).index, "return"]
    m = performance_summary(adjusted)
    print(n, m[["CAGR", "Sharpe", "MDD", "Calmar"]].round(4).to_dict())

