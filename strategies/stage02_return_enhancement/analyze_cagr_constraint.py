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
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, asset_returns)
cfg = StrategyConfig()
bt = run_backtest(asset_returns, signals, cfg)

asset_stats = pd.DataFrame({asset: performance_summary(asset_returns.loc[bt.index, asset]) for asset in ASSETS}).T
weights = bt[[f"w_{a}" for a in ASSETS]].copy()
weights.columns = ASSETS
gross_contrib = weights.mul(asset_returns.loc[bt.index, ASSETS].to_numpy(), axis=0)

print("=== Asset stats on strategy sample ===")
print(asset_stats[["CAGR", "Volatility", "Sharpe", "MDD", "Calmar"]].round(4).to_string())
print("\n=== Weight / contribution audit ===")
audit = pd.DataFrame({
    "AvgWeight": weights.mean(),
    "AnnualGrossContribution": gross_contrib.mean() * 12,
    "ContributionShare": gross_contrib.mean() / gross_contrib.mean().sum(),
})
print(audit.round(4).to_string())
print("\nAnnual costs", float((bt["trade_cost"] + bt["fx_cost"]).mean() * 12))

by_regime = pd.concat([bt[["regime", "return"]], weights], axis=1).groupby("regime").agg(
    N=("return", "size"),
    MeanReturn=("return", "mean"),
    Vol=("return", "std"),
    **{f"Avg_{a}": (a, "mean") for a in ASSETS},
)
by_regime["AnnReturn"] = by_regime["MeanReturn"] * 12
by_regime["AnnVol"] = by_regime["Vol"] * np.sqrt(12)
print("\n=== By predicted regime ===")
print(by_regime.round(4).to_string())

print("\n=== Asset forward returns by predicted regime ===")
regime_asset = asset_returns.loc[bt.index].join(bt["regime"]).groupby("regime")[ASSETS].mean() * 12
print(regime_asset.round(4).to_string())

rolling = bt["return"].rolling(36).apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(12) if x.std(ddof=1) > 0 else np.nan)
print("\n=== Worst rolling 36m Sharpe ===", float(rolling.min()), "at", rolling.idxmin())
print("=== Final performance ===")
print(performance_summary(bt["return"]).round(4).to_string())
