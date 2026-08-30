from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    compute_regime_signals,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

features, _ = load_macro_data()
returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, returns)
hard = run_backtest(returns, signals, StrategyConfig(), mode="hard")

stress = pd.read_csv(ROOT / "cache" / "stress_monthly.csv", index_col=0)
stress.index = pd.PeriodIndex(stress.index, freq="M")

market_daily = pd.read_csv(ROOT / "cache" / "market_daily.csv", parse_dates=["date"])
market_levels = market_daily.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
market_levels["GLD"] = market_levels["GLD"] * market_levels["USDKRW"]
market_levels["USO"] = market_levels["USO"] * market_levels["USDKRW"]
bond_daily = pd.read_csv(ROOT / "raw_data" / "krx_bond_index.csv", encoding="cp949")
bond_daily.index = pd.to_datetime(bond_daily.iloc[:, 0])
bond_level = pd.to_numeric(bond_daily.iloc[:, 1].astype(str).str.replace(",", "", regex=False), errors="coerce")
daily_levels = pd.concat(
    [market_levels["KODEX200"], bond_level.rename("BOND"), market_levels["GLD"], market_levels["USO"]],
    axis=1,
).sort_index()
daily_levels.columns = ASSETS
daily_levels = daily_levels.reindex(pd.date_range(daily_levels.index.min(), daily_levels.index.max(), freq="B")).ffill(limit=5)
daily_returns = daily_levels.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

rows = []
for month in hard.index:
    history = returns.loc[returns.index < month, ASSETS]
    if len(history) < 3:
        continue
    signal = signals.loc[month]
    base = hard_regime_weights(signal)
    proxy = pd.Series(history.to_numpy() @ base, index=history.index)
    stress_month = month - 1
    stress_row = stress.loc[stress_month] if stress_month in stress.index else pd.Series(dtype=float)

    row: dict[str, object] = {
        "month": month,
        "regime": signal["regime"],
        "p_growth_high": signal["p_growth_high"],
        "p_inflation_high": signal["p_inflation_high"],
        "hard_return": hard.loc[month, "return"],
        "loss5": int(hard.loc[month, "return"] < -0.05),
        "loss8": int(hard.loc[month, "return"] < -0.08),
    }
    for asset, weight in zip(ASSETS, base):
        row[f"base_{asset}"] = weight
    for window in [1, 2, 3, 6, 9, 12]:
        view = proxy.tail(window)
        row[f"proxy_mom{window}"] = float((1 + view).prod() - 1)
        if window >= 3:
            row[f"proxy_vol{window}"] = float(view.std(ddof=1) * np.sqrt(12))
            row[f"proxy_skew{window}"] = float(view.skew()) if window >= 6 else 0.0
    for window in [6, 12, 24]:
        corr = history.tail(window)[["KODEX200", "GLD", "USO"]].corr()
        vals = corr.to_numpy()[np.triu_indices(3, 1)]
        row[f"mean_corr{window}"] = float(np.nanmean(vals))
        row[f"max_corr{window}"] = float(np.nanmax(vals))

    # Two-calendar-day safety buffer avoids using a U.S. close that occurs
    # after the first Korean trading session of the target month.
    cutoff = month.to_timestamp(how="start") - pd.Timedelta(days=2)
    daily_history = daily_returns.loc[:cutoff, ASSETS].dropna(how="all")
    daily_proxy = pd.Series(daily_history.fillna(0.0).to_numpy() @ base, index=daily_history.index)
    for window in [21, 63, 126, 252]:
        view = daily_proxy.tail(window)
        row[f"daily_mom{window}"] = float((1 + view).prod() - 1)
        if window <= 63:
            row[f"daily_vol{window}"] = float(view.std(ddof=1) * np.sqrt(252))
            row[f"daily_skew{window}"] = float(view.skew())
            row[f"daily_downvol{window}"] = float(np.sqrt(np.mean(np.minimum(view, 0.0) ** 2)) * np.sqrt(252))
    for window in [63, 126]:
        corr = daily_history.tail(window)[["KODEX200", "GLD", "USO"]].corr()
        vals = corr.to_numpy()[np.triu_indices(3, 1)]
        row[f"daily_mean_corr{window}"] = float(np.nanmean(vals))
        row[f"daily_max_corr{window}"] = float(np.nanmax(vals))
    for col, value in stress_row.items():
        row[col] = value
    rows.append(row)

dataset = pd.DataFrame(rows).set_index("month")
dataset.to_csv(RESULTS / "hard_crash_features.csv")

print("=== EVENT COUNTS ===")
print(dataset[["loss5", "loss8"]].sum().to_string())
print("\n=== UNIVARIATE AUC FOR NEXT-MONTH HARD LOSS < -5% ===")
numeric = dataset.select_dtypes(include=[np.number]).drop(columns=["hard_return", "loss5", "loss8"])
auc_rows = []
for col in numeric:
    view = dataset[["loss5", col]].dropna()
    if view[col].nunique() < 2:
        continue
    auc = roc_auc_score(view["loss5"], view[col])
    auc_rows.append({"feature": col, "auc": auc, "directional_auc": max(auc, 1 - auc), "high_means_risk": auc >= 0.5})
print(pd.DataFrame(auc_rows).sort_values("directional_auc", ascending=False).head(30).round(3).to_string(index=False))

print("\n=== PRE-MONTH FEATURES FOR WORST HARD MONTHS ===")
show = [
    "regime", "hard_return", "proxy_mom1", "proxy_mom3", "proxy_mom6", "proxy_vol6", "proxy_skew12",
    "mean_corr12", "VIX_last", "VIX_max", "VIX_last_d1", "VIX_last_z60", "BAA_SPREAD_last",
    "BAA_SPREAD_last_d1", "BAA_SPREAD_last_z60", "NFCI_last", "NFCI_last_d1", "STLFSI_last", "STLFSI_last_d1",
]
print(dataset.nsmallest(25, "hard_return")[show].round(3).to_string())
