from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

features, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, asset_returns)
baseline = run_backtest(asset_returns, signals, StrategyConfig())
accelerated = pd.read_csv(RESULTS / "cagr_accelerator_backtest.csv", index_col=0)
accelerated.index = pd.PeriodIndex(accelerated.index, freq="M")

common = baseline.index.intersection(accelerated.index)
baseline = baseline.loc[common]
accelerated = accelerated.loc[common]


def summary_row(label: str, strategy: str, returns: pd.Series, turnover: pd.Series) -> dict:
    m = performance_summary(returns)
    return {"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(turnover.mean())}


periods = {
    "Full": (common.min(), common.max()),
    "2007-2012": (pd.Period("2007-04", "M"), pd.Period("2012-12", "M")),
    "2013-2017": (pd.Period("2013-01", "M"), pd.Period("2017-12", "M")),
    "2018-2022": (pd.Period("2018-01", "M"), pd.Period("2022-12", "M")),
    "2023-2026": (pd.Period("2023-01", "M"), common.max()),
}
subperiod_rows = []
for label, (start, end) in periods.items():
    for strategy, bt in [("Baseline", baseline), ("Accelerated", accelerated)]:
        view = bt.loc[start:end]
        subperiod_rows.append(summary_row(label, strategy, view["return"], view["turnover"]))
subperiod = pd.DataFrame(subperiod_rows)
print("=== SUBPERIOD ROBUSTNESS ===")
print(subperiod[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))


# Cost sensitivity on the realized target-weight path.  This keeps positions
# fixed, isolating the direct implementation-cost impact.
cost_rows = []
base_cost = baseline["trade_cost"] + baseline["fx_cost"]
acc_cost = accelerated["trade_cost"] + accelerated["fx_cost"]
for multiplier in [0.0, 1.0, 2.0, 3.0]:
    for strategy, bt, cost in [
        ("Baseline", baseline, base_cost),
        ("Accelerated", accelerated, acc_cost),
    ]:
        adjusted = bt["gross_return"] - multiplier * cost
        row = summary_row(f"Cost x{multiplier:.0f}", strategy, adjusted, bt["turnover"])
        row["AnnualCost"] = float((multiplier * cost).mean() * 12)
        cost_rows.append(row)
cost_table = pd.DataFrame(cost_rows)
print("\n=== COST SENSITIVITY ===")
print(cost_table[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AnnualCost"]].round(4).to_string(index=False))


episodes = {
    "GFC": ("2007-10", "2009-03"),
    "Taper/Gold shock": ("2013-01", "2013-12"),
    "COVID": ("2020-01", "2020-12"),
    "Inflation": ("2022-01", "2022-12"),
}
episode_rows = []
for label, (start, end) in episodes.items():
    for strategy, bt in [("Baseline", baseline), ("Accelerated", accelerated)]:
        view = bt.loc[start:end]
        m = performance_summary(view["return"])
        episode_rows.append({"Episode": label, "Strategy": strategy, "Return": float((1 + view["return"]).prod() - 1), "MDD": float(m["MDD"]), "Sharpe": float(m["Sharpe"])})
episodes_table = pd.DataFrame(episode_rows)
print("\n=== STRESS EPISODES ===")
print(episodes_table.round(4).to_string(index=False))


def paired_block_bootstrap(base_r: pd.Series, acc_r: pd.Series, n_boot: int = 3000, block: int = 12, seed: int = 20260824) -> pd.DataFrame:
    b = np.asarray(base_r, dtype=float)
    a = np.asarray(acc_r, dtype=float)
    n = len(b)
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx: list[int] = []
        while len(idx) < n:
            start = int(rng.choice(starts))
            idx.extend(range(start, start + block))
        idx = idx[:n]
        bm = performance_summary(pd.Series(b[idx]))
        am = performance_summary(pd.Series(a[idx]))
        rows.append({
            "DeltaCAGR": float(am["CAGR"] - bm["CAGR"]),
            "DeltaSharpe": float(am["Sharpe"] - bm["Sharpe"]),
            # Positive DeltaMDD means the accelerated drawdown is less severe.
            "DeltaMDD": float(am["MDD"] - bm["MDD"]),
            "DeltaCalmar": float(am["Calmar"] - bm["Calmar"]),
        })
    return pd.DataFrame(rows)


boot = paired_block_bootstrap(baseline["return"], accelerated["return"])
boot_ci = boot.quantile([0.025, 0.50, 0.975])
boot_prob = pd.Series({
    "P(Delta CAGR > 0)": float((boot["DeltaCAGR"] > 0).mean()),
    "P(Delta Sharpe > 0)": float((boot["DeltaSharpe"] > 0).mean()),
    "P(Delta MDD >= 0)": float((boot["DeltaMDD"] >= 0).mean()),
    "P(Delta Calmar > 0)": float((boot["DeltaCalmar"] > 0).mean()),
})
print("\n=== PAIRED 12-MONTH BLOCK BOOTSTRAP (3000) ===")
print(boot_ci.round(4).to_string())
print(boot_prob.round(4).to_string())


audit = {
    "baseline_mdd_month": str(baseline["drawdown"].idxmin()),
    "accelerated_mdd_month": str(accelerated["drawdown"].idxmin()),
    "baseline_avg_turnover": float(baseline["turnover"].mean()),
    "accelerated_avg_turnover": float(accelerated["turnover"].mean()),
    "accelerated_avg_sleeve": float(accelerated["sleeve_used"].mean()),
    "accelerated_avg_brake": float(accelerated["brake_used"].mean()),
    "accelerated_avg_early_guard": float(accelerated["early_guard_used"].mean()),
    "bootstrap_probabilities": boot_prob.to_dict(),
}
print("\n=== AUDIT ===")
print(json.dumps(audit, ensure_ascii=False, indent=2))

subperiod.to_csv(RESULTS / "cagr_accelerator_subperiods.csv", index=False)
cost_table.to_csv(RESULTS / "cagr_accelerator_cost_sensitivity.csv", index=False)
episodes_table.to_csv(RESULTS / "cagr_accelerator_stress_episodes.csv", index=False)
boot_ci.to_csv(RESULTS / "cagr_accelerator_bootstrap_ci.csv")
with (RESULTS / "cagr_accelerator_validation.json").open("w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
