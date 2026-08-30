from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", "M")

# Reuse the transparent allocation and cost implementation, excluding its ML
# search.  Here rank protection is disabled; VIX, realized volatility and the
# portfolio's own drawdown are the only risk controls.
source = (ROOT / "hard_crash_rank_experiment.py").read_text(encoding="utf-8")
source = source.split("\nfeature_data = pd.read_csv")[0]
exec(compile(source, str(ROOT / "hard_crash_rank_experiment.py"), "exec"), globals())

feature_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
feature_data.index = pd.PeriodIndex(feature_data.index, freq="M")
macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
base_cfg = StrategyConfig()
no_probability = pd.Series(0.0, index=feature_data.index)
no_rank = pd.Series(0.0, index=feature_data.index)

candidates = [
    RankConfig("none", 2.0, 0.0, vix, vix_fraction, vol_cap, dd_start, dd_fraction)
    for vix, vix_fraction, vol_cap, dd_start, dd_fraction in itertools.product(
        [20.0, 25.0, 30.0, 35.0, 40.0, 99.0],
        [0.50, 0.75, 1.00],
        [0.18, 0.22, 0.26, 0.30, 0.35, 99.0],
        [-0.03, -0.05, -0.07, -0.10],
        [0.50, 0.75, 1.00],
    )
]

rows = []
for candidate in candidates:
    backtest = run_rank_protected(
        asset_returns,
        signals,
        feature_data,
        no_probability,
        no_rank,
        candidate,
        end=str(CAL_END),
    )
    metrics = performance_summary(backtest["return"])
    rows.append(
        {
            "name": candidate.name,
            **asdict(candidate),
            **metrics.to_dict(),
            "AvgTurnover": float(backtest["turnover"].mean()),
            "AvgProtection": float(backtest["protection"].mean()),
            "MDD15Pass": bool(metrics["MDD"] >= -0.15),
        }
    )

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "simple_risk_overlay_calibration.csv", index=False)
eligible = ranking.loc[(ranking["MDD15Pass"]) & (ranking["Sharpe"] >= 1.0)]
eligible = eligible.sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
print("=== CALIBRATION MDD15 ELIGIBLE ===", len(eligible))
print(
    eligible[
        ["name", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection"]
    ].head(30).round(4).to_string(index=False)
)
if eligible.empty:
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = RankConfig(
    **{
        field: str(winner_row[field]) if field == "model_name" else float(winner_row[field])
        for field in RankConfig.__dataclass_fields__
    }
)
winner_full = run_rank_protected(
    asset_returns,
    signals,
    feature_data,
    no_probability,
    no_rank,
    winner,
)
hard = run_backtest(asset_returns, signals, base_cfg, mode="hard")
defensive = run_backtest(asset_returns, signals, base_cfg, mode="proposed")

comparison_rows = []
for period, start, end in [
    ("calibration", None, "2017-12"),
    ("locked_test", "2018-01", None),
    ("full", None, None),
]:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("SimpleRiskOverlay", winner_full)]:
        sample = backtest.loc[start:end] if start else backtest.loc[:end] if end else backtest
        metrics = performance_summary(sample["return"])
        comparison_rows.append(
            {
                "Period": period,
                "Strategy": strategy,
                **metrics.to_dict(),
                "AvgTurnover": float(sample["turnover"].mean()),
                "AvgProtection": float(sample.get("protection", pd.Series(0.0, index=sample.index)).mean()),
            }
        )
comparison = pd.DataFrame(comparison_rows)
print("\n=== PREDECLARED WINNER / LOCKED TEST ===")
print(winner)
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection"]
    ].round(4).to_string(index=False)
)

winner_full.to_csv(RESULTS / "simple_risk_overlay_backtest.csv")
comparison.to_csv(RESULTS / "simple_risk_overlay_comparison.csv", index=False)
with (RESULTS / "simple_risk_overlay_winner.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(winner), handle, ensure_ascii=False, indent=2)
