from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

source = (ROOT / "blend_leverage_experiment.py").read_text(encoding="utf-8")
source = source.split("\nmacro, _ = load_macro_data()")[0]
exec(compile(source, str(ROOT / "blend_leverage_experiment.py"), "exec"), globals())

macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
hard = run_backtest(asset_returns, signals, StrategyConfig(), mode="hard")
defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")

ranking = pd.read_csv(RESULTS / "blend_leverage_calibration.csv")
winner_row = ranking.loc[ranking["MDD"] >= -0.15].sort_values("CAGR", ascending=False).iloc[0]
winner = BlendConfig(
    hard_fraction=float(winner_row["hard_fraction"]),
    leverage=float(winner_row["leverage"]),
    financing_rate=float(winner_row["financing_rate"]),
)
winner_full = run_blend(asset_returns, signals, defensive, winner)

comparison_rows = []
for period, start, end in [
    ("calibration", None, "2017-12"),
    ("locked_test", "2018-01", None),
    ("full", None, None),
]:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("BlendMDD15Return", winner_full)]:
        sample = backtest.loc[start:end] if start else backtest.loc[:end] if end else backtest
        metrics = performance_summary(sample["return"])
        comparison_rows.append(
            {
                "Period": period,
                "Strategy": strategy,
                **metrics.to_dict(),
                "AvgTurnover": float(sample["turnover"].mean()),
            }
        )
comparison = pd.DataFrame(comparison_rows)
winner_full.to_csv(RESULTS / "blend_mdd15_return_backtest.csv")
comparison.to_csv(RESULTS / "blend_mdd15_return_comparison.csv", index=False)
with (RESULTS / "blend_mdd15_return_winner.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(winner), handle, ensure_ascii=False, indent=2)

print("=== CALIBRATION RETURN-MAXIMIZING MDD15 BLEND ===")
print(winner)
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
    ].round(4).to_string(index=False)
)
