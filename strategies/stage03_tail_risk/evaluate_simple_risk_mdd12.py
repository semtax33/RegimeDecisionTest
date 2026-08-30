from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

source = (ROOT / "hard_crash_rank_experiment.py").read_text(encoding="utf-8")
source = source.split("\nfeature_data = pd.read_csv")[0]
exec(compile(source, str(ROOT / "hard_crash_rank_experiment.py"), "exec"), globals())

feature_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
feature_data.index = pd.PeriodIndex(feature_data.index, freq="M")
macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
no_probability = pd.Series(0.0, index=feature_data.index)
no_rank = pd.Series(0.0, index=feature_data.index)

ranking = pd.read_csv(RESULTS / "simple_risk_overlay_calibration.csv")
eligible = ranking.loc[(ranking["MDD"] >= -0.12) & (ranking["Sharpe"] >= 1.0)]
winner_row = eligible.sort_values(["CAGR", "Sharpe"], ascending=False).iloc[0]
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
hard = run_backtest(asset_returns, signals, StrategyConfig(), mode="hard")
defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")

comparison_rows = []
for period, start, end in [
    ("calibration", None, "2017-12"),
    ("locked_test", "2018-01", None),
    ("full", None, None),
]:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("SimpleRiskMDD12", winner_full)]:
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
winner_full.to_csv(RESULTS / "simple_risk_mdd12_backtest.csv")
comparison.to_csv(RESULTS / "simple_risk_mdd12_comparison.csv", index=False)
with (RESULTS / "simple_risk_mdd12_winner.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(winner), handle, ensure_ascii=False, indent=2)

print("=== CALIBRATION 12% SAFETY-BUFFER WINNER ===")
print(winner)
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection"]
    ].round(4).to_string(index=False)
)
