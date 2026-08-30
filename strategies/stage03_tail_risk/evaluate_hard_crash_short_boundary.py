from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


# Reuse only the imports, model definitions, and backtest function.  The source
# script's calibration loop is deliberately excluded so this file performs a
# single locked evaluation of the preselected boundary candidate.
source = (ROOT / "hard_crash_short_experiment.py").read_text(encoding="utf-8")
source = source.split("\nfeature_data = pd.read_csv")[0]
exec(compile(source, str(ROOT / "hard_crash_short_experiment.py"), "exec"), globals())

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
    probability = walkforward_probability(
        feature_data,
        FEATURE_SETS[spec.feature_set],
        spec.target,
        make_model(spec.kind, spec.strength),
    )
    probabilities[spec.name] = probability
    ranks[spec.name] = expanding_rank(probability)

ranking = pd.read_csv(RESULTS / "hard_crash_short_calibration.csv")
boundary = (
    ranking.loc[ranking["MDD"] >= -0.10]
    .sort_values(["CAGR", "Sharpe"], ascending=False)
    .iloc[0]
)
winner = ShortConfig(
    **{
        field: str(boundary[field]) if field == "model_name" else float(boundary[field])
        for field in ShortConfig.__dataclass_fields__
    }
)

candidate = run_short_hedged(
    asset_returns,
    signals,
    feature_data,
    probabilities[winner.model_name],
    ranks[winner.model_name],
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
    for strategy, backtest in [
        ("Hard", hard),
        ("CurrentDefensive", defensive),
        ("ShortHedgedBoundary", candidate),
    ]:
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
candidate.to_csv(RESULTS / "hard_crash_short_boundary_backtest.csv")
comparison.to_csv(RESULTS / "hard_crash_short_boundary_comparison.csv", index=False)
with (RESULTS / "hard_crash_short_boundary.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(winner), handle, ensure_ascii=False, indent=2)

print("=== PRESELECTED CALIBRATION BOUNDARY CANDIDATE ===")
print(winner)
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
    ].round(4).to_string(index=False)
)
