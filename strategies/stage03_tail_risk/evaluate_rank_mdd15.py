from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Load definitions only; the original strict-9% search remains unchanged.
source = (ROOT / "hard_crash_rank_experiment.py").read_text(encoding="utf-8")
source = source.split("\nfeature_data = pd.read_csv")[0]
exec(compile(source, str(ROOT / "hard_crash_rank_experiment.py"), "exec"), globals())

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

# This is the sole primary candidate: maximize calibration CAGR subject to the
# user-approved 15% MDD ceiling.  Locked-test results are not used to select it.
ranking = pd.read_csv(RESULTS / "hard_crash_rank_calibration.csv")
eligible = ranking.loc[(ranking["MDD"] >= -0.15) & (ranking["Sharpe"] >= 1.0)]
primary_row = eligible.sort_values(["CAGR", "Sharpe"], ascending=False).iloc[0]
primary = RankConfig(
    **{
        field: str(primary_row[field]) if field == "model_name" else float(primary_row[field])
        for field in RankConfig.__dataclass_fields__
    }
)
primary_full = run_rank_protected(
    asset_returns,
    signals,
    feature_data,
    probabilities[primary.model_name],
    ranks[primary.model_name],
    primary,
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
        ("RankMDD15Primary", primary_full),
    ]:
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
primary_full.to_csv(RESULTS / "hard_crash_rank_mdd15_backtest.csv")
comparison.to_csv(RESULTS / "hard_crash_rank_mdd15_comparison.csv", index=False)
with (RESULTS / "hard_crash_rank_mdd15_primary.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(primary), handle, ensure_ascii=False, indent=2)

print("=== PREDECLARED PRIMARY CONFIGURATION ===")
print(primary)
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection"]
    ].round(4).to_string(index=False)
)
