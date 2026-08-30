from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

source = (ROOT / "blend_leverage_experiment.py").read_text(encoding="utf-8")
source = source.split("\nmacro, _ = load_macro_data()")[0]
exec(compile(source, str(ROOT / "blend_leverage_experiment.py"), "exec"), globals())


def metric_row(label: str, returns: pd.Series) -> dict[str, float | str]:
    return {"Strategy": label, **performance_summary(returns).to_dict()}


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    dd = wealth / running_peak - 1.0
    episodes = []
    in_episode = False
    start = None
    trough = None
    trough_dd = 0.0
    for month, value in dd.items():
        if value < -1e-12 and not in_episode:
            in_episode = True
            loc = dd.index.get_loc(month)
            start = dd.index[max(loc - 1, 0)]
            trough = month
            trough_dd = float(value)
        elif in_episode and value < trough_dd:
            trough = month
            trough_dd = float(value)
        if in_episode and value >= -1e-12:
            episodes.append(
                {
                    "Peak": str(start),
                    "Trough": str(trough),
                    "Recovery": str(month),
                    "MDD": trough_dd,
                    "MonthsToRecovery": int(dd.index.get_loc(month) - dd.index.get_loc(start)),
                }
            )
            in_episode = False
    if in_episode:
        episodes.append(
            {
                "Peak": str(start),
                "Trough": str(trough),
                "Recovery": "Unrecovered",
                "MDD": trough_dd,
                "MonthsToRecovery": int(len(dd) - 1 - dd.index.get_loc(start)),
            }
        )
    return pd.DataFrame(episodes).sort_values("MDD").reset_index(drop=True)


def sampled_metrics(values: np.ndarray) -> tuple[float, float, float]:
    wealth = np.cumprod(1.0 + values)
    cagr = wealth[-1] ** (12.0 / len(values)) - 1.0
    volatility = values.std(ddof=1) * np.sqrt(12.0)
    sharpe = values.mean() / values.std(ddof=1) * np.sqrt(12.0)
    mdd = np.min(wealth / np.maximum.accumulate(wealth) - 1.0)
    return float(cagr), float(sharpe), float(mdd)


def paired_block_bootstrap(
    series: dict[str, pd.Series],
    block_length: int,
    simulations: int = 3000,
    seed: int = 20260824,
) -> tuple[pd.DataFrame, dict[str, float]]:
    common = next(iter(series.values())).index
    for item in series.values():
        common = common.intersection(item.index)
    matrix = np.column_stack([series[name].loc[common].to_numpy(dtype=float) for name in series])
    names = list(series)
    n = len(matrix)
    rng = np.random.default_rng(seed + block_length)
    records = []
    comparisons = []
    for _ in range(simulations):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_length + 1))
            indices.extend(range(start, start + block_length))
        draw = matrix[np.asarray(indices[:n])]
        metrics = {name: sampled_metrics(draw[:, j]) for j, name in enumerate(names)}
        for name, (cagr, sharpe, mdd) in metrics.items():
            records.append(
                {
                    "BlockLength": block_length,
                    "Strategy": name,
                    "CAGR": cagr,
                    "Sharpe": sharpe,
                    "MDD": mdd,
                }
            )
        comparisons.append(
            {
                "mdd_better_than_hard": metrics["FinalBlend"][2] > metrics["Hard"][2],
                "sharpe_better_than_hard": metrics["FinalBlend"][1] > metrics["Hard"][1],
                "mdd_within_15": metrics["FinalBlend"][2] >= -0.15,
                "cagr_at_least_14": metrics["FinalBlend"][0] >= 0.14,
                "cagr_retention": metrics["FinalBlend"][0] / metrics["Hard"][0]
                if abs(metrics["Hard"][0]) > 1e-12
                else np.nan,
            }
        )
    raw = pd.DataFrame(records)
    summary_rows = []
    for (block, strategy), group in raw.groupby(["BlockLength", "Strategy"]):
        for metric in ["CAGR", "Sharpe", "MDD"]:
            summary_rows.append(
                {
                    "BlockLength": block,
                    "Strategy": strategy,
                    "Metric": metric,
                    "P05": float(group[metric].quantile(0.05)),
                    "Median": float(group[metric].median()),
                    "P95": float(group[metric].quantile(0.95)),
                }
            )
    comparison = pd.DataFrame(comparisons)
    probabilities = {
        f"block{block_length}_{col}": float(comparison[col].mean())
        for col in ["mdd_better_than_hard", "sharpe_better_than_hard", "mdd_within_15", "cagr_at_least_14"]
    }
    probabilities[f"block{block_length}_median_cagr_retention"] = float(comparison["cagr_retention"].median())
    return pd.DataFrame(summary_rows), probabilities


macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
hard = run_backtest(asset_returns, signals, StrategyConfig(), mode="hard")
defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")

# Central point next to the calibration frontier.  It uses only the four
# original assets and borrows 20% at an explicit 4% annual financing rate.
FINAL_CONFIG = BlendConfig(hard_fraction=0.40, leverage=1.20, financing_rate=0.04)
final = run_blend(asset_returns, signals, defensive, FINAL_CONFIG)

comparison_rows = []
for period, start, end in [
    ("calibration", None, "2017-12"),
    ("locked_test", "2018-01", None),
    ("full", None, None),
]:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("FinalBlend", final)]:
        sample = backtest.loc[start:end] if start else backtest.loc[:end] if end else backtest
        comparison_rows.append({"Period": period, **metric_row(strategy, sample["return"]), "AvgTurnover": float(sample["turnover"].mean())})
comparison = pd.DataFrame(comparison_rows)

cost_rows = []
for multiplier in [1.0, 2.0, 3.0]:
    scenario = run_blend(asset_returns, signals, defensive, FINAL_CONFIG, cost_multiplier=multiplier)
    cost_rows.append({"CostMultiplier": multiplier, **performance_summary(scenario["return"]).to_dict()})
cost_sensitivity = pd.DataFrame(cost_rows)

financing_rows = []
for rate in [0.04, 0.06, 0.08, 0.10]:
    cfg = BlendConfig(0.40, 1.20, rate)
    scenario = run_blend(asset_returns, signals, defensive, cfg)
    financing_rows.append({"FinancingRate": rate, **performance_summary(scenario["return"]).to_dict()})
financing_sensitivity = pd.DataFrame(financing_rows)

neighborhood_rows = []
for hard_fraction in [0.35, 0.375, 0.40, 0.425, 0.45]:
    for leverage in [1.15, 1.20, 1.25]:
        cfg = BlendConfig(hard_fraction, leverage, 0.04)
        scenario = run_blend(asset_returns, signals, defensive, cfg)
        for period, sample in [("locked_test", scenario.loc["2018-01":]), ("full", scenario)]:
            neighborhood_rows.append(
                {
                    "Period": period,
                    "HardFraction": hard_fraction,
                    "Leverage": leverage,
                    **performance_summary(sample["return"]).to_dict(),
                }
            )
neighborhood = pd.DataFrame(neighborhood_rows)

subperiod_ranges = [
    ("2007-2009", "2007-01", "2009-12"),
    ("2010-2013", "2010-01", "2013-12"),
    ("2014-2017", "2014-01", "2017-12"),
    ("2018-2020", "2018-01", "2020-12"),
    ("2021-2023", "2021-01", "2023-12"),
    ("2024-current", "2024-01", None),
]
subperiod_rows = []
for label, start, end in subperiod_ranges:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("FinalBlend", final)]:
        sample = backtest.loc[start:end]
        subperiod_rows.append({"Subperiod": label, **metric_row(strategy, sample["return"])})
subperiods = pd.DataFrame(subperiod_rows)

stress_ranges = [
    ("GlobalFinancialCrisis", "2008-01", "2009-06"),
    ("EuroCommodityShock", "2011-05", "2012-06"),
    ("OilDowncycle", "2014-07", "2016-02"),
    ("2018Selloff", "2018-10", "2018-12"),
    ("COVID", "2020-02", "2020-05"),
    ("InflationShock", "2022-01", "2022-10"),
    ("Recent2026", "2026-01", None),
]
stress_rows = []
for label, start, end in stress_ranges:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("FinalBlend", final)]:
        sample = backtest.loc[start:end]
        stress_rows.append({"Episode": label, **metric_row(strategy, sample["return"])})
stress = pd.DataFrame(stress_rows)

rolling_rows = []
for end_idx in range(59, len(final)):
    window = final["return"].iloc[end_idx - 59 : end_idx + 1]
    rolling_rows.append({"EndMonth": str(window.index[-1]), **performance_summary(window).to_dict()})
rolling = pd.DataFrame(rolling_rows)
drawdowns = drawdown_episodes(final["return"])

bootstrap_parts = []
bootstrap_probabilities = {}
for block_length in [6, 12]:
    summary, probabilities = paired_block_bootstrap(
        {"Hard": hard["return"], "CurrentDefensive": defensive["return"], "FinalBlend": final["return"]},
        block_length=block_length,
    )
    bootstrap_parts.append(summary)
    bootstrap_probabilities.update(probabilities)
bootstrap = pd.concat(bootstrap_parts, ignore_index=True)

regime_metrics = json.loads((RESULTS / "regime_metrics.json").read_text(encoding="utf-8"))
full_metrics = performance_summary(final["return"])
locked_metrics = performance_summary(final.loc["2018-01":, "return"])
validation = {
    "config": asdict(FINAL_CONFIG),
    "full": {key: float(value) for key, value in full_metrics.items()},
    "locked_test": {key: float(value) for key, value in locked_metrics.items()},
    "regime_metrics": regime_metrics,
    "bootstrap_probabilities": bootstrap_probabilities,
    "gates": {
        "full_mdd_within_15": bool(full_metrics["MDD"] >= -0.15),
        "full_sharpe_at_least_1": bool(full_metrics["Sharpe"] >= 1.0),
        "locked_mdd_within_15": bool(locked_metrics["MDD"] >= -0.15),
        "locked_sharpe_at_least_1": bool(locked_metrics["Sharpe"] >= 1.0),
    },
}

final.to_csv(RESULTS / "final_blend_backtest.csv")
comparison.to_csv(RESULTS / "final_blend_comparison.csv", index=False)
cost_sensitivity.to_csv(RESULTS / "final_blend_cost_sensitivity.csv", index=False)
financing_sensitivity.to_csv(RESULTS / "final_blend_financing_sensitivity.csv", index=False)
neighborhood.to_csv(RESULTS / "final_blend_parameter_neighborhood.csv", index=False)
subperiods.to_csv(RESULTS / "final_blend_subperiods.csv", index=False)
stress.to_csv(RESULTS / "final_blend_stress_episodes.csv", index=False)
rolling.to_csv(RESULTS / "final_blend_rolling60.csv", index=False)
drawdowns.to_csv(RESULTS / "final_blend_drawdown_episodes.csv", index=False)
bootstrap.to_csv(RESULTS / "final_blend_bootstrap.csv", index=False)
(RESULTS / "final_blend_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

print("=== FINAL CENTRAL BLEND ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))
print("\n=== COST SENSITIVITY ===")
print(cost_sensitivity[["CostMultiplier", "CAGR", "Sharpe", "MDD"]].round(4).to_string(index=False))
print("\n=== FINANCING SENSITIVITY ===")
print(financing_sensitivity[["FinancingRate", "CAGR", "Sharpe", "MDD"]].round(4).to_string(index=False))
print("\n=== TOP DRAWDOWNS ===")
print(drawdowns.head(8).round(4).to_string(index=False))
print("\n=== VALIDATION GATES ===")
print(json.dumps(validation["gates"], ensure_ascii=False, indent=2))
print("\n=== BOOTSTRAP PROBABILITIES ===")
print(json.dumps(bootstrap_probabilities, ensure_ascii=False, indent=2))
