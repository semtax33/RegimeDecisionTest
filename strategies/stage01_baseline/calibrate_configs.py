from __future__ import annotations

import itertools
from dataclasses import asdict

import pandas as pd

from strategies.core.regime_research import (
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


def score_metrics(metrics: pd.Series, turnover: float) -> float:
    # Pre-declared validation objective: risk-adjusted return and path risk,
    # with a strong penalty for breaching the user's 15% preferred MDD line.
    breach = max(abs(float(metrics["MDD"])) - 0.15, 0.0)
    return float(metrics["Sharpe"] + 0.35 * metrics["Calmar"] - 4.0 * breach - 0.20 * turnover)


features, _ = load_macro_data()
returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, returns)

rows = []
configs = []
for regime_strength, target_vol, invvol_tilt, drawdown_guard in itertools.product(
    [0.25, 0.50, 0.75],
    [0.08, 0.10, 0.12],
    [0.15, 0.35],
    [0.40, 0.75],
):
    cfg = StrategyConfig(
        name=f"rs{regime_strength}_tv{target_vol}_iv{invvol_tilt}_dg{drawdown_guard}",
        regime_strength=regime_strength,
        target_vol=target_vol,
        invvol_tilt=invvol_tilt,
        drawdown_guard=drawdown_guard,
        return_reward=1.15,
        vol_penalty=0.18,
        cdar_penalty=0.25,
        turnover_penalty=0.05,
        tracking_penalty=0.32,
        max_cdar=0.16,
    )
    bt = run_backtest(returns, signals, cfg, mode="proposed", end="2017-12")
    metrics = performance_summary(bt["return"])
    avg_turnover = float(bt["turnover"].mean())
    rows.append({"name": cfg.name, **asdict(cfg), **metrics.to_dict(), "AvgTurnover": avg_turnover, "ValidationScore": score_metrics(metrics, avg_turnover)})
    configs.append(cfg)

rank = pd.DataFrame(rows).sort_values("ValidationScore", ascending=False).reset_index(drop=True)
rank.to_csv("results/calibration_grid.csv", index=False)
print(rank[["name", "Sharpe", "MDD", "Calmar", "CAGR", "AvgTurnover", "ValidationScore"]].head(12).round(4).to_string(index=False))

winner_name = rank.iloc[0]["name"]
winner = next(c for c in configs if c.name == winner_name)
print("\nLOCKED WINNER", asdict(winner))
for label, start, end in [
    ("calibration", None, "2017-12"),
    ("locked_test", "2018-01", None),
    ("full", None, None),
]:
    bt = run_backtest(returns, signals, winner, mode="proposed", start=start, end=end)
    m = performance_summary(bt["return"])
    print(f"\n{label}\n{m.round(4).to_string()}")

