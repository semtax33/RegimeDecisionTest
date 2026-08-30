from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    compute_regime_signals,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", "M")


@dataclass(frozen=True)
class BlendConfig:
    hard_fraction: float
    leverage: float
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return f"blend_h{self.hard_fraction:.3f}_lev{self.leverage:.2f}"


def run_blend(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive_path: pd.DataFrame,
    cfg: BlendConfig,
    start: str | None = None,
    end: str | None = None,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index).intersection(defensive_path.index)
    if start:
        months = months[months >= pd.Period(start, "M")]
    if end:
        months = months[months <= pd.Period(end, "M")]

    rows = []
    pretrade_assets = np.zeros(len(ASSETS))
    pretrade_debt = 0.0
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        defensive = defensive_path.loc[month, [f"w_{asset}" for asset in ASSETS]].to_numpy(dtype=float)
        unlevered = cfg.hard_fraction * hard + (1.0 - cfg.hard_fraction) * defensive
        asset_weights = cfg.leverage * unlevered
        debt_weight = 1.0 - cfg.leverage

        asset_delta = asset_weights - pretrade_assets
        turnover = np.abs(asset_delta).sum() if first_trade else 0.5 * np.abs(asset_delta).sum()
        trade_cost = np.abs(asset_delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs(
            (asset_weights[2] + asset_weights[3]) - (pretrade_assets[2] + pretrade_assets[3])
        ) * 0.0005 * cost_multiplier
        financing = debt_weight * ((1.0 + cfg.financing_rate) ** (1.0 / 12.0) - 1.0)
        asset_r = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(asset_weights @ asset_r + financing)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)

        pretrade_assets = asset_weights * (1.0 + asset_r) / (1.0 + gross_return)
        pretrade_debt = debt_weight * (1.0 + cfg.financing_rate) ** (1.0 / 12.0) / (1.0 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "financing_return": financing,
                "regime": signals.loc[month, "regime"],
                "hard_fraction": cfg.hard_fraction,
                "leverage": cfg.leverage,
                "debt_weight": debt_weight,
                **{f"w_{asset}": asset_weights[i] for i, asset in enumerate(ASSETS)},
                **{f"hard_w_{asset}": hard[i] for i, asset in enumerate(ASSETS)},
                **{f"defensive_w_{asset}": defensive[i] for i, asset in enumerate(ASSETS)},
            }
        )
    return pd.DataFrame(rows).set_index("month")


macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
base_cfg = StrategyConfig()
hard = run_backtest(asset_returns, signals, base_cfg, mode="hard")
defensive = run_backtest(asset_returns, signals, base_cfg, mode="proposed")

candidates = [
    BlendConfig(hard_fraction, leverage)
    for hard_fraction, leverage in itertools.product(
        np.round(np.arange(0.0, 1.0001, 0.025), 3),
        np.round(np.arange(1.0, 1.3001, 0.05), 2),
    )
]

rows = []
for candidate in candidates:
    backtest = run_blend(asset_returns, signals, defensive, candidate, end=str(CAL_END))
    metrics = performance_summary(backtest["return"])
    rows.append(
        {
            "name": candidate.name,
            **asdict(candidate),
            **metrics.to_dict(),
            "AvgTurnover": float(backtest["turnover"].mean()),
            "MDD15Pass": bool(metrics["MDD"] >= -0.15),
        }
    )

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "blend_leverage_calibration.csv", index=False)
eligible = ranking.loc[(ranking["MDD15Pass"]) & (ranking["Sharpe"] >= 1.0)]
eligible = eligible.sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
print("=== CALIBRATION MDD15 ELIGIBLE ===", len(eligible))
print(
    eligible[["name", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
    .head(25).round(4).to_string(index=False)
)
if eligible.empty:
    raise SystemExit(2)

winner_row = eligible.iloc[0]
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
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("BlendLeverage", winner_full)]:
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
print("\n=== PREDECLARED WINNER / LOCKED TEST ===")
print(winner)
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
    ].round(4).to_string(index=False)
)

winner_full.to_csv(RESULTS / "blend_leverage_backtest.csv")
comparison.to_csv(RESULTS / "blend_leverage_comparison.csv", index=False)
with (RESULTS / "blend_leverage_winner.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(winner), handle, ensure_ascii=False, indent=2)
