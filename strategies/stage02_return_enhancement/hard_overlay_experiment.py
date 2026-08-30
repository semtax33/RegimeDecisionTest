from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    DEFENSIVE,
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


@dataclass(frozen=True)
class OverlayConfig:
    short_window: int
    long_window: int
    trend_rule: str
    protect_fraction: float
    vol_cap: float
    dd_start: float
    skew_limit: float = -0.75
    corr_limit: float = 0.60
    use_shape_flags: bool = True

    @property
    def name(self) -> str:
        return (
            f"s{self.short_window}_l{self.long_window}_{self.trend_rule}"
            f"_p{self.protect_fraction:.2f}_v{self.vol_cap:.2f}_dd{abs(self.dd_start):.2f}"
            f"_shape{int(self.use_shape_flags)}"
        )


def overlay_weights(
    base: np.ndarray,
    history: pd.DataFrame,
    current_dd: float,
    cfg: OverlayConfig,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if len(history) < max(24, cfg.long_window):
        return base, {"protection": 0.0, "trend_bad": False, "vol": np.nan, "skew": np.nan, "corr": np.nan, "shape_flags": 0}

    h = history[ASSETS]
    proxy = h.to_numpy() @ base
    short_mom = float(np.prod(1 + proxy[-cfg.short_window:]) - 1)
    long_mom = float(np.prod(1 + proxy[-cfg.long_window:]) - 1)
    if cfg.trend_rule == "both":
        trend_bad = short_mom < 0 and long_mom < 0
    elif cfg.trend_rule == "either":
        trend_bad = short_mom < 0 or long_mom < 0
    else:
        raise ValueError(cfg.trend_rule)

    vol_window = min(6, len(proxy))
    ann_vol = float(np.std(proxy[-vol_window:], ddof=1) * math.sqrt(12))
    vol_protection = float(np.clip(1 - cfg.vol_cap / max(ann_vol, 1e-9), 0.0, 1.0))

    shape_window = h.tail(12)
    selected = [i for i, w in enumerate(base) if w > 0.05 and ASSETS[i] != "BOND"]
    if selected:
        selected_proxy = shape_window.to_numpy()[:, selected] @ (base[selected] / base[selected].sum())
        skew = float(pd.Series(selected_proxy).skew())
    else:
        skew = 0.0
    risky_corr = shape_window[["KODEX200", "GLD", "USO"]].corr()
    corr_values = risky_corr.to_numpy()[np.triu_indices(3, 1)]
    mean_corr = float(np.nanmean(corr_values))
    shape_flags = int(skew < cfg.skew_limit) + int(mean_corr > cfg.corr_limit)

    protection = cfg.protect_fraction if trend_bad else 0.0
    protection = max(protection, vol_protection)
    if cfg.use_shape_flags and shape_flags >= 2:
        protection = max(protection, cfg.protect_fraction)
    if current_dd < cfg.dd_start:
        severity = float(np.clip((cfg.dd_start - current_dd) / 0.06, 0.0, 1.0))
        protection = max(protection, 0.55 + 0.45 * severity)

    w = (1 - protection) * base + protection * DEFENSIVE
    w = np.clip(w, 0.0, None)
    w /= w.sum()
    return w, {
        "protection": protection,
        "trend_bad": trend_bad,
        "short_mom": short_mom,
        "long_mom": long_mom,
        "vol": ann_vol,
        "skew": skew,
        "corr": mean_corr,
        "shape_flags": shape_flags,
    }


def run_overlay(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: OverlayConfig,
    start: str | None = None,
    end: str | None = None,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index)
    if start:
        months = months[months >= pd.Period(start, "M")]
    if end:
        months = months[months <= pd.Period(end, "M")]

    rows = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        signal = signals.loc[month]
        base = hard_regime_weights(signal)
        history = returns.loc[returns.index < month]
        current_dd = nav / peak - 1.0
        w, diag = overlay_weights(base, history, current_dd, cfg)

        delta = w - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((w[2] + w[3]) - (pretrade[2] + pretrade[3])) * 0.0005 * cost_multiplier
        asset_r = returns.loc[month, ASSETS].to_numpy()
        gross_return = float(w @ asset_r)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        end_w = w * (1 + asset_r) / (1 + gross_return)
        rows.append({
            "month": month,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            "regime": signal["regime"],
            **diag,
            **{f"w_{asset}": w[i] for i, asset in enumerate(ASSETS)},
            **{f"hard_w_{asset}": base[i] for i, asset in enumerate(ASSETS)},
        })
        pretrade = end_w
        first_trade = False
    return pd.DataFrame(rows).set_index("month")


features, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, asset_returns)
base_cfg = StrategyConfig()

cal_end = "2017-12"
test_start = "2018-01"
hard_cal = run_backtest(asset_returns, signals, base_cfg, mode="hard", end=cal_end)
defensive_cal = run_backtest(asset_returns, signals, base_cfg, mode="proposed", end=cal_end)
hard_cal_m = performance_summary(hard_cal["return"])
defensive_cal_m = performance_summary(defensive_cal["return"])

candidates = [
    OverlayConfig(s, l, rule, protect, vol_cap, dd_start, use_shape_flags=shape)
    for s, l, rule, protect, vol_cap, dd_start, shape in itertools.product(
        [1, 2, 3],
        [6, 9, 12],
        ["both", "either"],
        [0.50, 0.75, 1.00],
        [0.18, 0.22, 0.26, 0.30],
        [-0.03, -0.05],
        [False, True],
    )
]

rows = []
for i, candidate in enumerate(candidates, start=1):
    bt = run_overlay(asset_returns, signals, candidate, end=cal_end)
    m = performance_summary(bt["return"])
    rows.append({
        "name": candidate.name,
        **asdict(candidate),
        **m.to_dict(),
        "AvgTurnover": float(bt["turnover"].mean()),
        "AvgProtection": float(bt["protection"].mean()),
        "MDD9Pass": bool(m["MDD"] >= -0.09),
        "MDD10Pass": bool(m["MDD"] >= -0.10),
        "ReturnRetention": float(m["CAGR"] / hard_cal_m["CAGR"]),
    })
    if i % 200 == 0:
        print(f"calibrated {i}/{len(candidates)}")

ranking = pd.DataFrame(rows).sort_values(["MDD", "CAGR"], ascending=[False, False])
ranking.to_csv(RESULTS / "hard_overlay_calibration.csv", index=False)
eligible = ranking[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)

print("\n=== CALIBRATION BASELINES ===")
print("Hard", hard_cal_m.round(4).to_dict())
print("Defensive", defensive_cal_m.round(4).to_dict())
print("\n=== MDD <= 9% ELIGIBLE ===", len(eligible))
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection", "ReturnRetention"]].head(20).round(4).to_string(index=False))
print("\n=== BEST CAGR WITH MDD <= 10% ===")
print(ranking[ranking["MDD10Pass"]][["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgProtection", "ReturnRetention"]].sort_values("CAGR", ascending=False).head(20).round(4).to_string(index=False))

if eligible.empty:
    print("\nNo candidate meets the strict 9% calibration MDD gate.")
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = next(c for c in candidates if c.name == winner_row["name"])
print("\n=== LOCKED WINNER ===")
print(winner)

comparison_rows = []
for label, start, end in [("calibration", None, cal_end), ("locked_test", test_start, None), ("full", None, None)]:
    tests = [
        ("Hard", run_backtest(asset_returns, signals, base_cfg, mode="hard", start=start, end=end)),
        ("CurrentDefensive", run_backtest(asset_returns, signals, base_cfg, mode="proposed", start=start, end=end)),
        ("HardOverlay", run_overlay(asset_returns, signals, winner, start=start, end=end)),
    ]
    for strategy, bt in tests:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})

comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINE VS OVERLAY ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_full = run_overlay(asset_returns, signals, winner)
winner_full.to_csv(RESULTS / "hard_overlay_backtest.csv")
comparison.to_csv(RESULTS / "hard_overlay_comparison.csv", index=False)
with (RESULTS / "hard_overlay_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)

