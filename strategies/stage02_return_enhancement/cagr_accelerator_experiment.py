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
    StrategyConfig,
    compute_regime_signals,
    controlled_weights,
    ewma_cov,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RISKY = ["KODEX200", "GLD", "USO"]
CAPS = {"KODEX200": 0.62, "GLD": 0.60, "USO": 0.22}


@dataclass(frozen=True)
class AcceleratorConfig:
    sleeve: float
    vol_cap: float
    trend_window: int
    top_k: int
    brake_fraction: float = 0.0
    early_guard_start: float = -0.02
    early_guard_strength: float = 0.0
    dd_gate: float = -0.025
    bond_floor: float = 0.18

    @property
    def name(self) -> str:
        return (
            f"sl{self.sleeve:.2f}_vc{self.vol_cap:.3f}_tw{self.trend_window}_k{self.top_k}"
            f"_bf{self.brake_fraction:.2f}_egs{abs(self.early_guard_start):.3f}_eg{self.early_guard_strength:.2f}"
        )


def add_accelerator(
    base: np.ndarray,
    history: pd.DataFrame,
    current_dd: float,
    acfg: AcceleratorConfig,
) -> tuple[np.ndarray, str, float, float]:
    if len(history) < max(24, acfg.trend_window):
        return base, "OFF_HISTORY", 0.0, 0.0

    trailing = history.tail(acfg.trend_window)[ASSETS]
    mom = (1 + trailing).prod() - 1
    mom1 = history.iloc[-1][ASSETS]
    mom3 = (1 + history.tail(3)[ASSETS]).prod() - 1
    ann_vol = trailing.std(ddof=1) * math.sqrt(12)
    score = mom / ann_vol.clip(0.04)

    # Symmetric trend brake: the accelerator earns the right to take more risk
    # only if existing equity/oil risk is cut when both medium and short trends
    # are negative.  Gold is retained as the crisis diversifier.
    working = base.copy()
    brake_used = 0.0
    mom6 = (1 + history.tail(6)[ASSETS]).prod() - 1
    for asset in ["KODEX200", "USO"]:
        idx = ASSETS.index(asset)
        if mom3[asset] < 0 and mom6[asset] < 0:
            cut = acfg.brake_fraction * working[idx]
            working[idx] -= cut
            working[ASSETS.index("BOND")] += cut
            brake_used += cut

    if current_dd <= acfg.dd_gate:
        return working / working.sum(), "BRAKE_DD" if brake_used else "OFF_DD", 0.0, brake_used

    eligible = [a for a in RISKY if mom[a] > 0 and mom3[a] > 0 and mom1[a] > 0 and score[a] > 0]
    if not eligible:
        return working / working.sum(), "BRAKE" if brake_used else "OFF_TREND", 0.0, brake_used

    selected = sorted(eligible, key=lambda a: score[a], reverse=True)[: acfg.top_k]
    raw_strength = float(np.clip(max(mom[a] for a in selected) / 0.12, 0.25, 1.0))
    available = max(float(working[ASSETS.index("BOND")]) - acfg.bond_floor, 0.0)
    dd_scale = float(np.clip((current_dd - acfg.dd_gate) / (0.0 - acfg.dd_gate), 0.0, 1.0))
    desired = min(acfg.sleeve * raw_strength * dd_scale, available)
    if desired <= 1e-8:
        return working / working.sum(), "BRAKE_FLOOR" if brake_used else "OFF_FLOOR", 0.0, brake_used

    positive_scores = np.array([max(float(score[a]), 1e-6) for a in selected])
    shares = positive_scores / positive_scores.sum()

    def candidate(amount: float) -> np.ndarray:
        w = working.copy()
        w[ASSETS.index("BOND")] -= amount
        unallocated = amount
        for asset, share in zip(selected, shares):
            idx = ASSETS.index(asset)
            add = min(amount * float(share), CAPS[asset] - w[idx])
            w[idx] += max(add, 0.0)
            unallocated -= max(add, 0.0)
        w[ASSETS.index("BOND")] += max(unallocated, 0.0)
        return w / w.sum()

    cov = ewma_cov(history.tail(84), half_life=12.0, leverage=1.0)
    proposed = candidate(desired)
    proposed_vol = math.sqrt(max(float(proposed @ cov @ proposed), 0.0) * 12)
    if proposed_vol > acfg.vol_cap:
        lo, hi = 0.0, desired
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            w_mid = candidate(mid)
            vol_mid = math.sqrt(max(float(w_mid @ cov @ w_mid), 0.0) * 12)
            if vol_mid <= acfg.vol_cap:
                lo = mid
            else:
                hi = mid
        desired = lo
        proposed = candidate(desired)

    if desired <= 1e-5:
        return working / working.sum(), "BRAKE_VOL" if brake_used else "OFF_VOL", 0.0, brake_used
    label = "+".join(selected) + ("|BRAKE" if brake_used else "")
    return proposed, label, desired, brake_used


def run_accelerated(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    base_cfg: StrategyConfig,
    acfg: AcceleratorConfig,
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
        history = returns.loc[returns.index < month]
        current_dd = nav / peak - 1.0
        base = controlled_weights(signal, history, pretrade, current_dd, base_cfg)
        w, accelerator_asset, sleeve_used, brake_used = add_accelerator(base, history, current_dd, acfg)

        early_guard_used = 0.0
        if current_dd < acfg.early_guard_start and acfg.early_guard_strength > 0:
            severity = float(np.clip((acfg.early_guard_start - current_dd) / 0.05, 0.0, 1.0))
            early_guard_used = acfg.early_guard_strength * (0.30 + 0.70 * severity)
            w = (1 - early_guard_used) * w + early_guard_used * np.array([0.05, 0.72, 0.23, 0.00])
            w /= w.sum()

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
            "accelerator_asset": accelerator_asset,
            "sleeve_used": sleeve_used,
            "brake_used": brake_used,
            "early_guard_used": early_guard_used,
            **{f"w_{asset}": w[i] for i, asset in enumerate(ASSETS)},
            **{f"base_w_{asset}": base[i] for i, asset in enumerate(ASSETS)},
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
baseline_cal = run_backtest(asset_returns, signals, base_cfg, end=cal_end)
baseline_cal_metrics = performance_summary(baseline_cal["return"])
mdd_floor = float(baseline_cal_metrics["MDD"])

candidates = [
    # Previously selected strict-MDD candidate remains in the comparison set.
    AcceleratorConfig(
        sleeve=0.10,
        vol_cap=0.10,
        trend_window=12,
        top_k=2,
        brake_fraction=0.25,
        early_guard_start=-0.015,
        early_guard_strength=0.35,
    )
] + [
    AcceleratorConfig(
        sleeve=sleeve,
        vol_cap=vol_cap,
        trend_window=12,
        top_k=2,
        brake_fraction=brake_fraction,
        early_guard_start=early_guard_start,
        early_guard_strength=early_guard_strength,
    )
    for sleeve, vol_cap, brake_fraction, early_guard_start, early_guard_strength in itertools.product(
        [0.15, 0.20],
        [0.10, 0.11],
        [0.25, 0.50],
        [-0.010, -0.015, -0.020],
        [0.50, 0.65],
    )
]

rows = []
for i, acfg in enumerate(candidates, start=1):
    bt = run_accelerated(asset_returns, signals, base_cfg, acfg, end=cal_end)
    m = performance_summary(bt["return"])
    rows.append({
        "name": acfg.name,
        **asdict(acfg),
        **m.to_dict(),
        "AvgTurnover": float(bt["turnover"].mean()),
        "AvgSleeve": float(bt["sleeve_used"].mean()),
        "AvgBrake": float(bt["brake_used"].mean()),
        "AvgEarlyGuard": float(bt["early_guard_used"].mean()),
        "MDDPass": bool(m["MDD"] >= float(baseline_cal_metrics["MDD"])),
        "SharpePass": bool(m["Sharpe"] >= 1.0),
    })
    if i % 12 == 0:
        print(f"calibrated {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "cagr_accelerator_calibration.csv", index=False)
eligible = ranking[ranking["MDDPass"] & ranking["SharpePass"]].copy()
if eligible.empty:
    print("\n=== NO CANDIDATE PASSED THE PRE-DECLARED GATES ===")
    print("\nTop CAGR candidates")
    print(ranking.sort_values("CAGR", ascending=False)[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgSleeve", "AvgBrake", "AvgEarlyGuard"]].head(15).round(4).to_string(index=False))
    print("\nClosest MDD candidates")
    print(ranking.sort_values(["MDD", "CAGR"], ascending=[False, False])[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgSleeve", "AvgBrake", "AvgEarlyGuard"]].head(15).round(4).to_string(index=False))
    raise SystemExit(2)
eligible = eligible.sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
winner_row = eligible.iloc[0]
winner = next(c for c in candidates if c.name == winner_row["name"])

print("\n=== BASELINE CALIBRATION ===")
print(baseline_cal_metrics.round(4).to_string())
print("MDD floor", round(mdd_floor, 4))
print("\n=== ELIGIBLE TOP 15 (CALIBRATION ONLY) ===")
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgSleeve", "AvgBrake", "AvgEarlyGuard"]].head(15).round(4).to_string(index=False))
print("\n=== LOCKED WINNER ===")
print(winner)

periods = [
    ("calibration", None, cal_end),
    ("locked_test", test_start, None),
    ("full", None, None),
]
comparison_rows = []
for label, start, end in periods:
    base_bt = run_backtest(asset_returns, signals, base_cfg, start=start, end=end)
    acc_bt = run_accelerated(asset_returns, signals, base_cfg, winner, start=start, end=end)
    for strategy, bt in [("Baseline", base_bt), ("Accelerated", acc_bt)]:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})

comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINE VS WINNER ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_full = run_accelerated(asset_returns, signals, base_cfg, winner)
print("\n=== ACCELERATOR USAGE ===")
print(winner_full["accelerator_asset"].value_counts().to_string())
print("Avg sleeve", round(float(winner_full["sleeve_used"].mean()), 4), "Max sleeve", round(float(winner_full["sleeve_used"].max()), 4))
print("Avg brake", round(float(winner_full["brake_used"].mean()), 4), "Max brake", round(float(winner_full["brake_used"].max()), 4))
print("Avg early guard", round(float(winner_full["early_guard_used"].mean()), 4), "Max early guard", round(float(winner_full["early_guard_used"].max()), 4))

comparison.to_csv(RESULTS / "cagr_accelerator_comparison.csv", index=False)
winner_full.to_csv(RESULTS / "cagr_accelerator_backtest.csv")
with (RESULTS / "cagr_accelerator_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)
