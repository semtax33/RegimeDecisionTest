from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Reuse only the function/type definitions above the experiment's top-level
# execution marker.  This avoids rerunning its grid when importing the helpers.
common_source = (ROOT / "leveraged_daily_overlay_experiment.py").read_text(encoding="utf-8")
common_source = common_source.split("\ndaily_levels = load_daily_open_levels()")[0]
exec(compile(common_source, str(ROOT / "leveraged_daily_overlay_experiment.py"), "exec"), globals())


@dataclass(frozen=True)
class StopConfig:
    max_exposure: float
    target_vol: float
    sleeve_stop: float
    safe_exposure: float
    cooldown_days: int
    global_stop: float
    global_cooldown_days: int = 10
    vix_threshold: float = 35.0
    vix_multiplier: float = 0.50
    rebalance_band: float = 0.10
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return (
            f"mx{self.max_exposure:.2f}_tv{self.target_vol:.2f}_ss{self.sleeve_stop:.2f}"
            f"_se{self.safe_exposure:.2f}_cd{self.cooldown_days}_gs{self.global_stop:.2f}"
        )


def simulate_stop(
    arrays: dict[str, object],
    cfg: StopConfig,
    start: str | None = None,
    end: str | None = None,
    cost_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates: pd.DatetimeIndex = arrays["dates"]  # type: ignore[assignment]
    months: pd.PeriodIndex = arrays["months"]  # type: ignore[assignment]
    returns: np.ndarray = arrays["returns"]  # type: ignore[assignment]
    bases: np.ndarray = arrays["bases"]  # type: ignore[assignment]
    vols: np.ndarray = arrays["forecast_vol"]  # type: ignore[assignment]
    vix: np.ndarray = arrays["vix"]  # type: ignore[assignment]

    keep = np.ones(len(dates), dtype=bool)
    if start:
        keep &= months >= pd.Period(start, "M")
    if end:
        keep &= months <= pd.Period(end, "M")
    indices = np.flatnonzero(keep)

    nav = 1.0
    peak = 1.0
    pretrade_assets = np.zeros(len(ASSETS))
    pretrade_debt = 0.0
    current_exposure = 0.0
    previous_month: pd.Period | None = None
    sleeve_wealth = 1.0
    cooldown = 0
    global_cooldown = 0
    first_trade = True
    rf_daily = (1 + cfg.financing_rate) ** (1 / 252) - 1
    rows = []

    for j in indices:
        month = months[j]
        base = bases[j]
        month_boundary = previous_month is None or month != previous_month
        if month_boundary:
            sleeve_wealth = 1.0
            cooldown = 0

        base_daily_return = float(base @ returns[j])
        if cooldown == 0:
            sleeve_wealth *= 1 + base_daily_return

        ann_vol = vols[j]
        normal_exposure = min(cfg.max_exposure, cfg.target_vol / max(ann_vol, 1e-6))
        if vix[j] >= cfg.vix_threshold:
            normal_exposure = min(normal_exposure, cfg.max_exposure * cfg.vix_multiplier)

        stopped = cooldown > 0 or global_cooldown > 0
        desired = min(cfg.safe_exposure, normal_exposure) if stopped else normal_exposure
        desired = float(np.clip(desired, 0.0, cfg.max_exposure))
        event_rebalance = month_boundary or stopped or abs(desired - current_exposure) >= cfg.rebalance_band

        if event_rebalance:
            if desired <= 1.0:
                asset_w = desired * base
                asset_w[ASSETS.index("BOND")] += 1 - desired
                debt_w = 0.0
            else:
                asset_w = desired * base
                debt_w = 1 - desired
            current_exposure = desired
        else:
            asset_w = pretrade_assets.copy()
            debt_w = pretrade_debt

        delta = asset_w - pretrade_assets
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((asset_w[2] + asset_w[3]) - (pretrade_assets[2] + pretrade_assets[3])) * 0.0005 * cost_multiplier
        gross_return = float(asset_w @ returns[j] + debt_w * rf_daily)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade_assets = asset_w * (1 + returns[j]) / (1 + net_return)
        pretrade_debt = debt_w * (1 + rf_daily) / (1 + net_return)

        sleeve_trigger = cooldown == 0 and sleeve_wealth - 1 <= -cfg.sleeve_stop
        global_trigger = global_cooldown == 0 and nav / peak - 1 <= -cfg.global_stop
        if sleeve_trigger:
            cooldown = cfg.cooldown_days
        elif cooldown > 0:
            cooldown -= 1
            if cooldown == 0:
                sleeve_wealth = 1.0
        if global_trigger:
            global_cooldown = cfg.global_cooldown_days
        elif global_cooldown > 0:
            global_cooldown -= 1

        rows.append({
            "date": dates[j], "month": month, "return": net_return, "gross_return": gross_return,
            "nav": nav, "drawdown": nav / peak - 1, "turnover": turnover,
            "trade_cost": trade_cost, "fx_cost": fx_cost, "exposure": current_exposure,
            "forecast_vol": ann_vol, "vix": vix[j], "sleeve_wealth": sleeve_wealth,
            "cooldown": cooldown, "global_cooldown": global_cooldown, "debt_weight": debt_w,
            **{f"w_{a}": asset_w[k] for k, a in enumerate(ASSETS)},
        })
        previous_month = month
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda x: float(np.prod(1 + x))),
        gross_factor=("gross_return", lambda x: float(np.prod(1 + x))),
        turnover=("turnover", "sum"), trade_cost=("trade_cost", "sum"), fx_cost=("fx_cost", "sum"),
        avg_exposure=("exposure", "mean"), max_exposure=("exposure", "max"), min_exposure=("exposure", "min"),
        stopped_days=("cooldown", lambda x: int((x > 0).sum())),
    )
    monthly["return"] = monthly["return_factor"] - 1
    monthly["gross_return"] = monthly["gross_factor"] - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    return daily, monthly


daily_levels = load_daily_open_levels()
macro, _ = load_macro_data()
monthly_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, monthly_returns)
stress_raw = pd.read_csv(ROOT / "cache" / "stress_raw.csv", parse_dates=["date"]).set_index("date")
vix_daily = pd.to_numeric(stress_raw["VIX"], errors="coerce").dropna().sort_index()
arrays = prepare_daily_arrays(daily_levels, signals, vix_daily)

hard = run_backtest(monthly_returns, signals, StrategyConfig(), mode="hard")
defensive = run_backtest(monthly_returns, signals, StrategyConfig(), mode="proposed")
hard_cal_m = performance_summary(hard.loc[:"2017-12", "return"])

candidates = [
    StopConfig(mx, tv, stop, safe, cooldown, global_stop)
    for mx, tv, stop, safe, cooldown, global_stop in itertools.product(
        [1.00, 1.25, 1.50, 1.75, 2.00],
        [0.25, 0.30, 0.35, 0.40],
        [0.04, 0.06, 0.08],
        [0.00, 0.25],
        [5, 10, 20],
        [0.08, 0.09, 0.10],
    )
]

rows = []
for i, candidate in enumerate(candidates, start=1):
    _, monthly = simulate_stop(arrays, candidate, end="2017-12")
    m = performance_summary(monthly["return"])
    rows.append({
        "name": candidate.name, **asdict(candidate), **m.to_dict(),
        "AvgTurnover": float(monthly["turnover"].mean()), "AvgExposure": float(monthly["avg_exposure"].mean()),
        "StoppedDays": int(monthly["stopped_days"].sum()), "MDD9Pass": bool(m["MDD"] >= -0.09),
        "CAGRRetention": float(m["CAGR"] / hard_cal_m["CAGR"]),
    })
    if i % 500 == 0:
        print(f"tested {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "daily_stoploss_calibration.csv", index=False)
eligible = ranking[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
print("=== STRICT 9% MDD ELIGIBLE ===", len(eligible))
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgExposure", "StoppedDays", "CAGRRetention"]].head(30).round(4).to_string(index=False))
if eligible.empty:
    raise SystemExit(2)

winner_row = eligible.iloc[0]
field_types = {"cooldown_days": int, "global_cooldown_days": int}
winner_kwargs = {
    field: field_types.get(field, float)(winner_row[field])
    for field in StopConfig.__dataclass_fields__
}
winner = StopConfig(**winner_kwargs)
print("\n=== LOCKED WINNER ===")
print(winner)
winner_daily, winner_monthly = simulate_stop(arrays, winner)

comparison_rows = []
for label, start, end in [("calibration", None, "2017-12"), ("locked_test", "2018-01", None), ("full", None, None)]:
    for strategy, bt in [
        ("Hard", hard.loc[start:end] if start else hard.loc[:end] if end else hard),
        ("CurrentDefensive", defensive.loc[start:end] if start else defensive.loc[:end] if end else defensive),
        ("StopProtectedHard", winner_monthly.loc[start:end] if start else winner_monthly.loc[:end] if end else winner_monthly),
    ]:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})
comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINES VS WINNER ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_daily.to_csv(RESULTS / "daily_stoploss_backtest.csv")
winner_monthly.to_csv(RESULTS / "daily_stoploss_monthly.csv")
comparison.to_csv(RESULTS / "daily_stoploss_comparison.csv", index=False)
with (RESULTS / "daily_stoploss_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)

