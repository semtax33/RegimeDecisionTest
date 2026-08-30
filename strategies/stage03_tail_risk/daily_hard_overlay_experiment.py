from __future__ import annotations

import itertools
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    DEFENSIVE,
    RAW_DIR,
    StrategyConfig,
    compute_regime_signals,
    get_path,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load_daily_open_levels() -> pd.DataFrame:
    market = pd.read_csv(ROOT / "cache" / "market_daily.csv", parse_dates=["date"])

    with sqlite3.connect(get_path(RAW_DIR, "compass.db")) as con:
        proxy = pd.read_sql(
            "select date, open, close from etf_prices where symbol = ? order by date",
            con,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy[["open", "close"]] = proxy[["open", "close"]].apply(pd.to_numeric, errors="coerce")

    actual = market[market["symbol"] == "KODEX200"].dropna(subset=["open"]).copy()
    actual = actual[actual["date"] > pd.Timestamp("2009-03-31")]
    first_actual = actual["date"].min()
    actual_anchor = float(actual.loc[actual["date"] == first_actual, "open"].iloc[0])
    nearest = proxy.iloc[(proxy["date"] - first_actual).abs().argsort()[:1]]
    scale = actual_anchor / float(nearest["open"].iloc[0])
    proxy["open"] *= scale
    proxy = proxy[proxy["date"] < first_actual]
    kodex = pd.concat([proxy[["date", "open"]], actual[["date", "open"]]], ignore_index=True)
    kodex = kodex.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["open"]

    bond = pd.read_csv(get_path(RAW_DIR, "krx_bond_index.csv"), encoding="cp949")
    bond.index = pd.to_datetime(bond.iloc[:, 0])
    bond_level = pd.to_numeric(bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False), errors="coerce")
    bond_level.name = "BOND"

    pivot_open = market.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    pivot_close = market.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    fx = pivot_close["USDKRW"].reindex(pd.date_range(pivot_close.index.min(), pivot_close.index.max(), freq="D")).ffill()
    gld = pivot_open["GLD"] * fx.reindex(pivot_open.index).to_numpy()
    uso = pivot_open["USO"] * fx.reindex(pivot_open.index).to_numpy()

    levels = pd.concat([kodex.rename("KODEX200"), bond_level, gld.rename("GLD"), uso.rename("USO")], axis=1).sort_index()
    calendar = pd.date_range(levels.index.min(), levels.index.max(), freq="B")
    return levels.reindex(calendar).ffill(limit=5)[ASSETS]


@dataclass(frozen=True)
class DailyOverlayConfig:
    target_vol: float
    fast_window: int
    slow_window: int
    trend_cut: float
    vix_threshold: float
    vix_cut: float
    dd_floor: float
    dd_strength: float
    rebalance_band: float
    fallback: str

    @property
    def name(self) -> str:
        return (
            f"tv{self.target_vol:.2f}_f{self.fast_window}_s{self.slow_window}_tc{self.trend_cut:.2f}"
            f"_vx{self.vix_threshold:.0f}_vc{self.vix_cut:.2f}_fl{abs(self.dd_floor):.2f}"
            f"_ds{self.dd_strength:.2f}_rb{self.rebalance_band:.2f}_{self.fallback}"
        )


def run_daily_overlay(
    daily_levels: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: DailyOverlayConfig,
    vix_daily: pd.Series,
    start: str | None = None,
    end: str | None = None,
    cost_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forward_returns = daily_levels.shift(-1).div(daily_levels).sub(1.0)
    first_month = signals.index.min()
    last_month = signals.index.max()
    dates = daily_levels.index[
        (daily_levels.index.to_period("M") >= first_month)
        & (daily_levels.index.to_period("M") <= last_month)
    ]
    if start:
        dates = dates[dates.to_period("M") >= pd.Period(start, "M")]
    if end:
        dates = dates[dates.to_period("M") <= pd.Period(end, "M")]

    fallback = np.array([0.0, 1.0, 0.0, 0.0]) if cfg.fallback == "bond" else DEFENSIVE.copy()
    rows = []
    pretrade = np.zeros(len(ASSETS))
    current_exposure = 0.0
    previous_month: pd.Period | None = None
    nav = 1.0
    peak = 1.0
    first_trade = True

    hist_returns = daily_levels.pct_change(fill_method=None)
    for date in dates:
        month = date.to_period("M")
        if month not in signals.index or date not in forward_returns.index:
            continue
        asset_r = forward_returns.loc[date, ASSETS].to_numpy(dtype=float)
        if not np.isfinite(asset_r).all():
            continue

        base = hard_regime_weights(signals.loc[month])
        history = hist_returns.loc[hist_returns.index < date, ASSETS].tail(max(cfg.slow_window, 40)).fillna(0.0)
        proxy = history.to_numpy() @ base
        if len(proxy) >= 10:
            ann_vol = float(np.std(proxy[-20:], ddof=1) * math.sqrt(252))
            vol_exposure = min(1.0, cfg.target_vol / max(ann_vol, 1e-6)) if cfg.target_vol < 10 else 1.0
            fast_mom = float(np.prod(1 + proxy[-cfg.fast_window:]) - 1)
            slow_mom = float(np.prod(1 + proxy[-cfg.slow_window:]) - 1)
        else:
            ann_vol, fast_mom, slow_mom, vol_exposure = np.nan, 0.0, 0.0, 1.0
        trend_exposure = 1.0 - cfg.trend_cut if fast_mom < 0 and slow_mom < 0 else 1.0

        vix_value = float(vix_daily.asof(date - pd.Timedelta(days=1))) if len(vix_daily.loc[: date - pd.Timedelta(days=1)]) else np.nan
        vix_exposure = 1.0 - cfg.vix_cut if np.isfinite(vix_value) and vix_value >= cfg.vix_threshold else 1.0

        current_dd = nav / peak - 1.0
        cushion = float(np.clip((current_dd - cfg.dd_floor) / (0.0 - cfg.dd_floor), 0.0, 1.0))
        dd_exposure = (1.0 - cfg.dd_strength) + cfg.dd_strength * cushion
        desired_exposure = float(np.clip(min(vol_exposure, trend_exposure, vix_exposure, dd_exposure), 0.0, 1.0))

        month_boundary = previous_month is None or month != previous_month
        rebalance = month_boundary or abs(desired_exposure - current_exposure) >= cfg.rebalance_band
        if rebalance:
            w = desired_exposure * base + (1 - desired_exposure) * fallback
            w /= w.sum()
            current_exposure = desired_exposure
        else:
            w = pretrade.copy()

        delta = w - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = abs((w[2] + w[3]) - (pretrade[2] + pretrade[3])) * 0.0005 * cost_multiplier
        gross_return = float(w @ asset_r)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        end_w = w * (1 + asset_r) / (1 + gross_return)
        rows.append({
            "date": date,
            "month": month,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            "regime": signals.loc[month, "regime"],
            "exposure": current_exposure,
            "forecast_vol": ann_vol,
            "fast_mom": fast_mom,
            "slow_mom": slow_mom,
            "vix": vix_value,
            **{f"w_{asset}": w[i] for i, asset in enumerate(ASSETS)},
        })
        pretrade = end_w
        previous_month = month
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda x: float(np.prod(1 + x))),
        gross_factor=("gross_return", lambda x: float(np.prod(1 + x))),
        turnover=("turnover", "sum"),
        trade_cost=("trade_cost", "sum"),
        fx_cost=("fx_cost", "sum"),
        avg_exposure=("exposure", "mean"),
        min_exposure=("exposure", "min"),
    )
    monthly["return"] = monthly["return_factor"] - 1
    monthly["gross_return"] = monthly["gross_factor"] - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    return daily, monthly


daily_levels = load_daily_open_levels()
features, _ = load_macro_data()
monthly_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(features, monthly_returns)
stress_raw = pd.read_csv(ROOT / "cache" / "stress_raw.csv", parse_dates=["date"]).set_index("date")
vix_daily = pd.to_numeric(stress_raw["VIX"], errors="coerce").dropna().sort_index()

hard_monthly = run_backtest(monthly_returns, signals, StrategyConfig(), mode="hard")
defensive_monthly = run_backtest(monthly_returns, signals, StrategyConfig(), mode="proposed")

# A no-control daily reconstruction validates the frequency conversion before
# any candidate is selected.
baseline_cfg = DailyOverlayConfig(99.0, 10, 60, 0.0, 99.0, 0.0, -0.99, 0.0, 1.0, "bond")
_, daily_hard_monthly = run_daily_overlay(daily_levels, signals, baseline_cfg, vix_daily)
common = hard_monthly.index.intersection(daily_hard_monthly.index)
print("=== DAILY RECONSTRUCTION AUDIT ===")
print("Original monthly hard", performance_summary(hard_monthly.loc[common, "return"])[["CAGR", "Sharpe", "MDD"]].round(4).to_dict())
print("Daily-open reconstructed", performance_summary(daily_hard_monthly.loc[common, "return"])[["CAGR", "Sharpe", "MDD"]].round(4).to_dict())

candidates = [
    DailyOverlayConfig(target_vol, fast, slow, trend_cut, vix_threshold, vix_cut, -dd_floor, dd_strength, band, fallback)
    for target_vol, fast, slow, trend_cut, vix_threshold, vix_cut, dd_floor, dd_strength, band, fallback in itertools.product(
        [0.18, 0.22, 99.0],
        [5, 20],
        [60],
        [0.0, 0.50],
        [25.0, 35.0, 99.0],
        [0.50, 0.75],
        [0.08, 0.09, 0.10],
        [0.75, 1.00],
        [0.05],
        ["bond", "defensive"],
    )
]

cal_end = "2017-12"
hard_cal_m = performance_summary(hard_monthly.loc[:cal_end, "return"])
defensive_cal_m = performance_summary(defensive_monthly.loc[:cal_end, "return"])
rows = []
for i, candidate in enumerate(candidates, start=1):
    _, monthly = run_daily_overlay(daily_levels, signals, candidate, vix_daily, end=cal_end)
    m = performance_summary(monthly["return"])
    rows.append({
        "name": candidate.name,
        **asdict(candidate),
        **m.to_dict(),
        "AvgTurnover": float(monthly["turnover"].mean()),
        "AvgExposure": float(monthly["avg_exposure"].mean()),
        "MDD9Pass": bool(m["MDD"] >= -0.09),
        "MDD10Pass": bool(m["MDD"] >= -0.10),
        "ReturnRetention": float(m["CAGR"] / hard_cal_m["CAGR"]),
    })
    if i % 1000 == 0:
        print(f"tested {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "daily_hard_overlay_calibration.csv", index=False)
eligible = ranking[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
print("\n=== CALIBRATION BASELINES ===")
print("Hard", hard_cal_m.round(4).to_dict())
print("Current defensive", defensive_cal_m.round(4).to_dict())
print("\n=== STRICT 9% MDD ELIGIBLE ===", len(eligible))
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgExposure", "ReturnRetention"]].head(25).round(4).to_string(index=False))

if eligible.empty:
    print("No candidate passed the strict calibration gate.")
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = DailyOverlayConfig(**{field: winner_row[field] for field in DailyOverlayConfig.__dataclass_fields__})
print("\n=== LOCKED WINNER ===")
print(winner)

comparison_rows = []
winner_daily_full, winner_monthly_full = run_daily_overlay(daily_levels, signals, winner, vix_daily)
for label, start, end in [("calibration", None, "2017-12"), ("locked_test", "2018-01", None), ("full", None, None)]:
    tests = [
        ("Hard", hard_monthly.loc[start:end] if start else hard_monthly.loc[:end] if end else hard_monthly),
        ("CurrentDefensive", defensive_monthly.loc[start:end] if start else defensive_monthly.loc[:end] if end else defensive_monthly),
        ("DailyProtectedHard", winner_monthly_full.loc[start:end] if start else winner_monthly_full.loc[:end] if end else winner_monthly_full),
    ]
    for strategy, bt in tests:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})

comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINES VS WINNER ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_daily_full.to_csv(RESULTS / "daily_hard_overlay_daily.csv")
winner_monthly_full.to_csv(RESULTS / "daily_hard_overlay_monthly.csv")
comparison.to_csv(RESULTS / "daily_hard_overlay_comparison.csv", index=False)
with (RESULTS / "daily_hard_overlay_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)
