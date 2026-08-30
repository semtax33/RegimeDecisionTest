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
    anchor = float(actual.loc[actual["date"] == first_actual, "open"].iloc[0])
    nearest = proxy.iloc[(proxy["date"] - first_actual).abs().argsort()[:1]]
    proxy["open"] *= anchor / float(nearest["open"].iloc[0])
    kodex = pd.concat(
        [proxy.loc[proxy["date"] < first_actual, ["date", "open"]], actual[["date", "open"]]],
        ignore_index=True,
    ).sort_values("date").drop_duplicates("date", keep="last").set_index("date")["open"]

    bond = pd.read_csv(get_path(RAW_DIR, "krx_bond_index.csv"), encoding="cp949")
    bond.index = pd.to_datetime(bond.iloc[:, 0])
    bond_level = pd.to_numeric(bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False), errors="coerce")
    bond_level.name = "BOND"

    opens = market.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    closes = market.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    fx = closes["USDKRW"].reindex(pd.date_range(closes.index.min(), closes.index.max(), freq="D")).ffill()
    gld = opens["GLD"] * fx.reindex(opens.index).to_numpy()
    uso = opens["USO"] * fx.reindex(opens.index).to_numpy()
    levels = pd.concat([kodex.rename("KODEX200"), bond_level, gld.rename("GLD"), uso.rename("USO")], axis=1).sort_index()
    return levels.reindex(pd.date_range(levels.index.min(), levels.index.max(), freq="B")).ffill(limit=5)[ASSETS]


@dataclass(frozen=True)
class LeveragedConfig:
    max_exposure: float
    target_vol: float
    vix_threshold: float
    vix_multiplier: float
    dd_floor: float
    dd_strength: float
    rebalance_band: float
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return (
            f"mx{self.max_exposure:.2f}_tv{self.target_vol:.2f}_vx{self.vix_threshold:.0f}"
            f"_vm{self.vix_multiplier:.2f}_fl{abs(self.dd_floor):.2f}_ds{self.dd_strength:.2f}"
            f"_rb{self.rebalance_band:.2f}_rf{self.financing_rate:.2f}"
        )


def prepare_daily_arrays(daily_levels: pd.DataFrame, signals: pd.DataFrame, vix: pd.Series) -> dict[str, object]:
    forward = daily_levels.shift(-1).div(daily_levels).sub(1.0)
    backward = daily_levels.pct_change(fill_method=None)
    mask = (
        (daily_levels.index.to_period("M") >= signals.index.min())
        & (daily_levels.index.to_period("M") <= signals.index.max())
        & forward.notna().all(axis=1)
    )
    dates = daily_levels.index[mask]
    months = dates.to_period("M")
    asset_returns = forward.loc[dates, ASSETS].to_numpy(dtype=float)
    bases = np.vstack([hard_regime_weights(signals.loc[m]) for m in months])

    unique_bases = {tuple(row) for row in bases}
    vol_map: dict[tuple[float, ...], pd.Series] = {}
    for key in unique_bases:
        base = np.asarray(key)
        proxy = backward.fillna(0.0).to_numpy() @ base
        vol_map[key] = pd.Series(proxy, index=backward.index).rolling(20, min_periods=10).std(ddof=1) * math.sqrt(252)
    forecast_vol = np.array([float(vol_map[tuple(base)].loc[date]) for date, base in zip(dates, bases)])
    vix_lagged = vix.reindex(pd.date_range(vix.index.min(), vix.index.max(), freq="D")).ffill().shift(1)
    vix_values = vix_lagged.reindex(dates, method="ffill").to_numpy(dtype=float)
    return {
        "dates": dates,
        "months": months,
        "returns": asset_returns,
        "bases": bases,
        "forecast_vol": forecast_vol,
        "vix": vix_values,
    }


def simulate(arrays: dict[str, object], cfg: LeveragedConfig, start: str | None = None, end: str | None = None, cost_multiplier: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    idx = np.flatnonzero(keep)

    nav = 1.0
    peak = 1.0
    current_exposure = 0.0
    pretrade_assets = np.zeros(len(ASSETS))
    pretrade_debt = 0.0
    previous_month: pd.Period | None = None
    first_trade = True
    rf_daily = (1 + cfg.financing_rate) ** (1 / 252) - 1
    rows = []

    for j in idx:
        month = months[j]
        base = bases[j]
        ann_vol = vols[j]
        vol_exposure = min(cfg.max_exposure, cfg.target_vol / max(ann_vol, 1e-6))
        vix_exposure = cfg.max_exposure * cfg.vix_multiplier if vix[j] >= cfg.vix_threshold else cfg.max_exposure
        current_dd = nav / peak - 1.0
        cushion = float(np.clip((current_dd - cfg.dd_floor) / (0.0 - cfg.dd_floor), 0.0, 1.0))
        dd_exposure = cfg.max_exposure * ((1 - cfg.dd_strength) + cfg.dd_strength * cushion)
        desired = float(np.clip(min(vol_exposure, vix_exposure, dd_exposure), 0.0, cfg.max_exposure))

        month_boundary = previous_month is None or month != previous_month
        rebalance = month_boundary or abs(desired - current_exposure) >= cfg.rebalance_band
        if rebalance:
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
        if net_return <= -0.99:
            raise RuntimeError("insolvent path")
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade_assets = asset_w * (1 + returns[j]) / (1 + net_return)
        pretrade_debt = debt_w * (1 + rf_daily) / (1 + net_return)
        rows.append({
            "date": dates[j], "month": month, "return": net_return, "gross_return": gross_return,
            "nav": nav, "drawdown": nav / peak - 1, "turnover": turnover,
            "trade_cost": trade_cost, "fx_cost": fx_cost, "exposure": current_exposure,
            "forecast_vol": ann_vol, "vix": vix[j], "debt_weight": debt_w,
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
    LeveragedConfig(mx, tv, vx, vm, -floor, ds, band)
    for mx, tv, vx, vm, floor, ds, band in itertools.product(
        [1.25, 1.50, 1.75, 2.00, 2.25],
        [0.18, 0.22, 0.26, 0.30, 0.35],
        [25.0, 35.0, 99.0],
        [0.50, 0.75],
        [0.08, 0.09, 0.10],
        [0.75, 1.00],
        [0.05, 0.10],
    )
]

rows = []
for i, candidate in enumerate(candidates, start=1):
    _, monthly = simulate(arrays, candidate, end="2017-12")
    m = performance_summary(monthly["return"])
    rows.append({
        "name": candidate.name, **asdict(candidate), **m.to_dict(),
        "AvgTurnover": float(monthly["turnover"].mean()), "AvgExposure": float(monthly["avg_exposure"].mean()),
        "MDD9Pass": bool(m["MDD"] >= -0.09), "CAGRRetention": float(m["CAGR"] / hard_cal_m["CAGR"]),
    })
    if i % 500 == 0:
        print(f"tested {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "leveraged_daily_calibration.csv", index=False)
eligible = ranking[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
print("=== STRICT 9% MDD ELIGIBLE ===", len(eligible))
print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover", "AvgExposure", "CAGRRetention"]].head(30).round(4).to_string(index=False))
if eligible.empty:
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = LeveragedConfig(**{field: float(winner_row[field]) for field in LeveragedConfig.__dataclass_fields__})
print("\n=== LOCKED WINNER ===")
print(winner)
winner_daily, winner_monthly = simulate(arrays, winner)

comparison_rows = []
for label, start, end in [("calibration", None, "2017-12"), ("locked_test", "2018-01", None), ("full", None, None)]:
    for strategy, bt in [
        ("Hard", hard.loc[start:end] if start else hard.loc[:end] if end else hard),
        ("CurrentDefensive", defensive.loc[start:end] if start else defensive.loc[:end] if end else defensive),
        ("LeveragedProtectedHard", winner_monthly.loc[start:end] if start else winner_monthly.loc[:end] if end else winner_monthly),
    ]:
        m = performance_summary(bt["return"])
        comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})
comparison = pd.DataFrame(comparison_rows)
print("\n=== BASELINES VS WINNER ===")
print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

winner_daily.to_csv(RESULTS / "leveraged_daily_backtest.csv")
winner_monthly.to_csv(RESULTS / "leveraged_daily_monthly.csv")
comparison.to_csv(RESULTS / "leveraged_daily_comparison.csv", index=False)
with (RESULTS / "leveraged_daily_winner.json").open("w", encoding="utf-8") as f:
    json.dump(asdict(winner), f, ensure_ascii=False, indent=2)

