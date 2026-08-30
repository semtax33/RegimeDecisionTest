from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", "M")

# Reuse the lagged expanding-model machinery without running its search loop.
source = (ROOT / "hard_crash_short_experiment.py").read_text(encoding="utf-8")
source = source.split("\nfeature_data = pd.read_csv")[0]
exec(compile(source, str(ROOT / "hard_crash_short_experiment.py"), "exec"), globals())


@dataclass(frozen=True)
class PutConfig:
    model_name: str
    rank_threshold: float
    otm: float
    spread_width: float
    coverage: float
    dd_trigger: float
    premium_markup: float = 0.10

    @property
    def name(self) -> str:
        width = "put" if self.spread_width >= 0.90 else f"w{self.spread_width:.3f}"
        return (
            f"{self.model_name}_rq{self.rank_threshold:.2f}_otm{self.otm:.3f}_{width}"
            f"_cov{self.coverage:.2f}_dd{abs(self.dd_trigger):.2f}"
        )


def black_scholes_put(strike: float, sigma: float, rate: float = 0.03, term: float = 1 / 12) -> float:
    sigma = float(np.clip(sigma, 0.05, 2.50))
    root_t = math.sqrt(term)
    d1 = (math.log(1.0 / strike) + (rate + 0.5 * sigma * sigma) * term) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    return float(strike * math.exp(-rate * term) * norm.cdf(-d2) - norm.cdf(-d1))


def conservative_iv(asset: str, feature: pd.Series) -> float:
    realized = float(feature["daily_vol21"])
    vix = float(feature.get("VIX_last", np.nan)) / 100
    ovx = float(feature.get("OVX_last", np.nan)) / 100
    gvz = float(feature.get("GVZ_last", np.nan)) / 100
    if asset == "KODEX200":
        candidates = [0.18, 1.25 * realized, vix]
    elif asset == "USO":
        candidates = [0.30, 1.10 * realized, ovx]
    elif asset == "GLD":
        candidates = [0.15, 1.10 * realized, gvz]
    else:
        return 0.0
    return float(np.nanmax(candidates))


def option_quote(cfg: PutConfig, asset: str, feature: pd.Series) -> tuple[float, float, float]:
    sigma = conservative_iv(asset, feature)
    long_strike = 1.0 - cfg.otm
    short_strike = max(long_strike - cfg.spread_width, 0.01)
    long_put = black_scholes_put(long_strike, sigma)
    if cfg.spread_width >= 0.90:
        short_put = 0.0
        short_strike = 0.0
    else:
        short_put = black_scholes_put(short_strike, sigma)
    # Buy the long leg at 10% above mid and sell the short leg at 10% below
    # mid, plus 5 bp of notional.  This deliberately penalizes the proxy.
    premium = (1 + cfg.premium_markup) * long_put - (1 - cfg.premium_markup) * short_put + 0.0005
    return float(max(premium, 0.0005)), long_strike, short_strike


def option_payoff(underlying_growth: float, long_strike: float, short_strike: float) -> float:
    long_payoff = max(long_strike - underlying_growth, 0.0)
    short_payoff = max(short_strike - underlying_growth, 0.0) if short_strike > 0 else 0.0
    return float(long_payoff - short_payoff)


def load_usd_returns(returns: pd.DataFrame) -> pd.DataFrame:
    market = pd.read_csv(ROOT / "cache" / "market_daily.csv", parse_dates=["date"])
    first_open = {}
    for asset in ["GLD", "USO"]:
        data = market.loc[market["symbol"] == asset].dropna(subset=["open"]).sort_values("date").copy()
        data["month"] = data["date"].dt.to_period("M")
        first_open[asset] = data.groupby("month")["open"].first().astype(float)
    levels = pd.DataFrame(first_open)
    usd = levels.shift(-1).div(levels).sub(1.0)
    return usd.reindex(returns.index)


def run_put_overlay(
    returns: pd.DataFrame,
    usd_returns: pd.DataFrame,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    risk_rank: pd.Series,
    cfg: PutConfig,
    end: str | None = None,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index).intersection(features.index)
    if end:
        months = months[months <= pd.Period(end, "M")]

    rows = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        base = hard_regime_weights(signals.loc[month])
        feature = features.loc[month]
        rank_value = float(risk_rank.loc[month]) if pd.notna(risk_rank.loc[month]) else 0.0
        current_dd = nav / peak - 1.0
        active = bool(
            cfg.model_name == "always"
            or rank_value >= cfg.rank_threshold
            or (cfg.dd_trigger > -0.90 and current_dd <= cfg.dd_trigger)
        )

        quotes: dict[str, tuple[float, float, float]] = {}
        premium_ratio = 0.0
        if active:
            for i, asset in enumerate(ASSETS):
                if asset == "BOND" or base[i] <= 0:
                    continue
                quotes[asset] = option_quote(cfg, asset, feature)
                premium_ratio += base[i] * cfg.coverage * quotes[asset][0]

        # The option premium is funded from NAV, rather than silently borrowing
        # on top of a fully invested hard portfolio.
        scale = 1.0 / (1.0 + premium_ratio)
        invested = scale * base
        option_notional = {asset: scale * base[ASSETS.index(asset)] * cfg.coverage for asset in quotes}
        option_premium = 1.0 - scale

        delta = invested - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015
        fx_cost = abs((invested[2] + invested[3]) - (pretrade[2] + pretrade[3])) * 0.0005

        asset_r = returns.loc[month, ASSETS].to_numpy(dtype=float)
        base_profit = float(invested @ asset_r)
        option_value = 0.0
        for asset, notional in option_notional.items():
            if asset in {"GLD", "USO"} and pd.notna(usd_returns.loc[month, asset]):
                usd_growth = 1.0 + float(usd_returns.loc[month, asset])
                krw_growth = 1.0 + float(returns.loc[month, asset])
                fx_growth = krw_growth / usd_growth
                underlying_growth = usd_growth
            else:
                fx_growth = 1.0
                underlying_growth = 1.0 + float(returns.loc[month, asset])
            _, long_strike, short_strike = quotes[asset]
            option_value += notional * option_payoff(underlying_growth, long_strike, short_strike) * fx_growth

        net_return = scale + base_profit + option_value - trade_cost - fx_cost - 1.0
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        gross_base_growth = float(invested @ (1.0 + asset_r))
        terminal_total = gross_base_growth + option_value
        pretrade = invested * (1.0 + asset_r) / max(terminal_total, 1e-12)
        first_trade = False
        rows.append(
            {
                "month": month,
                "return": net_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "regime": signals.loc[month, "regime"],
                "risk_rank": rank_value,
                "option_active": active,
                "option_premium": option_premium,
                "option_payoff": option_value,
                "base_scale": scale,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **{f"w_{asset}": invested[i] for i, asset in enumerate(ASSETS)},
            }
        )
    return pd.DataFrame(rows).set_index("month")


features = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
features.index = pd.PeriodIndex(features.index, freq="M")
macro, _ = load_macro_data()
asset_returns, _ = load_monthly_asset_returns()
signals = compute_regime_signals(macro, asset_returns)
usd_returns = load_usd_returns(asset_returns)

specs = [
    ModelSpec("logit_s_l5_c01", "stress", "loss5", "logit", 0.1),
    ModelSpec("logit_s_l8_c01", "stress", "loss8", "logit", 0.1),
    ModelSpec("gbdt_s_l5_d1", "stress", "loss5", "gbdt", 1.0),
]
ranks = {}
for spec in specs:
    probability = walkforward_probability(
        features,
        FEATURE_SETS[spec.feature_set],
        spec.target,
        make_model(spec.kind, spec.strength),
    )
    ranks[spec.name] = expanding_rank(probability)
ranks["always"] = pd.Series(1.0, index=features.index)

model_thresholds = [(spec.name, q) for spec in specs for q in [0.80, 0.85, 0.90, 0.95]]
model_thresholds.append(("always", 0.0))
candidates = [
    PutConfig(model, q, otm, width, coverage, dd)
    for (model, q), otm, width, coverage, dd in itertools.product(
        model_thresholds,
        [0.025, 0.05, 0.075, 0.10],
        [0.05, 0.10, 0.15, 1.00],
        [0.50, 1.00, 1.50],
        [-0.03, -0.05, -0.99],
    )
]

rows = []
for i, candidate in enumerate(candidates, start=1):
    backtest = run_put_overlay(
        asset_returns,
        usd_returns,
        signals,
        features,
        ranks[candidate.model_name],
        candidate,
        end=str(CAL_END),
    )
    metrics = performance_summary(backtest["return"])
    rows.append(
        {
            "name": candidate.name,
            **asdict(candidate),
            **metrics.to_dict(),
            "ActiveMonths": int(backtest["option_active"].sum()),
            "AnnualPremium": float(backtest["option_premium"].mean() * 12),
            "AvgPayoff": float(backtest["option_payoff"].mean()),
            "AvgTurnover": float(backtest["turnover"].mean()),
            "MDD9Pass": bool(metrics["MDD"] >= -0.09),
        }
    )
    if i % 500 == 0:
        print(f"tested {i}/{len(candidates)}")

ranking = pd.DataFrame(rows)
ranking.to_csv(RESULTS / "synthetic_put_calibration.csv", index=False)
eligible = ranking.loc[ranking["MDD9Pass"]].sort_values(["CAGR", "Sharpe"], ascending=False)
print("=== STRICT 9% MDD ELIGIBLE ===", len(eligible))
print(
    eligible[
        ["name", "CAGR", "Sharpe", "MDD", "Calmar", "ActiveMonths", "AnnualPremium", "AvgTurnover"]
    ].head(25).round(4).to_string(index=False)
)
for limit in [-0.09, -0.10, -0.12, -0.15]:
    frontier = ranking.loc[ranking["MDD"] >= limit].sort_values("CAGR", ascending=False).head(1)
    print(f"\n=== BEST CAGR WITH MDD >= {limit:.0%} ===")
    print(frontier[["name", "CAGR", "Sharpe", "MDD", "AnnualPremium"]].round(4).to_string(index=False))

if eligible.empty:
    raise SystemExit(2)

winner_row = eligible.iloc[0]
winner = PutConfig(
    **{
        field: str(winner_row[field]) if field == "model_name" else float(winner_row[field])
        for field in PutConfig.__dataclass_fields__
    }
)
winner_full = run_put_overlay(
    asset_returns,
    usd_returns,
    signals,
    features,
    ranks[winner.model_name],
    winner,
)
hard = run_backtest(asset_returns, signals, StrategyConfig(), mode="hard")
defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")

comparison_rows = []
for period, start, end in [("calibration", None, "2017-12"), ("locked_test", "2018-01", None), ("full", None, None)]:
    for strategy, backtest in [("Hard", hard), ("CurrentDefensive", defensive), ("SyntheticPut", winner_full)]:
        sample = backtest.loc[start:end] if start else backtest.loc[:end] if end else backtest
        metrics = performance_summary(sample["return"])
        comparison_rows.append(
            {
                "Period": period,
                "Strategy": strategy,
                **metrics.to_dict(),
                "AnnualPremium": float(sample.get("option_premium", pd.Series(0.0, index=sample.index)).mean() * 12),
                "ActiveMonths": int(sample.get("option_active", pd.Series(False, index=sample.index)).sum()),
                "AvgTurnover": float(sample["turnover"].mean()),
            }
        )
comparison = pd.DataFrame(comparison_rows)
print("\n=== LOCKED COMPARISON ===")
print(
    comparison[
        ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AnnualPremium", "ActiveMonths"]
    ].round(4).to_string(index=False)
)

winner_full.to_csv(RESULTS / "synthetic_put_backtest.csv")
comparison.to_csv(RESULTS / "synthetic_put_comparison.csv", index=False)
with (RESULTS / "synthetic_put_winner.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(winner), handle, ensure_ascii=False, indent=2)
