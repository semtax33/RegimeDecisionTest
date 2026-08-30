from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from strategies.core.regime_research import (
    ASSETS,
    DEFENSIVE,
    STRATEGIC,
    StrategyConfig,
    cdar,
    compute_regime_signals,
    ewma_cov,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
    soft_anchor,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class AdaptiveVolConfig:
    calm_target: float
    stress_target: float
    trend_window: int
    normal_target: float = 0.08
    dd_stress: float = -0.025
    dd_calm: float = -0.01
    calm_vol_ratio: float = 1.10
    stress_vol_ratio: float = 1.35

    @property
    def name(self) -> str:
        return f"ct{self.calm_target:.3f}_st{self.stress_target:.3f}_tw{self.trend_window}"


def adaptive_state(history: pd.DataFrame, current_dd: float, acfg: AdaptiveVolConfig) -> tuple[str, float]:
    if len(history) < 24:
        return "NORMAL", acfg.normal_target
    eq = history["KODEX200"]
    mom = float((1 + eq.tail(acfg.trend_window)).prod() - 1)
    mom1 = float(eq.iloc[-1])
    mom3 = float((1 + eq.tail(3)).prod() - 1)
    short_vol = float(eq.tail(3).std(ddof=1) * math.sqrt(12))
    long_vol = float(eq.tail(24).std(ddof=1) * math.sqrt(12))

    stress = current_dd <= acfg.dd_stress or mom3 < 0 or short_vol > acfg.stress_vol_ratio * long_vol
    calm = (
        current_dd >= acfg.dd_calm
        and mom > 0
        and mom1 > 0
        and mom3 > 0
        and short_vol <= acfg.calm_vol_ratio * long_vol
    )
    if stress:
        return "STRESS", acfg.stress_target
    if calm:
        return "CALM", acfg.calm_target
    return "NORMAL", acfg.normal_target


def adaptive_weights(
    signal: pd.Series,
    history: pd.DataFrame,
    pretrade: np.ndarray,
    current_dd: float,
    cfg: StrategyConfig,
    acfg: AdaptiveVolConfig,
) -> tuple[np.ndarray, str, float]:
    anchor = soft_anchor(signal) if cfg.use_regime else STRATEGIC.copy()
    anchor = cfg.regime_strength * anchor + (1 - cfg.regime_strength) * STRATEGIC
    state, state_target = adaptive_state(history, current_dd, acfg)
    if not cfg.use_risk_control or len(history) < 24:
        return anchor, state, state_target

    cov = ewma_cov(history.tail(84), cfg.half_life, leverage=1.0)
    vols = np.sqrt(np.diag(cov)).clip(0.005, None)
    tilted = anchor * (np.median(vols) / vols) ** cfg.invvol_tilt
    tilted = tilted / tilted.sum()
    prior = 0.55 * anchor + 0.45 * tilted

    hist = history.tail(84)[ASSETS]
    long_mu = history[ASSETS].expanding(min_periods=24).mean().iloc[-1].to_numpy()
    recent_mu = hist.ewm(halflife=24, adjust=False).mean().iloc[-1].to_numpy()
    mu = np.clip(0.80 * long_mu + 0.20 * recent_mu, -0.006, 0.015)
    target = state_target * (0.86 + 0.20 * float(signal["p_growth_high"]))

    def objective(w: np.ndarray) -> float:
        ann_return = 12 * float(w @ mu)
        ann_vol = math.sqrt(max(float(w @ cov @ w), 0.0) * 12)
        path_cdar = abs(cdar(hist.to_numpy() @ w, 0.90))
        turnover = 0.5 * np.sum(np.sqrt((w - pretrade) ** 2 + 1e-6))
        tracking = float(np.sum((w - prior) ** 2))
        return (
            -cfg.return_reward * ann_return
            + cfg.vol_penalty * ann_vol
            + cfg.cdar_penalty * path_cdar
            + cfg.turnover_penalty * turnover
            + cfg.tracking_penalty * tracking
        )

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: target - math.sqrt(max(float(w @ cov @ w), 0.0) * 12)},
        {"type": "ineq", "fun": lambda w: cfg.max_cdar + cdar(hist.to_numpy() @ w, 0.90)},
    ]
    bounds = [(0.02, 0.68), (0.05, 0.88), (0.02, 0.62), (0.0, 0.38)]
    result = minimize(objective, prior, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 80, "ftol": 1e-8})
    w = result.x if result.success and np.isfinite(result.x).all() else prior
    w = np.clip(w, 0, None)
    w /= w.sum()

    # Keep the original state-dependent drawdown guard.  In stress it is
    # strengthened slightly because the calm state is allowed more risk.
    if current_dd < -0.05:
        severity = min(max((-current_dd - 0.05) / 0.12, 0.0), 1.0)
        guard = cfg.drawdown_guard * (1.15 if state == "STRESS" else 1.0)
        blend = min(guard * (0.25 + 0.50 * severity), 0.90)
        w = (1 - blend) * w + blend * DEFENSIVE
    return w / w.sum(), state, target


def run_adaptive(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: StrategyConfig,
    acfg: AdaptiveVolConfig,
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
    pretrade = np.zeros(4)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        signal = signals.loc[month]
        history = returns.loc[returns.index < month]
        current_dd = nav / peak - 1.0
        w, risk_state, target_vol = adaptive_weights(signal, history, pretrade, current_dd, cfg, acfg)
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
            "risk_state": risk_state,
            "target_vol": target_vol,
            "regime": signal["regime"],
            **{f"w_{a}": w[i] for i, a in enumerate(ASSETS)},
        })
        pretrade = end_w
        first_trade = False
    return pd.DataFrame(rows).set_index("month")


def main() -> None:
    features, _ = load_macro_data()
    returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(features, returns)
    cfg = StrategyConfig()
    cal_end = "2017-12"
    test_start = "2018-01"
    baseline_cal = run_backtest(returns, signals, cfg, end=cal_end)
    baseline_metrics = performance_summary(baseline_cal["return"])
    mdd_floor = float(baseline_metrics["MDD"] - 0.005)

    candidates = [
        AdaptiveVolConfig(calm_target=calm, stress_target=stress, trend_window=window)
        for calm, stress, window in itertools.product(
            [0.09, 0.10, 0.11, 0.12],
            [0.055, 0.065, 0.075],
            [6, 12],
        )
    ]
    rows = []
    for i, acfg in enumerate(candidates, start=1):
        bt = run_adaptive(returns, signals, cfg, acfg, end=cal_end)
        m = performance_summary(bt["return"])
        rows.append({
            "name": acfg.name,
            **asdict(acfg),
            **m.to_dict(),
            "AvgTurnover": float(bt["turnover"].mean()),
            "MDDPass": bool(m["MDD"] >= mdd_floor),
            "SharpePass": bool(m["Sharpe"] >= 1.0),
        })
        if i % 8 == 0:
            print(f"calibrated {i}/{len(candidates)}")

    ranking = pd.DataFrame(rows)
    ranking.to_csv(RESULTS / "adaptive_vol_calibration.csv", index=False)
    eligible = ranking[ranking["MDDPass"] & ranking["SharpePass"]].sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
    print("\n=== BASELINE CALIBRATION ===")
    print(baseline_metrics.round(4).to_string())
    print("MDD floor", round(mdd_floor, 4))
    if eligible.empty:
        print("\nNo candidate passed. Closest candidates:")
        print(ranking.sort_values(["MDD", "CAGR"], ascending=[False, False])[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].head(15).round(4).to_string(index=False))
        raise SystemExit(2)

    print("\n=== ELIGIBLE TOP 15 (CALIBRATION ONLY) ===")
    print(eligible[["name", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].head(15).round(4).to_string(index=False))
    winner_row = eligible.iloc[0]
    winner = next(c for c in candidates if c.name == winner_row["name"])
    print("\n=== LOCKED WINNER ===")
    print(winner)

    comparison_rows = []
    for label, start, end in [("calibration", None, cal_end), ("locked_test", test_start, None), ("full", None, None)]:
        base_bt = run_backtest(returns, signals, cfg, start=start, end=end)
        adaptive_bt = run_adaptive(returns, signals, cfg, winner, start=start, end=end)
        for strategy, bt in [("Baseline", base_bt), ("AdaptiveVol", adaptive_bt)]:
            m = performance_summary(bt["return"])
            comparison_rows.append({"Period": label, "Strategy": strategy, **m.to_dict(), "AvgTurnover": float(bt["turnover"].mean())})
    comparison = pd.DataFrame(comparison_rows)
    print("\n=== BASELINE VS WINNER ===")
    print(comparison[["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "MDD", "Calmar", "AvgTurnover"]].round(4).to_string(index=False))

    winner_full = run_adaptive(returns, signals, cfg, winner)
    print("\n=== RISK STATE COUNTS / WEIGHTS ===")
    print(winner_full["risk_state"].value_counts().to_string())
    print(winner_full.groupby("risk_state")[[f"w_{a}" for a in ASSETS]].mean().round(4).to_string())
    comparison.to_csv(RESULTS / "adaptive_vol_comparison.csv", index=False)
    winner_full.to_csv(RESULTS / "adaptive_vol_backtest.csv")
    with (RESULTS / "adaptive_vol_winner.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(winner), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
