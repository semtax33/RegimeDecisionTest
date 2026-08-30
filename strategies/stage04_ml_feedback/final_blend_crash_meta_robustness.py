from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import FactorBlendConfig, run_factor_blend
from strategies.core.regime_research import (
    StrategyConfig,
    compute_regime_signals,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load_factor(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"final_blend_crash_meta_{name}_factor.csv")
    frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
    return frame.set_index("month")


def record(test: str, strategy: str, backtest: pd.DataFrame) -> dict[str, float | str]:
    metrics = performance_summary(backtest["return"])
    return {
        "Test": test,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(backtest["turnover"].mean()),
    }


def main() -> None:
    macro, _ = load_macro_data()
    returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, returns)
    defensive = run_backtest(returns, signals, StrategyConfig(), mode="proposed")
    factors = {
        "Loss3Stress": (load_factor("loss3_stress"), 0.10),
        "Loss4Domestic": (load_factor("loss4_domestic"), 0.15),
        "Loss4Stress": (load_factor("loss4_stress"), 0.10),
    }
    neutral = next(iter(factors.values()))[0]
    rows: list[dict[str, float | str]] = []

    for cost in (0.5, 1.0, 2.0, 3.0):
        baseline = run_factor_blend(
            returns,
            signals,
            defensive,
            neutral,
            FactorBlendConfig(max_shift=0.0),
            cost_multiplier=cost,
        )
        rows.append(record(f"cost_{cost:.1f}x_full", "FinalBlend", baseline.loc["2007-01":]))
        rows.append(record(f"cost_{cost:.1f}x_locked", "FinalBlend", baseline.loc["2018-01":]))
        for name, (factor, shift) in factors.items():
            backtest = run_factor_blend(
                returns,
                signals,
                defensive,
                factor,
                FactorBlendConfig(max_shift=shift),
                cost_multiplier=cost,
            )
            rows.append(record(f"cost_{cost:.1f}x_full", name, backtest.loc["2007-01":]))
            rows.append(record(f"cost_{cost:.1f}x_locked", name, backtest.loc["2018-01":]))

    baseline = run_factor_blend(
        returns, signals, defensive, neutral, FactorBlendConfig(max_shift=0.0)
    )
    standard_backtests = {
        "FinalBlend": baseline,
        **{
            name: run_factor_blend(
                returns, signals, defensive, factor, FactorBlendConfig(max_shift=shift)
            )
            for name, (factor, shift) in factors.items()
        },
    }
    for label, start, end in (
        ("subperiod_2007_2012", "2007-01", "2012-12"),
        ("subperiod_2013_2017", "2013-01", "2017-12"),
        ("subperiod_2018_2021", "2018-01", "2021-12"),
        ("subperiod_2022_2026", "2022-01", "2026-12"),
    ):
        for name, backtest in standard_backtests.items():
            rows.append(record(label, name, backtest.loc[start:end]))

    # Locked-period sensitivity is diagnostic only; it is not used to select a shift.
    for name, (factor, _) in factors.items():
        for shift in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
            backtest = run_factor_blend(
                returns, signals, defensive, factor, FactorBlendConfig(max_shift=shift)
            )
            rows.append(record(f"locked_shift_{shift:.3f}", name, backtest.loc["2018-01":]))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "final_blend_crash_meta_robustness.csv", index=False)
    print(out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
