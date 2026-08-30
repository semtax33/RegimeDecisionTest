from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategies.stage04_ml_feedback.feedback_alternative_strategies_experiment import run_vol_target
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


def load_factor(path: Path) -> pd.DataFrame:
    factor = pd.read_csv(path)
    factor["target_month"] = pd.PeriodIndex(factor["target_month"], freq="M")
    return factor.set_index("target_month")


def record(test: str, strategy: str, backtest: pd.DataFrame) -> dict[str, float | str]:
    metrics = performance_summary(backtest["return"])
    return {"Test": test, "Strategy": strategy, **metrics.to_dict()}


def main() -> None:
    macro, _ = load_macro_data()
    returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, returns)
    defensive = run_backtest(returns, signals, StrategyConfig(), mode="proposed")
    hmm = load_factor(RESULTS / "alternative_factor_generalhmm.csv")
    rows: list[dict[str, float | str]] = []

    for multiplier in (0.5, 1.0, 2.0, 3.0):
        baseline = run_factor_blend(
            returns,
            signals,
            defensive,
            hmm,
            FactorBlendConfig(max_shift=0.0),
            cost_multiplier=multiplier,
        )
        hmm_test = run_factor_blend(
            returns,
            signals,
            defensive,
            hmm,
            FactorBlendConfig(max_shift=0.075),
            cost_multiplier=multiplier,
        )
        vol_target = run_vol_target(
            returns,
            signals,
            defensive,
            target_vol=0.15,
            cost_multiplier=multiplier,
        )
        for name, backtest in (
            ("FinalBlend", baseline),
            ("GeneralHMM", hmm_test),
            ("VolTarget15", vol_target),
        ):
            rows.append(record(f"cost_{multiplier:.1f}x_full_2007_2026", name, backtest.loc["2007-01":]))
            rows.append(record(f"cost_{multiplier:.1f}x_locked_2018_2026", name, backtest.loc["2018-01":]))

    baseline = run_factor_blend(
        returns, signals, defensive, hmm, FactorBlendConfig(max_shift=0.0)
    )
    hmm_test = run_factor_blend(
        returns, signals, defensive, hmm, FactorBlendConfig(max_shift=0.075)
    )
    vol_target = run_vol_target(returns, signals, defensive, target_vol=0.15)
    subperiods = (
        ("subperiod_2007_2012", "2007-01", "2012-12"),
        ("subperiod_2013_2017", "2013-01", "2017-12"),
        ("subperiod_2018_2021", "2018-01", "2021-12"),
        ("subperiod_2022_2026", "2022-01", "2026-12"),
    )
    for label, start, end in subperiods:
        for name, backtest in (
            ("FinalBlend", baseline),
            ("GeneralHMM", hmm_test),
            ("VolTarget15", vol_target),
        ):
            rows.append(record(label, name, backtest.loc[start:end]))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "feedback_strategy_robustness.csv", index=False)
    print(out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
