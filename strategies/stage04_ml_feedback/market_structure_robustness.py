from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage04_ml_feedback.feedback_alternative_strategies_experiment import run_vol_target
from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import (
    FactorBlendConfig,
    apply_factor_tilt,
    paired_block_bootstrap,
    run_factor_blend,
)
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


def load_factor(name: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"market_structure_{name}_factor.csv")
    frame["month"] = pd.PeriodIndex(frame["month"], freq="M")
    return frame.set_index("month")


def run_factor_vol_target(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive_path: pd.DataFrame,
    factor: pd.DataFrame,
    max_shift: float,
    target_vol: float = 0.15,
    cost_multiplier: float = 1.0,
    financing_rate: float = 0.04,
) -> pd.DataFrame:
    months = signals.index.intersection(returns.index).intersection(defensive_path.index)
    rows: list[dict[str, float | pd.Period | str]] = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        defensive = defensive_path.loc[month, [f"w_{asset}" for asset in ASSETS]].to_numpy(dtype=float)
        base = 0.40 * hard + 0.60 * defensive
        probability = float(factor.loc[month, "p_up"]) if month in factor.index else 0.5
        score = float(np.clip((probability - 0.5) / 0.15, -1, 1))
        unlevered = apply_factor_tilt(base, score, max_shift)

        history = returns.loc[returns.index < month, ASSETS].tail(24)
        if len(history) >= 12:
            conditional = history.to_numpy(dtype=float) @ unlevered
            weights = np.exp(np.linspace(-2.0, 0.0, len(conditional)))
            weights /= weights.sum()
            mean = float(weights @ conditional)
            variance = float(weights @ (conditional - mean) ** 2)
            forecast_vol = math.sqrt(max(variance, 1e-8) * 12)
            leverage = float(np.clip(target_vol / forecast_vol, 0.50, 1.50))
        else:
            forecast_vol = np.nan
            leverage = 1.20
        asset_weights = leverage * unlevered
        debt_weight = 1.0 - leverage
        delta = asset_weights - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = (
            abs((asset_weights[2] + asset_weights[3]) - (pretrade[2] + pretrade[3]))
            * 0.0005
            * cost_multiplier
        )
        financing = debt_weight * ((1 + financing_rate) ** (1 / 12) - 1)
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(asset_weights @ asset_return + financing)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = asset_weights * (1 + asset_return) / (1 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "forecast_vol": forecast_vol,
                "leverage": leverage,
                "risk_score": score,
                **{f"w_{asset}": asset_weights[i] for i, asset in enumerate(ASSETS)},
            }
        )
    return pd.DataFrame(rows).set_index("month")


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
        "CompositeDomestic": (load_factor("loss3_composite_domestic"), 0.15),
        "VolumeDomestic": (load_factor("loss3_volume_domestic"), 0.20),
        "IndexVolumeRaw": (load_factor("loss3_index_volume_raw_domestic"), 0.20),
        "IndexVolumeComposite": (
            load_factor("loss3_index_volume_composite_domestic"),
            0.20,
        ),
        "CompositePlusIndexVolume": (
            load_factor("loss3_composite_plus_index_volume_domestic"),
            0.20,
        ),
        "CorrelationDomestic": (load_factor("loss3_corr_domestic"), 0.20),
        "BreadthDomestic": (load_factor("loss3_breadth_domestic"), 0.20),
        "TailShapeDomestic": (load_factor("loss3_tail_domestic"), 0.10),
    }
    neutral = next(iter(factors.values()))[0]
    baseline = run_factor_blend(
        returns, signals, defensive, neutral, FactorBlendConfig(max_shift=0.0)
    )
    vol_target = run_vol_target(returns, signals, defensive, target_vol=0.15)
    standard: dict[str, pd.DataFrame] = {
        "FinalBlend": baseline,
        "VolTarget15": vol_target,
    }
    for name, (factor, shift) in factors.items():
        standard[name] = run_factor_blend(
            returns, signals, defensive, factor, FactorBlendConfig(max_shift=shift)
        )
    combined = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factors["CompositeDomestic"][0],
        max_shift=0.15,
        target_vol=0.15,
    )
    standard["CompositePlusVol15"] = combined
    proxy_combined = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factors["CompositePlusIndexVolume"][0],
        max_shift=0.20,
        target_vol=0.15,
    )
    standard["CompositeIndexVolumePlusVol15"] = proxy_combined

    rows: list[dict[str, float | str]] = []
    for label, start, end in (
        ("calibration_2007_2017", "2007-01", "2017-12"),
        ("locked_2018_2026", "2018-01", "2026-12"),
        ("full_2007_2026", "2007-01", "2026-12"),
        ("subperiod_2007_2012", "2007-01", "2012-12"),
        ("subperiod_2013_2017", "2013-01", "2017-12"),
        ("subperiod_2018_2021", "2018-01", "2021-12"),
        ("subperiod_2022_2026", "2022-01", "2026-12"),
    ):
        for name, backtest in standard.items():
            rows.append(record(label, name, backtest.loc[start:end]))

    for cost in (0.5, 1.0, 2.0, 3.0):
        baseline_cost = run_factor_blend(
            returns,
            signals,
            defensive,
            neutral,
            FactorBlendConfig(max_shift=0.0),
            cost_multiplier=cost,
        )
        composite_cost = run_factor_blend(
            returns,
            signals,
            defensive,
            factors["CompositeDomestic"][0],
            FactorBlendConfig(max_shift=0.15),
            cost_multiplier=cost,
        )
        index_volume_raw_cost = run_factor_blend(
            returns,
            signals,
            defensive,
            factors["IndexVolumeRaw"][0],
            FactorBlendConfig(max_shift=0.20),
            cost_multiplier=cost,
        )
        index_volume_composite_cost = run_factor_blend(
            returns,
            signals,
            defensive,
            factors["IndexVolumeComposite"][0],
            FactorBlendConfig(max_shift=0.20),
            cost_multiplier=cost,
        )
        composite_index_volume_cost = run_factor_blend(
            returns,
            signals,
            defensive,
            factors["CompositePlusIndexVolume"][0],
            FactorBlendConfig(max_shift=0.20),
            cost_multiplier=cost,
        )
        combined_cost = run_factor_vol_target(
            returns,
            signals,
            defensive,
            factors["CompositeDomestic"][0],
            max_shift=0.15,
            target_vol=0.15,
            cost_multiplier=cost,
        )
        proxy_combined_cost = run_factor_vol_target(
            returns,
            signals,
            defensive,
            factors["CompositePlusIndexVolume"][0],
            max_shift=0.20,
            target_vol=0.15,
            cost_multiplier=cost,
        )
        for name, backtest in (
            ("FinalBlend", baseline_cost),
            ("CompositeDomestic", composite_cost),
            ("IndexVolumeRaw", index_volume_raw_cost),
            ("IndexVolumeComposite", index_volume_composite_cost),
            ("CompositePlusIndexVolume", composite_index_volume_cost),
            ("CompositePlusVol15", combined_cost),
            ("CompositeIndexVolumePlusVol15", proxy_combined_cost),
        ):
            rows.append(record(f"cost_{cost:.1f}x_full", name, backtest.loc["2007-01":]))
            rows.append(record(f"cost_{cost:.1f}x_locked", name, backtest.loc["2018-01":]))

    # Locked-period shift sensitivity is diagnostic only, never used for selection.
    for name in (
        "CompositeDomestic",
        "VolumeDomestic",
        "IndexVolumeRaw",
        "IndexVolumeComposite",
        "CompositePlusIndexVolume",
    ):
        factor = factors[name][0]
        for shift in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20):
            backtest = run_factor_blend(
                returns, signals, defensive, factor, FactorBlendConfig(max_shift=shift)
            )
            rows.append(record(f"locked_shift_{shift:.3f}", name, backtest.loc["2018-01":]))

    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "market_structure_robustness.csv", index=False)
    baseline_locked = baseline.loc["2018-01":, "return"]
    bootstrap = {
        "CompositeDomestic": paired_block_bootstrap(
            baseline_locked, standard["CompositeDomestic"].loc["2018-01":, "return"]
        ),
        "CompositePlusVol15": paired_block_bootstrap(
            baseline_locked, combined.loc["2018-01":, "return"]
        ),
        "IndexVolumeRaw": paired_block_bootstrap(
            baseline_locked, standard["IndexVolumeRaw"].loc["2018-01":, "return"]
        ),
        "IndexVolumeComposite": paired_block_bootstrap(
            baseline_locked,
            standard["IndexVolumeComposite"].loc["2018-01":, "return"],
        ),
        "CompositePlusIndexVolume": paired_block_bootstrap(
            baseline_locked,
            standard["CompositePlusIndexVolume"].loc["2018-01":, "return"],
        ),
        "CompositeIndexVolumePlusVol15": paired_block_bootstrap(
            baseline_locked, proxy_combined.loc["2018-01":, "return"]
        ),
        "IndexVolumeRaw_vs_VolumeDomestic": paired_block_bootstrap(
            standard["VolumeDomestic"].loc["2018-01":, "return"],
            standard["IndexVolumeRaw"].loc["2018-01":, "return"],
        ),
        "CompositePlusIndexVolume_vs_CompositeDomestic": paired_block_bootstrap(
            standard["CompositeDomestic"].loc["2018-01":, "return"],
            standard["CompositePlusIndexVolume"].loc["2018-01":, "return"],
        ),
        "CompositeIndexVolumePlusVol15_vs_CompositePlusVol15": paired_block_bootstrap(
            combined.loc["2018-01":, "return"],
            proxy_combined.loc["2018-01":, "return"],
        ),
    }
    (RESULTS / "market_structure_robustness.json").write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        output.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== BOOTSTRAP ===")
    print(json.dumps(bootstrap, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
