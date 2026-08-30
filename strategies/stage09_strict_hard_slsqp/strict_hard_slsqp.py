from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    build_daily_vkospi_signals,
    load_daily_open_levels,
    prepare_arrays,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
    load_vkospi_daily as load_zero_vkospi_daily,
)
from strategies.stage08_vkospi_factorial.factorial_bridge import (
    run_monthly_path,
    simulate_zero_policy,
    zero_macro_signals,
    zero_vkospi_stress,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FULL_START = pd.Period("2007-04", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")
HARD_SHARE = 0.40
SLSQP_SHARE = 0.60


STRICT_HARD_MAPPING = {
    "Goldilocks": np.array([1.0, 0.0, 0.0, 0.0]),
    "Overheating": np.array([0.0, 0.0, 0.0, 1.0]),
    "Slowdown": np.array([0.0, 1.0, 0.0, 0.0]),
    "Stagflation": np.array([0.0, 0.0, 1.0, 0.0]),
}


def strict_hard_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """Map every regime to exactly one asset with a 100/0/0/0 allocation."""
    rows: list[dict[str, object]] = []
    for month, signal in signals.iterrows():
        weights = STRICT_HARD_MAPPING[str(signal["regime"])]
        rows.append(
            {
                "month": month,
                "regime": signal["regime"],
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    values = output[weight_columns].to_numpy(dtype=float)
    assert np.isin(values, [0.0, 1.0]).all()
    assert np.allclose(values.sum(axis=1), 1.0)
    assert np.all((values == 1.0).sum(axis=1) == 1)
    return output


def build_hard_slsqp_weights(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    hard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend strict hard 40% with the current causal SLSQP path 60%."""
    slsqp = run_backtest(
        returns, signals, StrategyConfig(), mode="proposed"
    )
    months = hard.index.intersection(slsqp.index).intersection(returns.index)
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    output = (
        HARD_SHARE * hard.loc[months, weight_columns]
        + SLSQP_SHARE * slsqp.loc[months, weight_columns]
    )
    assert np.allclose(output.sum(axis=1), 1.0)
    return output, slsqp


def apply_zero_tune_vkospi(
    monthly_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only the original Zero-Tune expanding-percentile daily overlay."""
    levels = load_daily_open_levels()
    arrays = prepare_arrays(
        levels, monthly_reference, build_daily_vkospi_signals()
    )
    zero_vkospi = load_zero_vkospi_daily()
    stress = zero_vkospi_stress(arrays, zero_vkospi)
    daily, monthly = simulate_zero_policy(arrays, stress)
    valid = daily["signal_date"].notna()
    assert (
        daily.index[valid].to_numpy()
        > pd.DatetimeIndex(daily.loc[valid, "signal_date"]).to_numpy()
    ).all()
    return daily, monthly


def metric_record(
    strategy: str,
    path: pd.DataFrame,
    period: str,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, object]:
    view = path.loc[start:end]
    metrics = performance_summary(view["return"])
    return {
        "Strategy": strategy,
        "Period": period,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{name: float(value) for name, value in metrics.items()},
        "AvgTurnover": (
            float(view["turnover"].mean()) if "turnover" in view else np.nan
        ),
    }


def load_reference_paths() -> dict[str, pd.DataFrame]:
    paths: dict[str, pd.DataFrame] = {}
    for name, relative in {
        "ZeroTune_VKOSPI": (
            "strategies/stage07_zero_tune_vkospi/outputs/zero_tune_monthly.csv"
        ),
        "Current_Robust_VKOSPI": (
            "results/balanced_logistic_no_sjm_final_reconciled.csv"
        ),
    }.items():
        frame = pd.read_csv(ROOT / relative, index_col=0)
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        paths[name] = frame
    return paths


def run_strict_hard_slsqp(save: bool = True) -> dict[str, object]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, macro_ranks = build_macro_probabilities(returns)
    signals = zero_macro_signals(probabilities)
    hard = strict_hard_weights(signals)
    blended_weights, slsqp = build_hard_slsqp_weights(
        returns, signals, hard
    )

    hard_monthly = run_monthly_path(
        returns, hard, factor=None, use_vol_target=False
    )
    blended_monthly = run_monthly_path(
        returns, blended_weights, factor=None, use_vol_target=False
    )
    hard_daily, hard_zero_vkospi = apply_zero_tune_vkospi(hard_monthly)
    blended_daily, blended_zero_vkospi = apply_zero_tune_vkospi(
        blended_monthly
    )

    paths = {
        "StrictHard100_Monthly": hard_monthly,
        "StrictHard100_ZeroTuneVKOSPI": hard_zero_vkospi,
        "StrictHard40_SLSQP60_Monthly": blended_monthly,
        "StrictHard40_SLSQP60_ZeroTuneVKOSPI": blended_zero_vkospi,
        **load_reference_paths(),
    }
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, object]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            rows.append(
                metric_record(name, path, period, start, common_end)
            )
    comparison = pd.DataFrame(rows)

    primary_full = comparison.loc[
        comparison["Period"].eq("full_2007_2026")
        & comparison["Strategy"].eq(
            "StrictHard40_SLSQP60_Monthly"
        )
    ].iloc[0]
    primary_locked = comparison.loc[
        comparison["Period"].eq("locked_2018_2026")
        & comparison["Strategy"].eq(
            "StrictHard40_SLSQP60_Monthly"
        )
    ].iloc[0]
    overlay_full = comparison.loc[
        comparison["Period"].eq("full_2007_2026")
        & comparison["Strategy"].eq(
            "StrictHard40_SLSQP60_ZeroTuneVKOSPI"
        )
    ].iloc[0]
    overlay_locked = comparison.loc[
        comparison["Period"].eq("locked_2018_2026")
        & comparison["Strategy"].eq(
            "StrictHard40_SLSQP60_ZeroTuneVKOSPI"
        )
    ].iloc[0]
    report = {
        "strategy": "StrictHard40_SLSQP60_Monthly",
        "definition": {
            "macro": "Zero-Tune expanding empirical macro ranks",
            "hard_mapping": {
                regime: {
                    asset: float(weights[index])
                    for index, asset in enumerate(ASSETS)
                }
                for regime, weights in STRICT_HARD_MAPPING.items()
            },
            "base": {
                "strict_hard_share": HARD_SHARE,
                "slsqp_share": SLSQP_SHARE,
            },
            "later_current_layers": {
                "tail_logistic": False,
                "vol_target": False,
                "leverage": False,
                "robust_vkospi_features": False,
                "current_overlay_policy": False,
                "frequency_reconciliation": False,
            },
            "vkospi": (
                "Original Zero-Tune expanding percentile and proportional "
                "equal bond/gold transfer"
            ),
        },
        "full_period": {
            name: float(primary_full[name])
            for name in ["Months", "CAGR", "Sharpe", "MDD", "Calmar"]
        },
        "locked_period": {
            name: float(primary_locked[name])
            for name in ["Months", "CAGR", "Sharpe", "MDD", "Calmar"]
        },
        "zero_tune_vkospi_overlay_variant": {
            "full_period": {
                name: float(overlay_full[name])
                for name in ["Months", "CAGR", "Sharpe", "MDD", "Calmar"]
            },
            "locked_period": {
                name: float(overlay_locked[name])
                for name in ["Months", "CAGR", "Sharpe", "MDD", "Calmar"]
            },
        },
        "checks": {
            "hard_weights_are_binary": True,
            "one_asset_has_100_percent_each_month": True,
            "macro_signal_precedes_target_month": bool(
                (signals["signal_month"] < signals.index).all()
            ),
            "vkospi_signal_precedes_action_date": True,
            "asset_weights_unlevered_before_daily_overlay": bool(
                np.allclose(blended_weights.sum(axis=1), 1.0)
            ),
        },
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probabilities.to_csv(OUTPUT_DIR / "zero_tune_macro_probabilities.csv")
        macro_ranks.to_csv(OUTPUT_DIR / "zero_tune_macro_ranks.csv")
        hard.to_csv(OUTPUT_DIR / "strict_hard_weights.csv")
        slsqp.to_csv(OUTPUT_DIR / "slsqp_path.csv")
        blended_weights.to_csv(
            OUTPUT_DIR / "strict_hard40_slsqp60_weights.csv"
        )
        hard_monthly.to_csv(OUTPUT_DIR / "strict_hard_monthly.csv")
        hard_zero_vkospi.to_csv(
            OUTPUT_DIR / "strict_hard_zero_tune_vkospi_monthly.csv"
        )
        blended_monthly.to_csv(
            OUTPUT_DIR / "strict_hard40_slsqp60_monthly.csv"
        )
        blended_zero_vkospi.to_csv(
            OUTPUT_DIR
            / "strict_hard40_slsqp60_zero_tune_vkospi_monthly.csv"
        )
        hard_daily.to_csv(
            OUTPUT_DIR / "strict_hard_zero_tune_vkospi_daily.csv"
        )
        blended_daily.to_csv(
            OUTPUT_DIR
            / "strict_hard40_slsqp60_zero_tune_vkospi_daily.csv"
        )
        comparison.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        (OUTPUT_DIR / "strict_hard_slsqp_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "probabilities": probabilities,
        "signals": signals,
        "hard": hard,
        "slsqp": slsqp,
        "blended_weights": blended_weights,
        "hard_monthly": hard_monthly,
        "hard_daily": hard_daily,
        "hard_zero_vkospi": hard_zero_vkospi,
        "blended_monthly": blended_monthly,
        "blended_daily": blended_daily,
        "blended_zero_vkospi": blended_zero_vkospi,
        "comparison": comparison,
        "report": report,
    }


def main() -> None:
    result = run_strict_hard_slsqp(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print("saved", OUTPUT_DIR)


if __name__ == "__main__":
    main()
