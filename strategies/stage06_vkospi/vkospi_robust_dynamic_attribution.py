from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import strategies.stage06_vkospi.vkospi_dynamic_risk_experiment as base
import strategies.stage06_vkospi.vkospi_robust_dynamic_experiment as robust
RESULTS = robust.RESULTS


def diagnostic_components(
    features: pd.DataFrame, level_threshold: float, shock_threshold: float
) -> dict[str, np.ndarray]:
    percentile = features["percentile_252"].fillna(features["percentile_126"])
    level = np.clip(
        (percentile.to_numpy(dtype=float) - level_threshold)
        / max(1 - level_threshold, 1e-6),
        0,
        1,
    )
    shock = np.clip(
        (features["shock_5"].to_numpy(dtype=float) - shock_threshold) / 2.5,
        0,
        1,
    )
    acceleration = np.clip(
        (features["acceleration_z5"].to_numpy(dtype=float) - shock_threshold) / 2.5,
        0,
        1,
    )
    level_shock = np.nan_to_num((0.40 * level + 0.35 * shock) / 0.75).clip(0, 1)
    selected = np.nan_to_num((0.40 * level + 0.35 * shock + 0.25 * acceleration).clip(0, 1))
    return {
        "LevelOnly": np.nan_to_num(level).clip(0, 1),
        "ShockOnly": np.nan_to_num(shock).clip(0, 1),
        "AccelerationOnly": np.nan_to_num(acceleration).clip(0, 1),
        "LevelShockRenormalized": level_shock,
        "SelectedAccelerationBlend": selected,
    }


def performance_row(
    experiment: str, period: str, monthly: pd.DataFrame
) -> dict[str, object]:
    metrics = base.performance_summary(monthly["return"])
    return {
        "Experiment": experiment,
        "Period": period,
        **metrics.to_dict(),
        "AvgTurnover": float(monthly["turnover"].mean()),
        "AvgStress": float(monthly["avg_stress"].mean()),
        "AvgTransfer": float(monthly["avg_transfer"].mean()),
    }


def main() -> None:
    reference = base.load_reference_weights()
    arrays = base.prepare_arrays(
        base.load_daily_open_levels(), reference, base.build_daily_vkospi_signals()
    )
    features = robust.align_features_to_arrays(
        robust.build_robust_daily_features(), arrays
    )
    old_report = json.loads(
        (RESULTS / "vkospi_dynamic_validation.json").read_text(encoding="utf-8")
    )
    new_report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(
            encoding="utf-8"
        )
    )
    old_config = base.DynamicRiskConfig(**old_report["winner"])
    winner_spec = robust.RobustStressConfig(**new_report["winner"])
    winner_config = winner_spec.dynamic_config()
    components = diagnostic_components(
        features, winner_spec.level_threshold, winner_spec.shock_threshold
    )
    selected_stress = components["SelectedAccelerationBlend"]

    _, neutral = base.simulate(arrays, None, keep_daily=False)
    old_daily, old_actual = base.simulate(arrays, old_config, keep_daily=True)

    step_configs = {
        "1_ExistingDynamic": (old_config, None),
        "2_RobustSignal_OldPolicy": (
            replace(winner_config, max_risk_transfer=0.25, rebalance_band=0.15),
            selected_stress,
        ),
        "3_RobustSignal_Transfer35": (
            replace(winner_config, max_risk_transfer=0.35, rebalance_band=0.15),
            selected_stress,
        ),
        "4_RobustWinner_Band20": (winner_config, selected_stress),
    }
    reconciled_paths: dict[str, pd.DataFrame] = {}
    daily_paths: dict[str, pd.DataFrame] = {"1_ExistingDynamic": old_daily}
    rows: list[dict[str, object]] = []
    for name, (config, stress) in step_configs.items():
        if name == "1_ExistingDynamic":
            actual = old_actual
            daily = old_daily
        else:
            daily, actual = base.simulate(
                arrays, config, keep_daily=True, stress_override=stress
            )
        daily_paths[name] = daily
        reconciled = base.reconcile_to_monthly_reference(reference, neutral, actual)
        reconciled_paths[name] = reconciled
        for period, start, end in (
            ("locked_2018_2026", base.TEST_START, reference.index.max()),
            ("full_2007_2026", reference.index.min(), reference.index.max()),
        ):
            rows.append(performance_row(name, period, reconciled.loc[start:end]))

    component_rows: list[dict[str, object]] = []
    for name, stress in components.items():
        _, actual = base.simulate(
            arrays, winner_config, keep_daily=False, stress_override=stress
        )
        reconciled = base.reconcile_to_monthly_reference(reference, neutral, actual)
        for period, start, end in (
            ("calibration_2007_2017", reference.index.min(), base.CAL_END),
            ("locked_2018_2026", base.TEST_START, reference.index.max()),
        ):
            component_rows.append(
                performance_row(name, period, reconciled.loc[start:end])
            )

    old_locked_daily = daily_paths["1_ExistingDynamic"].loc[
        daily_paths["1_ExistingDynamic"].index.to_period("M") >= base.TEST_START
    ]
    new_locked_daily = daily_paths["4_RobustWinner_Band20"].loc[
        daily_paths["4_RobustWinner_Band20"].index.to_period("M") >= base.TEST_START
    ]
    signal_stats = pd.DataFrame(
        {
            "ExistingDynamic": {
                "TradingDays": len(old_locked_daily),
                "StressPositiveDays": int((old_locked_daily["stress"] > 0).sum()),
                "StressAbove025Days": int((old_locked_daily["stress"] >= 0.25).sum()),
                "TransferPositiveDays": int(
                    (old_locked_daily["transfer_fraction"] > 0).sum()
                ),
                "AverageStress": float(old_locked_daily["stress"].mean()),
                "MaximumStress": float(old_locked_daily["stress"].max()),
                "AverageTransfer": float(
                    old_locked_daily["transfer_fraction"].mean()
                ),
                "MaximumTransfer": float(
                    old_locked_daily["transfer_fraction"].max()
                ),
                "AverageTurnover": float(old_locked_daily["turnover"].mean()),
            },
            "RobustWinner": {
                "TradingDays": len(new_locked_daily),
                "StressPositiveDays": int((new_locked_daily["stress"] > 0).sum()),
                "StressAbove025Days": int((new_locked_daily["stress"] >= 0.25).sum()),
                "TransferPositiveDays": int(
                    (new_locked_daily["transfer_fraction"] > 0).sum()
                ),
                "AverageStress": float(new_locked_daily["stress"].mean()),
                "MaximumStress": float(new_locked_daily["stress"].max()),
                "AverageTransfer": float(
                    new_locked_daily["transfer_fraction"].mean()
                ),
                "MaximumTransfer": float(
                    new_locked_daily["transfer_fraction"].max()
                ),
                "AverageTurnover": float(new_locked_daily["turnover"].mean()),
            },
        }
    ).rename_axis("Statistic").reset_index()

    old_monthly = reconciled_paths["1_ExistingDynamic"].loc[base.TEST_START:].copy()
    new_monthly = reconciled_paths["4_RobustWinner_Band20"].loc[base.TEST_START:].copy()
    common = old_monthly.index.intersection(new_monthly.index)
    contributions = pd.DataFrame(index=common)
    contributions["ExistingReturn"] = old_monthly.loc[common, "return"]
    contributions["RobustReturn"] = new_monthly.loc[common, "return"]
    contributions["ReturnDelta"] = (
        contributions["RobustReturn"] - contributions["ExistingReturn"]
    )
    contributions["RelativeFactor"] = (
        (1 + contributions["RobustReturn"]) / (1 + contributions["ExistingReturn"])
    )
    contributions["CumulativeRelative"] = contributions["RelativeFactor"].cumprod() - 1
    contributions["ExistingAvgStress"] = old_monthly.loc[common, "avg_stress"]
    contributions["RobustAvgStress"] = new_monthly.loc[common, "avg_stress"]
    contributions["ExistingAvgTransfer"] = old_monthly.loc[common, "avg_transfer"]
    contributions["RobustAvgTransfer"] = new_monthly.loc[common, "avg_transfer"]
    contributions.index.name = "Month"

    pd.DataFrame(rows).to_csv(
        RESULTS / "vkospi_robust_dynamic_stepwise_attribution.csv", index=False
    )
    pd.DataFrame(component_rows).to_csv(
        RESULTS / "vkospi_robust_dynamic_component_ablation.csv", index=False
    )
    signal_stats.to_csv(
        RESULTS / "vkospi_robust_dynamic_signal_statistics.csv", index=False
    )
    contributions.to_csv(
        RESULTS / "vkospi_robust_dynamic_monthly_contribution.csv"
    )
    print("\n=== STEPWISE LOCKED ===")
    print(
        pd.DataFrame(rows)
        .loc[lambda frame: frame["Period"] == "locked_2018_2026"]
        [["Experiment", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .to_string(index=False, float_format=lambda value: f"{value:.5f}")
    )
    print("\n=== COMPONENT LOCKED ===")
    print(
        pd.DataFrame(component_rows)
        .loc[lambda frame: frame["Period"] == "locked_2018_2026"]
        [["Experiment", "CAGR", "Sharpe", "MDD", "Calmar"]]
        .to_string(index=False, float_format=lambda value: f"{value:.5f}")
    )
    print("\n=== SIGNAL STATISTICS ===")
    print(signal_stats.to_string(index=False))
    print("\n=== TOP/BOTTOM MONTHS ===")
    ranked = contributions.sort_values("ReturnDelta")
    print(
        pd.concat([ranked.head(5), ranked.tail(5)])[
            ["ExistingReturn", "RobustReturn", "ReturnDelta", "RobustAvgStress"]
        ].to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
