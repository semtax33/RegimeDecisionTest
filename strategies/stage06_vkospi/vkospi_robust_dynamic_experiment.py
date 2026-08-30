from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    CAL_END,
    TEST_START,
    DynamicRiskConfig,
    build_daily_vkospi_signals,
    causal_percentile,
    load_daily_open_levels,
    load_reference_weights,
    load_vkospi_daily,
    paired_multiobjective_bootstrap,
    performance_summary,
    prepare_arrays,
    reconcile_to_monthly_reference,
    simulate,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
VALIDATION_START = pd.Period("2013-01", freq="M")
RANDOM_STATE = 20260827


@dataclass(frozen=True)
class RobustStressConfig:
    mode: str
    level_threshold: float
    shock_threshold: float
    max_risk_transfer: float
    bond_share: float
    rebalance_band: float
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_lt{self.level_threshold:.2f}_st{self.shock_threshold:.2f}"
            f"_rt{self.max_risk_transfer:.2f}_bs{self.bond_share:.2f}"
            f"_rb{self.rebalance_band:.2f}"
        )

    def dynamic_config(self) -> DynamicRiskConfig:
        return DynamicRiskConfig(
            mode="level",
            level_threshold=self.level_threshold,
            momentum_window=5,
            spike_threshold=0.0,
            max_risk_transfer=self.max_risk_transfer,
            bond_share=self.bond_share,
            rebalance_band=self.rebalance_band,
            financing_rate=self.financing_rate,
        )


def robust_zscore(series: pd.Series, window: int, minimum: int) -> pd.Series:
    median = series.rolling(window, min_periods=minimum).median()
    mad = (series - median).abs().rolling(window, min_periods=minimum).median()
    return ((series - median) / (1.4826 * mad.replace(0, np.nan))).clip(-6, 6)


def build_robust_daily_features() -> pd.DataFrame:
    daily = load_vkospi_daily()
    close = daily["close"].astype(float)
    log_close = np.log(close.where(close > 0))
    log_return = log_close.diff()
    output = pd.DataFrame(index=daily.index)
    output["close"] = close
    output["percentile_126"] = causal_percentile(close, window=126, minimum=84)
    output["percentile_252"] = causal_percentile(close, window=252, minimum=126)
    output["robust_z_63"] = robust_zscore(log_close, 63, 42)
    output["robust_z_252"] = robust_zscore(log_close, 252, 126)
    for window in (5, 10, 21):
        scale = log_return.rolling(63, min_periods=42).std(ddof=1) * math.sqrt(window)
        output[f"shock_{window}"] = (
            log_close.diff(window) / scale.replace(0, np.nan)
        ).clip(-8, 8)
    output["acceleration_5"] = log_close.diff(5) - log_close.diff(5).shift(5)
    acceleration_scale = log_return.rolling(63, min_periods=42).std(ddof=1) * math.sqrt(5)
    output["acceleration_z5"] = (
        output["acceleration_5"] / acceleration_scale.replace(0, np.nan)
    ).clip(-8, 8)

    high = daily["high"].astype(float).where(daily["high"].astype(float) > 0, close)
    low = daily["low"].astype(float).where(daily["low"].astype(float) > 0, close)
    high21 = high.rolling(21, min_periods=10).max()
    low21 = low.rolling(21, min_periods=10).min()
    output["distance_high21"] = close / high21 - 1
    output["close_location21"] = (
        (close - low21) / (high21 - low21).replace(0, np.nan)
    ).clip(0, 1)
    output["positive_fraction5"] = (log_return > 0).rolling(5, min_periods=3).mean()
    output["positive_fraction21"] = (log_return > 0).rolling(21, min_periods=10).mean()
    output["fast_slow"] = log_close.diff(5) - 5 / 21 * log_close.diff(21)
    return output.replace([np.inf, -np.inf], np.nan)


def align_features_to_arrays(
    features: pd.DataFrame, arrays: dict[str, object]
) -> pd.DataFrame:
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])
    aligned = features.reindex(signal_dates)
    aligned.index = pd.RangeIndex(len(aligned))
    return aligned


def stress_from_features(
    aligned: pd.DataFrame,
    mode: str,
    level_threshold: float,
    shock_threshold: float,
) -> np.ndarray:
    percentile = aligned["percentile_252"].fillna(aligned["percentile_126"])
    level = np.clip(
        (percentile.to_numpy(dtype=float) - level_threshold)
        / max(1 - level_threshold, 1e-6),
        0,
        1,
    )
    shock_raw = aligned["shock_5"].to_numpy(dtype=float)
    shock = np.clip((shock_raw - shock_threshold) / 2.5, 0, 1)
    acceleration = np.clip(
        (aligned["acceleration_z5"].to_numpy(dtype=float) - shock_threshold) / 2.5,
        0,
        1,
    )
    confirmation = np.nan_to_num(
        aligned["close_location21"].to_numpy(dtype=float), nan=0.5
    ).clip(0, 1)
    distance = np.nan_to_num(
        aligned["distance_high21"].to_numpy(dtype=float), nan=0.0
    )
    falling = np.clip(-shock_raw / 2.5, 0, 1)
    exhaustion = np.clip(level * (-distance).clip(0, 0.5) / 0.5 * falling, 0, 1)
    if mode == "robust_mean":
        stress = 0.50 * level + 0.50 * shock
    elif mode == "robust_max":
        stress = np.maximum(level, shock)
    elif mode == "confirmed":
        stress = (0.45 * level + 0.55 * shock) * (0.35 + 0.65 * confirmation)
    elif mode == "acceleration":
        stress = 0.40 * level + 0.35 * shock + 0.25 * acceleration
    elif mode == "exhaustion_adjusted":
        stress = (0.50 * level + 0.50 * shock) * (1 - 0.75 * exhaustion)
    else:
        raise ValueError(mode)
    return np.nan_to_num(stress, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)


def metric_row(
    period: str,
    strategy: str,
    monthly: pd.DataFrame,
) -> dict[str, object]:
    metrics = performance_summary(monthly["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(monthly["turnover"].mean()),
        "AvgStress": float(monthly.get("avg_stress", pd.Series(0, index=monthly.index)).mean()),
        "AvgTransfer": float(monthly.get("avg_transfer", pd.Series(0, index=monthly.index)).mean()),
    }


def main() -> None:
    reference = load_reference_weights()
    levels = load_daily_open_levels()
    arrays = prepare_arrays(levels, reference, build_daily_vkospi_signals())
    robust_features = align_features_to_arrays(build_robust_daily_features(), arrays)
    _, baseline_monthly = simulate(arrays, None, keep_daily=False)
    existing_report = json.loads(
        (RESULTS / "vkospi_dynamic_validation.json").read_text(encoding="utf-8")
    )
    existing_cfg = DynamicRiskConfig(**existing_report["winner"])
    _, existing_monthly = simulate(arrays, existing_cfg, keep_daily=False)
    existing_cal = performance_summary(existing_monthly.loc[:CAL_END, "return"])
    existing_validation = performance_summary(
        existing_monthly.loc[VALIDATION_START:CAL_END, "return"]
    )

    stress_cache: dict[tuple[str, float, float], np.ndarray] = {}
    for mode, level_threshold, shock_threshold in itertools.product(
        ("robust_mean", "robust_max", "confirmed", "acceleration", "exhaustion_adjusted"),
        (0.70, 0.80, 0.90),
        (0.0, 0.5, 1.0),
    ):
        stress_cache[(mode, level_threshold, shock_threshold)] = stress_from_features(
            robust_features, mode, level_threshold, shock_threshold
        )

    rows = []
    paths: dict[str, pd.DataFrame] = {}
    configs = []
    for mode, level_threshold, shock_threshold, transfer, bond_share, band in itertools.product(
        ("robust_mean", "robust_max", "confirmed", "acceleration", "exhaustion_adjusted"),
        (0.70, 0.80, 0.90),
        (0.0, 0.5, 1.0),
        (0.15, 0.25, 0.35),
        (0.0, 0.5),
        (0.10, 0.15, 0.20),
    ):
        config = RobustStressConfig(
            mode,
            level_threshold,
            shock_threshold,
            transfer,
            bond_share,
            band,
        )
        stress = stress_cache[(mode, level_threshold, shock_threshold)]
        _, monthly = simulate(
            arrays,
            config.dynamic_config(),
            keep_daily=False,
            stress_override=stress,
        )
        paths[config.name] = monthly
        configs.append(config)
        cal = performance_summary(monthly.loc[:CAL_END, "return"])
        val = performance_summary(monthly.loc[VALIDATION_START:CAL_END, "return"])
        rows.append(
            {
                "Config": config.name,
                **asdict(config),
                **{f"Cal_{key}": value for key, value in cal.to_dict().items()},
                **{f"Validation_{key}": value for key, value in val.to_dict().items()},
                "Cal_CAGRDelta": float(cal["CAGR"] - existing_cal["CAGR"]),
                "Cal_SharpeDelta": float(cal["Sharpe"] - existing_cal["Sharpe"]),
                "Cal_MDDDelta": float(cal["MDD"] - existing_cal["MDD"]),
                "Validation_CAGRDelta": float(
                    val["CAGR"] - existing_validation["CAGR"]
                ),
                "Validation_SharpeDelta": float(
                    val["Sharpe"] - existing_validation["Sharpe"]
                ),
                "Validation_MDDDelta": float(
                    val["MDD"] - existing_validation["MDD"]
                ),
                "AvgTurnover": float(monthly.loc[:CAL_END, "turnover"].mean()),
                "AvgStress": float(monthly.loc[:CAL_END, "avg_stress"].mean()),
            }
        )
    calibration = pd.DataFrame(rows)
    for metric in (
        "Cal_CAGR",
        "Cal_Sharpe",
        "Cal_MDD",
        "Cal_Calmar",
        "Validation_CAGR",
        "Validation_Sharpe",
        "Validation_MDD",
        "Validation_Calmar",
    ):
        calibration[f"Rank_{metric}"] = calibration[metric].rank(pct=True)
    calibration["MultiObjectiveScore"] = calibration[
        [column for column in calibration if column.startswith("Rank_")]
    ].mean(axis=1)
    strict = calibration.loc[
        (calibration["Cal_CAGRDelta"] > 0)
        & (calibration["Cal_SharpeDelta"] > 0)
        & (calibration["Cal_MDDDelta"] >= 0)
        & (calibration["Validation_CAGRDelta"] > 0)
        & (calibration["Validation_SharpeDelta"] > 0)
        & (calibration["Validation_MDDDelta"] >= 0)
        & (calibration["AvgStress"] > 0.002)
    ]
    selection_rule = "all three improve versus existing dynamic in 2007-2017 and 2013-2017"
    eligible = strict
    if eligible.empty:
        eligible = calibration.loc[
            (calibration["Cal_CAGR"] >= 0.995 * existing_cal["CAGR"])
            & (calibration["Cal_SharpeDelta"] > 0)
            & (calibration["Cal_MDDDelta"] >= 0)
            & (calibration["Validation_CAGR"] >= 0.995 * existing_validation["CAGR"])
            & (calibration["Validation_SharpeDelta"] > 0)
            & (calibration["Validation_MDDDelta"] >= 0)
            & (calibration["AvgStress"] > 0.002)
        ]
        selection_rule = "Sharpe and MDD improve with at least 99.5% CAGR retention in both pre-2018 windows"
    if eligible.empty:
        eligible = calibration.loc[calibration["AvgStress"] > 0.002]
        selection_rule = "highest pre-2018 multiobjective rank; no strict improvement candidate"
    winner_row = eligible.sort_values(
        ["MultiObjectiveScore", "Cal_Sharpe", "Validation_Sharpe", "Cal_CAGR"],
        ascending=False,
    ).iloc[0]
    calibration["Selected"] = False
    calibration.loc[winner_row.name, "Selected"] = True
    winner = RobustStressConfig(
        mode=str(winner_row["mode"]),
        level_threshold=float(winner_row["level_threshold"]),
        shock_threshold=float(winner_row["shock_threshold"]),
        max_risk_transfer=float(winner_row["max_risk_transfer"]),
        bond_share=float(winner_row["bond_share"]),
        rebalance_band=float(winner_row["rebalance_band"]),
        financing_rate=float(winner_row["financing_rate"]),
    )
    winner_stress = stress_cache[
        (winner.mode, winner.level_threshold, winner.shock_threshold)
    ]
    winner_daily, winner_monthly = simulate(
        arrays,
        winner.dynamic_config(),
        keep_daily=True,
        stress_override=winner_stress,
    )

    existing_reconciled = pd.read_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col="month"
    )
    existing_reconciled.index = pd.PeriodIndex(existing_reconciled.index, freq="M")
    winner_reconciled = reconcile_to_monthly_reference(
        reference, baseline_monthly, winner_monthly
    )
    comparison_rows = []
    periods = (
        ("calibration_2007_2017", reference.index.min(), CAL_END),
        ("validation_2013_2017", VALIDATION_START, CAL_END),
        ("locked_2018_2026", TEST_START, reference.index.max()),
        ("full_2007_2026", reference.index.min(), reference.index.max()),
    )
    for period, start, end in periods:
        for strategy, monthly in (
            ("ExistingDynamicActual", existing_monthly),
            ("RobustDynamicActual", winner_monthly),
            ("ExistingDynamicReconciled", existing_reconciled),
            ("RobustDynamicReconciled", winner_reconciled),
        ):
            comparison_rows.append(metric_row(period, strategy, monthly.loc[start:end]))
    comparison = pd.DataFrame(comparison_rows)

    existing_locked = existing_reconciled.loc[TEST_START:]
    robust_locked = winner_reconciled.loc[TEST_START:]
    existing_locked_metrics = performance_summary(existing_locked["return"])
    robust_locked_metrics = performance_summary(robust_locked["return"])
    locked_deltas = {
        metric: float(robust_locked_metrics[metric] - existing_locked_metrics[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    cost_rows = []
    for multiplier in (1.0, 2.0):
        _, existing_cost = simulate(
            arrays, existing_cfg, cost_multiplier=multiplier, keep_daily=False
        )
        _, robust_cost = simulate(
            arrays,
            winner.dynamic_config(),
            cost_multiplier=multiplier,
            keep_daily=False,
            stress_override=winner_stress,
        )
        cost_rows.append(
            metric_row(
                f"cost_{multiplier:.1f}x_locked",
                "ExistingDynamic",
                existing_cost.loc[TEST_START:],
            )
        )
        cost_rows.append(
            metric_row(
                f"cost_{multiplier:.1f}x_locked",
                "RobustDynamic",
                robust_cost.loc[TEST_START:],
            )
        )

    report = {
        "objective": "Improve the existing VKOSPI dynamic overlay with robust daily transformations",
        "processing": {
            "level": "causal 126/252-day percentiles and rolling median/MAD z-scores",
            "shock": "5/10/21-day log changes normalized by trailing 63-day volatility",
            "path": "acceleration, 21-day high distance, close location and fast-minus-slow change",
            "states": "confirmation and exhaustion-adjusted continuous stress variants",
        },
        "calibration_end": str(CAL_END),
        "locked_start": str(TEST_START),
        "candidate_count": int(len(calibration)),
        "strict_eligible_count": int(len(strict)),
        "selection_rule": selection_rule,
        "winner": asdict(winner),
        "locked": {
            "existing": existing_locked_metrics.to_dict(),
            "robust": robust_locked_metrics.to_dict(),
            "deltas": locked_deltas,
            "passes_all_three": bool(
                locked_deltas["CAGR"] > 0
                and locked_deltas["Sharpe"] > 0
                and locked_deltas["MDD"] >= 0
            ),
            "bootstrap": paired_multiobjective_bootstrap(
                existing_locked["return"], robust_locked["return"]
            ),
        },
    }
    calibration.to_csv(
        RESULTS / "vkospi_robust_dynamic_calibration.csv", index=False
    )
    comparison.to_csv(RESULTS / "vkospi_robust_dynamic_comparison.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(
        RESULTS / "vkospi_robust_dynamic_cost_sensitivity.csv", index=False
    )
    winner_daily.to_csv(RESULTS / "vkospi_robust_dynamic_daily.csv")
    winner_monthly.to_csv(RESULTS / "vkospi_robust_dynamic_monthly.csv")
    winner_reconciled.to_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv"
    )
    (RESULTS / "vkospi_robust_dynamic_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== WINNER ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n=== COMPARISON ===")
    print(
        comparison[
            ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== COST SENSITIVITY ===")
    print(
        pd.DataFrame(cost_rows)[
            ["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
