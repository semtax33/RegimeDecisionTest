from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strategies.core.regime_research import load_macro_data, load_monthly_asset_returns, performance_summary


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CAL_END = pd.Period("2017-12", freq="M")
TEST_START = pd.Period("2018-01", freq="M")

DOMESTIC_FEATURES = [
    "base_USO",
    "base_GLD",
    "base_KODEX200",
    "p_inflation_high",
    "proxy_mom1",
    "proxy_mom6",
    "proxy_vol6",
    "daily_mom21",
    "daily_mom252",
    "daily_vol21",
    "daily_downvol21",
    "daily_mean_corr63",
]
OAP_COMPOSITES = [
    "oap_momentum_trend_stress",
    "oap_reversal_crowding_stress",
    "oap_low_risk_tail_stress",
    "oap_liquidity_activity_stress",
]
TAIL_FEATURES = DOMESTIC_FEATURES + OAP_COMPOSITES


def make_model() -> Pipeline:
    """Reproduce the deployed balanced L2 logistic pipeline without heavy imports."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _metric_or_nan(function, y: pd.Series, values: pd.Series) -> float:
    if len(y) == 0 or y.nunique() < 2:
        return float("nan")
    return float(function(y, values))


def _ece(y: pd.Series, probability: pd.Series, bins: int = 5) -> float:
    frame = pd.DataFrame({"y": y, "p": probability}).dropna()
    if frame.empty:
        return float("nan")
    labels = pd.cut(frame["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    value = 0.0
    for _, group in frame.groupby(labels, observed=True):
        if len(group):
            value += len(group) / len(frame) * abs(group["y"].mean() - group["p"].mean())
    return float(value)


def _macro_components() -> tuple[pd.DataFrame, pd.DataFrame]:
    macro, core = load_macro_data()
    signals = pd.read_csv(RESULTS / "regime_signals.csv", index_col=0)
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    rows: list[dict[str, float | pd.Period]] = []
    for target_month, signal in signals.iterrows():
        signal_month = pd.Period(str(signal["signal_month"]), freq="M")
        features = macro.loc[signal_month.to_timestamp("M")]
        rows.append(
            {
                "target_month": target_month,
                "signal_month": signal_month,
                "growth_level": float(
                    features["growth"][["GDP_level", "Export_level", "BSI_level"]].mean()
                ),
                "growth_d3": float(
                    features["growth"][["GDP_level_d3", "Export_level_d3", "BSI_level_d3"]].mean()
                ),
                "inflation_level": float(
                    features["inflation"][["CPI_level", "PPI_level", "ImportPrice_level"]].mean()
                ),
                "inflation_d3": float(
                    features["inflation"][["CPI_level_d3", "PPI_level_d3", "ImportPrice_level_d3"]].mean()
                ),
                "growth_sjm": float(signal["p_growth_sjm"]),
                "inflation_sjm": float(signal["p_inflation_sjm"]),
            }
        )
    components = pd.DataFrame(rows).set_index("target_month")

    realized = pd.DataFrame(
        {
            "growth": core[["GDP", "Export", "BSI"]].mean(axis=1),
            "inflation": core[["CPI", "PPI", "ImportPrice"]].mean(axis=1),
        }
    )
    target = pd.DataFrame(index=realized.index, dtype=float)
    for name in ("growth", "inflation"):
        future = pd.concat([realized[name].shift(-step) for step in (1, 2, 3)], axis=1)
        target[name] = (future.mean(axis=1) >= 0).where(future.notna().sum(axis=1) == 3)
    return components, target


def _macro_probabilities(
    components: pd.DataFrame,
    d3_weight: float,
    sigmoid_scale: float,
    sjm_weight: float,
    current_weight: float,
) -> pd.DataFrame:
    output = pd.DataFrame(index=components.index, dtype=float)
    previous = {"growth": 0.5, "inflation": 0.5}
    for target_month, row in components.iterrows():
        for name in ("growth", "inflation"):
            composite = float(
                expit((row[f"{name}_level"] + d3_weight * row[f"{name}_d3"]) / sigmoid_scale)
            )
            raw = sjm_weight * row[f"{name}_sjm"] + (1 - sjm_weight) * composite
            probability = current_weight * raw + (1 - current_weight) * previous[name]
            output.loc[target_month, name] = probability
            previous[name] = probability
    return output


def _macro_metric_rows(
    probabilities: pd.DataFrame,
    components: pd.DataFrame,
    targets: pd.DataFrame,
    parameter: str,
    value: float,
    settings: dict[str, float],
) -> list[dict[str, object]]:
    joined = probabilities.copy()
    for name in ("growth", "inflation"):
        joined[f"y_{name}"] = [
            targets.loc[signal_month.to_timestamp("M"), name]
            if signal_month.to_timestamp("M") in targets.index
            else np.nan
            for signal_month in components["signal_month"]
        ]
    rows: list[dict[str, object]] = []
    for period, mask in (
        ("calibration_through_2017", joined.index <= CAL_END),
        ("locked_2018_2026", joined.index >= TEST_START),
    ):
        view = joined.loc[mask].dropna()
        record: dict[str, object] = {
            "parameter": parameter,
            "value": value,
            "period": period,
            "observations": len(view),
            **settings,
        }
        for name in ("growth", "inflation"):
            y = view[f"y_{name}"].astype(int)
            p = view[name].clip(1e-6, 1 - 1e-6)
            record[f"{name}_auc"] = _metric_or_nan(roc_auc_score, y, p)
            record[f"{name}_brier"] = float(brier_score_loss(y, p))
            record[f"{name}_balanced_accuracy"] = float(
                balanced_accuracy_score(y, p >= 0.5)
            )
        record["quadrant_accuracy"] = float(
            (
                (view["growth"] >= 0.5) == view["y_growth"].astype(bool)
            ).mul((view["inflation"] >= 0.5) == view["y_inflation"].astype(bool)).mean()
        )
        record["mean_brier"] = float(
            (record["growth_brier"] + record["inflation_brier"]) / 2
        )
        rows.append(record)
    return rows


def macro_constant_sensitivity() -> pd.DataFrame:
    components, targets = _macro_components()
    base = {
        "d3_weight": 0.20,
        "sigmoid_scale": 0.55,
        "sjm_weight": 0.10,
        "current_weight": 0.85,
    }
    variants: list[tuple[str, float, dict[str, float]]] = [("deployed", np.nan, base.copy())]
    grids = {
        "d3_weight": (0.0, 0.10, 0.20, 0.30, 0.40),
        "sigmoid_scale": (0.35, 0.45, 0.55, 0.65, 0.75),
        "sjm_weight": (0.0, 0.05, 0.10, 0.15, 0.20),
        "current_weight": (0.70, 0.80, 0.85, 0.90, 1.00),
    }
    for parameter, values in grids.items():
        for value in values:
            if math.isclose(value, base[parameter]):
                continue
            settings = base.copy()
            settings[parameter] = value
            variants.append((parameter, value, settings))
    rows: list[dict[str, object]] = []
    for parameter, value, settings in variants:
        probabilities = _macro_probabilities(components, **settings)
        rows.extend(
            _macro_metric_rows(
                probabilities,
                components,
                targets,
                parameter,
                value,
                settings,
            )
        )
    return pd.DataFrame(rows)


def _tail_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    domestic = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    factor = pd.read_csv(RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0)
    for frame in (domestic, composites, factor):
        frame.index = pd.PeriodIndex(frame.index, freq="M")
    data = domestic[DOMESTIC_FEATURES].join(composites[OAP_COMPOSITES]).join(
        factor[["tail_event"]]
    )
    probability = pd.Series(np.nan, index=data.index, dtype=float)
    coefficient_rows: list[dict[str, object]] = []
    template = make_model()
    for number, month in enumerate(data.index):
        train_end = number - 2
        if train_end < 36:
            continue
        train = data.iloc[:train_end].dropna(subset=["tail_event"])
        y = train["tail_event"].astype(int)
        if y.sum() < 4 or (len(y) - y.sum()) < 12:
            continue
        model = clone(template)
        model.fit(train[TAIL_FEATURES], y)
        probability.loc[month] = float(model.predict_proba(data.loc[[month], TAIL_FEATURES])[:, 1][0])
        coefficients = model.named_steps["model"].coef_[0]
        for feature, coefficient in zip(TAIL_FEATURES, coefficients):
            coefficient_rows.append(
                {"month": month, "feature": feature, "coefficient": float(coefficient)}
            )
    coefficient_frame = pd.DataFrame(coefficient_rows)
    if not coefficient_frame.empty:
        coefficient_frame["month"] = pd.PeriodIndex(coefficient_frame["month"], freq="M")
    data["refit_probability"] = probability
    return data, coefficient_frame


def tail_feature_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    data, coefficients = _tail_data()
    stored = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0
    )
    stored.index = pd.PeriodIndex(stored.index, freq="M")
    comparison = data["refit_probability"].dropna().to_frame("refit").join(
        stored[["p_tail_raw"]].rename(columns={"p_tail_raw": "stored"}), how="inner"
    ).dropna()
    reproduction = {
        "observations": int(len(comparison)),
        "max_absolute_probability_difference": float(
            (comparison["refit"] - comparison["stored"]).abs().max()
        ),
    }
    feature_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    calibration_rate = float(
        data.loc[data.index <= CAL_END].dropna(subset=["refit_probability", "tail_event"])[
            "tail_event"
        ].mean()
    )
    for period, mask in (
        ("calibration_through_2017", data.index <= CAL_END),
        ("locked_2018_2026", data.index >= TEST_START),
    ):
        view = data.loc[mask].dropna(subset=["refit_probability", "tail_event"])
        y = view["tail_event"].astype(int)
        p = view["refit_probability"].clip(1e-6, 1 - 1e-6)
        cutoff = float(p.quantile(0.80))
        selected = p >= cutoff
        prediction_rows.append(
            {
                "period": period,
                "observations": len(view),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "roc_auc": _metric_or_nan(roc_auc_score, y, p),
                "average_precision": _metric_or_nan(average_precision_score, y, p),
                "brier_score": float(brier_score_loss(y, p)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "ece_5bin": _ece(y, p, bins=5),
                "calibration_prevalence_brier": float(
                    np.mean((y - calibration_rate) ** 2)
                ),
                "oracle_period_prevalence_brier": float(y.mean() * (1 - y.mean())),
                "recall_at_top_20pct": float(y[selected].sum() / max(y.sum(), 1)),
                "precision_at_top_20pct": float(y[selected].mean()),
                "mean_probability_event": float(p[y == 1].mean()),
                "mean_probability_nonevent": float(p[y == 0].mean()),
            }
        )
        period_coefficients = coefficients.loc[
            (coefficients["month"] <= CAL_END)
            if period.startswith("calibration")
            else (coefficients["month"] >= TEST_START)
        ]
        for feature in TAIL_FEATURES:
            feature_view = view[[feature, "tail_event"]].dropna()
            fy = feature_view["tail_event"].astype(int)
            raw_auc = _metric_or_nan(roc_auc_score, fy, feature_view[feature])
            values = period_coefficients.loc[
                period_coefficients["feature"] == feature, "coefficient"
            ]
            median_coefficient = float(values.median()) if len(values) else np.nan
            median_sign = np.sign(median_coefficient)
            sign_stability = (
                float((np.sign(values) == median_sign).mean()) if len(values) else np.nan
            )
            feature_rows.append(
                {
                    "period": period,
                    "feature": feature,
                    "group": "domestic" if feature in DOMESTIC_FEATURES else "oap_composite",
                    "observations": len(feature_view),
                    "missing_rate": float(view[feature].isna().mean()),
                    "raw_univariate_auc": raw_auc,
                    "direction_free_auc": max(raw_auc, 1 - raw_auc)
                    if np.isfinite(raw_auc)
                    else np.nan,
                    "median_standardized_logit_coefficient": median_coefficient,
                    "median_absolute_coefficient": abs(median_coefficient)
                    if np.isfinite(median_coefficient)
                    else np.nan,
                    "coefficient_sign_stability": sign_stability,
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(prediction_rows), reproduction


def extended_performance() -> pd.DataFrame:
    existing = pd.read_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col="month"
    )
    robust = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col="month"
    )
    existing.index = pd.PeriodIndex(existing.index, freq="M")
    robust.index = pd.PeriodIndex(robust.index, freq="M")
    paths = {
        "ReferenceMediumHorizonOAPVol15": robust["reference_return"],
        "ExistingVKOSPIDynamic": existing["return"],
        "RobustVKOSPIDynamic": robust["return"],
    }
    rows: list[dict[str, object]] = []
    for strategy in paths:
        rows.append(
            {
                "period": "requested_2005_01_2026_07",
                "strategy": strategy,
                "start": "2005-01",
                "end": "2026-07",
                "status": "unavailable_same_strategy",
                "note": "Four-asset common returns begin 2006-04 and the 24-month regime warm-up makes 2007-04 the first tradable month.",
            }
        )
    periods = (
        ("available_full_2007_04_2026_07", pd.Period("2007-04", "M"), pd.Period("2026-07", "M")),
        ("early_calibration_2007_04_2012_12", pd.Period("2007-04", "M"), pd.Period("2012-12", "M")),
        ("validation_2013_01_2017_12", pd.Period("2013-01", "M"), pd.Period("2017-12", "M")),
        ("locked_2018_01_2026_07", pd.Period("2018-01", "M"), pd.Period("2026-07", "M")),
        ("locked_early_2018_01_2021_12", pd.Period("2018-01", "M"), pd.Period("2021-12", "M")),
        ("locked_late_2022_01_2026_07", pd.Period("2022-01", "M"), pd.Period("2026-07", "M")),
    )
    for period, start, end in periods:
        for strategy, returns in paths.items():
            view = returns.loc[start:end].dropna()
            metrics = performance_summary(view)
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "start": str(view.index.min()),
                    "end": str(view.index.max()),
                    "status": "measured",
                    "note": "same reconciled strategy path",
                    **metrics.to_dict(),
                }
            )

    _, levels = load_monthly_asset_returns(False)
    kodex = levels["KODEX200"].shift(-1).div(levels["KODEX200"]).sub(1).dropna()
    kodex = kodex.loc[pd.Period("2005-01", "M") : pd.Period("2026-07", "M")]
    metrics = performance_summary(kodex)
    rows.append(
        {
            "period": "requested_2005_01_2026_07",
            "strategy": "KODEX200ProxyBenchmark",
            "start": str(kodex.index.min()),
            "end": str(kodex.index.max()),
            "status": "measured_benchmark_only",
            "note": "KOSPI200 proxy benchmark; not the four-asset strategy and no overlay or costs.",
            **metrics.to_dict(),
        }
    )
    return pd.DataFrame(rows)


def overfitting_diagnostics(performance: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    calibration = pd.read_csv(RESULTS / "vkospi_robust_dynamic_calibration.csv")
    report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(encoding="utf-8")
    )
    winner = calibration.loc[calibration["Selected"]].iloc[0]
    axes = [
        "mode",
        "level_threshold",
        "shock_threshold",
        "max_risk_transfer",
        "bond_share",
        "rebalance_band",
    ]
    distance = pd.Series(0, index=calibration.index)
    for axis in axes:
        distance += calibration[axis].astype(str).ne(str(winner[axis])).astype(int)
    neighborhood = calibration.loc[distance <= 1].copy()
    neighborhood["grid_distance_from_winner"] = distance.loc[neighborhood.index]
    neighborhood["strict_pass"] = (
        (neighborhood["Cal_CAGRDelta"] > 0)
        & (neighborhood["Cal_SharpeDelta"] > 0)
        & (neighborhood["Cal_MDDDelta"] >= 0)
        & (neighborhood["Validation_CAGRDelta"] > 0)
        & (neighborhood["Validation_SharpeDelta"] > 0)
        & (neighborhood["Validation_MDDDelta"] >= 0)
        & (neighborhood["AvgStress"] > 0.002)
    )
    neighborhood = neighborhood.sort_values("MultiObjectiveScore", ascending=False)

    strict = calibration.loc[
        (calibration["Cal_CAGRDelta"] > 0)
        & (calibration["Cal_SharpeDelta"] > 0)
        & (calibration["Cal_MDDDelta"] >= 0)
        & (calibration["Validation_CAGRDelta"] > 0)
        & (calibration["Validation_SharpeDelta"] > 0)
        & (calibration["Validation_MDDDelta"] >= 0)
        & (calibration["AvgStress"] > 0.002)
    ].sort_values("MultiObjectiveScore", ascending=False)
    runner = strict.iloc[1] if len(strict) > 1 else None

    existing = pd.read_csv(
        RESULTS / "vkospi_dynamic_reconciled_monthly.csv", index_col="month"
    )["return"]
    robust = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col="month"
    )["return"]
    existing.index = pd.PeriodIndex(existing.index, freq="M")
    robust.index = pd.PeriodIndex(robust.index, freq="M")
    yearly = pd.DataFrame({"existing": existing, "robust": robust}).loc[TEST_START:]
    annual = yearly.groupby(yearly.index.year).apply(
        lambda frame: pd.Series(
            {
                "existing": (1 + frame["existing"]).prod() - 1,
                "robust": (1 + frame["robust"]).prod() - 1,
            }
        )
    )
    annual["delta"] = annual["robust"] - annual["existing"]

    available = performance.loc[
        performance["period"].isin(
            [
                "early_calibration_2007_04_2012_12",
                "validation_2013_01_2017_12",
                "locked_early_2018_01_2021_12",
                "locked_late_2022_01_2026_07",
            ]
        )
    ].copy()
    pivot = available.pivot(index="period", columns="strategy", values=["CAGR", "Sharpe", "MDD"])
    subperiod_deltas: dict[str, dict[str, float]] = {}
    for period in pivot.index:
        subperiod_deltas[period] = {
            metric: float(
                pivot.loc[period, (metric, "RobustVKOSPIDynamic")]
                - pivot.loc[period, (metric, "ExistingVKOSPIDynamic")]
            )
            for metric in ("CAGR", "Sharpe", "MDD")
        }

    output = {
        "candidate_count": int(len(calibration)),
        "strict_pass_count": int(len(strict)),
        "strict_pass_rate": float(len(strict) / len(calibration)),
        "selection_windows": {
            "outer": "2007-04 through 2017-12",
            "inner_validation": "2013-01 through 2017-12",
            "warning": "The inner window is nested inside the outer window, so the two gates are not independent tests.",
        },
        "winner_score": float(winner["MultiObjectiveScore"]),
        "runner_up_score": float(runner["MultiObjectiveScore"]) if runner is not None else None,
        "winner_runner_score_gap": float(
            winner["MultiObjectiveScore"] - runner["MultiObjectiveScore"]
        )
        if runner is not None
        else None,
        "one_axis_neighborhood_count": int(len(neighborhood)),
        "one_axis_neighborhood_strict_count": int(neighborhood["strict_pass"].sum()),
        "locked_bootstrap": report["locked"]["bootstrap"],
        "locked_years_robust_outperformed": int((annual["delta"] > 0).sum()),
        "locked_years_total": int(len(annual)),
        "subperiod_deltas_robust_minus_existing": subperiod_deltas,
        "controls_present": [
            "all coded selection ends at 2017-12",
            "2018-01 onward is a locked evaluation window",
            "strict CAGR, Sharpe and MDD gates are required in both pre-lock windows",
            "paired six-month block bootstrap and doubled-cost audit are reported",
        ],
        "residual_overfitting_risks": [
            "810 candidates create multiple-comparison selection pressure",
            "the inner 2013-2017 window is a subset of 2007-2017, not an independent validation set",
            "the winning score is only slightly above the runner-up score",
            "the macro constants and within-mode stress weights were fixed heuristics, not preregistered estimates",
            "revised macro data and repeated project-level experimentation can create researcher degrees of freedom",
            "no deflated Sharpe ratio, White reality check, CSCV/PBO, or external-market replication has been completed",
        ],
        "conclusion": "Code-level leakage controls are present, but overfitting cannot be ruled out. Evidence is suggestive rather than confirmatory.",
    }
    annual.reset_index(names="year").to_csv(
        RESULTS / "vkospi_locked_annual_relative_performance.csv", index=False
    )
    return neighborhood, output


def main() -> None:
    macro = macro_constant_sensitivity()
    feature, prediction, reproduction = tail_feature_diagnostics()
    performance = extended_performance()
    neighborhood, overfit = overfitting_diagnostics(performance)
    overfit["tail_probability_reproduction"] = reproduction

    macro.to_csv(RESULTS / "vkospi_macro_constant_sensitivity.csv", index=False)
    feature.to_csv(RESULTS / "vkospi_tail_feature_diagnostics.csv", index=False)
    prediction.to_csv(RESULTS / "vkospi_tail_prediction_diagnostics.csv", index=False)
    performance.to_csv(RESULTS / "vkospi_extended_period_performance.csv", index=False)
    neighborhood.to_csv(RESULTS / "vkospi_robust_grid_neighborhood.csv", index=False)
    (RESULTS / "vkospi_overfitting_diagnostics.json").write_text(
        json.dumps(overfit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("macro sensitivity", macro.shape)
    print("tail features", feature.shape)
    print("tail prediction", prediction.shape)
    print("period performance", performance.shape)
    print("grid neighborhood", neighborhood.shape)
    print(json.dumps(overfit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
