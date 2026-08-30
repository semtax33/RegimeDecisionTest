from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from strategies.core.regime_research import (
    StrategyConfig,
    load_monthly_asset_returns,
    run_backtest,
)
from strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy import (
    CAL_END,
    TEST_START,
    VALIDATION_START,
    balanced_logistic_spec,
    build_domestic_features,
    build_no_sjm_signals,
    fixed_robust_overlay,
    forward_path_loss,
    metric_record,
    run_neutral_factor_blend,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    load_daily_open_levels,
    paired_multiobjective_bootstrap,
)
from strategies.stage06_vkospi.vkospi_extended_diagnostics import (
    DOMESTIC_FEATURES,
    OAP_COMPOSITES,
    TAIL_FEATURES,
)
from strategies.stage06_vkospi.vkospi_model_robustness import (
    causal_percentile,
    make_logistic_model,
    run_factor_vol_target,
)
from strategies.stage06_vkospi.vkospi_robust_dynamic_experiment import (
    build_robust_daily_features,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

BRIDGE_PANEL_PATH = RESULTS / "balanced_logistic_early_start_bridge_panel.csv"
FACTOR_PATH = RESULTS / "balanced_logistic_early_start_factor.csv"
MEDIUM_PATH = RESULTS / "balanced_logistic_early_start_medium_backtest.csv"
FINAL_DAILY_PATH = RESULTS / "balanced_logistic_early_start_final_daily.csv"
FINAL_MONTHLY_PATH = RESULTS / "balanced_logistic_early_start_final_monthly.csv"
FINAL_RECONCILED_PATH = RESULTS / "balanced_logistic_early_start_final_reconciled.csv"
COMPARISON_PATH = RESULTS / "balanced_logistic_early_start_comparison.csv"
REPORT_PATH = RESULTS / "balanced_logistic_early_start_validation.json"

LEGACY_FACTOR_PATH = RESULTS / "balanced_logistic_no_sjm_factor.csv"
LEGACY_MEDIUM_PATH = RESULTS / "balanced_logistic_no_sjm_medium_backtest.csv"
LEGACY_FINAL_PATH = RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv"

BRIDGE_FEATURES = [
    "vkospi_percentile_252",
    "vkospi_shock_5",
    "vkospi_acceleration_z5",
    "vkospi_robust_z_63",
    "vkospi_robust_z_252",
    "kospi200_momentum_21",
    "kospi200_momentum_63",
    "kospi200_momentum_252",
    "kospi200_volatility_21",
    "kospi200_downside_volatility_21",
]

MIN_TRAIN = 36
MIN_POSITIVE = 4
MIN_NEGATIVE = 12
EMBARGO_MONTHS = 2

PERIODS = (
    (
        "bridge_2007_04_2010_05",
        pd.Period("2007-04", freq="M"),
        pd.Period("2010-05", freq="M"),
    ),
    ("calibration_2007_2017", None, CAL_END),
    ("validation_2013_2017", VALIDATION_START, CAL_END),
    ("locked_2018_2026", TEST_START, None),
    ("full_2007_2026", None, None),
)


def build_vkospi_bridge_panel() -> pd.DataFrame:
    """Build a point-in-time VKOSPI/KOSPI200 panel beginning before 2007.

    Features for target month t use observations no later than the final calendar
    day before t. The target is the two-month KOSPI200-proxy path minimum, which
    supplies genuine pre-2007 labels without fabricating macro/OAP history.
    """
    vkospi = build_robust_daily_features()
    kospi200 = load_daily_open_levels()["KODEX200"].dropna().sort_index()
    kospi200_return = kospi200.pct_change(fill_method=None)
    rows: list[dict[str, object]] = []

    last_month = min(
        vkospi.index.max().to_period("M"),
        kospi200.index.max().to_period("M"),
    )
    for month in pd.period_range("2003-01", last_month, freq="M"):
        cutoff = month.to_timestamp(how="start") - pd.Timedelta(days=1)
        vkospi_history = vkospi.loc[:cutoff].dropna(
            subset=[
                "percentile_252",
                "shock_5",
                "acceleration_z5",
                "robust_z_63",
                "robust_z_252",
            ]
        )
        price_history = kospi200.loc[:cutoff]
        return_history = kospi200_return.loc[:cutoff].dropna()
        if vkospi_history.empty or len(price_history) < 253 or len(return_history) < 252:
            continue

        latest = vkospi_history.iloc[-1]
        rows.append(
            {
                "month": month,
                "feature_cutoff": cutoff,
                "vkospi_percentile_252": float(latest["percentile_252"]),
                "vkospi_shock_5": float(latest["shock_5"]),
                "vkospi_acceleration_z5": float(latest["acceleration_z5"]),
                "vkospi_robust_z_63": float(latest["robust_z_63"]),
                "vkospi_robust_z_252": float(latest["robust_z_252"]),
                "kospi200_momentum_21": float(
                    price_history.iloc[-1] / price_history.iloc[-22] - 1
                ),
                "kospi200_momentum_63": float(
                    price_history.iloc[-1] / price_history.iloc[-64] - 1
                ),
                "kospi200_momentum_252": float(
                    price_history.iloc[-1] / price_history.iloc[-253] - 1
                ),
                "kospi200_volatility_21": float(
                    return_history.tail(21).std(ddof=1) * math.sqrt(252)
                ),
                "kospi200_downside_volatility_21": float(
                    np.sqrt(np.mean(np.minimum(return_history.tail(21), 0.0) ** 2))
                    * math.sqrt(252)
                ),
            }
        )

    panel = pd.DataFrame(rows).set_index("month").dropna(subset=BRIDGE_FEATURES)
    panel.index = pd.PeriodIndex(panel.index, freq="M")

    first_open = kospi200.groupby(kospi200.index.to_period("M")).first()
    monthly_return = first_open.shift(-1).div(first_open).sub(1.0)
    path_loss = forward_path_loss(monthly_return, horizon=2)
    panel["proxy_tail_event"] = (
        (path_loss < -0.05).where(path_loss.notna()).astype(float).reindex(panel.index)
    )
    cutoff_period = pd.PeriodIndex(
        pd.to_datetime(panel["feature_cutoff"]), freq="M"
    )
    assert (cutoff_period < panel.index).all()
    return panel


def fit_vkospi_bridge_probability(
    panel: pd.DataFrame,
    prediction_months: pd.PeriodIndex,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float | int | str]]:
    """Fit the bridge with an expanding window and a two-month label embargo."""
    probability = pd.Series(np.nan, index=prediction_months, dtype=float)
    rows: list[dict[str, object]] = []
    warnings_count = 0
    spec = balanced_logistic_spec()

    for month in prediction_months:
        if month not in panel.index:
            raise ValueError(f"Bridge feature month is unavailable: {month}")
        number = int(panel.index.get_loc(month))
        train_end = number - EMBARGO_MONTHS
        train = panel.iloc[: max(train_end, 0)].dropna(subset=["proxy_tail_event"])
        target = train["proxy_tail_event"].astype(int)
        positive = int(target.sum())
        negative = int(len(target) - positive)
        ready = (
            train_end >= MIN_TRAIN
            and positive >= MIN_POSITIVE
            and negative >= MIN_NEGATIVE
        )
        if not ready:
            raise ValueError(
                f"Insufficient causal bridge history for {month}: "
                f"n={len(train)}, positive={positive}, negative={negative}"
            )

        model = make_logistic_model(spec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train[BRIDGE_FEATURES], target)
        warnings_count += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        probability.loc[month] = float(
            model.predict_proba(panel.loc[[month], BRIDGE_FEATURES])[:, 1][0]
        )
        latest_training_month = train.index[-1]
        rows.append(
            {
                "month": month,
                "bridge_train_observations": int(len(train)),
                "bridge_train_positive": positive,
                "bridge_train_negative": negative,
                "bridge_latest_training_month": latest_training_month,
            }
        )

    audit = pd.DataFrame(rows).set_index("month")
    audit.index = pd.PeriodIndex(audit.index, freq="M")
    assert probability.notna().all()
    assert probability.between(0, 1).all()
    assert all(
        pd.Period(value, freq="M") <= month - (EMBARGO_MONTHS + 1)
        for month, value in audit["bridge_latest_training_month"].items()
    )
    return probability, audit, {
        "prediction_count": int(len(probability)),
        "first_prediction_month": str(probability.index[0]),
        "last_prediction_month": str(probability.index[-1]),
        "first_train_observations": int(audit.iloc[0]["bridge_train_observations"]),
        "first_train_positive": int(audit.iloc[0]["bridge_train_positive"]),
        "first_train_negative": int(audit.iloc[0]["bridge_train_negative"]),
        "convergence_warning_count": warnings_count,
    }


def make_early_start_factor(
    legacy_factor: pd.DataFrame,
    bridge_probability: pd.Series,
    bridge_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Join the early bridge to the mature factor without mixing probability scales."""
    output = legacy_factor.copy()
    mature = legacy_factor["p_tail_raw"].notna()
    bridge_months = output.index[~mature]
    if not bridge_months.equals(bridge_probability.index):
        raise ValueError("Bridge months must exactly fill the legacy probability warm-up")

    output.loc[bridge_months, "p_tail_raw"] = bridge_probability
    bridge_percentile = causal_percentile(bridge_probability)
    output.loc[bridge_months, "risk_percentile"] = bridge_percentile
    output.loc[bridge_months, "risk_severity"] = (
        (bridge_percentile - 0.80) / 0.20
    ).clip(0, 1)
    output.loc[bridge_months, "p_up"] = (
        0.50 - 0.15 * output.loc[bridge_months, "risk_severity"]
    )
    output.loc[bridge_months, "score"] = -output.loc[bridge_months, "risk_severity"]
    output["prediction_mode"] = np.where(
        mature, "standard_16_feature_expanding", "vkospi_pretrained_bridge"
    )
    return output.join(bridge_audit)


def prediction_metrics(frame: pd.DataFrame, target_name: str) -> dict[str, float | int]:
    view = frame.dropna(subset=["p_tail_raw", target_name])
    target = view[target_name].astype(int)
    probability = view["p_tail_raw"].clip(1e-6, 1 - 1e-6)
    return {
        "observations": int(len(view)),
        "events": int(target.sum()),
        "event_rate": float(target.mean()),
        "roc_auc": (
            float(roc_auc_score(target, probability))
            if target.nunique() > 1
            else np.nan
        ),
        "average_precision": (
            float(average_precision_score(target, probability))
            if target.nunique() > 1
            else np.nan
        ),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def _metric_delta(
    comparison: pd.DataFrame,
    period: str,
    early_name: str,
    legacy_name: str,
) -> dict[str, float]:
    view = comparison.loc[comparison["Period"].eq(period)].set_index("Strategy")
    return {
        metric: float(view.loc[early_name, metric] - view.loc[legacy_name, metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }


def main() -> None:
    returns, _ = load_monthly_asset_returns(False)
    signals, _ = build_no_sjm_signals(returns)
    defensive = run_backtest(returns, signals, StrategyConfig(), mode="proposed")
    domestic = build_domestic_features(signals, returns)
    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    composites.index = pd.PeriodIndex(composites.index, freq="M")

    neutral = run_neutral_factor_blend(returns, signals, defensive)
    data = domestic[DOMESTIC_FEATURES].join(composites[OAP_COMPOSITES], how="left")
    data = data.loc[data.index.intersection(neutral.index)].copy()
    path_loss = forward_path_loss(neutral.loc[data.index, "return"], horizon=2)
    data["tail_event"] = (path_loss < -0.05).where(path_loss.notna()).astype(float)

    legacy_factor = pd.read_csv(LEGACY_FACTOR_PATH, index_col=0)
    legacy_factor.index = pd.PeriodIndex(legacy_factor.index, freq="M")
    legacy_factor["tail_event"] = data["tail_event"].reindex(legacy_factor.index)
    bridge_months = legacy_factor.index[legacy_factor["p_tail_raw"].isna()]

    bridge_panel = build_vkospi_bridge_panel()
    bridge_probability, bridge_audit, fit_stats = fit_vkospi_bridge_probability(
        bridge_panel, bridge_months
    )
    factor = make_early_start_factor(
        legacy_factor, bridge_probability, bridge_audit
    )
    medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    final_daily, final_monthly, final_reconciled = fixed_robust_overlay(medium)

    legacy_medium = pd.read_csv(LEGACY_MEDIUM_PATH, index_col=0)
    legacy_medium.index = pd.PeriodIndex(legacy_medium.index, freq="M")
    legacy_final = pd.read_csv(LEGACY_FINAL_PATH, index_col=0)
    legacy_final.index = pd.PeriodIndex(legacy_final.index, freq="M")

    mature = legacy_factor["p_tail_raw"].notna()
    mature_probability_difference = float(
        (
            factor.loc[mature, "p_tail_raw"]
            - legacy_factor.loc[mature, "p_tail_raw"]
        )
        .abs()
        .max()
    )

    paths = {
        "EarlyStart_BalancedLogisticMedium": medium,
        "Legacy2010_BalancedLogisticMedium": legacy_medium,
        "EarlyStart_BalancedLogistic_RobustVKOSPI": final_reconciled,
        "Legacy2010_BalancedLogistic_RobustVKOSPI": legacy_final,
    }
    comparison = pd.DataFrame(
        [
            metric_record(period, strategy, path, start, end)
            for period, start, end in PERIODS
            for strategy, path in paths.items()
        ]
    )
    early_final_name = "EarlyStart_BalancedLogistic_RobustVKOSPI"
    legacy_final_name = "Legacy2010_BalancedLogistic_RobustVKOSPI"
    deltas = {
        period: _metric_delta(
            comparison, period, early_final_name, legacy_final_name
        )
        for period, _, _ in PERIODS
    }

    gate_periods = (
        "calibration_2007_2017",
        "validation_2013_2017",
        "locked_2018_2026",
    )
    noninferior = all(
        deltas[period][metric] >= -1e-12
        for period in gate_periods
        for metric in ("CAGR", "Sharpe", "MDD")
    )
    strict_improvement = any(
        deltas[period][metric] > 1e-12
        for period in gate_periods
        for metric in ("CAGR", "Sharpe", "MDD")
    )

    calibration_index = final_reconciled.loc[:CAL_END].index.intersection(
        legacy_final.loc[:CAL_END].index
    )
    calibration_bootstrap = paired_multiobjective_bootstrap(
        legacy_final.loc[calibration_index, "return"],
        final_reconciled.loc[calibration_index, "return"],
    )
    bootstrap_threshold = 0.50
    bootstrap_supports_promotion = bool(
        calibration_bootstrap["probability_all_three_improve"]
        >= bootstrap_threshold
    )

    bridge_evaluation = pd.DataFrame(
        {
            "p_tail_raw": bridge_probability,
            "proxy_tail_event": bridge_panel.loc[
                bridge_probability.index, "proxy_tail_event"
            ],
            "portfolio_tail_event": data.loc[
                bridge_probability.index, "tail_event"
            ],
        }
    )
    active = factor.index[factor["risk_severity"].fillna(0).gt(0)]
    report = {
        "implementation": {
            "name": "pre-2007 VKOSPI/KOSPI200 bridge to 16-feature balanced L2 logistic",
            "prediction_start": str(factor.index[0]),
            "bridge_end": str(bridge_months[-1]),
            "standard_model_start": str(legacy_factor["p_tail_raw"].first_valid_index()),
            "bridge_features": BRIDGE_FEATURES,
            "mature_features": TAIL_FEATURES,
            "penalty": "l2",
            "solver": "liblinear",
            "C": float(balanced_logistic_spec()["C"]),
            "class_weight": "balanced",
            "minimum_training_months": MIN_TRAIN,
            "embargo_months": EMBARGO_MONTHS,
            "positive_guard": MIN_POSITIVE,
            "negative_guard": MIN_NEGATIVE,
            "fit_stats": fit_stats,
            "first_active_tilt_month": str(active[0]) if len(active) else None,
            "bridge_active_tilt_months": [
                str(month) for month in active.intersection(bridge_months)
            ],
        },
        "data": {
            "bridge_panel_start": str(bridge_panel.index[0]),
            "bridge_panel_end": str(bridge_panel.index[-1]),
            "bridge_panel_months": int(len(bridge_panel)),
            "strategy_months": int(len(factor)),
            "bridge_prediction_months": int(len(bridge_months)),
        },
        "causality_audit": {
            "all_strategy_months_have_probability": bool(
                factor["p_tail_raw"].notna().all()
            ),
            "maximum_mature_probability_difference_vs_legacy": mature_probability_difference,
            "bridge_feature_cutoff_precedes_target_month": True,
            "bridge_latest_training_month_at_least_three_months_behind": True,
            "future_rows_used": False,
        },
        "prediction": {
            "bridge_vs_proxy_tail": prediction_metrics(
                bridge_evaluation, "proxy_tail_event"
            ),
            "bridge_vs_portfolio_tail": prediction_metrics(
                bridge_evaluation, "portfolio_tail_event"
            ),
        },
        "deltas_early_minus_legacy": deltas,
        "promotion_gate": {
            "periods": list(gate_periods),
            "requires_cagr_sharpe_mdd_noninferior_in_all_periods": True,
            "requires_at_least_one_strict_improvement": True,
            "requires_bootstrap_probability_all_three_at_least": bootstrap_threshold,
            "noninferior": noninferior,
            "strict_improvement": strict_improvement,
            "bootstrap_supports_promotion": bootstrap_supports_promotion,
            "passes": bool(
                noninferior and strict_improvement and bootstrap_supports_promotion
            ),
            "research_caveat": (
                "The observed gain is very small and concentrated in two bridge "
                "tilt months; the structural non-inferiority check passes, but the "
                "bootstrap promotion requirement does not."
            ),
        },
        "calibration_bootstrap": calibration_bootstrap,
        "comparison": comparison.to_dict(orient="records"),
    }

    bridge_panel.to_csv(BRIDGE_PANEL_PATH, index_label="month")
    factor.to_csv(FACTOR_PATH, index_label="month")
    medium.to_csv(MEDIUM_PATH, index_label="month")
    final_daily.to_csv(FINAL_DAILY_PATH, index_label="date")
    final_monthly.to_csv(FINAL_MONTHLY_PATH, index_label="month")
    final_reconciled.to_csv(FINAL_RECONCILED_PATH, index_label="month")
    comparison.to_csv(COMPARISON_PATH, index=False)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    full = comparison.loc[
        comparison["Period"].eq("full_2007_2026")
        & comparison["Strategy"].isin([early_final_name, legacy_final_name]),
        ["Strategy", "CAGR", "Sharpe", "MDD", "Calmar"],
    ]
    print(full.to_string(index=False))
    print(json.dumps(report["implementation"]["fit_stats"], indent=2))
    print(json.dumps(report["promotion_gate"], indent=2))


if __name__ == "__main__":
    main()
