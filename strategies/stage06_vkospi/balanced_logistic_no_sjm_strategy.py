from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    hard_regime_weights,
    load_macro_data,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    CAL_END,
    TEST_START,
    build_daily_vkospi_signals,
    load_daily_open_levels,
    paired_multiobjective_bootstrap,
    prepare_arrays,
    reconcile_to_monthly_reference,
    simulate,
)
from strategies.stage06_vkospi.vkospi_extended_diagnostics import (
    DOMESTIC_FEATURES,
    OAP_COMPOSITES,
    TAIL_FEATURES,
    _macro_probabilities,
)
from strategies.stage06_vkospi.vkospi_model_robustness import (
    build_macro_signals,
    fit_logistic_candidate,
    make_tail_factor,
    run_factor_vol_target,
)
from strategies.stage06_vkospi.vkospi_robust_dynamic_experiment import (
    RobustStressConfig,
    align_features_to_arrays,
    build_robust_daily_features,
    stress_from_features,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
VALIDATION_START = pd.Period("2013-01", freq="M")
LOCKED_LATE_START = pd.Period("2022-01", freq="M")

SIGNALS_PATH = RESULTS / "balanced_logistic_no_sjm_signals.csv"
FEATURES_PATH = RESULTS / "balanced_logistic_no_sjm_features.csv"
FACTOR_PATH = RESULTS / "balanced_logistic_no_sjm_factor.csv"
MEDIUM_PATH = RESULTS / "balanced_logistic_no_sjm_medium_backtest.csv"
FINAL_DAILY_PATH = RESULTS / "balanced_logistic_no_sjm_final_daily.csv"
FINAL_MONTHLY_PATH = RESULTS / "balanced_logistic_no_sjm_final_monthly.csv"
FINAL_RECONCILED_PATH = RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv"
COMPARISON_PATH = RESULTS / "balanced_logistic_no_sjm_comparison.csv"
REPORT_PATH = RESULTS / "balanced_logistic_no_sjm_validation.json"

PERIODS = (
    ("calibration_2007_2017", None, CAL_END),
    ("validation_2013_2017", VALIDATION_START, CAL_END),
    ("locked_2018_2026", TEST_START, None),
    ("locked_early_2018_2021", TEST_START, pd.Period("2021-12", freq="M")),
    ("locked_late_2022_2026", LOCKED_LATE_START, None),
    ("full_2007_2026", None, None),
)


def build_no_sjm_components(returns: pd.DataFrame) -> pd.DataFrame:
    """Build the causal macro component panel shared by no-SJM variants."""
    macro, _ = load_macro_data()
    rows: list[dict[str, float | pd.Period]] = []
    for target_month in returns.index:
        signal_month = target_month - 1
        history = macro.loc[: signal_month.to_timestamp("M")]
        if len(history) < 24:
            continue
        features = history.iloc[-1]
        rows.append(
            {
                "target_month": target_month,
                "signal_month": signal_month,
                "growth_level": float(
                    features["growth"][["GDP_level", "Export_level", "BSI_level"]].mean()
                ),
                "growth_d3": float(
                    features["growth"][
                        ["GDP_level_d3", "Export_level_d3", "BSI_level_d3"]
                    ].mean()
                ),
                "inflation_level": float(
                    features["inflation"][
                        ["CPI_level", "PPI_level", "ImportPrice_level"]
                    ].mean()
                ),
                "inflation_d3": float(
                    features["inflation"][
                        ["CPI_level_d3", "PPI_level_d3", "ImportPrice_level_d3"]
                    ].mean()
                ),
                # The probability function has one shared interface. These
                # placeholders are deliberately ignored when sjm_weight=0.
                "growth_sjm": 0.5,
                "inflation_sjm": 0.5,
            }
        )
    components = pd.DataFrame(rows).set_index("target_month")
    components.index = pd.PeriodIndex(components.index, freq="M")
    return components


def build_no_sjm_signals(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the transparent macro composite and set the SJM contribution to zero."""
    components = build_no_sjm_components(returns)
    probabilities = _macro_probabilities(
        components,
        d3_weight=0.20,
        sigmoid_scale=0.55,
        sjm_weight=0.0,
        current_weight=0.85,
    )
    signals = build_macro_signals(probabilities, components)
    assert (signals["signal_month"] < signals.index).all()
    assert np.isfinite(probabilities.to_numpy(dtype=float)).all()
    return signals, probabilities


def _daily_asset_returns() -> pd.DataFrame:
    market = pd.read_csv(ROOT / "cache" / "market_daily.csv", parse_dates=["date"])
    levels = market.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    levels["GLD"] = levels["GLD"] * levels["USDKRW"]
    levels["USO"] = levels["USO"] * levels["USDKRW"]
    bond = pd.read_csv(ROOT / "raw_data" / "krx_bond_index.csv", encoding="cp949")
    bond.index = pd.to_datetime(bond.iloc[:, 0])
    bond_level = pd.to_numeric(
        bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    daily_levels = pd.concat(
        [
            levels["KODEX200"],
            bond_level.rename("BOND"),
            levels["GLD"],
            levels["USO"],
        ],
        axis=1,
    ).sort_index()
    daily_levels.columns = ASSETS
    daily_levels = daily_levels.reindex(
        pd.date_range(daily_levels.index.min(), daily_levels.index.max(), freq="B")
    ).ffill(limit=5)
    return daily_levels.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def build_domestic_features(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the 12 deployed domestic inputs under the no-SJM regime path."""
    daily_returns = _daily_asset_returns()
    rows: list[dict[str, object]] = []
    for month in signals.index.intersection(returns.index):
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < 3:
            continue
        signal = signals.loc[month]
        base = hard_regime_weights(signal)
        proxy = pd.Series(history.to_numpy(dtype=float) @ base, index=history.index)
        cutoff = month.to_timestamp(how="start") - pd.Timedelta(days=2)
        daily_history = daily_returns.loc[:cutoff, ASSETS].dropna(how="all")
        daily_proxy = pd.Series(
            daily_history.fillna(0.0).to_numpy(dtype=float) @ base,
            index=daily_history.index,
        )

        row: dict[str, object] = {
            "month": month,
            "base_KODEX200": float(base[0]),
            "base_BOND": float(base[1]),
            "base_GLD": float(base[2]),
            "base_USO": float(base[3]),
            "p_inflation_high": float(signal["p_inflation_high"]),
            "proxy_mom1": float((1 + proxy.tail(1)).prod() - 1),
            "proxy_mom6": float((1 + proxy.tail(6)).prod() - 1),
            "proxy_vol6": float(proxy.tail(6).std(ddof=1) * math.sqrt(12)),
            "daily_mom21": float((1 + daily_proxy.tail(21)).prod() - 1),
            "daily_mom252": float((1 + daily_proxy.tail(252)).prod() - 1),
            "daily_vol21": float(daily_proxy.tail(21).std(ddof=1) * math.sqrt(252)),
            "daily_downvol21": float(
                np.sqrt(np.mean(np.minimum(daily_proxy.tail(21), 0.0) ** 2))
                * math.sqrt(252)
            ),
        }
        correlation = daily_history.tail(63)[["KODEX200", "GLD", "USO"]].corr()
        values = correlation.to_numpy()[np.triu_indices(3, 1)]
        row["daily_mean_corr63"] = float(np.nanmean(values))
        rows.append(row)

    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    missing = [column for column in DOMESTIC_FEATURES if column not in output]
    if missing:
        raise ValueError(f"Missing domestic inputs: {missing}")
    return output


def run_neutral_factor_blend(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    defensive: pd.DataFrame,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Reproduce the fixed 40% hard + 60% SLSQP, 1.2x baseline used by the label."""
    months = signals.index.intersection(returns.index).intersection(defensive.index)
    rows: list[dict[str, object]] = []
    pretrade = np.zeros(len(ASSETS))
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        defensive_weights = defensive.loc[
            month, [f"w_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        unlevered = 0.40 * hard + 0.60 * defensive_weights
        weights = 1.20 * unlevered
        debt_weight = -0.20
        delta = weights - pretrade
        turnover = np.abs(delta).sum() if first_trade else 0.5 * np.abs(delta).sum()
        trade_cost = np.abs(delta).sum() * 0.0015 * cost_multiplier
        fx_cost = (
            abs((weights[2] + weights[3]) - (pretrade[2] + pretrade[3]))
            * 0.0005
            * cost_multiplier
        )
        financing = debt_weight * ((1 + 0.04) ** (1 / 12) - 1)
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return + financing)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1 + asset_return) / (1 + gross_return)
        first_trade = False
        rows.append(
            {
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": float(turnover),
                "trade_cost": float(trade_cost),
                "fx_cost": float(fx_cost),
                **{f"w_{asset}": float(weights[index]) for index, asset in enumerate(ASSETS)},
            }
        )
    return pd.DataFrame(rows).set_index("month")


def forward_path_loss(returns: pd.Series, horizon: int = 2) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index, dtype=float)
    for position in range(len(returns)):
        future = returns.iloc[position : position + horizon]
        if len(future) == horizon:
            output.iloc[position] = float(((1 + future).cumprod() - 1).min())
    return output


def balanced_logistic_spec() -> dict[str, object]:
    return {
        "candidate": "l2_liblinear_c0.1_balanced",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 0.10,
        "class_weight": "balanced",
        "l1_ratio": np.nan,
    }


def prediction_metrics(
    factor: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, float | int]:
    view = factor
    if start is not None:
        view = view.loc[start:]
    if end is not None:
        view = view.loc[:end]
    view = view.dropna(subset=["p_tail_raw", "tail_event"])
    y = view["tail_event"].astype(int)
    probability = view["p_tail_raw"].clip(1e-6, 1 - 1e-6)
    return {
        "observations": int(len(view)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else np.nan,
        "average_precision": (
            float(average_precision_score(y, probability)) if y.nunique() > 1 else np.nan
        ),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def metric_record(
    period: str,
    strategy: str,
    path: pd.DataFrame,
    start: pd.Period | None,
    end: pd.Period | None,
) -> dict[str, object]:
    view = path
    if start is not None:
        view = view.loc[start:]
    if end is not None:
        view = view.loc[:end]
    metrics = performance_summary(view["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        "Months": int(len(view)),
        **{key: float(value) for key, value in metrics.to_dict().items()},
        "AvgTurnover": float(view["turnover"].mean()) if "turnover" in view else np.nan,
    }


def fixed_robust_overlay(
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = load_daily_open_levels()
    arrays = prepare_arrays(levels, reference, build_daily_vkospi_signals())
    features = align_features_to_arrays(build_robust_daily_features(), arrays)
    deployed_report = json.loads(
        (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(encoding="utf-8")
    )
    config = RobustStressConfig(**deployed_report["winner"])
    stress = stress_from_features(
        features,
        config.mode,
        config.level_threshold,
        config.shock_threshold,
    )
    _, neutral_monthly = simulate(arrays, None, keep_daily=False)
    daily, overlay_monthly = simulate(
        arrays,
        config.dynamic_config(),
        keep_daily=True,
        stress_override=stress,
    )
    reconciled = reconcile_to_monthly_reference(
        reference,
        neutral_monthly,
        overlay_monthly,
    )
    valid_signal = daily["signal_date"].notna()
    assert (
        daily.index[valid_signal].to_numpy()
        > pd.DatetimeIndex(daily.loc[valid_signal, "signal_date"]).to_numpy()
    ).all()
    return daily, overlay_monthly, reconciled


def audit_deployed_reproduction(returns: pd.DataFrame) -> dict[str, float | int]:
    """Prove that the variant pipeline changes SJM, not unrelated implementation details."""
    signals = pd.read_csv(RESULTS / "regime_signals.csv", index_col=0)
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    defensive = pd.read_csv(RESULTS / "proposed_backtest.csv", index_col=0)
    defensive.index = pd.PeriodIndex(defensive.index, freq="M")
    rebuilt_features = build_domestic_features(signals, returns)
    stored_features = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    stored_features.index = pd.PeriodIndex(stored_features.index, freq="M")
    feature_index = rebuilt_features.index.intersection(stored_features.index)
    feature_difference = (
        rebuilt_features.loc[feature_index, DOMESTIC_FEATURES]
        - stored_features.loc[feature_index, DOMESTIC_FEATURES]
    ).abs()

    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    composites.index = pd.PeriodIndex(composites.index, freq="M")
    neutral = run_neutral_factor_blend(returns, signals, defensive)
    data = rebuilt_features[DOMESTIC_FEATURES].join(
        composites[OAP_COMPOSITES], how="left"
    )
    data = data.loc[data.index.intersection(neutral.index)].copy()
    path_loss = forward_path_loss(neutral.loc[data.index, "return"], horizon=2)
    data["tail_event"] = (path_loss < -0.05).where(path_loss.notna()).astype(float)
    probability, _ = fit_logistic_candidate(data, balanced_logistic_spec())
    factor = make_tail_factor(probability, data["tail_event"])
    stored_factor = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0
    )
    stored_factor.index = pd.PeriodIndex(stored_factor.index, freq="M")
    factor_index = factor.index.intersection(stored_factor.index)
    label_view = pd.concat(
        [factor.loc[factor_index, "tail_event"], stored_factor.loc[factor_index, "tail_event"]],
        axis=1,
    ).dropna()
    probability_view = pd.concat(
        [
            factor.loc[factor_index, "p_tail_raw"],
            stored_factor.loc[factor_index, "p_tail_raw"],
        ],
        axis=1,
    ).dropna()

    medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    stored_medium = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_backtest.csv", index_col=0
    )
    stored_medium.index = pd.PeriodIndex(stored_medium.index, freq="M")
    medium_index = medium.index.intersection(stored_medium.index)
    _, _, final = fixed_robust_overlay(medium)
    stored_final = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col=0
    )
    stored_final.index = pd.PeriodIndex(stored_final.index, freq="M")
    final_index = final.index.intersection(stored_final.index)

    audit = {
        "feature_observations": int(len(feature_index)),
        "max_absolute_domestic_feature_difference": float(
            feature_difference.to_numpy(dtype=float).max()
        ),
        "label_observations": int(len(label_view)),
        "label_disagreements": int(
            (label_view.iloc[:, 0] != label_view.iloc[:, 1]).sum()
        ),
        "probability_observations": int(len(probability_view)),
        "max_absolute_probability_difference": float(
            (probability_view.iloc[:, 0] - probability_view.iloc[:, 1]).abs().max()
        ),
        "medium_return_observations": int(len(medium_index)),
        "max_absolute_medium_return_difference": float(
            (
                medium.loc[medium_index, "return"]
                - stored_medium.loc[medium_index, "return"]
            ).abs().max()
        ),
        "final_return_observations": int(len(final_index)),
        "max_absolute_final_return_difference": float(
            (
                final.loc[final_index, "return"]
                - stored_final.loc[final_index, "return"]
            ).abs().max()
        ),
    }
    assert audit["feature_observations"] > 0
    assert audit["max_absolute_domestic_feature_difference"] < 1e-12
    assert audit["label_disagreements"] == 0
    assert audit["max_absolute_probability_difference"] < 1e-12
    assert audit["max_absolute_medium_return_difference"] < 1e-12
    assert audit["max_absolute_final_return_difference"] < 1e-12
    return audit


def main() -> None:
    returns, _ = load_monthly_asset_returns(False)
    reproduction_audit = audit_deployed_reproduction(returns)
    signals, macro_probabilities = build_no_sjm_signals(returns)
    defensive = run_backtest(returns, signals, StrategyConfig(), mode="proposed")
    domestic = build_domestic_features(signals, returns)
    composites = pd.read_csv(RESULTS / "openassetpricing_composites.csv", index_col=0)
    composites.index = pd.PeriodIndex(composites.index, freq="M")

    neutral_baseline = run_neutral_factor_blend(returns, signals, defensive)
    data = domestic[DOMESTIC_FEATURES].join(composites[OAP_COMPOSITES], how="left")
    data = data.loc[data.index.intersection(neutral_baseline.index)].copy()
    path_loss = forward_path_loss(neutral_baseline.loc[data.index, "return"], horizon=2)
    data["tail_event"] = (path_loss < -0.05).where(path_loss.notna()).astype(float)
    probability, fit_stats = fit_logistic_candidate(data, balanced_logistic_spec())
    factor = make_tail_factor(probability, data["tail_event"])
    medium = run_factor_vol_target(
        returns,
        signals,
        defensive,
        factor,
        max_shift=0.20,
        target_vol=0.15,
    )
    final_daily, final_monthly, final_reconciled = fixed_robust_overlay(medium)

    deployed_medium = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_backtest.csv", index_col=0
    )
    deployed_medium.index = pd.PeriodIndex(deployed_medium.index, freq="M")
    deployed_final = pd.read_csv(
        RESULTS / "vkospi_robust_dynamic_reconciled_monthly.csv", index_col=0
    )
    deployed_final.index = pd.PeriodIndex(deployed_final.index, freq="M")
    deployed_proposed = pd.read_csv(RESULTS / "proposed_backtest.csv", index_col=0)
    deployed_proposed.index = pd.PeriodIndex(deployed_proposed.index, freq="M")

    paths = {
        "NoSJM_ProposedBase": defensive,
        "Deployed_SJM10_ProposedBase": deployed_proposed,
        "NoSJM_BalancedLogisticMedium": medium,
        "Deployed_SJM10_BalancedLogisticMedium": deployed_medium,
        "NoSJM_BalancedLogistic_RobustVKOSPI": final_reconciled,
        "Deployed_SJM10_BalancedLogistic_RobustVKOSPI": deployed_final,
    }
    rows = [
        metric_record(period, strategy, path, start, end)
        for period, start, end in PERIODS
        for strategy, path in paths.items()
    ]
    comparison = pd.DataFrame(rows)

    locked_current = deployed_final.loc[TEST_START:, "return"]
    locked_variant = final_reconciled.loc[TEST_START:, "return"]
    common = locked_current.index.intersection(locked_variant.index)
    bootstrap = paired_multiobjective_bootstrap(
        locked_current.loc[common],
        locked_variant.loc[common],
    )
    locked_rows = comparison.loc[
        comparison["Period"].eq("locked_2018_2026")
        & comparison["Strategy"].isin(
            [
                "NoSJM_BalancedLogistic_RobustVKOSPI",
                "Deployed_SJM10_BalancedLogistic_RobustVKOSPI",
            ]
        )
    ].set_index("Strategy")
    variant_locked = locked_rows.loc["NoSJM_BalancedLogistic_RobustVKOSPI"]
    deployed_locked = locked_rows.loc[
        "Deployed_SJM10_BalancedLogistic_RobustVKOSPI"
    ]
    locked_delta = {
        metric: float(variant_locked[metric] - deployed_locked[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }

    prelock_deltas: dict[str, dict[str, float]] = {}
    for period in ("calibration_2007_2017", "validation_2013_2017"):
        period_rows = comparison.loc[
            comparison["Period"].eq(period)
            & comparison["Strategy"].isin(
                [
                    "NoSJM_BalancedLogistic_RobustVKOSPI",
                    "Deployed_SJM10_BalancedLogistic_RobustVKOSPI",
                ]
            )
        ].set_index("Strategy")
        variant_row = period_rows.loc["NoSJM_BalancedLogistic_RobustVKOSPI"]
        deployed_row = period_rows.loc[
            "Deployed_SJM10_BalancedLogistic_RobustVKOSPI"
        ]
        prelock_deltas[period] = {
            metric: float(variant_row[metric] - deployed_row[metric])
            for metric in ("CAGR", "Sharpe", "MDD")
        }
    strict_prelock_pass = all(
        values["CAGR"] > 0 and values["Sharpe"] > 0 and values["MDD"] >= 0
        for values in prelock_deltas.values()
    )

    stored_factor = pd.read_csv(
        RESULTS / "openassetpricing_medium_horizon_factor.csv", index_col=0
    )
    stored_factor.index = pd.PeriodIndex(stored_factor.index, freq="M")
    label_view = factor[["tail_event", "p_tail_raw"]].join(
        stored_factor[["tail_event", "p_tail_raw"]],
        how="inner",
        lsuffix="_no_sjm",
        rsuffix="_deployed",
    )
    comparable_labels = label_view.dropna(subset=["tail_event_no_sjm", "tail_event_deployed"])
    comparable_probability = label_view.dropna(subset=["p_tail_raw_no_sjm", "p_tail_raw_deployed"])

    signals.to_csv(SIGNALS_PATH)
    domestic.to_csv(FEATURES_PATH)
    factor.to_csv(FACTOR_PATH)
    medium.to_csv(MEDIUM_PATH)
    final_daily.to_csv(FINAL_DAILY_PATH)
    final_monthly.to_csv(FINAL_MONTHLY_PATH)
    final_reconciled.to_csv(FINAL_RECONCILED_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)

    report = {
        "implementation": {
            "macro": {
                "d3_weight": 0.20,
                "sigmoid_scale": 0.55,
                "sjm_weight": 0.0,
                "current_weight": 0.85,
                "features": [
                    "GDP_level",
                    "GDP_level_d3",
                    "Export_level",
                    "Export_level_d3",
                    "BSI_level",
                    "BSI_level_d3",
                    "CPI_level",
                    "CPI_level_d3",
                    "PPI_level",
                    "PPI_level_d3",
                    "ImportPrice_level",
                    "ImportPrice_level_d3",
                ],
            },
            "logistic": {
                **balanced_logistic_spec(),
                "l1_ratio": None,
                "features": TAIL_FEATURES,
                "minimum_training_months": 36,
                "embargo_months": 2,
                "positive_guard": 4,
                "negative_guard": 12,
                "target": "two-month forward path minimum below -5%",
                "max_shift": 0.20,
                "target_vol": 0.15,
                "fit_stats": fit_stats,
            },
            "vkospi_overlay": json.loads(
                (RESULTS / "vkospi_robust_dynamic_validation.json").read_text(
                    encoding="utf-8"
                )
            )["winner"],
        },
        "data": {
            "signal_months": int(len(signals)),
            "feature_months": int(len(domestic)),
            "tail_events_calibration": int(data.loc[:CAL_END, "tail_event"].sum()),
            "tail_events_locked": int(data.loc[TEST_START:, "tail_event"].sum()),
            "label_comparison_observations": int(len(comparable_labels)),
            "label_agreement_with_deployed": float(
                (
                    comparable_labels["tail_event_no_sjm"]
                    == comparable_labels["tail_event_deployed"]
                ).mean()
            ),
            "probability_comparison_observations": int(len(comparable_probability)),
            "probability_correlation_with_deployed": float(
                comparable_probability[
                    ["p_tail_raw_no_sjm", "p_tail_raw_deployed"]
                ].corr().iloc[0, 1]
            ),
        },
        "prediction": {
            "calibration_2007_2017": prediction_metrics(factor, None, CAL_END),
            "validation_2013_2017": prediction_metrics(
                factor, VALIDATION_START, CAL_END
            ),
            "locked_2018_2026": prediction_metrics(factor, TEST_START, None),
        },
        "locked_final": {
            "variant": {
                key: float(variant_locked[key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar")
            },
            "deployed": {
                key: float(deployed_locked[key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar")
            },
            "delta_variant_minus_deployed": locked_delta,
            "bootstrap": bootstrap,
        },
        "prelock_gate": {
            "deltas_variant_minus_deployed": prelock_deltas,
            "requires_cagr_sharpe_mdd_improve_in_both_windows": True,
            "passes": bool(strict_prelock_pass),
        },
        "macro_probability_range": {
            "growth_min": float(macro_probabilities["growth"].min()),
            "growth_max": float(macro_probabilities["growth"].max()),
            "inflation_min": float(macro_probabilities["inflation"].min()),
            "inflation_max": float(macro_probabilities["inflation"].max()),
        },
        "deployed_reproduction_audit": reproduction_audit,
        "comparison": json.loads(comparison.to_json(orient="records")),
        "selection_note": (
            "This is a post-lock ablation requested by the user. All downstream "
            "features and labels are recomputed, but locked results are not used "
            "to tune C, class weighting, thresholds, or the VKOSPI overlay."
        ),
        "deployment_status": (
            "Implemented as an isolated candidate with separate result files. "
            "Existing deployed result files were not overwritten."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== BALANCED LOGISTIC + NO SJM ===")
    print("deployed reproduction", json.dumps(reproduction_audit, indent=2))
    print(json.dumps(report["locked_final"], ensure_ascii=False, indent=2))
    print("\n=== PERIOD COMPARISON ===")
    print(
        comparison[
            ["Period", "Strategy", "Months", "CAGR", "Sharpe", "MDD", "Calmar"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nsaved", REPORT_PATH)


if __name__ == "__main__":
    main()
