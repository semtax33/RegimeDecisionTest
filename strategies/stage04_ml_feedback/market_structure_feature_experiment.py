from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from strategies.stage04_ml_feedback.final_blend_crash_meta_experiment import (
    DOMESTIC_FEATURES,
    STRESS_FEATURES,
    make_factor,
    metric_record,
    predictive_metrics,
    walk_forward_probability,
)
from strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment import (
    CAL_END,
    TEST_START,
    FactorBlendConfig,
    paired_block_bootstrap,
    run_factor_blend,
)
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
DB_PATH = ROOT / "raw_data" / "compass.db"
KOSPI200 = "1028"
SECTORS = [
    "5043",  # KRX Autos
    "5044",  # KRX Semiconductors
    "5045",  # KRX Healthcare
    "5046",  # KRX Banks
    "5048",  # KRX Energy & Chemicals
    "5049",  # KRX Steel
    "5052",  # KRX Construction
    "5054",  # KRX Securities
    "5055",  # KRX Machinery
    "5056",  # KRX Insurance
    "5057",  # KRX Transportation
    "5061",  # KRX Consumer Discretionary
    "5062",  # KRX Consumer Staples
    "5063",  # KRX Media & Entertainment
    "5064",  # KRX Information Technology
    "5065",  # KRX Utilities
]
CYCLICAL = ["5043", "5044", "5048", "5049", "5052", "5055", "5061", "5064"]
DEFENSIVE_SECTORS = ["5045", "5046", "5056", "5062", "5065"]


def load_index_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(DB_PATH) as connection:
        raw = pd.read_sql_query(
            "SELECT symbol, date, close, volume FROM etf_prices",
            connection,
            parse_dates=["date"],
        )
    raw["symbol"] = raw["symbol"].astype(str)
    close = raw.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = raw.pivot(index="date", columns="symbol", values="volume").sort_index()
    return close, volume


def correlation_features(window_returns: pd.DataFrame, prefix: str) -> dict[str, float]:
    usable = window_returns.dropna(axis=1, thresh=max(10, int(len(window_returns) * 0.75)))
    if usable.shape[1] < 6:
        return {
            f"sector_mean_corr_{prefix}": np.nan,
            f"sector_corr_dispersion_{prefix}": np.nan,
            f"sector_eigen_ratio_{prefix}": np.nan,
        }
    corr = usable.corr(min_periods=max(10, len(usable) // 2)).dropna(how="all").dropna(axis=1, how="all")
    corr = corr.loc[corr.index.intersection(corr.columns), corr.index.intersection(corr.columns)]
    values = corr.to_numpy(dtype=float)
    pairwise = values[np.triu_indices(len(corr), 1)]
    pairwise = pairwise[np.isfinite(pairwise)]
    cleaned = np.nan_to_num(values, nan=0.0)
    np.fill_diagonal(cleaned, 1.0)
    eigenvalues = np.linalg.eigvalsh(cleaned)
    return {
        f"sector_mean_corr_{prefix}": float(np.mean(pairwise)) if len(pairwise) else np.nan,
        f"sector_corr_dispersion_{prefix}": float(np.std(pairwise, ddof=1)) if len(pairwise) > 1 else np.nan,
        f"sector_eigen_ratio_{prefix}": float(np.max(eigenvalues) / max(np.sum(eigenvalues), 1e-8)),
    }


def build_structure_features(target_months: pd.PeriodIndex) -> pd.DataFrame:
    close, volume = load_index_history()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    kospi_return = returns[KOSPI200]
    kospi_volume = volume[KOSPI200].replace(0, np.nan)
    log_volume = np.log1p(kospi_volume)
    index_volume_proxy = close[KOSPI200] * kospi_volume
    log_index_volume_proxy = np.log1p(index_volume_proxy)
    rows: list[dict[str, float | pd.Period]] = []

    for month in target_months:
        cutoff = month.to_timestamp(how="start") - pd.Timedelta(days=2)
        history_return = returns.loc[:cutoff]
        history_close = close.loc[:cutoff]
        history_volume = kospi_volume.loc[:cutoff]
        history_log_volume = log_volume.loc[:cutoff]
        history_index_volume = index_volume_proxy.loc[:cutoff].dropna()
        history_log_index_volume = log_index_volume_proxy.loc[:cutoff].dropna()
        if len(history_return) < 126:
            continue
        row: dict[str, float | pd.Period] = {"month": month}

        for window in (21, 63):
            row.update(correlation_features(history_return[SECTORS].tail(window), str(window)))

        for horizon in (5, 20):
            sector_history = history_close[SECTORS].dropna(axis=1, thresh=max(2, len(history_close) // 4))
            sector_change = sector_history.iloc[-1] / sector_history.shift(horizon).iloc[-1] - 1
            sector_change = sector_change.replace([np.inf, -np.inf], np.nan).dropna()
            row[f"sector_positive_breadth_{horizon}"] = float((sector_change > 0).mean())
            row[f"sector_return_dispersion_{horizon}"] = float(sector_change.std(ddof=1))

        for window in (21, 63, 126):
            view = kospi_return.loc[:cutoff].tail(window).dropna()
            row[f"kospi_skew_{window}"] = float(view.skew())
            if window in (21, 63):
                row[f"kospi_excess_kurt_{window}"] = float(view.kurt())
        view20 = pd.DataFrame(
            {"return": kospi_return.loc[:cutoff], "volume": history_volume}
        ).dropna().tail(20)
        view63_log = history_log_volume.dropna().tail(63)
        view5_volume = history_volume.dropna().tail(5)
        view20_volume = history_volume.dropna().tail(20)
        view63_volume = history_volume.dropna().tail(63)
        row["volume_ratio_5_20"] = float(view5_volume.mean() / view20_volume.mean())
        row["volume_ratio_20_63"] = float(view20_volume.mean() / view63_volume.mean())
        row["volume_z_63"] = float(
            (view63_log.iloc[-1] - view63_log.mean()) / max(view63_log.std(ddof=1), 1e-8)
        )
        total_volume = float(view20["volume"].sum())
        row["down_volume_share_20"] = float(
            view20.loc[view20["return"] < 0, "volume"].sum() / max(total_volume, 1e-8)
        )
        row["signed_volume_imbalance_20"] = float(
            (np.sign(view20["return"]) * view20["volume"]).sum() / max(total_volume, 1e-8)
        )

        proxy5 = history_index_volume.tail(5)
        proxy20 = history_index_volume.tail(20)
        proxy63 = history_index_volume.tail(63)
        proxy_log63 = history_log_index_volume.tail(63)
        proxy_return20 = history_index_volume.iloc[-1] / history_index_volume.iloc[-21] - 1
        proxy_frame20 = pd.DataFrame(
            {
                "return": kospi_return.loc[:cutoff],
                "proxy": history_index_volume,
            }
        ).dropna().tail(20)
        proxy_total20 = float(proxy_frame20["proxy"].sum())
        row["index_volume_proxy_ratio_5_20"] = float(proxy5.mean() / proxy20.mean())
        row["index_volume_proxy_ratio_20_63"] = float(proxy20.mean() / proxy63.mean())
        row["index_volume_proxy_z_63"] = float(
            (proxy_log63.iloc[-1] - proxy_log63.mean()) / max(proxy_log63.std(ddof=1), 1e-8)
        )
        row["index_volume_proxy_momentum_20"] = float(proxy_return20)
        row["index_volume_proxy_down_share_20"] = float(
            proxy_frame20.loc[proxy_frame20["return"] < 0, "proxy"].sum()
            / max(proxy_total20, 1e-8)
        )

        view63 = history_return.tail(63)
        row["corr_kospi_semiconductor_63"] = float(view63[KOSPI200].corr(view63["5044"]))
        row["corr_kospi_bank_63"] = float(view63[KOSPI200].corr(view63["5046"]))
        cyclical_return = view63[CYCLICAL].mean(axis=1, skipna=True)
        defensive_return = view63[DEFENSIVE_SECTORS].mean(axis=1, skipna=True)
        row["corr_cyclical_defensive_63"] = float(cyclical_return.corr(defensive_return))
        rows.append(row)

    out = pd.DataFrame(rows).set_index("month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.replace([np.inf, -np.inf], np.nan)


def causal_zscore(series: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
    prior_mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    prior_std = series.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return (series - prior_mean) / prior_std.replace(0, np.nan)


def build_structure_composites(structure: pd.DataFrame) -> pd.DataFrame:
    z = structure.apply(causal_zscore)
    composite = pd.DataFrame(index=structure.index)
    composite["systemic_correlation_stress"] = (
        z["sector_mean_corr_21"]
        + z["sector_eigen_ratio_21"]
        - z["sector_corr_dispersion_21"]
    ) / 3
    composite["breadth_dispersion_stress"] = (
        -z["sector_positive_breadth_5"]
        - z["sector_positive_breadth_20"]
        + z["sector_return_dispersion_5"]
        + z["sector_return_dispersion_20"]
    ) / 4
    composite["volume_stress"] = (
        z["volume_z_63"]
        + z["volume_ratio_20_63"]
        + z["down_volume_share_20"]
        - z["signed_volume_imbalance_20"]
    ) / 4
    # This sign convention is fixed by economic interpretation, not fitted:
    # more negative skew and more positive excess kurtosis mean greater tail risk.
    composite["tail_shape_stress"] = (
        -z["kospi_skew_63"] - z["kospi_skew_126"] + z["kospi_excess_kurt_63"]
    ) / 3
    composite["sector_linkage_stress"] = (
        z["corr_kospi_semiconductor_63"]
        + z["corr_kospi_bank_63"]
        + z["corr_cyclical_defensive_63"]
    ) / 3
    composite["index_volume_proxy_stress"] = (
        z["index_volume_proxy_z_63"]
        + z["index_volume_proxy_ratio_20_63"]
        + z["index_volume_proxy_down_share_20"]
    ) / 3
    return composite.replace([np.inf, -np.inf], np.nan)


def univariate_audit(data: pd.DataFrame, feature_columns: list[str], target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, mask in (
        ("calibration", data.index <= CAL_END),
        ("locked", data.index >= TEST_START),
    ):
        for feature in feature_columns:
            view = data.loc[mask, [target, feature]].dropna()
            if view.empty or view[target].nunique() < 2 or view[feature].nunique() < 2:
                continue
            auc = float(roc_auc_score(view[target], view[feature]))
            rows.append(
                {
                    "Period": period,
                    "Target": target,
                    "Feature": feature,
                    "AUC": auc,
                    "DirectionalAUC": max(auc, 1 - auc),
                    "HighMeansRisk": auc >= 0.5,
                    "N": len(view),
                    "Events": int(view[target].sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    base_data = pd.read_csv(RESULTS / "hard_crash_features.csv", index_col=0)
    base_data.index = pd.PeriodIndex(base_data.index, freq="M")
    structure = build_structure_features(base_data.index)
    structure.to_csv(RESULTS / "market_structure_features.csv")
    composites = build_structure_composites(structure)
    composites.to_csv(RESULTS / "market_structure_composites.csv")
    data = base_data.join(structure, how="left").join(composites, how="left")

    macro, _ = load_macro_data()
    asset_returns, _ = load_monthly_asset_returns()
    signals = compute_regime_signals(macro, asset_returns)
    defensive = run_backtest(asset_returns, signals, StrategyConfig(), mode="proposed")
    neutral = pd.DataFrame({"p_up": 0.5}, index=data.index)
    baseline = run_factor_blend(
        asset_returns, signals, defensive, neutral, FactorBlendConfig(max_shift=0.0)
    )
    data = data.loc[data.index.intersection(baseline.index)].copy()
    baseline_return = baseline.loc[data.index, "return"]
    data["final_loss3"] = (baseline_return < -0.03).astype(int)
    data["final_loss4"] = (baseline_return < -0.04).astype(int)

    structure_columns = list(structure.columns)
    core_composite_columns = [
        "systemic_correlation_stress",
        "breadth_dispersion_stress",
        "volume_stress",
        "tail_shape_stress",
        "sector_linkage_stress",
    ]
    proxy_raw_columns = [
        "index_volume_proxy_ratio_5_20",
        "index_volume_proxy_ratio_20_63",
        "index_volume_proxy_z_63",
        "index_volume_proxy_momentum_20",
        "index_volume_proxy_down_share_20",
    ]
    proxy_composite_columns = ["index_volume_proxy_stress"]
    composite_columns = core_composite_columns + proxy_composite_columns
    specifications = {
        "loss3_base_stress": ("final_loss3", STRESS_FEATURES),
        "loss3_structure_domestic": ("final_loss3", DOMESTIC_FEATURES + structure_columns),
        "loss3_structure_stress": ("final_loss3", STRESS_FEATURES + structure_columns),
        "loss4_base_stress": ("final_loss4", STRESS_FEATURES),
        "loss4_structure_domestic": ("final_loss4", DOMESTIC_FEATURES + structure_columns),
        "loss4_structure_stress": ("final_loss4", STRESS_FEATURES + structure_columns),
        "loss3_composite_domestic": ("final_loss3", DOMESTIC_FEATURES + core_composite_columns),
        "loss3_composite_stress": ("final_loss3", STRESS_FEATURES + core_composite_columns),
        "loss4_composite_domestic": ("final_loss4", DOMESTIC_FEATURES + core_composite_columns),
        "loss4_composite_stress": ("final_loss4", STRESS_FEATURES + core_composite_columns),
        "loss3_index_volume_raw_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + proxy_raw_columns,
        ),
        "loss3_index_volume_composite_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + proxy_composite_columns,
        ),
        "loss3_composite_plus_index_volume_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + core_composite_columns + proxy_composite_columns,
        ),
        "loss3_corr_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + ["systemic_correlation_stress", "sector_linkage_stress"],
        ),
        "loss3_breadth_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + ["breadth_dispersion_stress"],
        ),
        "loss3_volume_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + ["volume_stress"],
        ),
        "loss3_tail_domestic": (
            "final_loss3",
            DOMESTIC_FEATURES + ["tail_shape_stress"],
        ),
    }
    factors: dict[str, pd.DataFrame] = {}
    for name, (target, selected) in specifications.items():
        probability = walk_forward_probability(data, target, selected)
        factor = make_factor(probability, data[target], name)
        factors[name] = factor
        factor.to_csv(RESULTS / f"market_structure_{name}_factor.csv")

    audit = pd.concat(
        [univariate_audit(data, structure_columns, target) for target in ("final_loss3", "final_loss4")],
        ignore_index=True,
    )
    audit.to_csv(RESULTS / "market_structure_univariate_audit.csv", index=False)

    shifts = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    baseline_calibration = performance_summary(baseline.loc[:CAL_END, "return"])
    calibration_rows: list[dict[str, object]] = []
    backtests: dict[tuple[str, float], pd.DataFrame] = {}
    winners: dict[str, tuple[float, pd.DataFrame]] = {}
    for name, factor in factors.items():
        for shift in shifts:
            backtest = run_factor_blend(
                asset_returns, signals, defensive, factor, FactorBlendConfig(max_shift=shift)
            )
            backtests[(name, shift)] = backtest
            metrics = performance_summary(backtest.loc[:CAL_END, "return"])
            calibration_rows.append(
                {
                    "Strategy": name,
                    "max_shift": shift,
                    **metrics.to_dict(),
                    "AvgTurnover": float(backtest.loc[:CAL_END, "turnover"].mean()),
                }
            )
        table = pd.DataFrame([row for row in calibration_rows if row["Strategy"] == name])
        eligible = table[
            (table["MDD"] >= -0.15)
            & (table["CAGR"] >= 0.95 * float(baseline_calibration["CAGR"]))
        ]
        pool = eligible if not eligible.empty else table
        winner = pool.sort_values(["Calmar", "Sharpe", "CAGR"], ascending=False).iloc[0]
        shift = float(winner["max_shift"])
        winners[name] = (shift, backtests[(name, shift)])

    comparison_rows: list[dict[str, object]] = []
    for period, start, end in (
        ("calibration_2007_2017", baseline.index.min(), CAL_END),
        ("locked_2018_2026", TEST_START, None),
        ("full_2007_2026", baseline.index.min(), None),
    ):
        strategies = {"FinalBlend": baseline, **{name: value[1] for name, value in winners.items()}}
        for name, backtest in strategies.items():
            view = backtest.loc[start:end] if end is not None else backtest.loc[start:]
            comparison_rows.append(metric_record(period, name, view))
    comparison = pd.DataFrame(comparison_rows)

    baseline_locked = baseline.loc[TEST_START:]
    baseline_metrics = performance_summary(baseline_locked["return"])
    validation: dict[str, object] = {}
    for name, (shift, backtest) in winners.items():
        locked = backtest.loc[TEST_START:]
        metrics = performance_summary(locked["return"])
        validation[name] = {
            "selected_shift": shift,
            "prediction_calibration": predictive_metrics(factors[name].loc[:CAL_END]),
            "prediction_locked": predictive_metrics(factors[name], TEST_START),
            "locked_deltas": {
                key: float(metrics[key] - baseline_metrics[key])
                for key in ("CAGR", "Sharpe", "MDD", "Calmar")
            },
            "bootstrap": paired_block_bootstrap(baseline_locked["return"], locked["return"]),
        }
        backtest.to_csv(RESULTS / f"market_structure_{name}_backtest.csv")

    report = {
        "method": {
            "new_features": structure_columns,
            "economic_composites": composite_columns,
            "market_cap_status": "not tested: point-in-time constituent market cap is absent locally and KRX API requires credentials",
            "model": "same embargoed balanced logistic meta-label as the prior experiment",
            "calibration_end": str(CAL_END),
            "locked_start": str(TEST_START),
        },
        "validation": validation,
    }
    pd.DataFrame(calibration_rows).to_csv(RESULTS / "market_structure_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "market_structure_comparison.csv", index=False)
    (RESULTS / "market_structure_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== STRUCTURE FEATURE COVERAGE ===")
    print(structure.notna().mean().sort_values().to_string(float_format=lambda value: f"{value:.3f}"))
    print("\n=== TOP UNIVARIATE FEATURES ===")
    print(
        audit.sort_values(["Period", "Target", "DirectionalAUC"], ascending=[True, True, False])
        .groupby(["Period", "Target"])
        .head(8)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\n=== WINNERS ===")
    print(pd.DataFrame({name: {"max_shift": value[0]} for name, value in winners.items()}).T)
    print("\n=== COMPARISON ===")
    print(
        comparison[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== VALIDATION ===")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
