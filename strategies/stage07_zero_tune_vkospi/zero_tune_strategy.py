from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    get_path,
    load_monthly_asset_returns,
    performance_summary,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw_data"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# These are execution-cost assumptions shared with the reference backtest.
# They are not selected by the strategy and are reported separately.
DOMESTIC_TRADE_COST = 0.0015
FOREIGN_WEIGHT_CHANGE_COST = 0.0005

# The absence of values here is a machine-readable part of the design contract.
TUNABLE_HYPERPARAMETERS: tuple[()] = ()


def causal_expanding_percentile(series: pd.Series) -> pd.Series:
    """Return the empirical mid-rank using observations available through each row.

    There is no rolling-window length, minimum-history rule, decay, scale, or
    clipping constant. Ties receive their empirical mid-rank.
    """
    output = pd.Series(np.nan, index=series.index, dtype=float)
    history: list[float] = []
    for index, value in series.items():
        if not np.isfinite(value):
            continue
        history.append(float(value))
        reference = np.asarray(history, dtype=float)
        less = float(np.sum(reference < value))
        equal = float(np.sum(reference == value))
        output.loc[index] = (less + 0.5 * equal) / len(reference)
    return output


def load_macro_levels() -> pd.DataFrame:
    """Load the six macro levels without rolling z-scores or momentum windows."""
    gdp = pd.read_excel(
        get_path(RAW_DIR, "GDP 성장률.xlsx"), index_col=0, skiprows=6
    )
    gdp.columns = ["GDP_QoQ", "GDP_YoY"]
    gdp.index = (
        pd.PeriodIndex(gdp.index, freq="Q")
        .asfreq("M", how="end")
        .to_timestamp("M")
        + pd.offsets.MonthEnd(1)
    )
    gdp = gdp.resample("ME").ffill()

    trade = pd.read_excel(
        get_path(RAW_DIR, "수출입 총괄_20260816.xlsx"),
        index_col=0,
        skiprows=4,
    )
    trade = trade[["수출 금액", "수입금액"]].iloc[1:].copy()
    for column in trade:
        trade[column] = (
            trade[column].astype(str).str.replace(",", "", regex=False).astype(float)
        )
    trade.index = pd.to_datetime(trade.index, format="%Y.%m") + pd.offsets.MonthEnd(1)
    trade["Export_YoY"] = trade["수출 금액"].pct_change(12) * 100

    bsi = pd.read_csv(
        get_path(RAW_DIR, "기업경기조사(전망).csv"), encoding="cp949"
    )
    bsi = bsi[
        (bsi["업종코드별"] == "제 조 업")
        & (bsi["BSI코드별"] == "업황전망BSI 1)")
    ].iloc[:, 2:4]
    bsi["시점"] = (
        bsi["시점"]
        .str.replace("월", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    bsi["시점"] = pd.to_datetime(bsi["시점"], format="%Y.%m") + pd.offsets.MonthEnd(1)
    bsi = bsi.set_index("시점")
    bsi.columns = ["BSI"]

    cpi = pd.read_excel(
        get_path(RAW_DIR, "소비자물가 상승률.xlsx"), index_col=0, skiprows=6
    )
    cpi.columns = ["CPI_QoQ", "CPI_YoY"]
    cpi.index = pd.to_datetime(cpi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    ppi = pd.read_excel(
        get_path(RAW_DIR, "생산자물가 상승률.xlsx"), index_col=0, skiprows=6
    )
    ppi.columns = ["PPI_QoQ", "PPI_YoY"]
    ppi.index = pd.to_datetime(ppi.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    prices = pd.read_excel(
        get_path(RAW_DIR, "수출입물가 상승률.xlsx"), index_col=0, skiprows=6
    )
    prices.columns = ["ExportPrice_YoY", "ImportPrice_YoY"]
    prices.index = pd.to_datetime(prices.index, format="%Y-%m") + pd.offsets.MonthEnd(2)

    levels = pd.concat(
        [
            gdp["GDP_YoY"],
            trade["Export_YoY"],
            bsi["BSI"],
            cpi["CPI_YoY"],
            ppi["PPI_YoY"],
            prices["ImportPrice_YoY"],
        ],
        axis=1,
    ).sort_index()
    return levels


def build_macro_probabilities(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map six macro levels to expanding empirical probabilities.

    All three growth ranks and all three inflation ranks receive equal weight.
    No lookback, standardization scale, sigmoid, momentum, or smoothing weight
    is available to tune.
    """
    levels = load_macro_levels()
    ranks = levels.apply(causal_expanding_percentile)
    rows: list[dict[str, object]] = []
    for target_month in returns.index:
        signal_month = target_month - 1
        known = ranks.loc[: signal_month.to_timestamp("M")]
        if known.empty:
            continue
        current = known.iloc[-1]
        if current.isna().any():
            continue
        p_growth = float(
            current[["GDP_YoY", "Export_YoY", "BSI"]].mean()
        )
        p_inflation = float(
            current[["CPI_YoY", "PPI_YoY", "ImportPrice_YoY"]].mean()
        )
        rows.append(
            {
                "target_month": target_month,
                "signal_month": signal_month,
                "p_growth_high": p_growth,
                "p_inflation_high": p_inflation,
            }
        )
    probabilities = pd.DataFrame(rows).set_index("target_month")
    probabilities.index = pd.PeriodIndex(probabilities.index, freq="M")
    probabilities["p_Goldilocks"] = probabilities["p_growth_high"] * (
        1 - probabilities["p_inflation_high"]
    )
    probabilities["p_Overheating"] = (
        probabilities["p_growth_high"] * probabilities["p_inflation_high"]
    )
    probabilities["p_Slowdown"] = (1 - probabilities["p_growth_high"]) * (
        1 - probabilities["p_inflation_high"]
    )
    probabilities["p_Stagflation"] = (
        (1 - probabilities["p_growth_high"])
        * probabilities["p_inflation_high"]
    )
    return probabilities, ranks


def macro_asset_weights(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Assign each economic quadrant to its canonical single asset.

    Goldilocks -> Korean equity, Overheating -> oil, Slowdown -> bonds,
    Stagflation -> gold. The four quadrant probabilities already sum to one,
    so no anchor weights, optimizer, target volatility, or leverage is needed.
    """
    weights = pd.DataFrame(index=probabilities.index)
    weights["w_KODEX200"] = probabilities["p_Goldilocks"]
    weights["w_BOND"] = probabilities["p_Slowdown"]
    weights["w_GLD"] = probabilities["p_Stagflation"]
    weights["w_USO"] = probabilities["p_Overheating"]
    assert np.allclose(weights.sum(axis=1), 1.0)
    return weights


def load_vkospi_daily() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DIR / "VKOSPIData.csv", encoding="utf-8-sig")
    daily = raw.iloc[:, :7].copy()
    daily.columns = ["date", "close", "change", "return_pct", "open", "high", "low"]
    daily["date"] = pd.to_datetime(daily["date"], format="%Y/%m/%d", errors="coerce")
    daily["close"] = pd.to_numeric(
        daily["close"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    daily = daily.dropna(subset=["date", "close"]).set_index("date").sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    daily["stress"] = causal_expanding_percentile(daily["close"])
    return daily


def load_daily_open_levels() -> pd.DataFrame:
    market = pd.read_csv(CACHE_DIR / "market_daily.csv", parse_dates=["date"])
    with sqlite3.connect(get_path(RAW_DIR, "compass.db")) as connection:
        proxy = pd.read_sql(
            "select date, open from etf_prices where symbol = ? order by date",
            connection,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy["open"] = pd.to_numeric(proxy["open"], errors="coerce")
    actual = market.loc[
        (market["symbol"] == "KODEX200")
        & market["open"].notna()
        & (market["date"] > pd.Timestamp("2009-03-31"))
    ].copy()
    first_actual = actual["date"].min()
    actual_anchor = float(actual.loc[actual["date"] == first_actual, "open"].iloc[0])
    nearest = proxy.iloc[(proxy["date"] - first_actual).abs().argsort()[:1]]
    proxy["open"] *= actual_anchor / float(nearest["open"].iloc[0])
    proxy = proxy.loc[proxy["date"] < first_actual]
    kodex = pd.concat(
        [proxy[["date", "open"]], actual[["date", "open"]]], ignore_index=True
    )
    kodex = (
        kodex.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["open"]
        .rename("KODEX200")
    )

    bond = pd.read_csv(get_path(RAW_DIR, "krx_bond_index.csv"), encoding="cp949")
    bond.index = pd.to_datetime(bond.iloc[:, 0])
    bond_level = pd.to_numeric(
        bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).rename("BOND")

    open_prices = market.pivot_table(
        index="date", columns="symbol", values="open", aggfunc="last"
    ).sort_index()
    close_prices = market.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    fx = close_prices["USDKRW"].reindex(
        pd.date_range(close_prices.index.min(), close_prices.index.max(), freq="D")
    ).ffill()
    gld = (open_prices["GLD"] * fx.reindex(open_prices.index).to_numpy()).rename("GLD")
    uso = (open_prices["USO"] * fx.reindex(open_prices.index).to_numpy()).rename("USO")
    levels = pd.concat([kodex, bond_level, gld, uso], axis=1).sort_index()
    calendar = pd.date_range(levels.index.min(), levels.index.max(), freq="B")
    return levels.reindex(calendar).ffill(limit=5)[ASSETS]


def apply_parameter_free_overlay(
    base_weights: np.ndarray,
    vkospi_percentile: float,
) -> np.ndarray:
    """Move the percentile fraction of equity/oil equally to bonds/gold."""
    desired = np.asarray(base_weights, dtype=float).copy()
    stress = float(np.clip(vkospi_percentile, 0.0, 1.0))
    removed = desired[[0, 3]] * stress
    desired[[0, 3]] -= removed
    desired[[1, 2]] += removed.sum() / len(desired[[1, 2]])
    assert math.isclose(float(desired.sum()), 1.0, abs_tol=1e-12)
    return desired


def simulate_daily(
    levels: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    vkospi: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an always-rebalanced, unlevered, strictly lagged daily simulation."""
    forward_returns = levels.shift(-1).div(levels).sub(1.0)
    dates = levels.index[
        (levels.index.to_period("M") >= monthly_weights.index.min())
        & (levels.index.to_period("M") <= monthly_weights.index.max())
    ]
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    rows: list[dict[str, object]] = []
    pretrade = np.zeros(len(ASSETS))
    first_trade = True
    nav = 1.0
    peak = 1.0

    for date in dates:
        month = date.to_period("M")
        if month not in monthly_weights.index:
            continue
        asset_return = forward_returns.loc[date, ASSETS].to_numpy(dtype=float)
        if not np.isfinite(asset_return).all():
            continue
        known = vkospi.loc[: date - pd.Timedelta(days=1), "stress"].dropna()
        if known.empty:
            continue
        signal_date = known.index[-1]
        stress = float(known.iloc[-1])
        base = monthly_weights.loc[month, weight_columns].to_numpy(dtype=float)
        desired = apply_parameter_free_overlay(base, stress)

        delta = desired - pretrade
        turnover = (
            float(np.abs(delta).sum())
            if first_trade
            else 0.5 * float(np.abs(delta).sum())
        )
        trade_cost = float(np.abs(delta).sum()) * DOMESTIC_TRADE_COST
        fx_cost = (
            abs(
                (desired[2] + desired[3])
                - (pretrade[2] + pretrade[3])
            )
            * FOREIGN_WEIGHT_CHANGE_COST
        )
        gross_return = float(desired @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = desired * (1 + asset_return) / (1 + gross_return)
        rows.append(
            {
                "date": date,
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "vkospi_percentile": stress,
                "signal_date": signal_date,
                **{
                    f"base_{asset}": base[index]
                    for index, asset in enumerate(ASSETS)
                },
                **{
                    f"w_{asset}": desired[index]
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda values: float(np.prod(1 + values))),
        gross_factor=("gross_return", lambda values: float(np.prod(1 + values))),
        turnover=("turnover", "sum"),
        trade_cost=("trade_cost", "sum"),
        fx_cost=("fx_cost", "sum"),
        avg_vkospi_percentile=("vkospi_percentile", "mean"),
        max_vkospi_percentile=("vkospi_percentile", "max"),
    )
    monthly["return"] = monthly["return_factor"] - 1
    monthly["gross_return"] = monthly["gross_factor"] - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    return daily, monthly


def metric_record(
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
        "Strategy": strategy,
        "Start": str(view.index.min()),
        "End": str(view.index.max()),
        **{name: float(value) for name, value in metrics.items()},
        "AvgTurnover": (
            float(view["turnover"].mean()) if "turnover" in view else np.nan
        ),
    }


def compare_with_reference(zero_tune: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(
        ROOT / "results" / "balanced_logistic_no_sjm_final_reconciled.csv",
        index_col=0,
    )
    reference.index = pd.PeriodIndex(reference.index, freq="M")
    common_start = max(zero_tune.index.min(), reference.index.min())
    common_end = min(zero_tune.index.max(), reference.index.max())
    rows: list[dict[str, object]] = []
    for period, start in [
        ("full_2007_2026", common_start),
        ("locked_2018_2026", pd.Period("2018-01", freq="M")),
    ]:
        for name, path in [
            ("ZeroTune_VKOSPI", zero_tune),
            ("Current_Robust_VKOSPI", reference),
        ]:
            row = metric_record(name, path, start, common_end)
            row["Period"] = period
            rows.append(row)
    return pd.DataFrame(rows)


def run_zero_tune_research(save: bool = True) -> dict[str, object]:
    monthly_returns, _ = load_monthly_asset_returns(False)
    probabilities, macro_ranks = build_macro_probabilities(monthly_returns)
    weights = macro_asset_weights(probabilities)
    levels = load_daily_open_levels()
    vkospi = load_vkospi_daily()
    daily, monthly = simulate_daily(levels, weights, vkospi)
    comparison = compare_with_reference(monthly)

    valid_signals = daily["signal_date"].notna()
    assert (
        daily.index[valid_signals].to_numpy()
        > pd.DatetimeIndex(daily.loc[valid_signals, "signal_date"]).to_numpy()
    ).all()
    assert (probabilities["signal_month"] < probabilities.index).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert TUNABLE_HYPERPARAMETERS == ()

    period_results: dict[str, dict[str, object]] = {}
    for period, period_rows in comparison.groupby("Period", sort=False):
        indexed = period_rows.set_index("Strategy")
        zero = indexed.loc["ZeroTune_VKOSPI"]
        reference = indexed.loc["Current_Robust_VKOSPI"]
        period_results[str(period)] = {
            "zero_tune": {
                name: float(zero[name])
                for name in ["Months", "CAGR", "Sharpe", "MDD", "Calmar"]
            },
            "current_robust": {
                name: float(reference[name])
                for name in ["Months", "CAGR", "Sharpe", "MDD", "Calmar"]
            },
            "zero_tune_minus_current": {
                name: float(zero[name] - reference[name])
                for name in ["CAGR", "Sharpe", "MDD", "Calmar"]
            },
        }
    report: dict[str, object] = {
        "strategy": "ZeroTune_VKOSPI",
        "definition": (
            "Expanding empirical macro ranks, quadrant-probability asset weights, "
            "and direct expanding VKOSPI percentile risk transfer."
        ),
        "selection_procedure": (
            "One predeclared deterministic specification; no candidate grid "
            "or parameter search was run."
        ),
        "tunable_hyperparameters": list(TUNABLE_HYPERPARAMETERS),
        "execution_assumptions": {
            "domestic_trade_cost": DOMESTIC_TRADE_COST,
            "foreign_weight_change_cost": FOREIGN_WEIGHT_CHANGE_COST,
            "execution": "next open-to-open return using VKOSPI through prior day",
            "leverage": "none; asset weights sum to one",
            "rebalance_rule": "always move to the deterministic desired weight",
        },
        "period_results": period_results,
        "causality_checks": {
            "macro_signal_precedes_target": True,
            "vkospi_signal_precedes_action": True,
        },
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probabilities.to_csv(OUTPUT_DIR / "macro_probabilities.csv")
        macro_ranks.to_csv(OUTPUT_DIR / "macro_expanding_ranks.csv")
        weights.to_csv(OUTPUT_DIR / "macro_asset_weights.csv")
        daily.to_csv(OUTPUT_DIR / "zero_tune_daily.csv")
        monthly.to_csv(OUTPUT_DIR / "zero_tune_monthly.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        (OUTPUT_DIR / "zero_tune_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "probabilities": probabilities,
        "macro_ranks": macro_ranks,
        "weights": weights,
        "daily": daily,
        "monthly": monthly,
        "comparison": comparison,
        "report": report,
    }


def main() -> None:
    result = run_zero_tune_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    print("saved", OUTPUT_DIR)


if __name__ == "__main__":
    main()
