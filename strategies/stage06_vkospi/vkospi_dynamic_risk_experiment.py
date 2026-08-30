from __future__ import annotations

import itertools
import json
import math
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw_data"
RESULTS = ROOT / "results"
REFERENCE_PATH = RESULTS / "vkospi_selected_backtest.csv"
VKOSPI_PATH = ROOT / "raw_data" / "VKOSPIData.csv"
ASSETS = ["KODEX200", "BOND", "GLD", "USO"]
CAL_END = pd.Period("2017-12", freq="M")
TEST_START = pd.Period("2018-01", freq="M")
CAL_VALIDATION_START = pd.Period("2013-01", freq="M")


def get_path(directory: Path, filename: str) -> Path:
    """Resolve Korean filenames robustly across NFC/NFD filesystems."""
    target = unicodedata.normalize("NFC", filename)
    for path in directory.iterdir():
        if unicodedata.normalize("NFC", path.name) == target:
            return path
    raise FileNotFoundError(filename)


def performance_summary(returns: pd.Series) -> pd.Series:
    """Return the monthly performance metrics used by the reference strategy."""
    r = pd.Series(returns).dropna()
    wealth = (1 + r).cumprod()
    years = len(r) / 12
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    std = r.std(ddof=1)
    volatility = std * math.sqrt(12)
    sharpe = r.mean() / std * math.sqrt(12) if std > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1
    mdd = drawdown.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(12)
    sortino = r.mean() * 12 / downside if downside > 0 else np.nan
    return pd.Series(
        {
            "Months": len(r),
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MDD": mdd,
            "Calmar": calmar,
            "FinalMultiple": wealth.iloc[-1],
            "PositiveMonths": (r > 0).mean(),
        }
    )


@dataclass(frozen=True)
class DynamicRiskConfig:
    mode: str
    level_threshold: float
    momentum_window: int
    spike_threshold: float
    max_risk_transfer: float
    bond_share: float = 0.50
    rebalance_band: float = 0.05
    financing_rate: float = 0.04

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_lt{self.level_threshold:.2f}_mw{self.momentum_window}"
            f"_st{self.spike_threshold:.2f}_rt{self.max_risk_transfer:.2f}"
            f"_bs{self.bond_share:.2f}_rb{self.rebalance_band:.2f}"
        )


def load_vkospi_daily(path: Path = VKOSPI_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if raw.shape[1] < 7:
        raise ValueError(f"VKOSPI file must contain seven columns: {path}")
    daily = raw.iloc[:, :7].copy()
    daily.columns = ["date", "close", "change", "return_pct", "open", "high", "low"]
    daily["date"] = pd.to_datetime(
        daily["date"], format="%Y/%m/%d", errors="coerce"
    )
    for column in daily.columns[1:]:
        daily[column] = pd.to_numeric(
            daily[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    daily = daily.dropna(subset=["date", "close"]).set_index("date").sort_index()
    return daily[~daily.index.duplicated(keep="last")]


def load_daily_open_levels() -> pd.DataFrame:
    market = pd.read_csv(ROOT / "cache" / "market_daily.csv", parse_dates=["date"])
    with sqlite3.connect(get_path(RAW_DIR, "compass.db")) as connection:
        proxy = pd.read_sql(
            "select date, open, close from etf_prices where symbol = ? order by date",
            connection,
            params=("1028",),
        )
    proxy["date"] = pd.to_datetime(proxy["date"])
    proxy[["open", "close"]] = proxy[["open", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
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
    )

    bond = pd.read_csv(get_path(RAW_DIR, "krx_bond_index.csv"), encoding="cp949")
    bond.index = pd.to_datetime(bond.iloc[:, 0])
    bond_level = pd.to_numeric(
        bond.iloc[:, 1].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).rename("BOND")

    pivot_open = market.pivot_table(
        index="date", columns="symbol", values="open", aggfunc="last"
    ).sort_index()
    pivot_close = market.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    fx = pivot_close["USDKRW"].reindex(
        pd.date_range(pivot_close.index.min(), pivot_close.index.max(), freq="D")
    ).ffill()
    gld = (pivot_open["GLD"] * fx.reindex(pivot_open.index).to_numpy()).rename("GLD")
    uso = (pivot_open["USO"] * fx.reindex(pivot_open.index).to_numpy()).rename("USO")
    levels = pd.concat(
        [kodex.rename("KODEX200"), bond_level, gld, uso], axis=1, sort=False
    ).sort_index()
    calendar = pd.date_range(levels.index.min(), levels.index.max(), freq="B")
    return levels.reindex(calendar).ffill(limit=5)[ASSETS]


def causal_percentile(series: pd.Series, window: int = 252, minimum: int = 126) -> pd.Series:
    def last_rank(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < minimum:
            return np.nan
        return float((np.sum(finite[:-1] <= finite[-1]) + 1) / len(finite))

    return series.rolling(window + 1, min_periods=minimum).apply(last_rank, raw=True)


def build_daily_vkospi_signals() -> pd.DataFrame:
    daily = load_vkospi_daily()
    close = daily["close"]
    output = pd.DataFrame(index=daily.index)
    output["vkospi_close"] = close
    output["level_percentile"] = causal_percentile(close)
    for window in (5, 10, 21):
        output[f"momentum_{window}"] = close.pct_change(window, fill_method=None)
    return output.replace([np.inf, -np.inf], np.nan)


def load_reference_weights() -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_PATH, index_col=0)
    reference.index = pd.PeriodIndex(reference.index, freq="M")
    required = [f"w_{asset}" for asset in ASSETS]
    missing = [column for column in required if column not in reference]
    if missing:
        raise ValueError(f"Reference strategy is missing weights: {missing}")
    return reference


def prepare_arrays(
    levels: pd.DataFrame,
    reference: pd.DataFrame,
    vkospi_signals: pd.DataFrame,
) -> dict[str, object]:
    forward = levels.shift(-1).div(levels).sub(1.0)
    dates = levels.index[
        (levels.index.to_period("M") >= reference.index.min())
        & (levels.index.to_period("M") <= reference.index.max())
    ]
    valid_dates: list[pd.Timestamp] = []
    returns: list[np.ndarray] = []
    base_weights: list[np.ndarray] = []
    months: list[pd.Period] = []
    level_percentile: list[float] = []
    momentum = {window: [] for window in (5, 10, 21)}
    signal_dates: list[pd.Timestamp | pd.NaT] = []
    weight_columns = [f"w_{asset}" for asset in ASSETS]

    for date in dates:
        month = date.to_period("M")
        if month not in reference.index:
            continue
        asset_return = forward.loc[date, ASSETS].to_numpy(dtype=float)
        if not np.isfinite(asset_return).all():
            continue
        cutoff = date - pd.Timedelta(days=1)
        known = vkospi_signals.loc[:cutoff]
        signal = known.iloc[-1] if not known.empty else pd.Series(dtype=float)
        valid_dates.append(date)
        returns.append(asset_return)
        base_weights.append(reference.loc[month, weight_columns].to_numpy(dtype=float))
        months.append(month)
        signal_dates.append(known.index[-1] if not known.empty else pd.NaT)
        level_percentile.append(float(signal.get("level_percentile", np.nan)))
        for window in momentum:
            momentum[window].append(float(signal.get(f"momentum_{window}", np.nan)))

    return {
        "dates": pd.DatetimeIndex(valid_dates),
        "returns": np.asarray(returns),
        "base_weights": np.asarray(base_weights),
        "months": pd.PeriodIndex(months, freq="M"),
        "signal_dates": pd.DatetimeIndex(signal_dates),
        "level_percentile": np.asarray(level_percentile),
        "momentum": {window: np.asarray(values) for window, values in momentum.items()},
    }


def _stress_series(arrays: dict[str, object], cfg: DynamicRiskConfig) -> np.ndarray:
    percentile = np.asarray(arrays["level_percentile"], dtype=float)
    momentum = np.asarray(arrays["momentum"][cfg.momentum_window], dtype=float)
    level = np.clip(
        (percentile - cfg.level_threshold) / max(1 - cfg.level_threshold, 1e-6), 0, 1
    )
    spike = np.clip(
        (momentum - cfg.spike_threshold) / 0.25,
        0,
        1,
    )
    level = np.nan_to_num(level, nan=0.0)
    spike = np.nan_to_num(spike, nan=0.0)
    if cfg.mode == "level":
        return level
    if cfg.mode == "momentum":
        return spike
    if cfg.mode == "max":
        return np.maximum(level, spike)
    if cfg.mode == "mean":
        return 0.5 * (level + spike)
    raise ValueError(f"Unknown stress mode: {cfg.mode}")


def simulate(
    arrays: dict[str, object],
    cfg: DynamicRiskConfig | None,
    start: pd.Period | None = None,
    end: pd.Period | None = None,
    cost_multiplier: float = 1.0,
    keep_daily: bool = True,
    stress_override: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(arrays["dates"])
    months = pd.PeriodIndex(arrays["months"], freq="M")
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= months >= start
    if end is not None:
        mask &= months <= end
    positions = np.flatnonzero(mask)
    if stress_override is not None:
        stress = np.asarray(stress_override, dtype=float)
        if stress.shape != (len(dates),):
            raise ValueError(
                f"stress_override must have shape {(len(dates),)}, got {stress.shape}"
            )
        stress = np.nan_to_num(stress, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
    else:
        stress = np.zeros(len(dates)) if cfg is None else _stress_series(arrays, cfg)
    returns = np.asarray(arrays["returns"], dtype=float)
    base_weights = np.asarray(arrays["base_weights"], dtype=float)
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])

    pretrade = np.zeros(len(ASSETS))
    previous_month: pd.Period | None = None
    nav = 1.0
    peak = 1.0
    first_trade = True
    rf_daily = (1 + (cfg.financing_rate if cfg else 0.04)) ** (1 / 252) - 1
    rows: list[dict[str, object]] = []

    for position in positions:
        date = dates[position]
        month = months[position]
        base = base_weights[position].copy()
        severity = float(stress[position])
        transfer_fraction = 0.0 if cfg is None else cfg.max_risk_transfer * severity
        desired = base.copy()
        removed_equity = desired[0] * transfer_fraction
        removed_oil = desired[3] * transfer_fraction
        desired[0] -= removed_equity
        desired[3] -= removed_oil
        removed = removed_equity + removed_oil
        bond_share = 0.50 if cfg is None else cfg.bond_share
        desired[1] += removed * bond_share
        desired[2] += removed * (1 - bond_share)

        month_boundary = previous_month is None or month != previous_month
        band = math.inf if cfg is None else cfg.rebalance_band
        desired_turnover = 0.5 * float(np.abs(desired - pretrade).sum())
        rebalance = month_boundary or desired_turnover >= band
        weights = desired if rebalance else pretrade.copy()
        delta = weights - pretrade
        turnover = float(np.abs(delta).sum()) if first_trade else 0.5 * float(
            np.abs(delta).sum()
        )
        trade_cost = float(np.abs(delta).sum()) * 0.0015 * cost_multiplier
        fx_cost = (
            abs((weights[2] + weights[3]) - (pretrade[2] + pretrade[3]))
            * 0.0005
            * cost_multiplier
        )
        debt_weight = 1.0 - float(weights.sum())
        asset_return = returns[position]
        gross_return = float(weights @ asset_return + debt_weight * rf_daily)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1 + asset_return) / (1 + gross_return)
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
                "stress": severity,
                "transfer_fraction": transfer_fraction,
                "signal_date": signal_dates[position],
                **{f"w_{asset}": weights[i] for i, asset in enumerate(ASSETS)},
            }
        )
        previous_month = month
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda x: float(np.prod(1 + x))),
        gross_factor=("gross_return", lambda x: float(np.prod(1 + x))),
        turnover=("turnover", "sum"),
        trade_cost=("trade_cost", "sum"),
        fx_cost=("fx_cost", "sum"),
        avg_stress=("stress", "mean"),
        max_stress=("stress", "max"),
        avg_transfer=("transfer_fraction", "mean"),
    )
    monthly["return"] = monthly["return_factor"] - 1
    monthly["gross_return"] = monthly["gross_factor"] - 1
    wealth = (1 + monthly["return"]).cumprod()
    monthly["nav"] = wealth
    monthly["drawdown"] = wealth / wealth.cummax() - 1
    return (daily if keep_daily else pd.DataFrame()), monthly


def metric_row(period: str, strategy: str, monthly: pd.DataFrame) -> dict[str, object]:
    metrics = performance_summary(monthly["return"])
    return {
        "Period": period,
        "Strategy": strategy,
        **metrics.to_dict(),
        "AvgTurnover": float(monthly["turnover"].mean()),
        "TotalCost": float(monthly[["trade_cost", "fx_cost"]].sum().sum()),
    }


def reconcile_to_monthly_reference(
    monthly_reference: pd.DataFrame,
    daily_baseline: pd.DataFrame,
    daily_overlay: pd.DataFrame,
) -> pd.DataFrame:
    """Apply only the daily overlay's relative return to the validated monthly path.

    The daily reconstruction differs slightly from the original monthly engine
    because financing and costs compound at a different frequency. Taking the
    overlay/base relative factor isolates the incremental VKOSPI allocation
    effect without pretending that this frequency-conversion gap is alpha.
    """
    common = (
        monthly_reference.index.intersection(daily_baseline.index)
        .intersection(daily_overlay.index)
        .sort_values()
    )
    relative_factor = (1 + daily_overlay.loc[common, "return"]).div(
        1 + daily_baseline.loc[common, "return"]
    )
    output = pd.DataFrame(index=common)
    output["reference_return"] = monthly_reference.loc[common, "return"]
    output["overlay_relative_return"] = relative_factor - 1
    output["return"] = (1 + output["reference_return"]) * relative_factor - 1
    output["gross_return"] = output["return"]
    output["turnover"] = daily_overlay.loc[common, "turnover"]
    output["trade_cost"] = daily_overlay.loc[common, "trade_cost"]
    output["fx_cost"] = daily_overlay.loc[common, "fx_cost"]
    output["avg_stress"] = daily_overlay.loc[common, "avg_stress"]
    output["max_stress"] = daily_overlay.loc[common, "max_stress"]
    output["avg_transfer"] = daily_overlay.loc[common, "avg_transfer"]
    wealth = (1 + output["return"]).cumprod()
    output["nav"] = wealth
    output["drawdown"] = wealth / wealth.cummax() - 1
    return output


def paired_multiobjective_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    block_length: int = 6,
    simulations: int = 5_000,
) -> dict[str, float]:
    common = baseline.index.intersection(candidate.index)
    matrix = np.column_stack([baseline.loc[common], candidate.loc[common]]).astype(float)
    count = len(matrix)
    rng = np.random.default_rng(20260827 + block_length)
    deltas = np.empty((simulations, 3), dtype=float)
    for simulation in range(simulations):
        indices: list[int] = []
        while len(indices) < count:
            start = int(rng.integers(0, count - block_length + 1))
            indices.extend(range(start, start + block_length))
        draw = matrix[np.asarray(indices[:count])]
        metrics = []
        for column in range(2):
            values = draw[:, column]
            years = count / 12
            cagr = float(np.prod(1 + values) ** (1 / years) - 1)
            sharpe = float(values.mean() / values.std(ddof=1) * math.sqrt(12))
            wealth = np.cumprod(1 + values)
            mdd = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
            metrics.append((cagr, sharpe, mdd))
        deltas[simulation] = np.asarray(metrics[1]) - np.asarray(metrics[0])
    names = ("cagr", "sharpe", "mdd")
    output: dict[str, float] = {
        f"probability_{name}_improves": float(np.mean(deltas[:, index] > 0))
        for index, name in enumerate(names)
    }
    output["probability_all_three_improve"] = float(np.mean((deltas > 0).all(axis=1)))
    for index, name in enumerate(names):
        output[f"{name}_delta_p05"] = float(np.quantile(deltas[:, index], 0.05))
        output[f"{name}_delta_median"] = float(np.median(deltas[:, index]))
        output[f"{name}_delta_p95"] = float(np.quantile(deltas[:, index], 0.95))
    return output


def candidate_configs() -> list[DynamicRiskConfig]:
    configs: dict[str, DynamicRiskConfig] = {}
    for mode in ("level", "momentum", "max", "mean"):
        level_values = (0.70, 0.80, 0.90) if mode != "momentum" else (0.80,)
        momentum_windows = (5, 10, 21) if mode != "level" else (10,)
        spike_values = (0.05, 0.10, 0.15) if mode != "level" else (0.10,)
        for level, window, spike, risk_cut, band in itertools.product(
            level_values,
            momentum_windows,
            spike_values,
            (0.25, 0.50, 0.75, 1.00),
            (0.05, 0.10),
        ):
            cfg = DynamicRiskConfig(mode, level, window, spike, risk_cut, 0.50, band)
            configs[cfg.name] = cfg
    return list(configs.values())


def main() -> None:
    levels = load_daily_open_levels()
    reference = load_reference_weights()
    vkospi_signals = build_daily_vkospi_signals()
    arrays = prepare_arrays(levels, reference, vkospi_signals)
    if (pd.DatetimeIndex(arrays["signal_dates"]) >= pd.DatetimeIndex(arrays["dates"])).any():
        raise AssertionError("VKOSPI daily signal uses same-day or future information")

    _, baseline_full = simulate(arrays, None, keep_daily=False)
    baseline_cal = baseline_full.loc[:CAL_END]
    baseline_validation = baseline_full.loc[CAL_VALIDATION_START:CAL_END]
    baseline_cal_metrics = performance_summary(baseline_cal["return"])
    baseline_validation_metrics = performance_summary(baseline_validation["return"])

    rows: list[dict[str, object]] = []
    configs = candidate_configs()
    for number, cfg in enumerate(configs, start=1):
        _, monthly = simulate(arrays, cfg, end=CAL_END, keep_daily=False)
        full_metrics = performance_summary(monthly["return"])
        validation_metrics = performance_summary(
            monthly.loc[CAL_VALIDATION_START:CAL_END, "return"]
        )
        rows.append(
            {
                "name": cfg.name,
                "stage": "coarse",
                **asdict(cfg),
                **{f"Full_{key}": float(full_metrics[key]) for key in ("CAGR", "Sharpe", "MDD", "Calmar")},
                **{
                    f"Validation_{key}": float(validation_metrics[key])
                    for key in ("CAGR", "Sharpe", "MDD", "Calmar")
                },
                "AvgTurnover": float(monthly["turnover"].mean()),
                "TotalCost": float(monthly[["trade_cost", "fx_cost"]].sum().sum()),
            }
        )
        if number % 100 == 0:
            print(f"calibrated {number}/{len(configs)}")

    calibration = pd.DataFrame(rows)
    full_gate = (
        (calibration["Full_CAGR"] >= float(baseline_cal_metrics["CAGR"]))
        & (calibration["Full_Sharpe"] >= float(baseline_cal_metrics["Sharpe"]))
        & (calibration["Full_MDD"] >= float(baseline_cal_metrics["MDD"]))
    )
    validation_gate = (
        (calibration["Validation_CAGR"] >= float(baseline_validation_metrics["CAGR"]))
        & (calibration["Validation_Sharpe"] >= float(baseline_validation_metrics["Sharpe"]))
        & (calibration["Validation_MDD"] >= float(baseline_validation_metrics["MDD"]))
    )
    eligible = calibration.loc[full_gate & validation_gate].copy()
    if eligible.empty:
        raise RuntimeError("No VKOSPI dynamic overlay improved all three calibration objectives")

    # Refine only the coarse candidates that passed every objective in both
    # calibration windows. This keeps the search local and prevents the locked
    # test from choosing the defensive destination or trading-band parameters.
    refinement: dict[str, DynamicRiskConfig] = {}
    for _, seed in eligible.iterrows():
        for bond_share, risk_transfer, band in itertools.product(
            (0.00, 0.25, 0.50, 0.75, 1.00),
            (0.125, 0.25, 0.375),
            (0.075, 0.10, 0.15),
        ):
            cfg = DynamicRiskConfig(
                mode=str(seed["mode"]),
                level_threshold=float(seed["level_threshold"]),
                momentum_window=int(seed["momentum_window"]),
                spike_threshold=float(seed["spike_threshold"]),
                max_risk_transfer=risk_transfer,
                bond_share=bond_share,
                rebalance_band=band,
            )
            refinement[cfg.name] = cfg

    refinement_rows: list[dict[str, object]] = []
    for number, cfg in enumerate(refinement.values(), start=1):
        _, monthly = simulate(arrays, cfg, end=CAL_END, keep_daily=False)
        full_metrics = performance_summary(monthly["return"])
        validation_metrics = performance_summary(
            monthly.loc[CAL_VALIDATION_START:CAL_END, "return"]
        )
        refinement_rows.append(
            {
                "name": cfg.name,
                "stage": "refined",
                **asdict(cfg),
                **{
                    f"Full_{key}": float(full_metrics[key])
                    for key in ("CAGR", "Sharpe", "MDD", "Calmar")
                },
                **{
                    f"Validation_{key}": float(validation_metrics[key])
                    for key in ("CAGR", "Sharpe", "MDD", "Calmar")
                },
                "AvgTurnover": float(monthly["turnover"].mean()),
                "TotalCost": float(
                    monthly[["trade_cost", "fx_cost"]].sum().sum()
                ),
            }
        )
        if number % 100 == 0:
            print(f"refined {number}/{len(refinement)}")

    calibration = (
        pd.concat([calibration, pd.DataFrame(refinement_rows)], ignore_index=True)
        .sort_values("stage")
        .drop_duplicates("name", keep="last")
        .reset_index(drop=True)
    )
    full_gate = (
        (calibration["Full_CAGR"] >= float(baseline_cal_metrics["CAGR"]))
        & (calibration["Full_Sharpe"] >= float(baseline_cal_metrics["Sharpe"]))
        & (calibration["Full_MDD"] >= float(baseline_cal_metrics["MDD"]))
    )
    validation_gate = (
        (calibration["Validation_CAGR"] >= float(baseline_validation_metrics["CAGR"]))
        & (calibration["Validation_Sharpe"] >= float(baseline_validation_metrics["Sharpe"]))
        & (calibration["Validation_MDD"] >= float(baseline_validation_metrics["MDD"]))
    )
    eligible = calibration.loc[full_gate & validation_gate].copy()
    rank_columns = []
    for prefix in ("Full", "Validation"):
        for metric in ("CAGR", "Sharpe", "MDD"):
            rank = f"{prefix}_{metric}_rank"
            eligible[rank] = eligible[f"{prefix}_{metric}"].rank(pct=True)
            rank_columns.append(rank)
    eligible["MultiObjectiveScore"] = eligible[rank_columns].mean(axis=1)
    winner_row = eligible.sort_values(
        ["MultiObjectiveScore", "Validation_Calmar", "Full_Calmar"], ascending=False
    ).iloc[0]
    winner = DynamicRiskConfig(
        mode=str(winner_row["mode"]),
        level_threshold=float(winner_row["level_threshold"]),
        momentum_window=int(winner_row["momentum_window"]),
        spike_threshold=float(winner_row["spike_threshold"]),
        max_risk_transfer=float(winner_row["max_risk_transfer"]),
        bond_share=float(winner_row["bond_share"]),
        rebalance_band=float(winner_row["rebalance_band"]),
        financing_rate=float(winner_row["financing_rate"]),
    )

    winner_daily, winner_monthly = simulate(arrays, winner)
    reconciled_monthly = reconcile_to_monthly_reference(
        reference, baseline_full, winner_monthly
    )
    comparison_rows: list[dict[str, object]] = []
    for period, start, end in (
        ("calibration_2007_2017", baseline_full.index.min(), CAL_END),
        ("validation_2013_2017", CAL_VALIDATION_START, CAL_END),
        ("locked_2018_2026", TEST_START, baseline_full.index.max()),
        ("full_2007_2026", baseline_full.index.min(), baseline_full.index.max()),
    ):
        comparison_rows.append(metric_row(period, "ReferenceDaily", baseline_full.loc[start:end]))
        comparison_rows.append(
            metric_row(period, "VKOSPIDynamicActual", winner_monthly.loc[start:end])
        )
        comparison_rows.append(
            metric_row(period, "ReferenceMonthly", reference.loc[start:end])
        )
        comparison_rows.append(
            metric_row(
                period,
                "VKOSPIDynamicReconciled",
                reconciled_monthly.loc[start:end],
            )
        )
    comparison = pd.DataFrame(comparison_rows)

    base_locked = baseline_full.loc[TEST_START:]
    winner_locked = winner_monthly.loc[TEST_START:]
    base_locked_metrics = performance_summary(base_locked["return"])
    winner_locked_metrics = performance_summary(winner_locked["return"])
    locked_deltas = {
        metric: float(winner_locked_metrics[metric] - base_locked_metrics[metric])
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    reference_monthly_locked = reference.loc[TEST_START:]
    reconciled_locked = reconciled_monthly.loc[TEST_START:]
    reference_monthly_locked_metrics = performance_summary(
        reference_monthly_locked["return"]
    )
    reconciled_locked_metrics = performance_summary(reconciled_locked["return"])
    reconciled_locked_deltas = {
        metric: float(
            reconciled_locked_metrics[metric] - reference_monthly_locked_metrics[metric]
        )
        for metric in ("CAGR", "Sharpe", "MDD", "Calmar")
    }
    actual_pass = bool(
        locked_deltas["CAGR"] > 0
        and locked_deltas["Sharpe"] > 0
        and locked_deltas["MDD"] >= 0
    )
    reconciled_pass = bool(
        reconciled_locked_deltas["CAGR"] > 0
        and reconciled_locked_deltas["Sharpe"] > 0
        and reconciled_locked_deltas["MDD"] >= 0
    )
    promoted = actual_pass and reconciled_pass

    cost_rows: list[dict[str, object]] = []
    for cost in (0.5, 1.0, 1.5, 2.0):
        _, base_cost = simulate(arrays, None, cost_multiplier=cost, keep_daily=False)
        _, winner_cost = simulate(arrays, winner, cost_multiplier=cost, keep_daily=False)
        cost_rows.append(metric_row(f"cost_{cost:.1f}x_locked", "ReferenceDaily", base_cost.loc[TEST_START:]))
        cost_rows.append(metric_row(f"cost_{cost:.1f}x_locked", "VKOSPIDynamic", winner_cost.loc[TEST_START:]))

    subperiod_rows: list[dict[str, object]] = []
    for start, end in (
        (pd.Period("2018-01", "M"), pd.Period("2020-12", "M")),
        (pd.Period("2021-01", "M"), pd.Period("2023-12", "M")),
        (pd.Period("2024-01", "M"), winner_monthly.index.max()),
    ):
        label = f"{start}_{end}"
        subperiod_rows.append(metric_row(label, "ReferenceDaily", baseline_full.loc[start:end]))
        subperiod_rows.append(
            metric_row(label, "VKOSPIDynamicActual", winner_monthly.loc[start:end])
        )
        subperiod_rows.append(
            metric_row(label, "ReferenceMonthly", reference.loc[start:end])
        )
        subperiod_rows.append(
            metric_row(
                label,
                "VKOSPIDynamicReconciled",
                reconciled_monthly.loc[start:end],
            )
        )

    calibration.to_csv(RESULTS / "vkospi_dynamic_calibration.csv", index=False)
    comparison.to_csv(RESULTS / "vkospi_dynamic_comparison.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(RESULTS / "vkospi_dynamic_cost_sensitivity.csv", index=False)
    pd.DataFrame(subperiod_rows).to_csv(RESULTS / "vkospi_dynamic_subperiods.csv", index=False)
    winner_daily.to_csv(RESULTS / "vkospi_dynamic_daily.csv")
    winner_monthly.to_csv(RESULTS / "vkospi_dynamic_monthly.csv")
    reconciled_monthly.to_csv(RESULTS / "vkospi_dynamic_reconciled_monthly.csv")
    report = {
        "method": {
            "interpretation": "VKOSPI is the Korean equity implied-volatility index (domestic VIX)",
            "timing": "previous VKOSPI close only; position earns next open-to-open return",
            "overlay": "transfer a stress-scaled fraction of KODEX200 and USO to BOND and GLD",
            "reconciliation": "validated monthly reference multiplied by dynamic/base daily relative return; isolates overlay alpha from frequency conversion",
            "selection": "all CAGR, Sharpe, and MDD objectives must improve in both 2007-2017 and 2013-2017",
        },
        "winner": asdict(winner),
        "eligible_candidates": len(eligible),
        "total_candidates": len(calibration),
        "locked": {
            "actual_daily": {
                "reference": base_locked_metrics.to_dict(),
                "candidate": winner_locked_metrics.to_dict(),
                "deltas": locked_deltas,
                "passes_all_three": actual_pass,
            },
            "monthly_reference_reconciled": {
                "reference": reference_monthly_locked_metrics.to_dict(),
                "candidate": reconciled_locked_metrics.to_dict(),
                "deltas": reconciled_locked_deltas,
                "passes_all_three": reconciled_pass,
            },
            "promoted": promoted,
            "actual_daily_bootstrap": paired_multiobjective_bootstrap(
                base_locked["return"], winner_locked["return"]
            ),
            "reconciled_multiobjective_bootstrap": paired_multiobjective_bootstrap(
                reference_monthly_locked["return"], reconciled_locked["return"]
            ),
        },
    }
    (RESULTS / "vkospi_dynamic_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== DAILY RECONSTRUCTION VS MONTHLY REFERENCE ===")
    monthly_reference = performance_summary(reference.loc[baseline_full.index, "return"])
    print(
        pd.DataFrame(
            {"DailyReconstruction": performance_summary(baseline_full["return"]), "MonthlyReference": monthly_reference}
        ).loc[["CAGR", "Sharpe", "MDD", "Calmar"]]
    )
    print("\n=== WINNER ===")
    print(winner)
    print(f"eligible={len(eligible)}/{len(calibration)}")
    print("\n=== COMPARISON ===")
    print(
        comparison[["Period", "Strategy", "CAGR", "Sharpe", "MDD", "Calmar", "AvgTurnover"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n=== LOCKED DELTAS ===")
    print(json.dumps({"actual_daily": locked_deltas, "reconciled": reconciled_locked_deltas}, indent=2))
    print(f"PROMOTED={promoted}")


if __name__ == "__main__":
    main()
