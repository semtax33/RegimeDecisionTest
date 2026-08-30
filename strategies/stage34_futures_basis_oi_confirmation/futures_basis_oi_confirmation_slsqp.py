from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr

from strategies.core.regime_research import ASSETS, cdar, load_monthly_asset_returns
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    DOMESTIC_TRADE_COST,
    FOREIGN_WEIGHT_CHANGE_COST,
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    CATASTROPHE_ANNUAL_VOLATILITY,
    CATASTROPHE_CDAR,
    CDAR_CONFIDENCE,
    FULL_START,
    LOCKED_START,
    ONE_CALENDAR_YEAR,
    REGIME_COLUMNS,
    SLSQP_MAX_ITERATIONS,
    SLSQP_TOLERANCE,
    build_daily_stress_features,
    build_monthly_stress_signals,
    causal_expanding_midrank,
    estimate_conditional_moments,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    UNCONSTRAINED_LONG_ONLY_BOUNDS,
    expected_transaction_cost,
    metric_row,
    project_to_long_only_simplex,
    solver_summary,
)
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20,
)
from strategies.stage30_abnormal_surface_erp import (
    abnormal_surface_erp_slsqp as stage30,
)
from strategies.stage31_long_iv_state_dependence import (
    long_iv_state_dependence_slsqp as stage31,
)
from strategies.stage32_fear_premium_validation import (
    fear_premium_validation as stage32,
)
from strategies.stage33_wingasym_risk_context_validation import (
    wingasym_risk_context_validation as stage33,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
K200_FUTURES_XLSX = ROOT / "raw_data" / "260829_K200선물데이터.xlsx"
RESEARCH_END = pd.Period("2026-07", freq="M")
SIGNAL_DAYS = 20
MIN_CAUSAL_MONTHS = 60
SIGNIFICANCE_LEVEL = 0.10
HORIZONS = (1, 3, 6)
EQUITY_INDEX = ASSETS.index("KODEX200")

CONTROL_COLUMNS = (
    "vix6_stress_score",
    "recent_1m_return",
    "realized_vol_21d",
    "macro_fragility",
)
RISK_TARGETS = (
    "future_realized_vol_1m",
    "future_max_drawdown_1m",
    "future_max_drawdown_3m",
    "future_left_tail_1m",
)

SOURCE_FILES = (
    K200_FUTURES_XLSX,
    stage20.OHLCV_CACHE,
    stage20.COMPASS_PATH,
)
FROZEN_FILES = (
    Path(stage20.__file__),
    Path(stage30.__file__),
    Path(stage31.__file__),
    Path(stage32.__file__),
    Path(stage33.__file__),
    stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv",
    stage20.OUTPUT_DIR / "stage14_static_recomputed_monthly.csv",
    stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
)

FUTURES_COLUMNS = {
    1: "open",
    2: "high",
    3: "low",
    4: "close",
    5: "reference_price",
    6: "base_theoretical_price",
    7: "settlement_theoretical_price",
    8: "daily_contract_count",
    9: "daily_volume",
    12: "daily_notional_local_mn",
    13: "open_interest",
    15: "spread_deferred_close",
    17: "spread_deferred_volume",
    19: "spread_deferred_notional_local_mn",
    20: "provider_settlement_theory_gap",
    21: "dividend_index_future_value",
    22: "dividend_index_present_value",
    23: "implied_volatility_field",
    24: "days_to_expiry",
    25: "listing_date",
    26: "last_trading_date",
    27: "expiration_date",
    28: "contract_year_month",
    29: "strike_price_field",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def source_manifest() -> dict[str, dict[str, Any]]:
    return _manifest(SOURCE_FILES)


def frozen_manifest() -> dict[str, dict[str, Any]]:
    return _manifest(FROZEN_FILES)


def _read_futures_sheet(sheet_index: int, prefix: str) -> pd.DataFrame:
    positions = [0, *FUTURES_COLUMNS]
    raw = pd.read_excel(
        K200_FUTURES_XLSX,
        sheet_name=sheet_index,
        header=None,
        usecols=positions,
        engine="openpyxl",
    )
    data = raw.iloc[14:].copy()
    names = ["date", *[FUTURES_COLUMNS[position] for position in FUTURES_COLUMNS]]
    data.columns = names
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    for column in names[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[~data.index.duplicated(keep="last")]
    return data.add_prefix(f"{prefix}_")


def load_futures_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    near = _read_futures_sheet(0, "near")
    deferred = _read_futures_sheet(1, "deferred")
    daily = near.join(deferred, how="outer")

    valid_basis = (
        daily["near_close"].gt(0.0)
        & daily["near_settlement_theoretical_price"].gt(0.0)
    )
    daily["signed_close_theory_basis"] = np.where(
        valid_basis,
        daily["near_close"]
        / daily["near_settlement_theoretical_price"]
        - 1.0,
        np.nan,
    )
    same_contract = daily["near_contract_year_month"].eq(
        daily["near_contract_year_month"].shift(SIGNAL_DAYS)
    )
    daily["same_contract_20d"] = same_contract.astype(float)
    daily["basis_change_20d"] = daily["signed_close_theory_basis"].diff(
        SIGNAL_DAYS
    ).where(same_contract)
    valid_oi = (
        daily["near_open_interest"].gt(0.0)
        & daily["near_open_interest"].shift(SIGNAL_DAYS).gt(0.0)
        & same_contract
    )
    oi_ratio = (
        daily["near_open_interest"]
        / daily["near_open_interest"].shift(SIGNAL_DAYS)
    ).where(valid_oi)
    daily["oi_log_change_20d"] = np.log(oi_ratio)
    daily["futures_turnover"] = (
        daily["near_daily_volume"]
        / daily["near_open_interest"].shift(1)
    ).where(
        daily["near_open_interest"].shift(1).gt(0.0)
        & daily["near_contract_year_month"].eq(
            daily["near_contract_year_month"].shift(1)
        )
    )
    calendar_days = (
        daily["deferred_days_to_expiry"] - daily["near_days_to_expiry"]
    )
    valid_calendar = (
        daily["near_close"].gt(0.0)
        & daily["deferred_close"].gt(0.0)
        & calendar_days.gt(20.0)
    )
    daily["annualized_calendar_spread"] = (
        np.log(daily["deferred_close"] / daily["near_close"])
        * 365.0
        / calendar_days
    ).where(valid_calendar)
    daily["futures_dislocation"] = -daily["signed_close_theory_basis"]

    provider = daily["near_provider_settlement_theory_gap"]
    provider_early = provider.loc["2007":"2010"].dropna()
    provider_late = provider.loc["2011":].dropna()
    audit = {
        "source": str(K200_FUTURES_XLSX.resolve()),
        "sheets": ["최근월물", "차근월물"],
        "rows": int(len(daily)),
        "first_valid_basis_date": str(
            daily["signed_close_theory_basis"].dropna().index.min().date()
        ),
        "last_valid_basis_date": str(
            daily["signed_close_theory_basis"].dropna().index.max().date()
        ),
        "primary_formula": (
            "near close / near settlement theoretical price - 1"
        ),
        "primary_sign": (
            "positive=premium/buying pressure; negative=discount/selling pressure"
        ),
        "basis_change": (
            "20-trading-day difference, retained only when contract_year_month "
            "is unchanged"
        ),
        "oi_change": (
            "20-trading-day log change, retained only within the same contract"
        ),
        "provider_gap_used_as_signal": False,
        "provider_gap_reason": (
            "the field is an unsigned settlement/theory gap and becomes "
            "zero-heavy after 2011; it cannot identify premium versus discount"
        ),
        "provider_gap_nonzero_share_2007_2010": float(
            provider_early.ne(0.0).mean()
        ),
        "provider_gap_nonzero_share_2011_2026": float(
            provider_late.ne(0.0).mean()
        ),
        "implied_volatility_nonzero_observations": int(
            daily["near_implied_volatility_field"].fillna(0.0).ne(0.0).sum()
        ),
        "embedded_spread_close_observations": int(
            daily["near_spread_deferred_close"].notna().sum()
        ),
        "calendar_spread_source": (
            "direct deferred continuous-sheet close divided by near close; "
            "embedded spread fields are too sparse for the full study"
        ),
        "winsorization": False,
        "technical_indicator_search": False,
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit


def _causal_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean = values.shift(1).expanding(min_periods=MIN_CAUSAL_MONTHS).mean()
    std = values.shift(1).expanding(min_periods=MIN_CAUSAL_MONTHS).std(ddof=1)
    return ((values - mean) / std.where(std > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )


def build_monthly_futures_signals(daily: pd.DataFrame) -> pd.DataFrame:
    required = [
        "signed_close_theory_basis",
        "basis_change_20d",
        "oi_log_change_20d",
        "futures_turnover",
        "annualized_calendar_spread",
        "futures_dislocation",
        "near_days_to_expiry",
        "near_contract_year_month",
    ]
    rows: list[dict[str, Any]] = []
    for signal_month, group in daily.groupby(daily.index.to_period("M")):
        complete = group.dropna(
            subset=[
                "signed_close_theory_basis",
                "basis_change_20d",
                "oi_log_change_20d",
            ]
        )
        if complete.empty:
            continue
        current = complete.iloc[-1]
        rows.append(
            {
                "target_month": signal_month + 1,
                "futures_signal_month": signal_month,
                "futures_signal_date": complete.index[-1],
                **{column: float(current[column]) for column in required},
            }
        )
    signals = pd.DataFrame(rows).set_index("target_month").sort_index()
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    signals["basis_change_z"] = _causal_zscore(signals["basis_change_20d"])
    signals["oi_change_z"] = _causal_zscore(signals["oi_log_change_20d"])
    signals["basis_oi_interaction_score"] = (
        signals["basis_change_z"] * signals["oi_change_z"]
    )
    rank = causal_expanding_midrank(signals["futures_dislocation"])
    prior_count = signals["futures_dislocation"].notna().shift(1).fillna(False)
    prior_count = prior_count.astype(int).cumsum()
    signals["futures_dislocation_rank"] = rank.where(
        prior_count >= MIN_CAUSAL_MONTHS
    )
    signals["stress_confirmation_multiplier"] = (
        2.0 * signals["futures_dislocation_rank"]
    )
    return signals.replace([np.inf, -np.inf], np.nan)


def _forward_compound(series: pd.Series, horizon: int) -> pd.Series:
    legs = [series.shift(-offset) for offset in range(horizon)]
    frame = pd.concat(legs, axis=1)
    valid = frame.notna().all(axis=1)
    return frame.add(1.0).prod(axis=1).sub(1.0).where(valid)


def _realized_volatility_signal(close: pd.Series) -> pd.Series:
    daily = np.log(close.where(close > 0.0)).diff()
    rolling = daily.rolling(21, min_periods=15).std(ddof=1) * math.sqrt(252.0)
    monthly = rolling.groupby(rolling.index.to_period("M")).last()
    monthly.index = pd.PeriodIndex(monthly.index + 1, freq="M")
    return monthly.rename("realized_vol_21d")


def _forward_risk_targets(close: pd.Series) -> pd.DataFrame:
    close = close.dropna().sort_index()
    log_returns = np.log(close).diff()
    periods = pd.period_range(
        close.index.min().to_period("M"), RESEARCH_END, freq="M"
    )
    rows: list[dict[str, Any]] = []
    for period in periods:
        month_returns = log_returns.loc[
            log_returns.index.to_period("M") == period
        ].dropna()
        row: dict[str, Any] = {
            "target_month": period,
            "future_realized_vol_1m": (
                float(month_returns.std(ddof=1) * math.sqrt(252.0))
                if len(month_returns) >= 15
                else np.nan
            ),
        }
        for horizon, minimum_prices in ((1, 16), (3, 46)):
            end_period = period + horizon - 1
            before = close.loc[close.index < period.start_time]
            within = close.loc[
                (close.index >= period.start_time)
                & (close.index <= end_period.end_time)
            ]
            if before.empty or len(within) < minimum_prices - 1:
                value = np.nan
            else:
                path = pd.concat([before.iloc[[-1]], within])
                wealth = path / float(path.iloc[0])
                value = float(-(wealth / wealth.cummax() - 1.0).min())
            row[f"future_max_drawdown_{horizon}m"] = value
        rows.append(row)
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _causal_tail_event(monthly_return: pd.Series) -> pd.Series:
    threshold = (
        monthly_return.shift(1)
        .expanding(min_periods=MIN_CAUSAL_MONTHS)
        .quantile(0.05)
    )
    return pd.Series(
        np.where(
            monthly_return.notna() & threshold.notna(),
            (monthly_return <= threshold).astype(float),
            np.nan,
        ),
        index=monthly_return.index,
        name="future_left_tail_1m",
    )


def _load_period_csv(path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[column] = pd.PeriodIndex(frame[column], freq="M")
    return frame.set_index(column).sort_index()


def build_research_frame(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    equity_close: pd.Series,
) -> pd.DataFrame:
    frame = signals.copy()
    for horizon in HORIZONS:
        frame[f"future_{horizon}m_return"] = _forward_compound(
            returns["KODEX200"], horizon
        )
    frame["recent_1m_return"] = returns["KODEX200"].shift(1)
    frame["realized_vol_21d"] = _realized_volatility_signal(equity_close)
    frame["macro_fragility"] = (
        probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    )
    frame["vix6_stress_score"] = stress["stress_score"]
    frame = frame.join(_forward_risk_targets(equity_close), how="left")
    monthly_close = equity_close.groupby(equity_close.index.to_period("M")).last()
    monthly_return = monthly_close.pct_change().loc[:RESEARCH_END]
    frame["future_left_tail_1m"] = _causal_tail_event(monthly_return)

    stage20_path = _load_period_csv(
        stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv", "month"
    )
    stage14_path = _load_period_csv(
        stage20.OUTPUT_DIR / "stage14_static_recomputed_monthly.csv", "month"
    )
    frame["stage20_return"] = stage20_path["return"]
    frame["stage14_return"] = stage14_path["return"]
    frame["stage20_w_kodex200"] = stage20_path["w_KODEX200"]
    frame["stage14_w_kodex200"] = stage14_path["w_KODEX200"]
    frame["stage20_defense"] = np.where(
        frame["stage20_w_kodex200"].notna()
        & frame["stage14_w_kodex200"].notna(),
        (
            frame["stage20_w_kodex200"]
            < frame["stage14_w_kodex200"] - 1e-10
        ).astype(float),
        np.nan,
    )
    frame["stage20_false_positive_1m"] = np.where(
        frame["stage20_defense"].eq(1.0),
        (frame["stage20_return"] < frame["stage14_return"]).astype(float),
        np.nan,
    )
    return frame.loc[FULL_START:RESEARCH_END].replace(
        [np.inf, -np.inf], np.nan
    )


def _standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        std = float(frame[column].std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"No usable variation in {column}")
        output[column] = (frame[column] - frame[column].mean()) / std
    return output


def _fit_ols_hac(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    lags: int,
) -> tuple[Any, pd.DataFrame]:
    complete = frame[[target, *predictors]].dropna()
    x = _standardize(complete, predictors)
    fit = sm.OLS(complete[target], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return fit, complete


def return_predictive_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = {
        "BasisOnly": ["basis_change_20d"],
        "BasisFullControls": ["basis_change_20d", *CONTROL_COLUMNS],
        "BasisOIInteractionFullControls": [
            "basis_change_20d",
            "oi_log_change_20d",
            *CONTROL_COLUMNS,
        ],
    }
    for horizon in HORIZONS:
        target = f"future_{horizon}m_return"
        for model_name, predictors in models.items():
            complete = frame[[target, *predictors]].dropna()
            x = _standardize(complete, predictors)
            if model_name == "BasisOIInteractionFullControls":
                x["basis_x_oi"] = (
                    x["basis_change_20d"] * x["oi_log_change_20d"]
                )
            fit = sm.OLS(complete[target], sm.add_constant(x)).fit(
                cov_type="HAC", cov_kwds={"maxlags": horizon}
            )
            ic, ic_p = spearmanr(
                complete["basis_change_20d"], complete[target]
            )
            rows.append(
                {
                    "HorizonMonths": horizon,
                    "Model": model_name,
                    "Observations": int(len(complete)),
                    "BasisStandardizedBeta": float(
                        fit.params["basis_change_20d"]
                    ),
                    "BasisHACPValue": float(
                        fit.pvalues["basis_change_20d"]
                    ),
                    "BasisSpearmanIC": float(ic),
                    "BasisSpearmanICPValue": float(ic_p),
                    "OIStandardizedBeta": (
                        float(fit.params["oi_log_change_20d"])
                        if "oi_log_change_20d" in fit.params
                        else np.nan
                    ),
                    "OIHACPValue": (
                        float(fit.pvalues["oi_log_change_20d"])
                        if "oi_log_change_20d" in fit.pvalues
                        else np.nan
                    ),
                    "InteractionStandardizedBeta": (
                        float(fit.params["basis_x_oi"])
                        if "basis_x_oi" in fit.params
                        else np.nan
                    ),
                    "InteractionHACPValue": (
                        float(fit.pvalues["basis_x_oi"])
                        if "basis_x_oi" in fit.pvalues
                        else np.nan
                    ),
                    "AdjustedR2": float(fit.rsquared_adj),
                    "ExpectedBasisSign": "positive",
                    "ExpectedInteractionSign": "positive",
                    "HACLags": horizon,
                }
            )
    return pd.DataFrame(rows)


def risk_predictive_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in RISK_TARGETS:
        binary = target == "future_left_tail_1m"
        lags = 3 if target.endswith("3m") else 1
        predictors = ["futures_dislocation", *CONTROL_COLUMNS]
        complete = frame[[target, *predictors]].dropna()
        x = _standardize(complete, predictors)
        if binary:
            fit = sm.GLM(
                complete[target],
                sm.add_constant(x),
                family=sm.families.Binomial(),
            ).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            score = 1.0 - fit.llf / fit.llnull
            score_name = "McFaddenPseudoR2"
            events = int(complete[target].sum())
        else:
            fit = sm.OLS(complete[target], sm.add_constant(x)).fit(
                cov_type="HAC", cov_kwds={"maxlags": lags}
            )
            score = fit.rsquared_adj
            score_name = "AdjustedR2"
            events = np.nan
        rows.append(
            {
                "Target": target,
                "Observations": int(len(complete)),
                "Events": events,
                "DislocationStandardizedBeta": float(
                    fit.params["futures_dislocation"]
                ),
                "DislocationHACPValue": float(
                    fit.pvalues["futures_dislocation"]
                ),
                "ExpectedSign": "positive",
                "FitScoreName": score_name,
                "FitScore": float(score),
                "HACLags": lags,
            }
        )
    return pd.DataFrame(rows)


def false_positive_regression(frame: pd.DataFrame) -> pd.DataFrame:
    defense = frame.loc[frame["stage20_defense"].eq(1.0)]
    predictors = ["futures_dislocation", *CONTROL_COLUMNS]
    complete = defense[["stage20_false_positive_1m", *predictors]].dropna()
    x = _standardize(complete, predictors)
    fit = sm.GLM(
        complete["stage20_false_positive_1m"],
        sm.add_constant(x),
        family=sm.families.Binomial(),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return pd.DataFrame(
        [
            {
                "Observations": int(len(complete)),
                "FalsePositiveEvents": int(
                    complete["stage20_false_positive_1m"].sum()
                ),
                "DislocationStandardizedBeta": float(
                    fit.params["futures_dislocation"]
                ),
                "DislocationHACPValue": float(
                    fit.pvalues["futures_dislocation"]
                ),
                "ExpectedSign": "negative",
                "McFaddenPseudoR2": float(1.0 - fit.llf / fit.llnull),
            }
        ]
    )


def futures_confirmation_state_table(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(
        subset=["vix6_stress_score", "futures_dislocation_rank"]
    ).copy()
    vix_high = valid["vix6_stress_score"] >= 0.5
    futures_high = valid["futures_dislocation_rank"] >= 0.5
    valid["State"] = np.select(
        [
            ~vix_high & ~futures_high,
            vix_high & ~futures_high,
            ~vix_high & futures_high,
            vix_high & futures_high,
        ],
        [
            "VIX6 Low / Futures Normal",
            "VIX6 High / Futures Normal",
            "VIX6 Low / Futures Stress",
            "VIX6 High / Futures Stress",
        ],
        default="Unavailable",
    )
    rows: list[dict[str, Any]] = []
    for state, group in valid.groupby("State"):
        defense = group.loc[group["stage20_defense"].eq(1.0)]
        rows.append(
            {
                "State": state,
                "Months": int(len(group)),
                "DefenseMonths": int(group["stage20_defense"].sum()),
                "FalsePositiveRateAmongDefense": float(
                    defense["stage20_false_positive_1m"].mean()
                ),
                "MeanFutureReturn1M": float(
                    group["future_1m_return"].mean()
                ),
                "MeanFutureReturn3M": float(
                    group["future_3m_return"].mean()
                ),
                "MeanFutureRealizedVol1M": float(
                    group["future_realized_vol_1m"].mean()
                ),
                "MeanFutureMaxDrawdown3M": float(
                    group["future_max_drawdown_3m"].mean()
                ),
                "FutureLeftTailRate1M": float(
                    group["future_left_tail_1m"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def add_causal_return_calibration(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    output = signals.copy()
    rows: list[dict[str, Any]] = []
    for month in output.index:
        history = output.index[output.index < month].intersection(returns.index)
        calibration = pd.concat(
            [
                output.loc[history, "basis_change_z"],
                output.loc[history, "basis_oi_interaction_score"],
                returns.loc[history, "KODEX200"],
            ],
            axis=1,
        )
        calibration.columns = ["basis", "interaction", "return"]
        calibration = calibration.dropna()
        basis_slope = 0.0
        augmented_basis = 0.0
        augmented_interaction = 0.0
        if len(calibration) >= MIN_CAUSAL_MONTHS:
            y = calibration["return"].to_numpy(dtype=float)
            x_basis = calibration["basis"].to_numpy(dtype=float)
            centered_basis = x_basis - x_basis.mean()
            denominator = float(centered_basis @ centered_basis)
            if denominator > 0.0:
                raw = float(centered_basis @ (y - y.mean()) / denominator)
                basis_slope = max(raw, 0.0)
            x = calibration[["basis", "interaction"]].to_numpy(dtype=float)
            x_centered = x - x.mean(axis=0)
            y_centered = y - y.mean()
            if np.linalg.matrix_rank(x_centered) == 2:
                coefficients, _ = nnls(x_centered, y_centered)
                augmented_basis = float(coefficients[0])
                augmented_interaction = float(coefficients[1])
        current_basis = float(output.loc[month, "basis_change_z"])
        current_interaction = float(
            output.loc[month, "basis_oi_interaction_score"]
        )
        rows.append(
            {
                "target_month": month,
                "calibration_observations": int(len(calibration)),
                "basis_calibration_slope": basis_slope,
                "augmented_basis_slope": augmented_basis,
                "augmented_interaction_slope": augmented_interaction,
                "basis_mu_adjustment_KODEX200": basis_slope * current_basis,
                "basis_oi_mu_adjustment_KODEX200": (
                    augmented_basis * current_basis
                    + augmented_interaction * current_interaction
                ),
            }
        )
    calibration_frame = pd.DataFrame(rows).set_index("target_month")
    calibration_frame.index = pd.PeriodIndex(calibration_frame.index, freq="M")
    return output.join(calibration_frame)


def _solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    futures_signal: pd.Series,
    pretrade: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    original_expected_return, base_covariance, moment_detail = (
        estimate_conditional_moments(
            history=history,
            historical_probabilities=historical_probabilities,
            current_probabilities=current_probabilities,
            historical_stress=historical_stress,
            current_stress=current_stress,
            historical_recovery=historical_recovery,
            current_recovery=current_recovery,
            use_short_term_stress=True,
        )
    )
    macro_expected_return = np.asarray(
        moment_detail["macro_expected_monthly_return"], dtype=float
    )
    original_stress_adjustment = np.asarray(
        moment_detail["stress_return_adjustment"], dtype=float
    )
    technical = stage20.apply_technical_inputs(
        macro_expected_return, base_covariance, technical_signal
    )
    filtered_macro = np.asarray(
        technical["filtered_macro_expected_return"], dtype=float
    ).copy()
    stress_adjustment = original_stress_adjustment.copy()
    mu_adjustment = 0.0
    multiplier = 1.0
    if mode in {"basis_alpha", "combined"}:
        mu_adjustment = float(
            futures_signal["basis_mu_adjustment_KODEX200"]
        )
    elif mode == "basis_oi_alpha":
        mu_adjustment = float(
            futures_signal["basis_oi_mu_adjustment_KODEX200"]
        )
    if mode in {"risk_confirmation", "combined"}:
        multiplier = float(
            futures_signal["stress_confirmation_multiplier"]
        )
        stress_adjustment[EQUITY_INDEX] *= multiplier
    filtered_macro[EQUITY_INDEX] += mu_adjustment
    expected_return = filtered_macro + stress_adjustment
    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)

    common = history.index.intersection(historical_stress.dropna().index)
    historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
    initial = (
        project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        monthly_return = float(weights @ expected_return)
        monthly_variance = max(float(weights @ covariance @ weights), 0.0)
        realized_history = historical_returns @ weights
        downside_semivariance = float(
            np.mean(np.minimum(realized_history, 0.0) ** 2)
        )
        transaction_cost = expected_transaction_cost(weights, pretrade)
        utility = (
            monthly_return
            - 0.5 * monthly_variance
            - downside_semivariance
            - transaction_cost
        )
        return {
            "expected_monthly_return": monthly_return,
            "expected_monthly_variance": monthly_variance,
            "expected_annual_log_growth": 12.0
            * (monthly_return - 0.5 * monthly_variance),
            "downside_risk_aversion_lambda": 1.0,
            "variance_penalty": 0.5 * monthly_variance,
            "downside_semivariance": downside_semivariance,
            "downside_penalty": downside_semivariance,
            "estimated_transaction_cost": transaction_cost,
            "monthly_utility": utility,
        }

    def objective(weights: np.ndarray) -> float:
        return -portfolio_values(weights)["monthly_utility"]

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: (
                CATASTROPHE_ANNUAL_VOLATILITY - annual_volatility(weights)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                CATASTROPHE_CDAR
                + cdar(historical_returns @ weights, CDAR_CONFIDENCE)
            ),
        },
    ]
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": SLSQP_TOLERANCE},
    )
    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = project_to_long_only_simplex(result.x)
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": SLSQP_MAX_ITERATIONS,
                "ftol": SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                f"Stage34 and fallback solves failed: {result.message}; "
                f"{fallback.message}"
            )
        result = fallback
        weights = project_to_long_only_simplex(fallback.x)
        used_fallback = True

    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = cdar(historical_returns @ weights, CDAR_CONFIDENCE)
    detail = {
        **values,
        "policy": f"Stage34_{mode}",
        "solver_success": bool(result.success),
        "used_fallback": used_fallback,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_value": float(result.fun),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "sum_error": abs(float(weights.sum()) - 1.0),
        "volatility_slack": CATASTROPHE_ANNUAL_VOLATILITY - annual_vol,
        "cdar_slack": CATASTROPHE_CDAR + historical_cdar,
        "futures_mu_adjustment_KODEX200": mu_adjustment,
        "futures_stress_confirmation_multiplier": multiplier,
        "original_stress_mu_adjustment_KODEX200": float(
            original_stress_adjustment[EQUITY_INDEX]
        ),
        "confirmed_stress_mu_adjustment_KODEX200": float(
            stress_adjustment[EQUITY_INDEX]
        ),
        "macro_expected_monthly_return": macro_expected_return.tolist(),
        "stress_return_adjustment": stress_adjustment.tolist(),
        "original_expected_return": original_expected_return.tolist(),
        "filtered_expected_return": expected_return.tolist(),
        **technical,
    }
    detail.pop("adjusted_covariance")
    return weights, detail


def run_backtest(
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress_signals: pd.DataFrame,
    technical_signals: pd.DataFrame,
    futures_signals: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(futures_signals.dropna(
        subset=[
            "basis_mu_adjustment_KODEX200",
            "basis_oi_mu_adjustment_KODEX200",
            "stress_confirmation_multiplier",
        ]
    ).index)
    rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        history = returns.loc[returns.index < month, ASSETS]
        if len(history) < ONE_CALENDAR_YEAR:
            continue
        probability = probabilities.loc[month]
        stress = float(stress_signals.loc[month, "stress_score"])
        recovery = float(stress_signals.loc[month, "recovery_score"])
        technical_signal = technical_signals.loc[month]
        futures_signal = futures_signals.loc[month]
        weights, detail = _solve_weights(
            history,
            probabilities.loc[probabilities.index < month],
            probability,
            stress_signals.loc[
                stress_signals.index < month, "stress_score"
            ],
            stress,
            stress_signals.loc[
                stress_signals.index < month, "recovery_score"
            ],
            recovery,
            technical_signal,
            futures_signal,
            pretrade,
            mode,
        )
        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = float(np.abs(change).sum()) * DOMESTIC_TRADE_COST
        foreign_indices = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign_indices].sum()))
            * FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = returns.loc[month, ASSETS].to_numpy(dtype=float)
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False
        row: dict[str, Any] = {
            "month": month,
            "macro_signal_month": probability["signal_month"],
            "stress_signal_month": stress_signals.loc[
                month, "stress_signal_month"
            ],
            "technical_signal_month": technical_signal[
                "technical_signal_month"
            ],
            "futures_signal_month": futures_signal["futures_signal_month"],
            "futures_signal_date": futures_signal["futures_signal_date"],
            "stress_score": stress,
            "recovery_score": recovery,
            "return": net_return,
            "gross_return": gross_return,
            "nav": nav,
            "drawdown": nav / peak - 1.0,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "fx_cost": fx_cost,
            **{
                column: float(probability[column])
                for column in REGIME_COLUMNS
            },
            **{
                f"w_{asset}": float(weights[index])
                for index, asset in enumerate(ASSETS)
            },
            **{
                key: detail[key]
                for key in [
                    "policy",
                    "solver_success",
                    "used_fallback",
                    "solver_status",
                    "solver_message",
                    "solver_iterations",
                    "objective_value",
                    "expected_monthly_return",
                    "expected_monthly_variance",
                    "expected_annual_log_growth",
                    "downside_risk_aversion_lambda",
                    "estimated_transaction_cost",
                    "monthly_utility",
                    "expected_annual_volatility",
                    "historical_cdar",
                    "sum_error",
                    "volatility_slack",
                    "cdar_slack",
                    "futures_mu_adjustment_KODEX200",
                    "futures_stress_confirmation_multiplier",
                    "original_stress_mu_adjustment_KODEX200",
                    "confirmed_stress_mu_adjustment_KODEX200",
                ]
            },
        }
        for column in [
            "signed_close_theory_basis",
            "basis_change_20d",
            "oi_log_change_20d",
            "basis_change_z",
            "oi_change_z",
            "basis_oi_interaction_score",
            "futures_dislocation",
            "futures_dislocation_rank",
            "annualized_calendar_spread",
            "futures_turnover",
            "calibration_observations",
            "basis_calibration_slope",
            "augmented_basis_slope",
            "augmented_interaction_slope",
        ]:
            row[column] = float(futures_signal[column])
        rows.append(row)
    output = pd.DataFrame(rows).set_index("month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def _performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, Any]] = []
    for name, path in paths.items():
        rows.extend(
            [
                metric_row(name, path, "full_2007_2026", FULL_START, common_end),
                metric_row(
                    name,
                    path,
                    "early_2007_2017",
                    FULL_START,
                    LOCKED_START - 1,
                ),
                metric_row(
                    name,
                    path,
                    "locked_2018_2026",
                    LOCKED_START,
                    common_end,
                ),
            ]
        )
    return pd.DataFrame(rows)


def gate_decision(
    returns: pd.DataFrame,
    risks: pd.DataFrame,
    false_positive: pd.DataFrame,
    states: pd.DataFrame,
    performance: pd.DataFrame,
) -> dict[str, Any]:
    direction = returns.loc[
        returns["Model"].eq("BasisFullControls")
    ].set_index("HorizonMonths")
    interaction = returns.loc[
        returns["Model"].eq("BasisOIInteractionFullControls")
    ].set_index("HorizonMonths")
    direction_gates = {
        "one_month_basis_positive_p_below_10pct": bool(
            direction.loc[1, "BasisStandardizedBeta"] > 0.0
            and direction.loc[1, "BasisHACPValue"] < SIGNIFICANCE_LEVEL
        ),
        "basis_positive_at_3m_or_6m": bool(
            (direction.loc[[3, 6], "BasisStandardizedBeta"] > 0.0).any()
        ),
    }
    oi_gates = {
        "one_month_interaction_positive_p_below_10pct": bool(
            interaction.loc[1, "InteractionStandardizedBeta"] > 0.0
            and interaction.loc[1, "InteractionHACPValue"]
            < SIGNIFICANCE_LEVEL
        ),
        "interaction_positive_at_3m": bool(
            interaction.loc[3, "InteractionStandardizedBeta"] > 0.0
        ),
    }
    positive_risk = risks["DislocationStandardizedBeta"] > 0.0
    significant_risk = positive_risk & (
        risks["DislocationHACPValue"] < SIGNIFICANCE_LEVEL
    )
    material = risks["Target"].isin(
        [
            "future_max_drawdown_1m",
            "future_max_drawdown_3m",
            "future_left_tail_1m",
        ]
    )
    risk_gates = {
        "positive_at_three_of_four_risk_targets": bool(
            int(positive_risk.sum()) >= 3
        ),
        "significant_at_two_risk_targets": bool(
            int(significant_risk.sum()) >= 2
        ),
        "significant_for_drawdown_or_tail": bool(
            (significant_risk & material).any()
        ),
    }
    fp_row = false_positive.iloc[0]
    indexed_states = states.set_index("State")
    normal_fp = indexed_states.loc[
        "VIX6 High / Futures Normal", "FalsePositiveRateAmongDefense"
    ]
    stress_fp = indexed_states.loc[
        "VIX6 High / Futures Stress", "FalsePositiveRateAmongDefense"
    ]
    fp_gates = {
        "dislocation_reduces_false_positive_logit_p_below_10pct": bool(
            fp_row["DislocationStandardizedBeta"] < 0.0
            and fp_row["DislocationHACPValue"] < SIGNIFICANCE_LEVEL
        ),
        "vix_high_normal_futures_has_more_false_positives": bool(
            normal_fp > stress_fp
        ),
    }
    direction_pass = bool(all(direction_gates.values()))
    oi_pass = bool(all(oi_gates.values()))
    risk_pass = bool(all(risk_gates.values()))
    fp_pass = bool(all(fp_gates.values()))

    full = performance.loc[
        performance["Period"].eq("full_2007_2026")
    ].set_index("Strategy")
    baseline = full.loc["Stage20_Frozen"]
    candidate_names: list[str] = []
    if direction_pass:
        candidate_names.append("Stage34_BasisAlpha")
    if direction_pass and oi_pass:
        candidate_names.append("Stage34_BasisOIAlpha")
    if risk_pass and fp_pass:
        candidate_names.append("Stage34_RiskConfirmation")
    if direction_pass and risk_pass and fp_pass:
        candidate_names.append("Stage34_Combined")
    performance_gates: dict[str, dict[str, bool]] = {}
    eligible: list[str] = []
    for name in candidate_names:
        row = full.loc[name]
        checks = {
            "cagr_above_stage20": bool(row["CAGR"] > baseline["CAGR"]),
            "sharpe_not_below_stage20": bool(
                row["Sharpe"] >= baseline["Sharpe"]
            ),
            "mdd_not_worse_by_more_than_0_5pctpt": bool(
                row["MDD"] >= baseline["MDD"] - 0.005
            ),
        }
        performance_gates[name] = checks
        if all(checks.values()):
            eligible.append(name)
    promoted = (
        max(eligible, key=lambda name: float(full.loc[name, "CAGR"]))
        if eligible
        else None
    )
    decision = (
        f"promote_{promoted}"
        if promoted
        else "keep_stage20_frozen_futures_signals_fail_promotion_gate"
    )
    return {
        "direction_gates": direction_gates,
        "oi_confirmation_gates": oi_gates,
        "risk_sensor_gates": risk_gates,
        "false_positive_confirmation_gates": fp_gates,
        "direction_pass": direction_pass,
        "oi_confirmation_pass": oi_pass,
        "risk_sensor_pass": risk_pass,
        "false_positive_confirmation_pass": fp_pass,
        "performance_gates": performance_gates,
        "promoted_strategy": promoted,
        "decision": decision,
    }


def _plot_factor_mechanism(
    return_tests: pd.DataFrame,
    risk_tests: pd.DataFrame,
    path: Path,
) -> None:
    direction = return_tests.loc[
        return_tests["Model"].eq("BasisFullControls")
    ]
    interaction = return_tests.loc[
        return_tests["Model"].eq("BasisOIInteractionFullControls")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(3)
    axes[0].bar(
        x - 0.18,
        direction["BasisStandardizedBeta"] * 100.0,
        0.36,
        label="Basis change",
        color="#326f91",
    )
    axes[0].bar(
        x + 0.18,
        interaction["InteractionStandardizedBeta"] * 100.0,
        0.36,
        label="Basis × OI",
        color="#d28b45",
    )
    axes[0].axhline(0, color="#222", linewidth=1)
    axes[0].set_xticks(x, ["1M", "3M", "6M"])
    axes[0].set_ylabel("Forward-return beta (%p)")
    axes[0].set_title("Directional mechanism after controls")
    axes[0].legend(frameon=False)
    colors = [
        "#326f91" if value > 0 else "#c55752"
        for value in risk_tests["DislocationStandardizedBeta"]
    ]
    axes[1].bar(
        np.arange(len(risk_tests)),
        risk_tests["DislocationStandardizedBeta"],
        color=colors,
    )
    axes[1].axhline(0, color="#222", linewidth=1)
    axes[1].set_xticks(
        np.arange(len(risk_tests)),
        ["1M vol", "1M MDD", "3M MDD", "1M tail"],
        rotation=15,
        ha="right",
    )
    axes[1].set_title("Dislocation future-risk beta")
    axes[1].set_ylabel("Standardized beta")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_performance(performance: pd.DataFrame, path: Path) -> None:
    full = performance.loc[
        performance["Period"].eq("full_2007_2026")
    ].copy()
    x = np.arange(len(full))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics = [("CAGR", "CAGR"), ("Sharpe", "Sharpe"), ("MDD", "MDD")]
    for ax, (column, title) in zip(axes, metrics):
        values = full[column] * (100.0 if column != "Sharpe" else 1.0)
        colors = ["#315f78" if "Stage20" in name else "#8aa8b7" for name in full["Strategy"]]
        ax.bar(x, values, color=colors)
        ax.axhline(0.0, color="#222", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x, [name.replace("Stage34_", "") for name in full["Strategy"]], rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()
    daily, data_audit = load_futures_daily()
    signals = build_monthly_futures_signals(daily)
    returns, return_audit = load_monthly_asset_returns(False)
    probabilities, macro_audit = build_macro_probabilities(returns)
    stress = build_monthly_stress_signals(
        returns.index, build_daily_stress_features()
    )
    market, market_audit = stage20.load_daily_asset_ohlcv()
    equity_close = market["KODEX200"]["close"].dropna()
    research = build_research_frame(
        signals, returns, probabilities, stress, equity_close
    )
    return_tests = return_predictive_regressions(research)
    risk_tests = risk_predictive_regressions(research)
    false_positive = false_positive_regression(research)
    state_table = futures_confirmation_state_table(research)

    calibrated_signals = add_causal_return_calibration(signals, returns)
    technical = _load_period_csv(
        stage20.OUTPUT_DIR / "monthly_technical_signals.csv", "target_month"
    )
    modes = {
        "Stage34_NoChangeReproduction": "baseline_reproduction",
        "Stage34_BasisAlpha": "basis_alpha",
        "Stage34_BasisOIAlpha": "basis_oi_alpha",
        "Stage34_RiskConfirmation": "risk_confirmation",
        "Stage34_Combined": "combined",
    }
    candidate_paths = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated_signals,
            mode,
        )
        for name, mode in modes.items()
    }
    stage20_path = _load_period_csv(
        stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv", "month"
    )
    paths = {"Stage20_Frozen": stage20_path, **candidate_paths}
    performance = _performance_table(paths)
    gates = gate_decision(
        return_tests, risk_tests, false_positive, state_table, performance
    )
    bootstrap_rows: list[pd.DataFrame] = []
    for name, path in candidate_paths.items():
        summary = stage30.paired_block_bootstrap(
            stage20_path.loc[FULL_START:RESEARCH_END, "return"],
            path.loc[FULL_START:RESEARCH_END, "return"],
        )
        summary.insert(0, "Candidate", name)
        bootstrap_rows.append(summary)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)

    reproduction = candidate_paths["Stage34_NoChangeReproduction"]
    common = stage20_path.index.intersection(reproduction.index)
    max_return_error = float(
        (stage20_path.loc[common, "return"] - reproduction.loc[common, "return"])
        .abs()
        .max()
    )
    weight_columns = [f"w_{asset}" for asset in ASSETS]
    max_weight_error = float(
        (
            stage20_path.loc[common, weight_columns]
            - reproduction.loc[common, weight_columns]
        )
        .abs()
        .to_numpy()
        .max()
    )
    source_after = source_manifest()
    frozen_after = frozen_manifest()
    checks = {
        "source_files_unchanged": source_before == source_after,
        "stage20_to_stage33_files_unchanged": frozen_before == frozen_after,
        "futures_signal_precedes_target_month": bool(
            (calibrated_signals["futures_signal_month"] < calibrated_signals.index).all()
        ),
        "basis_and_oi_20d_same_contract_only": bool(
            daily.loc[
                daily["basis_change_20d"].notna()
                | daily["oi_log_change_20d"].notna(),
                "same_contract_20d",
            ].eq(1.0).all()
        ),
        "provider_unsigned_gap_not_used_as_direction_signal": True,
        "no_rsi_macd_breakout_or_horizon_search": True,
        "fixed_20d_signal_and_60m_calibration": bool(
            SIGNAL_DAYS == 20 and MIN_CAUSAL_MONTHS == 60
        ),
        "no_leverage_long_only_sum_to_one": bool(
            all(
                np.allclose(path[weight_columns].sum(axis=1), 1.0)
                and (path[weight_columns] >= -1e-10).all().all()
                and (path[weight_columns] <= 1.0 + 1e-10).all().all()
                for path in candidate_paths.values()
            )
        ),
        "no_change_mode_reproduces_stage20_returns": bool(
            max_return_error < 5e-7
        ),
        "no_change_mode_reproduces_stage20_weights": bool(
            max_weight_error < 5e-6
        ),
        "all_candidate_solvers_feasible": bool(
            all(
                path["solver_success"].all()
                and not path["used_fallback"].any()
                and path["volatility_slack"].min() >= -1e-7
                and path["cdar_slack"].min() >= -1e-7
                for path in candidate_paths.values()
            )
        ),
    }
    return_audit_summary = {
        "rows": int(len(return_audit)),
        "first_month": str(return_audit.index.min()),
        "last_month": str(return_audit.index.max()),
        "columns": list(return_audit.columns),
        "missing_by_column": {
            column: int(return_audit[column].isna().sum())
            for column in return_audit.columns
        },
    }
    macro_audit_summary = {
        "rows": int(len(macro_audit)),
        "first_date": str(macro_audit.index.min()),
        "last_date": str(macro_audit.index.max()),
        "columns": list(macro_audit.columns),
        "missing_by_column": {
            column: int(macro_audit[column].isna().sum())
            for column in macro_audit.columns
        },
    }
    report = {
        "study": "Stage34_KOSPI200FuturesBasisOIConfirmation",
        "decision": gates["decision"],
        "promoted_strategy": gates["promoted_strategy"],
        "scope": (
            "Stage20 is frozen. One signed close-to-settlement-theory basis, "
            "one 20-day OI confirmation, and one futures-stress confirmation "
            "mapping are evaluated without parameter search."
        ),
        "fixed_design": {
            "primary_direction_signal": (
                "20-day same-contract change in near close/settlement theory - 1"
            ),
            "risk_signal": "negative signed close-to-theory basis",
            "oi_confirmation": "20-day same-contract log OI change",
            "return_horizons_months": list(HORIZONS),
            "controls": list(CONTROL_COLUMNS),
            "causal_calibration_months": MIN_CAUSAL_MONTHS,
            "significance_level": SIGNIFICANCE_LEVEL,
            "strategy_mappings": {
                "basis_alpha": "causal non-negative slope times basis z-score",
                "basis_oi_alpha": (
                    "causal two-factor NNLS on basis and basis×OI"
                ),
                "risk_confirmation": (
                    "KODEX200 VIX6 stress mu adjustment times 2×causal "
                    "futures-dislocation percentile"
                ),
            },
            "searched_parameters": None,
        },
        "data_audit": data_audit,
        "return_audit": return_audit_summary,
        "macro_audit": macro_audit_summary,
        "market_audit": market_audit,
        "gate_results": gates,
        "return_predictive_regressions": json.loads(
            return_tests.to_json(orient="records", force_ascii=False)
        ),
        "risk_predictive_regressions": json.loads(
            risk_tests.to_json(orient="records", force_ascii=False)
        ),
        "false_positive_regression": json.loads(
            false_positive.to_json(orient="records", force_ascii=False)
        ),
        "confirmation_state_table": json.loads(
            state_table.to_json(orient="records", force_ascii=False)
        ),
        "performance_comparison": json.loads(
            performance.to_json(orient="records", force_ascii=False)
        ),
        "bootstrap_vs_stage20": json.loads(
            bootstrap.to_json(orient="records", force_ascii=False)
        ),
        "reproduction_audit": {
            "months": int(len(common)),
            "max_absolute_return_error": max_return_error,
            "max_absolute_weight_error": max_weight_error,
        },
        "solver_audit": {
            name: solver_summary(path) for name, path in candidate_paths.items()
        },
        "checks": checks,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(OUTPUT_DIR / "normalized_k200_futures_daily.csv")
        calibrated_signals.to_csv(
            OUTPUT_DIR / "monthly_futures_basis_oi_signals.csv"
        )
        research.to_csv(OUTPUT_DIR / "monthly_stage34_research_frame.csv")
        return_tests.to_csv(
            OUTPUT_DIR / "return_predictive_regressions.csv", index=False
        )
        risk_tests.to_csv(
            OUTPUT_DIR / "risk_predictive_regressions.csv", index=False
        )
        false_positive.to_csv(
            OUTPUT_DIR / "false_positive_regression.csv", index=False
        )
        state_table.to_csv(
            OUTPUT_DIR / "futures_confirmation_state_table.csv", index=False
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage20.csv", index=False
        )
        for name, path in candidate_paths.items():
            path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        _plot_factor_mechanism(
            return_tests,
            risk_tests,
            OUTPUT_DIR / "basis_oi_mechanism.png",
        )
        _plot_performance(
            performance, OUTPUT_DIR / "performance_comparison.png"
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
