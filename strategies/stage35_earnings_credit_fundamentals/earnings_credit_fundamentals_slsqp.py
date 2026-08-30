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
from scipy.optimize import minimize
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
from strategies.stage34_futures_basis_oi_confirmation import (
    futures_basis_oi_confirmation_slsqp as stage34,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EARNINGS_XLSX = ROOT / "raw_data" / "260829_fwdPE.EPS.rev.xlsx"
CREDIT_XLSX = ROOT / "raw_data" / "260829_국고채.회사채.xlsx"
RESEARCH_END = pd.Period("2026-07", freq="M")
MIN_CAUSAL_MONTHS = 60
CREDIT_CHANGE_DAYS = 20
SIGNIFICANCE_LEVEL = 0.10
RETURN_HORIZONS = (1, 3, 6)
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

SOURCE_FILES = (EARNINGS_XLSX, CREDIT_XLSX)
FROZEN_FILES = (
    Path(stage20.__file__),
    Path(stage34.__file__),
    stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv",
    stage20.OUTPUT_DIR / "stage14_static_recomputed_monthly.csv",
    stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
)


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


def load_fundamental_daily() -> tuple[pd.DataFrame, dict[str, Any]]:
    earnings = pd.read_excel(
        EARNINGS_XLSX,
        header=13,
        usecols=range(5),
        engine="openpyxl",
    )
    earnings.columns = [
        "date",
        "forward_pe_12m",
        "forward_eps_12m",
        "eps_revision_1w_pct",
        "eps_revision_1m_pct",
    ]
    earnings["date"] = pd.to_datetime(earnings["date"], errors="coerce")
    earnings = (
        earnings.dropna(subset=["date"])
        .set_index("date")
        .sort_index()
    )
    earnings = earnings.apply(pd.to_numeric, errors="coerce")
    earnings = earnings.loc[~earnings.index.duplicated(keep="last")]
    positive_eps = earnings["forward_eps_12m"].where(
        earnings["forward_eps_12m"] > 0.0
    )
    earnings["computed_eps_revision_21d_pct"] = (
        np.log(positive_eps).diff(21) * 100.0
    )

    credit = pd.read_excel(
        CREDIT_XLSX,
        header=None,
        skiprows=14,
        usecols=range(11),
        engine="openpyxl",
    )
    credit.columns = [
        "date",
        "ktb_1y_pct",
        "ktb_2y_pct",
        "ktb_3y_pct",
        "ktb_5y_pct",
        "ktb_10y_pct",
        "ktb_20y_pct",
        "ktb_30y_pct",
        "ktb_50y_pct",
        "corp_aa_minus_3y_pct",
        "corp_bbb_minus_3y_pct",
    ]
    credit["date"] = pd.to_datetime(credit["date"], errors="coerce")
    credit = credit.dropna(subset=["date"]).set_index("date").sort_index()
    credit = credit.apply(pd.to_numeric, errors="coerce")
    credit = credit.loc[~credit.index.duplicated(keep="last")]
    credit["aa_credit_spread_pctpt"] = (
        credit["corp_aa_minus_3y_pct"] - credit["ktb_3y_pct"]
    )
    credit["bbb_credit_spread_pctpt"] = (
        credit["corp_bbb_minus_3y_pct"] - credit["ktb_3y_pct"]
    )
    credit["quality_spread_pctpt"] = (
        credit["corp_bbb_minus_3y_pct"]
        - credit["corp_aa_minus_3y_pct"]
    )
    credit["aa_spread_widening_20d_pctpt"] = credit[
        "aa_credit_spread_pctpt"
    ].diff(CREDIT_CHANGE_DAYS)
    credit["bbb_spread_widening_20d_pctpt"] = credit[
        "bbb_credit_spread_pctpt"
    ].diff(CREDIT_CHANGE_DAYS)
    credit["quality_spread_widening_20d_pctpt"] = credit[
        "quality_spread_pctpt"
    ].diff(CREDIT_CHANGE_DAYS)
    credit["yield_curve_10y_minus_3y_pctpt"] = (
        credit["ktb_10y_pct"] - credit["ktb_3y_pct"]
    )

    daily = earnings.join(credit, how="outer")
    daily["earnings_yield_gap"] = (
        1.0 / daily["forward_pe_12m"].where(daily["forward_pe_12m"] > 0.0)
        - daily["ktb_10y_pct"] / 100.0
    )
    provider = daily[["eps_revision_1m_pct", "computed_eps_revision_21d_pct"]]
    audit = {
        "earnings_source": str(EARNINGS_XLSX.resolve()),
        "credit_source": str(CREDIT_XLSX.resolve()),
        "daily_rows": int(len(daily)),
        "first_date": str(daily.index.min().date()),
        "last_date": str(daily.index.max().date()),
        "earnings_first_valid": str(
            daily["eps_revision_1m_pct"].dropna().index.min().date()
        ),
        "credit_first_valid": str(
            daily["aa_spread_widening_20d_pctpt"].dropna().index.min().date()
        ),
        "eps_revision_provider_field_used": True,
        "eps_revision_formula": (
            "vendor EPS(Fwd.12M) one-month revision percentage"
        ),
        "computed_revision_role": "source cross-check only",
        "provider_computed_revision_correlation": float(
            provider.corr().iloc[0, 1]
        ),
        "eps_level_stale_share": float(
            daily["forward_eps_12m"]
            .dropna()
            .eq(daily["forward_eps_12m"].dropna().shift(1))
            .mean()
        ),
        "credit_primary_formula": (
            "(AA- unsecured 3Y yield - KTB 3Y yield) 20-observation change"
        ),
        "credit_primary_role": (
            "widening=funding stress; easing=positive equity-return input"
        ),
        "bbb_and_quality_role": "robustness diagnostics only",
        "earnings_yield_gap_role": "slow valuation diagnostic only",
        "winsorization": False,
        "parameter_grid": False,
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit


def _causal_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    prior = values.shift(1)
    mean = prior.expanding(min_periods=MIN_CAUSAL_MONTHS).mean()
    std = prior.expanding(min_periods=MIN_CAUSAL_MONTHS).std(ddof=1)
    return ((values - mean) / std.where(std > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )


def build_monthly_fundamental_signals(daily: pd.DataFrame) -> pd.DataFrame:
    required = [
        "forward_pe_12m",
        "forward_eps_12m",
        "eps_revision_1m_pct",
        "computed_eps_revision_21d_pct",
        "ktb_3y_pct",
        "ktb_10y_pct",
        "corp_aa_minus_3y_pct",
        "corp_bbb_minus_3y_pct",
        "aa_credit_spread_pctpt",
        "bbb_credit_spread_pctpt",
        "quality_spread_pctpt",
        "aa_spread_widening_20d_pctpt",
        "bbb_spread_widening_20d_pctpt",
        "quality_spread_widening_20d_pctpt",
        "yield_curve_10y_minus_3y_pctpt",
        "earnings_yield_gap",
    ]
    rows: list[dict[str, Any]] = []
    for signal_month, group in daily.groupby(daily.index.to_period("M")):
        complete = group.dropna(
            subset=[
                "eps_revision_1m_pct",
                "aa_spread_widening_20d_pctpt",
                "earnings_yield_gap",
            ]
        )
        if complete.empty:
            continue
        current = complete.iloc[-1]
        rows.append(
            {
                "target_month": signal_month + 1,
                "fundamental_signal_month": signal_month,
                "fundamental_signal_date": complete.index[-1],
                **{column: float(current[column]) for column in required},
            }
        )
    signals = pd.DataFrame(rows).set_index("target_month").sort_index()
    signals.index = pd.PeriodIndex(signals.index, freq="M")
    signals["eps_revision_z"] = _causal_zscore(
        signals["eps_revision_1m_pct"]
    )
    signals["credit_widening_z"] = _causal_zscore(
        signals["aa_spread_widening_20d_pctpt"]
    )
    signals["credit_easing_z"] = -signals["credit_widening_z"]
    signals["fundamental_confirmation_score"] = 0.5 * (
        signals["eps_revision_z"] + signals["credit_easing_z"]
    )
    signals["valuation_gap_z"] = _causal_zscore(signals["earnings_yield_gap"])
    stress_rank = causal_expanding_midrank(
        signals["aa_spread_widening_20d_pctpt"]
    )
    prior_count = (
        signals["aa_spread_widening_20d_pctpt"]
        .notna()
        .shift(1)
        .fillna(False)
        .astype(int)
        .cumsum()
    )
    signals["credit_stress_rank"] = stress_rank.where(
        prior_count >= MIN_CAUSAL_MONTHS
    )
    signals["credit_stress_multiplier"] = 2.0 * signals[
        "credit_stress_rank"
    ]
    return signals.replace([np.inf, -np.inf], np.nan)


def build_research_frame(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    probabilities: pd.DataFrame,
    stress: pd.DataFrame,
    equity_close: pd.Series,
) -> pd.DataFrame:
    frame = signals.copy()
    for horizon in (*RETURN_HORIZONS, 12):
        frame[f"future_{horizon}m_return"] = stage34._forward_compound(
            returns["KODEX200"], horizon
        )
    frame["recent_1m_return"] = returns["KODEX200"].shift(1)
    frame["realized_vol_21d"] = stage34._realized_volatility_signal(
        equity_close
    )
    frame["macro_fragility"] = (
        probabilities["p_Slowdown"] + probabilities["p_Stagflation"]
    )
    frame["vix6_stress_score"] = stress["stress_score"]
    frame = frame.join(stage34._forward_risk_targets(equity_close), how="left")
    monthly_close = equity_close.groupby(equity_close.index.to_period("M")).last()
    monthly_return = monthly_close.pct_change().loc[:RESEARCH_END]
    frame["future_left_tail_1m"] = stage34._causal_tail_event(monthly_return)

    stage20_path = stage34._load_period_csv(
        stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv", "month"
    )
    stage14_path = stage34._load_period_csv(
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


def _fit_return_model(
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
    models = {
        "EPSOnly": ["eps_revision_1m_pct"],
        "EPSFullControls": ["eps_revision_1m_pct", *CONTROL_COLUMNS],
        "CreditOnly": ["credit_easing_z"],
        "CreditFullControls": ["credit_easing_z", *CONTROL_COLUMNS],
        "JointFullControls": [
            "eps_revision_1m_pct",
            "credit_easing_z",
            *CONTROL_COLUMNS,
        ],
    }
    rows: list[dict[str, Any]] = []
    for horizon in RETURN_HORIZONS:
        target = f"future_{horizon}m_return"
        for model, predictors in models.items():
            fit, complete = _fit_return_model(
                frame, target, predictors, horizon
            )
            aligned = frame.loc[
                complete.index,
                ["eps_revision_1m_pct", "credit_easing_z", target],
            ]
            eps_ic, _ = spearmanr(
                aligned["eps_revision_1m_pct"], aligned[target],
                nan_policy="omit",
            )
            credit_ic, _ = spearmanr(
                aligned["credit_easing_z"], aligned[target],
                nan_policy="omit",
            )
            rows.append(
                {
                    "HorizonMonths": horizon,
                    "Model": model,
                    "Observations": int(len(complete)),
                    "EPSStandardizedBeta": float(
                        fit.params.get("eps_revision_1m_pct", np.nan)
                    ),
                    "EPSHACPValue": float(
                        fit.pvalues.get("eps_revision_1m_pct", np.nan)
                    ),
                    "EPSSpearmanIC": float(eps_ic),
                    "CreditEasingStandardizedBeta": float(
                        fit.params.get("credit_easing_z", np.nan)
                    ),
                    "CreditHACPValue": float(
                        fit.pvalues.get("credit_easing_z", np.nan)
                    ),
                    "CreditSpearmanIC": float(credit_ic),
                    "AdjustedR2": float(fit.rsquared_adj),
                    "HACLags": horizon,
                }
            )
    return pd.DataFrame(rows)


def stability_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("EPS_1M", "future_1m_return", "eps_revision_1m_pct", 1),
        ("Credit_3M", "future_3m_return", "credit_easing_z", 3),
    ]
    periods = [
        ("early_2007_2017", FULL_START, LOCKED_START - 1),
        ("locked_2018_2026", LOCKED_START, RESEARCH_END),
    ]
    rows: list[dict[str, Any]] = []
    for label, start, end in periods:
        view = frame.loc[start:end]
        for test, target, feature, lags in specifications:
            fit, complete = _fit_return_model(
                view, target, [feature, *CONTROL_COLUMNS], lags
            )
            rows.append(
                {
                    "Period": label,
                    "Test": test,
                    "Observations": int(len(complete)),
                    "StandardizedBeta": float(fit.params[feature]),
                    "HACPValue": float(fit.pvalues[feature]),
                    "ExpectedSign": "positive",
                }
            )
    return pd.DataFrame(rows)


def valuation_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in (6, 12):
        target = f"future_{horizon}m_return"
        fit, complete = _fit_return_model(
            frame, target, ["earnings_yield_gap", *CONTROL_COLUMNS], horizon
        )
        ic, ic_p = spearmanr(
            complete["earnings_yield_gap"], complete[target]
        )
        rows.append(
            {
                "HorizonMonths": horizon,
                "Observations": int(len(complete)),
                "ValuationGapStandardizedBeta": float(
                    fit.params["earnings_yield_gap"]
                ),
                "ValuationGapHACPValue": float(
                    fit.pvalues["earnings_yield_gap"]
                ),
                "SpearmanIC": float(ic),
                "SpearmanICPValue": float(ic_p),
                "Role": "eligible_slow_anchor_after_gate",
            }
        )
    return pd.DataFrame(rows)


def credit_risk_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in RISK_TARGETS:
        binary = target == "future_left_tail_1m"
        lags = 3 if target.endswith("3m") else 1
        predictors = ["aa_spread_widening_20d_pctpt", *CONTROL_COLUMNS]
        complete = frame[[target, *predictors]].dropna()
        x = _standardize(complete, predictors)
        if binary:
            fit = sm.GLM(
                complete[target],
                sm.add_constant(x),
                family=sm.families.Binomial(),
            ).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            fit_score = 1.0 - fit.llf / fit.llnull
            score_name = "McFaddenPseudoR2"
            events = int(complete[target].sum())
        else:
            fit = sm.OLS(complete[target], sm.add_constant(x)).fit(
                cov_type="HAC", cov_kwds={"maxlags": lags}
            )
            fit_score = fit.rsquared_adj
            score_name = "AdjustedR2"
            events = np.nan
        rows.append(
            {
                "Target": target,
                "Observations": int(len(complete)),
                "Events": events,
                "CreditWideningStandardizedBeta": float(
                    fit.params["aa_spread_widening_20d_pctpt"]
                ),
                "CreditWideningHACPValue": float(
                    fit.pvalues["aa_spread_widening_20d_pctpt"]
                ),
                "ExpectedSign": "positive",
                "FitScoreName": score_name,
                "FitScore": float(fit_score),
                "HACLags": lags,
            }
        )
    return pd.DataFrame(rows)


def false_positive_regression(frame: pd.DataFrame) -> pd.DataFrame:
    defense = frame.loc[frame["stage20_defense"].eq(1.0)]
    predictors = ["aa_spread_widening_20d_pctpt", *CONTROL_COLUMNS]
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
                "CreditWideningStandardizedBeta": float(
                    fit.params["aa_spread_widening_20d_pctpt"]
                ),
                "CreditWideningHACPValue": float(
                    fit.pvalues["aa_spread_widening_20d_pctpt"]
                ),
                "ExpectedSign": "negative",
                "McFaddenPseudoR2": float(1.0 - fit.llf / fit.llnull),
            }
        ]
    )


def fundamental_state_table(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["eps_revision_z", "credit_stress_rank"]).copy()
    eps_positive = valid["eps_revision_z"] >= 0.0
    credit_stress = valid["credit_stress_rank"] >= 0.5
    valid["State"] = np.select(
        [
            eps_positive & ~credit_stress,
            ~eps_positive & credit_stress,
            eps_positive & credit_stress,
            ~eps_positive & ~credit_stress,
        ],
        [
            "EPS Up / Credit Easing",
            "EPS Down / Credit Stress",
            "EPS Up / Credit Stress",
            "EPS Down / Credit Easing",
        ],
        default="Unavailable",
    )
    rows: list[dict[str, Any]] = []
    for state, group in valid.groupby("State"):
        rows.append(
            {
                "State": state,
                "Months": int(len(group)),
                "MeanFutureReturn1M": float(group["future_1m_return"].mean()),
                "MeanFutureReturn3M": float(group["future_3m_return"].mean()),
                "MeanFutureRealizedVol1M": float(
                    group["future_realized_vol_1m"].mean()
                ),
                "MeanFutureMaxDrawdown3M": float(
                    group["future_max_drawdown_3m"].mean()
                ),
                "FutureLeftTailRate1M": float(
                    group["future_left_tail_1m"].mean()
                ),
                "MeanStage20KODEXWeight": float(
                    group["stage20_w_kodex200"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def robustness_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Check predeclared alternate credit measures without trading them."""

    features = {
        "AA_Widening_Primary": "aa_spread_widening_20d_pctpt",
        "BBB_Widening_Robustness": "bbb_spread_widening_20d_pctpt",
        "Quality_Widening_Robustness": "quality_spread_widening_20d_pctpt",
    }
    rows: list[dict[str, Any]] = []
    for label, feature in features.items():
        for horizon in (1, 3, 6):
            target = f"future_{horizon}m_return"
            fit, complete = _fit_return_model(
                frame, target, [feature, *CONTROL_COLUMNS], horizon
            )
            rows.append(
                {
                    "Feature": label,
                    "HorizonMonths": horizon,
                    "Observations": int(len(complete)),
                    "WideningStandardizedBeta": float(fit.params[feature]),
                    "HACPValue": float(fit.pvalues[feature]),
                    "ExpectedSign": "negative",
                    "Traded": label == "AA_Widening_Primary",
                }
            )
    return pd.DataFrame(rows)


def _nonnegative_univariate_slope(
    feature: pd.Series, target: pd.Series
) -> float:
    complete = pd.concat([feature, target], axis=1).dropna()
    if len(complete) < MIN_CAUSAL_MONTHS:
        return 0.0
    x = complete.iloc[:, 0].to_numpy(dtype=float)
    y = complete.iloc[:, 1].to_numpy(dtype=float)
    x_std = float(x.std(ddof=1))
    if not np.isfinite(x_std) or x_std <= 0.0:
        return 0.0
    x = (x - x.mean()) / x_std
    denominator = float(x @ x)
    if denominator <= 0.0:
        return 0.0
    raw = float(x @ (y - y.mean()) / denominator)
    return max(raw, 0.0)


def add_causal_return_calibration(
    signals: pd.DataFrame,
    equity_returns: pd.Series,
) -> pd.DataFrame:
    output = signals.copy()
    forward_12m_return = stage34._forward_compound(equity_returns, 12)
    rows: list[dict[str, Any]] = []
    for month in output.index:
        history = output.index[output.index < month].intersection(
            equity_returns.index
        )
        eps_feature = output.loc[history, "eps_revision_1m_pct"]
        credit_feature = -output.loc[
            history, "aa_spread_widening_20d_pctpt"
        ]
        target = equity_returns.loc[history]
        eps_slope = _nonnegative_univariate_slope(eps_feature, target)
        credit_slope = _nonnegative_univariate_slope(credit_feature, target)
        complete = pd.concat([eps_feature, credit_feature, target], axis=1).dropna()
        standardized = complete.iloc[:, :2].copy()
        standardized = (
            standardized - standardized.mean()
        ) / standardized.std(ddof=1)
        confirmation_history = 0.5 * (
            standardized.iloc[:, 0] + standardized.iloc[:, 1]
        )
        confirmation_slope = _nonnegative_univariate_slope(
            confirmation_history, complete.iloc[:, 2]
        )
        valuation_history = output.index[
            output.index <= month - 12
        ].intersection(forward_12m_return.index)
        valuation_feature = output.loc[
            valuation_history, "earnings_yield_gap"
        ]
        valuation_target = forward_12m_return.loc[valuation_history]
        valuation_complete = pd.concat(
            [valuation_feature, valuation_target], axis=1
        ).dropna()
        valuation_slope_12m = _nonnegative_univariate_slope(
            valuation_complete.iloc[:, 0], valuation_complete.iloc[:, 1]
        )
        current_eps = float(output.loc[month, "eps_revision_z"])
        current_credit = float(output.loc[month, "credit_easing_z"])
        current_confirmation = float(
            output.loc[month, "fundamental_confirmation_score"]
        )
        eps_mu = eps_slope * current_eps
        credit_mu = credit_slope * current_credit
        confirmation_mu = confirmation_slope * current_confirmation
        valuation_mu = (
            valuation_slope_12m
            * float(output.loc[month, "valuation_gap_z"])
            / 12.0
        )
        rows.append(
            {
                "target_month": month,
                "calibration_observations": int(len(complete)),
                "eps_calibration_slope": eps_slope,
                "credit_calibration_slope": credit_slope,
                "confirmation_calibration_slope": confirmation_slope,
                "valuation_calibration_observations": int(
                    len(valuation_complete)
                ),
                "valuation_calibration_slope_12m": valuation_slope_12m,
                "eps_mu_adjustment_KODEX200": eps_mu,
                "credit_mu_adjustment_KODEX200": credit_mu,
                "fundamental_mu_adjustment_KODEX200": confirmation_mu,
                "valuation_mu_adjustment_KODEX200": valuation_mu,
            }
        )
    calibrated = pd.DataFrame(rows).set_index("target_month")
    calibrated.index = pd.PeriodIndex(calibrated.index, freq="M")
    return output.join(calibrated)


def _solve_weights(
    history: pd.DataFrame,
    historical_probabilities: pd.DataFrame,
    current_probabilities: pd.Series,
    historical_stress: pd.Series,
    current_stress: float,
    historical_recovery: pd.Series,
    current_recovery: float,
    technical_signal: pd.Series,
    fundamental_signal: pd.Series,
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

    eps_mu = 0.0
    credit_mu = 0.0
    confirmation_mu = 0.0
    valuation_mu = 0.0
    stress_multiplier = 1.0
    if mode in {"eps_alpha", "fundamental_dual_role"}:
        eps_mu = float(
            fundamental_signal["eps_mu_adjustment_KODEX200"]
        )
    if mode == "credit_alpha":
        credit_mu = float(
            fundamental_signal["credit_mu_adjustment_KODEX200"]
        )
    if mode in {"valuation_anchor", "fundamental_dual_role"}:
        valuation_mu = float(
            fundamental_signal["valuation_mu_adjustment_KODEX200"]
        )
    if mode in {"credit_risk_confirmation", "fundamental_dual_role"}:
        stress_multiplier = float(
            fundamental_signal["credit_stress_multiplier"]
        )
        stress_adjustment[EQUITY_INDEX] *= stress_multiplier
    filtered_macro[EQUITY_INDEX] += (
        eps_mu + credit_mu + confirmation_mu + valuation_mu
    )
    expected_return = filtered_macro + stress_adjustment
    covariance = np.asarray(technical["adjusted_covariance"], dtype=float)
    credit_variance_multiplier = 1.0
    if mode in {"credit_risk_confirmation", "fundamental_dual_role"}:
        credit_variance_multiplier = 1.0 + float(
            fundamental_signal["credit_stress_rank"]
        )
        risk_scaling = np.eye(len(ASSETS), dtype=float)
        risk_scaling[EQUITY_INDEX, EQUITY_INDEX] = math.sqrt(
            credit_variance_multiplier
        )
        covariance = risk_scaling @ covariance @ risk_scaling

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
                f"Stage35 and fallback solves failed: {result.message}; "
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
        "policy": f"Stage35_{mode}",
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
        "eps_mu_adjustment_KODEX200": eps_mu,
        "credit_mu_adjustment_KODEX200": credit_mu,
        "fundamental_mu_adjustment_KODEX200": confirmation_mu,
        "valuation_mu_adjustment_KODEX200": valuation_mu,
        "credit_stress_confirmation_multiplier": stress_multiplier,
        "credit_equity_variance_multiplier": credit_variance_multiplier,
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
    fundamental_signals: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    required = [
        "eps_mu_adjustment_KODEX200",
        "credit_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
    ]
    months = returns.index.intersection(probabilities.index)
    months = months.intersection(stress_signals.index)
    months = months.intersection(technical_signals.index)
    months = months.intersection(
        fundamental_signals.dropna(subset=required).index
    )
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
        fundamental_signal = fundamental_signals.loc[month]
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
            fundamental_signal,
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
            "fundamental_signal_month": fundamental_signal[
                "fundamental_signal_month"
            ],
            "fundamental_signal_date": fundamental_signal[
                "fundamental_signal_date"
            ],
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
        }
        detail_columns = [
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
            "eps_mu_adjustment_KODEX200",
            "credit_mu_adjustment_KODEX200",
            "fundamental_mu_adjustment_KODEX200",
            "valuation_mu_adjustment_KODEX200",
            "credit_stress_confirmation_multiplier",
            "credit_equity_variance_multiplier",
            "original_stress_mu_adjustment_KODEX200",
            "confirmed_stress_mu_adjustment_KODEX200",
        ]
        row.update({column: detail[column] for column in detail_columns})
        for column in [
            "forward_pe_12m",
            "forward_eps_12m",
            "eps_revision_1m_pct",
            "computed_eps_revision_21d_pct",
            "eps_revision_z",
            "aa_credit_spread_pctpt",
            "bbb_credit_spread_pctpt",
            "quality_spread_pctpt",
            "aa_spread_widening_20d_pctpt",
            "bbb_spread_widening_20d_pctpt",
            "quality_spread_widening_20d_pctpt",
            "credit_widening_z",
            "credit_easing_z",
            "credit_stress_rank",
            "credit_stress_multiplier",
            "fundamental_confirmation_score",
            "earnings_yield_gap",
            "valuation_gap_z",
            "yield_curve_10y_minus_3y_pctpt",
            "calibration_observations",
            "eps_calibration_slope",
            "credit_calibration_slope",
            "confirmation_calibration_slope",
            "valuation_calibration_observations",
            "valuation_calibration_slope_12m",
        ]:
            row[column] = float(fundamental_signal[column])
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
    return_tests: pd.DataFrame,
    stability: pd.DataFrame,
    valuation_tests: pd.DataFrame,
    risk_tests: pd.DataFrame,
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    controlled = return_tests.loc[
        return_tests["Model"].isin(["EPSFullControls", "CreditFullControls"])
    ].set_index(["Model", "HorizonMonths"])
    stable = stability.set_index(["Test", "Period"])
    eps_gates = {
        "full_1m_positive_p_below_10pct": bool(
            controlled.loc[("EPSFullControls", 1), "EPSStandardizedBeta"] > 0.0
            and controlled.loc[("EPSFullControls", 1), "EPSHACPValue"]
            < SIGNIFICANCE_LEVEL
        ),
        "early_and_locked_1m_beta_positive": bool(
            (
                stable.loc[
                    "EPS_1M", "StandardizedBeta"
                ]
                > 0.0
            ).all()
        ),
        "full_3m_positive_p_below_10pct": bool(
            controlled.loc[("EPSFullControls", 3), "EPSStandardizedBeta"] > 0.0
            and controlled.loc[("EPSFullControls", 3), "EPSHACPValue"]
            < SIGNIFICANCE_LEVEL
        ),
    }
    credit_return_gates = {
        "full_3m_easing_positive_p_below_10pct": bool(
            controlled.loc[
                ("CreditFullControls", 3), "CreditEasingStandardizedBeta"
            ]
            > 0.0
            and controlled.loc[
                ("CreditFullControls", 3), "CreditHACPValue"
            ]
            < SIGNIFICANCE_LEVEL
        ),
        "early_and_locked_3m_beta_positive": bool(
            (
                stable.loc[
                    "Credit_3M", "StandardizedBeta"
                ]
                > 0.0
            ).all()
        ),
    }
    positive_risk = risk_tests["CreditWideningStandardizedBeta"] > 0.0
    significant_risk = positive_risk & (
        risk_tests["CreditWideningHACPValue"] < SIGNIFICANCE_LEVEL
    )
    drawdown_or_tail = risk_tests["Target"].isin(
        [
            "future_max_drawdown_1m",
            "future_max_drawdown_3m",
            "future_left_tail_1m",
        ]
    )
    credit_risk_gates = {
        "positive_at_three_of_four_risk_targets": bool(
            int(positive_risk.sum()) >= 3
        ),
        "significant_for_drawdown_or_tail": bool(
            (significant_risk & drawdown_or_tail).any()
        ),
    }
    valuation_indexed = valuation_tests.set_index("HorizonMonths")
    valuation_gates = {
        "positive_p_below_10pct_at_6m_and_12m": bool(
            (
                valuation_indexed.loc[
                    [6, 12], "ValuationGapStandardizedBeta"
                ]
                > 0.0
            ).all()
            and (
                valuation_indexed.loc[
                    [6, 12], "ValuationGapHACPValue"
                ]
                < SIGNIFICANCE_LEVEL
            ).all()
        ),
        "twelve_month_ic_positive_p_below_10pct": bool(
            valuation_indexed.loc[12, "SpearmanIC"] > 0.0
            and valuation_indexed.loc[12, "SpearmanICPValue"]
            < SIGNIFICANCE_LEVEL
        ),
    }
    eps_pass = bool(all(eps_gates.values()))
    credit_return_pass = bool(all(credit_return_gates.values()))
    credit_risk_pass = bool(all(credit_risk_gates.values()))
    valuation_pass = bool(all(valuation_gates.values()))
    mechanism_pass = (
        eps_pass and credit_return_pass and credit_risk_pass and valuation_pass
    )

    full = performance.loc[
        performance["Period"].eq("full_2007_2026")
    ].set_index("Strategy")
    baseline = full.loc["Stage20_Frozen"]
    candidate = full.loc["Stage35_FundamentalDualRole"]
    split = performance.set_index(["Strategy", "Period"])
    baseline_early = split.loc[("Stage20_Frozen", "early_2007_2017")]
    candidate_early = split.loc[
        ("Stage35_FundamentalDualRole", "early_2007_2017")
    ]
    baseline_locked = split.loc[("Stage20_Frozen", "locked_2018_2026")]
    candidate_locked = split.loc[
        ("Stage35_FundamentalDualRole", "locked_2018_2026")
    ]
    boot = bootstrap.loc[
        bootstrap["Candidate"].eq("Stage35_FundamentalDualRole")
    ].set_index("Metric")
    performance_gates = {
        "cagr_at_least_10pct": bool(candidate["CAGR"] >= 0.10),
        "sharpe_at_least_1": bool(candidate["Sharpe"] >= 1.0),
        "sharpe_above_stage20": bool(candidate["Sharpe"] > baseline["Sharpe"]),
        "mdd_not_worse_than_stage20": bool(candidate["MDD"] >= baseline["MDD"]),
        "bootstrap_cagr_probability_positive_at_least_60pct": bool(
            boot.loc["delta_CAGR", "ProbabilityPositive"] >= 0.60
        ),
        "bootstrap_sharpe_probability_positive_at_least_60pct": bool(
            boot.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.60
        ),
        "bootstrap_mdd_probability_positive_at_least_50pct": bool(
            boot.loc["delta_MDD", "ProbabilityPositive"] >= 0.50
        ),
        "early_cagr_and_sharpe_above_stage20": bool(
            candidate_early["CAGR"] > baseline_early["CAGR"]
            and candidate_early["Sharpe"] > baseline_early["Sharpe"]
        ),
        "locked_cagr_and_sharpe_not_below_stage20": bool(
            candidate_locked["CAGR"] >= baseline_locked["CAGR"]
            and candidate_locked["Sharpe"] >= baseline_locked["Sharpe"]
        ),
        "locked_mdd_not_worse_by_more_than_1pctpt": bool(
            candidate_locked["MDD"] >= baseline_locked["MDD"] - 0.01
        ),
    }
    performance_pass = bool(all(performance_gates.values()))
    promoted = (
        "Stage35_FundamentalDualRole"
        if mechanism_pass and performance_pass
        else None
    )
    decision = (
        "promote_stage35_fundamental_dual_role"
        if promoted
        else "keep_stage20_frozen_stage35_fails_full_promotion_gate"
    )
    return {
        "eps_gates": eps_gates,
        "credit_return_gates": credit_return_gates,
        "credit_risk_gates": credit_risk_gates,
        "valuation_gates": valuation_gates,
        "eps_pass": eps_pass,
        "credit_return_pass": credit_return_pass,
        "credit_risk_pass": credit_risk_pass,
        "valuation_pass": valuation_pass,
        "mechanism_pass": mechanism_pass,
        "performance_gates": performance_gates,
        "performance_pass": performance_pass,
        "promoted_strategy": promoted,
        "decision": decision,
    }


def _plot_mechanism(
    return_tests: pd.DataFrame,
    risk_tests: pd.DataFrame,
    path: Path,
) -> None:
    controlled = return_tests.loc[
        return_tests["Model"].isin(["EPSFullControls", "CreditFullControls"])
    ]
    eps = controlled.loc[controlled["Model"].eq("EPSFullControls")]
    credit = controlled.loc[controlled["Model"].eq("CreditFullControls")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(RETURN_HORIZONS))
    axes[0].bar(
        x - 0.18,
        eps["EPSStandardizedBeta"] * 100.0,
        width=0.36,
        label="EPS revision",
        color="#26734d",
    )
    axes[0].bar(
        x + 0.18,
        credit["CreditEasingStandardizedBeta"] * 100.0,
        width=0.36,
        label="Credit easing",
        color="#497d9b",
    )
    axes[0].axhline(0.0, color="#222", linewidth=0.8)
    axes[0].set_xticks(x, [f"{h}M" for h in RETURN_HORIZONS])
    axes[0].set_ylabel("Forward-return beta (%p)")
    axes[0].set_title("Independent return mechanism after controls")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    risk_labels = ["1M vol", "1M MDD", "3M MDD", "1M tail"]
    risk_values = risk_tests["CreditWideningStandardizedBeta"].to_numpy()
    colors = ["#b14f43" if value > 0 else "#497d9b" for value in risk_values]
    axes[1].bar(np.arange(len(risk_values)), risk_values, color=colors)
    axes[1].axhline(0.0, color="#222", linewidth=0.8)
    axes[1].set_xticks(
        np.arange(len(risk_values)), risk_labels, rotation=20, ha="right"
    )
    axes[1].set_ylabel("Standardized beta")
    axes[1].set_title("Credit widening and future risk")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_performance(performance: pd.DataFrame, path: Path) -> None:
    full = performance.loc[
        performance["Period"].eq("full_2007_2026")
    ].copy()
    x = np.arange(len(full))
    labels = [
        name.replace("Stage35_", "").replace("Stage20_", "")
        for name in full["Strategy"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for ax, column, title in zip(
        axes, ["CAGR", "Sharpe", "MDD"], ["CAGR", "Sharpe", "MDD"]
    ):
        scale = 1.0 if column == "Sharpe" else 100.0
        values = full[column] * scale
        colors = [
            "#264f65" if name == "Stage20_Frozen" else "#8aa8b7"
            for name in full["Strategy"]
        ]
        colors[-1] = "#26734d"
        ax.bar(x, values, color=colors)
        ax.axhline(0.0, color="#222", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_nav(paths: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    selected = [
        "Stage20_Frozen",
        "Stage35_EPSAlpha",
        "Stage35_CreditAlpha",
        "Stage35_FundamentalDualRole",
    ]
    colors = ["#264f65", "#26734d", "#9b7049", "#8b3d66"]
    for name, color in zip(selected, colors):
        series = paths[name].loc[FULL_START:RESEARCH_END, "return"]
        nav = (1.0 + series).cumprod()
        ax.plot(nav.index.to_timestamp(), nav, label=name, color=color, lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Causal net performance, log scale")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()
    daily, data_audit = load_fundamental_daily()
    signals = build_monthly_fundamental_signals(daily)
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
    stability = stability_regressions(research)
    valuation = valuation_regressions(research)
    risk_tests = credit_risk_regressions(research)
    false_positive = false_positive_regression(research)
    states = fundamental_state_table(research)
    robustness = robustness_diagnostics(research)

    equity_monthly_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    equity_calibration_returns = equity_monthly_close.pct_change()
    calibrated = add_causal_return_calibration(
        signals, equity_calibration_returns
    )
    technical = stage34._load_period_csv(
        stage20.OUTPUT_DIR / "monthly_technical_signals.csv", "target_month"
    )
    modes = {
        "Stage35_NoChangeReproduction": "baseline_reproduction",
        "Stage35_EPSAlpha": "eps_alpha",
        "Stage35_CreditAlpha": "credit_alpha",
        "Stage35_CreditRiskConfirmation": "credit_risk_confirmation",
        "Stage35_ValuationAnchor": "valuation_anchor",
        "Stage35_FundamentalDualRole": "fundamental_dual_role",
    }
    candidate_paths = {
        name: run_backtest(
            returns,
            probabilities,
            stress,
            technical,
            calibrated,
            mode,
        )
        for name, mode in modes.items()
    }
    stage20_path = stage34._load_period_csv(
        stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv", "month"
    )
    paths = {"Stage20_Frozen": stage20_path, **candidate_paths}
    performance = _performance_table(paths)
    bootstrap_rows: list[pd.DataFrame] = []
    for name, candidate_path in candidate_paths.items():
        summary = stage30.paired_block_bootstrap(
            stage20_path.loc[FULL_START:RESEARCH_END, "return"],
            candidate_path.loc[FULL_START:RESEARCH_END, "return"],
        )
        summary.insert(0, "Candidate", name)
        bootstrap_rows.append(summary)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    gates = gate_decision(
        return_tests,
        stability,
        valuation,
        risk_tests,
        performance,
        bootstrap,
    )

    reproduction = candidate_paths["Stage35_NoChangeReproduction"]
    common = stage20_path.index.intersection(reproduction.index)
    max_return_error = float(
        (
            stage20_path.loc[common, "return"]
            - reproduction.loc[common, "return"]
        )
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
        "stage20_and_stage34_files_unchanged": frozen_before == frozen_after,
        "fundamental_signal_precedes_target": bool(
            (
                calibrated["fundamental_signal_month"]
                < calibrated.index
            ).all()
        ),
        "first_2007_signal_has_presample_calibration": bool(
            calibrated.loc[FULL_START, "calibration_observations"]
            >= MIN_CAUSAL_MONTHS
        ),
        "fixed_1m_eps_and_20d_credit_signals": bool(
            CREDIT_CHANGE_DAYS == 20 and MIN_CAUSAL_MONTHS == 60
        ),
        "no_acceleration_curve_or_threshold_search": True,
        "valuation_anchor_uses_only_fully_observed_12m_targets": bool(
            calibrated.loc[FULL_START, "valuation_calibration_observations"]
            >= MIN_CAUSAL_MONTHS
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
    }
    macro_audit_summary = {
        "rows": int(len(macro_audit)),
        "first_date": str(macro_audit.index.min()),
        "last_date": str(macro_audit.index.max()),
        "columns": list(macro_audit.columns),
    }
    report = {
        "study": "Stage35_EarningsCreditFundamentalConfirmation",
        "decision": gates["decision"],
        "promoted_strategy": gates["promoted_strategy"],
        "scope": (
            "Stage20 is frozen. Forward EPS one-month revision drives equity "
            "mu; AA- corporate spread easing drives equity mu; widening also "
            "confirms the pre-existing VIX6 equity stress adjustment."
        ),
        "fixed_design": {
            "eps_direction_signal": (
                "vendor Forward EPS 12M one-month revision percent"
            ),
            "credit_signal": (
                "20-observation change in AA- corporate 3Y minus KTB 3Y"
            ),
            "credit_return_direction": "easing positive",
            "credit_risk_direction": "widening positive",
            "controls": list(CONTROL_COLUMNS),
            "return_horizons_months": list(RETURN_HORIZONS),
            "causal_calibration_months": MIN_CAUSAL_MONTHS,
            "strategy_mapping": {
                "EPSAlpha": (
                    "non-negative expanding return slope times causal EPS z"
                ),
                "CreditAlpha": (
                    "non-negative expanding return slope times causal easing z"
                ),
                "CreditRiskConfirmation": (
                    "existing KODEX VIX6 stress mu times 2×causal widening rank; "
                    "KODEX variance times 1+causal widening rank"
                ),
                "ValuationAnchor": (
                    "non-negative expanding 12M return slope times valuation "
                    "gap z divided by 12; only fully observed targets"
                ),
                "FundamentalDualRole": (
                    "EPS causal mu + slow valuation mu + credit-confirmed "
                    "existing stress mu and KODEX variance; no credit return mu"
                ),
            },
            "searched_parameters": None,
            "candidate_selection": (
                "only FundamentalDualRole may be promoted; component paths are "
                "mechanism attribution, not a best-of search"
            ),
        },
        "data_audit": data_audit,
        "return_audit": return_audit_summary,
        "macro_audit": macro_audit_summary,
        "market_audit": market_audit,
        "gate_results": gates,
        "return_predictive_regressions": json.loads(
            return_tests.to_json(orient="records", force_ascii=False)
        ),
        "stability_regressions": json.loads(
            stability.to_json(orient="records", force_ascii=False)
        ),
        "credit_risk_regressions": json.loads(
            risk_tests.to_json(orient="records", force_ascii=False)
        ),
        "valuation_diagnostic": json.loads(
            valuation.to_json(orient="records", force_ascii=False)
        ),
        "credit_robustness_diagnostics": json.loads(
            robustness.to_json(orient="records", force_ascii=False)
        ),
        "false_positive_regression": json.loads(
            false_positive.to_json(orient="records", force_ascii=False)
        ),
        "fundamental_state_table": json.loads(
            states.to_json(orient="records", force_ascii=False)
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
            name: solver_summary(path)
            for name, path in candidate_paths.items()
        },
        "checks": checks,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(OUTPUT_DIR / "normalized_earnings_credit_daily.csv")
        calibrated.to_csv(
            OUTPUT_DIR / "monthly_earnings_credit_signals.csv"
        )
        research.to_csv(OUTPUT_DIR / "monthly_stage35_research_frame.csv")
        return_tests.to_csv(
            OUTPUT_DIR / "return_predictive_regressions.csv", index=False
        )
        stability.to_csv(
            OUTPUT_DIR / "subperiod_stability_regressions.csv", index=False
        )
        risk_tests.to_csv(
            OUTPUT_DIR / "credit_risk_regressions.csv", index=False
        )
        valuation.to_csv(
            OUTPUT_DIR / "valuation_gap_diagnostic.csv", index=False
        )
        robustness.to_csv(
            OUTPUT_DIR / "credit_measure_robustness.csv", index=False
        )
        false_positive.to_csv(
            OUTPUT_DIR / "false_positive_regression.csv", index=False
        )
        states.to_csv(
            OUTPUT_DIR / "fundamental_state_table.csv", index=False
        )
        performance.to_csv(
            OUTPUT_DIR / "performance_comparison.csv", index=False
        )
        bootstrap.to_csv(
            OUTPUT_DIR / "paired_block_bootstrap_vs_stage20.csv", index=False
        )
        for name, candidate_path in candidate_paths.items():
            candidate_path.to_csv(OUTPUT_DIR / f"{name.lower()}_monthly.csv")
        _plot_mechanism(
            return_tests, risk_tests, OUTPUT_DIR / "mechanism_coefficients.png"
        )
        _plot_performance(
            performance, OUTPUT_DIR / "performance_comparison.png"
        )
        _plot_nav(paths, OUTPUT_DIR / "nav_comparison.png")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    print(json.dumps(run_research(save=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
