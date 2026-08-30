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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

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


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STUDY_START = pd.Period("2007-04", freq="M")
RESEARCH_END = pd.Period("2026-07", freq="M")
RETURN_HORIZONS = (1, 3, 6)
MIN_STATE_HISTORY = 60
MIN_TAIL_HISTORY = 60
TAIL_QUANTILE = 0.05
SIGNIFICANCE_LEVEL = 0.10

VIX_COLUMN = "vix6_stress_score"
WING_COLUMN = "wing_asym_near"
CONTROL_COLUMNS = (
    "recent_1m_return",
    "realized_vol_21d",
    "macro_fragility",
)
FROZEN_STRATEGY_FILES = (
    Path(stage20.__file__),
    Path(stage30.__file__),
    Path(stage31.__file__),
    Path(stage32.__file__),
)
FROZEN_RESULT_FILES = (
    stage20.OUTPUT_DIR / "daily_technical_confidence_monthly.csv",
    stage20.OUTPUT_DIR / "stage14_static_recomputed_monthly.csv",
    stage32.OUTPUT_DIR / "monthly_fear_premium_research_frame.csv",
    stage32.OUTPUT_DIR / "validation_report.json",
)
SOURCE_FILES = (
    stage31.LONG_IV_XLSX,
    stage31.K200_FUTURES_XLSX,
    stage20.OHLCV_CACHE,
    stage20.COMPASS_PATH,
    ROOT / "raw_data" / "KOSPI200OptionPrice.csv",
)

RISK_TARGETS = {
    "future_realized_vol_1m": "higher_is_worse",
    "future_max_drawdown_1m": "higher_is_worse",
    "future_max_drawdown_3m": "higher_is_worse",
    "future_left_tail_1m": "higher_is_worse",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "last_write_time_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def source_manifest() -> dict[str, dict[str, Any]]:
    return _file_manifest(SOURCE_FILES)


def frozen_manifest() -> dict[str, dict[str, Any]]:
    return _file_manifest(FROZEN_STRATEGY_FILES + FROZEN_RESULT_FILES)


def _load_period_csv(path: Path, index_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[index_column] = pd.PeriodIndex(frame[index_column], freq="M")
    return frame.set_index(index_column).sort_index()


def _forward_compound(series: pd.Series, horizon: int) -> pd.Series:
    legs = [series.shift(-offset) for offset in range(horizon)]
    frame = pd.concat(legs, axis=1)
    valid = frame.notna().all(axis=1)
    result = frame.add(1.0).prod(axis=1).sub(1.0).where(valid)
    return result


def _causal_expanding_quantile(
    series: pd.Series,
    quantile: float,
    min_history: int,
) -> pd.Series:
    values = series.astype(float)
    return values.shift(1).expanding(min_periods=min_history).quantile(quantile)


def _causal_expanding_median(series: pd.Series) -> pd.Series:
    return _causal_expanding_quantile(series, 0.5, MIN_STATE_HISTORY)


def _forward_daily_risk_targets(close: pd.Series) -> pd.DataFrame:
    close = close.dropna().sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    log_returns = np.log(close).diff()
    periods = pd.period_range(
        close.index.min().to_period("M"),
        RESEARCH_END,
        freq="M",
    )
    rows: list[dict[str, Any]] = []
    for period in periods:
        month_mask = close.index.to_period("M") == period
        month_log_returns = log_returns.loc[month_mask].dropna()
        future_vol = (
            float(month_log_returns.std(ddof=1) * math.sqrt(252.0))
            if len(month_log_returns) >= 15
            else np.nan
        )
        row: dict[str, Any] = {
            "target_month": period,
            "future_realized_vol_1m": future_vol,
        }
        for horizon, minimum_prices in ((1, 16), (3, 46)):
            final_period = period + horizon - 1
            start_date = period.start_time
            end_date = final_period.end_time
            before = close.loc[close.index < start_date]
            within = close.loc[
                (close.index >= start_date) & (close.index <= end_date)
            ]
            if before.empty or len(within) < minimum_prices - 1:
                max_drawdown = np.nan
            else:
                path = pd.concat([before.iloc[[-1]], within])
                wealth = path / float(path.iloc[0])
                drawdown = wealth / wealth.cummax() - 1.0
                max_drawdown = float(-drawdown.min())
            row[f"future_max_drawdown_{horizon}m"] = max_drawdown
        rows.append(row)
    output = pd.DataFrame(rows).set_index("target_month")
    output.index = pd.PeriodIndex(output.index, freq="M")
    return output


def build_research_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, stage32_audit = stage32.build_research_frame()
    market, market_audit = stage20.load_daily_asset_ohlcv()
    equity_close = market["KODEX200"]["close"].dropna()
    risk = _forward_daily_risk_targets(equity_close)
    frame = frame.join(risk, how="left")

    monthly_equity_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    monthly_equity_return = monthly_equity_close.pct_change().loc[:RESEARCH_END]
    tail_threshold = _causal_expanding_quantile(
        monthly_equity_return,
        TAIL_QUANTILE,
        MIN_TAIL_HISTORY,
    )
    frame["future_equity_return_1m"] = monthly_equity_return
    frame["causal_left_tail_threshold"] = tail_threshold
    frame["future_left_tail_1m"] = np.where(
        frame["future_equity_return_1m"].notna()
        & frame["causal_left_tail_threshold"].notna(),
        (
            frame["future_equity_return_1m"]
            <= frame["causal_left_tail_threshold"]
        ).astype(float),
        np.nan,
    )

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
    frame["stage20_lower_equity_than_stage14"] = np.where(
        frame["stage20_w_kodex200"].notna()
        & frame["stage14_w_kodex200"].notna(),
        (
            frame["stage20_w_kodex200"]
            < frame["stage14_w_kodex200"] - 1e-10
        ).astype(float),
        np.nan,
    )
    for horizon in RETURN_HORIZONS:
        stage20_forward = _forward_compound(stage20_path["return"], horizon)
        stage14_forward = _forward_compound(stage14_path["return"], horizon)
        frame[f"stage20_future_{horizon}m_return"] = stage20_forward
        frame[f"stage14_future_{horizon}m_return"] = stage14_forward
        frame[f"stage20_defense_opportunity_cost_{horizon}m"] = (
            stage14_forward - stage20_forward
        )
    frame["stage20_false_positive_1m"] = np.where(
        frame["stage20_lower_equity_than_stage14"].eq(1.0)
        & frame["stage20_defense_opportunity_cost_1m"].notna(),
        (frame["stage20_defense_opportunity_cost_1m"] > 0.0).astype(float),
        np.nan,
    )

    frame["causal_vix6_median"] = _causal_expanding_median(frame[VIX_COLUMN])
    frame["causal_wing_median"] = _causal_expanding_median(frame[WING_COLUMN])
    valid_state = (
        frame[VIX_COLUMN].notna()
        & frame[WING_COLUMN].notna()
        & frame["causal_vix6_median"].notna()
        & frame["causal_wing_median"].notna()
    )
    vix_high = frame[VIX_COLUMN] >= frame["causal_vix6_median"]
    wing_high = frame[WING_COLUMN] >= frame["causal_wing_median"]
    state = pd.Series(np.nan, index=frame.index, dtype=object)
    state.loc[valid_state & ~vix_high & ~wing_high] = "VIX6 Low / Wing Low"
    state.loc[valid_state & vix_high & ~wing_high] = "VIX6 High / Wing Low"
    state.loc[valid_state & ~vix_high & wing_high] = "VIX6 Low / Wing High"
    state.loc[valid_state & vix_high & wing_high] = "VIX6 High / Wing High"
    frame["causal_2x2_state"] = state
    frame = frame.loc[STUDY_START:RESEARCH_END].replace(
        [np.inf, -np.inf], np.nan
    )

    audit = {
        "stage32": stage32_audit,
        "daily_equity": market_audit["assets"]["KODEX200"],
        "risk_price_series": (
            "Stage20 KODEX200 daily close, extended backward with the same "
            "KOSPI200 proxy used by the strategy"
        ),
        "risk_target_sign_convention": (
            "realized volatility and maximum drawdown are positive loss/risk "
            "magnitudes; left-tail event is one"
        ),
        "left_tail_definition": (
            "target-month KODEX200 return at or below the 5th percentile of "
            "returns observed strictly before that month; minimum 60 months"
        ),
        "state_definition": (
            "high/low uses each signal's median estimated strictly from the "
            "prior 60 or more observations"
        ),
        "defense_definition": (
            "Stage20 KODEX200 weight below the saved Stage14 comparator weight"
        ),
        "opportunity_cost_definition": (
            "saved Stage14 forward compound return minus saved Stage20 "
            "forward compound return; positive means Stage20 defense cost"
        ),
        "parameter_search": False,
        "study_period": f"{STUDY_START} to {RESEARCH_END}",
        "residual_wingasym": False,
        "fear_term_slope": False,
    }
    return frame, audit


def _standardize_predictors(
    complete: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    output = pd.DataFrame(index=complete.index)
    for column in columns:
        std = float(complete[column].std(ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Predictor has no usable variation: {column}")
        output[column] = (complete[column] - complete[column].mean()) / std
    return output


def _fit_continuous_hac(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lags: int,
) -> tuple[Any, pd.DataFrame]:
    complete = frame[[target, *predictors]].dropna()
    x = _standardize_predictors(complete, predictors)
    fit = sm.OLS(
        complete[target].astype(float), sm.add_constant(x)
    ).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return fit, complete


def _fit_binary_glm(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    max_lags: int,
) -> tuple[Any, pd.DataFrame]:
    complete = frame[[target, *predictors]].dropna()
    x = _standardize_predictors(complete, predictors)
    fit = sm.GLM(
        complete[target].astype(float),
        sm.add_constant(x),
        family=sm.families.Binomial(),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return fit, complete


def incremental_risk_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = {
        "VIX6PlusWing": [VIX_COLUMN, WING_COLUMN],
        "FullControls": [VIX_COLUMN, WING_COLUMN, *CONTROL_COLUMNS],
    }
    for target in RISK_TARGETS:
        binary = target == "future_left_tail_1m"
        max_lags = 3 if target.endswith("3m") else 1
        for model_name, predictors in models.items():
            nested_predictors = [p for p in predictors if p != WING_COLUMN]
            columns = [target, *predictors]
            complete = frame[columns].dropna()
            if binary:
                fit, _ = _fit_binary_glm(
                    complete, target, predictors, max_lags
                )
                nested, _ = _fit_binary_glm(
                    complete, target, nested_predictors, max_lags
                )
                fit_score = 1.0 - fit.llf / fit.llnull
                nested_score = 1.0 - nested.llf / nested.llnull
                score_name = "McFaddenPseudoR2"
            else:
                fit, _ = _fit_continuous_hac(
                    complete, target, predictors, max_lags
                )
                nested, _ = _fit_continuous_hac(
                    complete, target, nested_predictors, max_lags
                )
                fit_score = fit.rsquared_adj
                nested_score = nested.rsquared_adj
                score_name = "AdjustedR2"
            rows.append(
                {
                    "Target": target,
                    "Model": model_name,
                    "ModelFamily": "BinomialLogit" if binary else "OLS",
                    "Observations": int(len(complete)),
                    "Events": (
                        int(complete[target].sum()) if binary else np.nan
                    ),
                    "WingStandardizedBeta": float(fit.params[WING_COLUMN]),
                    "WingHACStandardError": float(fit.bse[WING_COLUMN]),
                    "WingHACPValue": float(fit.pvalues[WING_COLUMN]),
                    "ExpectedDirectRiskSign": "positive",
                    "FitScoreName": score_name,
                    "FitScore": float(fit_score),
                    "NestedWithoutWingFitScore": float(nested_score),
                    "IncrementalFitScore": float(fit_score - nested_score),
                    "HACLags": max_lags,
                }
            )
    return pd.DataFrame(rows)


def expanding_tail_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = "future_left_tail_1m"
    models = {
        "HistoricalBaseRate": [],
        "VIX6Only": [VIX_COLUMN],
        "VIX6PlusWing": [VIX_COLUMN, WING_COLUMN],
        "FullControls": [VIX_COLUMN, WING_COLUMN, *CONTROL_COLUMNS],
    }
    required = [target, VIX_COLUMN, WING_COLUMN, *CONTROL_COLUMNS]
    complete = frame[required].dropna().sort_index()
    predictions: list[dict[str, Any]] = []
    for position in range(MIN_TAIL_HISTORY, len(complete)):
        train = complete.iloc[:position]
        test = complete.iloc[[position]]
        if train[target].nunique() < 2 or int(train[target].sum()) < 3:
            continue
        for model_name, predictors in models.items():
            if not predictors:
                probability = float(train[target].mean())
            else:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(train[predictors])
                x_test = scaler.transform(test[predictors])
                model = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                    random_state=33,
                )
                model.fit(x_train, train[target].astype(int))
                probability = float(model.predict_proba(x_test)[0, 1])
            predictions.append(
                {
                    "target_month": str(test.index[0]),
                    "Model": model_name,
                    "Actual": int(test[target].iloc[0]),
                    "Probability": float(np.clip(probability, 1e-8, 1 - 1e-8)),
                    "TrainingMonths": int(len(train)),
                    "TrainingEvents": int(train[target].sum()),
                }
            )
    detail = pd.DataFrame(predictions)
    metric_rows: list[dict[str, Any]] = []
    for model_name, group in detail.groupby("Model", sort=False):
        y = group["Actual"].to_numpy()
        probability = group["Probability"].to_numpy()
        metric_rows.append(
            {
                "Model": model_name,
                "OOSMonths": int(len(group)),
                "OOSEvents": int(y.sum()),
                "AUC": float(roc_auc_score(y, probability)),
                "BrierScore": float(brier_score_loss(y, probability)),
                "LogLoss": float(log_loss(y, probability, labels=[0, 1])),
                "FirstOOSMonth": str(group["target_month"].min()),
                "LastOOSMonth": str(group["target_month"].max()),
            }
        )
    return pd.DataFrame(metric_rows), detail


def _interaction_regression(
    frame: pd.DataFrame,
    target: str,
    controls: list[str],
    max_lags: int,
    binary: bool = False,
) -> dict[str, Any]:
    predictors = [VIX_COLUMN, WING_COLUMN, *controls]
    complete = frame[[target, *predictors]].dropna()
    x = _standardize_predictors(complete, predictors)
    x["vix6_x_wing"] = x[VIX_COLUMN] * x[WING_COLUMN]
    nested_x = x.drop(columns="vix6_x_wing")
    if binary:
        fit = sm.GLM(
            complete[target].astype(float),
            sm.add_constant(x),
            family=sm.families.Binomial(),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        nested = sm.GLM(
            complete[target].astype(float),
            sm.add_constant(nested_x),
            family=sm.families.Binomial(),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        fit_score = 1.0 - fit.llf / fit.llnull
        nested_score = 1.0 - nested.llf / nested.llnull
        score_name = "McFaddenPseudoR2"
    else:
        fit = sm.OLS(
            complete[target].astype(float), sm.add_constant(x)
        ).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        nested = sm.OLS(
            complete[target].astype(float), sm.add_constant(nested_x)
        ).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        fit_score = fit.rsquared_adj
        nested_score = nested.rsquared_adj
        score_name = "AdjustedR2"
    return {
        "Observations": int(len(complete)),
        "Events": int(complete[target].sum()) if binary else np.nan,
        "InteractionStandardizedBeta": float(fit.params["vix6_x_wing"]),
        "InteractionHACStandardError": float(fit.bse["vix6_x_wing"]),
        "InteractionHACPValue": float(fit.pvalues["vix6_x_wing"]),
        "FitScoreName": score_name,
        "FitScore": float(fit_score),
        "NestedWithoutInteractionFitScore": float(nested_score),
        "IncrementalFitScore": float(fit_score - nested_score),
        "HACLags": max_lags,
    }


def context_interaction_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_controls = {
        "BasicContext": [],
        "FullControlsContext": list(CONTROL_COLUMNS),
    }
    for horizon in RETURN_HORIZONS:
        target = f"future_{horizon}m_return"
        for model_name, controls in model_controls.items():
            result = _interaction_regression(
                frame, target, controls, max_lags=horizon
            )
            rows.append(
                {
                    "TestFamily": "K200ForwardReturn",
                    "Target": target,
                    "HorizonMonths": horizon,
                    "Model": model_name,
                    "ExpectedContextSign": "positive",
                    **result,
                }
            )
    for horizon in RETURN_HORIZONS:
        target = f"stage20_defense_opportunity_cost_{horizon}m"
        for model_name, controls in model_controls.items():
            result = _interaction_regression(
                frame, target, controls, max_lags=horizon
            )
            rows.append(
                {
                    "TestFamily": "Stage20DefenseOpportunityCost",
                    "Target": target,
                    "HorizonMonths": horizon,
                    "Model": model_name,
                    "ExpectedContextSign": "positive",
                    **result,
                }
            )
    defense_frame = frame.loc[
        frame["stage20_lower_equity_than_stage14"].eq(1.0)
    ]
    for model_name, controls in model_controls.items():
        result = _interaction_regression(
            defense_frame,
            "stage20_false_positive_1m",
            controls,
            max_lags=1,
            binary=True,
        )
        rows.append(
            {
                "TestFamily": "Stage20FalsePositive",
                "Target": "stage20_false_positive_1m",
                "HorizonMonths": 1,
                "Model": model_name,
                "ExpectedContextSign": "positive",
                **result,
            }
        )
    return pd.DataFrame(rows)


def causal_state_table(frame: pd.DataFrame) -> pd.DataFrame:
    states = (
        "VIX6 Low / Wing Low",
        "VIX6 High / Wing Low",
        "VIX6 Low / Wing High",
        "VIX6 High / Wing High",
    )
    rows: list[dict[str, Any]] = []
    for state in states:
        group = frame.loc[frame["causal_2x2_state"].eq(state)]
        defense = group.loc[group["stage20_lower_equity_than_stage14"].eq(1.0)]
        row: dict[str, Any] = {
            "State": state,
            "Months": int(len(group)),
            "FirstMonth": str(group.index.min()) if len(group) else None,
            "LastMonth": str(group.index.max()) if len(group) else None,
            "MeanVIX6": float(group[VIX_COLUMN].mean()),
            "MeanWingAsym": float(group[WING_COLUMN].mean()),
            "MeanFutureRealizedVol1M": float(
                group["future_realized_vol_1m"].mean()
            ),
            "MeanFutureMaxDrawdown1M": float(
                group["future_max_drawdown_1m"].mean()
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
            "DefenseMonthsVsStage14": int(
                group["stage20_lower_equity_than_stage14"].sum()
            ),
            "FalsePositiveRateAmongDefense1M": float(
                defense["stage20_false_positive_1m"].mean()
            ),
        }
        for horizon in RETURN_HORIZONS:
            row[f"MeanFutureK200Return{horizon}M"] = float(
                group[f"future_{horizon}m_return"].mean()
            )
            row[f"MeanStage20DefenseOpportunityCost{horizon}M"] = float(
                group[f"stage20_defense_opportunity_cost_{horizon}m"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def gate_decision(
    risk_regressions: pd.DataFrame,
    interactions: pd.DataFrame,
    state_table: pd.DataFrame,
) -> tuple[dict[str, Any], bool, bool, str]:
    full_risk = risk_regressions.loc[
        risk_regressions["Model"].eq("FullControls")
    ].set_index("Target")
    positive = full_risk["WingStandardizedBeta"] > 0.0
    significant = positive & (
        full_risk["WingHACPValue"] < SIGNIFICANCE_LEVEL
    )
    material_targets = {
        "future_max_drawdown_1m",
        "future_max_drawdown_3m",
        "future_left_tail_1m",
    }
    significant_material = [
        target for target in material_targets if bool(significant.loc[target])
    ]
    direct_gates = {
        "positive_sign_at_least_three_of_four_targets": bool(
            int(positive.sum()) >= 3
        ),
        "positive_and_significant_at_least_two_targets": bool(
            int(significant.sum()) >= 2
        ),
        "at_least_one_significant_drawdown_or_tail_target": bool(
            significant_material
        ),
    }
    direct_pass = bool(all(direct_gates.values()))

    return_full = interactions.loc[
        interactions["TestFamily"].eq("K200ForwardReturn")
        & interactions["Model"].eq("FullControlsContext")
        & interactions["HorizonMonths"].isin([3, 6])
    ].set_index("HorizonMonths")
    return_positive = return_full["InteractionStandardizedBeta"] > 0.0
    return_significant = return_positive & (
        return_full["InteractionHACPValue"] < SIGNIFICANCE_LEVEL
    )
    high_high = state_table.loc[
        state_table["State"].eq("VIX6 High / Wing High")
    ].iloc[0]
    context_gates = {
        "positive_return_interaction_at_3m_and_6m": bool(
            return_positive.reindex([3, 6]).fillna(False).all()
        ),
        "return_interaction_significant_at_3m_or_6m": bool(
            return_significant.reindex([3, 6]).fillna(False).any()
        ),
        "high_high_opportunity_cost_positive_at_3m_and_6m": bool(
            high_high["MeanStage20DefenseOpportunityCost3M"] > 0.0
            and high_high["MeanStage20DefenseOpportunityCost6M"] > 0.0
        ),
        "high_high_has_at_least_15_causal_months": bool(
            high_high["Months"] >= 15
        ),
    }
    context_pass = bool(all(context_gates.values()))
    if direct_pass:
        decision = "preserve_wingasym_as_incremental_risk_diagnostic_only"
    elif context_pass:
        decision = "preserve_wingasym_as_vix6_context_diagnostic_only"
    else:
        decision = "close_wingasym_branch_move_to_independent_information_source"
    gates = {
        "direct_risk_sensor": direct_gates,
        "vix6_context_reentry": context_gates,
        "significant_direct_targets": sorted(
            full_risk.index[significant].tolist()
        ),
    }
    return gates, direct_pass, context_pass, decision


def _plot_risk_coefficients(risk: pd.DataFrame, path: Path) -> None:
    subset = risk.loc[risk["Model"].eq("FullControls")].copy()
    labels = [
        "1M realized vol",
        "1M max drawdown",
        "3M max drawdown",
        "1M left-tail logit",
    ]
    x = np.arange(len(subset))
    z_statistic = (
        subset["WingStandardizedBeta"]
        / subset["WingHACStandardError"]
    ).to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [
        "#2f6f9f" if value > 0 else "#c4514e" for value in z_statistic
    ]
    ax.bar(x, z_statistic, color=colors, alpha=0.88)
    ax.axhline(0.0, color="#222", linewidth=1)
    ax.axhline(1.645, color="#777", linewidth=1, linestyle="--")
    ax.axhline(-1.645, color="#777", linewidth=1, linestyle="--")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("WingAsym HAC z-statistic")
    ax.set_title("Stage 33: WingAsym incremental future-risk significance")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_context_states(state_table: pd.DataFrame, path: Path) -> None:
    labels = state_table["State"].str.replace(" / ", "\n", regex=False)
    x = np.arange(len(state_table))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.bar(
        x - width / 2,
        state_table["MeanFutureK200Return3M"] * 100.0,
        width,
        label="Future K200 return, 3M",
        color="#2f6f9f",
    )
    ax.bar(
        x + width / 2,
        state_table["MeanStage20DefenseOpportunityCost3M"] * 100.0,
        width,
        label="Stage20 defense opportunity cost, 3M",
        color="#d18b47",
    )
    ax.axhline(0.0, color="#222", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.set_title("Causal expanding-median 2×2 state diagnostic")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_research(save: bool = True) -> dict[str, Any]:
    source_before = source_manifest()
    frozen_before = frozen_manifest()
    frame, audit = build_research_frame()
    risk = incremental_risk_regressions(frame)
    tail_scores, tail_predictions = expanding_tail_scores(frame)
    interactions = context_interaction_regressions(frame)
    states = causal_state_table(frame)
    gates, direct_pass, context_pass, decision = gate_decision(
        risk, interactions, states
    )

    source_after = source_manifest()
    frozen_after = frozen_manifest()
    source_unchanged = source_before == source_after
    frozen_unchanged = frozen_before == frozen_after
    checks = {
        "source_files_unchanged": source_unchanged,
        "stage20_to_stage32_code_and_results_unchanged": frozen_unchanged,
        "signals_strictly_precede_target_month": bool(
            (frame["bucket_signal_month"] < frame.index).all()
        ),
        "tail_threshold_uses_only_prior_returns": True,
        "state_cutoffs_use_only_prior_signal_history": True,
        "fixed_5_percent_tail_and_60_month_history": bool(
            TAIL_QUANTILE == 0.05
            and MIN_TAIL_HISTORY == 60
            and MIN_STATE_HISTORY == 60
        ),
        "no_threshold_window_bucket_or_sign_search": True,
        "no_residual_wingasym": True,
        "fear_term_slope_retired": True,
        "no_strategy_weights_or_expected_returns_changed": True,
    }
    stage20_performance = pd.read_csv(
        stage32.OUTPUT_DIR / "frozen_stage20_performance.csv"
    )
    report = {
        "study": "Stage33_WingAsymIncrementalRiskAndVIX6Context",
        "decision": decision,
        "direct_risk_sensor_pass": direct_pass,
        "vix6_context_reentry_pass": context_pass,
        "scope": (
            "Final WingAsym branch diagnostic. No trading rule, expected-return "
            "adjustment, risk-aversion adjustment, or overlay is implemented."
        ),
        "fixed_design": {
            "signal": "near-month OTM2 put IV minus call IV",
            "risk_targets": list(RISK_TARGETS),
            "return_horizons_months": list(RETURN_HORIZONS),
            "controls": [VIX_COLUMN, *CONTROL_COLUMNS],
            "interaction": "z(VIX6 stress) multiplied by z(WingAsym)",
            "left_tail_quantile": TAIL_QUANTILE,
            "minimum_causal_history_months": MIN_TAIL_HISTORY,
            "significance_level": SIGNIFICANCE_LEVEL,
            "searched_parameters": None,
        },
        "gate_rules": {
            "direct_risk": (
                "Full-control WingAsym coefficient must be positive for at "
                "least 3 of 4 targets, positive with p<10% for at least 2, "
                "and significant for at least one drawdown or tail target."
            ),
            "context_reentry": (
                "Full-control VIX6×WingAsym return interaction must be positive "
                "at both 3m and 6m and p<10% at either; the causal high-high "
                "state must have positive Stage20 defense opportunity cost at "
                "both horizons and at least 15 months."
            ),
        },
        "gate_results": gates,
        "source_audit": audit,
        "incremental_risk_regressions": json.loads(
            risk.to_json(orient="records", force_ascii=False)
        ),
        "tail_oos_scores": json.loads(
            tail_scores.to_json(orient="records", force_ascii=False)
        ),
        "context_interaction_regressions": json.loads(
            interactions.to_json(orient="records", force_ascii=False)
        ),
        "causal_state_table": json.loads(
            states.to_json(orient="records", force_ascii=False)
        ),
        "frozen_stage20_performance": json.loads(
            stage20_performance.to_json(orient="records", force_ascii=False)
        ),
        "checks": checks,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
        "frozen_manifest_before": frozen_before,
        "frozen_manifest_after": frozen_after,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUTPUT_DIR / "monthly_stage33_research_frame.csv")
        risk.to_csv(
            OUTPUT_DIR / "future_risk_incremental_regressions.csv", index=False
        )
        tail_scores.to_csv(
            OUTPUT_DIR / "left_tail_expanding_oos_scores.csv", index=False
        )
        tail_predictions.to_csv(
            OUTPUT_DIR / "left_tail_expanding_oos_predictions.csv", index=False
        )
        interactions.to_csv(
            OUTPUT_DIR / "vix6_wing_context_interactions.csv", index=False
        )
        states.to_csv(OUTPUT_DIR / "causal_2x2_state_diagnostic.csv", index=False)
        stage20_performance.to_csv(
            OUTPUT_DIR / "frozen_stage20_performance.csv", index=False
        )
        _plot_risk_coefficients(
            risk, OUTPUT_DIR / "incremental_risk_coefficients.png"
        )
        _plot_context_states(states, OUTPUT_DIR / "causal_2x2_context.png")
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> None:
    report = run_research(save=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
