from __future__ import annotations

"""Stage36 계열 overlay를 실제로 켜고 끄는 검증 전용 연구 스크립트.

운영 전략인 ``explainable_regime_allocator.py``는 이 파일을 import하지 않는다.
여기서는 새 단일 전략에서 이미 제거된 레이어의 증분 성과를 검증하기 위해 동결된
Stage36을 'legacy reference adapter' 한 개로만 불러온다. 이 경계 덕분에 운영
코드는 과거 Stage 모듈 그래프에 의존하지 않으면서도, 반사실 실험은 원래 계산식과
데이터를 정확히 재사용한다.

실험 원칙
---------
* 한 경로에서 한 레이어만 제거한다(one-at-a-time ablation).
* 수익률 표본, 거래비용, 목적함수, 13% 변동성 및 16% CDaR 정책은 고정한다.
* 전체 레이어 경로가 저장된 Stage36 월별 수익률/비중을 재현하는지 먼저 확인한다.
* KODEX-only stress 공분산은 KODEX 분산비만 D Sigma D로 적용한다. 이 정의는
  대칭·양의 준정부호를 보존하고 다른 자산의 자체 분산을 직접 바꾸지 않는다.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from strategies.stage36_asset_implied_volatility_risk import (
    asset_implied_volatility_risk_slsqp as legacy,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ASSETS = tuple(legacy.ASSETS)
EQUITY_INDEX = ASSETS.index("KODEX200")
GOLD_INDEX = ASSETS.index("GLD")
OIL_INDEX = ASSETS.index("USO")
WEIGHT_COLUMNS = tuple(f"w_{asset}" for asset in ASSETS)


@dataclass(frozen=True)
class LayerSwitches:
    """True는 해당 Stage36 레이어가 활성화되었다는 뜻이다."""

    name: str
    stress_expected_return: bool = True
    stress_covariance: bool = True
    technical_confidence: bool = True
    atr_covariance: bool = True
    credit_stress_and_risk: bool = True
    gvz_ovx_variance: bool = True
    stress_scope: str = "all_assets"  # all_assets | kodex200_only


REFERENCE = LayerSwitches("AllLayersReference")
EXPERIMENTS = (
    REFERENCE,
    replace(
        REFERENCE,
        name="NoStressExpectedReturn",
        stress_expected_return=False,
    ),
    replace(
        REFERENCE,
        name="NoStressCovarianceBlend",
        stress_covariance=False,
    ),
    replace(
        REFERENCE,
        name="NoTechnicalConfidence",
        technical_confidence=False,
    ),
    replace(
        REFERENCE,
        name="NoATRCovarianceScaling",
        atr_covariance=False,
    ),
    replace(
        REFERENCE,
        name="NoCreditStressRiskScaling",
        credit_stress_and_risk=False,
    ),
    replace(
        REFERENCE,
        name="NoGVZOVXVarianceScaling",
        gvz_ovx_variance=False,
    ),
    replace(
        REFERENCE,
        name="StressKODEX200Only",
        stress_scope="kodex200_only",
    ),
)


@dataclass(frozen=True)
class ResearchInputs:
    returns: pd.DataFrame
    probabilities: pd.DataFrame
    stress: pd.DataFrame
    technical: pd.DataFrame
    fundamental: pd.DataFrame
    asset_volatility: pd.DataFrame


def load_research_inputs() -> ResearchInputs:
    """동결 Stage36이 사용한 원자료와 causal signal을 한 번만 구성한다."""

    implied_daily, _ = legacy.load_asset_implied_volatility_daily()
    returns, _ = legacy.stage35.load_monthly_asset_returns(False)
    probabilities, _ = legacy.stage35.build_macro_probabilities(returns)
    stress = legacy.stage35.build_monthly_stress_signals(
        returns.index, legacy.stage35.build_daily_stress_features()
    )
    market, _ = legacy.stage35.stage20.load_daily_asset_ohlcv()
    raw_fundamental, _ = legacy.stage35.load_fundamental_daily()
    fundamental = legacy.stage35.build_monthly_fundamental_signals(
        raw_fundamental
    )
    equity_close = market["KODEX200"]["close"].dropna()
    monthly_equity_close = equity_close.groupby(
        equity_close.index.to_period("M")
    ).last()
    calibrated = legacy.stage35.add_causal_return_calibration(
        fundamental, monthly_equity_close.pct_change()
    )
    technical = legacy.stage35.stage34._load_period_csv(
        legacy.stage35.stage20.OUTPUT_DIR / "monthly_technical_signals.csv",
        "target_month",
    )
    asset_volatility = legacy.build_monthly_asset_volatility_signals(
        implied_daily, returns.index
    )
    return ResearchInputs(
        returns=returns,
        probabilities=probabilities,
        stress=stress,
        technical=technical,
        fundamental=calibrated,
        asset_volatility=asset_volatility,
    )


def _nearest_positive_definite(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(
        float(np.trace(symmetric)) / len(symmetric),
        1e-12,
    )
    floor = scale * 1e-10
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _kodex_only_stress_covariance(
    macro_covariance: np.ndarray, all_asset_stress_covariance: np.ndarray
) -> np.ndarray:
    """KODEX 자체 분산의 stress 비율만 D Sigma D로 전달한다.

    이 정의에서 BOND/GLD/USO의 자체 분산은 macro covariance와 동일하다. KODEX와
    다른 자산의 공분산은 KODEX 표준편차 변화에 비례해 변한다. 완전히 0으로 두면
    상관구조를 깨뜨리므로, PSD를 보존하는 가장 작은 축 변환을 사용한다.
    """

    base_variance = max(
        float(macro_covariance[EQUITY_INDEX, EQUITY_INDEX]), 1e-16
    )
    stress_variance = max(
        float(all_asset_stress_covariance[EQUITY_INDEX, EQUITY_INDEX]), 1e-16
    )
    variance_ratio = stress_variance / base_variance
    scaling = np.eye(len(ASSETS), dtype=float)
    scaling[EQUITY_INDEX, EQUITY_INDEX] = math.sqrt(variance_ratio)
    return scaling @ macro_covariance @ scaling


def build_forecast(
    inputs: ResearchInputs,
    month: pd.Period,
    switches: LayerSwitches,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """한 달의 Stage36 mu/Sigma에서 지정된 레이어만 제거한다."""

    history = inputs.returns.loc[inputs.returns.index < month, ASSETS]
    probabilities = inputs.probabilities.loc[
        inputs.probabilities.index < month
    ]
    historical_stress = inputs.stress.loc[
        inputs.stress.index < month, "stress_score"
    ]
    historical_recovery = inputs.stress.loc[
        inputs.stress.index < month, "recovery_score"
    ]
    common_arguments = {
        "history": history,
        "historical_probabilities": probabilities,
        "current_probabilities": inputs.probabilities.loc[month],
        "historical_stress": historical_stress,
        "current_stress": float(inputs.stress.loc[month, "stress_score"]),
        "historical_recovery": historical_recovery,
        "current_recovery": float(inputs.stress.loc[month, "recovery_score"]),
    }
    _, macro_covariance, macro_detail = (
        legacy.stage35.estimate_conditional_moments(
            **common_arguments, use_short_term_stress=False
        )
    )
    _, all_asset_stress_covariance, stress_detail = (
        legacy.stage35.estimate_conditional_moments(
            **common_arguments, use_short_term_stress=True
        )
    )
    macro_mu = np.asarray(
        macro_detail["macro_expected_monthly_return"], dtype=float
    )
    stress_adjustment = np.asarray(
        stress_detail["stress_return_adjustment"], dtype=float
    )
    if not switches.stress_expected_return:
        stress_adjustment = np.zeros_like(stress_adjustment)
    if switches.stress_scope == "kodex200_only":
        scoped = np.zeros_like(stress_adjustment)
        scoped[EQUITY_INDEX] = stress_adjustment[EQUITY_INDEX]
        stress_adjustment = scoped

    covariance = (
        all_asset_stress_covariance
        if switches.stress_covariance
        else macro_covariance
    )
    if switches.stress_covariance and switches.stress_scope == "kodex200_only":
        covariance = _kodex_only_stress_covariance(
            macro_covariance, all_asset_stress_covariance
        )

    technical = legacy.stage35.stage20.apply_technical_inputs(
        macro_mu, covariance, inputs.technical.loc[month]
    )
    filtered_macro = (
        np.asarray(technical["filtered_macro_expected_return"], dtype=float)
        if switches.technical_confidence
        else macro_mu.copy()
    )
    covariance = (
        np.asarray(technical["adjusted_covariance"], dtype=float)
        if switches.atr_covariance
        else covariance.copy()
    )

    fundamental = inputs.fundamental.loc[month]
    filtered_macro[EQUITY_INDEX] += float(
        fundamental["eps_mu_adjustment_KODEX200"]
    ) + float(fundamental["valuation_mu_adjustment_KODEX200"])
    credit_stress_multiplier = 1.0
    credit_variance_multiplier = 1.0
    if switches.credit_stress_and_risk:
        credit_stress_multiplier = float(
            fundamental["credit_stress_multiplier"]
        )
        credit_variance_multiplier = 1.0 + float(
            fundamental["credit_stress_rank"]
        )
        stress_adjustment[EQUITY_INDEX] *= credit_stress_multiplier
        credit_scaling = np.eye(len(ASSETS), dtype=float)
        credit_scaling[EQUITY_INDEX, EQUITY_INDEX] = math.sqrt(
            credit_variance_multiplier
        )
        covariance = credit_scaling @ covariance @ credit_scaling

    gvz_multiplier = 1.0
    ovx_multiplier = 1.0
    if switches.gvz_ovx_variance:
        asset_volatility = inputs.asset_volatility.loc[month]
        gvz_multiplier = float(
            asset_volatility["gvz_gld_variance_multiplier"]
        )
        ovx_multiplier = float(
            asset_volatility["ovx_uso_variance_multiplier"]
        )
        asset_scaling = np.eye(len(ASSETS), dtype=float)
        asset_scaling[GOLD_INDEX, GOLD_INDEX] = math.sqrt(gvz_multiplier)
        asset_scaling[OIL_INDEX, OIL_INDEX] = math.sqrt(ovx_multiplier)
        covariance = asset_scaling @ covariance @ asset_scaling

    expected_return = filtered_macro + stress_adjustment
    detail = {
        "history": history,
        "historical_stress": historical_stress,
        "regime_beta": stress_detail["regime_stress_beta"],
        "credit_stress_multiplier": credit_stress_multiplier,
        "credit_variance_multiplier": credit_variance_multiplier,
        "gvz_multiplier": gvz_multiplier,
        "ovx_multiplier": ovx_multiplier,
    }
    # 모든 입력 공분산은 이미 PSD이고 이후 변환은 D Sigma D뿐이므로 다시
    # eigenvalue flooring하지 않는다. 불필요한 flooring은 동결 Stage36과
    # 수 마이크로 단위의 비중 차이를 만들 수 있다.
    return expected_return, covariance, detail


def _solve_stage36_objective(
    expected_return: np.ndarray,
    covariance: np.ndarray,
    historical_returns: np.ndarray,
    pretrade: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stage36의 downside 포함 목적함수와 위험제약을 그대로 푼다."""

    initial = (
        legacy.stage35.project_to_long_only_simplex(pretrade)
        if np.isfinite(pretrade).all() and pretrade.sum() > 0.99
        else np.repeat(1.0 / len(ASSETS), len(ASSETS))
    )

    def portfolio_values(weights: np.ndarray) -> dict[str, float]:
        mean = float(weights @ expected_return)
        variance = max(float(weights @ covariance @ weights), 0.0)
        realized = historical_returns @ weights
        downside = float(np.mean(np.minimum(realized, 0.0) ** 2))
        cost = legacy.stage35.expected_transaction_cost(weights, pretrade)
        return {
            "expected_monthly_return": mean,
            "expected_monthly_variance": variance,
            "downside_semivariance": downside,
            "estimated_transaction_cost": cost,
            "monthly_utility": mean - 0.5 * variance - downside - cost,
        }

    def annual_volatility(weights: np.ndarray) -> float:
        return math.sqrt(max(float(weights @ covariance @ weights), 0.0) * 12.0)

    constraints = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: legacy.stage35.CATASTROPHE_ANNUAL_VOLATILITY
            - annual_volatility(weights),
        },
        {
            "type": "ineq",
            "fun": lambda weights: legacy.stage35.CATASTROPHE_CDAR
            + legacy.stage35.cdar(
                historical_returns @ weights,
                legacy.stage35.CDAR_CONFIDENCE,
            ),
        },
    ]
    result = minimize(
        lambda weights: -portfolio_values(weights)["monthly_utility"],
        initial,
        method="SLSQP",
        bounds=legacy.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
        constraints=constraints,
        options={
            "maxiter": legacy.stage35.SLSQP_MAX_ITERATIONS,
            "ftol": legacy.stage35.SLSQP_TOLERANCE,
        },
    )
    used_fallback = False
    if result.success and np.isfinite(result.x).all():
        weights = legacy.stage35.project_to_long_only_simplex(result.x)
    else:
        fallback = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=legacy.stage35.UNCONSTRAINED_LONG_ONLY_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": legacy.stage35.SLSQP_MAX_ITERATIONS,
                "ftol": legacy.stage35.SLSQP_TOLERANCE,
            },
        )
        if not fallback.success or not np.isfinite(fallback.x).all():
            raise RuntimeError(
                f"Primary and fallback solves failed: {result.message}; "
                f"{fallback.message}"
            )
        result = fallback
        weights = legacy.stage35.project_to_long_only_simplex(fallback.x)
        used_fallback = True
    values = portfolio_values(weights)
    annual_vol = annual_volatility(weights)
    historical_cdar = legacy.stage35.cdar(
        historical_returns @ weights, legacy.stage35.CDAR_CONFIDENCE
    )
    return weights, {
        **values,
        "used_fallback": used_fallback,
        "solver_success": bool(result.success),
        "expected_annual_volatility": annual_vol,
        "historical_cdar": historical_cdar,
        "volatility_slack": legacy.stage35.CATASTROPHE_ANNUAL_VOLATILITY
        - annual_vol,
        "cdar_slack": legacy.stage35.CATASTROPHE_CDAR + historical_cdar,
    }


def _eligible_months(inputs: ResearchInputs) -> pd.PeriodIndex:
    required_fundamental = [
        "eps_mu_adjustment_KODEX200",
        "valuation_mu_adjustment_KODEX200",
        "credit_stress_multiplier",
        "credit_stress_rank",
    ]
    months = inputs.returns.index.intersection(inputs.probabilities.index)
    months = months.intersection(inputs.stress.index)
    months = months.intersection(inputs.technical.index)
    months = months.intersection(inputs.asset_volatility.index)
    months = months.intersection(
        inputs.fundamental.dropna(subset=required_fundamental).index
    )
    return months[
        (months >= legacy.FULL_START) & (months <= legacy.RESEARCH_END)
    ]


def run_path(
    inputs: ResearchInputs, switches: LayerSwitches
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    pretrade = np.zeros(len(ASSETS), dtype=float)
    first_trade = True
    nav = 1.0
    peak = 1.0

    for month in _eligible_months(inputs):
        expected_return, covariance, detail = build_forecast(
            inputs, month, switches
        )
        history = detail["history"]
        historical_stress = detail["historical_stress"]
        common = history.index.intersection(historical_stress.dropna().index)
        historical_returns = history.loc[common, ASSETS].to_numpy(dtype=float)
        weights, solve = _solve_stage36_objective(
            expected_return, covariance, historical_returns, pretrade
        )
        change = weights - pretrade
        turnover = (
            float(np.abs(change).sum())
            if first_trade
            else 0.5 * float(np.abs(change).sum())
        )
        trade_cost = (
            float(np.abs(change).sum()) * legacy.stage35.DOMESTIC_TRADE_COST
        )
        foreign = [ASSETS.index("GLD"), ASSETS.index("USO")]
        fx_cost = (
            abs(float(change[foreign].sum()))
            * legacy.stage35.FOREIGN_WEIGHT_CHANGE_COST
        )
        asset_return = inputs.returns.loc[month, list(ASSETS)].to_numpy(
            dtype=float
        )
        gross_return = float(weights @ asset_return)
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1.0 + net_return
        peak = max(peak, nav)
        rows.append(
            {
                "Strategy": switches.name,
                "month": month,
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1.0,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
                **solve,
            }
        )
        if switches == REFERENCE:
            for regime, kinds in detail["regime_beta"].items():
                for beta_kind, coefficients in kinds.items():
                    for asset, coefficient in zip(ASSETS, coefficients):
                        beta_rows.append(
                            {
                                "month": month,
                                "regime": regime,
                                "beta_kind": beta_kind,
                                "asset": asset,
                                "beta": float(coefficient),
                            }
                        )
        pretrade = weights * (1.0 + asset_return) / (1.0 + gross_return)
        first_trade = False

    path = pd.DataFrame(rows).set_index("month")
    path.index = pd.PeriodIndex(path.index, freq="M")
    return path, pd.DataFrame(beta_rows)


def _performance_summary(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    wealth = (1.0 + clean).cumprod()
    years = len(clean) / 12.0
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    volatility = float(clean.std(ddof=1) * math.sqrt(12.0))
    sharpe = float(clean.mean() / clean.std(ddof=1) * math.sqrt(12.0))
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "Months": float(len(clean)),
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "MDD": float(drawdown.min()),
    }


def build_performance_table(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = {
        "full_2007_2026": (legacy.FULL_START, legacy.RESEARCH_END),
        "locked_2018_2026": (
            legacy.stage35.LOCKED_START,
            legacy.RESEARCH_END,
        ),
    }
    for name, path in paths.items():
        for period, (start, end) in periods.items():
            view = path.loc[start:end]
            rows.append(
                {
                    "Strategy": name,
                    "Period": period,
                    "Start": str(view.index.min()),
                    "End": str(view.index.max()),
                    **_performance_summary(view["return"]),
                    "AvgTurnover": float(view["turnover"].mean()),
                    "TotalCost": float(
                        view[["trade_cost", "fx_cost"]].sum().sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_ablation_deltas(performance: pd.DataFrame) -> pd.DataFrame:
    indexed = performance.set_index(["Strategy", "Period"])
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS[1:]:
        for period in ("full_2007_2026", "locked_2018_2026"):
            reference = indexed.loc[(REFERENCE.name, period)]
            candidate = indexed.loc[(experiment.name, period)]
            rows.append(
                {
                    "Ablation": experiment.name,
                    "Period": period,
                    "DeltaCAGR": float(candidate["CAGR"] - reference["CAGR"]),
                    "DeltaVolatility": float(
                        candidate["Volatility"] - reference["Volatility"]
                    ),
                    "DeltaSharpe": float(
                        candidate["Sharpe"] - reference["Sharpe"]
                    ),
                    "DeltaMDD": float(candidate["MDD"] - reference["MDD"]),
                    "DeltaAvgTurnover": float(
                        candidate["AvgTurnover"] - reference["AvgTurnover"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_paired_bootstrap(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """시계열 의존을 보존하는 12개월 circular block bootstrap."""

    rows: list[pd.DataFrame] = []
    reference = paths[REFERENCE.name]
    periods = {
        "full_2007_2026": (legacy.FULL_START, legacy.RESEARCH_END),
        "locked_2018_2026": (
            legacy.stage35.LOCKED_START,
            legacy.RESEARCH_END,
        ),
    }
    for experiment in EXPERIMENTS[1:]:
        candidate = paths[experiment.name]
        for period, (start, end) in periods.items():
            summary = legacy.stage35.stage30.paired_block_bootstrap(
                reference.loc[start:end, "return"],
                candidate.loc[start:end, "return"],
                replications=2_000,
                block_months=12,
            )
            summary.insert(0, "Reference", REFERENCE.name)
            summary.insert(1, "Candidate", experiment.name)
            summary.insert(2, "Period", period)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def beta_stability_summary(beta_history: pd.DataFrame) -> pd.DataFrame:
    """regime/asset/beta별 sign와 magnitude 안정성을 요약한다."""

    rows: list[dict[str, Any]] = []
    keys = ["regime", "asset", "beta_kind"]
    for key, group in beta_history.sort_values("month").groupby(keys):
        values = group["beta"].to_numpy(dtype=float)
        group_months = pd.PeriodIndex(group["month"], freq="M")
        early_values = values[group_months < legacy.stage35.LOCKED_START]
        locked_values = values[group_months >= legacy.stage35.LOCKED_START]
        early_median = float(np.median(early_values))
        locked_median = float(np.median(locked_values))
        tolerance = 1e-12
        signs = np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))
        nonzero_signs = signs[signs != 0]
        sign_switches = (
            int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
            if len(nonzero_signs) > 1
            else 0
        )
        absolute = np.abs(values)
        nonzero_absolute = absolute[absolute > tolerance]
        median_absolute = (
            float(np.median(nonzero_absolute)) if len(nonzero_absolute) else 0.0
        )
        p90_absolute = (
            float(np.quantile(nonzero_absolute, 0.90))
            if len(nonzero_absolute)
            else 0.0
        )
        rows.append(
            {
                "regime": key[0],
                "asset": key[1],
                "beta_kind": key[2],
                "observations": int(len(values)),
                "first_beta": float(values[0]),
                "last_beta": float(values[-1]),
                "mean_beta": float(np.mean(values)),
                "median_beta": float(np.median(values)),
                "early_2007_2017_median_beta": early_median,
                "locked_2018_2026_median_beta": locked_median,
                "subperiod_median_sign_changed": bool(
                    abs(early_median) > tolerance
                    and abs(locked_median) > tolerance
                    and np.sign(early_median) != np.sign(locked_median)
                ),
                "locked_to_early_median_magnitude": (
                    abs(locked_median) / abs(early_median)
                    if abs(early_median) > tolerance
                    else math.nan
                ),
                "min_beta": float(np.min(values)),
                "max_beta": float(np.max(values)),
                "positive_fraction": float(np.mean(signs > 0)),
                "negative_fraction": float(np.mean(signs < 0)),
                "zero_fraction": float(np.mean(signs == 0)),
                "nonzero_sign_switches": sign_switches,
                "median_absolute_beta": median_absolute,
                "p90_absolute_beta": p90_absolute,
                "p90_to_median_magnitude": (
                    p90_absolute / median_absolute
                    if median_absolute > tolerance
                    else math.nan
                ),
                "largest_one_month_change": float(
                    np.max(np.abs(np.diff(values))) if len(values) > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _causal_zscore(series: pd.Series, minimum: int = 60) -> pd.Series:
    output = pd.Series(np.nan, index=series.index, dtype=float)
    for position in range(len(series)):
        history = series.iloc[: position + 1].dropna()
        if len(history) < minimum:
            continue
        standard_deviation = float(history.std(ddof=1))
        if standard_deviation <= 1e-12:
            continue
        output.iloc[position] = float(
            (series.iloc[position] - history.mean()) / standard_deviation
        )
    return output


def _expanding_univariate_bayesian_loss(
    feature: pd.Series,
    target: pd.Series,
    start: pd.Period,
) -> dict[str, float]:
    """같은 Bayesian ridge로 rank와 z의 one-step return MSE를 비교한다."""

    frame = pd.concat({"feature": feature, "target": target}, axis=1)
    predictions: list[float] = []
    actuals: list[float] = []
    for month in frame.index[frame.index >= start]:
        history = frame.loc[frame.index < month].dropna()
        current = frame.loc[month]
        if len(history) < 20 or current.isna().any():
            continue
        mean = float(history["feature"].mean())
        std = float(history["feature"].std(ddof=1))
        if std <= 1e-12:
            continue
        x = ((history["feature"] - mean) / std).to_numpy().reshape(-1, 1)
        current_x = np.array([[(float(current["feature"]) - mean) / std]])
        model = BayesianRidge(
            fit_intercept=True,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6,
        )
        model.fit(x, history["target"].to_numpy(dtype=float))
        predictions.append(float(model.predict(current_x)[0]))
        actuals.append(float(current["target"]))
    errors = np.asarray(actuals) - np.asarray(predictions)
    return {
        "observations": float(len(errors)),
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
    }


def credit_encoding_diagnostics(
    inputs: ResearchInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """credit rank-only/z-only/current mixed 선택을 역할별 OOS loss로 비교한다.

    비교 정의
    ---------
    * rank: AA spread 20일 변화의 causal empirical CDF.
    * z: 같은 원자료의 causal expanding z-score.
    * mixed: return에는 signed magnitude가 있는 z, risk에는 bounded rank를 쓴다.

    서로 단위가 다른 return MSE와 variance loss를 억지로 더하지 않는다. 대신 각
    역할에서 rank와 z 중 무엇이 낮은 OOS loss를 냈는지 보고 mixed 선택을 판정한다.
    """

    fundamental = inputs.fundamental.copy()
    raw = fundamental["aa_spread_widening_20d_pctpt"]
    rank = fundamental["credit_stress_rank"]
    zscore = fundamental["credit_widening_z"]

    return_rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        for encoding, feature in {"rank": rank, "zscore": zscore}.items():
            loss = _expanding_univariate_bayesian_loss(
                feature,
                inputs.returns[asset],
                legacy.FULL_START,
            )
            return_rows.append(
                {"role": "next_month_return", "asset": asset, "encoding": encoding, **loss}
            )

    # 위험 역할은 다음 달 제곱수익률의 log를 예측한다. 완전한 realized variance는
    # 아니지만 네 자산에 공통으로 정의되고 방향 예측과 분산 상태를 분리한다.
    risk_rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        target = np.log(inputs.returns[asset].pow(2).clip(lower=1e-10))
        for encoding, feature in {"rank": rank, "zscore": zscore}.items():
            loss = _expanding_univariate_bayesian_loss(
                feature,
                target,
                legacy.FULL_START,
            )
            risk_rows.append(
                {"role": "next_month_log_squared_return", "asset": asset, "encoding": encoding, **loss}
            )

    losses = pd.DataFrame(return_rows + risk_rows)
    decisions: list[dict[str, Any]] = []
    for asset in ASSETS:
        return_view = losses.loc[
            losses["role"].eq("next_month_return") & losses["asset"].eq(asset)
        ].set_index("encoding")
        risk_view = losses.loc[
            losses["role"].eq("next_month_log_squared_return")
            & losses["asset"].eq(asset)
        ].set_index("encoding")
        best_return = str(return_view["mse"].idxmin())
        best_risk = str(risk_view["mse"].idxmin())
        decisions.append(
            {
                "asset": asset,
                "best_return_encoding": best_return,
                "best_risk_encoding": best_risk,
                "supports_current_mixed_z_return_rank_risk": bool(
                    best_return == "zscore" and best_risk == "rank"
                ),
                "rank_only_total_wins": int(best_return == "rank")
                + int(best_risk == "rank"),
                "zscore_only_total_wins": int(best_return == "zscore")
                + int(best_risk == "zscore"),
            }
        )
    decisions_frame = pd.DataFrame(decisions)

    # 역할별 최소 MSE로 나누면 return과 log-risk의 단위 차이를 제거할 수 있다.
    # 1.0이 해당 task의 최선이며, 여러 task의 평균 상대손실이 낮을수록 좋다.
    policies = {
        "rank_only": {
            "next_month_return": "rank",
            "next_month_log_squared_return": "rank",
        },
        "zscore_only": {
            "next_month_return": "zscore",
            "next_month_log_squared_return": "zscore",
        },
        "current_mixed_z_return_rank_risk": {
            "next_month_return": "zscore",
            "next_month_log_squared_return": "rank",
        },
    }
    policy_rows: list[dict[str, Any]] = []
    for policy, role_encoding in policies.items():
        relative_losses: list[float] = []
        task_wins = 0
        for (role, asset), group in losses.groupby(["role", "asset"]):
            indexed = group.set_index("encoding")
            selected = role_encoding[str(role)]
            minimum = float(indexed["mse"].min())
            selected_loss = float(indexed.loc[selected, "mse"])
            relative_losses.append(selected_loss / minimum)
            task_wins += int(np.isclose(selected_loss, minimum))
        policy_rows.append(
            {
                "policy": policy,
                "tasks": int(len(relative_losses)),
                "task_wins": int(task_wins),
                "mean_relative_mse": float(np.mean(relative_losses)),
                "worst_relative_mse": float(np.max(relative_losses)),
            }
        )
    return losses, decisions_frame, pd.DataFrame(policy_rows)


def run_validation(save: bool = True) -> dict[str, Any]:
    inputs = load_research_inputs()
    paths: dict[str, pd.DataFrame] = {}
    beta_history = pd.DataFrame()
    for experiment in EXPERIMENTS:
        path, betas = run_path(inputs, experiment)
        paths[experiment.name] = path
        if experiment == REFERENCE:
            beta_history = betas

    performance = build_performance_table(paths)
    deltas = build_ablation_deltas(performance)
    bootstrap = build_paired_bootstrap(paths)
    beta_summary = beta_stability_summary(beta_history)
    encoding_losses, encoding_decisions, encoding_policies = (
        credit_encoding_diagnostics(inputs)
    )

    frozen = legacy.stage35.stage34._load_period_csv(
        legacy.OUTPUT_DIR / "stage36_gvz_ovxassetrisk_monthly.csv", "month"
    )
    reference = paths[REFERENCE.name]
    common = frozen.index.intersection(reference.index)
    max_return_error = float(
        (frozen.loc[common, "return"] - reference.loc[common, "return"])
        .abs()
        .max()
    )
    max_weight_error = float(
        np.abs(
            frozen.loc[common, list(WEIGHT_COLUMNS)].to_numpy(dtype=float)
            - reference.loc[common, list(WEIGHT_COLUMNS)].to_numpy(dtype=float)
        ).max()
    )

    stress_scope = performance.loc[
        performance["Strategy"].isin(
            [REFERENCE.name, "StressKODEX200Only"]
        )
    ].copy()
    report = {
        "design": {
            "ablation": "one layer removed while every other layer and policy is fixed",
            "stress_scope_kodex_only": "only KODEX200 variance ratio enters through a PSD-preserving D Sigma D map",
            "encoding_comparison": "credit next-month return MSE and log-squared-return MSE are reported separately",
            "beta_stability": "causal expanding coefficients, sign switches ignore exact zeros",
        },
        "checks": {
            "reference_months_match_frozen": bool(len(common) == len(frozen)),
            "reference_return_max_abs_error_below_1e_8": bool(
                max_return_error < 1e-8
            ),
            "reference_weight_max_abs_error_below_1e_7": bool(
                max_weight_error < 1e-7
            ),
            "all_paths_finite": bool(
                all(np.isfinite(path["return"]).all() for path in paths.values())
            ),
            "all_paths_long_only": bool(
                all(
                    path[list(WEIGHT_COLUMNS)].min().min() >= -1e-8
                    for path in paths.values()
                )
            ),
            "all_paths_fully_invested": bool(
                all(
                    np.allclose(
                        path[list(WEIGHT_COLUMNS)].sum(axis=1), 1.0, atol=1e-8
                    )
                    for path in paths.values()
                )
            ),
        },
        "reference_reproduction": {
            "months": int(len(common)),
            "max_return_absolute_error": max_return_error,
            "max_weight_absolute_error": max_weight_error,
        },
        "beta_instability_flags": {
            "series_with_nonzero_sign_switches": int(
                beta_summary["nonzero_sign_switches"].gt(0).sum()
            ),
            "series_total": int(len(beta_summary)),
            "series_with_p90_to_median_above_3": int(
                beta_summary["p90_to_median_magnitude"].gt(3.0).sum()
            ),
            "series_with_early_locked_median_sign_change": int(
                beta_summary["subperiod_median_sign_changed"].sum()
            ),
            "unconstrained_bond_gold_series_with_sign_switches": int(
                beta_summary.loc[
                    beta_summary["asset"].isin(["BOND", "GLD"]),
                    "nonzero_sign_switches",
                ]
                .gt(0)
                .sum()
            ),
            "unconstrained_bond_gold_series_total": int(
                beta_summary["asset"].isin(["BOND", "GLD"]).sum()
            ),
        },
        "credit_encoding_support": {
            "assets_supporting_mixed": int(
                encoding_decisions[
                    "supports_current_mixed_z_return_rank_risk"
                ].sum()
            ),
            "assets_total": int(len(encoding_decisions)),
            "best_policy_by_mean_relative_mse": str(
                encoding_policies.set_index("policy")["mean_relative_mse"].idxmin()
            ),
        },
    }
    report["all_checks_pass"] = bool(all(report["checks"].values()))

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [path.reset_index() for path in paths.values()], ignore_index=True
        ).to_csv(OUTPUT_DIR / "layer_ablation_monthly.csv", index=False)
        performance.to_csv(
            OUTPUT_DIR / "layer_ablation_performance.csv", index=False
        )
        deltas.to_csv(OUTPUT_DIR / "layer_ablation_deltas.csv", index=False)
        bootstrap.to_csv(
            OUTPUT_DIR / "layer_ablation_bootstrap.csv", index=False
        )
        stress_scope.to_csv(
            OUTPUT_DIR / "stress_scope_performance.csv", index=False
        )
        beta_history.to_csv(
            OUTPUT_DIR / "regime_beta_history.csv", index=False
        )
        beta_summary.to_csv(
            OUTPUT_DIR / "regime_beta_stability.csv", index=False
        )
        encoding_losses.to_csv(
            OUTPUT_DIR / "credit_encoding_oos_losses.csv", index=False
        )
        encoding_decisions.to_csv(
            OUTPUT_DIR / "credit_encoding_decisions.csv", index=False
        )
        encoding_policies.to_csv(
            OUTPUT_DIR / "credit_encoding_policy_comparison.csv", index=False
        )
        with (OUTPUT_DIR / "ablation_validation_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    return {
        "paths": paths,
        "performance": performance,
        "deltas": deltas,
        "bootstrap": bootstrap,
        "stress_scope": stress_scope,
        "beta_history": beta_history,
        "beta_stability": beta_summary,
        "encoding_losses": encoding_losses,
        "encoding_decisions": encoding_decisions,
        "encoding_policies": encoding_policies,
        "report": report,
    }


def main() -> None:
    result = run_validation(save=True)
    print(result["performance"].to_string(index=False))
    print("\nAblation deltas versus all layers:\n")
    print(result["deltas"].to_string(index=False))
    print("\nValidation:", "PASS" if result["report"]["all_checks_pass"] else "FAIL")


if __name__ == "__main__":
    main()
