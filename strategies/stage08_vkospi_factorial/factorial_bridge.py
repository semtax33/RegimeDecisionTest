from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.core.regime_research import (
    ASSETS,
    StrategyConfig,
    hard_regime_weights,
    load_monthly_asset_returns,
    performance_summary,
    run_backtest,
)
from strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy import (
    balanced_logistic_spec,
    build_domestic_features,
    build_no_sjm_signals,
    forward_path_loss,
)
from strategies.stage06_vkospi.vkospi_dynamic_risk_experiment import (
    DynamicRiskConfig,
    build_daily_vkospi_signals,
    load_daily_open_levels,
    prepare_arrays,
    reconcile_to_monthly_reference,
    simulate,
)
from strategies.stage06_vkospi.vkospi_extended_diagnostics import (
    DOMESTIC_FEATURES,
    OAP_COMPOSITES,
)
from strategies.stage06_vkospi.vkospi_model_robustness import (
    apply_factor_tilt,
    fit_logistic_candidate,
    make_tail_factor,
)
from strategies.stage06_vkospi.vkospi_robust_dynamic_experiment import (
    align_features_to_arrays,
    build_robust_daily_features,
    stress_from_features,
)
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    FOREIGN_WEIGHT_CHANGE_COST,
    DOMESTIC_TRADE_COST,
    build_macro_probabilities,
    causal_expanding_percentile,
    load_vkospi_daily as load_zero_vkospi_daily,
    macro_asset_weights,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FULL_START = pd.Period("2007-04", freq="M")
LOCKED_START = pd.Period("2018-01", freq="M")


@dataclass(frozen=True)
class Component:
    key: str
    zero: str
    robust: str


COMPONENTS = (
    Component(
        "macro_engine",
        "거시수준 expanding rank와 무평활 확률",
        "rolling z-score·3개월 변화·sigmoid·85/15 평활",
    ),
    Component(
        "base_allocation",
        "사분면 확률을 네 자산 비중으로 직접 사용",
        "hard 국면 40% + SLSQP 위험제어 60%",
    ),
    Component(
        "tail_logistic",
        "꼬리위험 모델과 자산 tilt 없음",
        "16변수 균형 L2 로지스틱과 최대 20% tilt",
    ),
    Component(
        "vol_target_leverage",
        "총 자산노출 1배 고정",
        "15% 변동성 타깃, 0.5~1.5배 노출",
    ),
    Component(
        "vkospi_signal",
        "VKOSPI expanding percentile",
        "252일 수준·5일 충격·가속도 robust stress",
    ),
    Component(
        "overlay_policy",
        "stress 비율 전부를 채권·금에 균등 이전, 매일 조정",
        "최대 35%를 금으로 이전, 20% 밴드",
    ),
    Component(
        "frequency_reconciliation",
        "일간 재구성 수익률을 그대로 사용",
        "월간 기준경로에 일간 overlay 상대효과만 결합",
    ),
)
COMPONENT_KEYS = tuple(component.key for component in COMPONENTS)


def zero_macro_signals(probabilities: pd.DataFrame) -> pd.DataFrame:
    signals = probabilities.copy()
    regime_columns = [
        "p_Goldilocks",
        "p_Overheating",
        "p_Slowdown",
        "p_Stagflation",
    ]
    signals["regime"] = (
        signals[regime_columns].idxmax(axis=1).str.removeprefix("p_")
    )
    return signals


def current_unlevered_base(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return current 40% hard + 60% SLSQP weights before tail and leverage."""
    defensive = run_backtest(
        returns, signals, StrategyConfig(), mode="proposed"
    )
    months = signals.index.intersection(returns.index).intersection(defensive.index)
    rows: list[dict[str, float | pd.Period]] = []
    for month in months:
        hard = hard_regime_weights(signals.loc[month])
        slsqp = defensive.loc[
            month, [f"w_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        weights = 0.40 * hard + 0.60 * slsqp
        rows.append(
            {
                "month": month,
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
    base = pd.DataFrame(rows).set_index("month")
    base.index = pd.PeriodIndex(base.index, freq="M")
    return base, defensive


def zero_unlevered_base(probabilities: pd.DataFrame) -> pd.DataFrame:
    return macro_asset_weights(probabilities)


def run_fixed_leverage_label_path(
    returns: pd.DataFrame,
    base_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 1.2x path on which the current two-month tail label is defined."""
    months = returns.index.intersection(base_weights.index)
    rows: list[dict[str, float | pd.Period]] = []
    pretrade = np.zeros(len(ASSETS))
    first_trade = True
    nav = 1.0
    peak = 1.0
    for month in months:
        unlevered = base_weights.loc[
            month, [f"w_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        weights = 1.20 * unlevered
        debt_weight = -0.20
        delta = weights - pretrade
        turnover = (
            float(np.abs(delta).sum())
            if first_trade
            else 0.5 * float(np.abs(delta).sum())
        )
        trade_cost = float(np.abs(delta).sum()) * DOMESTIC_TRADE_COST
        fx_cost = (
            abs(
                (weights[2] + weights[3])
                - (pretrade[2] + pretrade[3])
            )
            * FOREIGN_WEIGHT_CHANGE_COST
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
                "turnover": turnover,
            }
        )
    return pd.DataFrame(rows).set_index("month")


def build_tail_factor(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    base_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit the current tail model conditional on the selected macro/base modules."""
    domestic = build_domestic_features(signals, returns)
    composites = pd.read_csv(
        ROOT / "results" / "openassetpricing_composites.csv",
        index_col=0,
    )
    composites.index = pd.PeriodIndex(composites.index, freq="M")
    label_path = run_fixed_leverage_label_path(returns, base_weights)
    data = domestic[DOMESTIC_FEATURES].join(
        composites[OAP_COMPOSITES], how="left"
    )
    data = data.loc[data.index.intersection(label_path.index)].copy()
    path_loss = forward_path_loss(
        label_path.loc[data.index, "return"], horizon=2
    )
    data["tail_event"] = (path_loss < -0.05).where(
        path_loss.notna()
    ).astype(float)
    probability, fit_stats = fit_logistic_candidate(
        data, balanced_logistic_spec()
    )
    return make_tail_factor(probability, data["tail_event"]), fit_stats


def run_monthly_path(
    returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    factor: pd.DataFrame | None,
    use_vol_target: bool,
) -> pd.DataFrame:
    """Crossable monthly engine matching the current medium path at the robust corner."""
    months = returns.index.intersection(base_weights.index)
    rows: list[dict[str, float | pd.Period]] = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    for month in months:
        base = base_weights.loc[
            month, [f"w_{asset}" for asset in ASSETS]
        ].to_numpy(dtype=float)
        if factor is None:
            probability = 0.5
            score = 0.0
            unlevered = base
        else:
            probability = (
                float(factor.loc[month, "p_up"])
                if month in factor.index
                else 0.5
            )
            score = float(np.clip((probability - 0.5) / 0.15, -1, 1))
            unlevered = apply_factor_tilt(base, score, max_shift=0.20)

        history = returns.loc[returns.index < month, ASSETS].tail(24)
        if use_vol_target and len(history) >= 12:
            conditional = history.to_numpy(dtype=float) @ unlevered
            observation_weights = np.exp(
                np.linspace(-2.0, 0.0, len(conditional))
            )
            observation_weights /= observation_weights.sum()
            mean = float(observation_weights @ conditional)
            variance = float(
                observation_weights @ (conditional - mean) ** 2
            )
            forecast_vol = math.sqrt(max(variance, 1e-8) * 12)
            leverage = float(np.clip(0.15 / forecast_vol, 0.50, 1.50))
        elif use_vol_target:
            forecast_vol = np.nan
            leverage = 1.20
        else:
            forecast_vol = np.nan
            leverage = 1.0

        weights = leverage * unlevered
        debt_weight = 1.0 - leverage
        delta = weights - pretrade
        turnover = (
            float(np.abs(delta).sum())
            if first_trade
            else 0.5 * float(np.abs(delta).sum())
        )
        trade_cost = float(np.abs(delta).sum()) * DOMESTIC_TRADE_COST
        fx_cost = (
            abs(
                (weights[2] + weights[3])
                - (pretrade[2] + pretrade[3])
            )
            * FOREIGN_WEIGHT_CHANGE_COST
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
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "forecast_vol": forecast_vol,
                "leverage": leverage,
                "risk_score": score,
                **{
                    f"w_{asset}": float(weights[index])
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
    return pd.DataFrame(rows).set_index("month")


def zero_vkospi_stress(
    arrays: dict[str, object],
    daily: pd.DataFrame,
) -> np.ndarray:
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])
    return (
        daily["stress"]
        .reindex(signal_dates)
        .to_numpy(dtype=float)
    )


def current_vkospi_stress(
    arrays: dict[str, object],
    robust_features: pd.DataFrame,
) -> np.ndarray:
    aligned = align_features_to_arrays(
        robust_features, arrays
    )
    return stress_from_features(
        aligned,
        mode="acceleration",
        level_threshold=0.90,
        shock_threshold=1.00,
    )


def simulate_zero_policy(
    arrays: dict[str, object],
    stress: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the exact Zero-Tune daily transfer and always-rebalance policy."""
    dates = pd.DatetimeIndex(arrays["dates"])
    months = pd.PeriodIndex(arrays["months"], freq="M")
    asset_returns = np.asarray(arrays["returns"], dtype=float)
    base_weights = np.asarray(arrays["base_weights"], dtype=float)
    signal_dates = pd.DatetimeIndex(arrays["signal_dates"])
    stress = np.nan_to_num(
        np.asarray(stress, dtype=float), nan=0.0, posinf=1.0, neginf=0.0
    ).clip(0, 1)
    rows: list[dict[str, object]] = []
    pretrade = np.zeros(len(ASSETS))
    nav = 1.0
    peak = 1.0
    first_trade = True
    rf_daily = (1 + 0.04) ** (1 / 252) - 1

    for position, date in enumerate(dates):
        base = base_weights[position].copy()
        severity = float(stress[position])
        removed = base[[0, 3]] * severity
        desired = base.copy()
        desired[[0, 3]] -= removed
        desired[[1, 2]] += removed.sum() / 2

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
        debt_weight = 1.0 - float(desired.sum())
        gross_return = float(
            desired @ asset_returns[position] + debt_weight * rf_daily
        )
        net_return = gross_return - trade_cost - fx_cost
        nav *= 1 + net_return
        peak = max(peak, nav)
        pretrade = (
            desired * (1 + asset_returns[position]) / (1 + gross_return)
        )
        rows.append(
            {
                "date": date,
                "month": months[position],
                "return": net_return,
                "gross_return": gross_return,
                "nav": nav,
                "drawdown": nav / peak - 1,
                "turnover": turnover,
                "trade_cost": trade_cost,
                "fx_cost": fx_cost,
                "stress": severity,
                "transfer_fraction": severity,
                "signal_date": signal_dates[position],
                **{
                    f"w_{asset}": float(desired[index])
                    for index, asset in enumerate(ASSETS)
                },
            }
        )
        first_trade = False

    daily = pd.DataFrame(rows).set_index("date")
    monthly = daily.groupby("month").agg(
        return_factor=("return", lambda values: float(np.prod(1 + values))),
        gross_factor=(
            "gross_return", lambda values: float(np.prod(1 + values))
        ),
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
    return daily, monthly


def simulate_current_policy(
    arrays: dict[str, object],
    stress: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = DynamicRiskConfig(
        mode="level",
        level_threshold=0.90,
        momentum_window=5,
        spike_threshold=0.0,
        max_risk_transfer=0.35,
        bond_share=0.0,
        rebalance_band=0.20,
        financing_rate=0.04,
    )
    return simulate(
        arrays,
        config,
        keep_daily=True,
        stress_override=stress,
    )


def combination_id(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def combination_name(bits: tuple[int, ...]) -> str:
    enabled = [
        COMPONENT_KEYS[index] for index, bit in enumerate(bits) if bit
    ]
    return "ZeroTune" if not enabled else "+".join(enabled)


def summarize_path(path: pd.DataFrame) -> dict[str, float]:
    metrics = performance_summary(path["return"])
    return {name: float(value) for name, value in metrics.items()}


def evaluate_periods(path: pd.DataFrame) -> dict[str, float]:
    full = path.loc[FULL_START:]
    locked = path.loc[LOCKED_START:]
    output: dict[str, float] = {}
    for prefix, view in [("Full", full), ("Locked", locked)]:
        metrics = summarize_path(view)
        output.update(
            {
                f"{prefix}_Months": metrics["Months"],
                f"{prefix}_CAGR": metrics["CAGR"],
                f"{prefix}_Sharpe": metrics["Sharpe"],
                f"{prefix}_MDD": metrics["MDD"],
                f"{prefix}_Calmar": metrics["Calmar"],
                f"{prefix}_FinalMultiple": metrics["FinalMultiple"],
            }
        )
    return output


def prepare_research_inputs() -> dict[str, object]:
    returns, _ = load_monthly_asset_returns(False)
    zero_probabilities, _ = build_macro_probabilities(returns)
    zero_signals = zero_macro_signals(zero_probabilities)
    current_signals, current_probabilities = build_no_sjm_signals(returns)

    signals = {0: zero_signals, 1: current_signals}
    probabilities = {
        0: zero_probabilities,
        1: current_probabilities,
    }
    bases: dict[tuple[int, int], pd.DataFrame] = {}
    defensives: dict[tuple[int, int], pd.DataFrame | None] = {}
    for macro_bit in (0, 1):
        bases[(macro_bit, 0)] = zero_unlevered_base(
            signals[macro_bit]
        )
        defensives[(macro_bit, 0)] = None
        current_base, defensive = current_unlevered_base(
            returns, signals[macro_bit]
        )
        bases[(macro_bit, 1)] = current_base
        defensives[(macro_bit, 1)] = defensive

    factors: dict[tuple[int, int], pd.DataFrame] = {}
    factor_fit_stats: dict[str, dict[str, float]] = {}
    for macro_bit, base_bit in itertools.product((0, 1), repeat=2):
        factor, fit_stats = build_tail_factor(
            returns,
            signals[macro_bit],
            bases[(macro_bit, base_bit)],
        )
        factors[(macro_bit, base_bit)] = factor
        factor_fit_stats[f"{macro_bit}{base_bit}"] = fit_stats

    levels = load_daily_open_levels()
    vkospi_signal_frame = build_daily_vkospi_signals()
    zero_vkospi = load_zero_vkospi_daily()
    robust_vkospi_features = build_robust_daily_features()
    return {
        "returns": returns,
        "signals": signals,
        "probabilities": probabilities,
        "bases": bases,
        "defensives": defensives,
        "factors": factors,
        "factor_fit_stats": factor_fit_stats,
        "levels": levels,
        "vkospi_signal_frame": vkospi_signal_frame,
        "zero_vkospi": zero_vkospi,
        "robust_vkospi_features": robust_vkospi_features,
    }


def run_all_combinations() -> dict[str, object]:
    inputs = prepare_research_inputs()
    returns = inputs["returns"]
    bases = inputs["bases"]
    factors = inputs["factors"]
    levels = inputs["levels"]
    vkospi_signal_frame = inputs["vkospi_signal_frame"]
    zero_vkospi = inputs["zero_vkospi"]
    robust_vkospi_features = inputs["robust_vkospi_features"]

    monthly_paths: dict[tuple[int, int, int, int], pd.DataFrame] = {}
    for macro_bit, base_bit, tail_bit, vol_bit in itertools.product(
        (0, 1), repeat=4
    ):
        monthly_paths[(macro_bit, base_bit, tail_bit, vol_bit)] = (
            run_monthly_path(
                returns,
                bases[(macro_bit, base_bit)],
                factors[(macro_bit, base_bit)] if tail_bit else None,
                use_vol_target=bool(vol_bit),
            )
        )

    rows: list[dict[str, object]] = []
    return_columns: dict[str, pd.Series] = {}
    paths: dict[str, pd.DataFrame] = {}
    for monthly_key, reference in monthly_paths.items():
        macro_bit, base_bit, tail_bit, vol_bit = monthly_key
        arrays = prepare_arrays(levels, reference, vkospi_signal_frame)
        stress_by_signal = {
            0: zero_vkospi_stress(arrays, zero_vkospi),
            1: current_vkospi_stress(arrays, robust_vkospi_features),
        }
        _, neutral_monthly = simulate(arrays, None, keep_daily=False)
        overlay_cache: dict[tuple[int, int], pd.DataFrame] = {}
        for signal_bit, policy_bit in itertools.product((0, 1), repeat=2):
            stress = stress_by_signal[signal_bit]
            if policy_bit:
                _, overlay_monthly = simulate_current_policy(arrays, stress)
            else:
                _, overlay_monthly = simulate_zero_policy(arrays, stress)
            overlay_cache[(signal_bit, policy_bit)] = overlay_monthly

        for signal_bit, policy_bit, reconciliation_bit in itertools.product(
            (0, 1), repeat=3
        ):
            overlay_monthly = overlay_cache[(signal_bit, policy_bit)]
            final = (
                reconcile_to_monthly_reference(
                    reference, neutral_monthly, overlay_monthly
                )
                if reconciliation_bit
                else overlay_monthly
            )
            bits = (
                macro_bit,
                base_bit,
                tail_bit,
                vol_bit,
                signal_bit,
                policy_bit,
                reconciliation_bit,
            )
            identifier = combination_id(bits)
            paths[identifier] = final
            return_columns[identifier] = final["return"]
            row: dict[str, object] = {
                "Combination": identifier,
                "Name": combination_name(bits),
                "RobustComponents": int(sum(bits)),
                **{
                    COMPONENT_KEYS[index]: int(bit)
                    for index, bit in enumerate(bits)
                },
                **evaluate_periods(final),
            }
            rows.append(row)

    combinations = pd.DataFrame(rows).sort_values("Combination")
    returns_matrix = pd.concat(return_columns, axis=1).sort_index()
    return {
        **inputs,
        "monthly_paths": monthly_paths,
        "paths": paths,
        "combinations": combinations,
        "returns_matrix": returns_matrix,
    }


def shapley_attribution(
    combinations: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Allocate the endpoint difference over all 7! possible transition orders."""
    count = len(COMPONENTS)
    indexed = combinations.set_index("Combination")
    rows: list[dict[str, object]] = []
    for metric in metrics:
        values = indexed[metric].to_dict()
        contributions: dict[str, float] = {}
        for component_index, component in enumerate(COMPONENTS):
            contribution = 0.0
            other_indices = [
                index for index in range(count) if index != component_index
            ]
            for subset_size in range(count):
                weight = (
                    math.factorial(subset_size)
                    * math.factorial(count - subset_size - 1)
                    / math.factorial(count)
                )
                for subset in itertools.combinations(
                    other_indices, subset_size
                ):
                    off = [0] * count
                    for index in subset:
                        off[index] = 1
                    on = off.copy()
                    on[component_index] = 1
                    contribution += weight * (
                        values[combination_id(tuple(on))]
                        - values[combination_id(tuple(off))]
                    )
            contributions[component.key] = contribution

        endpoint_delta = (
            float(values["1" * count]) - float(values["0" * count])
        )
        for component in COMPONENTS:
            rows.append(
                {
                    "Metric": metric,
                    "Component": component.key,
                    "ShapleyContribution": contributions[component.key],
                    "EndpointDelta": endpoint_delta,
                    "ContributionShare": (
                        contributions[component.key] / endpoint_delta
                        if not math.isclose(endpoint_delta, 0.0)
                        else np.nan
                    ),
                }
            )
        assert math.isclose(
            sum(contributions.values()),
            endpoint_delta,
            abs_tol=1e-11,
        )
    return pd.DataFrame(rows)


def factorial_main_effects(
    combinations: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for component in COMPONENTS:
            enabled = combinations.loc[
                combinations[component.key].eq(1), metric
            ]
            disabled = combinations.loc[
                combinations[component.key].eq(0), metric
            ]
            rows.append(
                {
                    "Metric": metric,
                    "Component": component.key,
                    "MeanWhenRobust": float(enabled.mean()),
                    "MeanWhenZero": float(disabled.mean()),
                    "AverageMainEffect": float(
                        enabled.mean() - disabled.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def pairwise_interactions(
    combinations: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Average difference-in-differences over every setting of other modules."""
    indexed = combinations.set_index("Combination")
    count = len(COMPONENTS)
    rows: list[dict[str, object]] = []
    for metric in metrics:
        values = indexed[metric].to_dict()
        for first, second in itertools.combinations(range(count), 2):
            effects: list[float] = []
            other_indices = [
                index
                for index in range(count)
                if index not in {first, second}
            ]
            for other_bits in itertools.product(
                (0, 1), repeat=len(other_indices)
            ):
                base = [0] * count
                for index, bit in zip(other_indices, other_bits):
                    base[index] = bit
                key00 = base.copy()
                key10 = base.copy()
                key10[first] = 1
                key01 = base.copy()
                key01[second] = 1
                key11 = base.copy()
                key11[first] = 1
                key11[second] = 1
                effects.append(
                    values[combination_id(tuple(key11))]
                    - values[combination_id(tuple(key10))]
                    - values[combination_id(tuple(key01))]
                    + values[combination_id(tuple(key00))]
                )
            rows.append(
                {
                    "Metric": metric,
                    "FirstComponent": COMPONENTS[first].key,
                    "SecondComponent": COMPONENTS[second].key,
                    "AverageInteraction": float(np.mean(effects)),
                    "MeanAbsoluteInteraction": float(
                        np.mean(np.abs(effects))
                    ),
                }
            )
    return pd.DataFrame(rows)


def transition_tables(
    combinations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = combinations.set_index("Combination")
    ordered_rows: list[dict[str, object]] = []
    bits = [0] * len(COMPONENTS)
    previous: pd.Series | None = None
    for step in range(len(COMPONENTS) + 1):
        identifier = combination_id(tuple(bits))
        current = indexed.loc[identifier]
        changed = "Zero-Tune start" if step == 0 else COMPONENTS[step - 1].key
        record = {
            "Step": step,
            "ChangedComponent": changed,
            "Combination": identifier,
            **{
                metric: float(current[metric])
                for metric in [
                    "Full_CAGR",
                    "Full_Sharpe",
                    "Full_MDD",
                    "Locked_CAGR",
                    "Locked_Sharpe",
                    "Locked_MDD",
                ]
            },
        }
        if previous is not None:
            for metric in [
                "Full_CAGR",
                "Full_Sharpe",
                "Full_MDD",
                "Locked_CAGR",
                "Locked_Sharpe",
                "Locked_MDD",
            ]:
                record[f"Delta_{metric}"] = float(
                    current[metric] - previous[metric]
                )
        ordered_rows.append(record)
        previous = current
        if step < len(COMPONENTS):
            bits[step] = 1

    local_rows: list[dict[str, object]] = []
    zero = [0] * len(COMPONENTS)
    robust = [1] * len(COMPONENTS)
    for index, component in enumerate(COMPONENTS):
        one_on = zero.copy()
        one_on[index] = 1
        one_off = robust.copy()
        one_off[index] = 0
        for comparison, start_bits, end_bits in [
            ("OneAtATimeFromZero", zero, one_on),
            ("LeaveOneOutFromRobust", one_off, robust),
        ]:
            start = indexed.loc[combination_id(tuple(start_bits))]
            end = indexed.loc[combination_id(tuple(end_bits))]
            local_rows.append(
                {
                    "Comparison": comparison,
                    "Component": component.key,
                    **{
                        f"Delta_{metric}": float(
                            end[metric] - start[metric]
                        )
                        for metric in [
                            "Full_CAGR",
                            "Full_Sharpe",
                            "Full_MDD",
                            "Locked_CAGR",
                            "Locked_Sharpe",
                            "Locked_MDD",
                        ]
                    },
                }
            )
    return pd.DataFrame(ordered_rows), pd.DataFrame(local_rows)


def endpoint_audit(result: dict[str, object]) -> dict[str, float | int]:
    zero_expected = pd.read_csv(
        ROOT
        / "strategies"
        / "stage07_zero_tune_vkospi"
        / "outputs"
        / "zero_tune_monthly.csv",
        index_col=0,
    )
    current_expected = pd.read_csv(
        ROOT
        / "results"
        / "balanced_logistic_no_sjm_final_reconciled.csv",
        index_col=0,
    )
    medium_expected = pd.read_csv(
        ROOT
        / "results"
        / "balanced_logistic_no_sjm_medium_backtest.csv",
        index_col=0,
    )
    for frame in (zero_expected, current_expected, medium_expected):
        frame.index = pd.PeriodIndex(frame.index, freq="M")

    zero_actual = result["paths"]["0" * len(COMPONENTS)]
    current_actual = result["paths"]["1" * len(COMPONENTS)]
    medium_actual = result["monthly_paths"][(1, 1, 1, 1)]
    zero_common = zero_actual.index.intersection(zero_expected.index)
    current_common = current_actual.index.intersection(current_expected.index)
    medium_common = medium_actual.index.intersection(medium_expected.index)
    audit = {
        "combination_count": int(len(result["combinations"])),
        "zero_endpoint_months": int(len(zero_common)),
        "zero_endpoint_max_return_difference": float(
            (
                zero_actual.loc[zero_common, "return"]
                - zero_expected.loc[zero_common, "return"]
            ).abs().max()
        ),
        "current_medium_months": int(len(medium_common)),
        "current_medium_max_return_difference": float(
            (
                medium_actual.loc[medium_common, "return"]
                - medium_expected.loc[medium_common, "return"]
            ).abs().max()
        ),
        "current_endpoint_months": int(len(current_common)),
        "current_endpoint_max_return_difference": float(
            (
                current_actual.loc[current_common, "return"]
                - current_expected.loc[current_common, "return"]
            ).abs().max()
        ),
    }
    assert audit["combination_count"] == 2 ** len(COMPONENTS)
    assert audit["zero_endpoint_months"] > 0
    assert audit["current_endpoint_months"] > 0
    assert audit["zero_endpoint_max_return_difference"] < 1e-12
    assert audit["current_medium_max_return_difference"] < 1e-12
    assert audit["current_endpoint_max_return_difference"] < 1e-12
    return audit


def run_factorial_research(save: bool = True) -> dict[str, object]:
    result = run_all_combinations()
    combinations = result["combinations"]
    metrics = [
        "Full_CAGR",
        "Full_Sharpe",
        "Full_MDD",
        "Locked_CAGR",
        "Locked_Sharpe",
        "Locked_MDD",
    ]
    shapley = shapley_attribution(combinations, metrics)
    main_effects = factorial_main_effects(combinations, metrics)
    interactions = pairwise_interactions(combinations, metrics)
    ordered, local = transition_tables(combinations)
    audit = endpoint_audit(result)

    zero = combinations.loc[
        combinations["Combination"].eq("0" * len(COMPONENTS))
    ].iloc[0]
    current = combinations.loc[
        combinations["Combination"].eq("1" * len(COMPONENTS))
    ].iloc[0]
    report = {
        "objective": (
            "Evaluate every binary combination on the path from Zero-Tune "
            "VKOSPI to the current Robust VKOSPI."
        ),
        "component_count": len(COMPONENTS),
        "combination_count": 2 ** len(COMPONENTS),
        "components": [asdict(component) for component in COMPONENTS],
        "periods": {
            "full": "2007-04 through 2026-07",
            "locked": "2018-01 through 2026-07",
        },
        "endpoint_audit": audit,
        "zero_endpoint": {
            metric: float(zero[metric]) for metric in metrics
        },
        "current_endpoint": {
            metric: float(current[metric]) for metric in metrics
        },
        "endpoint_delta": {
            metric: float(current[metric] - zero[metric])
            for metric in metrics
        },
        "interpretation": {
            "ordered_transition": (
                "Path-dependent change when modules are enabled in the listed "
                "economic pipeline order."
            ),
            "shapley": (
                "Order-neutral average marginal contribution over all 7! "
                "possible module orders."
            ),
            "main_effect": (
                "Difference between the metric mean over 64 combinations with "
                "a component on and the 64 combinations with it off."
            ),
            "interaction": (
                "Average pairwise difference-in-differences over all settings "
                "of the other five components."
            ),
        },
        "warning": (
            "The 128-combination table is an attribution and sensitivity audit, "
            "not a new model-selection search. Ranking combinations on the same "
            "sample would create additional selection bias."
        ),
    }
    result.update(
        {
            "shapley": shapley,
            "main_effects": main_effects,
            "interactions": interactions,
            "ordered_transition": ordered,
            "local_effects": local,
            "audit": audit,
            "report": report,
        }
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(component) for component in COMPONENTS]).to_csv(
            OUTPUT_DIR / "component_catalog.csv", index=False
        )
        combinations.to_csv(
            OUTPUT_DIR / "all_128_combinations.csv", index=False
        )
        result["returns_matrix"].to_csv(
            OUTPUT_DIR / "all_128_monthly_returns.csv"
        )
        shapley.to_csv(OUTPUT_DIR / "shapley_attribution.csv", index=False)
        main_effects.to_csv(
            OUTPUT_DIR / "factorial_main_effects.csv", index=False
        )
        interactions.to_csv(
            OUTPUT_DIR / "pairwise_interactions.csv", index=False
        )
        ordered.to_csv(OUTPUT_DIR / "ordered_transition.csv", index=False)
        local.to_csv(OUTPUT_DIR / "local_component_effects.csv", index=False)
        (OUTPUT_DIR / "factorial_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    result = run_factorial_research(save=True)
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))
    print(result["ordered_transition"].to_string(index=False))
    print(
        result["shapley"]
        .loc[result["shapley"]["Metric"].isin(["Full_CAGR", "Full_Sharpe", "Full_MDD"])]
        .to_string(index=False)
    )
    print("saved", OUTPUT_DIR)


if __name__ == "__main__":
    main()
