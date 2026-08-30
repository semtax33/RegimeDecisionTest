from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.core.regime_research import ASSETS, load_monthly_asset_returns
from strategies.stage07_zero_tune_vkospi.zero_tune_strategy import (
    build_macro_probabilities,
)
from strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp import (
    FULL_START,
    LOCKED_START,
    build_daily_stress_features,
    build_monthly_stress_signals,
)
from strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp import (
    concentration_summary,
    metric_row,
    solver_summary,
)
from strategies.stage17_dynamic_risk_shape.dynamic_risk_shape_slsqp import (
    drawdown_episodes,
)
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20,
)
from strategies.stage28_option_directional_surface import (
    option_directional_surface_slsqp as stage28,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def apply_two_axis_confirmation(daily: pd.DataFrame) -> pd.DataFrame:
    """Implement the supplied four-cell Bear Pressure / ERP interpretation.

    Positive standardized bear pressure with non-positive ERP is bearish.
    Non-positive bear pressure with positive ERP is bullish.  High/high and
    low/low states are neutral.  Zero is the economically natural standardized
    boundary, not a fitted cutoff.
    """

    output = daily.copy()
    for horizon in ["fast", "slow"]:
        bear = output[f"bear_pressure_{horizon}"]
        compensation = output[f"z_option_erp_{horizon}"]
        bullish = (bear <= 0.0) & (compensation > 0.0)
        bearish = (bear > 0.0) & (compensation <= 0.0)
        direction = pd.Series(0.0, index=output.index)
        direction.loc[bullish] = compensation.loc[bullish]
        direction.loc[bearish] = -bear.loc[bearish]
        output[f"two_axis_state_{horizon}"] = np.select(
            [bullish, bearish, (bear > 0.0) & (compensation > 0.0)],
            ["bullish", "bearish", "fear_with_compensation"],
            default="uninformative",
        )
        output[f"option_direction_{horizon}"] = direction
    output["option_direction"] = output[
        ["option_direction_fast", "option_direction_slow"]
    ].mean(axis=1)
    output["option_direction_score"] = output["option_direction"] / (
        1.0 + output["option_direction"].abs()
    )
    return output


def _changes(left: pd.Series, right: pd.Series) -> dict[str, float]:
    return {
        "cagr": float(left["CAGR"] - right["CAGR"]),
        "sharpe": float(left["Sharpe"] - right["Sharpe"]),
        "mdd": float(left["MDD"] - right["MDD"]),
        "volatility": float(left["Volatility"] - right["Volatility"]),
    }


def run_research(save: bool = True) -> dict[str, Any]:
    returns, _ = load_monthly_asset_returns(False)
    probabilities, _ = build_macro_probabilities(returns)
    original_daily_stress = build_daily_stress_features()
    original_stress = build_monthly_stress_signals(
        returns.index, original_daily_stress
    )
    vkospi_daily = stage28.build_daily_vkospi_only_stress_features()
    vkospi_stress = stage28.build_monthly_vkospi_only_stress_signals(
        returns.index, vkospi_daily
    )
    daily_technical, technical_audit = stage28.build_daily_technical_features()
    technical = stage28.build_monthly_technical_signals(
        returns.index, daily_technical
    )
    daily_scalar, option_audit = stage28.build_daily_option_direction_features()
    scalar_signals = stage28.build_monthly_option_direction_signals(
        returns.index, daily_scalar
    )
    daily_two_axis = apply_two_axis_confirmation(daily_scalar)
    two_axis_signals = stage28.build_monthly_option_direction_signals(
        returns.index, daily_two_axis
    )

    stage20_path = stage20.run_backtest(
        returns, probabilities, original_stress, technical
    )
    scalar_path = stage28.run_backtest(
        returns,
        probabilities,
        vkospi_stress,
        technical,
        scalar_signals,
        use_option_direction=True,
    )
    two_axis_path = stage28.run_backtest(
        returns,
        probabilities,
        vkospi_stress,
        technical,
        two_axis_signals,
        use_option_direction=True,
    )
    two_axis_path["policy"] = "OptionTwoAxisConfirmation_StaticLambda"
    paths = {
        "Stage20_VIX6Decomposition": stage20_path,
        "Stage28_ODS_Difference": scalar_path,
        "Stage29_OptionTwoAxisConfirmation": two_axis_path,
    }
    common_end = min(path.index.max() for path in paths.values())
    rows: list[dict[str, Any]] = []
    for period, start in [
        ("full_2007_2026", FULL_START),
        ("locked_2018_2026", LOCKED_START),
    ]:
        for name, path in paths.items():
            rows.append(metric_row(name, path, period, start, common_end))
    comparison = pd.DataFrame(rows)
    full = comparison.loc[comparison["Period"] == "full_2007_2026"].set_index(
        "Strategy"
    )
    baseline = full.loc["Stage20_VIX6Decomposition"]
    scalar = full.loc["Stage28_ODS_Difference"]
    candidate = full.loc["Stage29_OptionTwoAxisConfirmation"]
    quintiles, forward_summary = stage28.option_forward_diagnostics(
        daily_two_axis, two_axis_signals, returns
    )
    episodes = pd.concat(
        [
            drawdown_episodes(path, returns, name)
            for name, path in paths.items()
        ],
        ignore_index=True,
    )
    top_episodes = (
        episodes.sort_values(["Strategy", "EpisodeMDD"])
        .groupby("Strategy", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    weights = [f"w_{asset}" for asset in ASSETS]
    other_assets = [asset for asset in ASSETS if asset != "KODEX200"]
    state_counts = {
        horizon: daily_two_axis[f"two_axis_state_{horizon}"].value_counts().to_dict()
        for horizon in ["fast", "slow"]
    }
    checks = {
        "macro_signal_precedes_target": bool(
            (two_axis_path["macro_signal_month"] < two_axis_path.index).all()
        ),
        "vkospi_signal_precedes_target": bool(
            (two_axis_path["stress_signal_month"] < two_axis_path.index).all()
        ),
        "technical_signal_precedes_target": stage28.verify_technical_signal_dates(
            two_axis_path
        ),
        "option_signal_precedes_target": stage28.verify_option_signal_dates(
            two_axis_path
        ),
        "monthly_option_signals_match_daily": stage28.verify_monthly_option_signals(
            two_axis_signals, daily_two_axis
        ),
        "candidate_stress_is_vkospi_only": bool(
            not any("vix6" in column.lower() for column in vkospi_daily.columns)
        ),
        "two_axis_states_use_zero_not_fitted_cutoffs": True,
        "ods_changes_only_equity_expected_mu": bool(
            all(
                np.allclose(
                    two_axis_path[f"filtered_expected_mu_{asset}"],
                    scalar_path[f"filtered_expected_mu_{asset}"],
                )
                for asset in other_assets
            )
        ),
        "weights_sum_to_one": bool(
            np.allclose(two_axis_path[weights].sum(axis=1), 1.0)
        ),
        "weights_are_long_only": bool(
            (two_axis_path[weights] >= -1e-10).all().all()
        ),
        "no_leverage": bool(
            np.allclose(two_axis_path[weights].sum(axis=1), 1.0)
        ),
        "static_lambda_equals_one": bool(
            np.allclose(two_axis_path["downside_risk_aversion_lambda"], 1.0)
        ),
        "all_solvers_succeeded": bool(
            two_axis_path["solver_success"].all()
            and not two_axis_path["used_fallback"].any()
        ),
        "no_hard_asset_cap": True,
        "no_hard_regime_weights": True,
        "no_post_optimizer_overlay": True,
        "no_hyperparameter_search": True,
    }
    report = {
        "strategy": "Stage29_OptionTwoAxisConfirmation",
        "base_strategy": "Stage20_VIX6Decomposition",
        "reason_for_follow_up": (
            "Stage28 scalar ODS raised CAGR but its high-ERP states also had "
            "worse tail losses. Stage29 implements the supplied four-cell table "
            "literally so high bear pressure plus high ERP is neutral."
        ),
        "research_status": "exploratory follow-up after Stage28 result",
        "technical_audit": technical_audit,
        "option_audit": option_audit,
        "two_axis_formula": {
            "bullish": "bear<=0 and ERP_z>0 => +ERP_z",
            "bearish": "bear>0 and ERP_z<=0 => -bear",
            "fear_with_compensation": "bear>0 and ERP_z>0 => 0",
            "uninformative": "bear<=0 and ERP_z<=0 => 0",
            "fast_slow_combination": "equal mean",
            "bounded_score": "direction/(1+abs(direction))",
            "searched_parameters": None,
        },
        "state_counts": state_counts,
        "performance": json.loads(
            comparison.to_json(orient="records", force_ascii=False)
        ),
        "full_changes_vs_stage20": _changes(candidate, baseline),
        "full_changes_vs_stage28_scalar": _changes(candidate, scalar),
        "forward_diagnostics": forward_summary,
        "concentration": {
            name: concentration_summary(path) for name, path in paths.items()
        },
        "top_drawdown_episodes": json.loads(
            top_episodes.to_json(orient="records", force_ascii=False)
        ),
        "solver": solver_summary(two_axis_path),
        "checks": checks,
    }
    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stage20_path.to_csv(OUTPUT_DIR / "stage20_vix6_monthly.csv")
        scalar_path.to_csv(OUTPUT_DIR / "stage28_scalar_ods_monthly.csv")
        two_axis_path.to_csv(OUTPUT_DIR / "option_two_axis_monthly.csv")
        daily_two_axis.to_csv(OUTPUT_DIR / "daily_option_two_axis_features.csv")
        two_axis_signals.to_csv(OUTPUT_DIR / "monthly_option_two_axis_signals.csv")
        comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)
        quintiles.to_csv(OUTPUT_DIR / "option_forward_quintiles.csv", index=False)
        top_episodes.to_csv(
            OUTPUT_DIR / "drawdown_episode_attribution.csv", index=False
        )
        (OUTPUT_DIR / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return {
        "comparison": comparison,
        "two_axis_path": two_axis_path,
        "scalar_path": scalar_path,
        "daily_two_axis": daily_two_axis,
        "two_axis_signals": two_axis_signals,
        "quintiles": quintiles,
        "episodes": top_episodes,
        "report": report,
    }


def main() -> None:
    result = run_research(save=True)
    print(result["comparison"].to_string(index=False))
    print(result["quintiles"].to_string(index=False))
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
