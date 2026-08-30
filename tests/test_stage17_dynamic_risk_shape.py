from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage17_dynamic_risk_shape.dynamic_risk_shape_slsqp import (
    OUTPUT_DIR,
    POLICY_17A,
    POLICY_17B,
    POLICY_17C,
    run_research,
)


@pytest.fixture(scope="module")
def research() -> dict:
    return run_research(save=False)


def test_only_three_frozen_economic_hypotheses_are_candidates(
    research: dict,
) -> None:
    assert set(research["candidate_paths"]) == {
        POLICY_17A.name,
        POLICY_17B.name,
        POLICY_17C.name,
    }


def test_stage16_evidence_is_reused_without_refit(research: dict) -> None:
    frozen = pd.read_csv(
        "strategies/stage16_confirmed_crash_risk/outputs/confirmed_crash_signals.csv",
        index_col=0,
    )
    frozen.index = pd.PeriodIndex(frozen.index, freq="M")
    signals = research["signals"]
    common = signals.index.intersection(frozen.index)
    assert np.allclose(
        signals.loc[common, "crash_evidence"],
        frozen.loc[common, "crash_evidence"],
    )
    assert np.allclose(
        signals.loc[common, "crash_pressure"],
        frozen.loc[common, "crash_pressure"],
    )


@pytest.mark.parametrize("policy", [POLICY_17A, POLICY_17B, POLICY_17C])
def test_candidate_is_causal_fully_invested_and_has_fixed_lambda(
    research: dict,
    policy,
) -> None:
    path = research["candidate_paths"][policy.name]
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert (path["macro_signal_month"] < path.index).all()
    assert (path["stress_signal_month"] < path.index).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-10).all().all()
    assert (weights <= 1.0 + 1e-10).all().all()
    assert np.allclose(path["downside_risk_aversion_lambda"], 1.0)
    assert (
        path["portfolio_crash_pressure"]
        <= path["market_crash_pressure"] + 1e-12
    ).all()
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()
    assert (path["volatility_slack"] >= -1e-7).all()
    assert (path["cdar_slack"] >= -1e-7).all()


def test_tail_candidates_use_literal_one_plus_q_es_squared(
    research: dict,
) -> None:
    for policy in [POLICY_17B, POLICY_17C]:
        path = research["candidate_paths"][policy.name]
        assert np.allclose(
            path["tail_multiplier"],
            1.0 + path["portfolio_crash_pressure"],
        )
        assert np.allclose(
            path["tail_penalty"],
            path["tail_multiplier"] * path["expected_shortfall"] ** 2,
        )


def test_performance_tradeoff_and_gates_are_reported(research: dict) -> None:
    comparison = research["comparison"].set_index(["Strategy", "Period"])
    a = comparison.loc[(POLICY_17A.name, "full_2007_2026")]
    b = comparison.loc[(POLICY_17B.name, "full_2007_2026")]
    c = comparison.loc[(POLICY_17C.name, "full_2007_2026")]
    assert a["CAGR"] == pytest.approx(0.110400, abs=1e-6)
    assert a["MDD"] == pytest.approx(-0.233914, abs=1e-6)
    assert b["CAGR"] == pytest.approx(0.093160, abs=1e-6)
    assert b["Sharpe"] == pytest.approx(0.855814, abs=1e-6)
    assert b["MDD"] == pytest.approx(-0.170066, abs=1e-6)
    assert c["CAGR"] == pytest.approx(0.093151, abs=1e-6)
    assert c["MDD"] == pytest.approx(-0.174638, abs=1e-6)
    gates = research["report"]["success_gates"]
    assert gates[POLICY_17A.name]["cagr_at_least_10_5_percent"]
    assert not gates[POLICY_17A.name]["mdd_no_worse_than_18_percent"]
    assert not gates[POLICY_17B.name]["cagr_at_least_10_5_percent"]
    assert gates[POLICY_17B.name]["mdd_no_worse_than_18_percent"]


def test_no_internal_cost_path_is_diagnostic_only_and_pays_real_cost(
    research: dict,
) -> None:
    path = research["no_internal_cost_path"]
    assert np.allclose(path["estimated_transaction_cost"], 0.0)
    assert float(path[["trade_cost", "fx_cost"]].sum().sum()) > 0.04
    assert float(path["turnover"].mean()) > float(
        research["candidate_paths"][POLICY_17C.name]["turnover"].mean()
    )


def test_drawdown_episode_identifies_gold_as_stage14_max_dd_driver(
    research: dict,
) -> None:
    episodes = research["episodes"]
    assert set(episodes["Strategy"]) == {
        "Stage14_StaticLambda",
        POLICY_17A.name,
        POLICY_17B.name,
        POLICY_17C.name,
    }
    static = episodes[episodes["Strategy"] == "Stage14_StaticLambda"].iloc[0]
    assert static["EpisodeMDD"] == pytest.approx(-0.231921, abs=1e-6)
    assert static["GLD_Contribution"] < -0.20
    assert abs(static["KODEX200_Contribution"]) < 0.03


def test_saved_report_proves_no_search_or_future_label() -> None:
    report = json.loads(
        (OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert all(report["checks"].values())
    assert report["parameter_policy"]["searched_parameters"] is None
    assert report["parameter_policy"]["candidate_count"] == 3
