from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage20_daily_technical_confidence import (
    daily_technical_confidence_slsqp as stage20,
)
from strategies.stage21_one_sided_confidence import (
    one_sided_confidence_slsqp as stage21,
)
from strategies.stage22_k_ratio_primary import k_ratio_primary_slsqp as stage22
from strategies.stage23_relative_atr import relative_atr_slsqp as stage23
from strategies.stage24_equity_k_ratio_only import (
    equity_k_ratio_only_slsqp as stage24,
)
from strategies.stage25_conflict_only_veto import conflict_only_veto_slsqp as stage25
from strategies.stage26_equity_conflict_veto import (
    equity_conflict_veto_slsqp as stage26,
)
from strategies.stage27_k_ratio_equity_veto import (
    k_ratio_equity_veto_slsqp as stage27,
)


MODULES = [stage21, stage22, stage23, stage24, stage25, stage26, stage27]


def _signal(direction: float = 1.0, atr: float = 0.5) -> pd.Series:
    return pd.Series(
        {
            **{f"technical_direction_{asset}": direction for asset in ASSETS},
            **{f"atr_percentile_{asset}": atr for asset in ASSETS},
        }
    )


def test_candidate_output_directories_are_isolated_from_stage20_and_each_other() -> None:
    output_dirs = [module.OUTPUT_DIR.resolve() for module in MODULES]
    assert len(set(output_dirs)) == len(output_dirs)
    assert all(path != stage20.OUTPUT_DIR.resolve() for path in output_dirs)
    assert all(path.parent.name.startswith("stage2") for path in output_dirs)


def test_stage21_never_raises_below_neutral_macro_forecast() -> None:
    macro = np.array([0.02, 0.01, 0.03, 0.00])
    detail = stage21.apply_technical_inputs(macro, np.eye(4), _signal(1.0))
    # Positive technical direction conflicts with the two below-neutral views.
    # Stage20 would lift them to neutral; Stage21 must retain the original view.
    assert np.allclose(
        detail["filtered_macro_expected_return"], macro
    )


def test_stage22_rsi_confirmation_cannot_reverse_k_ratio_sign() -> None:
    frame = pd.read_csv(
        stage22.OUTPUT_DIR / "daily_technical_features_KODEX200.csv"
    ).dropna(subset=["k_score", "rsi_confirmation", "technical_direction"])
    assert frame["rsi_confirmation"].between(0.0, 1.0).all()
    assert np.allclose(
        frame["technical_direction"],
        frame["k_score"] * frame["rsi_confirmation"],
    )
    nonzero = frame["technical_direction"].abs() > 1e-12
    assert np.array_equal(
        np.sign(frame.loc[nonzero, "technical_direction"]),
        np.sign(frame.loc[nonzero, "k_score"]),
    )


def test_stage23_relative_atr_scale_has_cross_sectional_mean_one() -> None:
    path = pd.read_csv(stage23.OUTPUT_DIR / "relative_atr_monthly.csv")
    scales = path[[f"atr_variance_scale_{asset}" for asset in ASSETS]]
    assert np.allclose(scales.mean(axis=1), 1.0)
    assert (scales > 0.0).all().all()


@pytest.mark.parametrize("module", [stage24, stage27])
def test_k_ratio_only_candidates_retain_rsi_but_use_k_score_direction(module) -> None:
    frame = pd.read_csv(
        module.OUTPUT_DIR / "daily_technical_features_KODEX200.csv"
    ).dropna(subset=["k_score", "technical_direction", "price_rsi", "volume_rsi"])
    assert np.allclose(frame["technical_direction"], frame["k_score"])
    assert frame["price_rsi"].between(0.0, 100.0).all()
    assert frame["volume_rsi"].between(0.0, 100.0).all()


def test_conflict_veto_confidence_mappings_are_as_declared() -> None:
    macro = np.array([0.02, 0.01, 0.03, 0.00])
    neutral_signal = _signal(0.0)
    all_assets = stage25.apply_technical_inputs(macro, np.eye(4), neutral_signal)
    equity_only = stage26.apply_technical_inputs(macro, np.eye(4), neutral_signal)
    assert np.allclose(all_assets["macro_confidence"], 1.0)
    assert equity_only["macro_confidence"][ASSETS.index("KODEX200")] == pytest.approx(1.0)
    for asset in ["BOND", "GLD", "USO"]:
        assert equity_only["macro_confidence"][ASSETS.index(asset)] == pytest.approx(0.5)


@pytest.mark.parametrize("module", MODULES)
def test_saved_candidate_paths_are_causal_long_only_and_solver_clean(module) -> None:
    report = json.loads(
        (module.OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
    )
    assert all(report["checks"].values())
    candidate_files = [
        path
        for path in module.OUTPUT_DIR.glob("*_monthly.csv")
        if path.name != "stage14_static_recomputed_monthly.csv"
        and path.name != "monthly_technical_signals.csv"
    ]
    assert len(candidate_files) == 1
    path = pd.read_csv(candidate_files[0])
    weights = path[[f"w_{asset}" for asset in ASSETS]]
    assert len(path) == 232
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-9).all().all()
    assert (weights <= 1.0 + 1e-9).all().all()
    assert path["solver_success"].all()
    assert not path["used_fallback"].any()


def test_review_identifies_only_stage24_as_pareto_improvement_and_no_target_pass() -> None:
    review_dir = (
        stage20.ROOT
        / "strategies"
        / "stage20_improvement_review"
        / "outputs"
    )
    comparison = pd.read_csv(review_dir / "candidate_performance_and_gates.csv")
    full = comparison.loc[comparison["Period"] == "full_2007_2026"]
    assert not full["PassAllPointTargets"].any()
    pareto = full.loc[full["ParetoImprovesStage20"], "Strategy"].tolist()
    assert pareto == ["Stage24_EquityKRatioOnly"]

    stage24_full = full.set_index("Strategy").loc["Stage24_EquityKRatioOnly"]
    assert stage24_full["CAGR"] == pytest.approx(0.095182, abs=1e-6)
    assert stage24_full["Sharpe"] == pytest.approx(0.996774, abs=1e-6)
    assert stage24_full["MDD"] == pytest.approx(-0.136289, abs=1e-6)

    report = json.loads((review_dir / "review_report.json").read_text(encoding="utf-8"))
    assert not report["target_achieved"]
    assert report["recommended_deployment_candidate"] == "Stage24_EquityKRatioOnly"
