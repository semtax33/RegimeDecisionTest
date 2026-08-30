from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from strategies.stage07_zero_tune_vkospi import zero_tune_strategy as strategy


def test_zero_tune_contract_has_no_candidate_or_model_hyperparameters() -> None:
    spec = json.loads(
        (strategy.OUTPUT_DIR.parent / "design_spec.json").read_text(encoding="utf-8")
    )
    assert strategy.TUNABLE_HYPERPARAMETERS == ()
    assert spec["tunable_hyperparameters"] == []
    assert "no candidate grid" in spec["selection_procedure"]

    source = inspect.getsource(strategy)
    assert "sklearn" not in source
    assert "scipy.optimize" not in source
    assert "LogisticRegression" not in source
    assert "minimize(" not in source


def test_expanding_midrank_and_overlay_are_deterministic() -> None:
    values = pd.Series([1.0, 2.0, 1.0])
    actual = strategy.causal_expanding_percentile(values)
    expected = pd.Series([0.5, 0.75, 1 / 3])
    pd.testing.assert_series_equal(actual, expected)

    base = np.array([0.40, 0.20, 0.10, 0.30])
    no_stress = strategy.apply_parameter_free_overlay(base, 0.0)
    full_stress = strategy.apply_parameter_free_overlay(base, 1.0)
    np.testing.assert_allclose(no_stress, base)
    np.testing.assert_allclose(full_stress, [0.0, 0.55, 0.45, 0.0])
    assert full_stress.sum() == pytest.approx(1.0)


def test_zero_tune_full_reproduction_and_causality() -> None:
    result = strategy.run_zero_tune_research(save=False)
    comparison = result["comparison"].set_index(["Period", "Strategy"])
    full = comparison.loc[("full_2007_2026", "ZeroTune_VKOSPI")]
    locked = comparison.loc[("locked_2018_2026", "ZeroTune_VKOSPI")]

    assert int(full["Months"]) == 232
    assert float(full["CAGR"]) == pytest.approx(0.0713687275831425, abs=1e-12)
    assert float(full["Sharpe"]) == pytest.approx(0.8446994945896235, abs=1e-12)
    assert float(full["MDD"]) == pytest.approx(-0.14012928924826884, abs=1e-12)
    assert int(locked["Months"]) == 103
    assert float(locked["CAGR"]) == pytest.approx(0.07038611121058924, abs=1e-12)
    assert float(locked["Sharpe"]) == pytest.approx(1.1336473538343195, abs=1e-12)
    assert float(locked["MDD"]) == pytest.approx(-0.14012928924826884, abs=1e-12)

    probabilities = result["probabilities"]
    daily = result["daily"]
    assert (probabilities["signal_month"] < probabilities.index).all()
    valid = daily["signal_date"].notna()
    assert (
        daily.index[valid].to_numpy()
        > pd.DatetimeIndex(daily.loc[valid, "signal_date"]).to_numpy()
    ).all()
