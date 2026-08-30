from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_research import ASSETS
from strategies.stage09_strict_hard_slsqp.strict_hard_slsqp import (
    HARD_SHARE,
    OUTPUT_DIR,
    SLSQP_SHARE,
    strict_hard_weights,
)


def test_strict_hard_mapping_is_exactly_one_hot() -> None:
    signals = pd.DataFrame(
        {
            "regime": [
                "Goldilocks",
                "Overheating",
                "Slowdown",
                "Stagflation",
            ]
        },
        index=pd.period_range("2020-01", periods=4, freq="M"),
    )
    actual = strict_hard_weights(signals)
    columns = [f"w_{asset}" for asset in ASSETS]
    np.testing.assert_array_equal(actual[columns].to_numpy(), np.eye(4)[[0, 3, 1, 2]])


def test_saved_hard_and_blended_weights_obey_contract() -> None:
    hard = pd.read_csv(OUTPUT_DIR / "strict_hard_weights.csv", index_col=0)
    slsqp = pd.read_csv(OUTPUT_DIR / "slsqp_path.csv", index_col=0)
    blended = pd.read_csv(
        OUTPUT_DIR / "strict_hard40_slsqp60_weights.csv", index_col=0
    )
    columns = [f"w_{asset}" for asset in ASSETS]
    assert np.isin(hard[columns].to_numpy(), [0.0, 1.0]).all()
    assert np.allclose(hard[columns].sum(axis=1), 1.0)
    assert np.all((hard[columns] == 1.0).sum(axis=1) == 1)
    common = (
        hard.index.intersection(slsqp.index).intersection(blended.index)
    )
    expected = (
        HARD_SHARE * hard.loc[common, columns]
        + SLSQP_SHARE * slsqp.loc[common, columns]
    )
    np.testing.assert_allclose(
        blended.loc[common, columns].to_numpy(),
        expected.to_numpy(),
        atol=1e-14,
    )
    assert np.allclose(blended[columns].sum(axis=1), 1.0)


def test_requested_monthly_and_zero_vkospi_metrics_are_reproduced() -> None:
    comparison = pd.read_csv(OUTPUT_DIR / "performance_comparison.csv")
    full = comparison.loc[
        comparison["Period"].eq("full_2007_2026")
    ].set_index("Strategy")
    monthly = full.loc["StrictHard40_SLSQP60_Monthly"]
    overlay = full.loc["StrictHard40_SLSQP60_ZeroTuneVKOSPI"]
    assert int(monthly["Months"]) == 232
    assert monthly["CAGR"] == pytest.approx(0.09599958285278087, abs=1e-12)
    assert monthly["Sharpe"] == pytest.approx(1.04889762675527, abs=1e-12)
    assert monthly["MDD"] == pytest.approx(-0.11448602277520559, abs=1e-12)
    assert overlay["CAGR"] == pytest.approx(0.07351675575924244, abs=1e-12)
    assert overlay["Sharpe"] == pytest.approx(0.9334052390017522, abs=1e-12)
    assert overlay["MDD"] == pytest.approx(-0.1272163382188587, abs=1e-12)


def test_report_confirms_causality_and_disabled_later_layers() -> None:
    report = json.loads(
        (OUTPUT_DIR / "strict_hard_slsqp_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(report["checks"].values())
    assert not any(
        report["definition"]["later_current_layers"].values()
    )
    hard_mapping = report["definition"]["hard_mapping"]
    for weights in hard_mapping.values():
        assert sorted(weights.values()) == [0.0, 0.0, 0.0, 1.0]
