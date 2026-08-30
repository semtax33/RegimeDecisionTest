from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.stage08_vkospi_factorial.factorial_bridge import (
    COMPONENT_KEYS,
    OUTPUT_DIR,
)


def _combinations() -> pd.DataFrame:
    frame = pd.read_csv(
        OUTPUT_DIR / "all_128_combinations.csv",
        dtype={"Combination": str},
    )
    frame["Combination"] = frame["Combination"].str.zfill(len(COMPONENT_KEYS))
    return frame


def test_every_binary_component_combination_is_present_once() -> None:
    combinations = _combinations()
    assert len(combinations) == 2 ** len(COMPONENT_KEYS)
    assert combinations["Combination"].nunique() == len(combinations)
    assert set(combinations["Combination"]) == {
        format(number, f"0{len(COMPONENT_KEYS)}b")
        for number in range(2 ** len(COMPONENT_KEYS))
    }
    for component in COMPONENT_KEYS:
        assert set(combinations[component]) == {0, 1}
        assert int(combinations[component].sum()) == len(combinations) // 2


def test_both_endpoints_are_exact_monthly_reproductions() -> None:
    report = json.loads(
        (OUTPUT_DIR / "factorial_report.json").read_text(encoding="utf-8")
    )
    audit = report["endpoint_audit"]
    assert audit["combination_count"] == 128
    assert audit["zero_endpoint_months"] == 244
    assert audit["current_medium_months"] == 232
    assert audit["current_endpoint_months"] == 232
    assert audit["zero_endpoint_max_return_difference"] < 1e-12
    assert audit["current_medium_max_return_difference"] < 1e-12
    assert audit["current_endpoint_max_return_difference"] < 1e-12


def test_shapley_contributions_close_to_each_endpoint_delta() -> None:
    combinations = _combinations().set_index("Combination")
    shapley = pd.read_csv(OUTPUT_DIR / "shapley_attribution.csv")
    for metric, rows in shapley.groupby("Metric"):
        endpoint_delta = float(
            combinations.loc["1" * len(COMPONENT_KEYS), metric]
            - combinations.loc["0" * len(COMPONENT_KEYS), metric]
        )
        assert len(rows) == len(COMPONENT_KEYS)
        assert np.isclose(
            rows["ShapleyContribution"].sum(),
            endpoint_delta,
            atol=1e-11,
            rtol=0,
        )


def test_transition_and_return_outputs_cover_the_full_experiment() -> None:
    ordered = pd.read_csv(
        OUTPUT_DIR / "ordered_transition.csv",
        dtype={"Combination": str},
    )
    returns = pd.read_csv(
        OUTPUT_DIR / "all_128_monthly_returns.csv",
        index_col=0,
    )
    interactions = pd.read_csv(OUTPUT_DIR / "pairwise_interactions.csv")
    assert len(ordered) == len(COMPONENT_KEYS) + 1
    assert ordered.iloc[0]["Combination"].zfill(len(COMPONENT_KEYS)) == "0" * len(
        COMPONENT_KEYS
    )
    assert ordered.iloc[-1]["Combination"].zfill(len(COMPONENT_KEYS)) == "1" * len(
        COMPONENT_KEYS
    )
    assert returns.shape[1] == 2 ** len(COMPONENT_KEYS)
    assert len(interactions) == 6 * 21
