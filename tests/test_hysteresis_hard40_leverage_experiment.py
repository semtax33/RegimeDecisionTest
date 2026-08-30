from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def report() -> dict:
    return json.loads(
        (RESULTS / "hysteresis_hard40_leverage_validation.json").read_text(
            encoding="utf-8"
        )
    )


def test_hysteresis_configuration_and_scope_are_explicit() -> None:
    implementation = report()["implementation"]
    hysteresis = implementation["hysteresis"]
    allocation = implementation["allocation"]
    assert hysteresis["upper"] == 0.2
    assert hysteresis["lower"] == -0.2
    assert hysteresis["scope"] == "representative regime used by Hard 40% only"
    assert allocation["hard_weight"] == 0.4
    assert allocation["defensive_slsqp_weight"] == 0.6
    assert allocation["leverage_caps_tested"] == [1.0, 1.1, 1.2, 1.3]


def test_hysteresis_switches_only_beyond_the_notebook_thresholds() -> None:
    signals = pd.read_csv(RESULTS / "hysteresis_hard40_signals.csv", index_col=0)
    for state_column, score_column in (
        ("growth_state", "growth_score"),
        ("inflation_state", "inflation_score"),
    ):
        state = signals[state_column]
        changed = state.ne(state.shift()) & state.shift().notna()
        switched_high = changed & state.eq(1)
        switched_low = changed & state.eq(-1)
        assert signals.loc[switched_high, score_column].gt(0.2).all()
        assert signals.loc[switched_low, score_column].lt(-0.2).all()


def test_signal_schedule_is_causal_and_regimes_are_complete() -> None:
    signals = pd.read_csv(RESULTS / "hysteresis_hard40_signals.csv", index_col=0)
    target = pd.PeriodIndex(signals.index, freq="M")
    known = pd.PeriodIndex(signals["signal_month"], freq="M")
    assert (known < target).all()
    assert set(signals["regime"]) <= {
        "Goldilocks",
        "Overheating",
        "Slowdown",
        "Stagflation",
    }
    assert report()["regime_audit"]["hysteresis_switches"] == 23
    assert report()["regime_audit"]["probability_argmax_switches"] == 44


@pytest.mark.parametrize("cap", [1.0, 1.1, 1.2, 1.3])
def test_each_medium_path_respects_its_leverage_cap(cap: float) -> None:
    token = f"{cap:.1f}".replace(".", "p")
    medium = pd.read_csv(
        RESULTS / f"hysteresis_hard40_leverage_cap_{token}_medium.csv"
    )
    assert len(medium) == 232
    assert np.isfinite(medium["return"]).all()
    assert medium["leverage"].between(0.5, cap + 1e-12).all()


def test_selection_is_prelock_only_and_current_strategy_is_preserved() -> None:
    selection = report()["selection"]
    assert selection["uses_locked_period_for_selection"] is False
    assert selection["selected_leverage_cap"] == 1.0
    assert selection["strict_eligible_count"] == 0
    assert selection["selected_strict_prelock_pass"] is False
    assert selection["promotion_status"] == (
        "research_candidate_only_current_strategy_preserved"
    )
    assert (RESULTS / "balanced_logistic_no_sjm_final_reconciled.csv").exists()


def test_selected_candidate_tradeoff_is_reported_without_overclaiming() -> None:
    full = report()["full_audit_2007_2026"]
    locked = report()["locked_audit_2018_2026"]
    assert full["selected"]["CAGR"] == pytest.approx(0.1302106282547184)
    assert full["selected"]["Sharpe"] == pytest.approx(1.1575294378369256)
    assert full["selected"]["MDD"] == pytest.approx(-0.11168978163312282)
    assert full["delta_selected_minus_current"]["CAGR"] < 0
    assert full["delta_selected_minus_current"]["Sharpe"] > 0
    assert full["delta_selected_minus_current"]["MDD"] > 0
    assert locked["passes_all_three"] is False
    assert locked["bootstrap"]["probability_all_three_improve"] == pytest.approx(
        0.0078
    )


def test_html_colab_and_bundle_deliverables_are_self_consistent() -> None:
    html_path = ROOT / "artifacts/reports/hysteresis_hard40_leverage_report.html"
    notebook_path = ROOT / "artifacts/notebooks/hysteresis_hard40_leverage_colab.ipynb"
    bundle_path = ROOT / "artifacts/bundles/hysteresis_hard40_leverage_colab_bundle.zip"

    text = html_path.read_text(encoding="utf-8")
    assert "현재 전략은 유지" in text
    assert "국면 전환은 44회에서 23회로 줄었다" in text
    assert re.search(r"__[A-Z_]+__", text) is None
    local_links = re.findall(r'href="([^"#][^"]*)"', text)
    assert local_links
    for link in local_links:
        assert (html_path.parent / link).resolve().exists(), link

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 6
    for cell in code_cells:
        ast.parse("".join(cell["source"]))

    with zipfile.ZipFile(bundle_path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    required = {
        "RegimeDecisionTest/strategies/stage09_hysteresis/hysteresis_hard40_leverage_experiment.py",
        "RegimeDecisionTest/strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py",
        "RegimeDecisionTest/raw_data/compass.db",
        "RegimeDecisionTest/cache/market_daily.csv",
        "RegimeDecisionTest/results/openassetpricing_composites.csv",
    }
    assert required <= names
