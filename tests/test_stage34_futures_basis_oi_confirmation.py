from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE34 = ROOT / "strategies" / "stage34_futures_basis_oi_confirmation"
OUTPUT = STAGE34 / "outputs"
REPORT_PATH = OUTPUT / "validation_report.json"
SOURCE_PATH = ROOT / "raw_data" / "260829_K200선물데이터.xlsx"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _performance() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "performance_comparison.csv")


def test_stage34_preserves_source_and_frozen_research_files() -> None:
    report = _report()
    expected_hash = (
        "4d37798c636a4716b7a7d03b71549d7195067e35718635da0db2420f019d0818"
    )
    assert _sha256(SOURCE_PATH) == expected_hash
    assert report["checks"]["source_files_unchanged"]
    assert report["checks"]["stage20_to_stage33_files_unchanged"]
    assert report["source_manifest_before"] == report["source_manifest_after"]
    assert report["frozen_manifest_before"] == report["frozen_manifest_after"]


def test_stage34_basis_formula_and_roll_guard_are_explicit() -> None:
    daily = pd.read_csv(
        OUTPUT / "normalized_k200_futures_daily.csv",
        parse_dates=["date"],
    )
    valid = daily[
        daily["near_close"].gt(0)
        & daily["near_settlement_theoretical_price"].gt(0)
    ].copy()
    expected = (
        valid["near_close"] / valid["near_settlement_theoretical_price"] - 1
    )
    np.testing.assert_allclose(
        valid["signed_close_theory_basis"], expected, rtol=0, atol=1e-14
    )
    changed = daily[
        daily["basis_change_20d"].notna()
        | daily["oi_log_change_20d"].notna()
    ]
    assert changed["same_contract_20d"].eq(1.0).all()

    audit = _report()["data_audit"]
    assert not audit["provider_gap_used_as_signal"]
    assert audit["provider_gap_nonzero_share_2007_2010"] > 0.90
    assert audit["provider_gap_nonzero_share_2011_2026"] < 0.10
    assert audit["implied_volatility_nonzero_observations"] == 0
    assert not audit["winsorization"]
    assert not audit["technical_indicator_search"]


def test_stage34_signals_are_causal_and_design_is_fixed() -> None:
    signals = pd.read_csv(OUTPUT / "monthly_futures_basis_oi_signals.csv")
    target = pd.PeriodIndex(signals["target_month"], freq="M")
    signal = pd.PeriodIndex(signals["futures_signal_month"], freq="M")
    assert (signal < target).all()

    report = _report()
    assert report["fixed_design"]["causal_calibration_months"] == 60
    assert report["fixed_design"]["searched_parameters"] is None
    assert report["checks"]["fixed_20d_signal_and_60m_calibration"]
    assert report["checks"]["no_rsi_macd_breakout_or_horizon_search"]


def test_stage34_direction_and_oi_mechanism_gates_fail() -> None:
    report = _report()
    gates = report["gate_results"]
    assert not gates["direction_pass"]
    assert not gates["oi_confirmation_pass"]
    assert not any(gates["direction_gates"].values())
    assert not any(gates["oi_confirmation_gates"].values())

    regressions = pd.read_csv(OUTPUT / "return_predictive_regressions.csv")
    one_month = regressions[
        (regressions["HorizonMonths"] == 1)
        & (regressions["Model"] == "BasisOIInteractionFullControls")
    ].iloc[0]
    three_month = regressions[
        (regressions["HorizonMonths"] == 3)
        & (regressions["Model"] == "BasisFullControls")
    ].iloc[0]
    assert one_month["InteractionStandardizedBeta"] < 0
    assert one_month["InteractionHACPValue"] > 0.10
    assert three_month["BasisStandardizedBeta"] < 0


def test_stage34_risk_and_false_positive_gates_fail() -> None:
    report = _report()
    gates = report["gate_results"]
    assert not gates["risk_sensor_pass"]
    assert not gates["false_positive_confirmation_pass"]

    risk = pd.read_csv(OUTPUT / "risk_predictive_regressions.csv")
    assert (risk["DislocationHACPValue"] > 0.10).all()
    false_positive = pd.read_csv(OUTPUT / "false_positive_regression.csv").iloc[0]
    assert false_positive["DislocationStandardizedBeta"] < 0
    assert false_positive["DislocationHACPValue"] > 0.10


def test_stage34_performance_does_not_justify_promotion() -> None:
    full = _performance().query("Period == 'full_2007_2026'").set_index(
        "Strategy"
    )
    baseline = full.loc["Stage20_Frozen"]
    basis = full.loc["Stage34_BasisAlpha"]
    combined = full.loc["Stage34_Combined"]

    assert np.isclose(baseline["CAGR"], 0.0939740153, atol=1e-9)
    assert np.isclose(baseline["Sharpe"], 0.986502924, atol=1e-9)
    assert np.isclose(baseline["MDD"], -0.140148068, atol=1e-9)
    assert basis["CAGR"] > baseline["CAGR"]
    assert basis["MDD"] > baseline["MDD"]
    assert basis["Sharpe"] < baseline["Sharpe"]
    assert basis["TotalCost"] > 2.5 * baseline["TotalCost"]
    assert combined["CAGR"] < 0.10
    assert combined["Sharpe"] < 1.0

    report = _report()
    assert report["promoted_strategy"] is None
    assert report["decision"] == (
        "keep_stage20_frozen_futures_signals_fail_promotion_gate"
    )


def test_stage34_reproduction_and_candidate_constraints_pass() -> None:
    report = _report()
    checks = report["checks"]
    assert all(checks.values())
    assert report["reproduction_audit"]["months"] == 232
    assert report["reproduction_audit"]["max_absolute_return_error"] < 5e-7
    assert report["reproduction_audit"]["max_absolute_weight_error"] < 5e-6

    weight_columns = ["w_KODEX200", "w_BOND", "w_GLD", "w_USO"]
    for path in OUTPUT.glob("stage34_*_monthly.csv"):
        frame = pd.read_csv(path)
        np.testing.assert_allclose(
            frame[weight_columns].sum(axis=1), 1.0, rtol=0, atol=1e-8
        )
        assert frame[weight_columns].min().min() >= -1e-9
        assert frame[weight_columns].max().max() <= 1.0 + 1e-9
        assert frame["solver_success"].all()
        assert not frame["used_fallback"].any()


class _ReportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections = 0
        self.tables = 0
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            self.sections += 1
        elif tag == "table":
            self.tables += 1
        elif tag == "img":
            self.images.append(dict(attrs).get("src") or "")


def test_stage34_html_report_and_charts_are_complete() -> None:
    html_path = STAGE34 / "stage34_futures_basis_oi_report.html"
    parser = _ReportHTMLParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    assert parser.sections >= 10
    assert parser.tables >= 5
    assert {
        "outputs/basis_oi_mechanism.png",
        "outputs/performance_comparison.png",
    }.issubset(set(parser.images))
    for image in parser.images:
        assert (STAGE34 / image).stat().st_size > 40_000
