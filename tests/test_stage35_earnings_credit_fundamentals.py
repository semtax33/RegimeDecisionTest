from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE35 = ROOT / "strategies" / "stage35_earnings_credit_fundamentals"
OUTPUT = STAGE35 / "outputs"
REPORT_PATH = OUTPUT / "validation_report.json"
EARNINGS = ROOT / "raw_data" / "260829_fwdPE.EPS.rev.xlsx"
CREDIT = ROOT / "raw_data" / "260829_국고채.회사채.xlsx"


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


def test_stage35_preserves_both_sources_and_frozen_strategies() -> None:
    assert _sha256(EARNINGS) == (
        "36a65e5cf1bf060f74544e49a38ee72bcab6c989d0a1772bce4bffad53ce23fe"
    )
    assert _sha256(CREDIT) == (
        "8bd43c6737db28b7d3c431035a21f5bebedeb7d57c442e3bc9b39c11bc1a1e34"
    )
    report = _report()
    assert report["source_manifest_before"] == report["source_manifest_after"]
    assert report["frozen_manifest_before"] == report["frozen_manifest_after"]
    assert report["checks"]["source_files_unchanged"]
    assert report["checks"]["stage20_and_stage34_files_unchanged"]


def test_stage35_credit_and_valuation_formulas_are_auditable() -> None:
    daily = pd.read_csv(
        OUTPUT / "normalized_earnings_credit_daily.csv", parse_dates=["date"]
    )
    valid_credit = daily[
        daily["corp_aa_minus_3y_pct"].notna()
        & daily["ktb_3y_pct"].notna()
    ]
    np.testing.assert_allclose(
        valid_credit["aa_credit_spread_pctpt"],
        valid_credit["corp_aa_minus_3y_pct"] - valid_credit["ktb_3y_pct"],
        rtol=0,
        atol=1e-13,
    )
    valid_value = daily[
        daily["forward_pe_12m"].gt(0) & daily["ktb_10y_pct"].notna()
    ]
    np.testing.assert_allclose(
        valid_value["earnings_yield_gap"],
        1.0 / valid_value["forward_pe_12m"]
        - valid_value["ktb_10y_pct"] / 100.0,
        rtol=0,
        atol=1e-13,
    )
    audit = _report()["data_audit"]
    assert audit["eps_revision_provider_field_used"]
    assert audit["bbb_and_quality_role"] == "robustness diagnostics only"
    assert not audit["winsorization"]
    assert not audit["parameter_grid"]


def test_stage35_signals_are_causal_and_have_2007_presample() -> None:
    signals = pd.read_csv(OUTPUT / "monthly_earnings_credit_signals.csv")
    target = pd.PeriodIndex(signals["target_month"], freq="M")
    signal = pd.PeriodIndex(signals["fundamental_signal_month"], freq="M")
    assert (signal < target).all()
    first = signals.loc[signals["target_month"].eq("2007-04")].iloc[0]
    assert int(first["calibration_observations"]) == 75
    assert int(first["valuation_calibration_observations"]) >= 60

    report = _report()
    assert report["checks"]["first_2007_signal_has_presample_calibration"]
    assert report["checks"][
        "valuation_anchor_uses_only_fully_observed_12m_targets"
    ]
    assert report["fixed_design"]["searched_parameters"] is None


def test_stage35_all_economic_mechanism_gates_pass() -> None:
    gates = _report()["gate_results"]
    assert gates["eps_pass"]
    assert gates["credit_return_pass"]
    assert gates["credit_risk_pass"]
    assert gates["valuation_pass"]
    assert gates["mechanism_pass"]
    assert all(gates["eps_gates"].values())
    assert all(gates["credit_return_gates"].values())
    assert all(gates["credit_risk_gates"].values())
    assert all(gates["valuation_gates"].values())

    returns = pd.read_csv(OUTPUT / "return_predictive_regressions.csv")
    eps = returns[
        (returns["Model"] == "EPSFullControls")
        & (returns["HorizonMonths"] == 1)
    ].iloc[0]
    credit = returns[
        (returns["Model"] == "CreditFullControls")
        & (returns["HorizonMonths"] == 3)
    ].iloc[0]
    assert eps["EPSStandardizedBeta"] > 0 and eps["EPSHACPValue"] < 0.10
    assert (
        credit["CreditEasingStandardizedBeta"] > 0
        and credit["CreditHACPValue"] < 0.10
    )


def test_stage35_hits_requested_full_period_performance_goals() -> None:
    full = _performance().query("Period == 'full_2007_2026'").set_index(
        "Strategy"
    )
    baseline = full.loc["Stage20_Frozen"]
    candidate = full.loc["Stage35_FundamentalDualRole"]
    assert np.isclose(candidate["CAGR"], 0.10608, atol=2e-5)
    assert np.isclose(candidate["Sharpe"], 1.05736, atol=2e-5)
    assert np.isclose(candidate["MDD"], -0.13742, atol=2e-5)
    assert candidate["CAGR"] >= 0.10
    assert candidate["Sharpe"] >= 1.0
    assert candidate["Sharpe"] > baseline["Sharpe"]
    assert candidate["MDD"] > baseline["MDD"]

    report = _report()
    assert report["promoted_strategy"] == "Stage35_FundamentalDualRole"
    assert report["decision"] == "promote_stage35_fundamental_dual_role"
    assert report["gate_results"]["performance_pass"]
    assert all(report["gate_results"]["performance_gates"].values())


def test_stage35_subperiods_and_block_bootstrap_are_reported() -> None:
    performance = _performance().set_index(["Strategy", "Period"])
    for period in ["early_2007_2017", "locked_2018_2026"]:
        baseline = performance.loc[("Stage20_Frozen", period)]
        candidate = performance.loc[("Stage35_FundamentalDualRole", period)]
        assert candidate["CAGR"] > baseline["CAGR"]
        assert candidate["Sharpe"] >= baseline["Sharpe"]

    bootstrap = pd.read_csv(
        OUTPUT / "paired_block_bootstrap_vs_stage20.csv"
    )
    candidate = bootstrap[
        bootstrap["Candidate"] == "Stage35_FundamentalDualRole"
    ].set_index("Metric")
    assert int(candidate["Replications"].min()) == 2000
    assert int(candidate["BlockMonths"].min()) == 12
    assert candidate.loc["delta_CAGR", "ProbabilityPositive"] >= 0.60
    assert candidate.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.60
    assert candidate.loc["delta_MDD", "ProbabilityPositive"] >= 0.50


def test_stage35_reproduction_constraints_and_solver_checks_pass() -> None:
    report = _report()
    assert all(report["checks"].values())
    assert report["reproduction_audit"]["months"] == 232
    assert report["reproduction_audit"]["max_absolute_return_error"] < 5e-7
    assert report["reproduction_audit"]["max_absolute_weight_error"] < 5e-6

    weights = ["w_KODEX200", "w_BOND", "w_GLD", "w_USO"]
    for path in OUTPUT.glob("stage35_*_monthly.csv"):
        frame = pd.read_csv(path)
        np.testing.assert_allclose(
            frame[weights].sum(axis=1), 1.0, rtol=0, atol=1e-8
        )
        assert frame[weights].min().min() >= -1e-9
        assert frame[weights].max().max() <= 1.0 + 1e-9
        assert frame["solver_success"].all()
        assert not frame["used_fallback"].any()


class _HTMLAudit(HTMLParser):
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


def test_stage35_html_report_and_charts_are_complete() -> None:
    html_path = STAGE35 / "stage35_earnings_credit_report.html"
    parser = _HTMLAudit()
    parser.feed(html_path.read_text(encoding="utf-8"))
    assert parser.sections >= 13
    assert parser.tables >= 7
    expected = {
        "outputs/mechanism_coefficients.png",
        "outputs/performance_comparison.png",
        "outputs/nav_comparison.png",
    }
    assert expected.issubset(set(parser.images))
    for image in expected:
        assert (STAGE35 / image).stat().st_size > 40_000
