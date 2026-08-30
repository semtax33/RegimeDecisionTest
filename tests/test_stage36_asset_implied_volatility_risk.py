from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGE36 = ROOT / "strategies" / "stage36_asset_implied_volatility_risk"
OUTPUT = STAGE36 / "outputs"
REPORT_PATH = OUTPUT / "validation_report.json"
GVZ = ROOT / "raw_data" / "GVZCLS.csv"
OVX = ROOT / "raw_data" / "OVXCLS.csv"


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


def test_stage36_preserves_gvz_ovx_and_stage35() -> None:
    assert _sha256(GVZ) == (
        "57ed6220e310eda54752ff115c31df7511306a352ce607e935fc43ac318a84be"
    )
    assert _sha256(OVX) == (
        "b0fcc2c9120b66bbcc084a8aecdcddf42476f5ad61342e656f0892ebd4b73bd6"
    )
    report = _report()
    assert report["source_manifest_before"] == report["source_manifest_after"]
    assert report["frozen_manifest_before"] == report["frozen_manifest_after"]
    assert report["checks"]["source_files_unchanged"]
    assert report["checks"]["stage35_frozen_files_unchanged"]


def test_stage36_does_not_backfill_and_waits_for_252_prior_observations() -> None:
    signals = pd.read_csv(
        OUTPUT / "monthly_asset_volatility_signals.csv",
        parse_dates=["gvz_signal_date", "ovx_signal_date"],
    )
    assert signals.loc[signals["target_month"].eq("2007-04"), "gvz_gld_variance_multiplier"].iloc[0] == 1.0
    assert signals.loc[signals["target_month"].eq("2007-04"), "ovx_uso_variance_multiplier"].iloc[0] == 1.0

    for sensor, asset in (("gvz", "gld"), ("ovx", "uso")):
        active = signals.loc[signals[f"{sensor}_active"]]
        inactive = signals.loc[~signals[f"{sensor}_active"]]
        assert active[f"{sensor}_prior_valid_observations"].min() >= 252
        assert inactive[f"{sensor}_{asset}_variance_multiplier"].eq(1.0).all()
        np.testing.assert_allclose(
            active[f"{sensor}_{asset}_variance_multiplier"],
            1.0 + active[f"{sensor}_causal_rank"],
            rtol=0,
            atol=1e-13,
        )
    report = _report()
    assert report["activation_audit"]["gvz_first_active_target"] == "2009-07"
    assert report["activation_audit"]["ovx_first_active_target"] == "2008-06"
    assert report["checks"]["no_backfill_before_minimum_history"]
    assert report["checks"]["minimum_252_prior_observations"]


def test_stage36_monthly_signals_are_causal() -> None:
    signals = pd.read_csv(
        OUTPUT / "monthly_asset_volatility_signals.csv",
        parse_dates=["gvz_signal_date", "ovx_signal_date"],
    )
    target = pd.PeriodIndex(signals["target_month"], freq="M")
    signal_month = pd.PeriodIndex(
        signals["asset_vol_signal_month"], freq="M"
    )
    assert (signal_month < target).all()
    for sensor in ("gvz", "ovx"):
        valid = signals[f"{sensor}_signal_date"].notna()
        last_allowed = pd.Series(
            [(period - 1).to_timestamp("M") for period in target[valid]],
            index=signals.index[valid],
        )
        observed = signals.loc[valid, f"{sensor}_signal_date"]
        assert (observed <= last_allowed).all()
    report = _report()
    assert report["checks"]["signal_month_precedes_target"]
    assert report["checks"]["signal_dates_not_after_prior_month_end"]


def test_stage36_uses_variance_only_and_no_directional_mu() -> None:
    for path in OUTPUT.glob("stage36_*_monthly.csv"):
        frame = pd.read_csv(path)
        assert frame["gvz_mu_adjustment_GLD"].eq(0.0).all()
        assert frame["ovx_mu_adjustment_USO"].eq(0.0).all()
        assert frame["gvz_gold_variance_multiplier"].between(1.0, 2.0).all()
        assert frame["ovx_oil_variance_multiplier"].between(1.0, 2.0).all()
    report = _report()
    assert report["fixed_design"]["directional_mu_effect"] is None
    assert report["fixed_design"]["searched_parameters"] is None
    assert report["checks"]["no_directional_mu_adjustment"]
    assert report["checks"]["variance_multipliers_within_one_and_two"]


def test_stage36_asset_sensors_predict_own_future_risk() -> None:
    tests = pd.read_csv(OUTPUT / "asset_risk_predictive_regressions.csv")
    view = tests.loc[
        tests["Period"].eq("common_2010_2026")
        & tests["Model"].eq("FullControls")
    ].set_index(["Sensor", "Target"])
    for sensor in ("GVZ", "OVX"):
        realized = view.loc[(sensor, "future_realized_vol_1m")]
        drawdown = view.loc[(sensor, "future_max_drawdown_3m")]
        assert realized["SensorStandardizedBeta"] > 0.0
        assert realized["SensorHACPValue"] < 0.10
        assert drawdown["SensorStandardizedBeta"] > 0.0
        assert drawdown["SensorHACPValue"] < 0.10
    report = _report()
    assert report["gate_results"]["gvz_risk_gate"]["pass"]
    assert report["gate_results"]["ovx_risk_gate"]["pass"]


def test_stage36_reproduces_stage35_and_all_solves_are_feasible() -> None:
    report = _report()
    audit = report["reproduction_audit"]
    assert audit["months"] == 232
    assert audit["max_absolute_return_error"] < 5e-7
    assert audit["max_absolute_weight_error"] < 5e-6

    for path in OUTPUT.glob("stage36_*_monthly.csv"):
        frame = pd.read_csv(path)
        np.testing.assert_allclose(
            frame[["w_KODEX200", "w_BOND", "w_GLD", "w_USO"]].sum(axis=1),
            1.0,
            rtol=0,
            atol=1e-8,
        )
        assert frame[["w_KODEX200", "w_BOND", "w_GLD", "w_USO"]].min().min() >= -1e-9
        assert frame[["w_KODEX200", "w_BOND", "w_GLD", "w_USO"]].max().max() <= 1.0 + 1e-9
        assert frame["solver_success"].all()
        assert not frame["used_fallback"].any()
        assert frame["volatility_slack"].min() >= -1e-7
        assert frame["cdar_slack"].min() >= -1e-7
    assert report["checks"]["all_candidate_solvers_feasible"]
    assert report["checks"]["no_leverage_long_only_sum_to_one"]


def test_stage36_full_and_common_performance_are_reported_honestly() -> None:
    perf = _performance().set_index(["Strategy", "Period"])
    full_base = perf.loc[("Stage35_Frozen", "full_2007_2026")]
    full_test = perf.loc[
        ("Stage36_GVZ_OVXAssetRisk", "full_2007_2026")
    ]
    assert np.isclose(full_test["CAGR"], 0.104994, atol=2e-6)
    assert np.isclose(full_test["Sharpe"], 1.104911, atol=2e-6)
    assert np.isclose(full_test["MDD"], -0.124071, atol=2e-6)
    assert full_test["CAGR"] >= 0.10
    assert full_test["Sharpe"] > full_base["Sharpe"]
    assert full_test["MDD"] > full_base["MDD"]

    common_base = perf.loc[("Stage35_Frozen", "common_2010_2026")]
    common_test = perf.loc[
        ("Stage36_GVZ_OVXAssetRisk", "common_2010_2026")
    ]
    assert common_test["Sharpe"] > common_base["Sharpe"]
    assert common_test["MDD"] > common_base["MDD"]
    assert common_test["CAGR"] >= common_base["CAGR"] - 0.005

    locked_base = perf.loc[("Stage35_Frozen", "locked_2018_2026")]
    locked_test = perf.loc[
        ("Stage36_GVZ_OVXAssetRisk", "locked_2018_2026")
    ]
    assert locked_test["CAGR"] < locked_base["CAGR"]
    assert locked_test["Sharpe"] < locked_base["Sharpe"]

    report = _report()
    assert report["gate_results"]["promote"]
    assert report["promoted_strategy"] == "Stage36_GVZ_OVXAssetRisk"


def test_stage36_block_bootstrap_supports_risk_efficiency_not_cagr() -> None:
    bootstrap = pd.read_csv(
        OUTPUT / "paired_block_bootstrap_vs_stage35.csv"
    )
    combined = bootstrap.loc[
        bootstrap["Candidate"].eq("Stage36_GVZ_OVXAssetRisk")
        & bootstrap["Period"].eq("common_2010_2026")
    ].set_index("Metric")
    assert int(combined["Replications"].min()) == 2000
    assert int(combined["BlockMonths"].min()) == 12
    assert combined.loc["delta_Sharpe", "ProbabilityPositive"] >= 0.60
    assert combined.loc["delta_MDD", "ProbabilityPositive"] >= 0.50
    assert combined.loc["delta_CAGR", "ProbabilityPositive"] < 0.50


class _HTMLAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections = 0
        self.tables = 0
        self.images: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "section":
            self.sections += 1
        elif tag == "table":
            self.tables += 1
        elif tag == "img":
            self.images.append(dict(attrs).get("src") or "")


def test_stage36_html_report_and_charts_are_complete() -> None:
    html_path = STAGE36 / "stage36_gvz_ovx_report.html"
    parser = _HTMLAudit()
    parser.feed(html_path.read_text(encoding="utf-8"))
    assert parser.sections >= 13
    assert parser.tables >= 3
    expected = {
        "outputs/sensor_history.png",
        "outputs/performance_comparison.png",
        "outputs/nav_comparison.png",
    }
    assert expected.issubset(set(parser.images))
    for image in expected:
        assert (STAGE36 / image).stat().st_size > 40_000


def test_stage36_detailed_implementation_guide_is_self_contained() -> None:
    html_path = STAGE36 / "stage36_implementation_economic_math_guide.html"
    text = html_path.read_text(encoding="utf-8")
    parser = _HTMLAudit()
    parser.feed(text)

    assert parser.sections >= 22
    assert parser.tables >= 7
    assert {
        "outputs/sensor_history.png",
        "outputs/performance_comparison.png",
        "outputs/nav_comparison.png",
    }.issubset(set(parser.images))
    for phrase in (
        "경제적 아이디어",
        "인과적 순위 변환",
        "Stage35에서 물려받은 기대수익·위험 엔진",
        "Stage36 공분산 오버레이의 수학",
        "SLSQP 목적함수와 원리",
        "전체 구현을 보려면 어느 소스코드를 읽어야 하나",
        "핵심 실제 구현 코드",
    ):
        assert phrase in text

    linked_files = (
        "asset_implied_volatility_risk_slsqp.py",
        "../stage35_earnings_credit_fundamentals/earnings_credit_fundamentals_slsqp.py",
        "../stage13_conditional_moments_slsqp/economic_conditional_slsqp.py",
        "../../tests/test_stage36_asset_implied_volatility_risk.py",
        "outputs/validation_report.json",
    )
    for relative in linked_files:
        assert (STAGE36 / relative).resolve().is_file()
