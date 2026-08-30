from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_report() -> dict:
    return json.loads(
        (RESULTS / "vkospi_model_robustness.json").read_text(encoding="utf-8")
    )


def test_candidate_grids_and_locked_boundary() -> None:
    report = load_report()
    assert report["schema_version"] == 2
    assert "through 2017-12" in report["method"]["selection_boundary"]
    assert "2018-01 onward" in report["method"]["locked_boundary"]
    assert report["logistic"]["candidate_count"] == 28
    assert report["sjm"]["attempted_candidate_count"] == 49
    assert report["sjm"]["valid_candidate_count"] == 37
    assert report["sjm"]["invalid_candidate_count"] == 12


def test_deployed_paths_are_exact_reproductions() -> None:
    report = load_report()
    logistic = report["logistic"]["deployed_reproduction"]
    sjm = report["sjm"]["deployed_reproduction"]
    assert logistic["probability_observations"] == 194
    assert logistic["portfolio_observations"] == 232
    assert logistic["max_absolute_probability_difference"] < 1e-12
    assert logistic["max_absolute_portfolio_return_difference"] < 1e-12
    assert sjm["observations"] == 232
    assert sjm["max_absolute_growth_probability_difference"] < 1e-12
    assert sjm["max_absolute_inflation_probability_difference"] < 1e-12


def test_numerical_failures_are_preserved_not_imputed() -> None:
    report = load_report()
    logistic = report["logistic"]
    assert logistic["candidates_with_convergence_warnings"] == 3
    assert logistic["total_convergence_warnings"] == 361
    assert logistic["zero_coefficient_candidates"] == [
        "l1_liblinear_c0.01_balanced",
        "l1_liblinear_c0.03_balanced",
    ]
    invalid = report["sjm"]["invalid_candidates"]
    assert len(invalid) == 12
    assert {item["jump_penalty"] for item in invalid} == {6.0}
    assert {item["keep_features"] for item in invalid} == {2, 4, 6}
    assert all("non-finite" in item["reason"] for item in invalid)


def test_selection_risk_is_reported_without_locked_retuning() -> None:
    report = load_report()
    logistic = report["logistic"]
    sjm = report["sjm"]
    assert logistic["deployed_calibration_portfolio_rank"] == 1
    assert logistic["deployed_locked_portfolio_rank"] == 1
    assert logistic["prelock_portfolio_sharpe_pbo"]["pbo"] > 0.90
    assert (
        logistic["prelock_portfolio_sharpe_pbo_excluding_warning_candidates"][
            "pbo"
        ]
        > 0.90
    )
    assert sjm["prelock_macro_brier_pbo"]["pbo"] > 0.90
    assert sjm["prelock_soft_sharpe_pbo"]["pbo"] > 0.70
    assert sjm["no_sjm_vs_deployed"][
        "bootstrap_probability_no_sjm_brier_better"
    ] > 0.95
    assert "cannot prove" in report["conclusion"]


def test_candidate_tables_contain_all_attempts() -> None:
    logistic = pd.read_csv(RESULTS / "vkospi_logistic_candidate_summary.csv")
    sjm = pd.read_csv(RESULTS / "vkospi_sjm_candidate_summary.csv")
    assert len(logistic) == 28
    assert len(sjm) == 49
    assert logistic["candidate"].nunique() == 28
    assert sjm["candidate"].nunique() == 49
    assert sjm["invalid_reason"].fillna("").ne("").sum() == 12


def test_html_notebook_and_bundle_include_robustness_audit() -> None:
    html_text = (ROOT / "artifacts/reports/vkospi_robust_dynamic_technical_report.html").read_text(
        encoding="utf-8"
    )
    assert 'id="model-robustness"' in html_text
    assert "95.7%" in html_text and "94.3%" in html_text
    assert "점프 벌점 6" in html_text
    assert "오버피팅이 없다" in html_text
    assert 'id="current-reference"' in html_text
    assert "현재 비교기준 2007–2026 · CAGR 15.64% · Sharpe 1.133 · MDD -12.96%" in html_text
    assert "사전 두 창 동시 관문은 <b>통과하지 못했습니다</b>" in html_text
    assert "pg_t = 0.85 × pg_raw,t + 0.15 × pg_(t−1)" in html_text
    assert "raw 확률을 문자 그대로 읽지 마십시오" in html_text
    assert "핵심 하이퍼파라미터와 결정 근거" in html_text
    assert 'id="rebuttal"' in html_text
    assert html_text.count("<summary>Q") == 22
    assert "point-in-time vintage" in html_text
    notebook_text = (
        ROOT / "artifacts/notebooks/vkospi_robust_dynamic_strategy_colab.ipynb"
    ).read_text(encoding="utf-8")
    assert "vkospi_model_robustness.py" in notebook_text
    bundle = ROOT / "artifacts/bundles/vkospi_robust_dynamic_colab_bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "RegimeDecisionTest/vkospi_model_robustness.py" in names
    assert "RegimeDecisionTest/results/vkospi_model_robustness.json" in names
