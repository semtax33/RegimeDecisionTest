from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
COLAB = (
    ROOT
    / "strategies"
    / "stage36_asset_implied_volatility_risk"
    / "colab"
)
NOTEBOOK = COLAB / "Stage36_GVZ_OVX_Colab.ipynb"
BUNDLE = COLAB / "stage36_colab_data.zip"
MANIFEST = COLAB / "stage36_colab_data_manifest.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_stage36_colab_notebook_is_valid_and_self_contained() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    prose = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert sum(cell.cell_type == "code" for cell in notebook.cells) >= 20
    assert sum(cell.cell_type == "markdown" for cell in notebook.cells) >= 20
    assert "from strategies" not in code
    assert "import strategies" not in code
    assert "\f" not in prose
    assert "\t" not in prose
    for function_name in (
        "load_monthly_asset_returns",
        "build_macro_probabilities",
        "build_daily_stress_features",
        "estimate_conditional_moments",
        "build_daily_technical_features",
        "build_monthly_fundamental_signals",
        "build_monthly_asset_volatility_signals",
        "solve_weights",
        "run_backtest",
        "paired_block_bootstrap",
        "asset_risk_predictive_regressions",
        "run_stage36_research",
    ):
        assert f"def {function_name}" in code

    for dependency in (
        "stage35/",
        "stage20/",
        "stage13/",
        "stage14/",
        "stage07/",
        "stage34/",
        "stage30/",
        "core/regime_research.py",
    ):
        assert dependency in prose
    assert "데이터 ZIP 하나" in prose
    assert "DΣD" in prose
    assert "μ를 조정하지 않습니다" in prose
    for explanation in (
        "Stage35가 본체이고 Stage36은 위험센서다",
        "전체 입력변수 지도",
        "경제 질문을 두 축으로 압축한다",
        "하드 라벨 대신 혼합상태를 쓰는 이유",
        "표본이 적으면 전체시장 통계로 수축한다",
        "기술신호는 새 기대수익을 더하지 않고",
        "신용스프레드: 금융시장의 혈압",
        "`μ35`가 실제 코드에서 조립되는 정확한 순서",
        "전체 알고리즘을 한 줄씩 추적하기",
    ):
        assert explanation in prose


def test_stage36_colab_bundle_contains_data_only_and_valid_hashes() -> None:
    external_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert external_manifest["bundle"] == "Stage36_GVZ_OVX_Colab_Data"
    assert external_manifest["code_included"] is False
    assert len(external_manifest["files"]) == 16

    with zipfile.ZipFile(BUNDLE) as archive:
        names = set(archive.namelist())
        assert "stage36_data/manifest.json" in names
        assert not any(
            name.lower().endswith((".py", ".ipynb", ".html"))
            for name in names
        )
        embedded = json.loads(
            archive.read("stage36_data/manifest.json").decode("utf-8")
        )
        for key in (
            "bundle",
            "schema_version",
            "root_directory",
            "code_included",
            "files",
        ):
            assert embedded[key] == external_manifest[key]
        for record in embedded["files"]:
            payload = archive.read(f"stage36_data/{record['path']}")
            assert len(payload) == record["bytes"]
            assert _sha256_bytes(payload) == record["sha256"]


def test_stage36_colab_has_reproduction_and_constraint_audits() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    for expected_check in (
        "signal_month_precedes_target",
        "no_backfill_before_252",
        "no_directional_gvz_ovx_mu",
        "no_leverage_long_only_sum_to_one",
        "all_solvers_feasible",
    ):
        assert expected_check in code
    assert "STAGE36_BOOTSTRAP_REPS" in code
    assert '"2000"' in code
    assert "0.1049938875" in code
    assert "1.1049112901" in code
    assert "-0.1240708722" in code
