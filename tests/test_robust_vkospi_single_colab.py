from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "artifacts"
    / "notebooks"
    / "robust_vkospi_reference_single_colab.ipynb"
)
DATA_BUNDLE = ROOT / "artifacts" / "bundles" / "robust_vkospi_colab_data.zip"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_exposes_modular_execution_code() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert len(notebook["cells"]) == 33
    all_source = "\n".join(str(cell["source"]) for cell in notebook["cells"])

    assert "files.upload()" in all_source
    assert "EMBEDDED_BUNDLE_B64" not in all_source
    assert "base64.b64decode" not in all_source
    assert "from strategies" not in all_source
    assert "import strategies" not in all_source

    for definition in (
        "def load_monthly_asset_returns",
        "def controlled_weights",
        "def build_no_sjm_signals",
        "def build_domestic_features",
        "def fit_logistic_candidate",
        "def run_factor_vol_target",
        "def build_robust_daily_features",
        "def run_robust_vkospi_overlay",
    ):
        assert definition in all_source

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])


def test_uploaded_bundle_contains_data_only() -> None:
    with zipfile.ZipFile(DATA_BUNDLE) as archive:
        names = set(archive.namelist())

    assert len(names) == 11
    assert not any(name.lower().endswith(".py") for name in names)
    assert not any("/results/" in name for name in names)
    assert not any("expected" in name.lower() for name in names)
    assert {
        "RegimeDecisionData/raw_data/compass.db",
        "RegimeDecisionData/raw_data/VKOSPIData.csv",
        "RegimeDecisionData/raw_data/krx_bond_index.csv",
        "RegimeDecisionData/cache/market_daily.csv",
        "RegimeDecisionData/input_data/openassetpricing_composites.csv",
    } <= names
