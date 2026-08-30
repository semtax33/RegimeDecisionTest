from __future__ import annotations

import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STAGES: dict[str, list[str]] = {
    "strategies/core": [
        "regime_research.py",
    ],
    "strategies/stage01_baseline": [
        "calibrate_configs.py",
    ],
    "strategies/stage02_return_enhancement": [
        "analyze_cagr_constraint.py",
        "adaptive_vol_experiment.py",
        "validate_cagr_accelerator.py",
        "cagr_accelerator_experiment.py",
        "analyze_hard_strategy.py",
        "hard_overlay_experiment.py",
    ],
    "strategies/stage03_tail_risk": [
        "build_crash_features.py",
        "hard_crash_model_experiment.py",
        "daily_hard_overlay_experiment.py",
        "leveraged_daily_overlay_experiment.py",
        "daily_stoploss_experiment.py",
        "hard_crash_rank_experiment.py",
        "daily_guard_experiment.py",
        "hard_crash_short_experiment.py",
        "evaluate_hard_crash_short_boundary.py",
        "download_stress_data.py",
        "synthetic_put_overlay_experiment.py",
        "evaluate_rank_mdd15.py",
        "blend_leverage_experiment.py",
        "simple_risk_overlay_experiment.py",
        "evaluate_simple_risk_mdd12.py",
        "evaluate_blend_mdd15_return.py",
        "validate_final_blend.py",
    ],
    "strategies/stage04_ml_feedback": [
        "regime_lightgbm_factor_experiment.py",
        "feedback_alternative_strategies_experiment.py",
        "feedback_strategy_robustness.py",
        "short_regime_tail_risk_experiment.py",
        "final_blend_crash_meta_experiment.py",
        "final_blend_crash_meta_robustness.py",
        "market_structure_feature_experiment.py",
        "market_structure_robustness.py",
    ],
    "strategies/stage05_openassetpricing": [
        "openassetpricing_signal_experiment.py",
    ],
    "strategies/stage06_vkospi": [
        "vkospi_feature_experiment.py",
        "vkospi_dynamic_risk_experiment.py",
        "vkospi_reprocessing_experiment.py",
        "vkospi_robust_dynamic_experiment.py",
        "vkospi_robust_dynamic_attribution.py",
        "vkospi_extended_diagnostics.py",
        "vkospi_model_robustness.py",
        "balanced_logistic_no_sjm_strategy.py",
    ],
    "strategies/stage07_regime_models": [
        "top3_regime_model_experiment.py",
    ],
    "strategies/stage08_options": [
        "vix6_case1_strategy.py",
        "vix6_case1_model_comparison.py",
        "vix6_processed_input_experiment.py",
        "option_asset_slippage_experiment.py",
    ],
    "strategies/stage09_hysteresis": [
        "hysteresis_hard40_leverage_experiment.py",
    ],
}

BUILDERS = [
    "build_validated_notebook.py",
    "build_cagr_enhanced_notebook.py",
    "build_final_blend_notebook.py",
    "build_colab_notebook.py",
    "build_implementation_guide.py",
    "build_openassetpricing_colab_notebook.py",
    "build_vkospi_dynamic_deliverables.py",
    "build_vkospi_robust_dynamic_deliverables.py",
    "build_vkospi_robust_dynamic_technical_report.py",
    "build_top3_regime_model_deliverables.py",
    "build_vix6_case1_html.py",
    "build_robust_vkospi_implementation_guide.py",
    "build_hysteresis_hard40_leverage_deliverables.py",
]

TESTS = [
    "test_vkospi_feature_experiment.py",
    "test_vkospi_dynamic_risk_experiment.py",
    "test_top3_regime_model_experiment.py",
    "test_vkospi_robust_dynamic_experiment.py",
    "test_vkospi_extended_diagnostics.py",
    "test_vkospi_model_robustness.py",
    "test_balanced_logistic_no_sjm_strategy.py",
    "test_vix6_case1_strategy.py",
    "test_vix6_processed_input_experiment.py",
    "test_option_asset_slippage_experiment.py",
    "test_hysteresis_hard40_leverage_experiment.py",
]


def _python_moves() -> dict[str, str]:
    moves: dict[str, str] = {}
    for directory, names in STAGES.items():
        for name in names:
            moves[name] = f"{directory}/{name}"
    moves.update({name: f"tools/builders/{name}" for name in BUILDERS})
    moves.update({name: f"tests/{name}" for name in TESTS})
    return moves


def _artifact_directory(suffix: str) -> str:
    return {
        ".html": "artifacts/reports",
        ".ipynb": "artifacts/notebooks",
        ".zip": "artifacts/bundles",
    }[suffix]


def _package_name(relative_path: str) -> str:
    return relative_path.removesuffix(".py").replace("/", ".")


def _replace_imports(text: str, module_paths: dict[str, str]) -> str:
    for old_module, new_module in sorted(module_paths.items(), key=lambda item: -len(item[0])):
        text = re.sub(
            rf"(?m)^(\s*)from\s+{re.escape(old_module)}\s+import\s+",
            rf"\1from {new_module} import ",
            text,
        )
        text = re.sub(
            rf"(?m)^(\s*)import\s+{re.escape(old_module)}\s+as\s+(\w+)\s*$",
            rf"\1import {new_module} as \2",
            text,
        )
        text = re.sub(
            rf"(?m)^(\s*)import\s+{re.escape(old_module)}\s*$",
            rf"\1import {new_module} as {old_module}",
            text,
        )
    return text


def _rewrite_python(
    path: Path,
    relative: str,
    python_moves: dict[str, str],
    artifact_moves: dict[str, str],
) -> None:
    text = path.read_text(encoding="utf-8")
    module_paths = {
        Path(old).stem: _package_name(new) for old, new in python_moves.items()
    }
    text = _replace_imports(text, module_paths)

    root_expression = (
        "ROOT = Path(__file__).resolve().parents[1]"
        if relative.startswith("tests/")
        else "ROOT = Path(__file__).resolve().parents[2]"
    )
    text = re.sub(
        r"(?m)^ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parent\s*$",
        root_expression,
        text,
    )

    if relative.startswith("tools/builders/"):
        for old, new in python_moves.items():
            text = text.replace(f'"{old}"', f'"{new}"')
            text = text.replace(f"'{old}'", f"'{new}'")

    for old, new in artifact_moves.items():
        text = text.replace(f'ROOT / "{old}"', f'ROOT / "{new}"')
        text = text.replace(f"ROOT / '{old}'", f"ROOT / '{new}'")

    if relative.startswith("tools/builders/"):
        for old, new in python_moves.items():
            text = text.replace(f'href="{new}"', f'href="../../{new}"')
        for name in artifact_moves:
            suffix = Path(name).suffix.lower()
            if suffix == ".ipynb":
                text = text.replace(f'href="{name}"', f'href="../notebooks/{name}"')
            elif suffix == ".zip":
                text = text.replace(f'href="{name}"', f'href="../bundles/{name}"')
        text = text.replace('href="results/', 'href="../../results/')

    path.write_text(text, encoding="utf-8")


def _rewrite_report(path: Path, python_moves: dict[str, str], artifact_names: set[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in python_moves.items():
        text = text.replace(f'href="{old}"', f'href="../../{new}"')
    for name in artifact_names:
        suffix = Path(name).suffix.lower()
        if suffix == ".ipynb":
            text = text.replace(f'href="{name}"', f'href="../notebooks/{name}"')
        elif suffix == ".zip":
            text = text.replace(f'href="{name}"', f'href="../bundles/{name}"')
    text = text.replace('href="results/', 'href="../../results/')
    path.write_text(text, encoding="utf-8")


def _write_package_files() -> None:
    package_dirs = {
        "strategies",
        *STAGES.keys(),
        "tools",
        "tools/builders",
        "tests",
    }
    for directory in sorted(package_dirs):
        path = ROOT / directory / "__init__.py"
        if not path.exists():
            path.write_text('"""Repository package created by the 2026-08 refactor."""\n', encoding="utf-8")


def main() -> None:
    python_moves = _python_moves()
    root_python = {path.name for path in ROOT.glob("*.py")}
    expected = set(python_moves)
    if root_python != expected:
        missing = sorted(expected - root_python)
        unexpected = sorted(root_python - expected)
        raise RuntimeError(f"Python mapping mismatch; missing={missing}, unexpected={unexpected}")

    artifact_paths = [
        path
        for suffix in ("*.html", "*.ipynb", "*.zip")
        for path in ROOT.glob(suffix)
    ]
    artifact_moves = {
        path.name: f"{_artifact_directory(path.suffix.lower())}/{path.name}"
        for path in artifact_paths
    }

    for directory in (
        *STAGES.keys(),
        "tools/builders",
        "tests",
        "docs",
        "artifacts/reports",
        "artifacts/notebooks",
        "artifacts/bundles",
        "artifacts/archive",
    ):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    snapshot = ROOT / "artifacts/archive/pre_refactor_python_20260828.zip"
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(root_python):
            archive.write(ROOT / name, arcname=name)

    rows: list[dict[str, object]] = []
    for old, new in {**python_moves, **artifact_moves}.items():
        source = ROOT / old
        target = ROOT / new
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(target)
        rows.append(
            {
                "old": old,
                "new": new,
                "bytes": source.stat().st_size,
                "modified": datetime.fromtimestamp(source.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

    manifest_path = ROOT / "docs/refactor_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "source_snapshot": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
                "moves": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for old, new in python_moves.items():
        target = ROOT / new
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ROOT / old), str(target))
    for old, new in artifact_moves.items():
        target = ROOT / new
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ROOT / old), str(target))

    _write_package_files()
    for relative in python_moves.values():
        _rewrite_python(ROOT / relative, relative, python_moves, artifact_moves)
    for name, relative in artifact_moves.items():
        if Path(name).suffix.lower() == ".html":
            _rewrite_report(ROOT / relative, python_moves, set(artifact_moves))

    print(f"moved python={len(python_moves)}, artifacts={len(artifact_moves)}")
    print(f"snapshot={snapshot.relative_to(ROOT)}")
    print(f"manifest={manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
