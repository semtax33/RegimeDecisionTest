from __future__ import annotations

import ast
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
NOTEBOOKS = ROOT / "artifacts" / "notebooks"
BUNDLES = ROOT / "artifacts" / "bundles"


def _python_checks(errors: list[str]) -> tuple[int, int]:
    files = sorted(
        path
        for directory in (ROOT / "strategies", ROOT / "tools", ROOT / "tests")
        for path in directory.rglob("*.py")
    )
    internal_stems = {path.stem for path in files}
    bare_imports = 0
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            errors.append(f"python parse: {path.relative_to(ROOT)}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in internal_stems:
                bare_imports += 1
                errors.append(f"bare internal import: {path.relative_to(ROOT)} -> {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in internal_stems:
                        bare_imports += 1
                        errors.append(f"bare internal import: {path.relative_to(ROOT)} -> {alias.name}")
    return len(files), bare_imports


def _html_checks(errors: list[str]) -> tuple[int, int]:
    reports = sorted(REPORTS.glob("*.html"))
    checked_links = 0
    for report in reports:
        text = report.read_text(encoding="utf-8")
        if re.search(r"__[A-Z][A-Z0-9_]+__", text):
            errors.append(f"unresolved template token: {report.relative_to(ROOT)}")
        for raw_link in re.findall(r'href=["\']([^"\']+)["\']', text):
            if raw_link.startswith(("#", "http://", "https://", "mailto:", "javascript:")):
                continue
            target_text = unquote(raw_link.split("#", 1)[0])
            if not target_text:
                continue
            checked_links += 1
            target = (report.parent / target_text).resolve()
            if not target.exists():
                errors.append(f"broken local link: {report.name} -> {raw_link}")
    return len(reports), checked_links


def _markdown_checks(errors: list[str]) -> tuple[int, int]:
    markdown_files = sorted(
        {
            ROOT / "README.md",
            ROOT / "results" / "README.md",
            ROOT / "artifacts" / "README.md",
            *list((ROOT / "docs").glob("*.md")),
            *list((ROOT / "strategies").rglob("README.md")),
        }
    )
    checked_links = 0
    for document in markdown_files:
        if not document.exists():
            continue
        text = document.read_text(encoding="utf-8")
        for raw_link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            target_text = raw_link.strip().strip("<>").split("#", 1)[0]
            if not target_text or target_text.startswith(
                ("#", "http://", "https://", "mailto:")
            ):
                continue
            checked_links += 1
            target = (document.parent / unquote(target_text)).resolve()
            if not target.exists():
                errors.append(
                    f"broken markdown link: {document.relative_to(ROOT)} -> {raw_link}"
                )
    return len([path for path in markdown_files if path.exists()]), checked_links


def _artifact_checks(errors: list[str]) -> tuple[int, int, int]:
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if notebook.get("nbformat") != 4:
                errors.append(f"unexpected notebook format: {path.name}")
        except Exception as exc:
            errors.append(f"notebook json: {path.name}: {exc}")

    bundles = sorted(BUNDLES.glob("*.zip"))
    for path in bundles:
        try:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
            if bad is not None:
                errors.append(f"corrupt zip member: {path.name} -> {bad}")
        except Exception as exc:
            errors.append(f"zip: {path.name}: {exc}")

    json_files = sorted((ROOT / "results").glob("*.json"))
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"result json: {path.name}: {exc}")
    return len(notebooks), len(bundles), len(json_files)


def main() -> None:
    errors: list[str] = []
    python_count, bare_imports = _python_checks(errors)
    report_count, link_count = _html_checks(errors)
    markdown_count, markdown_link_count = _markdown_checks(errors)
    notebook_count, bundle_count, json_count = _artifact_checks(errors)
    root_code = sorted(path.name for path in ROOT.glob("*.py") if path.name != "run_strategy.py")
    if root_code:
        errors.append(f"unexpected root Python files: {root_code}")

    print("=== REPOSITORY HEALTH ===")
    print(f"python={python_count}, bare_internal_imports={bare_imports}")
    print(f"reports={report_count}, checked_local_links={link_count}")
    print(f"markdown={markdown_count}, checked_local_links={markdown_link_count}")
    print(f"notebooks={notebook_count}, bundles={bundle_count}, result_json={json_count}")
    if errors:
        print(f"FAIL ({len(errors)})")
        for error in errors:
            print("-", error)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
