from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from strategies.catalog import STAGES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

STAGE_LABELS = {
    "01": "거시 국면 베이스라인",
    "02": "CAGR·Hard 배분 강화",
    "03": "꼬리위험·MDD 15%",
    "04": "ML·피드백·시장구조",
    "05": "Open Asset Pricing",
    "06": "VKOSPI·Robust VKOSPI",
    "07": "CJM·TVTP-HMM 비교",
    "08": "VIX6·옵션",
    "09": "히스테리시스·상한",
    "10": "VIX6 조건부 위기 라우터",
    "other": "공통·미분류",
}


def classify_result(name: str) -> str:
    if name.startswith("vix6_router_"):
        return "10"
    if name.startswith("hysteresis_"):
        return "09"
    if name.startswith(("vix6_", "option_")):
        return "08"
    if name.startswith("top3_"):
        return "07"
    if name.startswith(("vkospi_", "balanced_logistic_")):
        return "06"
    if name.startswith("openassetpricing_"):
        return "05"
    if name.startswith(
        (
            "regime_lightgbm_",
            "feedback_",
            "alternative_",
            "short_",
            "market_",
            "final_blend_crash_meta_",
        )
    ):
        return "04"
    if name.startswith(
        (
            "hard_",
            "daily_",
            "leveraged_",
            "blend_",
            "simple_",
            "final_blend_",
            "crisis_",
            "bootstrap_",
            "cost_",
        )
    ):
        return "03"
    if name.startswith(("adaptive_", "cagr_")):
        return "02"
    if name.startswith(("regime_", "proposed_", "summary.", "config.")):
        return "01"
    return "other"


def first_description(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
    except Exception:
        docstring = None
    if docstring:
        return " ".join(docstring.strip().split())
    return path.stem.replace("_", " ")


def build_stage_readmes() -> None:
    stage_by_dir = {
        "stage01_baseline": STAGES[0],
        "stage02_return_enhancement": STAGES[1],
        "stage03_tail_risk": STAGES[2],
        "stage04_ml_feedback": STAGES[3],
        "stage05_openassetpricing": STAGES[4],
        "stage06_vkospi": STAGES[5],
        "stage07_regime_models": STAGES[6],
        "stage08_options": STAGES[7],
        "stage09_hysteresis": STAGES[8],
    }
    for directory, stage in stage_by_dir.items():
        target = ROOT / "strategies" / directory
        files = sorted(path for path in target.glob("*.py") if path.name != "__init__.py")
        lines = [
            f"# Stage {stage.id} — {stage.title}",
            "",
            stage.objective,
            "",
            f"- 상태: `{stage.status}`",
            f"- 대표 결과: {stage.outcome}",
            f"- 대표 실행: `python -m {stage.module}`",
        ]
        if stage.report:
            lines.append(f"- 보고서: [`{Path(stage.report).name}`](../../{stage.report})")
        lines.extend(["", "## 파일", ""])
        for path in files:
            lines.append(f"- [`{path.name}`]({path.name}) — {first_description(path)}")
        lines.append("")
        (target / "README.md").write_text("\n".join(lines), encoding="utf-8")

    core = ROOT / "strategies" / "core"
    (core / "README.md").write_text(
        "# Core engine\n\n"
        "모든 단계가 공유하는 거시 데이터 로더, Sparse Jump Model, 국면 확률, "
        "SLSQP 자산배분, 성과 측정과 월별 백테스트 엔진입니다.\n\n"
        "- [`regime_research.py`](regime_research.py)\n",
        encoding="utf-8",
    )


def build_results_index() -> None:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(RESULTS.iterdir()):
        if path.is_file() and path.name != "README.md":
            groups[classify_result(path.name)].append(path)

    key_files = [
        ("01", "summary.csv", "기본 6개 전략 성과"),
        ("03", "final_blend_validation.json", "MDD 15% final blend 검증"),
        ("04", "regime_lightgbm_validation.json", "LightGBM 비승격 판정"),
        ("05", "openassetpricing_validation.json", "OAP 입력 선택"),
        ("06", "vkospi_robust_dynamic_validation.json", "Robust VKOSPI 선택·잠금"),
        ("06", "balanced_logistic_no_sjm_validation.json", "현재 비교 기준"),
        ("07", "top3_regime_model_validation.json", "CJM·TVTP 비교"),
        ("08", "option_asset_slippage_validation.json", "옵션 비승격 판정"),
        ("09", "hysteresis_hard40_leverage_validation.json", "최신 히스테리시스 실험"),
        ("10", "vix6_router_validation.json", "VIX6 조건부 라우터 선택·잠금"),
        ("10", "vix6_router_option_validation.json", "상태별 옵션 구조 비승격 판정"),
    ]
    lines = [
        "# Results index",
        "",
        f"`results`에는 경로 호환성을 위해 {sum(len(v) for v in groups.values())}개 산출물을 평면 구조로 유지합니다.",
        "",
        "## 가장 먼저 볼 파일",
        "",
        "| 단계 | 파일 | 의미 |",
        "|---:|---|---|",
    ]
    for stage, name, meaning in key_files:
        lines.append(f"| {stage} | [`{name}`](../results/{name}) | {meaning} |")
    lines.extend(["", "## 단계별 개수", "", "| 단계 | 분류 | 파일 수 | 용량 |", "|---:|---|---:|---:|"])
    for stage in (*[f"{number:02d}" for number in range(1, 11)], "other"):
        paths = groups.get(stage, [])
        size_mb = sum(path.stat().st_size for path in paths) / (1024 * 1024)
        lines.append(f"| {stage} | {STAGE_LABELS[stage]} | {len(paths)} | {size_mb:.2f} MB |")

    lines.extend(["", "## 전체 파일", ""])
    for stage in (*[f"{number:02d}" for number in range(1, 11)], "other"):
        paths = groups.get(stage, [])
        if not paths:
            continue
        lines.append(f"<details><summary>{stage} · {STAGE_LABELS[stage]} ({len(paths)}개)</summary>")
        lines.append("")
        for path in paths:
            lines.append(f"- [`{path.name}`](../results/{path.name}) — {path.stat().st_size:,} bytes")
        lines.extend(["", "</details>", ""])
    (ROOT / "docs" / "RESULTS_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    build_stage_readmes()
    build_results_index()
    print("generated stage READMEs and docs/RESULTS_INDEX.md")


if __name__ == "__main__":
    main()
