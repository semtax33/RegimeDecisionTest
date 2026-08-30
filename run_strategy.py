from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from strategies.catalog import STAGE_BY_ID, STAGES


ROOT = Path(__file__).resolve().parent


def list_stages() -> None:
    print("ID  상태               전략 단계")
    print("--  -----------------  --------------------------------")
    for stage in STAGES:
        print(f"{stage.id}  {stage.status:<17}  {stage.title}")
        print(f"    {stage.outcome}")


def run_module(module: str) -> int:
    return subprocess.call([sys.executable, "-m", module], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RegimeDecisionTest 전략 카탈로그·실행·테스트 진입점"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="전략 시도 순서를 표시")
    run_parser = subparsers.add_parser("run", help="대표 전략을 실행")
    run_parser.add_argument("stage", choices=sorted(STAGE_BY_ID))
    build_parser = subparsers.add_parser("build", help="단계 보고서/노트북을 재생성")
    build_parser.add_argument("stage", choices=sorted(STAGE_BY_ID))
    subparsers.add_parser("test", help="전체 회귀 테스트 실행")
    subparsers.add_parser("health", help="구조·링크·JSON·ZIP 건강검사")
    args = parser.parse_args()

    if args.command == "list":
        list_stages()
        return 0
    if args.command == "run":
        return run_module(STAGE_BY_ID[args.stage].module)
    if args.command == "build":
        builder = STAGE_BY_ID[args.stage].builder
        if builder is None:
            parser.error(f"stage {args.stage}에는 등록된 builder가 없습니다")
        return run_module(builder)
    if args.command == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)
    if args.command == "health":
        return run_module("tools.repository_health")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

