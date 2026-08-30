# 저장소 리팩터링 및 검증 보고서

## 결과 요약

루트에 섞여 있던 Python 72개와 사람이 읽는 산출물 37개를 역할과 연구 순서에 따라 재배치했습니다. 루트 실행 파일은 `run_strategy.py` 하나만 남겼고, 과거 코드는 ZIP으로 보존했습니다. 전략 로직과 결과 데이터는 삭제하지 않았습니다.

현재 구조에서는 다음 세 명령으로 전체 흐름을 확인할 수 있습니다.

```powershell
$py = "D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe"
& $py run_strategy.py list
& $py run_strategy.py test
& $py run_strategy.py health
```

## 정리 전후

| 항목 | 정리 전 | 정리 후 |
|---|---|---|
| 루트 Python | 72개가 평면 배치 | `run_strategy.py` 1개 |
| 전략 코드 | 연구 순서가 파일명과 수정시각에 흩어짐 | `strategies/stage01`부터 `stage09`까지 시간순 배치 |
| 공통 엔진 | 루트의 `regime_research.py` | `strategies/core/regime_research.py` |
| 테스트 | 루트와 코드 사이에 혼재 | `tests/` 12개 모듈 |
| 문서 생성기 | 전략 코드와 혼재 | `tools/builders/` 13개 생성기 |
| HTML | 루트 13개 | `artifacts/reports/` 13개 |
| 노트북 | 루트 15개 | `artifacts/notebooks/` 15개 |
| ZIP 번들 | 루트 9개 | `artifacts/bundles/` 9개 |
| 결과 데이터 | 평면 `results/` | 기존 상대경로 호환을 위해 평면 구조 유지, 인덱스 추가 |

## 전략 순서

1. `stage01_baseline`: 성장·물가 4국면, Sparse Jump Model, SLSQP 배분
2. `stage02_return_enhancement`: 목표변동성·모멘텀·Hard 국면·레버리지
3. `stage03_tail_risk`: crash 특징, 일간 guard, MDD 15% 제약, final blend
4. `stage04_ml_feedback`: LightGBM, 피드백 대안, 시장구조·tail meta
5. `stage05_openassetpricing`: Open Asset Pricing 아이디어의 한국형 합성입력
6. `stage06_vkospi`: VKOSPI 동적·Robust 오버레이와 현재 비교 기준
7. `stage07_regime_models`: CJM, TVTP-HMM, CJM+LightGBM
8. `stage08_options`: VIX6 decomposition, KOSPI200 옵션, 슬리피지
9. `stage09_hysteresis`: ±0.2 히스테리시스와 레버리지 상한 실험

각 단계의 코드·판정·대표 결과는 [`STRATEGY_ROADMAP.md`](STRATEGY_ROADMAP.md)와 각 stage의 `README.md`에 연결했습니다.

## 경로와 import 정리

- 내부 import를 `strategies.stageNN_...`와 `tools.builders...` 형식의 패키지 경로로 통일했습니다.
- 전략, 테스트, 생성기의 `ROOT` 계산을 새 디렉터리 깊이에 맞췄습니다.
- 생성기를 import했을 때 곧바로 파일을 만들거나 장시간 계산하지 않도록 `main()` 경계를 정리했습니다.
- HTML 생성기의 상대경로를 `artifacts/reports` 기준으로 수정했습니다.
- Colab 노트북의 로컬 스크립트 실행을 파일경로 호출에서 `python -m 패키지.모듈` 방식으로 바꿨습니다.
- 대형 `KOSPI200OptionPrice.csv`가 일반 Colab 입력 번들에 중복 포함되지 않도록 제외했습니다.
- 잘못 잘린 `daily_hard_overlay_winner.json`은 보정 결과의 승자 행으로 복구했습니다.

전체 옛 경로와 새 경로는 [`refactor_manifest.json`](refactor_manifest.json)에 기록했습니다. 리팩터링 전 루트 Python 원본은 `artifacts/archive/pre_refactor_python_20260828.zip`에 보존되어 있습니다.

## 실행 검증

최종 검증 결과는 다음과 같습니다.

| 검사 | 결과 |
|---|---|
| Python 구문 컴파일 | 전략·도구·테스트 90개 PASS |
| 내부 bare import | 0개 |
| pytest 회귀 테스트 | 75 passed, 12.42초 |
| HTML 보고서 | 13개, 로컬 링크 124개 PASS |
| Markdown 문서 | 17개, 로컬 링크 434개 PASS |
| Jupyter notebook JSON | 15개 PASS |
| Colab ZIP 무결성 | 9개 PASS |
| 결과 JSON 파싱 | 38개 PASS |
| 생성기 import 안전성 | 13개 PASS, import 시 생성 부작용 없음 |

실제 전략 실행도 확인했습니다.

- `run_strategy.py run 01`: 거시 256×12, 자산 244×4, 신호 232×16을 생성하고 기준 전략 요약을 정상 출력했습니다.
- `run_strategy.py run 09`: 2007-04–2026-07 전체 및 2018-01–2026-07 비교 결과를 다시 만들었습니다. 현재 전략 수치는 CAGR 15.64%, Sharpe 1.133, MDD -12.96%로 리팩터링 전과 같았습니다.
- Stage 09 Colab 번들을 새 폴더에 풀어 패키지 모듈 실행까지 확인했습니다. 선택 상한 1.0의 전체 CAGR은 13.02%, 잠금 Sharpe는 1.429로 원 결과와 일치했습니다.
- 등록된 산출물 생성 경로 01, 02, 03, 05, 06, 07, 08, 09를 실행했습니다. Stage 04는 별도 배포 생성기가 없는 탐색 단계입니다.

## 업로드 노트북 비교

업로드한 `economic_regime_asset_allocation_backtest_multi_asset.ipynb`도 같은 시장 캐시와 비용 정의로 재계산했습니다. 2007-04–2026-07에서 업로드 Hard 4국면 전략은 CAGR 20.46%, Sharpe 0.929, MDD -25.25%였고, 현재 Robust VKOSPI는 각각 15.64%, 1.133, -12.96%였습니다.

즉 업로드 노트북은 CAGR이 4.82%p 높지만 현재 전략은 Sharpe가 0.204 높고 MDD가 12.29%p 얕습니다. 상세 알고리즘, 2018년 이후, 2025년 말 기준 민감도는 [`UPLOADED_NOTEBOOK_COMPARISON.md`](UPLOADED_NOTEBOOK_COMPARISON.md)에 정리했습니다.

## 남겨 둔 호환성 선택과 한계

- `results/`는 359개 산출물의 기존 참조를 한꺼번에 깨지 않도록 평면 구조를 유지했습니다. 파일별 설명은 [`RESULTS_INDEX.md`](RESULTS_INDEX.md)에 있습니다.
- `_workspace`의 중간 작업물은 삭제하지 않고 `artifacts/archive/humanization_sessions`와 `artifacts/archive/qa_runs`로 옮겼습니다. 현재 `_workspace`는 비어 있습니다.
- 모든 과거 후보 탐색을 처음부터 전부 다시 돌린 것은 아닙니다. 810개 VKOSPI 탐색처럼 비용이 큰 과거 실험은 저장 결과, 회귀 테스트, 대표 전략 재실행, 산출물 생성기로 검증했습니다. 따라서 이번 검증은 코드 이동으로 인한 회귀를 확인한 것이며, 과거 연구의 통계적 유효성을 새로 증명한 것은 아닙니다.
- 현재 무SJM 기준은 post-lock ablation입니다. `current_reference`는 저장소의 비교 기준이라는 뜻이며, 완전한 신규 홀드아웃 승자를 뜻하지 않습니다.
