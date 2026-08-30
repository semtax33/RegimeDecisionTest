# RegimeDecisionTest

한국 거시 국면 기반 4자산 배분에서 출발해, 꼬리위험·Open Asset Pricing·VKOSPI·VIX6·KOSPI200 옵션까지 순서대로 검증한 연구 저장소입니다.

## 먼저 볼 것

```powershell
$py = "D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe"

# 전략 시도 순서와 현재 판정
& $py run_strategy.py list

# 전체 회귀 테스트
& $py run_strategy.py test

# 구조·보고서 링크·노트북·ZIP·JSON 검사
& $py run_strategy.py health

# 예: 10번 VIX6 조건부 라우터 재실행
& $py run_strategy.py run 10
```

상세한 발전 과정은 [전략 로드맵](docs/STRATEGY_ROADMAP.md), 결과 파일은 [결과 인덱스](docs/RESULTS_INDEX.md), 옛 경로와 새 경로의 대응은 [리팩터링 매니페스트](docs/refactor_manifest.json)에서 확인할 수 있습니다. 작업 내용과 검증 결과는 [리팩터링 보고서](docs/REFACTOR_REPORT.md), 업로드 노트북과의 동일 구간 비교는 [업로드 노트북 비교](docs/UPLOADED_NOTEBOOK_COMPARISON.md)에 따로 정리했습니다.

## 디렉터리

```text
RegimeDecisionTest/
├─ strategies/               # 시간순 전략 코드
│  ├─ core/                  # 공통 거시·백테스트 엔진
│  ├─ stage01_baseline/
│  ├─ stage02_return_enhancement/
│  ├─ stage03_tail_risk/
│  ├─ stage04_ml_feedback/
│  ├─ stage05_openassetpricing/
│  ├─ stage06_vkospi/
│  ├─ stage07_regime_models/
│  ├─ stage08_options/
│  ├─ stage09_hysteresis/
│  └─ stage10_vix6_router/
├─ tests/                    # 회귀·인과성·산출물 테스트
├─ tools/builders/           # HTML·노트북·Colab 번들 생성기
├─ artifacts/                # 사람이 읽거나 내려받는 완성물
│  ├─ reports/
│  ├─ notebooks/
│  ├─ bundles/
│  └─ archive/               # 리팩터링 전 Python 백업
├─ raw_data/                 # 원천자료
├─ cache/                    # 시장 데이터 캐시
├─ results/                  # 코드 호환성을 위해 유지한 평면 산출물 데이터 레이크
├─ docs/                     # 로드맵·결과 인덱스·이동 기록
└─ run_strategy.py           # 단일 실행 진입점
```

## 현재 판단

- 현재 비교 기준: 무SJM 거시 국면 + 균형 L2 로지스틱 + Robust VKOSPI.
- 전체 2007-04–2026-07: CAGR 15.64%, Sharpe 1.133, MDD -12.96%.
- 2018-01–2026-07: CAGR 20.91%, Sharpe 1.497, MDD -9.76%.
- 노트북식 히스테리시스 + 상한 1.0은 전체 Sharpe와 MDD를 개선했지만 CAGR과 2018년 이후 Sharpe가 낮아 현재 전략을 교체하지 않았습니다.
- VIX6 조건부 위기 라우터와 Put/Call Spread·Covered Call도 사전 채택 조건을 통과하지 못해, 성과가 같은 Robust VKOSPI 안전 폴백을 유지했습니다.
- 업로드한 Hard 4국면 노트북은 동일한 2007-04–2026-07 입력 기준으로 CAGR 20.46%, Sharpe 0.929, MDD -25.25%였습니다. CAGR은 높지만 변동성·낙폭·위험조정 성과가 나빠 현재 전략의 대체안이 아니라 공격형 비교선으로 분류했습니다.

수치는 프로젝트 공통 월수익 정의를 따르며, 과거 시뮬레이션은 미래 성과를 보장하지 않습니다.
