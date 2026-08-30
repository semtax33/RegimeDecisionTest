# 전략 발전 로드맵

아래 순서는 파일 수정시각과 실험 의존관계를 함께 반영했습니다. `status`는 성과를 숨기지 않고 현재 코드에서 실제로 어떻게 취급하는지 나타냅니다.

| 단계 | 핵심 시도 | 대표 결과 | status | 코드 |
|---:|---|---|---|---|
| 01 | 성장·물가 4국면, Sparse Jump Model, SLSQP 방어배분 | Proposed 전체 CAGR 8.12%, Sharpe 1.144, MDD -8.93% | baseline | `strategies/core`, `stage01_baseline` |
| 02 | 목표변동성·모멘텀·Hard 국면·레버리지로 CAGR 강화 | 후속 MDD 제약 전략의 재료가 된 탐색 단계 | superseded | `stage02_return_enhancement` |
| 03 | Crash 특징, 일별 guard, stop-loss, synthetic put, 최종 blend | 전체 CAGR 14.30%, Sharpe 1.024, MDD -14.90%; 네 gate 통과 | milestone | `stage03_tail_risk` |
| 04 | LightGBM, 피드백 대안, tail meta, 시장구조 | LightGBM 승자 `max_shift=0`; 잠금 개선 실패 | not promoted | `stage04_ml_feedback` |
| 05 | Open Asset Pricing 아이디어를 한국 시장 합성변수로 변환 | OAP 합성 4개가 후속 균형 L2 로지스틱 입력에 편입 | feature adopted | `stage05_openassetpricing` |
| 06 | VKOSPI 수준·충격·가속, Robust 일간 위험예산 | 현재 비교 기준 전체 15.64% / 1.133 / -12.96% | current reference | `stage06_vkospi` |
| 07 | CJM, TVTP-HMM, CJM+LightGBM | 기준 대비 CAGR·Sharpe·MDD 동시 개선 후보 없음 | not promoted | `stage07_regime_models` |
| 08 | VIX6 decomposition, long KOSPI200 put, 슬리피지 | 강건 옵션 후보 0개; 옵션 비중 0% 유지 | not promoted | `stage08_options` |
| 09 | ±0.2 히스테리시스, Hard 40%, 레버리지 상한 1.0~1.3 | 상한 1.0은 Sharpe·MDD 개선/CAGR 하락; 사전 gate 실패 | research only | `stage09_hysteresis` |

## 외부 비교선: 업로드 노트북

`economic_regime_asset_allocation_backtest_multi_asset.ipynb`는 성장·물가 합성점수에 ±0.2 히스테리시스를 적용하고, 네 국면마다 한 가지 Hard 자산배분을 사용합니다. 학습 모델, 국면확률, SLSQP, VKOSPI/VIX6 오버레이는 사용하지 않습니다. 프로젝트와 같은 2007-04–2026-07 월수익·비용 입력으로 재계산한 결과는 CAGR 20.46%, Sharpe 0.929, MDD -25.25%였습니다. 현재 기준보다 CAGR은 4.82%p 높지만 Sharpe는 0.204 낮고 MDD는 12.29%p 더 깊습니다. 자세한 비교와 재현 조건은 [`UPLOADED_NOTEBOOK_COMPARISON.md`](UPLOADED_NOTEBOOK_COMPARISON.md)에 있습니다.

## 최종 전략을 읽는 순서

1. `strategies/core/regime_research.py` — 거시 입력, 확률, SLSQP, 기본 백테스트.
2. `strategies/stage05_openassetpricing/openassetpricing_signal_experiment.py` — OAP 합성 입력.
3. `strategies/stage06_vkospi/balanced_logistic_no_sjm_strategy.py` — Hard 40% + SLSQP 60% + 균형 L2 로지스틱.
4. `strategies/stage06_vkospi/vkospi_robust_dynamic_experiment.py` — Robust VKOSPI 일간 스트레스.
5. `strategies/stage08_options/option_asset_slippage_experiment.py` — 옵션을 채택하지 않은 이유.
6. `strategies/stage09_hysteresis/hysteresis_hard40_leverage_experiment.py` — 최신 히스테리시스/상한 ablation.

## 해석상 주의

- 무SJM 경로는 사용자 요청에 따른 post-lock ablation이며 2007–2017 사전 승격 gate를 통과하지 않았습니다. 그래서 `current_reference`는 현재 비교 기준이라는 뜻이지, 깨끗한 신규 홀드아웃 승자라는 뜻은 아닙니다.
- 2018년 이후는 여러 후속 연구 과정에서 이미 관측됐습니다. 최신 실험은 파라미터 선택에 2018년 이후를 쓰지 않았지만, 저장소 전체를 완전한 미관측 holdout 연구로 볼 수는 없습니다.
- Sharpe는 월수익을 연율화한 프로젝트 공통 정의이며 별도 무위험 초과수익 차감은 없습니다.
