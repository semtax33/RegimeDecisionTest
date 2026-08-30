# Zero-Tune VKOSPI

이 폴더는 현재 기준 전략과 분리한 **비교용 제로튜닝 구현**이다. 목적은 더 좋은
백테스트 숫자를 고르는 것이 아니라, 성과를 본 뒤 바꿀 수 있는 자유 파라미터를
없앴을 때 성과가 어디까지 남는지 측정하는 데 있다.

## 실행

프로젝트 루트에서 다음 명령을 실행한다.

    & 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' -m strategies.stage07_zero_tune_vkospi.zero_tune_strategy

결과는 이 폴더의 outputs 폴더에 저장된다.

## 사전 고정 규칙

1. 여섯 거시 수준값을 각 계열의 **인과적 expanding empirical mid-rank**로
   변환한다. 창 길이, 최소 관측치, 감쇠, clipping, z-score scale은 없다.
2. GDP·수출·BSI rank를 똑같이 평균해 성장확률로, CPI·PPI·수입물가 rank를
   똑같이 평균해 물가확률로 쓴다. 3개월 변화, sigmoid scale, 전월확률
   smoothing은 없다.
3. 사분면 확률 자체를 자산비중으로 쓴다.
   Goldilocks=KODEX200, Overheating=USO, Slowdown=BOND,
   Stagflation=GLD다. anchor와 SLSQP는 없다.
4. 로지스틱 꼬리위험, 손실 임계값, 예측 horizon, 정규화 C,
   class weight, 변동성 타깃, 레버리지는 전부 제거한다.
5. VKOSPI 종가의 인과적 expanding percentile을 그대로 스트레스 비율로 쓴다.
   창 길이, 수준·충격 임계값, 모멘텀, 가속도, 특징 혼합비가 없다.
6. 그 비율만큼 KODEX200·USO를 줄이고 빠진 비중은 BOND·GLD에 똑같이
   나눈다. 최대 이전비율, bond share, rebalance band는 없다.
7. 월간 거시신호는 목표월보다 앞서고, 일간 VKOSPI는 행동일 전일까지의 값만
   사용한다. 총 자산비중은 항상 1이므로 레버리지와 차입비용이 없다.

## 무엇이 하이퍼파라미터가 아닌가

“제로튜닝”은 가정이 0개라는 뜻이 아니다. 다음은 전략 후보를 고르기 위한
하이퍼파라미터가 아니라 비교를 성립시키는 데이터·집행 규약으로 고정했다.

- 자산군과 경제 사분면의 의미적 연결
- 월간 거시자료와 일간 VKOSPI라는 관측 빈도
- 다음 시가부터 수익을 적용하는 인과적 실행 순서
- 기준 전략과 동일한 국내 15bp 및 해외비중 변화 5bp 비용
- 수출증가율을 전년동월 대비로 정의하는 12개월 자료 변환

이 규약까지 없애면 어떤 자산을 언제 거래했는지 정의할 수 없으므로 백테스트
자체가 성립하지 않는다. 대신 이 구현에는 성과를 보고 고를 후보군, 탐색 grid,
학습된 threshold 또는 조정 가능한 Config가 없다.

## 측정 결과

공통 비교기간은 2007-04~2026-07, 232개월이다.

| 전략 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| Zero-Tune VKOSPI | 7.14% | 0.845 | -14.01% |
| 현재 Robust VKOSPI | 15.64% | 1.133 | -12.96% |

2018-01~2026-07 구간:

| 전략 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| Zero-Tune VKOSPI | 7.04% | 1.134 | -14.01% |
| 현재 Robust VKOSPI | 20.91% | 1.497 | -9.76% |

따라서 현재 데이터에서는 하이퍼파라미터를 없앤 쪽이 세 핵심 지표 모두
나빠졌다. 이 결과는 현재 전략의 모든 숫자가 정당하다는 증거는 아니다.
다만 “어떠한 조정값도 두지 않는 규칙”이 자동으로 더 robust하거나 더 좋은
성과를 내지는 않는다는 비교 기준을 제공한다.

## 파일

- zero_tune_strategy.py: 전체 계산과 실행 진입점
- design_spec.json: 제로튜닝 계약과 남겨 둔 실행 가정
- outputs/performance_comparison.csv: 동일기간 성과표
- outputs/zero_tune_report.json: 실행 규칙과 검증 결과
- outputs/zero_tune_monthly.csv: 월별 최종 수익률
- outputs/zero_tune_daily.csv: 일별 신호·비중·비용
