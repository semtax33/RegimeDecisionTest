# Zero-Tune → Robust VKOSPI 전 조합 실험

이 폴더는 Zero-Tune VKOSPI에서 현재 Robust VKOSPI로 바뀐 부분을 7개의
이진 모듈로 나누고, 가능한 모든 128개 조합을 동일 자료에서 실행한 결과다.

## 조합 번호 읽는 법

조합 번호는 아래 순서의 7자리 비트다. 0은 Zero-Tune 규칙, 1은 현재 Robust
규칙이다.

| 자리 | 변화축 | 0: Zero-Tune | 1: 현재 Robust |
|---:|---|---|---|
| 1 | macro_engine | expanding 거시 rank | z-score·3개월 변화·sigmoid·85/15 평활 |
| 2 | base_allocation | 사분면 확률 직접배분 | hard 40% + SLSQP 60% |
| 3 | tail_logistic | 미사용 | 16변수 균형 L2 로지스틱·20% tilt |
| 4 | vol_target_leverage | 1배 고정 | 15% 변동성 타깃·0.5~1.5배 |
| 5 | vkospi_signal | expanding VKOSPI percentile | 수준·충격·가속도 robust stress |
| 6 | overlay_policy | 전량비례·채권/금 균등·매일 조정 | 최대35%·금 이전·20% 밴드 |
| 7 | frequency_reconciliation | 일간 수익 직접 사용 | 월간 기준경로에 상대효과 결합 |

예를 들어 1101111은 현재 거시·기본배분을 사용하고 tail logistic만 끈 뒤,
변동성 타깃부터 이후 모듈은 현재 방식을 사용한다.

## 끝점 재현

- 0000000은 기존 Zero-Tune 월별 경로와 최대 오차 9.89e-17로 일치했다.
- 1111111의 월간 중간경로는 현재 medium 경로와 최대 오차 9.87e-17이다.
- 1111111의 최종경로는 현재 Robust 경로와 최대 오차 9.89e-17이다.

따라서 128개 조합은 두 다른 전략을 비슷하게 흉내 낸 것이 아니라, 확인된 두
끝점 사이에서 각 모듈만 교체한 결과다.

## 자연스러운 구현 순서로 하나씩 바꾼 결과

2007-04~2026-07 기준이다.

| 단계 | 새로 현재 방식으로 바꾼 부분 | CAGR | Sharpe | MDD |
|---:|---|---:|---:|---:|
| 0 | Zero-Tune | 7.14% | 0.845 | -14.01% |
| 1 | 거시확률 | 8.65% | 0.752 | -17.34% |
| 2 | 기본배분 | 8.57% | 0.971 | -12.96% |
| 3 | 꼬리 로지스틱 | 8.65% | 0.978 | -12.76% |
| 4 | 변동성 타깃·레버리지 | 9.86% | 0.902 | -16.94% |
| 5 | Robust VKOSPI 신호 | 13.09% | 1.017 | -16.49% |
| 6 | 현재 오버레이 정책 | 15.40% | 1.121 | -12.98% |
| 7 | 빈도 reconciliation | 15.64% | 1.133 | -12.96% |

이 표의 증분은 **순서에 의존한다**. 예를 들어 거시 엔진만 Zero-Tune 위에
올리면 Sharpe가 내려가지만, SLSQP 기본배분과 결합한 뒤에는 결과가 달라진다.
따라서 단일 순서표만으로 기여도를 단정하지 않고 모든 순서를 평균한 Shapley
값을 함께 계산했다.

## 순서 중립 Shapley 기여도

전체기간에서 현재 전략과 Zero-Tune의 차이는 CAGR +8.50%p, Sharpe +0.288,
MDD +1.05%p다. MDD 기여도가 양수면 낙폭이 덜 나빠졌다는 뜻이다.

| 변화축 | CAGR 기여 | Sharpe 기여 | MDD 기여 |
|---|---:|---:|---:|
| macro_engine | +1.68%p | -0.008 | -2.27%p |
| base_allocation | +2.22%p | +0.211 | +4.50%p |
| tail_logistic | -0.10%p | -0.003 | +0.39%p |
| vol_target_leverage | +0.78%p | -0.053 | -1.82%p |
| vkospi_signal | +1.50%p | +0.035 | -0.72%p |
| overlay_policy | +2.34%p | +0.092 | +0.98%p |
| frequency_reconciliation | +0.07%p | +0.014 | -0.01%p |

핵심은 다음과 같다.

- Sharpe 개선의 가장 큰 원천은 SLSQP를 포함한 기본배분이다.
- CAGR에는 기본배분과 오버레이 정책의 기여가 가장 크고, 거시엔진과
  VKOSPI 신호가 그다음이다.
- MDD는 기본배분이 크게 개선했지만 거시엔진과 변동성 타깃·레버리지가
  반대 방향으로 작용했다.
- tail logistic은 평균적으로 CAGR·Sharpe 기여가 거의 없거나 소폭
  음수였고, MDD에는 작은 양의 기여가 있었다.
- 기여도의 합은 각 끝점 차이와 수치적으로 일치한다.

2018-01~2026-07에서도 Sharpe 기여는 기본배분 +0.260이 가장 크다.
CAGR에는 오버레이 정책 +3.58%p, 거시엔진 +3.27%p, VKOSPI 신호
+3.25%p가 크게 기여했다.

## 상호작용

Full Sharpe에서 가장 큰 평균 상호작용은 다음과 같다.

| 조합 | 평균 상호작용 |
|---|---:|
| 거시엔진 × 기본배분 | +0.135 |
| 기본배분 × 오버레이 정책 | +0.073 |
| VKOSPI 신호 × 오버레이 정책 | -0.071 |
| 기본배분 × VKOSPI 신호 | +0.066 |
| 기본배분 × 변동성 타깃 | -0.039 |

즉 “한 모듈만 켠 성과”와 “완성 전략에서 그 모듈을 뺀 성과”가 크게 다른
이유는 모듈 사이 상호작용 때문이다.

## 모든 조합에서 관찰된 후보

- Full Sharpe 최고는 1110011: CAGR 12.70%, Sharpe 1.183,
  MDD -11.94%다.
- Full CAGR 최고는 1101111: CAGR 15.75%, Sharpe 1.144,
  MDD -13.42%다. 이 조합은 tail logistic을 사용하지 않는다.
- Full MDD가 가장 얕은 조합은 0100111: CAGR 11.81%,
  Sharpe 1.133, MDD -11.45%다.
- 현재 1111111을 전체기간 CAGR·Sharpe·MDD에서 동시에 지배한 다른 조합은
  없었다.
- 2018년 이후에는 1101111이 세 지표에서 현재 전략보다 좋았지만, 이는
  128개 결과를 본 뒤 발견한 사후 결과이므로 새 전략의 미사용 표본 성과로
  취급하면 안 된다.

## 결과 파일

- factorial_bridge.py: 전 조합 실행·검증·기여도 계산 코드
- outputs/all_128_combinations.csv: 128개 조합의 전체·2018년 이후 성과
- outputs/all_128_monthly_returns.csv: 조합별 월 수익률
- outputs/ordered_transition.csv: 구현 순서별 단계 변화
- outputs/local_component_effects.csv: Zero에서 하나만 켠 효과와 Robust에서
  하나를 뺀 효과
- outputs/shapley_attribution.csv: 순서 중립 기여도
- outputs/factorial_main_effects.csv: 64개 on 대 64개 off 평균효과
- outputs/pairwise_interactions.csv: 21개 모듈 쌍의 상호작용
- outputs/factorial_report.json: 구성과 끝점 재현 감사

## 실행

프로젝트 루트에서 다음을 실행한다.

    & 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' -m strategies.stage08_vkospi_factorial.factorial_bridge

이 실험은 민감도·기여도 분석이다. 128개 중 최고값을 골라 같은 자료의 성과로
홍보하면 새로운 하이퍼파라미터 탐색과 선택편향이 생긴다.
