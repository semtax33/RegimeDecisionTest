# Stage17 — Dynamic Risk Shape

## 결론

Stage16 Crash Evidence와 0.5 dead zone을 동결하고 세 가지 경제 가설만 검증했다.

- **17A Dynamic Sigma**: Evidence가 공분산 구조를 변경
- **17B Dynamic ES**: 고정 λ, Evidence가 Expected Shortfall 꼬리 벌점을 변경
- **17C Sigma + ES**: 두 구조를 결합

결과는 명확한 trade-off다.

- 17A는 CAGR 11.04%를 유지했지만 MDD와 Sharpe가 Stage14보다 나빠졌다.
- 17B는 MDD를 -23.19%에서 -17.01%로 줄이고 Sharpe를 0.856으로 높였지만 CAGR이 9.32%로 하락했다.
- 17C는 MDD -17.46%를 달성했지만 CAGR 9.32%로 17B보다 나은 결합효과가 없었다.
- 세 후보 중 `CAGR >= 10.5%`, `MDD >= -18%`, `Sharpe >= Stage14 고정 λ`를 동시에 충족한 전략은 없다.

따라서 Stage17B는 **MDD 우선 연구 후보**지만 기존 전략을 대체할 최종안은 아니다.

## 성과

### 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Stage14 고정 λ | **11.00%** | 13.34% | 0.851 | **1.571** | -23.19% | 0.474 |
| Stage16 확인 λ | 10.93% | 13.26% | 0.851 | 1.563 | -23.19% | 0.471 |
| 17A Dynamic Sigma | **11.04%** | 13.58% | 0.841 | 1.545 | -23.39% | 0.472 |
| 17B Dynamic ES | 9.32% | **11.16%** | **0.856** | 1.535 | **-17.01%** | **0.548** |
| 17C Sigma + ES | 9.32% | 11.26% | 0.849 | 1.524 | -17.46% | 0.533 |

Stage14 고정 λ 대비 17B:

- CAGR `-1.69%p`
- 변동성 `-2.18%p`
- Sharpe `+0.0046`
- MDD `+6.19%p` 개선
- Calmar `0.474 → 0.548`

### 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage14 고정 λ | **16.13%** | 14.27% | 1.123 | -15.90% |
| 17A Dynamic Sigma | **16.20%** | 14.67% | 1.101 | -16.86% |
| 17B Dynamic ES | 12.64% | **11.20%** | **1.123** | **-15.24%** |
| 17C Sigma + ES | 12.67% | 11.23% | 1.123 | -15.28% |

최근 구간에서도 ES는 위험을 낮췄지만 CAGR 약 3.5%p를 희생했다.

## 중요한 선행 확인

Stage13/14의 조건부 모멘트 추정에는 이미 raw stress로 정상 공분산과 스트레스 공분산을 섞는 구조가 있었다.

```text
Sigma_stage14 = (1 - raw_stress) × Sigma_normal
                + raw_stress × Sigma_stress
```

따라서 17A는 동적 공분산을 새로 처음 도입한 것이 아니라, 혼합 비중을 설명력이 낮은 raw stress에서 Stage16의 확인된 포트폴리오 압력으로 교체한 실험이다.

## 동결한 Stage16 Evidence

Stage16의 다음 항목은 다시 학습하거나 문턱을 바꾸지 않았다.

- `min(shock, persistence)`
- 5거래일 상승 방향과 VIX6 왼쪽 꼬리의 동일가중 평균
- `P(Slowdown) + P(Stagflation)`
- 세 블록 동일가중
- 중앙값 0.5 dead zone
- High Level × Falling 정상화 규칙

```text
q_market = max(0, 2 × (Evidence - 0.5))
```

## Portfolio-specific pressure

VKOSPI/VIX6는 주식시장 중심 신호이므로 현재 포트폴리오가 이미 안전하면 반응을 줄인다.

각 시점에서 과거 수익률 시나리오를 당시 stress score로 가중하고 현재 전월 비중의 90% Expected Shortfall을 계산한다.

```text
Vulnerability = current portfolio stress-weighted ES
                / worst single-asset stress-weighted ES

q_portfolio = q_market × Vulnerability
```

비율은 0~1로 제한된다. 최대 손실 자산을 기준으로 정규화하기 때문에 학습 계수가 없다. 세 후보의 평균 portfolio pressure는 약 0.034로 시장 pressure보다 상당히 작았다. 즉 많은 달에 포트폴리오가 이미 분산돼 있어 추가 방어 필요성이 낮다고 판단했다.

## 17A — Dynamic Sigma

λ와 기존 하방 준분산 벌점은 1로 고정한다.

```text
Sigma* = (1 - q_portfolio) × Sigma_normal
         + q_portfolio × Sigma_stress
```

목적은 “위험을 얼마나 싫어할지”가 아니라 스트레스 때 자산의 변동성과 상관이 어떻게 바뀌는지를 알려주는 것이다.

결과적으로 CAGR은 유지됐지만 MDD가 줄지 않았다. 확인된 공분산만으로 기존 장기 낙폭의 원인이 해결되지 않았다는 뜻이다.

## 17B — Dynamic Expected Shortfall

분산계수와 λ는 고정하고 90% 월간 Expected Shortfall의 제곱만 동적으로 강화한다.

```text
ES = worst 10% historical monthly portfolio loss average
TailPenalty = (1 + q_portfolio) × ES²
```

ES를 제곱해 분산과 같은 수익률 제곱 단위로 만들었다. 별도 scale 계수는 없다. 이 구조가 MDD를 가장 많이 줄였지만 ES 벌점이 정상시에도 존재하기 때문에 상승 참여가 감소했다.

## 17C — Dynamic Sigma + ES

17A와 17B를 동시에 적용했다. 예상과 달리 17B보다 MDD·Sharpe가 좋아지지 않았다. 공분산 혼합과 ES가 같은 위험을 중복 반영하거나, 주식 중심 Evidence가 실제 최대 낙폭 자산을 설명하지 못했을 가능성이 크다.

## 실제 Drawdown Episode

Stage14 고정 λ의 최대 낙폭은 단일 폭락월이 아니라 2012-10~2014-10의 25개월 장기 에피소드였다.

| 항목 | 값 |
|---|---:|
| Episode MDD | -23.19% |
| KODEX200 기여 | -1.54% |
| 채권 기여 | +0.15% |
| GLD 기여 | **-23.69%** |
| USO 기여 | 0.00% |
| 주식-채권 상관 | 0.058 |

최대 MDD의 핵심 원인은 주식-채권 상관 붕괴가 아니라 높은 금 비중에서 발생한 장기 GLD 하락이었다. 이 기간 VKOSPI/VIX6 기반 Crash Evidence가 낮았기 때문에 17A/17C의 공분산 조정도 낙폭을 막지 못했다.

이 결과는 현재 주식 옵션 기반 신호만으로 4자산 전체 MDD를 안정적으로 -15% 이내로 제한하기 어렵다는 것을 보여준다. 다음 단계에는 금·채권·원유 자체의 포트폴리오별 tail 상태가 필요하지만, 이번 Stage에서는 새 변수를 추가하지 않았다.

## 세 단계 Precision

사후 진단을 다음과 같이 분리했다.

1. `Signal precision`: q가 켜진 달이 하위 10% 주식 손실월이었는가
2. `Decision precision`: q가 켜진 달 후보 수익률이 고정 λ보다 높았는가
3. `Portfolio precision`: 실제 위험비중을 줄인 달 후보 수익률이 고정 λ보다 높았는가

17B 전체 구간:

- signal activation 119개월
- signal crash precision 5.9%, 24개 폭락 중 7개 포착
- decision precision 52.1%
- portfolio action 206개월
- portfolio precision 47.6%

이를 통해 신호 오류와 거래비용·전월 비중에서 발생하는 포트폴리오 경로효과를 분리했다.

## 내부 거래비용 제거 진단

17C에서 optimizer 내부 거래비용만 제거하고 실제 성과에서는 모든 거래·환전비용을 그대로 차감했다.

| 경로 | CAGR | Sharpe | MDD | 평균 turnover | 총비용 |
|---|---:|---:|---:|---:|---:|
| 17C | 9.32% | 0.849 | -17.46% | 2.28% | 1.55% |
| 내부 비용 제거 진단 | 9.45% | **0.882** | **-16.73%** | 6.28% | 4.62% |

내부 거래비용 제약이 느린 재진입과 일부 MDD 악화에 영향을 준다는 증거는 있다. 하지만 turnover와 총비용이 크게 증가했고 CAGR은 여전히 10.5%에 못 미치므로 채택 후보가 아니라 원인 규명용 경로다.

## 과최적화 방지

- Stage16 Evidence 및 0.5 문턱 동결
- 후보는 경제 가설 17A·17B·17C 세 개로 사전 제한
- ES confidence는 기존 CDaR와 같은 90%
- ES²로 단위만 분산과 맞추고 별도 scale 없음
- λ는 모든 후보에서 정확히 1
- 미래 폭락·수익률 label은 진단에만 사용
- 성과를 보고 threshold, gamma, cost ratio를 선택하지 않음
- 후보 중 목표 gate를 통과하지 못한 결과도 그대로 보존

## 성공 Gate

| 전략 | CAGR ≥ 10.5% | MDD ≥ -18% | Sharpe ≥ Stage14 |
|---|---:|---:|---:|
| 17A | 통과 | 실패 | 실패 |
| 17B | 실패 | 통과 | 통과 |
| 17C | 실패 | 통과 | 실패 |

세 조건을 모두 만족한 후보가 없으므로 Stage14를 자동 교체하지 않는다.

## 검증

- 후보별 232개월 SLSQP 완료
- 경제 목적함수 fallback 0회
- 장기전용, 비중 합계 100%, 현금·레버리지 없음
- 단일자산 과반 제한 없음
- λ가 전 기간 1인지 검증
- portfolio pressure가 market pressure를 초과하지 않음
- 13% ex-ante 변동성 및 -16% CDaR guard 충족
- 거시·옵션 신호가 매매월보다 앞서는지 검증
- 미래 label 및 하이퍼파라미터 탐색 없음

## 실행

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage17_dynamic_risk_shape.dynamic_risk_shape_slsqp
```

## 파일

- `dynamic_risk_shape_slsqp.py`: Stage17 A/B/C 및 진단 코드
- `dynamic_risk_shape_report.html`: 상세 결과 보고서
- `outputs/performance_comparison.csv`: 전체·고정구간 성과
- `outputs/layered_precision.csv`: Signal/Decision/Portfolio precision
- `outputs/drawdown_episode_attribution.csv`: peak-to-trough 에피소드 분해
- `outputs/frozen_stage16_signals.csv`: 동결한 Evidence
- `outputs/stage17a_dynamicsigma_monthly.csv`: 17A 월별 경로
- `outputs/stage17b_dynamices_monthly.csv`: 17B 월별 경로
- `outputs/stage17c_dynamicsigmaes_monthly.csv`: 17C 월별 경로
- `outputs/stage17c_no_internal_cost_diagnostic_monthly.csv`: 거래비용 진단
- `outputs/validation_report.json`: 공식·검증·성과 gate

