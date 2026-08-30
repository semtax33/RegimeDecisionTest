# Stage16 — Confirmed Crash-Risk Lambda

## 결론

Stage14의 `lambda = 1 + stress`를 경제적 확인 게이트로 바꿨다. 직접적인 λ 작동은 232개월 전부에서 사실상 계속되던 구조에서 119개월로 줄었고, 실제 SLSQP 위험축소 월은 192개월에서 151개월, false positive는 110개월에서 87개월로 감소했다.

성과는 **부분 개선**이다.

- 전체 구간: CAGR과 Sharpe는 개선됐지만 MDD는 0.22%p 나빠졌다.
- 2018년 이후: CAGR은 개선됐고 MDD는 같지만 Sharpe는 소폭 낮아졌다.
- 무가중 고정 λ와 비교하면 전체 구간에서 거의 같은 성과다.

즉 과잉 방어 비용은 상당 부분 회수했지만, 원문이 기대한 FP 비율 25~35%까지 낮추거나 MDD를 동시에 개선하지는 못했다. 현재 VKOSPI/VIX6 블록만으로 미래 폭락을 정밀하게 분류할 수 있다는 근거는 확인되지 않았다.

## 성과

### 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | Sortino | MDD | 최종 배수 |
|---|---:|---:|---:|---:|---:|---:|
| Stage14 고정 λ | 11.00% | 13.34% | 0.851 | 1.571 | -23.19% | 7.53배 |
| Stage14 `1+stress` | 10.76% | 13.13% | 0.846 | 1.547 | -22.97% | 7.21배 |
| Stage16 확인 λ | **10.93%** | 13.26% | **0.851** | **1.563** | -23.19% | **7.43배** |

Stage14 동적 λ 대비:

- CAGR `+0.17%p`
- Sharpe `+0.0046`
- Sortino `+0.0154`
- 최종 배수 `+0.22배`
- MDD `-0.22%p` 악화

### 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD | 최종 배수 |
|---|---:|---:|---:|---:|---:|
| Stage14 고정 λ | 16.13% | 14.27% | 1.123 | -15.90% | 3.61배 |
| Stage14 `1+stress` | 15.78% | 13.87% | **1.130** | -15.90% | 3.52배 |
| Stage16 확인 λ | **15.96%** | 14.11% | 1.124 | -15.90% | **3.56배** |

Stage14 동적 λ 대비 CAGR은 `+0.18%p`, MDD는 동일하며 Sharpe는 `-0.0056`이다.

## 왜 기존 λ가 과잉 방어였나

Stage14는 스트레스 백분위가 0보다 크기만 하면 λ가 1보다 커졌다. 실제 백테스트의 λ 범위는 약 1.28~1.89였다. 스트레스가 평균적인 달에도 하방 준분산 벌점이 계속 커졌고, 위험축소가 232개월 중 192개월 발생했다.

사후 진단 결과도 “높은 VKOSPI 수준” 하나로 설명되지 않았다.

- 기존 FP 110개월 중 42개월은 직전 주식 수익률이 음수였다.
- 기존 FP의 평균 향후 주식 수익률은 +4.74%였다.
- FP의 평균 shock은 0.522였지만 persistence는 0.470이었다.
- 즉 일시적 충격 또는 이미 하락한 뒤의 정상화 국면이 섞여 있었다.

## Stage16 확인 공식

새로운 변수를 추가하거나 미래 수익률로 분류기를 학습하지 않았다. Stage14가 이미 가진 블록을 다음 세 가지 경제적 확인축으로 재구성했다.

### 1. 금융시장 확인

```text
financial_confirmation = min(shock rank, persistence rank)
```

빠른 충격과 한 달 지속성이 함께 높아야 높은 값이 된다. `High shock + Low persistence`는 일시적 소음으로 처리한다.

### 2. 방향·꼬리위험 확인

```text
timing_confirmation = mean(
    5-day rising-stress rank,
    VIX6 left-tail repricing rank
)
```

현재 수준이 아니라 상승 방향과 왼쪽 꼬리의 재가격을 본다. 5거래일은 조정한 lookback이 아니라 한 거래주라는 달력 단위다.

### 3. 거시 취약성

```text
macro_vulnerability = P(Slowdown) + P(Stagflation)
```

성장둔화 또는 스태그플레이션 확률이 높을수록 같은 금융충격을 더 위험하게 본다. Hard regime allocation은 사용하지 않는다.

세 블록은 설명 불가능한 계수 없이 동일 가중한다.

```text
crash_evidence = mean(
    financial_confirmation,
    timing_confirmation,
    macro_vulnerability
)
```

## Dead zone과 정상화

```text
crash_evidence <= 0.5  -> crash_pressure = 0, lambda = 1
crash_evidence >  0.5  -> crash_pressure = 2 × (evidence - 0.5)
lambda = 1 + crash_pressure
```

0.5는 탐색한 문턱이 아니다. 인과적 백분위의 중앙값이자 확률의 과반 경계다. evidence 0.5를 기준으로 lambda 1~2에 선형 대응시켜 추가 곡률 파라미터도 없앴다.

다음 상태에서는 λ를 1로 되돌린다.

```text
VKOSPI level > median
AND stress direction <= median
-> normalization, lambda = 1
```

이는 원문의 `High Level × Falling = Risk Normalization`을 그대로 구현한 것이다.

## 신호 및 행동 진단

| 항목 | Stage14 | Stage16 |
|---|---:|---:|
| 실제 위험축소 월 | 192 | 151 |
| 실제 FP 월 | 110 | 87 |
| 실제 FP 비율 | 57.3% | 57.6% |
| 포착한 하위 10% 폭락월 | 18/24 | 9/24 |
| 직접 λ 활성화 월 | 사실상 전 기간 | 119/232 |
| 직접 신호 FP 월 | 해당 없음 | 66 |
| 직접 신호 FP 비율 | 해당 없음 | 55.5% |

FP의 절대 개수는 줄었지만 비율은 좋아지지 않았다. 또한 폭락 포착이 감소했다. 방어 횟수를 줄이면 우연히 포함되던 폭락월도 함께 빠진다는 의미다.

직접 λ 활성화는 119개월이지만 실제 위험축소가 151개월인 이유는 SLSQP가 경로의존적이기 때문이다. 전월 비중과 거래비용이 목적함수에 들어가므로 λ가 1로 돌아와도 기존 방어 포지션을 즉시 전부 되돌리지 않는다. 이를 강제로 되돌리면 post-optimizer overlay가 되므로 적용하지 않았다.

## 의도적으로 넣지 않은 것

- 0.65, 0.70, 0.75 문턱 비교
- FP 비용 2:1, 3:1, 4:1 탐색
- gamma 곡률 탐색
- 미래 1개월·3개월 수익률을 이용한 meta-label 학습
- 학습 표본에서 가장 성과가 좋은 규칙 선택
- Logistic, tree, boosting 등 추가 분류기

이 방법들은 성과를 더 높일 가능성은 있지만 작은 월별 표본에서 선택 편향과 설명하기 어려운 하이퍼파라미터를 만든다. 미래 수익률과 폭락 label은 사후 진단에만 사용했다.

## 변하지 않은 Stage14 구조

- 거시 확률과 조건부 기대수익·공분산 추정
- VKOSPI/VIX6가 기대수익과 공분산에 미치는 연속 효과
- SLSQP 100% 완전투자
- 현금 및 레버리지 없음
- 단일자산 과반 제한 없음
- 연환산 ex-ante 변동성 13% guard
- 과거 CDaR 90% -16% guard
- 거래비용과 환전비용

## 검증

- 232개월 SLSQP 성공, fallback 0회
- 비중 합계 최대오차 `2.22e-16`
- 장기전용, 자산별 0~100%
- 거시·스트레스 신호가 매매월보다 앞서는지 검증
- dead zone에서 λ가 정확히 1인지 검증
- λ 최대 2 이하
- 변동성·CDaR 제약 충족
- 미래 label 미사용 및 하이퍼파라미터 탐색 없음

## 실행

프로젝트 루트에서:

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage16_confirmed_crash_risk.confirmed_crash_risk_slsqp
```

## 파일

- `confirmed_crash_risk_slsqp.py`: 전략 및 검증 코드
- `confirmed_crash_risk_report.html`: 브라우저용 상세 보고서
- `outputs/confirmed_crash_lambda_monthly.csv`: 월별 신호·비중·수익률
- `outputs/confirmed_crash_signals.csv`: 확인 블록과 λ 입력
- `outputs/performance_comparison.csv`: Stage14 고정/동적 및 Stage16 비교
- `outputs/stage16_minus_stage14.csv`: Stage14 동적 λ 대비 증감
- `outputs/attribution_comparison.csv`: FP·폭락 포착 비교
- `outputs/validation_report.json`: 공식·성과·제약 검증 결과

