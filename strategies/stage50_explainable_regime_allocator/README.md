# Stage50 Explainable Regime Allocator

이 폴더는 기존 코드에서 실제로 선택된 **거시 국면 자산배분 한 경로만** 독립시킨
리팩터링 결과다. 운영 파일에는 실행 모드, overlay 분기, 다른 Stage import가 없다.
따라서 처음 읽는 사람은 아래 한 방향만 이해하면 된다.

```text
원자료
  -> 월별 KRW 자산수익률 + 거시 국면 score + 일별 KRW 수익률
  -> BayesianRidge 월 기대수익률
  -> Ledoit-Wolf 월 공분산
  -> 거래비용·변동성·CDaR 제약을 둔 long-only 최적화
  -> 다음 달 실현수익률
```

## 파일 경계

- `explainable_regime_allocator.py`: 실제 전략의 유일한 실행 파일이다. 데이터 접근,
  기대수익률, 공분산, 비용, 최적화, 백테스트를 작은 책임으로 분리했지만 모든 의존
  코드는 이 파일 안에 있다.
- `layer_ablation_validation.py`: 과거 Stage36의 여섯 overlay가 실제로 어떤 역할을
  했는지 확인하는 검증 전용 파일이다. 운영 파일에서는 import하지 않는다. 레이어를
  제거하는 반사실은 원래 레이어가 구현된 동결 Stage36을 기준으로 해야 하므로, 이
  파일만 하나의 legacy reference adapter를 사용한다.
- `outputs/`: 월별 경로, 성과, 인과성/제약 감사, ablation, bootstrap, encoding 및
  beta 안정성 결과다.

운영 전략에서 사용하지 않는 stress expected-return, stress covariance blend,
technical confidence, ATR, credit, GVZ, OVX 코드는 모두 제거했다. 이 기능을 켜고
끄는 가짜 옵션도 두지 않았다.

## 모델의 경제·통계적 의미

### 1. 거시 국면 score

성장(GDP·수출·BSI)과 물가(CPI·PPI·수입물가)의 전년동월 변화율을 causal expanding
z-score로 표준화한다. 성장과 물가의 부호 조합으로 Goldilocks, Overheating,
Slowdown, Stagflation score를 만든다. 이 값은 네 상태의 상대 강도이지, calibration을
검증한 확률은 아니다.

### 2. 기대수익률

각 자산의 다음 달 수익률을 세 국면 score로 Bayesian ridge 회귀한다. 당월 의사결정은
직전 월까지의 데이터로만 학습한다. Goldilocks를 기준범주로 뺀 것은 다중공선성을
피하면서 기존 결과를 재현하기 위한 선택이다. Ridge에서는 기준범주 변경이 prior의
모양을 바꾸므로 완전히 중립적인 코딩은 아니라는 한계가 있다.

### 3. 공분산

의사결정 직전 252 거래일의 KRW 환산 일별 수익률에 Ledoit-Wolf shrinkage를 적용하고
21을 곱해 월 공분산으로 바꾼다. 표본 공분산보다 안정적이지만 252일과 21일은 추정된
최적값이 아니라 거래일 관행에 따른 휴리스틱이다.

### 4. 배분

목적함수는 월 기대 로그성장률의 2차 근사에서 예상 거래비용을 뺀 값이다. 비중 합은
1, 공매도와 레버리지는 금지하며, 연 13% ex-ante 변동성과 과거 90% CDaR 16% 한도를
둔다. 이 한도들은 통계적 결론이 아니라 투자정책 값이다.

SOLID는 인터페이스를 늘리기 위한 장식이 아니라 책임 경계에만 사용했다. 데이터
repository, 평균 forecaster, 공분산 forecaster, allocator를 각각 교체할 수 있고,
`BacktestEngine`은 그 계약만 의존한다. 실행 경로는 하나이므로 KISS도 유지한다.

## 독립 전략 성과

거래비용을 반영한 저장 결과는 다음과 같다.

| 기간 | CAGR | 변동성 | Sharpe | MDD | 월평균 turnover |
|---|---:|---:|---:|---:|---:|
| 2007-04~2026-07 | 11.70% | 11.46% | 1.025 | -15.73% | 5.49% |
| 2018-01~2026-07 | 17.03% | 12.47% | 1.329 | -11.45% | 4.12% |

저장된 기존 기준 경로와 비교하면 기대수익률은 최대 절대오차 `2e-16` 이내로 같고,
월 수익률은 `5.2e-7`, 비중은 `4.1e-6` 이내다. 후자의 미세 차이는 같은 연속 최적화
문제를 다시 푼 SLSQP 수치 경로 차이다.

주의할 점은 거시 원자료가 point-in-time vintage인지 확인되지 않았다는 사실이다.
따라서 이 결과는 코드 재현과 구조 검증 결과이지 즉시 운용 가능한 성과 보장은 아니다.

## Stage36 레이어 ablation

각 실험은 기준 전략에서 **한 레이어만** 제거하고 데이터, 비용, 목적함수, 위험한도를
고정했다. 수치는 `후보 - 전체 레이어 기준`이다.

| 한 레이어 제거 | 전체 CAGR Δ | 전체 변동성 Δ | 전체 Sharpe Δ | 전체 MDD Δ |
|---|---:|---:|---:|---:|
| stress expected-return | +0.26%p | +0.10%p | +0.015 | -0.39%p |
| stress covariance blend | +0.18%p | +0.22%p | -0.006 | -0.02%p |
| technical confidence | +0.49%p | +0.24%p | +0.021 | -2.02%p |
| ATR covariance scaling | +1.90%p | +2.60%p | -0.073 | -2.26%p |
| credit stress/risk scaling | +0.78%p | +1.18%p | -0.045 | -0.97%p |
| GVZ/OVX variance scaling | +0.11%p | +0.57%p | -0.048 | -1.34%p |

`MDD Δ`가 음수면 제거 후 낙폭이 더 나빠졌다는 뜻이다. 12개월 circular block
bootstrap 2,000회에서도 ATR·credit·GVZ/OVX 제거는 CAGR을 높이는 대신 Sharpe 또는
MDD를 악화시키는 방향이 강했다. 이 세 레이어는 alpha라기보다 명확한 **위험
throttle**로 작동했다. 반면 stress expected-return adjustment는 제거 후 CAGR 개선
확률 96.1%, Sharpe 개선 확률 88.3%로 성과상 정당화가 약했다. 이는 인과효과 검정이
아니며, 여러 ablation을 보았으므로 데이터 마이닝 가능성도 남는다.

## 질문에 대한 검증 결론

### 왜 `1 + rank`, 즉 1~2배 선형 variance scaling인가?

이 방식의 장점은 세 가지뿐이다. 과거 관측치 안의 서열만 써 이상치와 단위에
둔감하고, multiplier가 항상 1~2에 갇혀 폭주하지 않으며, 위험 신호가 커질수록
위험을 단조롭게 올린다. 그러나 2배 상한과 선형 모양은 확률모형에서 도출된 값이
아니라 **보수적인 bounded heuristic**이다. ablation 결과 ATR·credit·GVZ/OVX가
위험을 낮춘 것은 확인됐지만, 1~2가 최적이라는 증거는 아니다. 실제 운용 전에는
`1 + k*rank`, 비선형 mapping, 예상 분산 직접회귀를 별도 OOS로 비교해야 한다.

### 왜 credit을 alpha가 아니라 KODEX200 위험 확인 신호로 제한했는가?

Credit spread 확대는 위험회피와 금융여건 악화를 잘 나타내지만, 그것이 다음 달
수익률에 미치는 부호와 시차는 자산별·국면별로 달라질 수 있다. 그래서 Stage36은
방향성 수익 예측보다 equity 위험 확인에만 썼다. 이는 설계 의도이지 통계적 필연은
아니다. 실제 ablation에서 credit scaling 제거는 CAGR `+0.78%p`와 함께 변동성
`+1.18%p`, Sharpe `-0.045`, MDD `-0.97%p`를 만들었다. 즉 표본에서는 alpha보다는
risk throttle 설명이 더 잘 맞는다.

수익률 변화량에 직접 반영하는 대안은 가능하다. 다만 같은 credit signal을 평균과
분산에 동시에 넣으면 위험회피를 이중 반영할 수 있으므로, 다음 달 수익 예측의
walk-forward OOS 개선을 먼저 입증하고 평균 또는 분산 중 역할을 명시해야 한다.

### 왜 credit을 BOND·GLD·USO에 직접 적용하지 않았는가?

KODEX200에는 spread 확대→equity risk-off라는 경로가 비교적 직접적이다. BOND는
국채 duration과 회사채 신용위험이 반대로 움직일 수 있고, GLD와 USO는 달러·실질금리·
인플레이션·수급 경로가 섞여 국내 credit spread의 직접 mapping이 약하다. 따라서
다른 자산에 동일 배율을 복사하는 것은 경제적으로 안전하지 않다.

KOSPI 옵션 stress를 전 자산에 적용한 기존 방식과 KODEX200에만 적용한 반사실을
직접 비교한 결과는 다음과 같다.

| 기간 | KODEX200-only CAGR Δ | 변동성 Δ | Sharpe Δ | MDD Δ |
|---|---:|---:|---:|---:|
| 2007-04~2026-07 | +0.31%p | +0.18%p | +0.011 | -0.38%p |
| 2018-01~2026-07 | +0.41%p | +0.01%p | +0.035 | +0.17%p |

전체 표본에서 global stress는 MDD를 약간 더 방어했지만, KODEX200-only가 CAGR과
Sharpe는 높았다. 특히 locked 기간에서 KODEX200-only의 CAGR·Sharpe 개선 bootstrap
확률은 각각 100%, 99.95%였다. 이 표본에서는 VKOSPI 기반 shock을 금·채권·유가의
공분산까지 전파해야 할 강한 근거가 없다. 다만 KODEX200-only 공분산은 단순 대각원소
교체가 아니라 `D Σ D`로 구성해 PSD를 보존했다.

### VKOSPI stress 공분산 blend가 금·채권·유가에도 영향을 주는 이유는?

`Σ = (1-s)Σ_macro + sΣ_high-stress`이면 high-stress 표본에서 추정한 모든 분산과
공분산이 섞인다. 따라서 신호는 KOSPI 옵션에서 왔어도 과거 stress 때 함께 움직인
BOND·GLD·USO의 분산 및 상관까지 바뀐다. 수학적으로는 convex covariance blend라
PSD를 보존하지만, 경제적으로는 “VKOSPI가 전 자산의 systemic state proxy”라는 강한
가정이다. 위 scope 실험은 그 가정의 성과상 필요성이 제한적임을 보여준다.

### Rank-only, z-score-only, 현재 혼합 방식은 어느 쪽이 나은가?

Credit raw signal만 대상으로 causal expanding encoding을 만들고, 다음 달 수익률과
다음 달 log-squared-return을 각각 224개 OOS 관측으로 예측했다. 서로 단위가 다른 두
MSE를 억지로 더하지 않고, 각 task의 최저 MSE로 정규화한 뒤 정책을 비교했다.

| 정책 | 8개 task 중 최저 MSE 횟수 | 평균 상대 MSE | 최악 상대 MSE |
|---|---:|---:|---:|
| rank-only | 5 | 1.006 | 1.032 |
| z-score-only | 3 | 1.013 | 1.050 |
| z-return / rank-risk 혼합 | 5 | 1.011 | 1.050 |

자산별로 현재 혼합을 정확히 지지한 것은 USO 1개뿐이었다. 이 진단에서는 rank-only가
평균적으로 가장 나았지만 차이는 작고, 전체 portfolio P&L 비교가 아니라 signal-role
진단이다. 따라서 “rank는 risk, z-score는 alpha”라는 구분은 직관적 설명은 되지만
보편적인 통계 근거는 아니다.

### Regime별 stress/recovery beta는 안정적인가?

32개 causal expanding beta 시계열 중 14개가 0을 제외한 sign switch를 보였고,
28개는 `|beta|`의 p90/median 비율이 3을 넘었다. sign switch 14개는 모두 부호 제약을
두지 않은 BOND·GLD 16개 계열에서 나왔다. 반대로 KODEX200·USO의 부호가 안정적으로
보이는 것은 코드의 sign prior 영향이므로 실증적 안정성 증거가 아니다.

2007~2017 median과 2018~2026 median 사이의 부호 변경은 없었지만, 예를 들어 Slowdown
BOND stress의 두 기간 median 크기는 약 575배 차이였다. 결론은 **넓은 하위기간의
방향은 유지되지만 월별 sign과 magnitude는 구조계수라고 부를 만큼 안정적이지 않다**는
것이다. `R² × beta`는 이 불안정을 완화하려는 휴리스틱이지만, in-sample R²는 Bayesian
posterior shrinkage도 아니고 계수 불확실성의 올바른 척도도 아니다. 운영 전략에서는
이 레이어와 R² 곱을 모두 제거했다.

### Technical neutral을 왜 cross-sectional mean으로 뒀는가?

그 값은 현금 중립점이 아니라 **같은 시점 자산군 안의 상대적 중립점**이다. Technical
방향이 macro view를 지지하지 않으면 해당 자산의 극단적 forecast를 현재 자산군 평균
쪽으로 보내 cross-sectional conviction만 줄이려는 의도였다. 0, 무위험수익률, 과거
평균은 각각 절대수익 관점을 새로 섞으므로 원래의 상대배분 filter와 다른 모델이 된다.

하지만 macro forecast 전체가 잘못됐을 때 cross-sectional mean도 잘못된 중심이며,
“technical 반대=평균 자산 수익”이라는 통계적 근거는 없다. 실제로 technical filter를
제거하면 전체 CAGR `+0.49%p`, Sharpe `+0.021`이나 MDD는 `-2.02%p` 나빠졌고, locked
기간에는 CAGR·Sharpe·MDD가 모두 개선됐다. 즉 drawdown 완충 의도는 일부 보이나
안정적인 필수 레이어라는 증거는 약하다. 운영 전략에서는 이 filter도 제거했다.

## 실행과 재현

프로젝트 루트에서 AGENTS.md에 지정된 환경을 활성화한 뒤 실행한다.

```powershell
python -m strategies.stage50_explainable_regime_allocator.explainable_regime_allocator
python -m strategies.stage50_explainable_regime_allocator.layer_ablation_validation
pytest tests/test_stage50_explainable_regime_allocator.py -q
```

주요 결과 파일은 `monthly_results.csv`, `performance.csv`,
`validation_report.json`, `layer_ablation_performance.csv`,
`layer_ablation_bootstrap.csv`, `stress_scope_performance.csv`,
`credit_encoding_oos_losses.csv`, `credit_encoding_policy_comparison.csv`,
`regime_beta_history.csv`, `regime_beta_stability.csv`,
`ablation_validation_report.json`이다.
