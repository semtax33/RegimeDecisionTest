# Stage 49 — Prequential Evidence Model

Stage 49는 Stage 48을 다시 감사해 남아 있던 동일 표본 신뢰도, 미검증
technical shrink target, 임의 variance clipping, 위험항 중복을 제거한 연구
버전이다. Stage 36과 Stage 48의 원본 파일은 수정하지 않았다.

핵심 결론은 두 가지다.

1. 경제·수리통계적으로 해석 가능한 방식으로 모든 overlay를 다시 구현하는 것은
   가능했다.
2. 그렇다고 모든 overlay를 포트폴리오에 넣는 것이 좋은 것은 아니었다.
   Full 모델은 Stage 48보다 MDD는 개선했지만 Sharpe와 회전율 gate를 실패했다.
   오히려 overlay를 제거한 `Stage49_MacroBayes`가 Stage 48보다 안정적이었다.

따라서 `Stage49_FullPrequential`은 **승격 실패 연구 후보**이고,
`Stage49_MacroBayes`는 단순성 우선 ablation이다. 후자의 성과가 좋다는 이유만으로
사후적으로 최종 승격하지는 않았다.

## Stage 48에서 추가로 제거한 문제

### 같은 표본의 `beta/SE` reliability

Stage 48의 다음 식은 `|t|>1`이면 계수를 살린다.

```text
max(0, 1 - SE² / beta_hat²)
```

귀무가설에서도 정규근사상 계수 하나가 살아남을 확률이 약 31.7%이고, 이를
자산·레짐·월별로 반복하면 false discovery가 누적된다. 또한 이 값은 완전한
empirical-Bayes posterior probability가 아니다. Stage 49에서는 제거했다.

### 네 레짐별 stress/recovery WLS

soft regime probability를 WLS precision weight처럼 사용해 stress/recovery 계수를
레짐별로 반복 추정하면, 확률적 membership과 관측오차 precision의 의미가 섞이고
계수 수가 많아진다. Stage 49는 macro probability를 일반 predictor로 넣은 하나의
Bayesian ridge hierarchy로 교체했다.

### Technical을 cross-sectional mean으로 보내는 규칙

완전투자에서 cross-sectional mean이 active-return의 0점인 것은 맞지만,
technical 반대 신호의 크기만큼 그 평균으로 보내야 한다는 수익예측 근거는 없다.
Stage 49에는 neutral target 자체가 없다. Technical direction은 다음 달 수익률의
직접 predictor이며, technical을 추가한 nested model이 과거 one-step forecast를
개선한 비율만 반영된다.

### EPS·valuation의 일방향 slope

Stage 48은 Stage 35의 양수로 절단한 단변량 EPS/valuation slope를 그대로
사용했다. Stage 49에서는 KODEX200 Bayesian model의 마지막 nested block으로
포함해 부호를 강제하지 않는다.

### 과거 min/max variance clipping

과거 target 최솟값·최댓값은 predictive distribution의 신뢰구간이 아니다.
Stage 49는 이를 제거하고 Bayesian predictive mean/variance와 prequential QLIKE를
사용한다.

### 분산과 downside semivariance의 이중 차감

Stage 48 목적함수는 분산 전체와 downside semivariance를 단위계수로 동시에
차감했다. 비대칭 위험선호를 나타낼 수는 있지만 계수 1의 근거가 없고 downside가
이미 분산에 포함되어 있다. Stage 49는 다음 로그효용 2차 근사만 사용한다.

```text
certainty equivalent = w' mu - 0.5 w' Sigma w - expected transaction cost
```

13% 연환산 변동성, 16% 90%-CDaR 한도는 통계 추정치가 아니라 Stage 36과 같은
governance mandate로 명시해 유지했다.

## 질문 1·4: 왜 `1 + rank`, 1~2배 선형 scaling인가?

Stage 49에는 이 규칙이 없다. 기본 공분산은 의사결정월 직전 252개 complete daily
KRW return의 Ledoit–Wolf constant-correlation covariance다.

```text
Sigma_base(t) = 21 × LedoitWolf[daily returns through t-1 month-end]
```

ATR, credit rank, VKOSPI stress, GVZ, OVX는 다음 달 분산의 잔차만 예측한다.

```text
q(i,t) = log[realized_variance(i,t) / base_variance(i,t)]
```

KODEX200/BOND는 `stress, ATR, credit`, GLD/USO는 여기에 각각 GVZ/OVX와 availability
indicator를 추가한다. Bayesian ridge가 `q`의 predictive mean `m`과 variance
`s²`를 만들며 로그정규 재변환의 Jensen bias를 다음처럼 보정한다.

```text
full variance = base variance × exp(m + 0.5 s²)
```

과거에 그 당시 정보만으로 생성했던 base/full forecast를 산술적으로 섞었을 때
QLIKE가 최소가 되는 `w in [0,1]`를 다음 달에 사용한다.

```text
h_final = (1-w) h_base + w h_full
Sigma_final = D Sigma_base D
```

`D Sigma D`는 한 번만 적용된다. 따라서 고정 1~2배, 순차 곱, 과거 min/max
clipping이 없다.

## 질문 2: Credit을 alpha로 쓸 수 없는가?

쓸 수 있다. 다만 contemporaneous correlation이나 같은 표본 t값만으로 alpha라고
부를 수는 없다. Stage 49의 return hierarchy는 다음 순서다.

```text
M0: macro probabilities
M1: M0 + stress + recovery
M2: M1 + credit_widening_z
M3: M2 + technical direction
M4: M3 + EPS + valuation                 [KODEX200 only]
```

Credit alpha는 `forecast(M2)-forecast(M1)`이다. 과거에 실제 냈던 M1/M2 예측을
`base + w(full-base)`로 혼합할 때 MSE가 최소가 되는 convex stacking weight만
적용한다. Stress나 technical의 성과가 credit weight를 대신 만들어줄 수 없다.

Credit widening은 return에서 방향과 surprise 크기가 필요한 causal z-score로,
variance에서는 bounded risk state가 필요한 causal rank로 사용한다.

## 질문 3: 왜 KODEX200에만 credit을 적용했는가?

Stage 49에서는 그렇게 제한하지 않는다. KODEX200, BOND, GLD, USO 각각에 대해
macro와 stress/recovery를 통제한 뒤 credit의 추가 one-step 예측력을 별도로
측정한다. 이는 credit이 네 자산에 반드시 alpha라는 가정이 아니라, 적용 대상을
사전에 KODEX200으로 고정하지 않는다는 뜻이다.

최종 시점의 credit stacking weight는 KODEX200 1.000, BOND 0.625, GLD 0.741,
USO 0.803이었다. 다만 이것은 각 자산의 MSE-optimal forecast mixture이지
인과효과나 경제적 유의성을 뜻하지 않는다.

## 질문 5: Technical neutral은 왜 자산군 평균인가?

Stage 49에서는 평균, 0, risk-free rate, 과거 평균 중 어느 것도 technical neutral
target으로 사용하지 않는다. Technical signal 0은 technical predictor가 0이라는
뜻이고, 해당 block의 직접 return forecast increment만 계산한다. 과거 OOS
stacking weight가 0이면 increment도 0이다.

현재 최종 technical weight는 KODEX200 0, GLD 0, USO 0이었고 BOND만 약 0.996이었다.
이는 “반대면 평균으로 이동” 규칙이 아니라 BOND의 nested one-step MSE 결과다.

## 성과

거래비용 차감 후, 공통 기간 2007-04~2026-07:

| 전략 | CAGR | 변동성 | Sharpe | MDD | 평균 월 회전율 |
|---|---:|---:|---:|---:|---:|
| Stage36 | 10.50% | 9.47% | 1.105 | -12.41% | 3.70% |
| Stage48 | 12.31% | 12.90% | 0.967 | -16.46% | 10.14% |
| Stage49 MacroBayes | 11.70% | 11.46% | 1.025 | -15.73% | 5.49% |
| Stage49 ReturnEvidence | 10.42% | 11.64% | 0.911 | -15.20% | 9.44% |
| Stage49 FullPrequential | 10.03% | 11.24% | 0.908 | -15.54% | 10.29% |

2018-01~2026-07:

| 전략 | CAGR | 변동성 | Sharpe | MDD | 평균 월 회전율 |
|---|---:|---:|---:|---:|---:|
| Stage36 | 12.63% | 10.18% | 1.224 | -11.93% | 2.62% |
| Stage48 | 16.34% | 14.08% | 1.149 | -14.45% | 7.94% |
| Stage49 MacroBayes | 17.03% | 12.47% | 1.329 | -11.45% | 4.12% |
| Stage49 ReturnEvidence | 15.28% | 13.65% | 1.112 | -14.31% | 9.02% |
| Stage49 FullPrequential | 14.32% | 12.78% | 1.114 | -14.31% | 9.46% |

Stage48 대비 MacroBayes의 paired 12개월 block bootstrap 2,000회 결과:

- 전체: CAGR 개선 35.4%, Sharpe 개선 68.3%, MDD 개선 82.8%
- 2018년 이후: CAGR 개선 57.8%, Sharpe 개선 84.4%, MDD 개선 71.3%

반면 FullPrequential은 Stage48 대비 전체 Sharpe 개선 확률 30.5%, CAGR 개선 확률
6.0%에 그쳤다. Full 모델의 `promotion_pass=false`는 이 결과를 그대로 반영한다.

해석은 분명하다. 개별 자산 MSE나 variance QLIKE를 개선하는 신호라도, 작은
expected-return 차이를 증폭하는 최적화와 거래비용을 거치면 포트폴리오 Sharpe가
낮아질 수 있다. 이번 표본에서는 “더 많은 보정”보다 macro-only Bayesian shrinkage가
나았다.

## 파일과 검증

- `prequential_evidence_model.py`: 실제 추정·stacking·공분산·최적화·검증 코드
- `outputs/stage49_macro_bayes_monthly.csv`: overlay 없는 ablation
- `outputs/stage49_return_evidence_monthly.csv`: return overlay ablation
- `outputs/stage49_full_prequential_monthly.csv`: return+variance 후보
- `outputs/return_prequential_history.csv`: 당시 생성된 nested one-step forecast
- `outputs/variance_prequential_history.csv`: 당시 생성된 variance forecast와 QLIKE
- `outputs/performance_comparison.csv`: Stage36/48/49 비교
- `outputs/paired_block_bootstrap.csv`: 모든 Stage49 후보의 paired bootstrap
- `outputs/validation_report.json`: frozen hash, 인과성, PSD, 제약, gate

런타임 검증 28개와 pytest 9개가 모두 통과했다. 새 폴더에는 Stage48의 차트,
HTML, notebook, 후보 회귀표를 복사하지 않았다.

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1'
python -m strategies.stage49_prequential_evidence_model.prequential_evidence_model
python -m pytest tests\test_stage49_prequential_evidence_model.py -q
```

## 남은 한계

- Bayesian ridge의 Gaussian·선형 관계와 five-observations-per-parameter sufficiency
  rule은 명시적 모델 가정이지 경제 법칙이 아니다.
- Convex stacking weight도 유한한 과거 OOS forecast에 추정되므로 불확실하다.
- 2018년 이후 구간도 이전 연구 단계에서 반복 확인됐으므로 완전히 untouched한
  외부 검증 표본으로 볼 수 없다.
- 백테스트와 bootstrap은 미래 성과를 보장하지 않는다.
