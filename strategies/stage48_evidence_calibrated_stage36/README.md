# Stage 48 — Evidence-calibrated Stage 36

Stage 48은 Stage 36의 데이터 시점, 네 자산, 장기 레짐 평균, VKOSPI 기반
stress-conditioned covariance, 거래비용, long-only·완전투자 SLSQP 제약을
유지하면서 근거가 약했던 보정 규칙만 교체한 연구 버전이다. 원본 Stage 36
파일은 수정하지 않았다.

결론부터 말하면 구현·인과성·수치 제약 검증은 통과했지만 Stage 36 대체
승격 조건은 실패했다. CAGR은 높아졌으나 변동성, MDD, 회전율이 커져 Sharpe가
낮아졌다. 따라서 이 폴더의 결과는 성과가 더 좋은 최종안이 아니라, 질문받은
메커니즘을 통계적으로 해석 가능한 형태로 바꾼 정직한 비교 실험이다.

## 무엇을 바꿨는가

### 1. `beta × R²`를 제거한 stress/recovery expected return

각 레짐 `g`에서 soft regime probability를 WLS 가중치로 사용해 stress와
recovery를 동시에 추정한다.

```text
r(i,t) = alpha(i,g)
       + beta_s(i,g) [stress(t) - weighted_mean_g(stress)]
       + beta_r(i,g) [recovery(t) - weighted_mean_g(recovery)] + error(i,t)
```

두 변수를 함께 넣었기 때문에 한 계수는 다른 변수에 대한 부분효과다. 표준오차는
월별 자기상관을 고려한 HAC(1)이다. 추정계수에는 다음 positive-part
signal-to-noise reliability를 적용한다.

```text
reliability = max(0, 1 - HAC_SE² / beta_hat²)
beta_tilde  = beta_hat × reliability
```

이는 `R²`처럼 전체 회귀 적합도를 개별 계수에 임의로 곱하는 규칙이 아니다.
`beta_hat² - SE²`를 0 아래에서 절단한 추정 신호분산으로 보고, 해당 계수에서
sampling noise를 제외한 비중만 남긴다. 과거 Stage 36의 KODEX200·USO 부호
강제도 제거했다. 부호가 불안정하면 HAC 표준오차가 커져 0으로 수축된다.

### 2. VKOSPI stress covariance가 금·채권·유가에 미치는 영향

기본 공분산은 Stage 36의 연속형 조건부 구조를 유지한다.

```text
Sigma_base(t) = [1 - stress(t)] Sigma_macro(t)
              + stress(t) Sigma_high-stress(t)
```

따라서 VKOSPI stress는 KODEX200만이 아니라 BOND, GLD, USO의 조건부 분산과
자산 간 공분산에도 영향을 줄 수 있다. 이것은 경제적으로도 가능한 현상이다.
위기 때 안전자산 선호, 달러/원, 원자재 수요, 유동성 청산 때문에 네 자산의
변동성과 동행성이 함께 바뀔 수 있기 때문이다.

다만 이 추정은 “VKOSPI가 금이나 채권을 인과적으로 움직인다”는 주장이 아니다.
과거 VKOSPI stress가 높았던 달의 조건부 공동분포를 위험 추정에 사용하는
것이다. Stage 48은 이 구분을 유지하고, GVZ·OVX·ATR·credit가 같은 위험을 다시
여러 번 곱하지 않도록 아래의 단일 잔차분산 보정만 추가한다.

### 3. `1 + rank`의 1~2배 선형 scaling 및 연속 곱 제거

월 `t`의 의사결정 후 실제 월중 일별 수익률로 다음 값을 기록한다.

```text
q(i,t) = log[realized_variance(i,t) / Sigma_base(i,i,t)]
```

다음 의사결정에서는 오직 과거 60개월 이상의 `q`만 사용해 자산별 HAC(1)
회귀를 적합한다.

```text
q(i,t) = a(i) + b_ATR(i) ATR_rank_state(t)
        + b_credit(i) credit_rank_state(t)
        + b_IV(i) implied_vol_rank_state(t) + error(i,t)
```

`b_IV`는 GLD에 GVZ, USO에 OVX만 사용한다. 모든 계수는 위와 같은
signal-to-noise shrinkage를 거치며, 표본 밖 외삽을 막기 위해 예측 로그비율은
과거에 관측된 target 범위 안에 둔다. 배수는 `m_i=exp(q_hat_i)`이고 최종 보정은
딱 한 번만 적용한다.

```text
D = diag(sqrt(m_1), ..., sqrt(m_4))
Sigma_final = D Sigma_base D
```

`D Sigma D`는 PSD와 기존 correlation을 보존한다. 별도 `1+rank` 네 개를
순차적으로 곱하지 않으므로 기계적인 1~2배 범위나 최대 약 4배의 복합 scaling이
없다. 관측된 배수는 자산별로 1보다 작거나 클 수 있다.

### 4. credit을 KODEX200 전용 confirmation으로 제한하지 않음

기존의 `credit_stress_multiplier × 전체 stress/recovery adjustment`는 제거했다.
그 규칙은 credit stress가 높은데 recovery 조정이 양수인 달에는 오히려 양의
expected return을 증폭할 수 있었다.

Stage 48은 모든 자산에 대해 다음 부분회귀를 별도로 적합한다.

```text
r(i,t) = alpha_i + beta_credit_i credit_widening_z(t)
       + controls[stress, recovery, macro_fragility, recent_return_i] + error
credit_mu_adjustment(i,t) = beta_tilde_credit_i × credit_widening_z(t)
```

신호는 월 `t` 전에 알려진 20거래일 AA- credit spread 변화다. credit 계수만
expected return에 더하므로 이미 macro/stress 모형에 들어간 통제변수 효과를
이중으로 더하지 않는다. KODEX200, BOND, GLD, USO를 모두 추정하되 근거가 약한
자산은 자산별 HAC uncertainty가 계수를 0 쪽으로 수축한다.

이는 “credit이 네 자산에 반드시 alpha다”라는 가정도 아니다. 적용 범위를 먼저
KODEX200으로 고정하지 않고, 각 자산에서 데이터가 지지하는 부분효과만 허용한다.
분산모형에서도 credit rank는 네 자산 모두의 residual variance 후보 설명변수다.

### 5. rank와 z-score를 나누는 기준

- **Causal expanding rank**: ATR, credit stress, GVZ, OVX처럼 “현재 위험 상태가
  과거 분포에서 얼마나 높은가”가 필요한 경우에 쓴다. 0~1로 bounded이고
  outlier magnitude에 덜 민감하므로 분산 상태 설명에 적합하다. 대신 부호와
  극단치 크기 정보가 약해져 return alpha에는 쓰지 않는다.
- **Causal expanding z-score**: credit widening처럼 방향과 surprise magnitude가
  expected return에 필요한 경우에 쓴다. 0은 중립, 양수는 widening, 음수는
  easing이다. Stage 48은 이미 causal z-score인 credit을 다시 표준화하지 않아
  이 0의 의미를 보존한다.

### 6. technical neutral과 cross-sectional mean

`mu_neutral = mean(mu_economic)`은 현금수익률이나 risk-free rate의 추정치가
아니다. 완전투자 제약 `sum(w)=1`에서는 모든 자산 expected return에 같은 상수
`c`를 더해도 목적함수에 `c`만 더해져 최적 비중이 바뀌지 않는다. 따라서
cross-sectional mean은 **active return의 정확한 0점**이다.

```text
active_i   = mu_i - mean(mu)
conflict_i = max(0, -sign(active_i) × technical_direction_i)
confidence_i = 1 - conflict_i
mu_filtered_i = mean(mu) + confidence_i × active_i
```

technical signal이 0이면 `confidence=1`이어서 expected return을 전혀 바꾸지
않는다. 같은 방향도 그대로 두고, 반대 방향일 때만 active view를 평균 쪽으로
축소한다. 기존처럼 neutral technical이 macro view를 자동으로 절반으로 만들지
않으며, 필터는 macro 조각만이 아니라 stress·EPS·valuation·credit까지 합친 전체
economic expected return에 적용한다. 과거 전체 평균, 0, risk-free rate를
neutral로 쓰면 기술 신호가 없는 상황에도 절대수익률 전망을 임의의 외부
앵커로 이동시키므로 cross-sectional veto 목적과 맞지 않는다.

## 성과 결과

기간은 두 전략에 공통인 2007-04~2026-07, 수익률은 거래비용 차감 후다.

| 전략 | CAGR | 변동성 | Sharpe | MDD | 평균 월 회전율 |
|---|---:|---:|---:|---:|---:|
| Stage36 | 10.50% | 9.47% | 1.105 | -12.41% | 3.70% |
| Stage48 | 12.31% | 12.90% | 0.967 | -16.46% | 10.14% |

2018-01~2026-07 locked 구간:

| 전략 | CAGR | 변동성 | Sharpe | MDD | 평균 월 회전율 |
|---|---:|---:|---:|---:|---:|
| Stage36 | 12.63% | 10.18% | 1.224 | -11.93% | 2.62% |
| Stage48 | 16.34% | 14.08% | 1.149 | -14.45% | 7.94% |

12개월 circular paired block bootstrap 2,000회의 전체 표본 결과는 CAGR 차이가
양수일 확률 92.0%, Sharpe 차이가 양수일 확률 2.3%, MDD 차이가 개선될 확률
1.2%였다. locked 구간도 각각 97.0%, 16.3%, 1.1%였다. 따라서 높은 CAGR만으로
승격하지 않았고 `promotion_pass=false`다.

## 검증과 파일

- `evidence_calibrated_stage36.py`: 실제로 호출되는 로더 연결, 추정, 최적화,
  백테스트, 검증만 포함한다.
- `outputs/stage48_monthly.csv`: 월별 비중·수익률·각 expected-return 조각·분산배수·
  학습 표본 종료월.
- `outputs/variance_calibration_history.csv`: 의사결정 이후에만 추가된 실현분산
  calibration record.
- `outputs/performance_comparison.csv`: Stage36/48 기간별 성과.
- `outputs/paired_block_bootstrap_vs_stage36.csv`: paired bootstrap 결과.
- `outputs/validation_report.json`: frozen hash, 인과성, PSD, 제약, 승격 gate.

Stage36의 후보별 연구 회귀표, 차트, HTML 보고서, notebook/Colab 경로, 사용하지
않는 overlay mode는 새 폴더에 복사하지 않았다. 자동검증은 14개 체크와 8개
pytest로 구성되며 모두 통과했다.

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\Activate.ps1'
python -m strategies.stage48_evidence_calibrated_stage36.evidence_calibrated_stage36
python -m pytest tests\test_stage48_evidence_calibrated_stage36.py -q
```

백테스트는 미래 성과를 보장하지 않는다. 특히 이번 결과는 경제적 설명 가능성을
높이는 것과 위험조정 성과 개선이 같은 명제가 아님을 보여준다.
