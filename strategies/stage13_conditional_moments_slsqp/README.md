# Stage13: 거시·VKOSPI·VIX6 조건부 모멘트 SLSQP

이 폴더는 Stage10의 `Sharpe + CAGR 점수 최대화` 코드를 그대로 덮어쓰지 않고,
별도 연구안으로 다시 설계한 버전이다. 핵심 목적은 숫자 조합으로 성과를 맞추는
것이 아니라 다음 질문에 코드가 직접 답하도록 만드는 것이다.

> 지금의 거시환경과 옵션시장이 말하는 단기 스트레스 아래에서 네 자산의
> 기대수익과 동반 하락 위험은 어떻게 달라지는가?

거시국면이나 VKOSPI 신호가 최종 비중을 직접 정하지 않는다. 두 정보는 오직
기대수익과 공분산, 하방위험 회피강도를 바꾸고, 최종 비중 100%는 한 번의
SLSQP 최적화가 결정한다. Hard 국면 비중, 사후 VKOSPI 비중 이동, 레버리지,
SJM, 로지스틱 회귀는 없다.

## 결론부터 보기

2007-04~2026-07의 저장된 결과는 다음과 같다.

| 전략 | CAGR | Sharpe | MDD | 실현 변동성 | 월평균 turnover |
|---|---:|---:|---:|---:|---:|
| Stage10 Sharpe+CAGR SLSQP | 6.86% | **1.329** | **-6.88%** | **5.10%** | 3.05% |
| Stage13 거시 조건부 모멘트 | **11.69%** | 0.887 | -21.22% | 13.53% | **2.56%** |
| Stage13 거시+스트레스 조건부 모멘트 | 11.50% | 0.901 | -21.32% | 13.07% | 2.70% |

Stage13은 Stage10보다 CAGR이 4.64%p 높지만 Sharpe와 MDD는 나쁘다. 따라서
이 결과만으로 기존 Stage10을 폐기하거나 Stage13을 운영 기준으로 승격하면
안 된다. Stage13의 개선은 **모형 구조와 설명 가능성의 개선**이며, 세 성과지표의
동시 개선을 뜻하지 않는다.

단기 스트레스 모형 자체를 공정하게 보려면 같은 Stage13 거시 모형끼리 비교해야
한다. 전체기간에는 스트레스를 넣은 뒤 CAGR이 0.19%p 낮아지고 Sharpe는 0.014
높아졌으며 MDD는 0.10%p 나빠졌다. 2018-01~2026-07에는 CAGR이 0.48%p
낮아지는 대신 Sharpe가 0.026 높아지고 MDD가 약 0.97%p 개선됐다. 즉,
스트레스 정보는 최근 구간의 위험조정성과에는 도움을 줬지만 전체기간에서 모든
지표를 동시에 개선하지는 못했다.

## Stage10에서 무엇을 바꿨나

### 1. 기대수익 추정

Stage10은 전체 평균 80%와 최근 EWMA 평균 20%를 섞은 뒤 월 -0.6%~1.5%로
잘랐다. 80:20과 상·하한은 결과에 큰 영향을 주지만 경제상태와 직접 연결되지
않는다.

Stage13은 과거 각 월의 네 국면확률을 관측 가중치로 사용한다.

    mu(r) = sum[P(r,s) * R(s)] / sum[P(r,s)]
    mu_macro(t) = sum[P(r,t) * mu(r)]

국면은 Goldilocks, Overheating, Slowdown, Stagflation 네 가지다. 과거 월을
하나의 국면으로 억지 분류하지 않고 당시의 확률 전체를 보존한다. 예를 들어
Goldilocks 확률이 0.6이었던 월은 Goldilocks 표본에 0.6만큼 기여한다.

### 2. 공분산 추정

각 국면의 확률가중 공분산을 구하고 현재 국면확률로 합친다.

    Sigma_macro(t) = sum[P(r,t) * Sigma(r)]

단기 스트레스를 쓸 때는 과거 월을 0/1로 나누지 않는다. 과거의 연속 스트레스
점수 자체를 가중치로 사용해 스트레스 공분산을 만들고, 현재 스트레스 점수만큼
거시 공분산에서 스트레스 공분산 쪽으로 이동한다.

    Sigma(t) = (1 - S(t)) * Sigma_macro(t)
               + S(t) * Sigma_high_stress(t)

### 3. VKOSPI/VIX6의 역할

Stage13에는 `VKOSPI가 높으면 주식을 몇 % 판다`는 코드가 없다. VKOSPI와
VIX6는 다음 세 경로로만 영향을 준다.

1. 국면별 기대수익의 스트레스 민감도
2. 현재 공분산의 스트레스 조건부 확대와 상관구조 변화
3. 목적함수의 하방 반분산 회피강도

따라서 SLSQP가 끝난 뒤 비중을 다시 수정하는 사후 overlay가 아니다. 보고서에서
`overlay attribution`이라고 부르는 것은 단기정보를 끈 동일 최적화와 켠 최적화의
성과 차이를 뜻한다.

### 4. 목적함수

Stage10처럼 Sharpe와 CAGR을 자산별 횡단면으로 표준화해 50:50으로 더하지
않는다. Stage13의 월 목적함수는 다음 한 식이다.

    U(w) = mu_p - 0.5 * variance_p
           - S(t) * downside_semivariance_p
           - estimated_transaction_cost(w, previous_w)

`mu - 0.5 * variance`는 기대 로그성장률, 즉 기대 CAGR의 월 단위 근사다.
하방 반분산은 음의 과거 수익률만 제곱한 평균이다. `S(t)`가 0~1 범위의 동적
위험회피계수 역할을 한다. turnover에는 임의의 벌점계수 대신 기존 백테스트에서
실제로 차감하는 국내 거래비용 15bp와 해외자산 비중변경비용 5bp를 그대로 쓴다.

연 단위로 쓰면 다음과 동일하다.

    12 * mu_p - 6 * variance_p
    - 12 * S(t) * downside_semivariance_p
    - 12 * estimated_transaction_cost

## 사용한 입력변수 전체

### 거시 원자료

| 축 | 원자료 | 변환 |
|---|---|---|
| 성장 | GDP 전년동월비 | causal expanding percentile |
| 성장 | 수출 전년동월비 | causal expanding percentile |
| 성장 | 제조업 업황전망 BSI | causal expanding percentile |
| 물가 | CPI 전년동월비 | causal expanding percentile |
| 물가 | PPI 전년동월비 | causal expanding percentile |
| 물가 | 수입물가 전년동월비 | causal expanding percentile |

세 성장 rank의 단순평균이 `p_growth_high`, 세 물가 rank의 단순평균이
`p_inflation_high`다. 두 확률로 다음 네 연속 국면확률을 만든다.

- `p_Goldilocks = growth * (1 - inflation)`
- `p_Overheating = growth * inflation`
- `p_Slowdown = (1 - growth) * (1 - inflation)`
- `p_Stagflation = (1 - growth) * inflation`

### VKOSPI 원자료와 가공변수

- `vkospi_close`: VKOSPI 종가
- `vkospi_log_change_5`: 다섯 관측치 로그변화
- `level_component`: VKOSPI 종가의 causal expanding percentile
- `vkospi_shock_rank`: 5관측치 충격의 causal expanding percentile

### VIX6 decomposition 여섯 원변수

- `sticky_strike`: 기초자산 이동만으로 예상되는 고정행사가 IV 변화
- `parallel_shift`: 실제 ATM IV 변화에서 sticky-strike 효과를 뺀 공통 이동
- `put_skew`: 풋 쪽 shoulder skew 변화
- `call_skew`: 콜 쪽 shoulder skew 변화
- `downside_convexity`: 왼쪽 꼬리 convexity 변화
- `upside_convexity`: 오른쪽 꼬리 convexity 변화

### VIX6 파생변수와 최종 스트레스 변수

- `vix6_left_impulse = put_skew + downside_convexity`
- `vix6_right_impulse = call_skew + upside_convexity`
- `vix6_tail_asymmetry = left_impulse - right_impulse`
- `parallel_shift_rank`: 광범위한 IV 재평가의 expanding percentile
- `left_impulse_rank`: 왼쪽 꼬리 충격의 expanding percentile
- `tail_asymmetry_rank`: 왼쪽 우위의 expanding percentile
- `shock_component`: VKOSPI 충격 rank와 parallel-shift rank의 동일가중 평균
- `tail_component`: left-impulse rank와 tail-asymmetry rank의 동일가중 평균
- `persistence_component`: level·shock·tail의 21관측치 평균
- `stress_raw`: level·shock·tail·persistence 네 블록의 동일가중 평균
- `stress_score`: `stress_raw`와 5관측치 평균 중 큰 값
- `recovery_score`: 5관측치 평균보다 현재 raw stress가 낮아진 정도의 expanding rank

동일가중을 쓴 이유는 네 블록 중 어느 하나가 더 중요하다는 사전 근거가 없기
때문이다. 5는 한 거래주, 21은 한 거래월이라는 시간 단위다. 상승 시 현재 점수를
즉시 사용하고 하락 시 5관측치 평균이 잠시 남게 해 risk-off는 빠르게, recovery는
한 주에 걸쳐 확인한다.

### 시장·실행 입력

- KODEX200, 채권, 원화환산 GLD, 원화환산 USO의 월수익률
- 직전 월말 이후 가격변동을 반영한 pre-trade 비중
- 국내 거래비용과 해외자산 비중변경비용

## 스트레스 기대수익 회귀

각 거시국면 안에서 자산별로 스트레스와 회복 민감도를 과거 데이터만으로
계산한다. 복잡한 분류기가 아니라 확률가중 단변량 OLS다.

    beta_stress(i,r) = Cov_r(stress, return_i) / Var_r(stress)
    beta_recovery(i,r) = Cov_r(recovery, return_i) / Var_r(recovery)

작은 표본의 우연한 큰 기울기를 그대로 쓰지 않기 위해 각 기울기에 해당 회귀의
R-squared를 곱한다. R-squared는 설명된 분산의 비율이므로 별도의 ridge 강도나
튜닝값이 필요 없다.

국면 평균과 공분산도 표본 초기에 그대로 믿지 않는다. 각 국면의 유효표본수와
한 달력연도(12개월)의 사전 표본을 이용해 expanding 전체 평균·공분산 쪽으로
축소한다. 유효표본이 쌓일수록 국면 추정치의 비중은 자동으로 커진다. 12는 성과를
보고 고른 shrinkage 강도가 아니라 최초 투자에도 요구한 한 연간계절과 같은 단위다.

주식과 원유에는 경제적 부호 제약을 둔다. 스트레스 민감도는 0 이하, 회복
민감도는 0 이상이다. 변동성 급등이 할인율과 유동성 프리미엄을 높이는 시점에
단순히 과거 반등이 자주 나왔다는 이유만으로 위험자산 기대수익을 올리는 것을
막는다. 채권과 금은 인플레이션 위기와 디플레이션 위기의 방향이 다르므로 부호를
데이터에 맡긴다.

    mu(t) = mu_macro(t)
            + sum_r P(r,t) * [
                beta_stress(r) * (S(t) - mean_r(S))
                + beta_recovery(r) * (Recovery(t) - mean_r(Recovery))
              ]

## SLSQP 제약과 숫자의 근거

| 값 | 역할 | 근거 |
|---:|---|---|
| 12개월 | 최초 추정 이력 | 한 번의 연간 계절을 관측한 뒤 시작 |
| 5관측치 | 충격·회복 확인 | 한 거래주 |
| 21관측치 | 지속성 | 한 거래월 |
| 13% | 예상 연변동성 상한 | 피드백의 12~13% 중 넓은 쪽을 택한 catastrophe guard |
| 16% | 과거 CDaR 상한 | Stage10에서 유지한 비상 제약 |
| 90% | CDaR 신뢰수준 | 최악 10% drawdown 평균 |
| 50% | 단일자산 상한 | 어느 한 자산도 포트폴리오 과반을 차지하지 못하게 함 |
| 15bp, 5bp | 비용 | 기존 실행비용 가정, 목적함수와 실현수익에 동일 적용 |

13% 변동성 상한은 Stage10의 약 8% 목표변동성처럼 평상시 비중을 정하는 장치가
아니다. 저장 결과에서 변동성 guard는 31/232개월, CDaR guard는 82/232개월
구속됐다. 여전히 적지 않으므로 향후에는 공분산 추정의 보수성을 먼저 점검해야
한다. 상한 숫자를 성과에 맞춰 더 높이는 방식은 권하지 않는다.

SLSQP의 `maxiter=300`, `ftol=1e-9`, 공분산 eigenvalue floor는 수치해석 설정이다.
경제 신호의 세기나 자산비중을 조절하는 하이퍼파라미터가 아니다.

## Risk Overlay Attribution

`macro_conditional_monthly.csv`는 단기 스트레스 효과만 끈 대조군이고,
`macro_stress_conditional_monthly.csv`가 완성형이다. 두 경로는 같은 거시확률,
같은 SLSQP, 같은 제약, 같은 거래비용을 쓴다.

전체 232개월에서 스트레스 모형은 위험자산(KODEX200+USO)을 줄인 달이 155개다.
사후 진단상 KODEX200 수익률 하위 10%인 24개월 중 12개월에 위험자산을 줄였고
12개월은 놓쳤다. 위험자산을 줄였지만 KODEX200이 오른 false positive는
93개월이다. 이 수치는 전략 입력이나 threshold 튜닝에 쓰지 않았으며, 사후
설명용이다.

스트레스 점수와 위험자산 감소량의 상관은 양(+)이고, 스트레스 최상위 20%에서
평균 위험자산 감소량도 가장 크다. 즉, 최종 비중을 직접 덮어쓰지 않아도 조건부
기대수익·공분산·하방위험 경로를 통해 의도한 risk-off 방향은 작동한다.

## 시점과 look-ahead 방지

투자월 `t`의 거시 신호월과 스트레스 신호월은 모두 `t-1`이다. 조건부 평균,
공분산, beta, R-squared에는 `t`보다 앞선 월수익률만 들어간다. expanding
percentile도 각 날짜까지 관측된 값으로만 계산한다. 저장 결과 232개월 모두에서
다음을 검사한다.

- 거시 신호월 < 투자월
- 스트레스 신호월 < 투자월
- 스트레스 실제 날짜 < 투자월
- 비중합 = 1
- 모든 비중 >= 0
- 모든 비중 <= 0.5
- 레버리지 없음
- solver fallback 0회

## 파일 안내

- `economic_conditional_slsqp.py`: 전체 입력 생성, 조건부 모멘트, SLSQP,
  백테스트, 귀속분석
- `economic_design_report.html`: 사람이 읽기 위한 상세 알고리즘 보고서
- `outputs/daily_stress_features.csv`: 일별 VKOSPI/VIX6 가공변수
- `outputs/monthly_stress_signals.csv`: 다음 달 투자에 실제 사용된 신호
- `outputs/macro_conditional_monthly.csv`: 스트레스를 끈 동일 구조 대조군
- `outputs/macro_stress_conditional_monthly.csv`: 최종 월별 비중·수익·solver 기록
- `outputs/risk_overlay_attribution.csv`: 월별 base/overlay 차이와 오탐·미탐 진단
- `outputs/performance_comparison.csv`: Stage10 포함 동일기간 성과표
- `outputs/research_report.json`: 설정, 검증, 귀속 결과의 기계판독 요약
- `tests/test_stage13_conditional_moments_slsqp.py`: 인과성·무레버리지·회귀 테스트

## 실행

프로젝트 루트에서 실행한다.

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage13_conditional_moments_slsqp.economic_conditional_slsqp
```

테스트는 다음과 같다.

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m pytest tests/test_stage13_conditional_moments_slsqp.py -q
```

## 해석상 주의

이 연구는 하이퍼파라미터 탐색을 하지 않았지만, 그렇다고 과적합 위험이 0이 되는
것은 아니다. 네 국면의 경제적 정의, 자산 universe, 거래비용, catastrophe guard
자체가 모두 모형 선택이다. 특히 전체기간 MDD가 Stage10보다 크고 최근 구간과
전체기간의 overlay 효과가 다르다. 따라서 다음 단계는 숫자를 더 탐색하는 것이
아니라 다음 검증이어야 한다.

1. 시작일 이동과 expanding-history 길이에 대한 안정성
2. 거래비용 2배 스트레스 테스트
3. 2008, 2020, 2022, 2026 위기별 기여도
4. 기대수익 beta를 완전히 제거했을 때의 성과 귀속
5. 13%·16% catastrophe guard가 구속된 달의 원인 감사
