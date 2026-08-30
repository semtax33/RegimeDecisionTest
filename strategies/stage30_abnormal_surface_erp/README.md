# Stage 30 — Pure ODS + 데이터 품질 + 인과적 μ 보정

## 결론부터

`Option Order Flow`는 완전히 제외했다. 추가 피드백에 따라 ERP도 주식 방향점수에서 뺐다. 최종 후보는 기존 Stage20의 VKOSPI/VIX6 위험 엔진을 그대로 두고, KOSPI200 옵션에서 추출한 **비정상 풋 스큐·IVA 변화**만 KODEX200 기대수익에 반영한다.

전체 2007-04~2026-07 결과는 CAGR과 MDD가 개선됐지만 Sharpe가 근소하게 낮아졌다. 세 지표를 동시에 개선하지 못했으므로 기본 전략은 Stage20으로 유지한다.

| 전략 | 구간 | CAGR | 변동성 | Sharpe | MDD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| Stage20 VIX6 | 2007-04~2026-07 | 9.3974% | 9.5981% | 0.9865 | -14.0148% | 0.6705 |
| Pure ODS, 품질 보정 없음 | 2007-04~2026-07 | 8.9451% | 10.1097% | 0.8999 | -16.9095% | 0.5290 |
| **Stage30 품질·인과 보정** | **2007-04~2026-07** | **9.7820%** | **10.0718%** | **0.9796** | **-13.5990%** | **0.7193** |

Stage30과 Stage20의 차이는 CAGR **+0.3845%p**, Sharpe **-0.0069**, MDD **+0.4158%p(개선)**다.

| 전략 | 구간 | CAGR | Sharpe | MDD |
|---|---|---:|---:|---:|
| Stage20 | 2007-04~2017-12 | 6.6493% | 0.7563 | -14.0148% |
| Stage30 | 2007-04~2017-12 | 6.5213% | 0.7404 | -13.5990% |
| Stage20 | 2018-01~2026-07 | 12.9392% | 1.2503 | -11.4190% |
| Stage30 | 2018-01~2026-07 | 14.0068% | 1.2405 | -11.5536% |

후기 구간에서는 수익률이 높아졌지만 Sharpe와 MDD가 소폭 나빠졌다. 초기 구간에서도 MDD만 개선됐다. 특정 구간에서만 압도적으로 좋아 보였던 Stage28보다 구간 차이는 줄었지만, 방향 알파가 안정적으로 강하다고 결론 내리기에는 아직 부족하다.

## 무엇을 바꿨나

### 1. 위험과 방향을 분리했다

```text
VKOSPI + VIX6 decomposition
        ↓
기존 Stage20 위험 엔진
        ↓
공분산·스트레스 기대수익·Vol/CDaR 제약

KOSPI200 옵션가격·IV·행사가·만기
        ↓
금리 반영 패리티 + 데이터 품질 감사
        ↓
비정상 Put Skew / IVA 변화
        ↓
Pure Directional ODS
        ↓
인과적 μ 보정 → KODEX200 μ만 조정
```

Stage28처럼 VIX6를 빼거나 ERP를 방향점수에 더하지 않는다. Stage20은 브레이크, Stage30 옵션 신호는 방향 판단이라는 역할 분리를 유지한다.

### 2. 무이자 put-call parity를 폐기했다

같은 만기·행사가의 콜과 풋으로 매일 다음 식을 횡단면 회귀한다.

```text
C(K) - P(K) = D × F - D × K
```

회귀 절편은 `D×F`, 행사가 기울기는 `-D`다. 따라서 별도 금리 파일 없이 할인계수 `D`와 선도가격 `F`를 동시에 추정한다. Stage28의 `D=1` 가정보다 과거 고금리 구간의 선도·델타 왜곡을 줄인다. 패리티 회귀 NRMSE는 뒤의 데이터 품질 신뢰도에도 들어간다.

### 3. 수준이 아니라 비정상 변화만 남겼다

5일과 20일 각각에 대해 다음 네 개의 expanding one-step-ahead OLS를 계산한다.

```text
ΔPutSkew_5  ~ KOSPI수익_5  + ΔVKOSPI_5  + DTE거리 + Roll + 성장 + 물가
ΔIVA_5      ~ KOSPI수익_5  + ΔVKOSPI_5  + DTE거리 + Roll + 성장 + 물가
ΔPutSkew_20 ~ KOSPI수익_20 + ΔVKOSPI_20 + DTE거리 + Roll + 성장 + 물가
ΔIVA_20     ~ KOSPI수익_20 + ΔVKOSPI_20 + DTE거리 + Roll + 성장 + 물가
```

각 날짜의 예측에는 그 날짜를 넣지 않고 `t-1`까지의 관측치만 사용한다. 최소 사전표본은 252거래일이다. 옵션·VKOSPI·거시 자료가 2003년부터 겹치므로 Stage30 신호는 2005-10-21부터 완성되고, 실제 2007-04 백테스트 시작 전에 충분한 사전 학습 구간이 있다.

성장 설명변수는 GDP·수출·BSI의 직전 월 causal rank 평균, 물가는 CPI·PPI·수입물가의 직전 월 causal rank 평균이다. 현재 월 거시자료를 당월 옵션 잔차에 넣지 않는다.

잔차를 과거값만으로 표준화한 뒤 다음처럼 결합한다.

```text
Bear_fast = mean(Z[abnormal ΔPutSkew_5],  Z[abnormal ΔIVA_5])
Bear_slow = mean(Z[abnormal ΔPutSkew_20], Z[abnormal ΔIVA_20])
PureDirection = -mean(Bear_fast, Bear_slow)
```

두 잔차가 모두 존재할 때만 평균을 낸다. 한쪽이 결측이면 남은 한쪽을 100% 가중하는 식으로 처리하지 않는다.

### 4. 데이터 품질이 나쁘면 신호를 줄였다

성과가 나쁜 과거 연도를 지정해서 끄지 않았다. 매 시점에 관찰 가능한 품질만 사용한다.

```text
Q = Q_DTE × Q_coverage × Q_parity × Q_arbitrage × Q_roll
UsableDirection = Q × PureDirection
Score = UsableDirection / (1 + |UsableDirection|)
```

- `Q_DTE`: 30일 목표 만기에서 멀수록 연속적으로 감소
- `Q_coverage`: 현재 OTM 행사가 폭을 그때까지 관측한 최대 폭으로 나눔
- `Q_parity`: `1 / (1 + parity NRMSE)`
- `Q_arbitrage`: 원가격의 수직 단조성과 convexity 충족 비율 평균
- `Q_roll`: 최근 5일 안에 만기 롤이 있으면 0, 아니면 1

옵션 표면은 4,995일이다. 정확한 30일 만기는 103일뿐이고 4,892일은 nearest-listed proxy다. 완성된 Pure ODS는 3,937일이다.

품질 보정 없는 대조군은 전체 CAGR 8.95%, Sharpe 0.90, MDD -16.91%였다. 품질·인과 보정을 적용하면 9.78%, 0.98, -13.60%로 세 지표가 모두 나아졌다. 이 비교는 품질 계층의 필요성을 보여주는 ablation이며, 여러 품질공식을 탐색해 가장 좋은 것을 고른 결과가 아니다.

### 5. ODS를 μ로 바꾸는 배율도 인과적으로 추정했다

Stage28의 `Score × 거시 μ 횡단면 표준편차`는 옵션 예측력과 배율의 경제적 연결이 약했다. Stage30은 매월 다음 관계의 기울기를 과거에 결과가 확정된 월만으로 다시 추정한다.

```text
KODEX200_return = a + b × OptionScore + error
Δμ_KODEX200 = max(b, 0) × 현재 OptionScore
```

최소 12개월의 과거 쌍을 요구한다. 비음수 제약을 둔 이유는 `OptionScore>0`을 사전에 강세로 정의했기 때문이다. 추정기울기가 음수라고 해서 신호 의미를 뒤집어 역베팅하지 않고, 그 시기에는 예측근거가 없다고 보고 조정을 0으로 축소한다.

검증 과정에서 음의 기울기까지 허용한 중간안은 전체 CAGR 11.00%, Sharpe 1.062, MDD -13.60%를 기록했다. 그러나 초기 12~20개월의 작은 표본에서 최대 월 3.42%p 조정이 발생했고, 절반가량의 월에 신호 의미를 반대로 사용했다. 경제적 설명과 과최적화 방지 원칙에 맞지 않아 그 결과는 공식 후보에서 폐기했다.

## ERP와 Option Order Flow 처리

- `Option Order Flow`: 거래량·거래대금·미결제약정 열을 표면 계산 전에 버린다. CPV나 `Volume×IV change` 대용치도 사용하지 않는다.
- `Option-implied ERP`: OTM 옵션가격 적분으로 만든 Martin식 SVIX² 대용치를 진단용으로만 저장한다. 방향점수·μ·공분산·제약·비중에는 들어가지 않는다.

ERP 진단 IC는 5일 `+0.0862`, 20일 `+0.1332`, 월간 `+0.1505`였지만 VIX6/VKOSPI 스트레스와 월간 상관도 `+0.8523`이었다. 즉 기대보상과 꼬리위험을 함께 담은 값이라서 방향점수에서 제외한 판단이 타당하다.

## 만기·데이터 품질 감사 결과

| 항목 | 2007~2017 | 2018~2026 |
|---|---:|---:|
| 평균 DTE | 17.98일 | 18.01일 |
| 평균 상장 계약 수 | 61.0 | 226.8 |
| 평균 log-moneyness coverage | 0.316 | 0.705 |
| 패리티 NRMSE | 0.0241 | 0.0389 |
| 단조성 품질 | 0.994 | 0.972 |
| convexity 품질 | 0.772 | 0.780 |
| IV 실패 비율 | 8.32% | 4.99% |
| 25Δ 최근접 행사가 거리 | 0.270% | 0.182% |
| 종합 품질 신뢰도 | 0.149 | 0.225 |

2018년 이후에는 계약 수와 행사가 커버리지가 크게 늘고 25Δ 보간 거리가 줄었다. 반면 zero/missing close 비율은 8.0%에서 48.4%로 높아졌고 패리티 오차도 조금 커졌다. 따라서 후기 성과를 단순한 시장 구조 변화 하나로 설명할 수 없다. 데이터 구성과 유동성 분포도 함께 바뀌었다.

순수 신호의 DTE별 IC도 일정하지 않다.

| DTE | 5일 IC | 20일 IC |
|---|---:|---:|
| 7~14일 | +0.0398 | +0.0543 |
| 15~30일 | -0.0548 | -0.0338 |
| 31~45일 | -0.0361 | +0.1727 |

31~45일은 165건뿐이며 대부분 롤 직후라 실제 사용할 때는 `Q_roll=0`으로 꺼진다. 롤 직후의 raw IC는 5일 -0.0029, 20일 +0.0195로 거의 0에 가까웠다. 롤 신호를 그대로 믿지 않도록 한 구조와 일치한다.

## 통계적 강도

월간 Pure ODS 점수의 KODEX200 동일 목표월 Spearman IC는 `+0.0261`로 약하다. 12개월 paired circular block bootstrap 2,000회 결과는 다음과 같다.

| Stage30-Stage20 | 평균 | 5% | 95% | 개선확률 |
|---|---:|---:|---:|---:|
| CAGR | +0.402%p | -0.137%p | +1.149%p | 84.2% |
| Sharpe | -0.0081 | -0.0393 | +0.0210 | 33.9% |
| MDD | +0.158%p | -0.477%p | +1.051%p | 52.4% |

CAGR 개선 가능성은 비교적 높지만 5% 경계는 음수다. Sharpe와 MDD는 우위가 통계적으로 분명하지 않다. 그래서 Stage30을 바로 실전 기본안으로 승격하지 않는다.

## 전체 입력변수와 변경되지 않은 부분

새 옵션 방향 블록이 직접 읽는 원자료는 다음뿐이다.

- 옵션: 일자, 콜/풋, 만기, 행사가, 종가, 내재변동성
- 현물: KOSPI200/KODEX200 종가 수익률
- 변동성: VKOSPI 종가 변화
- 거시: GDP, 수출, BSI, CPI, PPI, 수입물가의 직전 월 causal rank

Stage20에서 그대로 유지한 입력·알고리즘은 다음과 같다.

- 거시 4국면 확률: Goldilocks, Overheating, Slowdown, Stagflation
- 위험 엔진: VKOSPI 수준·충격, VIX6 parallel shift, put/call skew, downside/upside convexity, tail asymmetry, persistence, recovery
- 기술 신뢰도: 126일 K-ratio, 14일 ATR/NATR, 가격 RSI, 거래량 RSI
- 조건부 기대수익·공분산, 역사적 하방 semivariance, 거래비용
- SLSQP 목적함수와 λ=1
- 연 변동성 13% guard, CDaR 16% guard
- 롱온리, 합계 100%, 레버리지 없음
- 단일자산 50% 상한 없음, 사후 비중 오버레이 없음

Stage30이 바꾸는 것은 **KODEX200의 월 기대수익 μ 하나뿐**이다. 채권·금·원유 μ, 공분산, λ, Vol/CDaR 제약은 옵션 방향신호 때문에 바뀌지 않는다.

## 파일과 실행법

- `abnormal_surface_erp_slsqp.py`: 옵션 표면, 품질 감사, 잔차 OLS, 인과 μ 보정, SLSQP, 성과·검증 전체
- `outputs/performance_comparison.csv`: 전체·초기·후기 성과
- `outputs/daily_abnormal_surface_erp_features.csv`: 일별 표면·잔차·품질·ERP 진단
- `outputs/monthly_option_alpha_signals.csv`: 다음 달에 사용하는 신호와 과거기반 보정계수
- `outputs/option_construction_diagnostics.csv`: DTE·롤·시기별 품질 감사
- `outputs/factor_diagnostics.csv`: 5일·20일·월간 IC와 위험엔진 상관
- `outputs/paired_block_bootstrap.csv`: paired block bootstrap
- `outputs/validation_report.json`: 설계·수치·불변조건 종합 보고서
- `tests/test_stage30_abnormal_surface_erp.py`: 인과성·산식·성과 회귀 테스트

PowerShell에서 실행한다.

```powershell
$py = 'd:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage30_abnormal_surface_erp.abnormal_surface_erp_slsqp
& $py -m pytest tests\test_stage30_abnormal_surface_erp.py -q
```

최종 판정은 `retain_stage20`이다. Stage30은 데이터 품질 보정과 방향·위험 분리를 구현한 연구 후보로 보존한다. 백테스트 수치는 미래 성과를 보장하지 않는다.
