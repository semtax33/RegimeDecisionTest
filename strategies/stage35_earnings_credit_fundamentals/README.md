# Stage 35 — Earnings·Credit Fundamental Confirmation

## 결론

Stage20을 동결한 상태에서 기업의 **돈벌이**, **상대가치**, **자금조달 환경**을 각각 한 가지 역할로 추가했다.

- 기업이익 방향: Forward EPS 12개월 전망의 1개월 revision → KODEX200 기대수익
- 상대가치: Forward earnings yield − 국고채 10년 → 느린 기대수익 앵커
- 신용환경: AA− 회사채 3년 − 국고채 3년의 20일 확대 → 기존 VIX6 위험판단과 KODEX200 공분산 확인

최종 전략은 2007-04~2026-07에서 다음 성과를 기록했다.

| 전략 | CAGR | Sharpe | MDD | 월평균 회전율 | 누적 거래비용 |
|---|---:|---:|---:|---:|---:|
| Stage20 동결 | 9.397% | 0.987 | -14.015% | 2.68% | 1.86% |
| EPSAlpha | 9.864% | 0.996 | -14.015% | 2.87% | 2.02% |
| CreditAlpha | 8.370% | 0.875 | -17.024% | 4.90% | 3.45% |
| CreditRiskConfirmation | 9.132% | 1.011 | -13.818% | 2.56% | 1.77% |
| ValuationAnchor | 11.052% | 1.018 | -13.800% | 3.16% | 2.24% |
| **FundamentalDualRole** | **10.608%** | **1.057** | **-13.742%** | **3.66%** | **2.61%** |

Stage35 최종 전략은 Stage20보다 CAGR +1.211%p, Sharpe +0.071, MDD +0.273%p 개선됐다. 레버리지와 단일자산 과반금지 조건은 사용하지 않았고, 롱온리 합계 100%만 유지했다.

## 왜 세 변수를 사용했나

### 1. Forward EPS revision: 기업이 앞으로 더 벌 것으로 예상되는가

Forward EPS 수준은 장기 성장과 rolling window 효과가 섞여 비정상 시계열이 되기 쉽다. 따라서 원자료가 제공하는 `EPS(Fwd.12M) 변화율(1개월, 지배)`만 방향 신호로 썼다.

```text
애널리스트의 향후 12개월 이익 전망 상향
→ 기업 현금흐름 기대 개선
→ KODEX200 기대수익 상향
```

1주 revision, EPS acceleration, 여러 revision 기간은 탐색하지 않았다. 직접 계산한 21거래일 EPS 변화는 원자료 필드 교차검산에만 사용한다.

### 2. Earnings-yield gap: 주식이 장기채보다 얼마나 싼가

```text
EarningsYieldGap = 1 / ForwardPE12M − KTB10Y / 100
```

이 신호는 1개월 타이밍 지표로 보지 않았다. 6개월과 12개월 통제회귀가 사전 예상인 양수이고 10% 유의수준을 통과한 뒤에만 느린 anchor로 승격했다.

과거 12개월 선행수익이 완전히 관측된 표본만 사용해 expanding slope를 계산한다. 구한 12개월 효과를 12로 나누어 월간 KODEX200 μ에 넣는다. 미래 12개월 수익을 현재 계수 학습에 넣는 look-ahead는 없다.

### 3. Credit spread: 기업이 실제로 돈을 빌리기 어려워졌는가

```text
AACreditSpread = 회사채(무보증 3년) AA− − 국고채 3년
CreditWidening = AACreditSpread(t) − AACreditSpread(t−20거래관측)
```

절대 회사채 금리가 아니라 같은 만기의 국고채를 차감했다. 국고채 금리 자체의 통화정책·인플레이션 영향을 걷어내고 기업 신용 프리미엄을 남기기 위해서다.

CreditAlpha는 통계적 방향성이 있었지만 실제 인과적 백테스트에서는 CAGR·Sharpe·MDD가 모두 악화됐다. 따라서 최종 전략에서는 credit easing을 별도 수익 알파로 더하지 않는다. 회사채 확대가 미래 3개월 최대낙폭을 유의하게 설명한 결과에 맞춰 다음 두 위험경로만 확인한다.

1. 기존 VIX6 KODEX200 stress μ를 `2 × 과거 credit-widening 백분위`로 확인
2. KODEX200 조건부 분산을 `1 + 과거 credit-widening 백분위`만큼 확대

두 배수는 모두 과거 백분위의 자연스러운 0~1 범위에서 중립값이 각각 1이 되도록 만든 항등 매핑이다. 성과표를 보고 고른 문턱이 아니다.

## 통계적 메커니즘

기존 통제변수는 VIX6 stress, 최근 1개월 KODEX200 수익률, 21일 실현변동성, 거시 취약도다. 표준오차는 horizon에 맞춘 HAC를 사용했다.

| 검정 | 표준화 beta | HAC p값 | 판정 |
|---|---:|---:|---|
| EPS revision → 향후 1M 수익 | +1.809%p | 0.027 | 통과 |
| EPS revision → 향후 3M 수익 | +3.156%p | 0.016 | 통과 |
| Credit easing → 향후 3M 수익 | +3.471%p | <0.001 | 통과 |
| Credit widening → 향후 3M MDD | +1.087%p | 0.033 | 통과 |
| EY gap → 향후 6M 수익 | +3.434%p | 0.083 | 통과 |
| EY gap → 향후 12M 수익 | +11.460%p | 0.068 | 통과 |
| EY gap 12M Spearman IC | +0.161 | 0.016 | 통과 |

기간분할에서도 EPS 1개월 beta와 credit easing 3개월 beta는 2007~2017 및 2018~2026 모두 사전 부호인 양수였다. 유의성은 시기별로 달랐으므로 “모든 시기에 똑같이 강한 알파”로 해석하지 않는다.

## 최종 알고리즘

매월 목표월 직전 월의 마지막 공통 가용 관측치를 선택한다.

1. 원자료 신호를 직전 자료만으로 expanding 표준화한다.
2. 2000년 이후 KODEX200 월수익으로 EPS revision의 비음수 expanding slope를 계산한다.
3. 완전히 관측된 12개월 선행수익만으로 EY gap의 비음수 12개월 slope를 계산하고 월간으로 환산한다.
4. Stage20의 거시 4국면 확률, VIX6, 기술 신뢰도와 조건부 모멘트를 그대로 계산한다.
5. KODEX200 μ에 EPS와 valuation 조정만 더한다.
6. AA credit widening 과거순위로 기존 VIX6 stress μ와 KODEX200 공분산을 확인한다.
7. 기존 목적함수와 변동성·CDaR 제약을 둔 SLSQP가 네 자산 비중을 연속적으로 결정한다.
8. 국내 거래비용과 GLD·USO 외화 비중 변경비용을 차감한다.

```text
μ_K200(final)
= μ_K200(Stage20)
+ β_EPS(t, causal) × z_EPS(t)
+ β_EYGap,12M(t, causal) × z_EYGap(t) / 12
+ confirmed VIX6 stress adjustment
```

가중치에는 hard regime 전환, 레버리지, 단일자산 상한, 사후 overlay가 없다.

## 사용한 입력변수

### 새 원자료와 가공변수

| 구분 | 입력변수 | 최종 전략 사용 |
|---|---|---|
| 이익 | Forward P/E 12M | earnings-yield 계산 |
| 이익 | Forward EPS 12M | 교차검산 |
| 이익 | EPS revision 1주 | 감사만 수행 |
| 이익 | EPS revision 1개월 | 주식 μ |
| 이익 | 계산 EPS 21일 revision | 교차검산 |
| 가치 | 1/Forward PE − 국고10년 | 느린 주식 μ |
| 금리 | 국고채 1·2·3·5·10·20·30·50년 | 3년·10년 사용, 나머지 보존 |
| 신용 | 회사채 AA− 3년, BBB− 3년 | AA− 주 위험축, BBB− 강건성 |
| 신용 | AA−−국고3년 | 주 credit spread |
| 신용 | BBB−−국고3년 | 강건성 진단 |
| 신용 | BBB−−AA− | quality spread 진단 |
| 신용 | 각 spread 20일 변화 | AA 변화만 최종 위험확인 |
| 금리 | 국고10년−국고3년 | 중복 가능성 진단, 미거래 |

### Stage20에서 그대로 유지한 입력

- 성장축: GDP YoY, 수출 YoY, BSI와 각 3개월 변화
- 물가축: CPI YoY, PPI YoY, 수입물가 YoY와 각 3개월 변화
- 국면확률: Goldilocks, Overheating, Slowdown, Stagflation
- 일간 위험축: VIX6 decomposition 기반 stress/recovery
- 기술축: 126일 K-ratio, Wilder ATR(14), KODEX200 가격 RSI(14), 거래량 RSI(14)
- 자산 수익률·조건부 공분산: KODEX200, 국내채권, GLD, USO
- 거래 전 비중과 국내/외화 거래비용

## 과최적화를 줄인 장치

- EPS 기간은 공급자 1개월 필드 하나로 고정
- credit 변화는 20거래관측 하나로 고정
- z-score·계수·순위는 모두 현재월 이전 데이터만 사용
- 계수는 비음수 인과적 OLS slope이며 임의 배율 없음
- valuation은 12개월 목표가 완전히 끝난 표본만 학습
- RSI·MACD·breakout, acceleration, 문턱 grid, horizon best-pick 없음
- AA 신호가 주 신호이며 BBB·quality는 거래하지 않고 부호 강건성만 확인
- 구성요소 경로는 역할 귀속용이고 최종 전략은 하나만 승격 가능
- 2000년 pre-sample 75개월을 이용해 2007-04부터 계수 작동
- 전체, 2007~2017, 2018~2026, 12개월 블록 부트스트랩을 모두 보고

## 기간분할

| 전략·기간 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| Stage20 · 2007~2017 | 6.649% | 0.756 | -14.015% |
| Stage35 · 2007~2017 | 8.106% | 0.884 | -13.742% |
| Stage20 · 2018~2026 | 12.939% | 1.250 | -11.419% |
| Stage35 · 2018~2026 | 13.817% | 1.251 | -11.977% |

2018년 이후 MDD는 Stage20보다 0.56%p 나쁘다. 전체기간 MDD는 개선됐지만 하위기간의 낙폭 우위가 일관되다고 주장하지 않는다.

## 부트스트랩

12개월 paired block bootstrap 2,000회 기준 Stage35−Stage20 개선확률은 다음과 같다.

- CAGR: 89.7%
- Sharpe: 76.4%
- MDD: 57.2%

CAGR과 Sharpe 증거는 비교적 강하지만 MDD 개선 증거는 약하다. 따라서 -13.74%라는 단일 MDD 숫자를 구조적 우위로 과장해서는 안 된다.

## 실행

프로젝트 루트에서 다음을 실행한다.

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage35_earnings_credit_fundamentals.earnings_credit_fundamentals_slsqp
```

주요 산출물은 `outputs/validation_report.json`, `performance_comparison.csv`, `monthly_earnings_credit_signals.csv`, 각 전략 월별 경로와 세 개의 PNG 차트다.

## 최종 판정

사전 메커니즘과 성과 게이트를 모두 통과했으므로 `Stage35_FundamentalDualRole`을 Stage20의 후속 후보로 승격한다. 다만 이는 표본 내 역사적 백테스트 결과이며 미래 성과를 보장하지 않는다. 실전 도입 전에는 데이터 입수시각과 수정 이력, 세금·슬리피지, 실제 상품 괴리를 별도로 검증해야 한다.
