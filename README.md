# RegimeDecisionTest — Stage36 GVZ·OVX 위험 오버레이

이 저장소는 한국 거시국면을 기반으로 `KODEX200·국내채권·GLD·USO`를 배분하고,
VKOSPI·VIX6·기술지표·기업이익·밸류에이션·신용·자산별 옵션 내재변동성을
순차적으로 결합해 온 연구 프로젝트다.

현재 이 README가 중심적으로 설명하는 경로는 **Stage36 — GVZ·OVX 자산별
내재변동성 위험 오버레이**다.

이 문서의 전략 설명과 순서는
[`Stage36_GVZ_OVX_Colab.ipynb`](strategies/stage36_asset_implied_volatility_risk/colab/Stage36_GVZ_OVX_Colab.ipynb)의
22개 설명 셀을 기준으로 작성했다. 메인 README에서는 노트북 코드를 그대로
나열하기보다, 각 코드 덩어리가 어떤 경제 질문에 답하고 다음 단계로 무엇을
넘기는지를 중심으로 설명한다.

> Stage35가 거시·공포·기술·펀더멘털 정보를 기대수익 `μ35`와 위험행렬
> `Σ35`로 압축하고, Stage36은 기대수익을 바꾸지 않은 채 GVZ를 GLD,
> OVX를 USO의 위험축에만 추가한다.

## 먼저 볼 파일

| 목적 | 파일 |
|---|---|
| Stage36 전체 설명 | [`strategies/stage36_asset_implied_volatility_risk/README.md`](strategies/stage36_asset_implied_volatility_risk/README.md) |
| Google Colab 실행 | [`Stage36_GVZ_OVX_Colab.ipynb`](strategies/stage36_asset_implied_volatility_risk/colab/Stage36_GVZ_OVX_Colab.ipynb) |
| Colab 업로드 데이터 | [`stage36_colab_data.zip`](strategies/stage36_asset_implied_volatility_risk/colab/stage36_colab_data.zip) |
| 프로젝트 통합형 코드 | [`asset_implied_volatility_risk_slsqp.py`](strategies/stage36_asset_implied_volatility_risk/asset_implied_volatility_risk_slsqp.py) |
| 경제·수학·코드 상세 가이드 | [`stage36_implementation_economic_math_guide.html`](strategies/stage36_asset_implied_volatility_risk/stage36_implementation_economic_math_guide.html) |
| 성과 시각화 보고서 | [`stage36_gvz_ovx_report.html`](strategies/stage36_asset_implied_volatility_risk/stage36_gvz_ovx_report.html) |
| 기계판독 검증결과 | [`outputs/validation_report.json`](strategies/stage36_asset_implied_volatility_risk/outputs/validation_report.json) |

전체 구현을 분석하거나 수정할 때는 루트 README만 보지 말고 반드시
[`strategies/stage36_asset_implied_volatility_risk/`](strategies/stage36_asset_implied_volatility_risk/)
폴더를 함께 참고해야 한다. 해당 폴더에는 통합 소스, 독립 Colab 코드, 상세 HTML,
월별 결과, 차트, bootstrap과 검증 JSON이 모두 들어 있다.

## 노트북 설명을 기준으로 한 읽기 순서

| 노트북 절 | 먼저 이해할 질문 | 메인 README 위치 |
|---|---|---|
| 1~2 | 어떤 자산·제약을 쓰고 데이터 ZIP을 어떻게 검증하는가 | 2절·9절 |
| 3~4 | 미래를 보지 않는 변환과 원화 월수익률은 어떻게 만드는가 | 2절 |
| 5~7 | 거시확률·VKOSPI/VIX6가 기본 `μ·Σ`로 어떻게 연결되는가 | 2절 |
| 8~9 | K-ratio·RSI·ATR·EPS·밸류·신용의 역할은 무엇인가 | 2절 |
| 10~11 | GVZ/OVX `DΣD`와 SLSQP는 무엇을 하는가 | 3~4절 |
| 12~14 | 비용·성과지표·bootstrap·미래위험 진단은 어떻게 계산하는가 | 5·7·8절 |
| 15~19 | 네 비교경로를 어떻게 실행·감사·저장하는가 | 6·9·12·14절 |

노트북을 실제로 열어 볼 때도 위 순서대로 설명 셀을 먼저 읽고 바로 다음 코드
셀을 확인하는 방식이 가장 이해하기 쉽다.

---

## 1. Stage36의 핵심 아이디어

GVZ는 금 옵션시장의 기대변동성, OVX는 원유 옵션시장의 기대변동성을 나타낸다.
Stage36은 이를 가격 방향 예측값으로 쓰지 않는다.

```text
GVZ 상승 → GLD 기대수익을 낮춘다       X
OVX 상승 → USO 매도신호를 만든다       X

GVZ 상승 → GLD 위험을 더 크게 계산한다 O
OVX 상승 → USO 위험을 더 크게 계산한다 O
```

경제적 질문은 다음과 같다.

> 옵션시장이 해당 자산의 미래 변동성 보험을 비싸게 평가할 때, 같은 자산비중을
> 평소와 동일한 위험으로 계산해도 되는가?

이 구조는 방향 알파를 억지로 추가하는 대신, 독립적인 옵션시장 위험정보를 기존
공분산에 연결한다.

### Stage36의 고정 경계

- GVZ는 GLD에만 연결
- OVX는 USO에만 연결
- 현재값 이전 252개 유효 일간관측 후 활성화
- 목표월에는 직전 월말까지 알려진 마지막 값만 사용
- 출시 전 backfill 없음
- 활성화 전 위험배수 1.0
- 활성화 후 위험배수 `1 + causal rank`, 범위 1~2
- GVZ/OVX 기대수익 조정은 항상 0
- long-only, 비중합 100%, 무레버리지·무공매도
- 단일자산 과반금지 없음
- Stage36 센서문턱·배수에 대한 grid search 없음

---

## 2. Stage35에서 물려받는 본체

Stage36만 읽으면 `μ35·Σ35`가 갑자기 만들어지는 것처럼 보일 수 있다. 실제
계산은 다음 계층을 누적한다.

| 계층 | 입력 | 역할 |
|---|---|---|
| 성장 | GDP YoY, 수출 YoY, BSI | 성장축 `g` |
| 물가 | CPI YoY, PPI YoY, 수입물가 YoY | 물가축 `π` |
| 거시국면 | `g`, `π` | 네 soft-regime 확률과 조건부 `μ·Σ` |
| 시장공포 | VKOSPI 수준·5일 변화, VIX6 표면 | stress/recovery |
| 추세확인 | 126일 K-ratio | 거시 기대수익 신뢰도 |
| 주식 기술 | price RSI·volume RSI | KODEX200 방향 확인 |
| 실현위험 | ATR/가격 | 자산별 공분산 축 확대 |
| 기업이익 | forward EPS·1M revision | KODEX200 기대수익 |
| 밸류에이션 | `1/Forward PER−국고채10Y` | KODEX200 장기 기대수익 |
| 신용 | AA- 회사채3Y−국고채3Y, 20일 변화 | 주식 stress·위험 확인 |
| 최적화 상태 | 과거 월수익·직전 drift 비중 | downside·CDaR·거래비용 |

### 인과적 변환: 왜 expanding percentile을 쓰는가

GDP 3%, BSI 95, GVZ 24처럼 단위가 다른 값을 그대로 평균하면 숫자가 큰 변수가
결과를 지배한다. 각 관측을 “그때까지의 자기 역사에서 어느 위치인가”라는 0~1
척도로 바꿔야 서로 비교할 수 있다.

```text
q_t = [현재보다 작은 과거 관측수 + 동률 midrank] / 현재까지의 관측수
```

중요한 것은 2020년 값을 판단할 때 2021년 이후 자료를 분포에 넣지 않는다는
점이다. `causal_zscore`도 현재월을 제외한 과거 평균·표준편차만 사용한다.

- 거시와 시장신호: expanding percentile·midrank
- EPS·밸류·신용 z-score: 현재월 이전 최소 60개월
- GVZ·OVX: 현재일 이전 최소 252개 유효 일간관측
- 이력이 부족하면 값을 억지로 채우지 않고 해당 효과를 비활성화

이 구조는 전체표본 평균·표준편차를 미리 아는 look-ahead를 막고 서로 다른 단위의
변수를 공통척도로 만든다.

### 네 자산의 원화 월수익률

전략 수익률은 목표월 첫 거래일 시가에서 다음 달 첫 거래일 시가까지 계산한다.
직전 월말 신호를 확인한 뒤 실제로 거래 가능한 다음 시점부터 수익을 측정하려는
시간순서다.

- KODEX200: 2009년 3월까지 `compass.db` KOSPI200 프록시를 실제 ETF의 첫
  관측레벨에 맞춰 연결
- BOND: KRX 채권 총수익지수
- GLD·USO: USD 가격에 USDKRW를 곱해 원화 OHLC와 월수익률 생성

프록시 레벨 정합은 초기 수익흐름을 연결하는 작업이지 미래 수익을 채우는
backfill이 아니다. 해외자산의 기술지표도 원화 OHLC에서 계산하므로 신호와
백테스트 수익률의 통화기준이 일치한다.

### 거시확률

성장·물가 원변수는 각각 자기 과거에서의 인과적 percentile로 바꾼다.

```text
g = mean(rank(GDP), rank(Export), rank(BSI))
π = mean(rank(CPI), rank(PPI), rank(ImportPrice))

p_Goldilocks  = g × (1-π)
p_Overheating = g × π
p_Slowdown    = (1-g) × (1-π)
p_Stagflation = (1-g) × π
```

HMM·SJM·CJM·로지스틱 회귀로 국면을 학습하지 않는다. 네 확률을 직접 계산하고
과거 월수익을 확률가중해 조건부 평균과 공분산을 만든다. 국면 유효표본이 작으면
`n_eff/(n_eff+12)` 신뢰도로 무조건부 통계에 수축한다.

예를 들어 `g=0.7`, `π=0.8`이면 다음과 같은 혼합상태가 된다.

| 국면 | 확률 |
|---|---:|
| Goldilocks | 14% |
| Overheating | 56% |
| Slowdown | 6% |
| Stagflation | 24% |

“이번 달은 무조건 과열”이라고 하드 분류하지 않고 과열 성격 56%, 스태그플레이션
성격 24%처럼 나눈다. 경계 부근의 작은 데이터 변화가 포트폴리오를 통째로
뒤집는 것을 줄이기 위해서다. GDP·수출에는 한 달, 물가에는 두 달의 공표지연을
반영한다.

### VKOSPI·VIX6: 거시보다 빠른 공포를 읽는다

거시자료는 느리므로 시장에서 직접 관찰되는 공포를 네 블록으로 만든다.

1. VKOSPI 수준: 공포가 구조적으로 높은가
2. VKOSPI 5일 로그변화와 VIX6 parallel shift: 공포가 갑자기 폭발했는가
3. put/call skew와 downside/upside convexity: 하락꼬리 보험이 얼마나 비싼가
4. 앞선 세 블록의 21거래일 평균: 하루 충격이 아니라 지속되는 상태인가

네 블록을 동일가중한 `stress_raw`와 최근 5일 평균 중 큰 값을 `stress_score`로
사용한다.

```text
stress_score = max(stress_raw, recent_5day_mean)
```

공포 급등은 바로 반영하지만 하루의 안도랠리만으로 완전히 회복했다고 판단하지
않는 단순 히스테리시스다. 현재 stress가 5일 평균 아래로 내려간 정도는
`recovery_score`로 변환한다. 목표월에는 직전 월말까지 관찰된 마지막 일간값만
사용한다.

### soft-regime 조건부 평균과 공분산

과거 월 `i`의 Goldilocks 확률이 70%라면 그 월 수익은 Goldilocks 통계에 0.7만큼
기여한다. 이렇게 국면별 가중평균수익과 가중공분산을 계산한다.

확률가중치가 몇 달에 집중되면 달력상의 월수보다 실제 정보량이 작아지므로
effective sample size를 계산한다.

```text
n_eff,k = sum(p_i,k)^2 / sum(p_i,k^2)
c_k     = n_eff,k / (n_eff,k + 12)

μ_k = c_k × raw_μ_k + (1-c_k) × μ_all
Σ_k = c_k × raw_Σ_k + (1-c_k) × Σ_all
```

표본이 충분하면 국면 고유통계를 많이 믿고, 부족하면 전체시장 통계로 후퇴한다.
`12`는 한 해의 무조건부 prior이며 결과를 보고 탐색한 값이 아니다.

현재 네 국면확률로 국면통계를 다시 섞어 `μ_macro·Σ_macro`를 만든다.
VKOSPI/VIX6 stress와 recovery의 국면별 OLS 기울기는 자기 R²에 해당하는
reliability로 축소한다. 주식·원유에는 `stress beta≤0`, `recovery beta≥0`의
경제부호를 둬 표본우연이 공포를 호재로 해석하는 것을 막는다. 공분산의 작은
음의 고유값은 PSD 보정해 `w'Σw`가 음수가 되는 수치오류도 차단한다.

### 기술신뢰도

K-ratio와 RSI는 별도 알파를 기대수익에 더하지 않는다. 거시 상대전망과 기술방향이
일치하는 정도에 따라 거시전망을 보존하거나 네 자산 평균 쪽으로 축소한다.

```text
agreement_i  = sign(μ_i - mean(μ)) × tech_i
confidence_i = clip((1 + agreement_i) / 2, 0, 1)
filtered_μ_i = mean(μ) + confidence_i × (μ_i - mean(μ))
```

`confidence`는 예측이 맞을 확률이 아니라 거시 상대전망의 보존계수다.
`agreement=-1, 0, +1`을 `confidence=0, 0.5, 1`로 옮기는 대칭 선형식이며,
충돌하더라도 기대수익 부호를 반대로 뒤집지 않는다.

#### K-ratio·RSI·ATR의 구체적 역할

126일 K-ratio는 로그가격 추세의 기울기를 그 기울기의 표준오차로 나눠 방향과
안정성을 함께 본다. 같은 상승률이어도 들쭉날쭉한 경로보다 꾸준한 상승경로의
점수가 높다.

```text
k_score = K / (1 + abs(K))       # -1과 1 사이로 압축
price_strength  = (price_RSI - 50) / 50
volume_strength = (volume_RSI - 50) / 50
```

KODEX200의 기술방향은 `k_score·price_strength·volume_strength`의 평균이고,
다른 자산은 K-score를 사용한다. 가격은 방향, 거래량은 참여도를 확인한다.

ATR은 방향이 아니라 위험이다. `NATR=ATR/가격`의 causal rank가 0.9이면 해당
자산의 variance multiplier는 1.9다. `D=diag(sqrt(1+rank))`로 `DΣD`를 계산해
분산과 다른 자산과의 공분산을 같은 위험축에서 조정한다.

### EPS·밸류에이션·신용

Forward EPS의 절대수준보다 1개월 revision을 사용해 기업이익 전망의 개선·악화를
본다. 현재 revision은 과거 60개월 이상으로 만든 causal z-score로 표준화하고,
과거 KODEX200 월수익과의 expanding slope로 기대수익 단위에 맞춘다.

```text
β_EPS = max(0, expanding univariate slope)
Δμ_EPS = β_EPS × EPS_revision_z
```

과거에 도움이 없으면 0으로 두며 음의 부호로 뒤집어 새 알파를 만들지 않는다.

밸류에이션은 PER만 보지 않고 채권 대안수익률과 비교한다.

```text
EarningsYieldGap = 1 / Forward_PER - KTB_10Y_yield
```

PER 10배라도 국고채가 3%일 때와 8%일 때 주식의 상대매력은 다르다. EYG는
완전히 관측된 과거 12개월 선행수익으로 인과 보정하고 월 단위로 KODEX200
기대수익에 반영한다.

신용위험은 `AA-회사채3Y−국고채3Y`와 그 20거래일 변화를 사용한다. 확대분의
60개월 causal rank를 `q_credit`이라 하면 주식 stress 조정에는 `2q_credit`,
KODEX200 공분산 축에는 `1+q_credit`을 적용한다. 신용악화는 독립 매도알파가
아니라 기존 공포신호와 주식위험을 확인하는 층이다.

---

## 3. GVZ·OVX를 `Σ35`에 연결하는 방법

각 센서의 현재값은 현재보다 앞선 관측만 들어 있는 분포에서 순위화한다.

```text
q_GVZ,t = causal expanding rank(GVZ_t)
q_OVX,t = causal expanding rank(OVX_t)

m_GLD,t = 1 + q_GVZ,t
m_USO,t = 1 + q_OVX,t
```

대각행렬을 만든 뒤 Stage35 공분산의 양쪽에 곱한다.

```text
D   = diag(1, 1, sqrt(m_GLD), sqrt(m_USO))
Σ36 = D @ Σ35 @ D
μ36 = μ35
```

제곱근을 사용하므로 GLD와 USO의 분산은 각각 정확히 `m_GLD`, `m_USO`배가
되고 다른 자산과의 공분산도 같은 위험축에 맞게 조정된다.

실제 활성화 시점:

| 센서 | 최초 활성 목표월 | 활성 전 처리 |
|---|---|---|
| GVZ | 2009-07 | GLD multiplier 1.0 |
| OVX | 2008-06 | USO multiplier 1.0 |

---

## 4. SLSQP 목적함수와 제약

매월 다음 효용을 최대화한다.

```text
U(w) = w'μ
       - 0.5 × w'Σw
       - mean(min(R_history @ w, 0)^2)
       - transaction_cost(w, pretrade_w)
```

- 기대수익을 보상
- 조건부 분산을 벌점
- 과거 손실구간의 하방반분산을 추가 벌점
- 직전 drift 비중에서 새 비중으로 이동하는 거래비용을 차감

제약:

```text
sum(w) = 1
0 <= w_i <= 1
expected annual volatility <= 13%
historical CDaR(90%) >= -16%
```

13%는 Stage36 결과로 새로 찾은 최적값이 아니라 이전 Stage에서 상속한 예상
연변동성 guard다. -16%도 Stage10 이후 유지된 역사적 경로위험 guard다.
CDaR는 최악 10% 과거 drawdown의 평균이며 미래 MDD를 보장하지 않는다.

현재 Stage36 결합경로에서는 232개월 중 변동성 제약이 106개월, CDaR 제약이
3개월에 구속된다. 따라서 13%는 원래 catastrophe guard로 도입됐지만 현재
전략에서는 실질적인 위험예산으로도 작동한다.

---

## 5. 월별 워크포워드

1. 목표월보다 앞선 월수익만 추정이력으로 사용한다.
2. 직전 월말까지 알려진 거시·VKOSPI/VIX6·기술·펀더멘털 신호를 읽는다.
3. Stage35의 `μ35·Σ35`를 만든다.
4. GVZ·OVX로 `Σ36`을 만든다.
5. 직전 drift 비중을 초기점으로 SLSQP를 실행한다.
6. 비중을 먼저 확정한 뒤 해당 월 실현수익을 적용한다.
7. 국내 비중변화 15bp와 해외 순비중변화 5bp를 차감한다.
8. 실현수익으로 다음 달 리밸런싱 직전 비중을 계산한다.

이 순서를 통해 당월 수익이 당월 비중결정에 들어가는 look-ahead를 차단한다.

---

## 6. 비교경로

| 경로 | 목적 |
|---|---|
| `Stage35_Frozen` | 동결 기준 |
| `Stage36_NoOverlayReproduction` | Stage35 완전 재현 감사 |
| `Stage36_GVZGoldRisk` | GVZ→GLD 단독 귀속 |
| `Stage36_OVXOilRisk` | OVX→USO 단독 귀속 |
| `Stage36_GVZ_OVXAssetRisk` | 사전 선언된 최종 결합경로 |

단독경로는 가장 좋은 센서를 선택하기 위한 후보탐색이 아니라 결합효과의 원인을
나누어 보는 ablation이다.

---

## 7. 성과

### 전체구간 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 10.608% | 10.043% | 1.057 | -13.743% |
| GVZ→GLD | 10.262% | 9.511% | 1.078 | -12.407% |
| OVX→USO | 10.846% | 10.005% | 1.083 | -13.743% |
| **GVZ+OVX** | **10.499%** | **9.472%** | **1.105** | **-12.407%** |

### 센서 공통구간 2010-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 9.653% | 9.371% | 1.033 | -13.743% |
| **GVZ+OVX** | **9.232%** | **8.719%** | **1.059** | **-12.407%** |

### 최근 잠금구간 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 13.821% | 10.861% | 1.251 | -11.984% |
| **GVZ+OVX** | **12.626%** | **10.178%** | **1.224** | **-11.931%** |

최근구간에서는 위험이 조금 낮아졌지만 CAGR과 Sharpe도 낮아졌다. Stage36은
모든 시기에 Stage35를 지배하는 신규 알파가 아니라 더 방어적인 위험관리 선택이다.

### 2,000회 paired block bootstrap

| 구간 | ΔCAGR > 0 | ΔSharpe > 0 | ΔMDD > 0 |
|---|---:|---:|---:|
| 전체 | 39.20% | 97.15% | 95.10% |
| 공통 2010+ | 12.10% | 86.85% | 93.30% |

Sharpe와 MDD 개선 증거는 비교적 강하지만 CAGR 개선 증거는 없다.

---

## 8. 미래위험 검증

GVZ·OVX는 미래수익 방향이 아니라 자기자산의 다음 1개월 실현변동성,
1·3개월 최대낙폭과 왼쪽꼬리를 설명하는지 검사한다. 최근 자기자산 수익률·
21일 실현변동성·VIX6 stress·거시 취약도를 통제한 HAC 회귀를 사용한다.

2010년 이후 주요 결과:

| 센서 | 목적변수 | 표준화 beta | HAC p-value | Spearman IC |
|---|---|---:|---:|---:|
| GVZ | GLD 미래 1M 실현변동성 | +0.01955 | 0.0029 | +0.492 |
| GVZ | GLD 미래 3M MDD 크기 | +0.00931 | 0.0871 | +0.398 |
| OVX | USO 미래 1M 실현변동성 | +0.04984 | 0.0010 | +0.588 |
| OVX | USO 미래 3M MDD 크기 | +0.02322 | 0.0369 | +0.170 |

이 회귀는 옵션센서를 공분산에 넣는 경제적 근거를 진단하기 위한 것이며 월별
비중결정에는 사용되지 않는다.

---

## 9. Google Colab에서 재현

필요한 파일:

1. [`Stage36_GVZ_OVX_Colab.ipynb`](strategies/stage36_asset_implied_volatility_risk/colab/Stage36_GVZ_OVX_Colab.ipynb)
2. [`stage36_colab_data.zip`](strategies/stage36_asset_implied_volatility_risk/colab/stage36_colab_data.zip)

실행방법:

1. 노트북을 Colab에서 연다.
2. **런타임 → 모두 실행**을 선택한다.
3. 업로드 창에 데이터 ZIP 하나만 올린다.
4. 성과·비중·위험회귀·bootstrap·감사결과를 확인한다.
5. 마지막 셀에서 결과 ZIP을 내려받는다.

노트북은 프로젝트 모듈을 import하지 않는다. 실제 Stage36 의존경로의 코드를
21개 실행 셀 안에 포함하고, 22개 설명 셀로 경제·수학·구현원리를 설명한다.

---

## 10. 로컬 실행

프로젝트 루트에서:

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage36_asset_implied_volatility_risk.asset_implied_volatility_risk_slsqp
```

Stage36 및 Colab 테스트:

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m pytest tests\test_stage36_asset_implied_volatility_risk.py `
            tests\test_stage36_colab_notebook.py -q
```

---

## 11. Stage36 폴더에서 무엇을 볼 것인가

권장 읽기 순서:

1. [`Stage36 README`](strategies/stage36_asset_implied_volatility_risk/README.md)
2. [`Colab 노트북`](strategies/stage36_asset_implied_volatility_risk/colab/Stage36_GVZ_OVX_Colab.ipynb)
3. [`프로젝트 통합형 메인 소스`](strategies/stage36_asset_implied_volatility_risk/asset_implied_volatility_risk_slsqp.py)
4. [`Colab 코드 셀 원본`](strategies/stage36_asset_implied_volatility_risk/colab/stage36_colab_cells.py)
5. [`경제·수학 상세 가이드`](strategies/stage36_asset_implied_volatility_risk/stage36_implementation_economic_math_guide.html)
6. [`시각화 보고서`](strategies/stage36_asset_implied_volatility_risk/stage36_gvz_ovx_report.html)
7. [`검증 JSON`](strategies/stage36_asset_implied_volatility_risk/outputs/validation_report.json)
8. [`Stage36 테스트`](tests/test_stage36_asset_implied_volatility_risk.py)
9. [`Colab 테스트`](tests/test_stage36_colab_notebook.py)

메인 소스에서 우선 확인할 함수:

- `load_asset_implied_volatility_daily()`
- `build_monthly_asset_volatility_signals()`
- `build_asset_risk_research_frame()`
- `asset_risk_predictive_regressions()`
- `_solve_weights()`
- `run_backtest()`
- `gate_decision()`
- `run_research()`

실제 소스 의존관계:

```text
Stage36
└─ Stage35 EPS·밸류·신용
   ├─ Stage20 K-ratio·RSI·ATR
   ├─ Stage13 조건부 μ·Σ·VKOSPI/VIX6
   ├─ Stage14 비용·bounds·solver 보조
   ├─ Stage30 paired block bootstrap
   ├─ Stage34 미래위험 목적변수
   ├─ Stage07 거시확률·비용률
   └─ core 자산목록·월수익·CDaR
```

Stage20은 drawdown 진단을 위해 Stage17 보조함수도 사용한다. 더 세부적인 파일별
역할은 Stage36 폴더의 README에 정리돼 있다.

---

## 12. 주요 Stage36 결과 파일

| 파일 | 내용 |
|---|---|
| `monthly_asset_volatility_signals.csv` | 월별 GVZ·OVX 순위·활성화·배수 |
| `asset_risk_predictive_regressions.csv` | 미래위험 HAC 회귀·IC |
| `performance_comparison.csv` | 전체·공통·잠금 성과 |
| `paired_block_bootstrap_vs_stage35.csv` | 2,000회 재표집 결과 |
| `stage36_gvz_ovxassetrisk_monthly.csv` | 최종 월별 비중·수익·solver 기록 |
| `validation_report.json` | 해시·인과성·제약·gate 감사 |
| `sensor_history.png` | 센서역사와 활성구간 |
| `performance_comparison.png` | 기간별 성과비교 |
| `nav_comparison.png` | NAV·drawdown |

모든 파일은
[`strategies/stage36_asset_implied_volatility_risk/outputs/`](strategies/stage36_asset_implied_volatility_risk/outputs/)
에 저장된다.

---

## 13. 저장소 구조

```text
RegimeDecisionTest/
├─ strategies/
│  ├─ core/                                  # 공통 자산·수익·CDaR
│  ├─ stage07_zero_tune_vkospi/              # 거시확률·비용
│  ├─ stage13_conditional_moments_slsqp/     # 조건부 μ·Σ, VKOSPI/VIX6
│  ├─ stage14_unconstrained_dynamic_risk_slsqp/
│  ├─ stage20_daily_technical_confidence/    # K-ratio·RSI·ATR
│  ├─ stage30_abnormal_surface_erp/          # bootstrap 보조
│  ├─ stage34_futures_basis_oi_confirmation/ # 미래위험 목적변수
│  ├─ stage35_earnings_credit_fundamentals/  # EPS·밸류·신용
│  └─ stage36_asset_implied_volatility_risk/ # 현재 README의 중심
├─ tests/                                    # 회귀·인과성·산출물 검사
├─ raw_data/                                 # 원천자료
├─ cache/                                    # 시장·OHLCV 캐시
├─ results/                                  # 공통 중간 결과
├─ artifacts/                                # 과거 보고서·노트북·번들
├─ docs/                                     # 전략 발전과 리팩터링 기록
└─ run_strategy.py                           # 이전 전략 공통 실행 도구
```

이전 Stage의 발전과 실패·보존판정은 [전략 로드맵](docs/STRATEGY_ROADMAP.md),
전체 산출물 위치는 [결과 인덱스](docs/RESULTS_INDEX.md), 업로드 노트북 비교는
[업로드 노트북 비교](docs/UPLOADED_NOTEBOOK_COMPARISON.md)에서 확인할 수 있다.

---

## 14. 검증과 한계

현재 Stage36 검증결과:

- GVZ·OVX 및 Stage35 동결파일 실행 전후 SHA-256 동일
- 무오버레이 Stage36과 Stage35 수익·비중 오차 `1e-16` 수준
- 모든 센서 신호월이 목표월보다 앞섬
- 활성화 전 배수 1, 활성 후 현재 이전 252개 이상 유효관측
- GVZ/OVX 방향성 `μ` 조정 0
- 모든 월 long-only·비중합 1·무레버리지
- 모든 SLSQP 해 성공, fallback 0
- 독립 Colab 노트북 전체 셀 실행 성공
- 12개월 블록·2,000회 bootstrap 실행

주의할 점:

- 전체표본의 Sharpe·MDD 개선이 최근구간의 우월성을 뜻하지 않는다.
- GVZ 효과는 대부분 GLD 비중 감소에 집중된다.
- USO 기본비중이 작아 OVX의 포트폴리오 영향은 제한적이다.
- 13% 예상변동성과 -16% 역사적 CDaR는 미래 손실의 보증선이 아니다.
- 세금·시장충격·괴리율·상품존속위험은 완전히 반영되지 않았다.
- 결과를 보고 센서문턱·최소비중·위험상한을 다시 맞추면 과최적화 위험이 커진다.
- 과거 시뮬레이션은 미래성과를 보장하지 않으며 투자조언이 아니다.
