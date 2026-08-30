# Stage36 — GVZ·OVX 자산별 내재변동성 위험 오버레이

Stage36은 Stage35의 기대수익·위험 엔진을 동결한 뒤, 금 옵션시장의 `GVZ`와
원유 옵션시장의 `OVX`를 각각 `GLD`와 `USO`의 **위험축에만** 연결한 월간
자산배분 연구다.

> Stage35가 거시·시장공포·기술·기업이익·밸류에이션·신용정보를 `μ35`와
> `Σ35`로 압축하는 본체라면, Stage36은 `μ35`를 그대로 두고 `Σ35`에
> GVZ·OVX 위험센서를 추가하는 마지막 계층이다.

연구기간은 2007-04~2026-07이고, 최종 결합경로의 전 기간 성과는 다음과 같다.

| 전략 | CAGR | 연변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 동결 기준 | 10.608% | 10.043% | 1.057 | -13.743% |
| **Stage36 GVZ+OVX** | **10.499%** | **9.472%** | **1.105** | **-12.407%** |

Stage36은 Stage35보다 CAGR이 0.108%p 낮지만, 연변동성은 0.571%p 낮고
Sharpe는 0.048 높으며 MDD는 1.336%p 완화됐다. 따라서 새로운 방향 알파라기보다
일부 기대수익을 포기해 위험효율을 개선한 방어적 변형으로 해석해야 한다.

---

## 1. 무엇을 바꾸고 무엇을 바꾸지 않았나

### Stage36에서 새로 추가한 것

- `GVZCLS → GLD covariance`
- `OVXCLS → USO covariance`
- 각 센서를 자기 과거에서 계산한 `causal expanding midrank`로 변환
- 현재 관측 이전 252개 유효 일간관측이 쌓인 뒤에만 활성화
- 활성화 후 variance multiplier는 `1 + causal rank`, 즉 1~2배
- GVZ-only, OVX-only, GVZ+OVX 경로를 같은 조건으로 비교
- GVZ·OVX가 VIX6 이후에도 자기자산의 미래위험을 설명하는지 HAC 회귀로 검증

### Stage35에서 그대로 유지한 것

- GDP·수출·BSI와 CPI·PPI·수입물가로 만든 네 soft-regime 확률
- 국면확률로 가중한 조건부 기대수익과 공분산
- VKOSPI·VIX6 stress/recovery 조정
- K-ratio·price RSI·volume RSI 기술 신뢰도
- ATR/NATR 공분산 위험조정
- EPS revision·earnings-yield gap·AA- 신용스프레드
- SLSQP 목적함수와 거래비용
- long-only, 비중합 100%, 무레버리지, 공매도 없음
- 연환산 예상변동성 13%, 과거 CDaR(90%) -16% guard

### 의도적으로 하지 않은 것

- GVZ로 GLD 기대수익 `μ`를 올리거나 내리지 않음
- OVX로 USO 매수·매도 방향을 결정하지 않음
- 센서 출시 전 값을 실현변동성이나 다른 지수로 backfill하지 않음
- 절대수준 문턱·분위수 문턱·최대배수 후보를 반복 탐색하지 않음
- USO 영향력을 키우기 위한 최소비중이나 별도 알파를 추가하지 않음
- SJM·CJM·HMM·로지스틱 회귀로 Stage36 상태를 새로 학습하지 않음
- 단일자산 과반금지 및 HHI 집중도 벌점을 사용하지 않음

---

## 2. 전체 입력변수 지도

| 계층 | 원 입력 | 인과적 가공 | 최종 역할 |
|---|---|---|---|
| 성장 | GDP YoY, 수출 YoY, BSI | 과거 percentile 후 평균 | 성장축 `g` |
| 물가 | CPI YoY, PPI YoY, 수입물가 YoY | 과거 percentile 후 평균 | 물가축 `π` |
| 거시국면 | `g`, `π` | 네 soft-regime 확률 | 기본 `μ`, `Σ` |
| 시장공포 | VKOSPI 수준·5일 로그변화 | expanding midrank | stress 수준·충격 |
| 옵션표면 | VIX6 parallel shift, skew, convexity | 꼬리 비대칭·21일 지속성 | stress/recovery |
| 추세 | 126일 K-ratio | `K/(1+abs(K))` | 거시 `μ`의 신뢰도 |
| 주식 기술 | 14일 price RSI·volume RSI | `(RSI-50)/50` | KODEX200 방향 확인 |
| 실현위험 | 14일 ATR/가격 | causal rank 후 `1+rank` | 각 자산 `Σ` 축 |
| 기업이익 | 12M forward EPS, 1M revision | 60M causal z·expanding slope | KODEX200 `μ` |
| 밸류에이션 | 12M forward PER, 국고채 10Y | `1/PER-y10Y` | KODEX200 장기 `μ` |
| 신용 | AA- 회사채3Y−국고채3Y, 20일 변화 | 60M rank·z | 주식 stress·`Σ` |
| Stage36 | GVZ, OVX | 252일 causal rank | GLD·USO `Σ` 축 |
| 최적화 상태 | 과거 4자산 월수익·직전 비중 | downside·CDaR·drift | 위험·비용·초기점 |

투자자 관점에서는 다음 순서로 읽으면 된다.

```text
GDP·수출·BSI → 성장축 g
CPI·PPI·수입물가 → 물가축 π
g, π → 네 soft-regime 확률
과거 확률가중 월수익 → μ_macro, Σ_macro
VKOSPI·VIX6 → stress/recovery
K-ratio·RSI → μ_macro confidence filter
ATR → 네 자산의 Σ 위험축
EPS revision·밸류에이션 → KODEX200 μ
AA- spread → 주식 stress·Σ
= Stage35의 μ35, Σ35
GVZ→GLD, OVX→USO → Σ36만 추가 조정
SLSQP → 다음 달 long-only·무레버리지 비중
```

---

## 3. 거시확률과 Stage35의 `μ35·Σ35`

성장축과 물가축은 각 원 변수의 인과적 percentile 평균이다.

```text
g = mean(rank(GDP), rank(Export), rank(BSI))
π = mean(rank(CPI), rank(PPI), rank(ImportPrice))

p_Goldilocks  = g × (1-π)
p_Overheating = g × π
p_Slowdown    = (1-g) × (1-π)
p_Stagflation = (1-g) × π
```

별도의 국면 분류모델을 학습하지 않으며 네 확률의 합은 1이다. 과거 각 월의
국면확률을 가중치로 사용해 국면별 평균과 공분산을 만든다. 특정 국면의 유효표본이
작으면 다음 credibility로 무조건부 통계에 수축한다.

```text
n_eff       = sum(p)^2 / sum(p^2)
credibility = n_eff / (n_eff + 12)
```

현재 네 국면확률로 국면별 통계를 다시 섞어 `μ_macro`와 `Σ_macro`를 만든다.
VKOSPI/VIX6 stress·recovery 회귀는 자기 R²에 해당하는 reliability로 축소하며,
주식과 원유에는 stress 계수≤0, recovery 계수≥0이라는 사전 경제부호를 둔다.

### 기술신뢰도 식

K-ratio와 RSI는 새 기대수익을 직접 더하지 않고 거시 상대전망을 얼마나 보존할지
결정한다.

```text
macro_direction_i = sign(μ_i - mean(μ))
agreement_i       = macro_direction_i × tech_i
confidence_i      = clip((1 + agreement_i) / 2, 0, 1)
filtered_μ_i      = mean(μ) + confidence_i × (μ_i - mean(μ))
```

`agreement`는 -1~1이다. `(-1, 0, +1)`을 confidence `(0, 0.5, 1)`로 옮기는
가장 단순한 대칭 선형식이 `(1+agreement)/2`다. 기술방향이 거시전망과 충돌해도
기대수익 부호를 반대로 뒤집지 않고 네 자산 평균 쪽으로만 축소한다.

confidence는 예측이 맞을 통계적 확률이 아니라 **거시 상대전망의 보존계수**다.
중립 기술신호에서 0.5를 주는 것은 수학적 필연이 아니라 “확인이 없으면 절반만
믿는다”는 고정 설계다.

### 펀더멘털과 신용

KODEX200에는 다음 두 기대수익 조정을 더한다.

```text
EPS adjustment       = max(0, expanding slope) × EPS revision z-score
Earnings-yield gap   = 1 / Forward PER - KTB 10Y yield
Valuation adjustment = max(0, expanding 12M slope) × gap z-score / 12
```

음의 회귀계수는 사후에 반대 알파로 뒤집지 않고 0으로 둔다. AA- 신용스프레드
20일 확대분의 causal rank `q_credit`은 주식 stress 조정에는 `2q_credit`,
KODEX200 공분산 축에는 `1+q_credit`로 연결된다.

---

## 4. Stage36 GVZ·OVX 오버레이

현재 센서값은 현재보다 앞선 관측만 들어 있는 정렬분포에서 평가한다. GVZ·OVX는
현재값 이전의 유효관측이 252개 미만이면 순위가 계산돼도 비활성화한다.

```text
q_GVZ,t = causal_rank(GVZ_t)
q_OVX,t = causal_rank(OVX_t)

m_GLD,t = 1 + q_GVZ,t
m_USO,t = 1 + q_OVX,t
```

목표 투자월 `t`에는 `t-1` 월말까지 알려진 마지막 센서값만 연결한다.

| 센서 | 최초 활성 목표월 | 활성 전 정책 |
|---|---|---|
| GVZ | 2009-07 | multiplier 1.0 |
| OVX | 2008-06 | multiplier 1.0 |

Stage35 공분산이 `Σ35`라면 다음 대각행렬을 만든다.

```text
D   = diag(1, 1, sqrt(m_GLD), sqrt(m_USO))
Σ36 = D @ Σ35 @ D
μ36 = μ35
```

제곱근을 쓰기 때문에 GLD variance는 정확히 `m_GLD`배, USO variance는
`m_USO`배가 된다. 다른 자산과의 covariance도 대응하는 제곱근 비율로 함께
조정돼 행렬의 일관성을 유지한다.

GVZ/OVX가 높다는 것은 해당 자산이 하락한다는 뜻이 아니다. 옵션시장이 미래
변동성 보험을 비싸게 평가하므로 같은 비중을 평소와 같은 위험으로 계산하지
않겠다는 의미다.

---

## 5. SLSQP 목적함수와 위험제약

매월 SLSQP는 다음 효용을 최대화한다.

```text
U(w) = w'μ
       - 0.5 × w'Σw
       - mean(min(R_history @ w, 0)^2)
       - transaction_cost(w, pretrade_w)
```

- `w'μ`: 다음 달 기대수익
- `0.5 w'Σw`: 현재 추정 분산 벌점
- downside semivariance: 과거 손실구간만 제곱한 추가 벌점
- transaction cost: 직전 drift 비중에서 새 목표비중으로 이동하는 비용

제약은 다음과 같다.

```text
sum(w) = 1
0 <= w_i <= 1
sqrt(12 × w'Σw) <= 13%
historical CDaR(90%) >= -16%
```

단일자산 과반금지는 없지만 비중합 1과 개별 0~1 때문에 레버리지와 공매도는
불가능하다.

### 13%와 -16%의 성격

- 13%는 이전 Stage에서 제안된 12~13% 중 넓은 쪽을 선택한 예상 연변동성
  catastrophe guard다. 백테스트로 찾은 Stage36 최적값이 아니다.
- -16%는 Stage10 이후 유지된 역사적 경로위험 비상제약이다. CDaR(90%)는 가장
  나쁜 10%의 과거 drawdown 평균이다.
- CDaR -16%는 미래 실현 MDD가 -16% 이내라는 보장이 아니다. 현재 비중을 과거
  수익경로에 적용한 조건부 검사이기 때문이다.

원래 13%는 평상시 비중을 정하지 않는 넓은 guard로 도입됐지만, 현재 저장된
Stage36 결합경로에서는 232개월 중 106개월에 변동성 제약이 구속되고 CDaR는
3개월에 구속된다. 따라서 현 Stage36에서 13%는 실질적인 위험예산으로 작동한다.
성과를 높이려고 상한을 사후 완화해서는 안 되며, 변경 필요 시 사전 고정한
12%·13%·14% 민감도와 잠금구간을 함께 보고해야 한다.

---

## 6. 월별 워크포워드 순서와 비용

1. 목표월보다 앞선 월수익만 `history`로 자른다.
2. 직전 월말까지 관측된 거시·stress·기술·펀더멘털·GVZ/OVX 신호를 읽는다.
3. Stage35의 `μ35·Σ35`를 만든다.
4. 선택한 Stage36 모드에 따라 `Σ36`을 만든다.
5. 직전 drift 비중을 초기점으로 SLSQP를 실행한다.
6. SLSQP 실패 시 동일 제약의 최소분산 문제로만 fallback한다.
7. 목표비중을 정한 뒤에만 해당 월 실현수익을 적용한다.
8. 국내 비중변화 15bp와 GLD·USO 해외 순비중변화 5bp를 차감한다.
9. 다음 달 리밸런싱 직전 drift 비중을 계산한다.

```text
pretrade_weight_i,next
  = w_i × (1 + asset_return_i) / (1 + portfolio_gross_return)
```

이 순서를 통해 당월 실현수익이 당월 비중결정에 들어가는 look-ahead를 차단한다.

---

## 7. 비교경로와 실험설계

| 경로 | 의미 |
|---|---|
| `Stage35_Frozen` | 저장된 Stage35 기준 |
| `Stage36_NoOverlayReproduction` | Stage36 코드가 Stage35를 재현하는지 검사 |
| `Stage36_GVZGoldRisk` | GVZ→GLD만 적용한 귀속실험 |
| `Stage36_OVXOilRisk` | OVX→USO만 적용한 귀속실험 |
| `Stage36_GVZ_OVXAssetRisk` | 사전 선언된 최종 결합경로 |

단일센서 경로는 결과가 가장 좋은 센서를 고르기 위한 후보탐색이 아니라 결합성과의
원인을 나누어 보기 위한 ablation이다. 최종 결합경로는 결과를 보기 전에 선언됐다.

- `full_2007_2026`: 2007-04~2026-07
- `common_2010_2026`: 두 센서가 충분히 활성화된 2010-01~2026-07
- `locked_2018_2026`: 최근 잠금 진단구간 2018-01~2026-07

---

## 8. 성과결과

### 전체구간 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 10.608% | 10.043% | 1.057 | -13.743% |
| GVZ→GLD | 10.262% | 9.511% | 1.078 | -12.407% |
| OVX→USO | 10.846% | 10.005% | 1.083 | -13.743% |
| **GVZ+OVX** | **10.499%** | **9.472%** | **1.105** | **-12.407%** |

### 공통구간 2010-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 9.653% | 9.371% | 1.033 | -13.743% |
| **GVZ+OVX** | **9.232%** | **8.719%** | **1.059** | **-12.407%** |

공통구간에서도 Sharpe와 MDD는 개선됐지만 CAGR은 0.421%p 낮아졌다.

### 잠금구간 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 13.821% | 10.861% | 1.251 | -11.984% |
| **GVZ+OVX** | **12.626%** | **10.178%** | **1.224** | **-11.931%** |

최근구간에서는 변동성과 MDD가 소폭 낮아졌지만 CAGR과 Sharpe도 낮아졌다.
Stage36이 모든 시대에 Stage35를 지배한다고 볼 수 없다.

### 12개월 paired circular block bootstrap

Stage35와 Stage36의 같은 월을 12개월 원형블록으로 2,000회 재표집했다.

| 구간 | 지표 | 개선확률 |
|---|---|---:|
| 전체 | ΔCAGR > 0 | 39.20% |
| 전체 | ΔSharpe > 0 | 97.15% |
| 전체 | ΔMDD > 0 | 95.10% |
| 공통 2010+ | ΔCAGR > 0 | 12.10% |
| 공통 2010+ | ΔSharpe > 0 | 86.85% |
| 공통 2010+ | ΔMDD > 0 | 93.30% |

위험효율 개선의 재표집 증거는 비교적 강하지만 CAGR 개선 증거는 없다.

---

## 9. 미래위험 진단

미래수익 방향이 아니라 자기자산의 다음 1개월 실현변동성, 1·3개월 최대낙폭
크기와 왼쪽꼬리를 종속변수로 사용한다. 통제변수는 자기자산 최근 1개월 수익,
21일 실현변동성, VIX6 stress, 거시 취약도다.

2010 공통구간 `FullControls` 결과:

| 센서 | 목적변수 | 표준화 beta | HAC p-value | Spearman IC |
|---|---|---:|---:|---:|
| GVZ | GLD 미래 1M 실현변동성 | +0.01955 | 0.0029 | +0.492 |
| GVZ | GLD 미래 1M MDD 크기 | +0.00733 | 0.0014 | +0.411 |
| GVZ | GLD 미래 3M MDD 크기 | +0.00931 | 0.0871 | +0.398 |
| OVX | USO 미래 1M 실현변동성 | +0.04984 | 0.0010 | +0.588 |
| OVX | USO 미래 1M MDD 크기 | +0.01281 | 0.0977 | +0.261 |
| OVX | USO 미래 3M MDD 크기 | +0.02322 | 0.0369 | +0.170 |

두 센서는 VIX6와 최근 자기자산 위험을 통제한 뒤에도 적어도 하나의 핵심
미래위험 목적변수에서 양의 추가 설명력을 보였다. 이 회귀는 경제적 역할을
검증하기 위한 진단이며 SLSQP 비중결정에는 들어가지 않는다.

---

## 10. 결과 해석

- 성과변화의 대부분은 GVZ가 GLD 위험을 높게 평가하면서 평균 GLD 비중을 약
  34.5%에서 30.0%로 낮추고 BOND로 위험예산을 옮긴 데서 왔다.
- Stage35의 USO 평균비중은 약 0.35%이고 양의 비중 월도 6개월뿐이다. OVX가
  USO 미래위험을 잘 설명해도 2010년 이후 포트폴리오 영향은 거의 없다.
- 전체구간 OVX-only 개선은 초기 몇 개의 USO 보유월에 집중돼 있다.
- 결합경로는 전 기간 CAGR 10%·Sharpe 1 기준을 유지하면서 MDD를 완화했다.
- 2018년 이후 CAGR·Sharpe 저하는 분명한 비용이다. Stage36은 Stage35의 절대
  대체재가 아니라 더 낮은 위험을 선호할 때 선택할 수 있는 변형이다.

---

## 11. Google Colab에서 실행

필요한 파일:

- [`colab/Stage36_GVZ_OVX_Colab.ipynb`](./colab/Stage36_GVZ_OVX_Colab.ipynb)
- [`colab/stage36_colab_data.zip`](./colab/stage36_colab_data.zip)

실행순서:

1. 노트북을 Google Colab에서 연다.
2. **런타임 → 모두 실행**을 누른다.
3. 업로드 창에서 `stage36_colab_data.zip` 하나만 올린다.
4. 성과표·월별 비중·HAC 회귀·2,000회 bootstrap·감사결과를 확인한다.
5. 마지막 셀에서 `stage36_colab_outputs.zip`을 내려받는다.

노트북은 로컬 `strategies` 패키지를 import하지 않는다. Stage36과 실제 실행에
필요한 Stage35·20·13·14·07·34·30·core 코드를 21개 기능별 실행 셀에 직접
포함하고, 22개 설명 셀에서 경제적 의미·수학식·인과성 경계를 설명한다.

데이터 ZIP에는 16개 데이터 파일과 SHA-256 manifest만 있고 Python 코드나
노트북은 들어 있지 않다. 해시가 다르면 실행을 중단한다.

---

## 12. 로컬에서 실행

프로젝트 루트에서:

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage36_asset_implied_volatility_risk.asset_implied_volatility_risk_slsqp
```

테스트:

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m pytest tests\test_stage36_asset_implied_volatility_risk.py `
            tests\test_stage36_colab_notebook.py -q
```

Colab 산출물 재생성:

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  strategies\stage36_asset_implied_volatility_risk\colab\build_stage36_colab_data_bundle.py

& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  strategies\stage36_asset_implied_volatility_risk\colab\build_stage36_colab_notebook.py
```

---

## 13. 전체 구현은 Stage36 폴더를 함께 참고할 것

README나 노트북만으로 계산의 모든 세부사항을 감사하려면 부족할 수 있다.
**전체 구현을 분석하거나 수정할 때는 반드시 다음 Stage36 폴더를 함께 참고한다.**

```text
strategies/stage36_asset_implied_volatility_risk/
```

권장 탐색순서:

1. [`README.md`](./README.md)
   - 전략 목적·경제적 경계·성과·실행법을 파악한다.
2. [`colab/Stage36_GVZ_OVX_Colab.ipynb`](./colab/Stage36_GVZ_OVX_Colab.ipynb)
   - 데이터가 입력변수와 비중으로 변하는 전체 순서를 설명과 함께 실행한다.
3. [`asset_implied_volatility_risk_slsqp.py`](./asset_implied_volatility_risk_slsqp.py)
   - 프로젝트 통합형 기준 구현이다. 센서·월매핑·HAC 회귀·SLSQP·gate·보고서
     생성을 확인한다.
4. [`colab/stage36_colab_cells.py`](./colab/stage36_colab_cells.py)
   - 독립 Colab 노트북에 삽입되는 전체 실행코드의 검토 가능한 원본이다.
5. [`stage36_implementation_economic_math_guide.html`](./stage36_implementation_economic_math_guide.html)
   - 경제적 가설, 수학, 함수 호출관계와 소스 탐색순서를 가장 자세히 설명한다.
6. [`stage36_gvz_ovx_report.html`](./stage36_gvz_ovx_report.html)
   - 성과표·센서역사·NAV·해석을 시각적으로 확인한다.
7. [`outputs/validation_report.json`](./outputs/validation_report.json)
   - 원본·Stage35 해시, 활성일, gate, 인과성, solver 감사를 확인한다.
8. [`../../tests/test_stage36_asset_implied_volatility_risk.py`](../../tests/test_stage36_asset_implied_volatility_risk.py)
   - 소스불변·252일 이력·인과성·μ 불변·제약·성과를 회귀검증한다.
9. [`../../tests/test_stage36_colab_notebook.py`](../../tests/test_stage36_colab_notebook.py)
   - 노트북 자기완결성, 데이터 전용 ZIP, 해시, 설명과 감사코드를 검증한다.

### Stage36 메인 소스의 핵심 함수

| 함수 | 확인할 내용 |
|---|---|
| `load_asset_implied_volatility_daily()` | FRED CSV 정리, 인과순위, 사전관측수 |
| `build_monthly_asset_volatility_signals()` | 직전 월말 매핑, 활성화, multiplier |
| `build_asset_risk_research_frame()` | 미래위험 진단 프레임 |
| `asset_risk_predictive_regressions()` | VIX6·최근위험 통제 HAC 회귀 |
| `_solve_weights()` | Stage35 재현, `DΣD`, 목적함수·제약 |
| `run_backtest()` | 월별 워크포워드, 비용, drift 비중 |
| `gate_decision()` | 성과·위험정보·bootstrap 승격조건 |
| `run_research()` | 네 경로 실행, 감사, 파일·차트·HTML 저장 |

### 실제 의존경로

```text
stage36_asset_implied_volatility_risk/
└─ asset_implied_volatility_risk_slsqp.py
   └─ stage35_earnings_credit_fundamentals/
      ├─ stage20_daily_technical_confidence/
      │  ├─ stage13_conditional_moments_slsqp/
      │  ├─ stage14_unconstrained_dynamic_risk_slsqp/
      │  └─ stage17_dynamic_risk_shape/        # drawdown 진단 보조
      ├─ stage13_conditional_moments_slsqp/
      ├─ stage14_unconstrained_dynamic_risk_slsqp/
      ├─ stage30_abnormal_surface_erp/          # bootstrap 보조
      ├─ stage34_futures_basis_oi_confirmation/ # 미래위험 목적변수
      ├─ stage07_zero_tune_vkospi/              # 거시확률·비용률
      └─ core/regime_research.py                # 자산·수익·CDaR
```

Stage14도 Stage13을 사용하고 Stage13·14는 Stage07과 core에 의존한다. 위 트리는
소스의 실제 import 관계를 요약한 것이며, Colab 노트북은 이 가운데 Stage36
실행에 실제로 필요한 함수만 복사해 자기완결형으로 구성한다.

---

## 14. 주요 산출물

| 파일 | 용도 |
|---|---|
| `outputs/monthly_asset_volatility_signals.csv` | 월별 센서값·순위·활성화·배수 |
| `outputs/normalized_gvz_ovx_daily.csv` | 정규화된 일간 센서 이력 |
| `outputs/asset_risk_predictive_regressions.csv` | 미래위험 HAC 회귀·IC |
| `outputs/monthly_asset_risk_research_frame.csv` | 회귀 입력과 위험 목적변수 |
| `outputs/performance_comparison.csv` | 전 기간·공통·잠금 성과 |
| `outputs/paired_block_bootstrap_vs_stage35.csv` | 2,000회 paired bootstrap |
| `outputs/stage36_nooverlayreproduction_monthly.csv` | Stage35 재현경로 |
| `outputs/stage36_gvzgoldrisk_monthly.csv` | GVZ-only 월별 기록 |
| `outputs/stage36_ovxoilrisk_monthly.csv` | OVX-only 월별 기록 |
| `outputs/stage36_gvz_ovxassetrisk_monthly.csv` | 최종 결합경로 월별 전체 기록 |
| `outputs/validation_report.json` | 설정·gate·해시·인과성·solver 감사 |
| `outputs/sensor_history.png` | GVZ·OVX 역사와 활성구간 |
| `outputs/performance_comparison.png` | 기간별 성과 비교 |
| `outputs/nav_comparison.png` | NAV·drawdown 비교 |

---

## 15. 검증상태

- 원본 GVZ·OVX 및 Stage35 동결파일 실행 전후 SHA-256 동일
- Stage36 무오버레이와 Stage35 수익·비중 최대오차 `1e-16` 수준
- 모든 센서 신호월이 목표월보다 앞서고 직전 월말 이후 신호 사용 없음
- 활성화 전 multiplier 1.0, 활성월은 현재 이전 252개 이상 유효관측
- variance multiplier 1~2, GVZ/OVX 방향성 `μ` 조정 0
- 모든 월 long-only·비중합 1·무레버리지
- 모든 후보 SLSQP 해 성공, fallback 0
- Colab 데이터 ZIP 16개 파일 해시 검증
- 독립 노트북 21개 코드 셀 전체 실행 성공
- paired circular block bootstrap 2,000회 실행

---

## 16. 한계와 사용상 주의

- 전체표본에서 위험효율은 개선됐지만 최근 잠금구간의 CAGR·Sharpe는 낮다.
- GVZ 영향은 실질적으로 GLD 위험예산 감소에 집중돼 있다.
- OVX는 USO 기본비중이 매우 작아 포트폴리오 영향이 제한적이다.
- 조건부 공분산과 기대수익 추정오차는 실거래에서 더 커질 수 있다.
- 비용은 반영했지만 세금·시장충격·괴리율·상품존속위험을 완전히 반영하지 않는다.
- 13% 예상변동성과 -16% 역사적 CDaR는 미래 위험의 보증선이 아니다.
- 결과를 보고 센서 문턱·최소비중·위험상한을 맞추면 과최적화 위험이 커진다.
- 이 결과는 연구용 역사적 시뮬레이션이며 투자조언이나 미래성과 보장이 아니다.
