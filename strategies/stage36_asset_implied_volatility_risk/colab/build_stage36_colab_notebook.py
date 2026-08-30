from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stage36_colab_cells.py"
OUTPUT = HERE / "Stage36_GVZ_OVX_Colab.ipynb"


ORIENTATION_GUIDE = r"""
## 먼저 잡아야 할 핵심: Stage35가 본체이고 Stage36은 위험센서다

Stage36 코드부터 읽으면 `μ35`와 `Σ35`가 갑자기 만들어지는 것처럼 보일 수 있습니다. 실제 계보는 다음과 같습니다.

> 거시경제가 **어떤 날씨인지** 판단하고 → VKOSPI·VIX6가 **폭풍이 얼마나 센지** 보고 → 가격·거래량이 **그 전망에 동의하는지** 확인하고 → 기업이익·밸류에이션이 **주식이 살 만한지** 검사하고 → 신용시장이 **금융시스템에 금이 가는지** 확인합니다. 이 정보가 Stage35의 기대수익 `μ35`와 위험행렬 `Σ35`로 압축됩니다. Stage36은 `μ35`는 그대로 두고 `Σ35`의 금·원유 축에 GVZ·OVX만 추가합니다.

### 전체 입력변수 지도

| 계층 | 원 입력 | 인과적 가공 | 최종 역할 |
|---|---|---|---|
| 성장 | GDP YoY, 수출 YoY, BSI | 과거만 이용한 percentile 후 평균 | 성장축 $g_t$ |
| 물가 | CPI YoY, PPI YoY, 수입물가 YoY | 과거만 이용한 percentile 후 평균 | 물가축 $\pi_t$ |
| 거시국면 | $g_t,\pi_t$ | 네 개 soft-regime 확률 | 기본 `μ`, 기본 `Σ` |
| 시장 공포 | VKOSPI 수준, 5일 로그변화 | expanding midrank | stress 수준·충격 |
| 옵션 표면 | VIX6 parallel shift, put/call skew, downside/upside convexity | 꼬리 비대칭·21일 지속성 | stress/recovery 보정 |
| 가격 추세 | 126일 K-ratio | $K/(1+\lvert K\rvert)$ | 거시 `μ`의 신뢰도 |
| 주식 확인 | 14일 price RSI, volume RSI | $(RSI-50)/50$ | KODEX200 방향 확인 |
| 실현 위험 | 14일 ATR/가격 | causal rank 후 $1+rank$ | 각 자산 `Σ` 축 확대 |
| 기업이익 | 12M forward EPS, 1M revision | 60개월 causal z-score·expanding slope | KODEX200 `μ` |
| 밸류에이션 | 12M forward PER, 국고채 10Y | $1/PER-y_{10Y}$ | KODEX200 장기 `μ` |
| 신용 | AA- 회사채 3Y−국고채 3Y, 20일 변화 | 60개월 rank·z-score | 주식 stress와 `Σ` |
| Stage36 | GVZ, OVX | 직전 월말 값의 252일 causal rank | 각각 GLD·USO `Σ` 축 |
| 최적화 상태 | 과거 4자산 월수익, 직전 보유비중 | downside·CDaR·drift | 위험제약·거래비용·초기점 |

### 코드가 최종적으로 만드는 두 장의 성적표

- `μ35`: 다음 한 달에 각 자산이 얼마나 유리할지 나타내는 월 기대수익 벡터입니다. 거시 조건부 수익을 기술신호로 덜 또는 더 신뢰하고, VKOSPI/VIX6 stress·recovery와 KODEX200의 EPS·밸류에이션을 반영합니다.
- `Σ35`: 각 자산의 흔들림과 동행관계를 나타내는 공분산 행렬입니다. 거시 조건부 공분산을 표본수에 따라 수축하고, ATR과 신용위험으로 해당 축을 확대합니다.

Stage36의 경계는 명확합니다.

\[
\mu_{36}=\mu_{35},\qquad
\Sigma_{36}=D_{GVZ,OVX}\Sigma_{35}D_{GVZ,OVX}
\]

즉 GVZ가 높다고 금의 기대수익을 깎거나, OVX가 높다고 원유를 매도하라는 방향 신호를 만들지 않습니다. 옵션시장이 비싸게 평가한 **미래 위험**만 포트폴리오 위험계산에 반영합니다.

### 이 노트북을 읽는 순서

1. 3~7절에서 인과적 변환 → 거시확률 → VKOSPI/VIX6 → 조건부 `μ·Σ`를 봅니다.
2. 8절에서 K-ratio·RSI가 `μ`를 뒤집지 않고 신뢰도만 조절하는 부분을 봅니다.
3. 9절에서 EPS·밸류에이션·신용이 KODEX200에 연결되는 정확한 식을 봅니다.
4. 10~11절에서 Stage36의 GVZ/OVX `DΣD`와 SLSQP 조립 순서를 봅니다.
5. 12~18절에서 월별 실행·성과·미래위험 진단·감사 결과를 확인합니다.

이 구현은 HMM·SJM·CJM·로지스틱 회귀로 국면을 학습하지 않습니다. 거시 여섯 변수로 네 확률을 직접 계산하며, 하이퍼파라미터 탐색으로 가장 좋아 보이는 문턱을 고르지도 않습니다.
"""


DESCRIPTIONS = {
    "01_environment": """
## 1. 실행환경과 고정 설계값

이 셀은 Stage36 전체에서 공유하는 자산 순서와 경제적·수치적 상수를 선언합니다.

- 자산: `KODEX200`, `BOND`, `GLD`, `USO`
- 무레버리지·공매도 금지: 각 비중 0~1, 합계 1
- 위험 guard: 연환산 변동성 13%, 과거 CDaR(90%) -16%
- 거래비용: 전체 비중변화 15bp, 해외 순비중 변화 추가 5bp
- GVZ/OVX 최소 이력: 현재 값을 제외한 252개 일간관측
- SLSQP의 300회와 `ftol=1e-9`는 경제적 파라미터가 아니라 수치해석 설정입니다.

`configure_data_root`는 압축을 푼 데이터 폴더만 가리키며, 로컬 `strategies` 패키지를 참조하지 않습니다. `validate_data_bundle`은 ZIP에 기록된 파일 크기와 SHA-256을 전부 확인합니다.
""",
    "02_upload_and_extract": """
## 2. 데이터 ZIP 하나 업로드하고 검증

Colab에서는 실행 시 표시되는 업로드 창에 `stage36_colab_data.zip` 하나만 올리면 됩니다. ZIP에는 코드가 없고 시장·거시·펀더멘털·GVZ/OVX 원천과 고정된 VIX6 6요인 일간 결과만 있습니다.

로컬 QA에서는 `STAGE36_DATA_ZIP` 환경변수를 사용합니다. 압축을 푼 뒤 16개 파일의 SHA-256이 manifest와 다르면 즉시 중단하므로, 깨진 업로드나 다른 버전의 데이터를 조용히 사용하는 일을 막습니다.
""",
    "03_causal_transforms": r"""
## 3. 인과적 변환: expanding percentile·midrank·z-score

현재 관측치는 현재까지 알려진 역사 안에서만 평가합니다. 미래 전체표본 평균이나 표준편차는 쓰지 않습니다.

\[
q_t=\frac{L_t+0.5(E_t+1)}{n_t+1}
\]

`causal_expanding_midrank`는 이 empirical CDF를 정렬목록으로 계산합니다. GVZ/OVX는 현재 값 이전의 유효관측이 252개 미만이면 순위가 있더라도 비활성화됩니다. 펀더멘털 z-score는 현재월을 제외한 최소 60개월의 평균과 표준편차를 사용합니다.
""",
    "04_market_returns": """
## 4. 네 자산의 원화 월수익률

전략 수익률은 월 첫 거래일 시가에서 다음 달 첫 거래일 시가까지 계산합니다. KODEX200은 2009년 3월까지 `compass.db`의 KOSPI200 프록시를 실제 ETF에 레벨 정합해 연결합니다. BOND는 KRX 채권 총수익지수, GLD와 USO는 USD 가격에 USDKRW를 곱해 한국 투자자의 원화 수익률로 바꿉니다.

이 셀은 인터넷에서 가격을 다시 받지 않습니다. 따라서 Colab 실행일과 무관하게 동일한 데이터 스냅샷을 사용합니다.
""",
    "05_macro_probabilities": r"""
## 5. 무학습 거시 국면 확률

GDP·수출·BSI의 인과적 백분위 평균을 성장확률 `g`, CPI·PPI·수입물가 평균을 물가확률 `π`로 둡니다.

\[
p_G=g(1-\pi),\quad p_O=g\pi,\quad
p_S=(1-g)(1-\pi),\quad p_{Stag}=(1-g)\pi
\]

하드 분류, 로지스틱 회귀, SJM/CJM 학습은 없습니다. 네 연속확률은 합이 1이며 과거 수익의 국면별 가중치가 됩니다. GDP와 수출에는 원 코드와 같은 한 달 공표 지연, 물가에는 두 달 공표 지연을 적용합니다.
""",
    "06_vkospi_vix6_stress": """
## 6. VKOSPI·VIX6 연속 스트레스와 회복

네 블록을 동일가중으로 결합합니다.

1. VKOSPI 수준
2. VKOSPI 5일 변화와 VIX6 parallel shift
3. put/call skew 및 downside/upside convexity의 왼쪽 꼬리 비대칭
4. 앞선 세 블록의 21거래일 지속성

공포 상승은 즉시 반영하고, 하락 시에는 현재 stress와 5일 평균 중 큰 값을 써서 하루짜리 안도 랠리가 완전한 회복으로 해석되는 것을 줄입니다. 목표월에는 직전 월말까지의 마지막 일간값만 연결합니다.
""",
    "07_conditional_moments": r"""
## 7. soft-regime 조건부 평균과 공분산

과거 각 월의 네 국면확률을 가중치로 사용해 국면별 평균과 공분산을 추정합니다. 유효표본이 작으면 무조건부 모멘트로 수축합니다.

\[
n_{eff}=\frac{(\sum p_i)^2}{\sum p_i^2},\qquad
c=\frac{n_{eff}}{n_{eff}+12}
\]

VKOSPI/VIX6 stress와 recovery의 국면별 OLS 기울기는 자기 R²로 신뢰도를 낮춥니다. 주식과 원유는 스트레스 기울기≤0, 회복 기울기≥0이라는 고정 경제부호를 둡니다. 음의 고유값은 수치오차 수준의 바닥만 적용해 PSD 행렬로 만듭니다.
""",
    "08_daily_technical_inputs": """
## 8. K-ratio·RSI·ATR 기술 입력

`K_RATIO_DAYS=126`은 약 6개월 로그가격 추세의 기울기 안정성을, Wilder 14일 RSI와 ATR은 방향 강도와 현재 위험을 측정합니다.

- 모든 자산: K-ratio와 ATR/NATR의 인과적 백분위
- KODEX200: price RSI와 거래량 RSI를 추가
- 기술 방향과 거시 상대방향이 일치할수록 거시 기대수익의 횡단면 차이를 더 신뢰
- ATR 순위는 각 자산 공분산 축을 `1+rank`로 확대

해외자산 OHLC도 USDKRW로 원화 환산하므로 월수익률과 기술지표의 통화기준이 같습니다.
""",
    "09_fundamental_inputs": """
## 9. EPS·밸류에이션·신용 입력

Stage35의 기준 기대수익을 재현합니다.

- KODEX200 μ: 1개월 선행 EPS revision과 `1/PER−국고채10년` 밸류에이션 갭
- 신용위험: `AA-회사채3년−국고채3년`의 20거래일 확대
- 인과 보정: 최소 60개월 과거만 이용한 비음수 단변량 기울기

신용스프레드 확대는 주식 stress μ를 확인하고 주식 공분산 축을 키웁니다. 실패한 신호를 사후에 부호반전하지 않습니다.
""",
    "10_gvz_ovx_overlay": r"""
## 10. Stage36의 신규 입력: GVZ→GLD, OVX→USO

FRED 형식 CSV에서 양수 지수값만 읽고 현재 이전 252개 유효관측이 쌓인 뒤 활성화합니다. 목표월 `t`에는 `t-1` 월말까지의 마지막 값만 사용합니다.

\[
m_G=1+q_{GVZ},\quad m_O=1+q_{OVX}
\]

비활성 기간의 배수는 1입니다. 출시 전 값을 실현변동성이나 다른 지수로 backfill하지 않습니다.
""",
    "11_slsqp_optimizer": r"""
## 11. DΣD 오버레이와 SLSQP

Stage36은 기대수익을 바꾸지 않고 공분산만 바꿉니다.

\[
D=diag(1,1,\sqrt{m_G},\sqrt{m_O}),\qquad
\Sigma_{36}=D\Sigma_{35}D,\qquad \mu_{36}=\mu_{35}
\]

최적화 목적은 다음 월별 효용 최대화입니다.

\[
w'\mu-\tfrac12w'\Sigma w-
E[\min(R_{hist}w,0)^2]-C(w,w^{pre})
\]

제약은 비중합 1, 각 비중 0~1, 연변동성≤13%, 역사적 CDaR(90%)≥-16%입니다. SLSQP가 실패할 때만 같은 제약의 최소분산 fallback을 사용합니다.
""",
    "12_monthly_backtest": r"""
## 12. 월별 비용 차감 백테스트

월 `t`보다 앞선 수익만 추정에 넣고, SLSQP 비중에 당월 실현수익을 적용합니다. 전체 비중변화의 15bp와 GLD·USO 순비중 변화의 추가 5bp를 차감합니다.

월말의 다음 사전비중은

\[
w^{pre}_{i,t+1}=\frac{w_{i,t}(1+r_{i,t})}{1+w_t'r_t}
\]

로 계산하므로 가격변동으로 자연스럽게 떠밀린 비중과 새 목표비중 사이의 실제 매매량을 비용에 반영합니다.
""",
    "13_performance_metrics": """
## 13. 성과지표와 paired circular block bootstrap

CAGR, 월수익 표준편차×√12, 무위험수익률 0 기준 Sharpe, 월말 NAV MDD를 계산합니다. 시계열 군집을 보존하기 위해 Stage35와 Stage36의 동일 월을 12개월 원형 블록으로 묶어 2,000회 재표집합니다. 이는 위험효율 개선의 불확실성을 보여주는 진단이지 새로운 최적화 파라미터가 아닙니다.
""",
    "14_future_risk_diagnostics": """
## 14. GVZ/OVX의 미래위험 설명력

GVZ와 OVX가 자기자산의 향후 1개월 실현변동성, 1·3개월 최대낙폭 크기, 왼쪽꼬리를 설명하는지 HAC 회귀로 검사합니다. 통제변수는 자기자산 최근 1개월 수익, 21일 실현변동성, VIX6 stress, 거시 취약도입니다.

이 미래 목적변수는 검증에만 쓰이며 월별 비중 산정에는 들어가지 않습니다.
""",
    "15_research_orchestration": """
## 15. 전체 조립과 네 경로 비교

지금까지 정의한 함수를 다음 순서로 연결합니다.

`월수익 → 거시확률 → VKOSPI/VIX6 → 기술신호 → EPS·신용 → GVZ/OVX → SLSQP`

Stage35, GVZ-only, OVX-only, GVZ+OVX를 동일 비용·동일 제약으로 실행합니다. 결과 CSV, 월별 비중, 위험회귀, bootstrap, solver audit를 `colab_outputs`에 저장합니다. 코드에 포함된 인과성·무레버리지·μ 불변 검사가 하나라도 실패하면 최종 감사 셀에서 오류가 납니다.
""",
    "16_results_and_charts": """
## 16. 결과 표시와 다운로드 묶음

성과표, 로그 NAV, 월말 drawdown, 인과성 및 solver 검사를 표시합니다. 마지막 함수는 모든 CSV·JSON 결과를 하나의 ZIP으로 묶습니다. Colab에서는 선택적으로 이 결과 ZIP을 바로 내려받을 수 있습니다.
""",
}


DETAILED_NOTES = {
    "03_causal_transforms": r"""
### 왜 이 변환이 필요한가

GDP 3%, BSI 95, GVZ 24처럼 단위가 다른 원값을 그대로 평균하면 숫자가 큰 변수가 결과를 지배합니다. percentile은 각 값을 “자기 역사에서 어느 위치인가”라는 공통 0~1 척도로 바꿉니다. 중요한 점은 전체 기간이 아니라 **그 시점까지 들어온 값만** 정렬목록에 넣는다는 것입니다.

예를 들어 현재 값보다 작은 과거 관측이 `L`, 같은 값까지 포함한 동률 수가 `E`, 현재를 포함한 관측수가 `n+1`이면 midrank를 사용합니다. 동일한 값이 반복돼도 먼저 나온 관측과 나중 관측을 자의적으로 다르게 취급하지 않기 위해서입니다.

`causal_zscore`도 같은 원칙을 따릅니다. 현재월 값은 현재월 이전 평균·표준편차와 비교합니다. 최소 60개월이 쌓이기 전에는 펀더멘털 신호를 활성화하지 않습니다. GVZ·OVX에는 더 촘촘한 일간자료를 쓰므로 현재일을 제외한 252개 유효관측을 요구합니다.

**코드에서 확인할 부분**

- `shift(1)`: 현재값이 자기 기준통계에 들어가는 것을 차단합니다.
- `prior_count >= 252`: 센서 출시 초기의 불안정한 순위를 중립 배수 1로 둡니다.
- expanding 방식: 2020년 값을 평가하면서 2025년 분포를 보는 전 표본 look-ahead를 막습니다.
""",
    "04_market_returns": r"""
### 왜 월 첫 시가에서 다음 달 첫 시가인가

전략은 직전 월말까지 알려진 신호를 이용해 다음 달 비중을 정합니다. 그러므로 신호를 관찰할 수 있는 시점과 체결 수익률의 시작점을 분리해야 합니다. 목표월 첫 거래일 시가에서 진입하고 다음 달 첫 거래일 시가까지 보유하는 정의는 “월말 정보를 본 뒤 다음 거래가능 시점에 체결한다”는 시간순서를 구현합니다.

GLD와 USO는 달러가격만 보면 한국 투자자의 실제 손익과 다릅니다. 코드가 일간 OHLC와 월수익률 모두에 USDKRW를 곱하는 이유는 환율 효과를 빠뜨리지 않고 기술신호와 백테스트 수익률의 통화기준도 일치시키기 위해서입니다.

KODEX200 ETF가 충분히 길지 않은 초기구간은 `compass.db`의 KOSPI200 프록시를 실제 ETF 첫 관측 레벨에 맞춰 연결합니다. 이 조정은 수익률 흐름을 이어 주기 위한 레벨 정합이며 미래 수익을 채우는 backfill과는 다릅니다.
""",
    "05_macro_probabilities": r"""
### 경제 질문을 두 축으로 압축한다

성장축은 “경제 엔진이 얼마나 잘 도는가”를 묻습니다. GDP는 경제 전체, 수출은 한국경제의 대외수요, BSI는 기업 현장의 전망을 보완합니다. 세 percentile이 각각 0.8, 0.7, 0.6이라면

\[
g_t=(0.8+0.7+0.6)/3=0.7
\]

로 읽습니다. 물가축은 “엔진이 과열됐는가”를 묻습니다. CPI는 소비자단계, PPI는 생산자단계, 수입물가는 해외 원가압력을 포착합니다.

### 하드 라벨 대신 혼합상태를 쓰는 이유

예를 들어 $g=0.7$, $\pi=0.8$이면 네 확률은 다음과 같습니다.

| 국면 | 계산 | 확률 |
|---|---:|---:|
| Goldilocks | $0.7(1-0.8)$ | 14% |
| Overheating | $0.7\times0.8$ | 56% |
| Slowdown | $(1-0.7)(1-0.8)$ | 6% |
| Stagflation | $(1-0.7)0.8$ | 24% |

현실 경제는 한 달 사이에 골디락스에서 스태그플레이션으로 완전히 점프하지 않습니다. “과열 성격이 가장 강하지만 스태그플레이션 위험도 일부 있다”처럼 확률을 나누면 경계 부근의 작은 데이터변화가 포트폴리오를 통째로 뒤집는 일을 줄일 수 있습니다.

이 확률은 예측모델이 출력한 class probability가 아닙니다. 성장·물가 percentile의 곱으로 직접 계산되므로 학습표본 부족으로 로지스틱 계수가 흔들리는 문제도 없습니다.
""",
    "06_vkospi_vix6_stress": r"""
### 거시데이터보다 빠른 시장의 공포를 읽는다

GDP가 양호해도 금융시장은 며칠 만에 무너질 수 있습니다. VKOSPI 수준은 공포가 구조적으로 높은지를, 5일 로그변화는 공포가 갑자기 폭발했는지를 구분합니다. VKOSPI 35가 한 달 지속된 경우와 18에서 35로 급등한 경우가 다른 이유입니다.

VIX6는 공포의 **높이뿐 아니라 모양**을 봅니다.

- `parallel_shift`: 옵션 전반의 내재변동성이 함께 이동했는지
- `put_skew`, `downside_convexity`: 하락보험과 극단 하락꼬리가 비싸졌는지
- `call_skew`, `upside_convexity`: 상승꼬리와 비교했을 때 하락꼬리 비대칭이 얼마나 큰지
- 21일 persistence: 하루짜리 충격인지 약 한 달 이어진 상태인지

`stress_score=max(stress_raw, 최근 5일 평균)`은 공포 급등은 바로 올리되, 하루 하락만으로 위험이 완전히 사라졌다고 보지 않는 간단한 히스테리시스입니다. `recovery_score`는 5일 평균보다 현재 stress가 낮아진 정도를 역사적 순위로 바꿉니다.

**시간축 확인:** 목표월 $t$에는 $t-1$ 월말까지 관찰된 마지막 일간값만 사용합니다. 이 셀의 미래 데이터는 월별 신호에 연결되지 않습니다.
""",
    "07_conditional_moments": r"""
### 네 국면확률이 기본 `μ`와 `Σ`가 되는 과정

과거 월 $i$가 Goldilocks 확률 70%였다면 그 월 수익은 Goldilocks 통계에 0.7만큼 기여합니다. 이렇게 각 국면의 가중평균수익 $\tilde\mu_k$와 가중공분산 $\tilde\Sigma_k$를 계산합니다. 질문을 사람말로 바꾸면 “과거에 이 국면의 냄새가 강했던 달에 네 자산이 평균적으로 어떻게 움직였나?”입니다.

### 표본이 적으면 전체시장 통계로 수축한다

확률가중치가 몇 달에 몰리면 실제 정보량은 달력상의 월수보다 작습니다.

\[
n_{eff,k}=\frac{(\sum_i p_{i,k})^2}{\sum_i p_{i,k}^2},\qquad
c_k=\frac{n_{eff,k}}{n_{eff,k}+12}
\]

따라서

\[
\mu_k=c_k\tilde\mu_k+(1-c_k)\mu_{all},\qquad
\Sigma_k=c_k\tilde\Sigma_k+(1-c_k)\Sigma_{all}
\]

이 됩니다. 리뷰 2개의 평점 5.0을 리뷰 5,000개의 평점 4.8만큼 믿지 않는 것과 같습니다. `12`는 1년의 무조건부 prior를 뜻하며 후보탐색으로 고른 값이 아닙니다.

현재 거시확률로 네 국면 통계를 다시 섞어 $\mu_{macro,t}=\sum_k p_{t,k}\mu_k$를 만듭니다. stress/recovery 회귀기울기는 자기 $R^2$에 해당하는 reliability로 축소합니다. 주식·원유에는 stress 계수≤0, recovery 계수≥0이라는 사전 경제부호를 두어 표본우연이 “공포 급등은 주식에 호재” 같은 역방향 규칙을 만드는 것을 막습니다.

공분산도 현재 stress가 높을수록 국면별 high-stress 공분산 쪽으로 이동합니다. 마지막 `nearest_psd`는 음의 고유값 때문에 $w'\Sigma w$가 음수가 되는 수치오류를 차단합니다.
""",
    "08_daily_technical_inputs": r"""
### K-ratio: 추세의 방향뿐 아니라 안정성을 본다

126일 로그가격에 직선을 적합하고 기울기를 그 기울기의 표준오차로 나눈 뒤 기간규모를 조정합니다. 같은 상승률이라도 들쭉날쭉한 가격보다 꾸준한 상승경로의 K-ratio가 큽니다. 극단값이 전략을 장악하지 않도록

\[
Kscore=\frac{K}{1+|K|},\qquad -1<Kscore<1
\]

로 압축합니다.

KODEX200에는 Wilder 14일 price RSI와 volume RSI도 추가합니다. 코드의 정확한 재스케일은 둘 다 $(RSI-50)/50$이며, KODEX200의 최종 기술방향은 `k_score`, `price_strength`, `volume_strength`의 단순평균입니다. 다른 세 자산은 K-score만 사용합니다. 가격은 방향을, 거래량은 그 움직임의 참여도를 확인한다는 해석입니다.

### 기술신호는 새 기대수익을 더하지 않고 거시전망의 확신을 조절한다

\[
confidence_i=clip\left[\frac12\{1+sign(\mu_i-\bar\mu)tech_i\},0,1\right]
\]

\[
\mu_{filtered,i}=\bar\mu+confidence_i(\mu_i-\bar\mu)
\]

거시가 평균보다 좋게 보는 자산의 가격추세도 좋으면 상대 기대수익을 더 신뢰합니다. 반대로 가격이 거시전망과 충돌하면 기대수익 부호를 뒤집지 않고 횡단면 평균 쪽으로 당겨 **덜 자신 있게** 만듭니다.

ATR은 방향이 아니라 위험입니다. `NATR=ATR/가격`의 causal rank가 0.9라면 variance scale은 1.9입니다. $D=diag(\sqrt{1+rank_i})$로 $D\Sigma D$를 계산하므로 해당 자산의 분산은 1.9배, 다른 자산과의 공분산은 기하평균 규모로 일관되게 조정됩니다.
""",
    "09_fundamental_inputs": r"""
### EPS revision: 기업이익 전망이 좋아지는가

Forward EPS 절대수준보다 애널리스트의 1개월 revision을 사용합니다. 전망치가 100→105→112라면 이익기대가 계속 상향되는 상태입니다. 현재 revision은 과거 60개월 이상으로 만든 causal z-score로 표준화하고, 과거 월수익과의 expanding 단변량 slope로 월 기대수익 단위에 맞춥니다.

\[
\beta^*=\max(0,\hat\beta),\qquad
\Delta\mu_{EPS}=\beta^* z_{EPS}
\]

과거에 도움이 없었으면 계수를 0으로 두며, 음의 부호로 뒤집어 새 규칙을 만들지 않습니다.

### 밸류에이션: PER만 보지 않고 채권수익률과 비교한다

\[
EYG=\frac{1}{Forward\ PER}-y_{10Y}
\]

PER 10배의 earnings yield는 10%입니다. 국고채 10년이 3%면 gap은 7%지만 국채가 8%면 2%뿐입니다. 같은 PER라도 채권 대안수익률이 높으면 주식의 상대매력이 낮다는 뜻입니다. EYG는 완전히 관측된 과거 12개월 선행수익만 이용해 slope를 추정하고 월 단위로 12분의 1하여 KODEX200 `μ`에 더합니다.

### 신용스프레드: 금융시장의 혈압

\[
CreditSpread=y_{AA-,3Y}-y_{Gov,3Y}
\]

20거래일 확대분의 60개월 causal rank를 $q_C$라 하면, 실제 코드는 주식 stress 조정에 `2q_C`를 곱하고 KODEX200 공분산 축에는 $1+q_C$를 적용합니다. 즉 신용악화는 새로운 독립 매도 알파가 아니라 기존 공포신호를 확인하고 주식 위험을 높이는 역할입니다.

이 셀에는 원문에서 소스 확인이 필요하다고 남겨졌던 정확한 산술식까지 그대로 구현되어 있습니다. `winsorization=False`, `parameter_grid=False`도 함께 기록해 사후 문턱탐색을 하지 않았음을 감사할 수 있습니다.
""",
    "10_gvz_ovx_overlay": r"""
### ATR과 GVZ/OVX는 무엇이 다른가

- ATR: 실제 가격에서 이미 나타난 최근 변동성으로 네 자산 모두를 측정합니다.
- GVZ/OVX: 옵션시장에서 금·원유의 앞으로의 변동성에 붙은 보험가격을 측정합니다.

따라서 Stage36의 질문은 “GVZ가 높으면 금이 떨어질까?”가 아니라 “옵션시장이 금 위험을 비싸게 보고 있을 때 동일한 금 비중을 평소와 같은 위험으로 계산해도 되는가?”입니다.

GVZ rank가 0.8이면 GLD variance multiplier는 1.8, OVX rank가 0.3이면 USO multiplier는 1.3입니다. 센서가 비활성인 초기구간에는 무조건 1입니다. 순위문턱을 여러 개 시험하지 않고 전체 0~1 순위를 연속적으로 쓰므로 경계 바로 양쪽의 관측이 전혀 다른 의사결정을 만드는 것을 피합니다.

**중요한 제한:** 이 셀은 신호만 만듭니다. GLD·USO의 기대수익 조정 열은 뒤의 optimizer에서 명시적으로 0으로 저장됩니다.
""",
    "11_slsqp_optimizer": r"""
### `μ35`가 실제 코드에서 조립되는 정확한 순서

1. `estimate_conditional_moments`에서 `macro_expected_return`과 `stress_return_adjustment`를 분리해 받습니다.
2. 거시 기대수익만 K-ratio·RSI confidence filter에 통과시킵니다.
3. KODEX200의 filtered macro에 EPS·valuation 조정을 더합니다.
4. 신용 stress multiplier를 KODEX200의 stress adjustment에만 곱합니다.
5. 마지막으로 `expected_return = filtered_macro + stress_adjustment`를 계산합니다.

개념식은 다음과 같습니다.

\[
\mu_{35}=Filter_{technical}(\mu_{macro})
+\Delta\mu_{EPS}+\Delta\mu_{valuation}
+Confirm_{credit}(\Delta\mu_{stress/recovery})
\]

### `Σ35`와 `Σ36`의 정확한 순서

\[
\Sigma_{macro/stress}
\rightarrow D_{ATR}\Sigma D_{ATR}
\rightarrow D_{credit}\Sigma D_{credit}
\rightarrow D_{GVZ,OVX}\Sigma D_{GVZ,OVX}
\]

Stage36은 마지막 화살표 하나만 추가합니다. `asset_scaling` 전후에 `expected_return`을 변경하는 코드가 없고 결과에도 `gvz_mu_adjustment_GLD=0`, `ovx_mu_adjustment_USO=0`을 기록합니다.

### SLSQP가 최대화하는 것

\[
U(w)=w'\mu-\frac12w'\Sigma w
-E[\min(R_{hist}w,0)^2]-C(w,w^{pre})
\]

- $w'\mu$: 기대수익 보상
- $\frac12w'\Sigma w$: 현재 추정 변동성 벌점
- 하방반분산: 과거 손실구간만 제곱해 부과하는 추가 벌점
- 거래비용: 현재 사전비중에서 새 비중으로 이동하는 마찰

제약은 $\sum w_i=1$, $0\le w_i\le1$, 연변동성 13% 이하, 과거 90% CDaR가 -16%보다 나쁘지 않을 것입니다. “단일자산 과반금지”는 없지만 비중합 1과 개별 0~1 때문에 레버리지와 공매도는 불가능합니다.

SLSQP는 목적함수의 기울기와 제약면을 함께 따라가는 비선형 최적화법입니다. 실패했을 때만 동일 제약의 최소분산 문제로 fallback합니다. `maxiter`와 `ftol`은 경제가설이 아니라 해를 얼마나 오래·정밀하게 찾을지 정한 수치설정입니다.
""",
    "12_monthly_backtest": r"""
### 직전 비중도 중요한 입력변수다

예측변수만으로 최적화하면 매달 100% 갈아타는 해가 좋아 보일 수 있습니다. 실제로는 지난달 목표비중이 자산수익률 때문에 자연스럽게 변한 `pretrade` 비중에서 출발해야 합니다.

예를 들어 지난달 GLD 목표가 30%였더라도 GLD만 크게 오르면 이번 달 리밸런싱 직전 비중은 30%보다 커집니다. 코드는 이 drift를 계산한 뒤 새 목표와의 차이에 비용을 매깁니다. 첫 진입에는 전체 절대변화량, 이후 보고용 turnover에는 양방향 매매의 중복계산을 피하려고 절반을 사용합니다. 실제 비용은 국내 15bp와 해외 순비중 변화 5bp를 모두 차감합니다.

각 반복에서 `history = returns[index < month]`로 잘라 현재월 실현수익이 추정과 최적화에 들어가지 않도록 합니다. 비중을 먼저 결정한 뒤에만 `returns.loc[month]`를 적용하는 순서를 코드에서 확인할 수 있습니다.
""",
    "15_research_orchestration": r"""
### 전체 알고리즘을 한 줄씩 추적하기

```text
GDP·수출·BSI → g
CPI·PPI·수입물가 → π
g, π → 네 soft-regime 확률
과거 확률가중 월수익 → μ_macro, Σ_macro
VKOSPI·VIX6 → stress/recovery adjustment
K-ratio·RSI → μ_macro confidence filter
ATR → Σ의 네 자산 위험축
EPS revision·EYG → KODEX200 μ
AA- spread → 주식 stress·Σ
= Stage35의 μ35, Σ35
GVZ→GLD, OVX→USO → Σ36만 추가 조정
SLSQP → 다음 달 long-only 무레버리지 비중
```

네 경로를 함께 계산하는 이유는 Stage36 결합결과만 보면 어느 센서가 영향을 냈는지 알기 어렵기 때문입니다. `Stage35_Frozen`은 기준, `GVZ-only`와 `OVX-only`는 개별 기여도, `GVZ+OVX`는 최종 후보입니다. 미래위험 회귀는 센서의 경제적 역할을 검사할 뿐 비중결정에는 사용되지 않습니다.
""",
    "16_results_and_charts": r"""
### 결과를 읽을 때 주의할 점

전체 2007~2026 구간에서 Stage36은 Stage35보다 Sharpe와 MDD가 좋아졌지만 CAGR은 약간 낮습니다. 2018 이후에는 CAGR과 Sharpe 모두 Stage35보다 낮으므로 “모든 시기에 우월한 알파”가 아니라 **일부 기대수익을 포기하고 금·원유 위험예산을 더 보수적으로 계산한 전략**으로 읽어야 합니다.

결과표에서 함께 볼 열은 `expected_mu_*`, 자산별 variance multiplier, 실제 `w_*`, `turnover`, `trade_cost`, `solver_success`, 두 위험제약의 slack입니다. 성과 숫자만 보지 않고 어떤 위험센서가 어떤 비중변화를 만들었는지 월별 CSV로 역추적할 수 있습니다.
""",
}


def split_sections(source: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"^# %% \[([^\]]+)\]\s*$", re.MULTILINE)
    matches = list(pattern.finditer(source))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        sections.append((match.group(1), source[start:end].strip() + "\n"))
    return sections


def build_notebook() -> Path:
    source = SOURCE.read_text(encoding="utf-8")
    sections = split_sections(source)
    if [name for name, _ in sections] != list(DESCRIPTIONS):
        raise ValueError("Section markers and notebook descriptions are out of sync")

    cells = [
        new_markdown_cell(
            """# Stage36 — GVZ·OVX 자산별 위험 오버레이

## 데이터 ZIP 하나로 재현하는 Google Colab 실행본

이 노트북은 로컬 `strategies` 패키지를 import하지 않습니다. Stage36과 실제 실행 경로에 필요한 Stage35·20·13·14·07·34·30·core 로직을 기능별 코드 셀로 풀어 넣었습니다.

### 사용 순서

1. 이 노트북을 Google Colab에서 엽니다.
2. **런타임 → 모두 실행**을 누릅니다.
3. 업로드 창이 뜨면 제공된 `stage36_colab_data.zip` 하나를 올립니다.
4. 약 수 분 뒤 2007-04~2026-07 성과표·위험회귀·bootstrap·월별 비중이 생성됩니다.

### 설계 경계

- GVZ는 GLD, OVX는 USO의 공분산 축만 조정합니다.
- GVZ/OVX로 기대수익 μ를 조정하지 않습니다.
- long-only, no leverage, 단일자산 과반금지 없음
- 출시 전 backfill·grid search·SJM/CJM·로지스틱 학습 없음
- 결과는 연구용 역사적 시뮬레이션이며 미래성과를 보장하지 않습니다.
"""
        ),
        new_markdown_cell(
            """## 원래 프로젝트 소스와 이 노트북 셀의 대응

| 원래 소스 | 이 노트북의 역할 |
|---|---|
| `stage36/.../asset_implied_volatility_risk_slsqp.py` | 10~16절: GVZ/OVX·DΣD·실험 |
| `stage35/.../earnings_credit_fundamentals_slsqp.py` | 9·11절: EPS·밸류·신용·기준 μ |
| `stage20/.../daily_technical_confidence_slsqp.py` | 8절: K-ratio·RSI·ATR |
| `stage13/.../economic_conditional_slsqp.py` | 3·6·7절: midrank·stress·조건부 μ/Σ |
| `stage14/.../dynamic_risk_slsqp.py` | 11~13절: bounds·비용·성과지표 |
| `stage07/.../zero_tune_strategy.py` | 4·5절: 원화 수익·거시확률·비용률 |
| `stage34/.../futures_basis_oi_confirmation_slsqp.py` | 14절: 미래위험 목적변수 |
| `stage30/.../abnormal_surface_erp_slsqp.py` | 13절: paired block bootstrap |
| `core/regime_research.py` | 4·11·13절: 자산목록·수익·CDaR |

노트북에는 이 경로에서 **실제로 호출되는 함수만** 포함합니다. 이전 Stage의 보고서·플롯 함수나 미사용 실험모델은 포함하지 않습니다.
"""
        ),
        new_markdown_cell(ORIENTATION_GUIDE.strip()),
        new_code_cell(
            """# Colab 기본환경에 없는 경우만 설치됩니다.
%pip -q install openpyxl statsmodels
"""
        ),
    ]

    for name, code in sections:
        explanation = DESCRIPTIONS[name].strip()
        if name in DETAILED_NOTES:
            explanation += "\n\n" + DETAILED_NOTES[name].strip()
        cells.append(new_markdown_cell(explanation))
        cells.append(new_code_cell(code))
        if name == "02_upload_and_extract":
            cells.append(
                new_code_cell(
                    """bundle_zip = locate_or_upload_bundle()
data_root = extract_bundle(bundle_zip)
print(f"사용 데이터 루트: {data_root}")
"""
                )
            )

    cells.extend(
        [
            new_markdown_cell(
                """## 17. 전체 실행

기본값은 원 프로젝트와 같은 12개월 블록·2,000회 bootstrap입니다. 로컬 자동검증에서만 환경변수로 반복수를 낮출 수 있으며 Colab에서는 자동으로 2,000회가 적용됩니다.
"""
            ),
            new_code_cell(
                """bootstrap_replications = int(os.environ.get("STAGE36_BOOTSTRAP_REPS", "2000"))
report = run_stage36_research(
    save=True,
    bootstrap_replications=bootstrap_replications,
)
display_stage36_results(report)
"""
            ),
            new_markdown_cell(
                """## 18. 기준 수치와 재현성 감사

Colab·SciPy 버전에 따른 SLSQP의 마지막 자리 차이는 허용하되, 핵심 성과는 기존 Stage36 결과와 0.01%p/0.001 이내여야 합니다. 모든 인과성·제약 검사도 참이어야 합니다.
"""
            ),
            new_code_cell(
                """expected = {
    ("Stage35_Frozen", "full_2007_2026"): {
        "CAGR": 0.1060756884, "Sharpe": 1.0573641671, "MDD": -0.1374348684,
    },
    ("Stage36_GVZ_OVXAssetRisk", "full_2007_2026"): {
        "CAGR": 0.1049938875, "Sharpe": 1.1049112901, "MDD": -0.1240708722,
    },
}
indexed = report["performance"].set_index(["Strategy", "Period"])
for key, targets in expected.items():
    for metric, target in targets.items():
        actual = float(indexed.loc[key, metric])
        tolerance = 1e-4 if metric != "Sharpe" else 1e-3
        assert abs(actual - target) <= tolerance, (key, metric, actual, target)
assert all(report["source_checks"].values()), report["source_checks"]
print("✅ Stage35·Stage36 핵심 성과와 모든 인과성·제약 검사가 통과했습니다.")
"""
            ),
            new_markdown_cell(
                """## 19. 주요 결과 파일 묶기

다음 셀은 월별 비중, 성과표, 위험회귀, bootstrap, validation JSON을 `stage36_colab_outputs.zip`으로 묶습니다. Colab이면 다운로드를 시작하고, 로컬이면 생성 경로만 출력합니다.
"""
            ),
            new_code_cell(
                """output_zip = zip_colab_outputs()
print(f"결과 ZIP: {output_zip}")
try:
    from google.colab import files
    files.download(str(output_zip))
except ImportError:
    pass
"""
            ),
        ]
    )

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"name": OUTPUT.name, "provenance": []},
        },
    )
    nbformat.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build_notebook())
