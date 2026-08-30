# Stage36 — GVZ / OVX Asset-Specific Implied-Volatility Risk

Stage35를 동결 기준으로 두고 자산별 옵션시장의 기대변동성을 해당 자산의 위험축에만 연결한 월간 자산배분 연구다.

- GVZCLS → GLD covariance
- OVXCLS → USO covariance
- 방향성 기대수익 mu 변경 없음
- 252개 과거 유효 일간관측 전에는 multiplier 1.0
- 출시 전 backfill, 실현변동성 대체, 다른 지수 splice 없음
- 활성화 후 variance multiplier는 1 + causal expanding midrank
- long-only, no leverage, 단일자산 상한 없음
- Stage35와 동일한 SLSQP 목적함수·13% 변동성 guard·−16% CDaR guard·거래비용

## 구현

주 실행 파일은 asset_implied_volatility_risk_slsqp.py다.

핵심 함수:

1. load_asset_implied_volatility_daily()
   - raw_data/GVZCLS.csv, raw_data/OVXCLS.csv를 읽는다.
   - FRED의 점 등 숫자가 아닌 값은 결측치로 처리한다.
   - 각 센서의 causal expanding midrank와 prior valid count를 만든다.

2. build_monthly_asset_volatility_signals()
   - 대상 월 t에는 t-1 월말까지 알려진 마지막 값만 사용한다.
   - 252개 prior observations가 없으면 센서 효과는 중립이다.
   - GVZ는 2009-07 target month, OVX는 2008-06 target month부터 활성화됐다.

3. _solve_weights()
   - Stage35의 EPS·valuation·credit/VIX6 결합을 그대로 재현한다.
   - 선택한 센서만 아래 대각 스케일링으로 공분산에 추가한다.

    q_GVZ,t = causal_rank(GVZ_t)
    q_OVX,t = causal_rank(OVX_t)

    D_GLD,GLD = sqrt(1 + q_GVZ,t)
    D_USO,USO = sqrt(1 + q_OVX,t)

    Sigma_stage36 = D @ Sigma_stage35 @ D
    mu_stage36 = mu_stage35

4. asset_risk_predictive_regressions()
   - 자기자산의 최근 1개월 수익률·21일 실현변동성, VIX6 stress, 거시 취약도를 통제한다.
   - 향후 1개월 실현변동성, 1·3개월 MDD, left-tail을 HAC 회귀로 검사한다.

5. run_research()
   - Stage35 재현, GVZ-only, OVX-only, GVZ+OVX 경로를 모두 실행한다.
   - 전체구간, 2010 공통구간, 2018 잠금구간 성과와 12개월 block bootstrap을 저장한다.

## 결과

### 전체 구간 2007-04 ~ 2026-07

| Strategy | CAGR | Volatility | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 10.608% | 10.043% | 1.057 | -13.743% |
| GVZ → Gold | 10.262% | 9.511% | 1.078 | -12.407% |
| OVX → Oil | 10.846% | 10.005% | 1.083 | -13.743% |
| **GVZ + OVX** | **10.499%** | **9.472%** | **1.105** | **-12.407%** |

결합전략은 Stage35 대비:

- CAGR −0.108%p
- 연환산 변동성 −0.571%p
- Sharpe +0.048
- MDD +1.336%p — 덜 음수인 방향

### 공통 구간 2010-01 ~ 2026-07

| Strategy | CAGR | Volatility | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 9.653% | 9.371% | 1.033 | -13.743% |
| GVZ + OVX | 9.232% | 8.719% | 1.059 | -12.407% |

2008 금융위기 초반의 GVZ 초기표본 효과를 제외해도 Sharpe와 MDD 개선은 유지됐지만 CAGR은 0.421%p 낮아졌다.

### 2018-01 ~ 2026-07

| Strategy | CAGR | Volatility | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage35 | 13.821% | 10.861% | 1.251 | -11.984% |
| GVZ + OVX | 12.626% | 10.178% | 1.224 | -11.931% |

최근 구간에서는 변동성과 MDD가 조금 낮아졌지만 CAGR과 Sharpe도 낮아졌다. Stage36이 모든 시대에 Stage35를 지배한다고 볼 수 없다.

## 미래 위험 진단

2010 공통구간 FullControls 결과:

| Sensor | Target | standardized beta | HAC p-value | Spearman IC |
|---|---|---:|---:|---:|
| GVZ | GLD future 1M realized vol | +0.01955 | 0.0029 | +0.492 |
| GVZ | GLD future 1M MDD magnitude | +0.00733 | 0.0014 | +0.411 |
| GVZ | GLD future 3M MDD magnitude | +0.00931 | 0.0871 | +0.398 |
| OVX | USO future 1M realized vol | +0.04984 | 0.0010 | +0.588 |
| OVX | USO future 1M MDD magnitude | +0.01281 | 0.0977 | +0.261 |
| OVX | USO future 3M MDD magnitude | +0.02322 | 0.0369 | +0.170 |

두 센서는 최근 자기자산 변동성과 VIX6를 통제한 뒤에도 적어도 하나의 핵심 위험목적변수에서 양의 추가 설명력을 보였다.

## 해석

- 성과 개선의 대부분은 GVZ가 GLD 평균비중을 약 34.5% → 30.0%로 낮추고 BOND로 위험예산을 옮긴 데서 왔다.
- Stage35의 USO 평균비중은 약 0.35%이고 양의 비중 월도 6개월뿐이었다. 그래서 OVX가 미래 USO 위험을 잘 설명해도 2010 이후 포트폴리오 성과 영향은 거의 없다.
- 전체구간 OVX-only 개선은 초기 몇 개의 USO 보유월에 집중되어 있다. 이를 더 키우려고 USO 최소비중이나 mu 조정을 추가하지 않았다.
- 결합전략은 전체 표본에서 기존 목표인 CAGR 10% 이상, Sharpe 1 이상을 유지하면서 MDD를 완화했다.
- 2018 이후 CAGR·Sharpe 저하는 명확한 비용이므로, 절대 우월이 아니라 더 방어적인 선택으로 이해해야 한다.

## 실행

프로젝트 루트에서:

    & d:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m strategies.stage36_asset_implied_volatility_risk.asset_implied_volatility_risk_slsqp

테스트:

    & d:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m pytest tests\test_stage36_asset_implied_volatility_risk.py -q

상세 보고서:

- stage36_gvz_ovx_report.html
- stage36_implementation_economic_math_guide.html
  - 경제적 가설, 인과적 순위, 조건부 모멘트, 공분산 변환,
    SLSQP 목적함수·제약·비용과 전체 소스코드 탐색 순서를 설명한다.

주요 산출물:

- outputs/monthly_asset_volatility_signals.csv
- outputs/asset_risk_predictive_regressions.csv
- outputs/performance_comparison.csv
- outputs/paired_block_bootstrap_vs_stage35.csv
- outputs/stage36_gvz_ovxassetrisk_monthly.csv
- outputs/validation_report.json

## Google Colab 단일 노트북

`colab/Stage36_GVZ_OVX_Colab.ipynb`는 Stage36의 실제 실행 의존경로를
프로젝트 모듈 import 없이 한 파일 안에 풀어 넣은 실행본이다. Stage36뿐 아니라
그 실행에 사용되는 Stage35·20·13·14·07·34·30 및 core 함수가 기능별 셀로
나뉘어 있으며, 각 코드 셀 앞에 경제적 역할·수학식·인과성 경계를 설명한다.

설명은 단순 함수 요약에 그치지 않는다. 먼저 “Stage35가 본체이고 Stage36은
위험센서”라는 관점에서 전체 입력변수 지도를 제시하고, 이어서 다음 내용을 실제
코드가 실행되는 순서에 맞춰 풀어 썼다.

- GDP·수출·BSI와 CPI·PPI·수입물가가 네 soft-regime 확률이 되는 과정
- 확률가중 조건부 평균·공분산과 effective sample shrinkage의 의미
- VKOSPI/VIX6 stress·recovery가 기대수익과 위험에 연결되는 방식
- K-ratio·price RSI·volume RSI가 거시 기대수익을 뒤집지 않고 신뢰도만 조절하는 식
- ATR, EPS revision, earnings-yield gap, AA- 신용스프레드의 정확한 역할
- Stage35의 `μ35·Σ35` 조립순서와 Stage36의 GVZ/OVX `DΣD` 경계
- SLSQP 목적함수·CDaR·거래비용·직전 drift 비중을 함께 쓰는 이유

Colab 사용 순서:

1. `colab/Stage36_GVZ_OVX_Colab.ipynb`를 Google Colab에서 연다.
2. 런타임의 **모두 실행**을 누른다.
3. 업로드 창에서 `colab/stage36_colab_data.zip` 하나만 선택한다.
4. 성과표·월별 비중·위험회귀·bootstrap·검증 JSON을 내려받는다.

데이터 ZIP은 16개 데이터 파일과 SHA-256 manifest만 담고 코드 파일은 담지
않는다. 즉 실행 코드는 노트북에서 그대로 열어 읽고 수정할 수 있다.

재생성 도구:

- `colab/build_stage36_colab_data_bundle.py`: 현재 데이터 스냅샷으로 ZIP 재생성
- `colab/stage36_colab_cells.py`: 노트북 코드 셀의 검토 가능한 단일 원본
- `colab/build_stage36_colab_notebook.py`: 설명 셀과 코드 셀을 조립해 ipynb 재생성

로컬 QA에서는 데이터 ZIP만 지정해 모든 셀을 실행했으며, 232개월 전 구간에서
Stage36 결합경로의 CAGR 10.499%, Sharpe 1.105, MDD -12.407%를 재현했다.
Stage35 기준경로도 CAGR 10.608%, Sharpe 1.057, MDD -13.743%로 재현했고,
모든 월의 SLSQP 해가 long-only·비중합 1·무레버리지 및 위험제약 감사를
통과했다.
