# Stage 34 — KOSPI200 선물 Basis·OI 확인축

## 최종 결론

`260829_K200선물데이터.xlsx`에서 선물시장 고유 정보인 basis, 미결제약정(OI), calendar spread를 만들고 Stage20에 적용해 성과까지 측정했다. 사전 경제 게이트는 통과하지 못했다.

> `keep_stage20_frozen_futures_signals_fail_promotion_gate`

Basis α 후보는 전체기간 CAGR과 MDD를 개선했지만 VIX6 통제 후 예측계수가 사라졌고, Sharpe는 개선되지 않았으며, 거래비용이 크게 늘었다. 따라서 결과표는 보존하되 기준 전략은 Stage20 그대로다.

## 원자료 감사에서 발견한 사항

엑셀은 최근월물과 차근월물 두 시트, 각 52열로 구성되어 있다. 한글 헤더는 일부 도구에서 깨져 보이지만 항목코드와 값은 정상적으로 읽힌다.

| 항목 | 감사 결과 | 사용 여부 |
|---|---|---|
| `P101800 정산가의이론가괴리율` | 프리미엄·할인의 부호가 없는 괴리 크기. 2011년 이후 0이 95.3% | 방향 신호에서 제외 |
| 종가·정산이론가 | 1996-05 이후 연속 관측 | 서명된 basis 재구성 |
| 미결제약정수량 | 1996-05 이후 거의 전 기간 | 동일 월물 20일 변화 |
| 최근월/차근월 종가·잔존일수 | 전 기간 | 연율화 calendar spread |
| 내재변동성 필드 | 유효구간 전체가 0 | 제외 |
| 내장 스프레드 필드 | 429일뿐 | 장기검정에서 제외 |

주신호는 다음처럼 고정했다.

```text
SignedBasis = 최근월물 종가 / 최근월물 정산이론가 - 1
BasisChange20D = SignedBasis(t) - SignedBasis(t-20)
OIChange20D = log(OI(t) / OI(t-20))
```

20거래일 전과 `contract_year_month`가 같은 경우에만 변화를 계산했다. 롤 전후 서로 다른 계약의 OI를 비교하지 않는다. 양의 basis 변화는 이론가 대비 프리미엄 확대 또는 할인 축소, 즉 선물 가격압력 개선으로 해석했다. 결과를 본 뒤 부호를 뒤집지 않았다.

## 연구 설계

연구기간은 Stage20과 같은 2007-04~2026-07, 232개월이다. 모든 월별 신호는 목표 월 직전 월의 마지막 완전 관측치다.

### 가설 A — Basis 방향정보

```text
Future KODEX200 Return
= BasisChange20D
+ VIX6 + recent return + realized vol + macro fragility
```

예상 부호는 양수다.

### 가설 B — OI 신규포지션 확인

```text
Future Return
= BasisChange20D + OIChange20D
+ BasisChange20D × OIChange20D
+ controls
```

상호작용의 예상 부호도 양수다. basis 개선에 OI 증가가 붙을수록 가격압력에 신규 포지션이 동반된다는 가설이다.

### 가설 C — VIX6 방어 확인축

```text
Future Risk
= FuturesDislocation + VIX6 + controls

FuturesDislocation = -SignedBasis
```

discount가 클수록 미래 변동성·MDD·왼쪽꼬리가 커질 것으로 예상했다. Stage20 방어월에서는 dislocation이 높을수록 false positive가 줄어야 한다.

## 방향 예측 결과

| 기간 | 모형 | Basis beta | p값 | IC | Basis×OI beta | p값 |
|---|---|---:|---:|---:|---:|---:|
| 1M | Basis 단독 | +0.4695%p | 0.148 | +0.055 | — | — |
| 1M | 전체 통제 | +0.1791%p | 0.670 | +0.055 | — | — |
| 1M | Basis·OI 상호작용 | +0.1550%p | 0.714 | +0.055 | -0.1791%p | 0.737 |
| 3M | 전체 통제 | -1.2563%p | 0.085 | -0.049 | — | — |
| 3M | Basis·OI 상호작용 | -1.3597%p | 0.055 | -0.049 | -0.7866%p | 0.429 |
| 6M | 전체 통제 | -1.3336%p | 0.332 | -0.057 | — | — |
| 6M | Basis·OI 상호작용 | -1.5332%p | 0.217 | -0.057 | -1.5728%p | 0.358 |

원시 1개월 관계는 약한 양수지만 VIX6와 기존 통제변수를 넣으면 beta가 0에 가까워진다. 3·6개월은 사전 가설과 반대인 음수다. OI 상호작용도 전 기간 음수이며 유의하지 않다.

## 미래위험과 false positive

| 목표 | Dislocation beta | p값 | 판정 |
|---|---:|---:|---|
| 다음 1M 실현변동성 | +0.00268 | 0.690 | 유의하지 않음 |
| 다음 1M 최대낙폭 | +0.00102 | 0.760 | 유의하지 않음 |
| 다음 3M 최대낙폭 | -0.00452 | 0.265 | 부호 반대 |
| 다음 달 왼쪽꼬리 | +0.0961 | 0.804 | 유의하지 않음 |

Stage20 방어월 204개에서 false-positive 로짓의 dislocation beta는 `-0.0468`, p값은 `0.746`이다. 방향은 예상과 맞지만 설명력이 없다.

기술통계에서는 VIX6가 높을 때 선물시장이 정상인 82개월의 방어 false-positive 비율이 56.5%, 선물 stress가 동반된 28개월은 45.5%였다. 그러나 표본 차이가 크고 회귀가 유의하지 않아 이 비율만으로 확인 규칙을 만들지 않았다.

## Stage20 적용 방식

파라미터 탐색 없이 네 후보와 무변경 재현 경로를 실행했다.

1. `BasisAlpha`: 과거 60개월 이상 자료로 추정한 비음수 단일회귀 slope를 KODEX200 μ에 추가
2. `BasisOIAlpha`: basis와 `basis×OI` 두 계수를 매월 NNLS로 추정해 KODEX200 μ에 추가
3. `RiskConfirmation`: 기존 KODEX200 VIX6 stress μ 조정을 `2 × 과거 dislocation 백분위`로 확인
4. `Combined`: BasisAlpha와 RiskConfirmation 동시 적용

모두 무레버리지·롱온리·합계 100%이며 Stage20의 SLSQP 변동성/CDaR 제약, 기술 신뢰도, 비용 계산을 유지한다.

## 성과

### 2007-04~2026-07

| 전략 | CAGR | Sharpe | MDD | 월평균 회전율 | 누적 거래비용 |
|---|---:|---:|---:|---:|---:|
| **Stage20 동결** | **9.397%** | **0.987** | **-14.015%** | 2.68% | 1.86% |
| BasisAlpha | 9.582% | 0.986 | -11.280% | 7.58% | 5.34% |
| BasisOIAlpha | 9.581% | 0.986 | -11.287% | 7.58% | 5.34% |
| RiskConfirmation | 9.399% | 0.987 | -14.024% | 2.68% | 1.86% |
| Combined | 9.584% | 0.986 | -11.280% | 7.57% | 5.33% |

BasisAlpha의 CAGR은 Stage20보다 +0.184%p, MDD는 +2.735%p 개선됐다. 그러나 Sharpe는 -0.0003 낮고 CAGR 10%, Sharpe 1을 넘지 못했다. OI를 추가해도 사실상 변화가 없다. RiskConfirmation도 거의 같은 경로다.

### 구간 성과

| 전략 | 2007~2017 CAGR / Sharpe / MDD | 2018~2026 CAGR / Sharpe / MDD |
|---|---|---|
| Stage20 | 6.65% / 0.756 / -14.01% | 12.94% / 1.250 / -11.42% |
| BasisAlpha | 6.41% / 0.777 / -10.96% | 13.69% / 1.209 / -11.28% |
| Combined | 6.41% / 0.778 / -11.00% | 13.69% / 1.209 / -11.28% |

BasisAlpha는 초기 CAGR을 낮추고 후기 CAGR을 높였다. Sharpe도 초기에는 개선, 후기에는 악화했다. 전 구간에 같은 방식으로 작동하는 안정적인 독립 알파라고 보기 어렵다.

### 12개월 블록 부트스트랩

BasisAlpha가 Stage20보다 좋아질 확률은 CAGR 59.5%, Sharpe 51.8%, MDD 86.6%였다. MDD 흔적은 있지만 CAGR·Sharpe 개선은 동전 던지기에서 크게 벗어나지 않는다.

## 왜 성과가 조금 좋아도 승격하지 않았나

성과표를 먼저 보고 basis 규칙을 채택하면 Stage34의 출발점인 “VIX6와 다른 독립 정보인가?”를 건너뛰게 된다.

- 1개월 통제 beta p=0.670
- Basis×OI p=0.737
- 3·6개월 basis 부호 반대
- 미래위험 네 목표 모두 p>0.26
- false-positive 로짓 p=0.746
- CAGR·Sharpe 부트스트랩 우위 확률 약 50~60%
- 거래비용 약 3.5%p 증가

MDD 개선만 남지만 이것도 고정된 메커니즘 검정에서 설명되지 않는다. 따라서 기준 전략 변경의 근거로 사용하지 않는다.

## 실행

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage34_futures_basis_oi_confirmation.futures_basis_oi_confirmation_slsqp
```

주요 파일:

- `futures_basis_oi_confirmation_slsqp.py`: 원자료 감사, 인과적 신호, 회귀, SLSQP 백테스트, 게이트
- `stage34_futures_basis_oi_report.html`: 상세 설명과 결과
- `outputs/normalized_k200_futures_daily.csv`: 정규화된 최근·차근월 원자료와 재구성 신호
- `outputs/monthly_futures_basis_oi_signals.csv`: 실제 매월 사용 신호와 인과적 보정값
- `outputs/return_predictive_regressions.csv`: 방향·OI 상호작용 회귀
- `outputs/risk_predictive_regressions.csv`: 미래위험 회귀
- `outputs/performance_comparison.csv`: 전체·초기·후기 성과
- `outputs/validation_report.json`: 설계·게이트·무결성 감사

