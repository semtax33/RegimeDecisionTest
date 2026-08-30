# Stage 28 — VIX6 분해 대체 KOSPI200 옵션 방향 전략

## 결론

Stage20 원본은 수정하지 않았다. Stage28 후보에서는 VIX6 여섯 성분을 제거하고 다음처럼 역할을 분리했다.

```text
VKOSPI                         → 위험의 크기
KOSPI200 옵션가격·거래대금    → 주식 방향
거시 4국면                    → 기본 기대수익
K-Ratio·ATR·RSI               → Stage20 기술 신뢰도·위험
```

옵션 방향값은 KODEX200 기대수익에만 더했다. λ=1, ATR 공분산, 연 13% 변동성 한도, 16% CDaR 한도, 롱온리·무레버리지와 다른 자산 기대수익은 변경하지 않았다.

전체 2007-04~2026-07에서는 CAGR이 9.40%에서 10.03%로 올랐지만 Sharpe는 0.987에서 0.978로 낮아지고 MDD는 -14.01%에서 -18.08%로 악화됐다. 따라서 **Stage20을 대체하지 않는다.**

## 성과

| 전략 | 구간 | CAGR | 변동성 | Sharpe | MDD |
|---|---|---:|---:|---:|---:|
| Stage20 VIX6 | 2007-04~2026-07 | 9.3974% | 9.5981% | 0.9865 | -14.0148% |
| **Stage28 ODS** | 2007-04~2026-07 | **10.0328%** | 10.3510% | 0.9779 | -18.0833% |
| Stage20 VIX6 | 2007-04~2017-12 | 6.6493% | 9.0712% | 0.7563 | -14.0148% |
| Stage28 ODS | 2007-04~2017-12 | 4.4493% | 9.5754% | 0.5026 | -18.0833% |
| Stage20 VIX6 | 2018-01~2026-07 | 12.9392% | 10.1861% | 1.2503 | -11.4190% |
| **Stage28 ODS** | 2018-01~2026-07 | **17.4486%** | 10.9952% | **1.5255** | **-11.0361%** |

2018년 이후에는 세 지표가 모두 좋아졌지만 2007~2017 성과가 크게 나빴다. 전체 표본에서 안정적인 대체 전략이 아니라 **기간 의존성이 큰 탐색 결과**다.

12개월 paired block bootstrap에서 Stage28-Stage20 CAGR 차이가 양수일 확률은 78.6%였지만 Sharpe는 59.4%, MDD 개선은 36.1%였다. 구간도 모두 0을 포함하므로 확정적인 우월성 근거가 아니다.

## 옵션 팩터

주간 KOSPI200 옵션 원자료를 사용한다. 구형 자료는 종목코드 첫 자리로 콜·풋을 복원하고, 야간 자료는 제외한다.

### 1. Put Wing

```text
PutSkew25 = IV(25-delta put) - ATM IV
```

선물가격 자료가 없으므로 같은 행사가의 콜·풋 종가를 이용한 무이자 put-call parity로 단기 forward를 추정한다.

### 2. Downside/Up­side implied variance

OTM 풋·콜 가격을 각각 적분한다.

```text
IVA = (downside IVar - upside IVar) / (downside IVar + upside IVar)
```

적분 전 풋 가격은 행사가에 따라 감소하지 않고 콜 가격은 증가하지 않도록 수직차익 단조성만 투영했다. 완전한 convex no-arbitrage surface는 아니다.

### 3. 거래대금 방향

```text
CPV = log(Call trading value / Put trading value)
```

원자료에는 bid/ask와 체결주도 방향이 없으므로 signed order flow가 아니라 일별 총 거래대금 비율이다.

### 4. 옵션 내재 ERP 대용치

```text
ERP proxy = downside IVar + upside IVar
```

이는 Martin식 SVIX² 아이디어의 대용치다. 무위험 할인곡선과 무한 행사가 꼬리가 없으므로 완전한 option-implied ERP로 표현하지 않았다.

## Fast/Slow 방향

5일과 20일 변화량을 각각 인과적 expanding Z-score로 바꾼다.

```text
Bear = [Z(Δ PutSkew) + Z(Δ IVA) - Z(Δ CPV)] / 3
Direction = Z(ERP proxy) - Bear
ODS = mean(Direction_5D, Direction_20D)
ODSScore = ODS / (1 + abs(ODS))
```

ODS는 단위가 없기 때문에 그달 거시 기대수익의 자산 간 표준편차를 곱해 월수익률 단위로 바꾼다.

```text
mu_KODEX200 += ODSScore × cross_sectional_std(macro_mu)
```

이 환산에는 학습계수나 탐색된 배율이 없다.

## 데이터 한계

피드백은 30일 고정만기 표면을 요구하지만 KRX 원자료는 과거 대부분의 날짜에 최근월물 하나만 제공한다.

- 생성된 일별 표면: 4,997일
- 정확히 30일 만기: 103일
- 두 만기 사이 30일 보간: 0일
- 7~60일 중 가장 가까운 상장만기 대용치: 4,894일

따라서 이 전략은 엄밀한 30일 constant-maturity surface가 아니라 **nearest-listed maturity proxy**다. 이 한계가 2007~2017과 2018년 이후의 성과 차이에 일부 영향을 줬을 가능성을 배제할 수 없다.

## 방향성 진단

- 일별 ODS → 다음 5일 수익률 IC: +0.068
- 일별 ODS → 다음 20일 수익률 IC: +0.135
- 월별 ODS → 해당 투자월 KODEX200 수익률 IC: +0.162
- 5분위 평균수익의 순위 상관: 5일 +0.706, 20일 +0.741
- Q1→Q5 평균수익은 엄격한 단조증가가 아님

Q5의 다음 20일 평균수익률은 3.58%였지만 Q5의 하위 10% 꼬리 평균도 -16.73%로 가장 나빴다. ERP가 높다는 것은 기대보상이 높다는 뜻이지 안전하다는 뜻은 아니다. 이 특성이 전체 MDD 악화와 연결된다.

## 추가 피드백 검토

추가 피드백의 `Order Flow + Abnormal Surface Residual + ERP` 구분은 타당하다. 다만 이번 자료와 연구 단계에서는 다음 이유로 사후 추가하지 않았다.

- 진짜 signed order flow: bid/ask 및 aggressor-side 체결자료가 없어 구현 불가능
- Abnormal Surface residual: VKOSPI·현물수익·거시변수를 사용한 새 회귀모델이 필요하며, Stage28 결과를 본 뒤 붙이면 사후 탐색이 됨
- ERP: 현재는 strike tail과 할인곡선이 없는 대용치이므로 별도 정교화 필요

다음 연구에서는 VIX6를 위험 엔진에 유지하고, signed flow 자료가 확보된 뒤 방향 α만 별도 사전등록하는 편이 낫다. 이번 결론을 바꾸기 위해 추가 피드백에 맞춰 코드를 사후 수정하지 않았다.

## 파일

- `option_directional_surface_slsqp.py`: 옵션 체인 정규화, 표면 팩터, VKOSPI-only 위험, SLSQP
- `evaluate_replacement.py`: 전체·초기·고정 구간과 bootstrap 비교
- `outputs/daily_option_direction_features.csv`: 일별 옵션 표면과 ODS
- `outputs/monthly_option_direction_signals.csv`: 다음 달에 사용하는 인과적 신호
- `outputs/option_directional_surface_monthly.csv`: Stage28 월별 비중과 수익
- `outputs/replacement_performance_by_subperiod.csv`: 구간별 최종 비교
- `outputs/replacement_paired_block_bootstrap.csv`: paired bootstrap
- `outputs/validation_report.json`: 공식·감사·제약·성과
- `outputs/replacement_decision.json`: 최종 유지/대체 판정

## 실행

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage28_option_directional_surface.option_directional_surface_slsqp
& $py -m strategies.stage28_option_directional_surface.evaluate_replacement
```
