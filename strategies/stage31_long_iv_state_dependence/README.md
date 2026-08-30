# 1997~2026 버킷 IV와 Stage 30 상태의존성 검증

## 판정

피드백에서 제안한 세 가지 핵심 가설을 `raw_data/260829_옵션내재변동성.xlsx`로 검증했다.

1. Stage 30 ODS의 경제적 관계가 2018년 전후로 구조적으로 변했는가?
2. ODS의 의미가 거시 취약도 또는 VIX6 스트레스에 따라 달라지는가?
3. 약한 ODS 계수를 reliability shrinkage로 줄이면 성과가 개선되는가?

결과는 다음과 같다.

- ODS의 IC는 2007~2017년 `-0.065`, 2018~2026년 `+0.111`로 표본상 부호가 바뀌었다.
- 그러나 2018년 구조변화 상호작용의 p값은 `0.405`다. 부호 변화는 보이지만 구조변화를 확정할 통계적 증거는 부족하다.
- `ODS × Macro Fragility`의 p값은 `0.779`, `ODS × VIX6 Stress`의 p값은 `0.725`다. 피드백에서 예상한 상태의존성은 확인되지 않았다.
- reliability shrinkage는 Stage 30보다 Sharpe를 `0.0025` 높였지만 CAGR을 `0.306%p` 낮추고 MDD를 `0.428%p` 악화시켰다.

따라서 **Fragility interaction은 전략에 넣지 않았으며, Stage 31 reliability 후보도 승격하지 않는다.** 기존 전략 중에서는 Stage 20을 공식 기준으로 유지하고, Stage 30은 더 높은 CAGR과 조금 나은 MDD를 원하는 연구 변형으로 남기는 것이 현재 증거에 맞다.

## 사용한 XLSX

### 옵션 버킷 IV

`260829_옵션내재변동성.xlsx`에는 두 시트가 있다.

- `최근월물`: CALL/PUT ATM, ITM1~4, OTM1~4
- `차근월물`: CALL/PUT ATM, ITM1~4, OTM1~4

실제 IV 관측은 1997-07-07부터 2026-08-28까지다. 피드백에서 제시한 식을 결과 확인 전에 주 검정으로 고정했다.

```text
WingAsym_near = IV(PUT OTM2, 최근월) - IV(CALL OTM2, 최근월)
BucketDirection_near = -WingAsym_near
```

`BucketDirection`의 부호는 Stage 30과 맞추기 위해 양수일수록 bullish가 되도록 뒤집었다. OTM1·OTM3·OTM4를 돌아가며 고르거나 윈도우·임계값을 탐색하지 않았다. 차근월 OTM2와 최근·차근월 동일가중은 주 검정의 강건성 확인에만 사용했다.

### KOSPI200 수익률

1997년부터 일관된 장기 수익률이 필요해 `260829_K200선물데이터.xlsx`의 최근월 종가 `P100400`을 사용했다. 자료는 1996-05-03부터 존재한다. 월 `t`의 신호는 `t-1` 마지막 완전 관측치로 만들고, 월 `t`의 선물 수익률을 예측하도록 배치했다.

연속선물의 롤 점프 가능성을 확인하기 위해 2000년 이후에는 기존 Stage 30의 KOSPI200 현물 종가로 같은 검정을 반복했다. 결과의 부호가 동일했기 때문에 핵심 결론은 선물 롤에 의존하지 않는다.

## 장기 버킷 IV가 알려준 것

### 주 검정

아래 IC는 `BucketDirection = -(PUT OTM2 - CALL OTM2)`와 다음 달 수익률의 Spearman 상관이다.

| 구간 | 최근월 주 검정 IC | 최근월 표준화 beta | HAC p값 |
|---|---:|---:|---:|
| 1997~2026 전체 | -0.138 | -1.94% | 0.098 |
| 1997~2006 | -0.177 | -3.42% | 0.116 |
| 2007~2017 | -0.103 | -0.05% | 0.951 |
| 2018~2026 | -0.089 | -1.72% | 0.086 |

방향 부호를 다시 원래의 `WingAsym`으로 읽으면 전체 IC는 `+0.138`이다. 즉 풋 OTM2 IV가 콜 OTM2 IV보다 높을수록 다음 달 수익률이 높은 **공포 이후 위험프리미엄 또는 반등** 성격에 가깝다.

이 관계는 Stage 30 ODS와 다르다. Stage 30은 스큐 수준 하나를 쓰는 것이 아니라, 5일·20일 풋 스큐와 내재분산 비대칭의 거시·VKOSPI 잔차 변화를 결합한다. 두 신호의 월별 상관은 전체 `0.193`으로 약한 양의 관계지만, 향후 수익률에 대한 부호와 시대별 움직임은 같지 않다. 따라서 버킷 IV를 Stage 30 방향 알파의 대체재로 바로 넣을 근거가 없다.

### 차근월과 현물 교차검증

전체 1997~2026 차근월 IC는 `-0.104`, 두 월물 평균은 `-0.114`다. 2000~2026 KOSPI200 현물 수익률을 사용했을 때도 최근월 IC는 `-0.113`, 차근월은 `-0.118`, 두 월물 평균은 `-0.123`이었다.

방향과 대략적인 크기가 유지되므로 다음 두 설명은 가능성이 낮다.

- 최근월 롤 점프가 우연히 만든 상관
- 특정 한 월물에서만 나타난 데이터 오류

## Stage 30 ODS의 구조변화

| 구간 | Option direction score IC | 표준화 beta | HAC p값 |
|---|---:|---:|---:|
| 2006~2026 전체 | 0.026 | 0.40% | 0.437 |
| 2007~2017 | -0.065 | -0.03% | 0.948 |
| 2018~2026 | 0.111 | 0.83% | 0.379 |

후기 IC가 초기보다 좋아진 것은 사실이다. 하지만 2018년 더미와 `ODS×후기더미`를 넣은 HAC 회귀에서 상호작용 p값은 `0.405`였다. 장기 버킷 IV의 2018년 구조변화 p값도 `0.355`다.

따라서 현재 가능한 표현은 다음 정도다.

> ODS의 표본상 유효성은 후기에서 좋아졌지만, 장기 독립 IV 자료까지 포함했을 때 2018년을 경계로 경제적 관계가 바뀌었다고 확정할 수준은 아니다.

## 거시와 VIX6 상태의존성

거시국면은 hard classification을 만들지 않고 각 월의 soft probability를 관측 가중치로 사용했다.

| Soft state | Stage 30 ODS 가중 IC |
|---|---:|
| Goldilocks | 0.015 |
| Overheating | 0.015 |
| Slowdown | 0.011 |
| Stagflation | 0.088 |
| VIX6 저스트레스 가중 | 0.025 |
| VIX6 고스트레스 가중 | 0.024 |

Fragility는 `p_Slowdown + p_Stagflation`으로 고정했다.

```text
KOSPI200_return(t)
  = a + beta*ODS(t)
      + delta*Fragility(t)
      + gamma*ODS(t)*Fragility(t) + error(t)
```

- Macro Fragility interaction: `gamma=0.0803`, p=`0.779`
- VIX6 Stress interaction: `gamma=-0.1300`, p=`0.725`

둘 다 오차가 매우 커서 상태에 따라 ODS의 의미가 바뀐다는 주장을 지지하지 않는다. 특히 Slowdown과 Stagflation에서 ODS IC가 음수가 될 것이라는 피드백의 예시는 실제 자료에서 재현되지 않았다. Stagflation 가중 IC가 오히려 가장 높았다.

상호작용을 백테스트 후보로 만들지 않은 이유도 여기에 있다. 진단이 불분명한 상태에서 interaction을 전략에 넣으면 결과를 본 뒤 자유도를 추가하는 셈이 된다.

## Reliability shrinkage 검증

Stage 30의 기존 보정은 과거 자료로 추정한 양(+)의 OLS slope를 그대로 사용한다.

```text
mu_adjustment = max(beta_hat, 0) * ODS_score
```

Stage 31 후보는 피드백의 식을 그대로 적용했다.

```text
z = beta_hat / SE(beta_hat)
Reliability = z^2 / (1 + z^2)
mu_adjustment = max(beta_hat, 0) * Reliability * ODS_score
```

임계값이나 튜닝 계수는 없다. 모든 beta와 표준오차는 해당 월보다 앞서 실현된 자료만으로 expanding 추정했다.

그러나 ODS slope의 불확실성이 커서 평균 reliability는 `0.110`, 중앙값은 `0.0159`에 불과했다. 이 때문에 KOSPI200 월 기대수익 조정의 평균 절댓값이 `0.000657`에서 `0.000174`로 약 74% 줄었다. 노이즈만 제거한 것이 아니라 Stage 30이 얻던 수익 기여도 대부분 같이 제거했다.

## 성과

### 2007-04~2026-07 전체

| 전략 | CAGR | Sharpe | MDD | Calmar | 평균 회전율 |
|---|---:|---:|---:|---:|---:|
| Stage 20 VIX6 | 9.397% | **0.987** | -14.015% | 0.671 | 2.681% |
| Stage 30 ODS | **9.782%** | 0.980 | **-13.599%** | **0.719** | 2.992% |
| Stage 31 Reliability | 9.476% | 0.982 | -14.027% | 0.676 | 2.725% |

Stage 31은 Stage 30 대비:

- CAGR `-0.306%p`
- Sharpe `+0.0025`
- MDD `-0.428%p` 악화
- Calmar `-0.0438`

목표였던 CAGR 10%, Sharpe 1에도 도달하지 못했다.

### 구간별

| 전략 | 2007~2017 CAGR / Sharpe / MDD | 2018~2026 CAGR / Sharpe / MDD |
|---|---|---|
| Stage 20 | 6.649% / 0.756 / -14.015% | 12.939% / **1.250** / **-11.419%** |
| Stage 30 | 6.521% / 0.740 / **-13.599%** | **14.007%** / 1.241 / -11.554% |
| Stage 31 | 6.582% / 0.750 / -14.027% | 13.211% / 1.244 / -11.468% |

12개월 원형 블록 부트스트랩 2,000회에서도 Stage 31의 CAGR 개선 확률은 `13.7%`에 불과했다. Sharpe 개선 확률은 `58.9%`였지만 크기가 작고 CAGR·MDD 손실을 보상하지 못한다.

## 최종 해석

이번 XLSX는 “ODS가 state-dependent alpha다”라는 설명을 강화하기보다, 서로 다른 옵션 신호를 구분해야 한다는 점을 보여줬다.

- 단순 OTM2 IV 수준 차이: 공포 수준과 이후 위험프리미엄/반등에 가까움
- Stage 30 ODS: 스큐·내재분산 변화의 비정상 잔차를 이용하는 약한 방향 보조신호
- VIX6: 현재 위험과 자산비중을 조절하는 위험 엔진

세 정보는 모두 옵션시장에서 나오지만 같은 신호가 아니다. 장기 버킷 IV가 Stage 30 후기 IC의 구조변화를 독립적으로 확인해주지 않았고, 거시·스트레스 interaction도 유의하지 않았다.

현재 근거로는 다음과 같이 판단한다.

1. 버킷 IV를 Stage 30 입력변수로 바로 추가하지 않는다.
2. Macro Fragility interaction을 추가하지 않는다.
3. reliability Stage 31은 승격하지 않는다.
4. 공식 기준은 Stage 20으로 유지한다.
5. Stage 30은 CAGR과 MDD의 교환관계를 감수할 때만 연구 변형으로 유지한다.

## 재현 파일

- `long_iv_state_dependence_slsqp.py`: XLSX 적재, 장기 IV·상태의존성 진단, Stage 31 백테스트
- `outputs/normalized_long_iv_daily.csv`: 원본을 수정하지 않고 만든 정규화 일별 버킷 IV
- `outputs/monthly_bucket_iv_signals.csv`: `t-1` 말 기준 월별 신호
- `outputs/long_iv_era_diagnostics.csv`: 선물수익률 기준 장기 검정
- `outputs/long_iv_spot_return_robustness.csv`: 현물수익률 교차검증
- `outputs/stage30_ods_era_diagnostics.csv`: Stage 30 ODS 시대별 IC·beta
- `outputs/soft_state_diagnostics.csv`: 거시 및 VIX6 soft-state 결과
- `outputs/continuous_interaction_hac.csv`: Fragility·VIX6 HAC interaction
- `outputs/rolling_stage30_ods_beta_ic.csv`: ODS expanding beta·IC
- `outputs/rolling_long_iv_beta_ic.csv`: 장기 버킷 IV expanding beta·IC
- `outputs/stage31_reliability_monthly.csv`: Stage 31 월별 백테스트
- `outputs/performance_comparison.csv`: Stage 20/30/31 성과
- `outputs/validation_report.json`: 가설·성과·무결성 감사 종합 결과

두 원본 XLSX는 분석 전후 SHA-256, 크기, 수정시각이 동일하다.

- 옵션 IV: `967f0a612eb8ccde36e47a5ad870e1a27c58bd43685dc9234ac4f7b46e403d6c`
- K200 선물: `4d37798c636a4716b7a7d03b71549d7195067e35718635da0db2420f019d0818`
