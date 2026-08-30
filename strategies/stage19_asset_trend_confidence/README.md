# Stage 19 — 자산별 추세를 이용한 매크로 기대수익 신뢰도 필터

## 결론

피드백의 핵심 원인 진단은 맞았다. Stage 14의 최대 낙폭은 주식 스트레스가 아니라 **금에 대한 낙관적인 매크로 기대수익과 금 자체의 장기 하락 추세가 충돌했는데도 높은 금 비중을 유지한 것**에서 발생했다.

이 문제를 고치기 위해 자산별 6개월·12개월 자체 추세를 매크로 기대수익의 신뢰도 필터로만 사용하는 단일 가설을 별도 Stage 19 폴더에 구현했다. 2012~2014 금 낙폭과 전체 MDD는 크게 개선됐지만 CAGR과 Sharpe 성공 기준은 통과하지 못했다. 따라서 **Stage 19를 Stage 14의 최종 대체 전략으로 채택하지 않는다.**

성과를 확인한 뒤 기간, 계수, 중립값이나 신뢰도 식을 다시 바꾸지 않았다.

## 구현 전 범인 확인

Stage 14 고정 λ 경로의 2012-10~2014-10 최대 낙폭 구간은 다음과 같았다.

| 항목 | Stage 14 진단 |
|---|---:|
| 기간 | 25개월 |
| 포트폴리오 MDD | -23.1921% |
| GLD 손익 기여 | -23.6886% |
| KODEX200 손익 기여 | -1.5445% |
| 평균 GLD 비중 | 49.4702% |
| 평균 GLD macro μ | 월 +0.9540% |
| 직전 6개월 GLD 추세 | 평균 -8.0226% |
| 직전 12개월 GLD 추세 | 평균 -15.4279% |
| 6·12개월 추세가 모두 음수인 달 | 25개월 중 21개월 |

즉 매크로 모델은 금에 계속 양의 기대수익을 부여했지만, 금 자체 가격은 이미 중기·장기 약세였다. 변동성 λ를 더 조절하는 것보다 금 기대수익의 확신을 낮춰야 한다는 피드백을 데이터가 뒷받침했다.

## 신뢰도 필터

각 자산 `i`에 대해 투자월 직전까지의 자체 수익률만 사용한다.

```text
M_i = 0.5 × [sign(직전 6개월 수익률) + sign(직전 12개월 수익률)]
```

따라서 `M_i`는 다음 세 값 중 하나다.

- `+1`: 6개월·12개월 추세가 모두 상승
- `0`: 두 기간의 방향이 충돌
- `-1`: 6개월·12개월 추세가 모두 하락

중립 기대수익은 해당 월 네 자산의 매크로 기대수익 단순 평균이다.

```text
neutral_mu = mean(macro_mu_KODEX200, macro_mu_BOND,
                  macro_mu_GLD, macro_mu_USO)

macro_direction_i = sign(macro_mu_i - neutral_mu)

confidence_i
  = [1 + macro_direction_i × M_i] / 2

filtered_macro_mu_i
  = neutral_mu
    + confidence_i × (macro_mu_i - neutral_mu)
```

- 매크로 상대 전망과 6·12개월 추세가 모두 일치: `confidence=1`
- 두 추세 중 하나만 일치: `confidence=0.5`
- 두 추세가 모두 매크로 전망과 충돌: `confidence=0`

추세 수익률을 기대수익에 직접 더하지 않는다. 충돌하는 매크로 상대 전망을 네 자산의 중립 기대수익으로 축소할 뿐이다. `beta`, 추세 배수, 임계값 또는 학습된 계수는 없다.

VKOSPI/VIX6 스트레스 기대수익 조정은 필터링 뒤에 Stage 14와 똑같이 더한다.

## 바꾸지 않은 조건

- 하방 위험회피 `λ=1` 고정
- 기존 조건부 공분산 추정
- 연 변동성 13% catastrophe guard
- 90% CDaR 16% guard
- 기존 거래비용
- 월별 SLSQP 100% 배분
- 가중치 범위 0~100%, 합계 100%
- 레버리지·현금·공매도 없음
- 단일자산 50% hard cap 없음
- HHI 집중도 벌점 없음
- Dynamic λ·Sigma·ES 및 사후 오버레이 없음

이번 실험은 추세 신뢰도 필터 하나만 검증한다. 집중도 벌점은 같은 실행에 섞지 않았다.

## 성과

거래비용 차감 후 결과이며 2026년 8월은 완결된 월간 자산수익률이 없어 2026년 7월까지 측정했다.

### 전체 구간: 2007-04~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|---:|
| Stage 14 고정 λ | 11.0041% | 13.3417% | 0.8512 | -23.1921% | 0.4745 |
| Stage 19 추세 신뢰도 | 9.7576% | 11.9747% | 0.8392 | -15.8979% | 0.6138 |
| 변화 | **-1.2465%p** | -1.3670%p | **-0.0120** | **+7.2941%p** | +0.1393 |

### 고정 검증 구간: 2018-01~2026-07

| 전략 | CAGR | 변동성 | Sharpe | MDD |
|---|---:|---:|---:|---:|
| Stage 14 고정 λ | 16.1276% | 14.2741% | 1.1229 | -15.8980% |
| Stage 19 추세 신뢰도 | 14.1417% | 12.8226% | 1.0997 | -15.8979% |

### 사전 성공 기준

| 기준 | 결과 |
|---|---:|
| CAGR ≥ 10.5% | 실패: 9.7576% |
| Sharpe ≥ 0.85 | 실패: 0.8392 |
| MDD ≥ -18% | 통과: -15.8979% |
| 레버리지 0 | 통과 |
| Hard regime 0 | 통과 |

MDD 하나만 통과했으므로 종합 채택 기준은 실패다.

## 금 문제에는 정확히 작동했다

2012-10~2014-10 구간에서:

| 항목 | Stage 14 | Stage 19 |
|---|---:|---:|
| 평균 GLD 비중 | 49.47% | 19.27% |
| GLD 평균 confidence | 해당 없음 | 0.08 |
| 해당 낙폭 | -23.19% | -11.22% |
| GLD 손실 기여 | -23.69% | -8.40% |

금의 두 추세가 모두 음수인 21개월에는 낙관적인 금 매크로 전망 대부분을 중립값으로 축소했다. 이로써 피드백이 지목한 구조적 오류와 2012~2014 낙폭은 크게 완화됐다.

## 그런데 왜 종합 성과는 실패했나

신뢰도를 낮추면 자산별 매크로 기대수익 차이가 줄어든다. 기대수익이 비슷해지자 SLSQP는 공분산과 하방 위험이 낮은 채권을 선택했다. 그 결과 금 집중은 줄었지만 **채권 집중이라는 새로운 문제가 생겼다.**

| 단일자산 비중 | Stage 14 해당 월 | Stage 19 해당 월 | Stage 19 주된 자산 |
|---|---:|---:|---|
| 50% 초과 | 128 | 104 | GLD 55, BOND 38 |
| 60% 초과 | 16 | 39 | BOND 25 |
| 70% 초과 | 12 | 37 | BOND 25 |
| 80% 초과 | 10 | 35 | BOND 25 |
| 90% 초과 | 2 | 22 | 주로 BOND |

즉 단순한 `50% 초과` 빈도는 감소했지만 더 심한 `80~90% 집중`은 오히려 늘었다. 필터가 “잘못된 금 전망”은 고쳤지만 기대수익을 중립화한 뒤 optimizer가 저위험 자산으로 쏠리는 Mean-Variance 특성까지 고치지는 못했다. CAGR과 Sharpe 하락은 이 방어적 채권 집중의 대가다.

이 결과를 보고 신뢰도 0을 0.2로 바꾸거나 6개월·12개월 가중치를 조정하면 사후 파라미터 탐색이 된다. 따라서 하지 않았다.

## 판단

1. 자산 자체 추세를 기대수익 신뢰도에 반영해야 한다는 피드백은 타당하다.
2. 해당 필터는 2012~2014 금 문제와 MDD를 실질적으로 개선했다.
3. 하지만 신뢰도 필터만으로는 수익률과 집중도를 동시에 해결하지 못했다.
4. Stage 14를 자동 교체하지 않는다.
5. 다음 독립 가설로 진행한다면 피드백에서 분리해 둔 `HHI soft concentration penalty` 또는 자산군 확대를 별도 Stage에서 검증해야 한다. 이번 성과를 이용해 벌점 계수를 고르면 안 된다.

## 파일

- `asset_trend_confidence_slsqp.py`: 전체 구현·진단·백테스트
- `outputs/stage14_static_recomputed_monthly.csv`: 동일 데이터로 재실행한 기준 경로
- `outputs/asset_trend_confidence_monthly.csv`: 월별 Stage 19 경로와 모든 추세·confidence
- `outputs/performance_comparison.csv`: 전체/고정 검증구간 성과
- `outputs/stage14_concentration_diagnostic.csv`: 232개월 비중, 3·6·12개월 후 성과, macro μ와 예측오차
- `outputs/stage14_gold_concentration_diagnostic.csv`: GLD가 50%를 초과한 달만 분리한 진단
- `outputs/stage14_concentration_threshold_summary.csv`: 50·60·70·80% 집중 진단
- `outputs/trend_strategy_concentration_diagnostic.csv`: Stage 19 집중 진단
- `outputs/drawdown_episode_attribution.csv`: 최대 낙폭 구간별 자산 기여
- `outputs/validation_report.json`: 가설, 성과, gate, 제약·인과성 감사
- `tests/test_stage19_asset_trend_confidence.py`: 계산식·미래정보·성과·실패 조건 회귀 테스트

## 실행

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m strategies.stage19_asset_trend_confidence.asset_trend_confidence_slsqp
```

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m pytest tests/test_stage19_asset_trend_confidence.py -q
```
