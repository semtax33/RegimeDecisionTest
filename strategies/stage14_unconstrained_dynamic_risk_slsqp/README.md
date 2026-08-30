# Stage14: 단일자산 상한 없는 동적 위험회피 SLSQP

이 폴더는 업로드한
`economic_regime_asset_allocation_backtest_multi_asset.ipynb`에 대한 피드백을
별도 실험으로 구현한 것이다. 원본 노트북과 Stage13 폴더는 수정하지 않았다.

## 결론

피드백의 수학적 지적은 맞다. 현금이 없고 네 자산 비중의 합을 항상 100%로
유지하면 `RiskScale <= 1`을 네 자산에 일괄 적용할 수 없다.

    scaled_weight_i = k * weight_i

를 다시 합계 1로 정규화하면

    normalized_weight_i
      = k * weight_i / sum(k * weight)
      = weight_i

가 되어 scaling 효과가 사라진다. 따라서 Stage14는 변동성 배율을 사용하지
않는다. 대신 VKOSPI/VIX6 스트레스가 높을수록 SLSQP 목적함수의 하방위험
회피계수 `lambda`를 높이고, 예상 연변동성 13%와 CDaR 16%를 사전 비상 제약으로
유지한다.

사용자의 추가 요청에 따라 **단일자산 과반금지 조건도 완전히 해제했다.**

    0 <= w_i <= 1
    sum(w_i) = 1

따라서 한 자산 100%도 수학적으로 가능하다. 현금, 차입, 음의 비중은 없다.

## 원본 노트북에서 바뀐 점

원본 노트북은 다음 Hard 배분을 사용한다.

| 국면 | 원본 목표비중 |
|---|---|
| Goldilocks | KODEX200 100% |
| Overheating | USO 100% |
| Slowdown | KODEX200 60% + 채권 40% |
| Stagflation | GLD 100% |

Stage14에는 이 표가 없다. GDP·수출·BSI와 CPI·PPI·수입물가로 만든 네 soft
국면확률은 기대수익과 공분산을 추정하는 정보로만 사용한다. VKOSPI/VIX6도
비중을 직접 수정하지 않는다. 최종 비중은 매월 SLSQP가 100% 결정한다.

원본 노트북에는 volatility targeting이나 SLSQP가 실제로 구현되어 있지 않고,
Hard regime mapping이 곧 최종 비중이다. 따라서 기존 비중에 단순 배율을 붙인
것이 아니라 배분 엔진 자체를 바꿨다.

## 목적함수

Stage14 최종안의 월 목적함수는 다음과 같다.

    maximize
        expected_return
        - 0.5 * conditional_variance
        - lambda_t * downside_semivariance
        - estimated_transaction_cost

    lambda_t = 1 + stress_t

`expected_return - 0.5 * variance`는 기대 로그성장률의 월 근사다. 하방
반분산은 과거 포트폴리오 월수익 중 음수만 제곱한 평균이다.

`lambda_0=1`은 하방 반분산이 공분산과 같은 월수익률 제곱 단위를 갖기 때문에
둔 기준값이다. `alpha=1`은 0~1 empirical percentile인 stress가 최댓값에
도달했을 때 위험회피를 정확히 두 배로 만든다는 피드백을 그대로 사용한 것이다.
두 숫자를 백테스트 탐색으로 고르지 않았다.

비교를 위해 `lambda=1`로 고정한 경로도 함께 계산한다. 이 경로와 최종안의
차이가 오직 동적 위험회피의 증분효과다.

## 제약

    sum(w) = 1
    0 <= each weight <= 1
    ex-ante annual volatility <= 13%
    historical CDaR(90%) >= -16%

- 자산별 50% 상한: 없음
- 현금: 없음
- 차입: 없음
- 변동성 배율: 없음
- 사후 VKOSPI 비중 overlay: 없음
- Hard regime 비중: 없음
- 레버리지: 없음

13% 상한은 목표 변동성 10%를 맞추는 scaling 장치가 아니라 폭주 방지용
catastrophe guard다. 다만 최종안에서 232개월 중 71개월, CDaR은 82개월
구속됐다. 따라서 실제로는 안전벨트가 자주 작동한다. 이를 숨기고 “평상시에는
전혀 작동하지 않는다”고 설명해서는 안 된다.

## 성과

### 2007-04~2026-07, 232개월

| 전략 | CAGR | Sharpe | MDD | 실현 변동성 | 월평균 turnover |
|---|---:|---:|---:|---:|---:|
| Stage10 | 6.86% | **1.329** | **-6.88%** | 5.10% | 3.05% |
| Stage13, 단일자산 50% 상한 | **11.50%** | 0.901 | -21.32% | 13.07% | 2.70% |
| Stage14, 상한 없음·고정 lambda | 11.00% | 0.851 | -23.19% | 13.34% | 2.98% |
| **Stage14, 상한 없음·동적 lambda** | **10.76%** | **0.846** | **-22.97%** | **13.13%** | **2.90%** |

단일자산 상한을 제거한 결과 Stage13보다 CAGR, Sharpe, MDD가 모두 나빠졌다.
이것은 코드 오류가 아니라 집중위험을 허용한 결과다. 사용자가 요청한 조건은
그대로 유지하되, 운영상 개선이라고 해석하지는 않는다.

동적 lambda는 같은 무상한·고정 lambda 경로 대비 다음 효과가 있었다.

- CAGR: 11.00% → 10.76%, **-0.25%p**
- Sharpe: 0.851 → 0.846, **-0.005**
- MDD: -23.19% → -22.97%, **+0.22%p 개선**
- 실현 변동성: 13.34% → 13.13%
- turnover: 2.98% → 2.90%

전체기간에는 수익 희생에 비해 위험개선이 작았다. 따라서 동적 위험회피를
적용할 수는 있지만, 이 결과만으로 유효성이 충분하다고 볼 수 없다.

### 2018-01~2026-07, 103개월

| 전략 | CAGR | Sharpe | MDD | 실현 변동성 |
|---|---:|---:|---:|---:|
| Stage13, 50% 상한 | 16.29% | 1.118 | -15.90% | 14.49% |
| Stage14, 상한 없음·고정 lambda | 16.13% | 1.123 | -15.90% | 14.27% |
| **Stage14, 상한 없음·동적 lambda** | **15.78%** | **1.130** | **-15.90%** | **13.87%** |

최근 구간에서는 동적 lambda가 Sharpe를 0.007 높이고 변동성을 0.40%p
낮췄지만, CAGR은 0.35%p 낮아졌고 MDD는 바뀌지 않았다.

## 단일자산 과반금지 해제 결과

최종 경로 232개월 중:

- 한 자산 비중이 50%를 넘은 달: 119개월
- 한 자산 비중이 90%를 넘은 달: 2개월
- 최대 단일자산 비중: 92.69%
- 평균 최대 자산비중: 51.46%
- 실질적인 100% 집중: 0개월

평균 비중은 KODEX200 34.31%, 채권 19.03%, 금 46.36%, USO 0.30%다.
상한은 없지만 13% 변동성·16% CDaR 제약과 위험벌점 때문에 완전한 100%
집중은 발생하지 않았다. 그렇다고 집중위험이 사라진 것은 아니다.

## Dynamic Risk Attribution

고정 lambda를 base, 동적 lambda를 overlay로 놓고 비교했다.

전체기간 동적 위험회피는 192개월에 위험자산 비중을 더 줄였다. KODEX200
수익률 하위 10%인 24개월 중 18개월에 위험자산을 줄였지만, 주식이 오른
달에도 위험자산을 줄인 false positive가 110개월이었다. 이 때문에 MDD는
0.22%p만 개선되고 CAGR은 0.25%p 희생됐다.

2018년 이후에는 11개 급락월 중 8개를 방어 방향으로 반영하고 3개를 놓쳤다.
그러나 MDD 자체는 동일했다. 위험 신호가 방향상 작동하는 것과 실제 최악의
누적 낙폭을 줄이는 것은 서로 다른 문제다.

## 업로드 노트북 성과와 직접 비교하지 않은 이유

업로드 노트북에 저장된 결과는 CAGR 18.63%, Sharpe 0.868, MDD -25.25%다.
하지만 실행 시점의 Yahoo Finance 데이터와 2026년 open 가격에 의존하고,
최근 월에는 KODEX200 월수익 +31%, USO -21%, KODEX200 -27% 같은 값이
포함돼 있다. 또한 백테스트 시작월과 거시 계산법도 다르다.

따라서 이 숫자를 Stage14와 한 성과표에 넣으면 전략 차이와 데이터 차이가
섞인다. Stage14 비교표는 동일한 프로젝트 월수익, 동일한 거래비용, 동일한
2007-04~2026-07 구간으로 Stage10·Stage13·Stage14만 비교한다.

## 파일

- `dynamic_risk_slsqp.py`: 조건부 모멘트, 무상한 SLSQP, 백테스트와 귀속분석
- `economic_regime_dynamic_risk_revision.ipynb`: 업로드 노트북 피드백을
  순서대로 실행·설명하는 수정 노트북
- `dynamic_risk_report.html`: 상세 설계와 결과 보고서
- `outputs/no_asset_cap_static_lambda_monthly.csv`: 고정 lambda 대조군
- `outputs/no_asset_cap_dynamic_lambda_monthly.csv`: 최종 동적 lambda 경로
- `outputs/dynamic_risk_attribution.csv`: 월별 동적 위험회피 증분효과
- `outputs/performance_comparison.csv`: 동일기간 성과표
- `outputs/validation_report.json`: 설정·집중도·solver·검증 요약
- `tests/test_stage14_unconstrained_dynamic_risk_slsqp.py`: 회귀 테스트

## 실행

```powershell
& 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe' `
  -m strategies.stage14_unconstrained_dynamic_risk_slsqp.dynamic_risk_slsqp
```

## 판단

피드백의 구조는 현금 없는 완전투자 포트폴리오에 맞으며, 변동성 배율보다
동적 위험벌점이 수학적으로 일관된다. 하지만 이번 데이터에서는 **동적 lambda의
전체기간 비용 대비 효과가 작았고, 단일자산 상한 해제는 세 성과지표를 모두
악화시켰다.**

그러므로 구현은 유지하되, “성과가 더 좋아진 최종안”으로 승격시키지는 않는다.
단일자산 상한을 쓰지 않는 것이 절대 조건이라면 현재 저장된 Stage14가 그 조건을
충족하는 기준 경로다.

