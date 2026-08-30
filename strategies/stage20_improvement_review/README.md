# Stage20 개선 실험 종합 판정

## 결론

Stage20 원본은 수정하지 않고 후보를 각각 독립 폴더에 구현했다. 전체 2007-04~2026-07에서 **CAGR 10% 이상, Sharpe 1 이상, MDD -14.0148%보다 악화되지 않음**을 동시에 만족한 후보는 없었다.

다만 Stage24는 Stage20 대비 CAGR, Sharpe, MDD를 모두 개선한 유일한 파레토 개선안이다. 현재 자료에서 배포 후보를 하나 고른다면 Stage24가 가장 낫지만, 절대 목표를 달성했다고 표현해서는 안 된다.

## 전체 기간 성과

| 전략 | 핵심 변경 | CAGR | Sharpe | MDD | 판정 |
|---|---|---:|---:|---:|---|
| Stage20 | 동결 기준 | 9.3974% | 0.9865 | -14.0148% | 기준 |
| Stage21 | 단방향 기대수익 축소 | 9.5985% | 0.9613 | -18.0468% | 기각 |
| Stage22 | K-Ratio 방향·RSI 크기 확인 | 9.3507% | 0.9990 | -13.1952% | CAGR 하락 |
| **Stage24** | **KODEX200 K-Ratio 단독 방향** | **9.5182%** | **0.9968** | **-13.6289%** | **유일한 파레토 개선** |
| Stage23 | 상대 ATR 정규화 | 9.6412% | 0.8661 | -16.1238% | 기각 |
| Stage25 | 전 자산 충돌 거부권 | 10.1253% | 0.9750 | -17.7915% | 위험 악화 |
| Stage26 | 주식만 충돌 거부권 | 10.2270% | 0.9752 | -16.3631% | 위험 악화 |
| Stage27 | K-Ratio 주식 거부권 결합 | 10.0036% | 0.9737 | -15.9480% | 위험 악화 |

Stage25~27은 CAGR 10% 부근까지 위험예산을 풀 수 있다는 것은 보여줬다. 하지만 CAGR 상승이 Sharpe 개선이 아니라 변동성과 2012~2013년 낙폭 확대를 동반했기 때문에 목표에 맞는 개선이 아니다.

## Stage24에서 실제로 달라진 것

| 항목 | Stage20 | Stage24 | 변화 |
|---|---:|---:|---:|
| KODEX200 평균 비중 | 25.39% | 26.42% | +1.03%p |
| 채권 평균 비중 | 36.50% | 37.48% | +0.98%p |
| 금 평균 비중 | 37.83% | 35.82% | -2.01%p |
| KODEX200 연환산 산술 기여 | 2.98%p | 3.27%p | +0.29%p |
| 평균 월 회전율 | 2.68% | 2.79% | +0.11%p |

14일 가격·거래량 RSI가 126일 K-Ratio의 부호를 희석하거나 뒤집지 못하게 하면서 주식 기여도가 소폭 회복됐고, 금 집중과 2013년 낙폭은 줄었다. 새로운 기간, 임계값, 가중치는 넣지 않았다.

## 고정 검증 구간

2018-01~2026-07에서 Stage24는 CAGR 13.1602%, Sharpe 1.2585%, MDD -11.4289%였다. Stage20은 각각 12.9392%, 1.2503, -11.4190%다. 수익과 Sharpe는 개선됐지만 MDD는 0.0099%p 나빠 사실상 같았다.

## 불확실성 판정

12개월 이동블록을 사용한 2,000회 paired bootstrap은 Stage24와 Stage20의 월수익률 차이를 같은 표본 순서로 재표집했다.

| 지표 차이(Stage24-Stage20) | 점추정 | 5~95% 구간 | 개선 확률 |
|---|---:|---:|---:|
| CAGR | +0.1208%p | -0.2881~+0.3927%p | 54.3% |
| Sharpe | +0.0103 | -0.0210~+0.0358 | 65.0% |
| MDD | +0.3859%p | -0.2278~+1.3954%p | 69.8% |

구간에 0이 포함되므로 Stage24의 우월성이 통계적으로 확정됐다고 볼 수 없다. 또한 Stage24~27은 1차 결과를 확인한 뒤 만든 탐색적 후속안이다. bootstrap은 불확실성 설명용이며 사후 선택의 유의확률로 해석하지 않는다.

## 폴더 분리

- `../stage21_one_sided_confidence`: 단방향 필터
- `../stage22_k_ratio_primary`: K-Ratio·RSI 계층화
- `../stage23_relative_atr`: 상대 ATR
- `../stage24_equity_k_ratio_only`: 권고 후보
- `../stage25_conflict_only_veto`: 전 자산 위험예산 해제 귀속
- `../stage26_equity_conflict_veto`: 주식 위험예산 해제 귀속
- `../stage27_k_ratio_equity_veto`: 결합 후보

각 폴더는 자체 실행 코드, `outputs`, 검증 보고서를 가진다. 종합 폴더는 전략을 실행하거나 섞지 않고 저장된 결과만 비교한다.

## 재실행

```powershell
$py = 'D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe'
& $py -m strategies.stage20_improvement_review.compare_candidates
```

주요 산출물:

- `outputs/candidate_performance_and_gates.csv`: 절대 성과와 목표 통과 여부
- `outputs/weight_and_return_attribution.csv`: 평균 비중과 자산별 기여도
- `outputs/worst_drawdown_windows.csv`: 최대 낙폭 구간
- `outputs/paired_block_bootstrap.csv`: paired bootstrap 결과
- `outputs/review_report.json`: 최종 기계판독 판정
