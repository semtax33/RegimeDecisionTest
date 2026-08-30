# Stage 27 — K-Ratio 단독 방향과 주식 충돌 거부권 결합

Stage24의 K-Ratio 단독 KODEX200 방향과 Stage26의 주식 충돌 거부권을 결합했다. 다른 자산의 기술 신뢰도와 ATR 위험 조정은 Stage20 그대로다.

| 구간 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| 2007-04~2026-07 | 10.0036% | 0.9737 | -15.9480% |
| 2018-01~2026-07 | 13.8636% | 1.2260 | -11.4377% |

CAGR 10%에는 간신히 도달했지만 Sharpe와 MDD 조건을 통과하지 못했다. 결합이 각 단일 구조의 약점을 해소하지 못했으므로 기각한다. 코드는 `k_ratio_equity_veto_slsqp.py`, 결과는 `outputs/k_ratio_equity_veto_monthly.csv`에 있다.
