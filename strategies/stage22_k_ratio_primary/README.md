# Stage 22 — K-Ratio 중심·RSI 확인

126일 K-Ratio가 KODEX200 방향을 정하고 14일 가격·거래량 RSI는 그 크기만 확인한다. RSI가 중기 추세의 부호를 뒤집지 못하게 한 단일 변경이다.

| 구간 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| 2007-04~2026-07 | 9.3507% | 0.9990 | -13.1952% |
| 2018-01~2026-07 | 12.7955% | 1.2617 | -11.4194% |

Sharpe와 MDD는 개선됐지만 RSI 확인계수가 K-Ratio까지 항상 약화해 CAGR이 낮아졌다. 목표 미달이므로 방어 연구 후보로만 보존한다. 코드는 `k_ratio_primary_slsqp.py`, 결과는 `outputs/k_ratio_primary_monthly.csv`에 있다.
