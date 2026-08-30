# Stage 21 — 단방향 기대수익 필터

Stage20의 대칭 축소 대신, 중립보다 높은 거시 기대수익만 기술 신뢰도로 삭감한다. 중립 이하 전망을 기술신호만으로 올리지 않는 단일 변경이며 Stage20 원본은 수정하지 않았다.

| 구간 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| 2007-04~2026-07 | 9.5985% | 0.9613 | -18.0468% |
| 2018-01~2026-07 | 13.4899% | 1.2456 | -11.3734% |

수익률은 Stage20보다 0.20%p 높아졌지만 전체 MDD가 4.03%p 악화되어 기각한다. 코드는 `one_sided_confidence_slsqp.py`, 월별 결과는 `outputs/one_sided_confidence_monthly.csv`에 있다.

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m strategies.stage21_one_sided_confidence.one_sided_confidence_slsqp
```
