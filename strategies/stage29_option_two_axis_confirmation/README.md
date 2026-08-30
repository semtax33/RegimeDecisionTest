# Stage 29 — Bear Pressure × Option ERP 두 축 확인

Stage28의 `ERP-Bear` 단일 차분이 높은 기대수익과 높은 꼬리위험을 동시에 매수하는 문제를 확인하기 위한 별도 후속 폴더다. Stage28 결과를 본 뒤 만든 탐색적 실험이며 확증 후보가 아니다.

피드백의 네 칸 표를 그대로 적용했다.

| Bear | ERP | 방향 |
|---|---|---|
| 0 이하 | 0 초과 | `+ERP`: bullish |
| 0 초과 | 0 이하 | `-Bear`: bearish |
| 0 초과 | 0 초과 | 0: fear with compensation |
| 0 이하 | 0 이하 | 0: uninformative |

표준화된 0만 경계로 사용하며 임계값을 탐색하지 않았다. Fast 5일과 Slow 20일 방향을 동일가중하고, KODEX200 기대수익만 변경한다.

## 성과

| 전략 | 구간 | CAGR | Sharpe | MDD |
|---|---|---:|---:|---:|
| Stage20 | 2007-04~2026-07 | 9.3974% | 0.9865 | -14.0148% |
| Stage28 단순 ODS | 2007-04~2026-07 | 10.0328% | 0.9779 | -18.0833% |
| **Stage29 두 축** | 2007-04~2026-07 | **9.6326%** | 0.9723 | **-15.3284%** |
| Stage20 | 2018-01~2026-07 | 12.9392% | 1.2503 | -11.4190% |
| Stage29 두 축 | 2018-01~2026-07 | 15.2589% | 1.4163 | -11.0356% |

두 축 확인은 Stage28보다 MDD를 2.75%p 줄였지만 전체 Stage20보다 MDD가 1.31%p 나쁘고 Sharpe도 0.014 낮았다. 따라서 Stage20 대체안으로 채택하지 않는다.

코드는 `two_axis_option_direction_slsqp.py`, 월별 결과는 `outputs/option_two_axis_monthly.csv`, 공식·검증은 `outputs/validation_report.json`에 있다.

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m strategies.stage29_option_two_axis_confirmation.two_axis_option_direction_slsqp
```
