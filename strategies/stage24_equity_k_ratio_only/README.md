# Stage 24 — KODEX200 K-Ratio 단독 방향

KODEX200 방향 입력에서 월간 예측 IC가 약했던 14일 가격·거래량 RSI의 투표권을 제거했다. 두 RSI는 출력과 진단에는 남기고, 126일 K-Ratio만 거시전망 신뢰도를 결정한다. 기간·계수·임계값은 새로 탐색하지 않았다.

| 구간 | CAGR | Sharpe | MDD |
|---|---:|---:|---:|
| 2007-04~2026-07 | **9.5182%** | **0.9968** | **-13.6289%** |
| 2018-01~2026-07 | **13.1602%** | **1.2585** | -11.4289% |

전체 구간에서 Stage20 대비 CAGR +0.1208%p, Sharpe +0.0103, MDD +0.3859%p로 세 지표가 함께 개선된 유일한 후보다. 다만 CAGR 10%·Sharpe 1이라는 절대 목표에는 미달했고, 2018년 이후 MDD는 0.0099%p 나빠 사실상 동일했다.

이 후보는 1차 실험 결과를 본 뒤 만든 탐색적 후속안이다. 신규 표본 확인 전까지는 연구용 권고안이지 확정된 우월 전략이 아니다. 코드는 `equity_k_ratio_only_slsqp.py`, 결과는 `outputs/equity_k_ratio_only_monthly.csv`에 있다.

```powershell
& D:\Programming\python_example\Arcana\.venv-llama\Scripts\python.exe -m strategies.stage24_equity_k_ratio_only.equity_k_ratio_only_slsqp
```
