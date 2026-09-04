# Stage55 Growth Inflation Bayesian Z

Stage55는 Stage52의 Bayesian z-score betting-size allocator와 `GSG` universe를
유지하되, BayesianRidge regression feature를 4국면 score가 아니라 세 개의 원천
macro score로 바꾼 실험이다. 기본 실행은 risk-free를 차감하지 않는 단순수익률
기준이다.

```text
월별 KRW 자산수익률
  + 일별 KRW 자산수익률
  + g_score
  + i_score
  + g_i_score = g_score * i_score
  -> BayesianRidge 월 기대수익률과 predictive std
  -> z_score = expected_return / predictive_std
  -> betting_size = clip(2 * NormalCDF(z_score) - 1, 0, 1)
  -> weight = betting_size / sum(betting_size)
  -> 다음 달 실현수익률
```

## Stage52 대비 변경점

- 자산 universe는 `KODEX200`, `BOND`, `GLD`, `GSG` 그대로 둔다.
- allocator도 Stage52와 같은 Bayesian z-score betting-size 규칙을 사용한다.
- regression feature는 `p_Overheating`, `p_Slowdown`, `p_Stagflation` 대신
  `g_score`, `i_score`, `g_i_score` 세 개만 사용한다.
- `g_i_score`는 단순 곱이며, rank 변환은 하지 않는다.
- `p_Goldilocks`, `p_Overheating`, `p_Slowdown`, `p_Stagflation`은 진단용으로
  `monthly_results.csv`에 남긴다.
- 기본 실행은 simple return이며, CD(91일) excess return은 옵션으로만 켠다.

## 저장 파일

- `outputs/historical_monthly_returns.csv`: allocator 입력으로 사용한 월별 자산수익률
- `outputs/historical_daily_returns.csv`: 공분산 입력으로 사용한 일별 자산수익률
- `outputs/monthly_cd_risk_free_returns.csv`: 월별 CD(91일) risk-free return. 기본 OFF에서는 0으로 저장
- `outputs/monthly_results.csv`: 월별 예측, z-score, betting size, 비중, macro score, 실현수익률
- `outputs/regime_betas.csv`: BayesianRidge macro score beta long table
- `outputs/performance.csv`: 성과 요약
- `outputs/validation_report.json`: 인과성·유한값·long-only·완전투자 검증과 위험 진단
- `outputs/weights_full_common.csv`, `outputs/weights_full_common.png`: 전체 구간 투자비중
- `outputs/weights_locked_2018_2026.csv`, `outputs/weights_locked_2018_2026.png`: locked 구간 투자비중
- `outputs/pnl_comparison_full_common.csv`, `outputs/pnl_comparison_full_common.png`: 전체 구간 전략 vs 개별 자산 누적 PnL
- `outputs/pnl_comparison_locked_2018_2026.csv`, `outputs/pnl_comparison_locked_2018_2026.png`: locked 구간 전략 vs 개별 자산 누적 PnL
- `outputs/individual_asset_performance.csv`: 전략과 개별 자산 100% buy-and-hold 성과 비교

## 실행

```bash
python -m strategies.stage55_growth_inflation_bayesian_z.explainable_regime_allocator
```

CD(91일) 초과수익률 실험을 다시 켜려면 옵션을 추가한다.

```bash
python -m strategies.stage55_growth_inflation_bayesian_z.explainable_regime_allocator --enable-risk-free
```

현재 공유 캐시에 `GSG`가 없으면 먼저 `yfinance`가 설치된 환경에서 다음을 실행해야 한다.

```bash
python -m strategies.stage55_growth_inflation_bayesian_z.explainable_regime_allocator --refresh-monthly-market-cache
```
