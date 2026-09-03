# Stage56 Rolling Z Growth Inflation Bayesian Z

Stage56은 Stage52의 Bayesian z-score betting-size allocator와 `GSG` universe를
유지하되, macro regression feature를 rolling z-score 기반 성장/물가 score로 바꾼
실험이다. 기본 실행은 risk-free를 차감하지 않는 단순수익률 기준이다.

```text
월별 KRW 자산수익률
  + 일별 KRW 자산수익률
  + g_score = mean(GDP YoY z72, Export YoY z36, BSI z24)
  + i_score = mean(CPI YoY z36, PPI YoY z36, Import Price YoY z36)
  + g_i_score = g_score * i_score
  -> BayesianRidge 월 기대수익률과 predictive std
  -> z_score = expected_return / predictive_std
  -> betting_size = clip(2 * NormalCDF(z_score) - 1, 0, 1)
  -> weight = betting_size / sum(betting_size)
  -> 다음 달 실현수익률
```

## Stage55 대비 변경점

- regression feature 이름은 `g_score`, `i_score`, `g_i_score` 그대로 유지한다.
- Stage55의 `g_score`, `i_score`는 expanding percentile rank 평균이었다.
- Stage56의 `g_score`, `i_score`는 rolling z-score 평균이다.
- `g_i_score`는 단순 곱이며 rank 변환은 하지 않는다.
- z-score는 각 지표별 rolling window 평균과 표준편차로 계산하고 `[-3, 3]`으로 clip한다.
- `p_Goldilocks`, `p_Overheating`, `p_Slowdown`, `p_Stagflation`은 진단용으로
  `monthly_results.csv`에 남긴다.

## Rolling Window

| Block | Indicator | Window |
| --- | --- | --- |
| Growth | `GDP_YoY` | 72 months |
| Growth | `Export_YoY` | 36 months |
| Growth | `BSI` | 24 months |
| Inflation | `CPI_YoY` | 36 months |
| Inflation | `PPI_YoY` | 36 months |
| Inflation | `ImportPrice_YoY` | 36 months |

## 성과

| Period | CAGR | Volatility | Sharpe | MDD | Final Multiple |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_common | 9.22% | 10.11% | 0.925 | -16.97% | 5.341 |
| locked_2018_2026 | 16.10% | 10.83% | 1.440 | -14.93% | 3.601 |

Stage55의 expanding-rank `g/i/g*i` 실험과 비교하면 full_common Sharpe는 `-0.062`,
locked_2018_2026 Sharpe는 `-0.050` 낮았다. Stage52의 regime probability
baseline과 비교하면 full_common Sharpe는 `-0.114`, locked_2018_2026 Sharpe는
`-0.057` 낮았다. 다만 locked_2018_2026 CAGR은 Stage55보다 `+0.904%p`,
Stage52보다 `+0.546%p` 높았다.

## 저장 파일

- `outputs/historical_monthly_returns.csv`: allocator 입력으로 사용한 월별 자산수익률
- `outputs/historical_daily_returns.csv`: 공분산 입력으로 사용한 일별 자산수익률
- `outputs/macro_score_inputs.csv`: rolling z-score 기반 macro score
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
python -m strategies.stage56_rolling_z_growth_inflation_bayesian_z.explainable_regime_allocator
```

CD(91일) 초과수익률 실험을 다시 켜려면 옵션을 추가한다.

```bash
python -m strategies.stage56_rolling_z_growth_inflation_bayesian_z.explainable_regime_allocator --enable-risk-free
```
