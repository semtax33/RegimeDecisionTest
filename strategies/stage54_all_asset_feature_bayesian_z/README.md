# Stage54 All Asset Feature Bayesian Z

Stage54는 Stage53의 Bayesian z-score betting-size allocator를 유지하되,
`KODEX200`뿐 아니라 `BOND`, `GLD`, `GSG`에도 자산군별 feature를 추가한 실험이다.
기본 실행은 risk-free를 차감하지 않는 단순수익률 기준이다.

```text
월별 KRW 자산수익률
  + 일별 KRW 자산수익률
  + 거시 국면 score
  + 자산별 전용 feature
  -> BayesianRidge 월 기대수익률과 predictive std
  -> z_score = expected_return / predictive_std
  -> betting_size = clip(2 * NormalCDF(z_score) - 1, 0, 1)
  -> weight = betting_size / sum(betting_size)
  -> 다음 달 실현수익률
```

## Stage53 대비 변경점

- 모든 자산은 기존처럼 `p_Overheating`, `p_Slowdown`, `p_Stagflation`을 사용한다.
- `KODEX200`에는 Stage53과 같은 EPS, valuation, credit, K-ratio, option feature를 넣는다.
- `BOND`에는 curve carry/roll-down/rate momentum/slope와 BOND technical feature를 넣는다.
- `GLD`에는 real-yield, FX, gold trend state, GVZ rank, GLD technical feature를 넣는다.
- `GSG`에는 OVX rank를 commodity/oil volatility proxy로 넣고, GSG 자체 lagged momentum/vol rank를 넣는다.
- `BOND`, `GLD`, `GSG` feature까지 모두 BayesianRidge의 expected return regression에 직접 들어간다.

## Feature

| Asset | Feature |
| --- | --- |
| `KODEX200` | `kospi_eps_revision_z`, `kospi_valuation_gap_z`, `kospi_credit_widening_z`, `kospi_k_score`, `kospi_option_direction_score`, `kospi_abnormal_bear_pressure_fast`, `kospi_abnormal_bear_pressure_slow` |
| `BOND` | `bond_carry_5y_monthly_z`, `bond_roll_down_5y_to_3y_monthly_z`, `bond_rate_momentum_5y_monthly_z`, `bond_curve_slope_10y_minus_3y_pctpt_z`, `bond_curve_total_return_proxy_z`, `bond_k_score`, `bond_atr_percentile` |
| `GLD` | `gld_real_yield_support`, `gld_fx_support`, `gld_gold_trend_support`, `gld_gold_composite_state`, `gld_gvz_causal_rank`, `gld_k_score`, `gld_atr_percentile` |
| `GSG` | `gsg_ovx_causal_rank`, `gsg_return_momentum_3m`, `gsg_return_momentum_12m`, `gsg_return_volatility_3m_rank` |

## Source

| Block | File |
| --- | --- |
| KOSPI fundamental | `stage35_earnings_credit_fundamentals/outputs/monthly_earnings_credit_signals.csv` |
| KOSPI/BOND/GLD technical | `stage24_equity_k_ratio_only/outputs/monthly_technical_signals.csv` |
| KOSPI ODS | `stage28_option_directional_surface/outputs/monthly_option_direction_signals.csv` |
| KOSPI abnormal option pressure | `stage30_abnormal_surface_erp/outputs/monthly_option_alpha_signals.csv` |
| BOND curve | `stage37_bond_curve_alpha/outputs/monthly_bond_curve_signals.csv` |
| GLD gold state | `stage38_gold_state_alpha/outputs/monthly_gold_state_signals.csv` |
| GLD/GSG volatility proxy | `stage36_asset_implied_volatility_risk/outputs/monthly_asset_volatility_signals.csv` |
| GSG own momentum/volatility | lagged `historical_monthly_returns.csv` derived inside Stage54 |

## 저장 파일

- `outputs/historical_monthly_returns.csv`: allocator 입력으로 사용한 월별 자산수익률
- `outputs/historical_daily_returns.csv`: 공분산 입력으로 사용한 일별 자산수익률
- `outputs/asset_feature_inputs.csv`: 자산별 regression에 사용한 월별 feature
- `outputs/monthly_results.csv`: 월별 예측, z-score, betting size, 비중, feature, 실현수익률
- `outputs/regime_betas.csv`: BayesianRidge regime/asset feature beta long table
- `outputs/performance.csv`: 성과 요약
- `outputs/validation_report.json`: 인과성·유한값·long-only·완전투자 검증과 위험 진단
- `outputs/weights_full_common.csv`, `outputs/weights_full_common.png`: 전체 구간 투자비중
- `outputs/weights_locked_2018_2026.csv`, `outputs/weights_locked_2018_2026.png`: locked 구간 투자비중
- `outputs/pnl_comparison_full_common.csv`, `outputs/pnl_comparison_full_common.png`: 전체 구간 전략 vs 개별 자산 누적 PnL
- `outputs/pnl_comparison_locked_2018_2026.csv`, `outputs/pnl_comparison_locked_2018_2026.png`: locked 구간 전략 vs 개별 자산 누적 PnL
- `outputs/individual_asset_performance.csv`: 전략과 개별 자산 100% buy-and-hold 성과 비교

## 실행

```bash
python -m strategies.stage54_all_asset_feature_bayesian_z.explainable_regime_allocator
```

CD(91일) 초과수익률 실험을 다시 켜려면 옵션을 추가한다.

```bash
python -m strategies.stage54_all_asset_feature_bayesian_z.explainable_regime_allocator --enable-risk-free
```

현재 공유 캐시에 `GSG`가 없으면 먼저 `yfinance`가 설치된 환경에서 다음을 실행해야 한다.

```bash
python -m strategies.stage54_all_asset_feature_bayesian_z.explainable_regime_allocator --refresh-monthly-market-cache
```

## 해석

Stage54는 "각 자산에는 서로 다른 전용 정보가 있다"는 가설을 테스트한다. feature 수가
늘기 때문에 Stage53보다 설명력은 좋아질 수 있지만, Bayesian shrinkage를 쓰더라도
overfit 위험은 더 커진다. 성과가 좋아지는지뿐 아니라 변동성, MDD, turnover, beta
안정성을 함께 봐야 한다.
