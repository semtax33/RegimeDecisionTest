# Stage53 KOSPI Feature Bayesian Z

Stage53은 Stage52의 `GSG` universe와 Bayesian z-score
betting-size allocator를 유지하되, `KODEX200` 기대수익 regression에만
fundamental/technical/option feature를 추가하는 실험이다.

```text
월별 KRW 자산수익률
  + 일별 KRW 자산수익률
  + 거시 국면 score
  + KODEX200 전용 feature
  -> BayesianRidge 월 기대수익률과 predictive std
  -> z_score = expected_return / predictive_std
  -> betting_size = clip(2 * NormalCDF(z_score) - 1, 0, 1)
  -> weight = betting_size / sum(betting_size)
  -> 다음 달 실현수익률
```

## Stage52 대비 변경점

- `KODEX200`, `BOND`, `GLD`, `GSG` 네 자산은 그대로 둔다.
- 모든 자산은 기존처럼 regime score를 regression feature로 사용한다.
- `KODEX200` regression에만 아래 7개 feature를 추가한다.
- `BOND`, `GLD`, `GSG` regression에는 추가 feature를 넣지 않는다.
- 기본 실행에서는 risk-free return을 차감하지 않은 단순 자산수익률을 사용한다.
- 공분산은 비중 산출에 직접 쓰지 않고, 사후 위험 진단에만 사용한다.

## KODEX200 Feature

| Feature | Source |
| --- | --- |
| `kospi_eps_revision_z` | `stage35_earnings_credit_fundamentals/outputs/monthly_earnings_credit_signals.csv` |
| `kospi_valuation_gap_z` | `stage35_earnings_credit_fundamentals/outputs/monthly_earnings_credit_signals.csv` |
| `kospi_credit_widening_z` | `stage35_earnings_credit_fundamentals/outputs/monthly_earnings_credit_signals.csv` |
| `kospi_k_score` | `stage24_equity_k_ratio_only/outputs/monthly_technical_signals.csv` |
| `kospi_option_direction_score` | `stage28_option_directional_surface/outputs/monthly_option_direction_signals.csv` |
| `kospi_abnormal_bear_pressure_fast` | `stage30_abnormal_surface_erp/outputs/monthly_option_alpha_signals.csv` |
| `kospi_abnormal_bear_pressure_slow` | `stage30_abnormal_surface_erp/outputs/monthly_option_alpha_signals.csv` |

## 저장 파일

- `outputs/historical_monthly_returns.csv`: allocator 입력으로 사용한 월별 자산수익률
- `outputs/historical_daily_returns.csv`: 공분산 입력으로 사용한 일별 자산수익률
- `outputs/monthly_cd_risk_free_returns.csv`: 월별 CD(91일) risk-free return. 기본 OFF에서는 0으로 저장
- `outputs/kospi_feature_inputs.csv`: KODEX200 regression에 사용한 월별 feature
- `outputs/monthly_results.csv`: 월별 예측, z-score, betting size, 비중, feature, 실현수익률
- `outputs/regime_betas.csv`: BayesianRidge regime/KOSPI feature beta long table
- `outputs/performance.csv`: 성과 요약
- `outputs/validation_report.json`: 인과성·유한값·long-only·완전투자 검증과 위험 진단
- `outputs/weights_full_common.csv`, `outputs/weights_full_common.png`: 전체 구간 투자비중
- `outputs/weights_locked_2018_2026.csv`, `outputs/weights_locked_2018_2026.png`: locked 구간 투자비중
- `outputs/pnl_comparison_full_common.csv`, `outputs/pnl_comparison_full_common.png`: 전체 구간 전략 vs 개별 자산 누적 PnL
- `outputs/pnl_comparison_locked_2018_2026.csv`, `outputs/pnl_comparison_locked_2018_2026.png`: locked 구간 전략 vs 개별 자산 누적 PnL
- `outputs/individual_asset_performance.csv`: 전략과 개별 자산 100% buy-and-hold 성과 비교

## 실행

프로젝트 루트에서 실행한다.

```bash
python -m strategies.stage53_kospi_feature_bayesian_z.explainable_regime_allocator
```

CD(91일) 초과수익률 실험을 다시 켜려면 옵션을 추가한다.

```bash
python -m strategies.stage53_kospi_feature_bayesian_z.explainable_regime_allocator --enable-risk-free
```

현재 공유 캐시에 `GSG`가 없으면 먼저 `yfinance`가 설치된 환경에서 다음을 실행해야 한다.

```bash
python -m strategies.stage53_kospi_feature_bayesian_z.explainable_regime_allocator --refresh-monthly-market-cache
```

## 해석

Stage53은 "KODEX200에는 주식시장 전용 정보가 더 있어야 한다"는 가설을 검증한다.
따라서 성과가 좋아져도 전체 자산 universe에 같은 feature를 확장했다는 뜻은 아니며,
KODEX200의 기대 초과수익률 예측에 추가 정보가 있었는지를 보는 실험으로 읽어야 한다.
