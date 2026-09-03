# Stage52 Bayesian Z Betting Size

Stage52는 Stage51의 `GSG` universe와 BayesianRidge 기대수익 추정은 유지하되,
SLSQP 최적화를 제거하고 Bayesian forecast의 신호대잡음비로 직접 long-only 비중을
정하는 실험이다.

```text
원자료
  -> 월별 KRW 자산수익률 - CD(91일) risk-free return
  -> 거시 국면 score + 일별 KRW excess return
  -> BayesianRidge 월 기대수익률과 predictive std
  -> z_score = expected_return / predictive_std
  -> betting_size = clip(2 * NormalCDF(z_score) - 1, 0, 1)
  -> weight = betting_size / sum(betting_size)
  -> 다음 달 실현수익률
```

## Stage51 대비 변경점

- `KODEX200`, `BOND`, `GLD`, `GSG` 네 자산은 그대로 둔다.
- BayesianRidge regression과 Ledoit-Wolf covariance 추정은 그대로 계산한다.
- 기본 실행에서는 월별/일별 자산수익률에서 `CD(91일)` risk-free return을 차감한
  excess return을 사용한다.
- Stage51의 SLSQP 목적함수와 Vol/CDaR 제약 최적화는 사용하지 않는다.
- 공분산은 비중 산출에 직접 쓰지 않고, 사후 위험 진단에만 사용한다.
- 모든 `betting_size`가 0이면 정규화할 수 없으므로 equal-weight로 후퇴하고
  `used_equal_weight_fallback=True`를 기록한다.

## 저장 파일

- `outputs/historical_monthly_returns.csv`: allocator 입력으로 사용한 월별 자산 초과수익률
- `outputs/monthly_cd_risk_free_returns.csv`: 월별 CD(91일) risk-free return
- `outputs/monthly_results.csv`: 월별 예측, z-score, betting size, 비중, 실현수익률
- `outputs/weights_full_common.csv`: 전체 공통 구간 월별 투자비중
- `outputs/weights_full_common.png`: 전체 공통 구간 투자비중 stacked area chart
- `outputs/weights_locked_2018_2026.csv`: 2018년 이후 locked 구간 월별 투자비중
- `outputs/weights_locked_2018_2026.png`: 2018년 이후 locked 구간 투자비중 stacked area chart
- `outputs/weight_summary_by_period.csv`: 구간별·자산별 평균/중앙값/최소/최대/마지막 비중 요약
- `outputs/pnl_comparison_full_common.csv`: 전체 공통 구간 전략 vs 개별 자산 누적 NAV/PnL
- `outputs/pnl_comparison_full_common.png`: 전체 공통 구간 전략 vs 개별 자산 누적 PnL chart
- `outputs/pnl_comparison_locked_2018_2026.csv`: 2018년 이후 locked 구간 전략 vs 개별 자산 누적 NAV/PnL
- `outputs/pnl_comparison_locked_2018_2026.png`: 2018년 이후 locked 구간 전략 vs 개별 자산 누적 PnL chart
- `outputs/individual_asset_performance.csv`: 전략과 개별 자산 100% buy-and-hold 성과 비교
- `outputs/regime_betas.csv`: BayesianRidge regime beta long table
- `outputs/performance.csv`: 성과 요약
- `outputs/validation_report.json`: 인과성·유한값·long-only·완전투자 검증과 위험 진단

## 실행

프로젝트 루트에서 실행한다.

```bash
python -m strategies.stage52_bayesian_z_betting_size.explainable_regime_allocator
```

기존처럼 단순 자산수익률을 쓰려면 risk-free 차감을 끈다.

```bash
python -m strategies.stage52_bayesian_z_betting_size.explainable_regime_allocator --disable-risk-free
```

현재 공유 캐시에 `GSG`가 없으면 먼저 `yfinance`가 설치된 환경에서 다음을 실행해야 한다.

```bash
python -m strategies.stage52_bayesian_z_betting_size.explainable_regime_allocator --refresh-monthly-market-cache
```

## 해석

이 전략은 "예상 수익률이 크고 예측 불확실성이 작을수록 더 크게 베팅한다"는 규칙을
검증하기 위한 단순 실험이다. `predictive_std`는 BayesianRidge의 forecast uncertainty
이며, 공분산 기반 포트폴리오 위험과는 다르다. 따라서 Stage52의 Vol/CDaR 값은 제약
조건이 아니라 사후 진단으로 읽어야 한다.
