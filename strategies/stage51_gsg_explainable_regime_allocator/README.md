# Stage51 GSG Explainable Regime Allocator

Stage51은 Stage50의 단일 거시 국면 allocator에서 원유 ETF `USO`를 broad commodity
ETF `GSG`로 바꾼 1차 실험 버전이다. 전략 구조, BayesianRidge 기대수익률, Ledoit-Wolf
공분산, 거래비용, long-only 제약, 연 13% 변동성 cap, 90% CDaR 16% cap은 Stage50과
같게 둔다.

```text
원자료
  -> 월별 KRW 자산수익률 + 거시 국면 score + 일별 KRW 수익률
  -> BayesianRidge 월 기대수익률
  -> Ledoit-Wolf 월 공분산
  -> 거래비용·변동성·CDaR 제약을 둔 long-only 최적화
  -> 다음 달 실현수익률
```

## Stage50 대비 변경점

- 자산군: `("KODEX200", "BOND", "GLD", "USO")`에서
  `("KODEX200", "BOND", "GLD", "GSG")`로 변경했다.
- `GLD`와 `GSG`는 모두 USD 자산으로 보고 각 거래일의 `USDKRW`로 원화 환산한다.
- 월간 기대수익률 회귀 target은 `return_GSG`로 바뀐다.
- 일별 공분산 추정에도 `GSG`의 KRW 환산 일별 수익률을 사용한다.
- Stage50의 USO 성과표는 복사하지 않았다. Stage51 성과는 Stage51을 새로 실행한 뒤
  `outputs/performance.csv`에서 확인해야 한다.

## 데이터 주의

현재 공유 캐시 `cache/market_daily.csv`와 `cache/regime_lightgbm_ohlcv.csv`에는
기본적으로 `GSG`가 없을 수 있다. 이 경우 먼저 다음 명령으로 `market_daily.csv`에
`GSG`를 내려받아야 한다.

```bash
python -m strategies.stage51_gsg_explainable_regime_allocator.explainable_regime_allocator --refresh-monthly-market-cache
```

Stage51의 일별 공분산 로더는 `regime_lightgbm_ohlcv.csv`에 `GSG`가 없으면
`market_daily.csv`의 `GSG` 종가를 보조 데이터로 사용한다. 따라서 `GSG`가 없는
상태에서 실행하면 명확한 에러를 내도록 했다.

## 실행

프로젝트 루트에서 실행한다.

```bash
python -m strategies.stage51_gsg_explainable_regime_allocator.explainable_regime_allocator
```

주요 산출물은 `outputs/historical_monthly_returns.csv`, `outputs/monthly_results.csv`,
`outputs/regime_betas.csv`, `outputs/performance.csv`, `outputs/validation_report.json`이다.

## 해석

이 Stage는 "USO 대신 더 넓은 commodity basket을 쓰면 macro allocator 성격이 어떻게
바뀌는가"를 보기 위한 비교 실험이다. USO는 원유 단일 상품 노출에 가깝고, GSG는 에너지
비중이 크더라도 여러 원자재를 포함하므로 물가/성장 국면에 대한 민감도가 달라질 수 있다.
따라서 Stage50보다 낫다는 전제가 아니라, 같은 allocator 위에서 commodity proxy만 바꾼
실험으로 읽어야 한다.
