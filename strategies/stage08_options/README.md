# Stage 08 — VIX6·KOSPI200 옵션

VIX6 decomposition으로 옵션 매수·청산·비중을 결정

- 상태: `not_promoted`
- 대표 결과: Base·보수적 슬리피지 모두에서 세 지표를 개선한 옵션 비중 0개; 옵션 비중 0% 유지
- 대표 실행: `python -m strategies.stage08_options.option_asset_slippage_experiment`
- 보고서: [`vix6_case1_strategy_report.html`](../../artifacts/reports/vix6_case1_strategy_report.html)

## 파일

- [`option_asset_slippage_experiment.py`](option_asset_slippage_experiment.py) — option asset slippage experiment
- [`vix6_case1_model_comparison.py`](vix6_case1_model_comparison.py) — vix6 case1 model comparison
- [`vix6_case1_strategy.py`](vix6_case1_strategy.py) — vix6 case1 strategy
- [`vix6_processed_input_experiment.py`](vix6_processed_input_experiment.py) — vix6 processed input experiment
