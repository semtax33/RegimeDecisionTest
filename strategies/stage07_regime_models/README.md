# Stage 07 — CJM·TVTP-HMM·CJM+LightGBM

Hard 전환을 연속확률 국면모델로 대체할 수 있는지 비교

- 상태: `not_promoted`
- 대표 결과: 세 후보 모두 기준 대비 CAGR·Sharpe·MDD 동시 개선에 실패
- 대표 실행: `python -m strategies.stage07_regime_models.top3_regime_model_experiment`
- 보고서: [`top3_regime_model_report.html`](../../artifacts/reports/top3_regime_model_report.html)

## 파일

- [`top3_regime_model_experiment.py`](top3_regime_model_experiment.py) — top3 regime model experiment
