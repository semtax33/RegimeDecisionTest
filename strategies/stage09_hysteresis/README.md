# Stage 09 — 노트북 히스테리시스·레버리지 상한

±0.2 히스테리시스와 상한 1.0~1.3을 Hard 40% 경로에 결합

- 상태: `research_only`
- 대표 결과: 상한 1.0은 전체 Sharpe·MDD 개선, CAGR 하락; 엄격 사전 gate 실패로 현재 전략 유지
- 대표 실행: `python -m strategies.stage09_hysteresis.hysteresis_hard40_leverage_experiment`
- 보고서: [`hysteresis_hard40_leverage_report.html`](../../artifacts/reports/hysteresis_hard40_leverage_report.html)

## 파일

- [`hysteresis_hard40_leverage_experiment.py`](hysteresis_hard40_leverage_experiment.py) — hysteresis hard40 leverage experiment
