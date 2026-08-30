# Stage 06 — VKOSPI·Robust VKOSPI

한국판 VIX의 수준·충격·가속으로 일별 위험예산을 조절

- 상태: `current_reference`
- 대표 결과: 현재 비교 기준 전체: CAGR 15.64%, Sharpe 1.133, MDD -12.96%; 2018+ Sharpe 1.497
- 대표 실행: `python -m strategies.stage06_vkospi.balanced_logistic_no_sjm_strategy`
- 보고서: [`vkospi_robust_dynamic_technical_report.html`](../../artifacts/reports/vkospi_robust_dynamic_technical_report.html)

## 파일

- [`balanced_logistic_early_start_strategy.py`](balanced_logistic_early_start_strategy.py) — balanced logistic early start strategy
- [`balanced_logistic_no_sjm_strategy.py`](balanced_logistic_no_sjm_strategy.py) — balanced logistic no sjm strategy
- [`vkospi_dynamic_risk_experiment.py`](vkospi_dynamic_risk_experiment.py) — vkospi dynamic risk experiment
- [`vkospi_extended_diagnostics.py`](vkospi_extended_diagnostics.py) — vkospi extended diagnostics
- [`vkospi_feature_experiment.py`](vkospi_feature_experiment.py) — vkospi feature experiment
- [`vkospi_model_robustness.py`](vkospi_model_robustness.py) — vkospi model robustness
- [`vkospi_reprocessing_experiment.py`](vkospi_reprocessing_experiment.py) — vkospi reprocessing experiment
- [`vkospi_robust_dynamic_attribution.py`](vkospi_robust_dynamic_attribution.py) — vkospi robust dynamic attribution
- [`vkospi_robust_dynamic_experiment.py`](vkospi_robust_dynamic_experiment.py) — vkospi robust dynamic experiment
