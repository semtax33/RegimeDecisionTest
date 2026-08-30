# Stage 03 — 꼬리위험·MDD 15% 제약

Crash feature, 일별 guard, stop-loss, synthetic put, blend를 비교

- 상태: `milestone`
- 대표 결과: Final blend 전체: CAGR 14.30%, Sharpe 1.024, MDD -14.90%; 네 검증 gate 통과
- 대표 실행: `python -m strategies.stage03_tail_risk.validate_final_blend`
- 보고서: [`economic_regime_allocation_four_asset_mdd15.html`](../../artifacts/reports/economic_regime_allocation_four_asset_mdd15.html)

## 파일

- [`blend_leverage_experiment.py`](blend_leverage_experiment.py) — blend leverage experiment
- [`build_crash_features.py`](build_crash_features.py) — build crash features
- [`daily_guard_experiment.py`](daily_guard_experiment.py) — daily guard experiment
- [`daily_hard_overlay_experiment.py`](daily_hard_overlay_experiment.py) — daily hard overlay experiment
- [`daily_stoploss_experiment.py`](daily_stoploss_experiment.py) — daily stoploss experiment
- [`download_stress_data.py`](download_stress_data.py) — download stress data
- [`evaluate_blend_mdd15_return.py`](evaluate_blend_mdd15_return.py) — evaluate blend mdd15 return
- [`evaluate_hard_crash_short_boundary.py`](evaluate_hard_crash_short_boundary.py) — evaluate hard crash short boundary
- [`evaluate_rank_mdd15.py`](evaluate_rank_mdd15.py) — evaluate rank mdd15
- [`evaluate_simple_risk_mdd12.py`](evaluate_simple_risk_mdd12.py) — evaluate simple risk mdd12
- [`hard_crash_model_experiment.py`](hard_crash_model_experiment.py) — hard crash model experiment
- [`hard_crash_rank_experiment.py`](hard_crash_rank_experiment.py) — hard crash rank experiment
- [`hard_crash_short_experiment.py`](hard_crash_short_experiment.py) — hard crash short experiment
- [`leveraged_daily_overlay_experiment.py`](leveraged_daily_overlay_experiment.py) — leveraged daily overlay experiment
- [`simple_risk_overlay_experiment.py`](simple_risk_overlay_experiment.py) — simple risk overlay experiment
- [`synthetic_put_overlay_experiment.py`](synthetic_put_overlay_experiment.py) — synthetic put overlay experiment
- [`validate_final_blend.py`](validate_final_blend.py) — validate final blend
