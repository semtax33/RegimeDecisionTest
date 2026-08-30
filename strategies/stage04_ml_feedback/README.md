# Stage 04 — LightGBM·피드백·시장구조

거시 확률을 ML·tail meta·시장구조 특징으로 보완

- 상태: `not_promoted`
- 대표 결과: LightGBM 최종 max_shift=0; 잠금 성과 개선 gate를 통과하지 못해 기준 경로 유지
- 대표 실행: `python -m strategies.stage04_ml_feedback.regime_lightgbm_factor_experiment`
- 보고서: [`regime_strategy_sharpe_1_1_explainer.html`](../../artifacts/reports/regime_strategy_sharpe_1_1_explainer.html)

## 파일

- [`feedback_alternative_strategies_experiment.py`](feedback_alternative_strategies_experiment.py) — feedback alternative strategies experiment
- [`feedback_strategy_robustness.py`](feedback_strategy_robustness.py) — feedback strategy robustness
- [`final_blend_crash_meta_experiment.py`](final_blend_crash_meta_experiment.py) — final blend crash meta experiment
- [`final_blend_crash_meta_robustness.py`](final_blend_crash_meta_robustness.py) — final blend crash meta robustness
- [`market_structure_feature_experiment.py`](market_structure_feature_experiment.py) — market structure feature experiment
- [`market_structure_robustness.py`](market_structure_robustness.py) — market structure robustness
- [`regime_lightgbm_factor_experiment.py`](regime_lightgbm_factor_experiment.py) — regime lightgbm factor experiment
- [`short_regime_tail_risk_experiment.py`](short_regime_tail_risk_experiment.py) — short regime tail risk experiment
