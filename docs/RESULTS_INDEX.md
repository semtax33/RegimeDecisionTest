# Results index

`results`에는 경로 호환성을 위해 378개 산출물을 평면 구조로 유지합니다.

## 가장 먼저 볼 파일

| 단계 | 파일 | 의미 |
|---:|---|---|
| 01 | [`summary.csv`](../results/summary.csv) | 기본 6개 전략 성과 |
| 03 | [`final_blend_validation.json`](../results/final_blend_validation.json) | MDD 15% final blend 검증 |
| 04 | [`regime_lightgbm_validation.json`](../results/regime_lightgbm_validation.json) | LightGBM 비승격 판정 |
| 05 | [`openassetpricing_validation.json`](../results/openassetpricing_validation.json) | OAP 입력 선택 |
| 06 | [`vkospi_robust_dynamic_validation.json`](../results/vkospi_robust_dynamic_validation.json) | Robust VKOSPI 선택·잠금 |
| 06 | [`balanced_logistic_no_sjm_validation.json`](../results/balanced_logistic_no_sjm_validation.json) | 현재 비교 기준 |
| 07 | [`top3_regime_model_validation.json`](../results/top3_regime_model_validation.json) | CJM·TVTP 비교 |
| 08 | [`option_asset_slippage_validation.json`](../results/option_asset_slippage_validation.json) | 옵션 비승격 판정 |
| 09 | [`hysteresis_hard40_leverage_validation.json`](../results/hysteresis_hard40_leverage_validation.json) | 최신 히스테리시스 실험 |
| 10 | [`vix6_router_validation.json`](../results/vix6_router_validation.json) | VIX6 조건부 라우터 선택·잠금 |
| 10 | [`vix6_router_option_validation.json`](../results/vix6_router_option_validation.json) | 상태별 옵션 구조 비승격 판정 |

## 단계별 개수

| 단계 | 분류 | 파일 수 | 용량 |
|---:|---|---:|---:|
| 01 | 거시 국면 베이스라인 | 5 | 0.20 MB |
| 02 | CAGR·Hard 배분 강화 | 13 | 0.17 MB |
| 03 | 꼬리위험·MDD 15% | 55 | 7.25 MB |
| 04 | ML·피드백·시장구조 | 100 | 3.28 MB |
| 05 | Open Asset Pricing | 40 | 1.25 MB |
| 06 | VKOSPI·Robust VKOSPI | 86 | 7.40 MB |
| 07 | CJM·TVTP-HMM 비교 | 12 | 0.46 MB |
| 08 | VIX6·옵션 | 34 | 18.86 MB |
| 09 | 히스테리시스·상한 | 16 | 0.68 MB |
| 10 | VIX6 조건부 위기 라우터 | 12 | 2.38 MB |
| other | 공통·미분류 | 5 | 0.63 MB |

## 전체 파일

<details><summary>01 · 거시 국면 베이스라인 (5개)</summary>

- [`config.json`](../results/config.json) — 356 bytes
- [`proposed_backtest.csv`](../results/proposed_backtest.csv) — 65,729 bytes
- [`regime_metrics.json`](../results/regime_metrics.json) — 380 bytes
- [`regime_signals.csv`](../results/regime_signals.csv) — 144,265 bytes
- [`summary.csv`](../results/summary.csv) — 1,092 bytes

</details>

<details><summary>02 · CAGR·Hard 배분 강화 (13개)</summary>

- [`adaptive_vol_backtest.csv`](../results/adaptive_vol_backtest.csv) — 60,958 bytes
- [`adaptive_vol_calibration.csv`](../results/adaptive_vol_calibration.csv) — 6,270 bytes
- [`adaptive_vol_comparison.csv`](../results/adaptive_vol_comparison.csv) — 1,313 bytes
- [`adaptive_vol_winner.json`](../results/adaptive_vol_winner.json) — 202 bytes
- [`cagr_accelerator_backtest.csv`](../results/cagr_accelerator_backtest.csv) — 81,620 bytes
- [`cagr_accelerator_bootstrap_ci.csv`](../results/cagr_accelerator_bootstrap_ci.csv) — 311 bytes
- [`cagr_accelerator_calibration.csv`](../results/cagr_accelerator_calibration.csv) — 16,935 bytes
- [`cagr_accelerator_comparison.csv`](../results/cagr_accelerator_comparison.csv) — 1,312 bytes
- [`cagr_accelerator_cost_sensitivity.csv`](../results/cagr_accelerator_cost_sensitivity.csv) — 1,857 bytes
- [`cagr_accelerator_stress_episodes.csv`](../results/cagr_accelerator_stress_episodes.csv) — 688 bytes
- [`cagr_accelerator_subperiods.csv`](../results/cagr_accelerator_subperiods.csv) — 2,088 bytes
- [`cagr_accelerator_validation.json`](../results/cagr_accelerator_validation.json) — 535 bytes
- [`cagr_accelerator_winner.json`](../results/cagr_accelerator_winner.json) — 215 bytes

</details>

<details><summary>03 · 꼬리위험·MDD 15% (55개)</summary>

- [`blend_leverage_backtest.csv`](../results/blend_leverage_backtest.csv) — 88,381 bytes
- [`blend_leverage_calibration.csv`](../results/blend_leverage_calibration.csv) — 63,912 bytes
- [`blend_leverage_comparison.csv`](../results/blend_leverage_comparison.csv) — 1,929 bytes
- [`blend_leverage_winner.json`](../results/blend_leverage_winner.json) — 76 bytes
- [`blend_mdd15_return_backtest.csv`](../results/blend_mdd15_return_backtest.csv) — 84,714 bytes
- [`blend_mdd15_return_comparison.csv`](../results/blend_mdd15_return_comparison.csv) — 1,938 bytes
- [`blend_mdd15_return_winner.json`](../results/blend_mdd15_return_winner.json) — 76 bytes
- [`bootstrap_ci.csv`](../results/bootstrap_ci.csv) — 217 bytes
- [`cost_sensitivity.csv`](../results/cost_sensitivity.csv) — 659 bytes
- [`crisis_episodes.csv`](../results/crisis_episodes.csv) — 958 bytes
- [`daily_guard_calibration.csv`](../results/daily_guard_calibration.csv) — 418,905 bytes
- [`daily_hard_overlay_calibration.csv`](../results/daily_hard_overlay_calibration.csv) — 293,729 bytes
- [`daily_hard_overlay_comparison.csv`](../results/daily_hard_overlay_comparison.csv) — 1,948 bytes
- [`daily_hard_overlay_daily.csv`](../results/daily_hard_overlay_daily.csv) — 1,315,113 bytes
- [`daily_hard_overlay_monthly.csv`](../results/daily_hard_overlay_monthly.csv) — 49,020 bytes
- [`daily_hard_overlay_winner.json`](../results/daily_hard_overlay_winner.json) — 221 bytes
- [`daily_stoploss_calibration.csv`](../results/daily_stoploss_calibration.csv) — 347,655 bytes
- [`final_blend_backtest.csv`](../results/final_blend_backtest.csv) — 87,992 bytes
- [`final_blend_bootstrap.csv`](../results/final_blend_bootstrap.csv) — 1,469 bytes
- [`final_blend_comparison.csv`](../results/final_blend_comparison.csv) — 1,920 bytes
- [`final_blend_cost_sensitivity.csv`](../results/final_blend_cost_sensitivity.csv) — 591 bytes
- [`final_blend_drawdown_episodes.csv`](../results/final_blend_drawdown_episodes.csv) — 1,552 bytes
- [`final_blend_financing_sensitivity.csv`](../results/final_blend_financing_sensitivity.csv) — 754 bytes
- [`final_blend_parameter_neighborhood.csv`](../results/final_blend_parameter_neighborhood.csv) — 5,494 bytes
- [`final_blend_rolling60.csv`](../results/final_blend_rolling60.csv) — 28,516 bytes
- [`final_blend_stress_episodes.csv`](../results/final_blend_stress_episodes.csv) — 3,836 bytes
- [`final_blend_subperiods.csv`](../results/final_blend_subperiods.csv) — 3,348 bytes
- [`final_blend_validation.json`](../results/final_blend_validation.json) — 1,958 bytes
- [`hard_crash_features.csv`](../results/hard_crash_features.csv) — 280,410 bytes
- [`hard_crash_model_auc.csv`](../results/hard_crash_model_auc.csv) — 570 bytes
- [`hard_crash_model_backtest.csv`](../results/hard_crash_model_backtest.csv) — 59,511 bytes
- [`hard_crash_model_calibration.csv`](../results/hard_crash_model_calibration.csv) — 838,452 bytes
- [`hard_crash_model_comparison.csv`](../results/hard_crash_model_comparison.csv) — 1,925 bytes
- [`hard_crash_model_winner.json`](../results/hard_crash_model_winner.json) — 197 bytes
- [`hard_crash_rank_calibration.csv`](../results/hard_crash_rank_calibration.csv) — 770,574 bytes
- [`hard_crash_rank_mdd15_backtest.csv`](../results/hard_crash_rank_mdd15_backtest.csv) — 57,197 bytes
- [`hard_crash_rank_mdd15_comparison.csv`](../results/hard_crash_rank_mdd15_comparison.csv) — 2,037 bytes
- [`hard_crash_rank_mdd15_primary.json`](../results/hard_crash_rank_mdd15_primary.json) — 209 bytes
- [`hard_crash_short_boundary.json`](../results/hard_crash_short_boundary.json) — 203 bytes
- [`hard_crash_short_boundary_backtest.csv`](../results/hard_crash_short_boundary_backtest.csv) — 60,962 bytes
- [`hard_crash_short_boundary_comparison.csv`](../results/hard_crash_short_boundary_comparison.csv) — 1,956 bytes
- [`hard_crash_short_calibration.csv`](../results/hard_crash_short_calibration.csv) — 298,830 bytes
- [`hard_overlay_calibration.csv`](../results/hard_overlay_calibration.csv) — 267,841 bytes
- [`leveraged_daily_backtest.csv`](../results/leveraged_daily_backtest.csv) — 1,030,713 bytes
- [`leveraged_daily_calibration.csv`](../results/leveraged_daily_calibration.csv) — 571,849 bytes
- [`leveraged_daily_comparison.csv`](../results/leveraged_daily_comparison.csv) — 1,957 bytes
- [`leveraged_daily_monthly.csv`](../results/leveraged_daily_monthly.csv) — 50,436 bytes
- [`leveraged_daily_winner.json`](../results/leveraged_daily_winner.json) — 201 bytes
- [`simple_risk_mdd12_backtest.csv`](../results/simple_risk_mdd12_backtest.csv) — 55,334 bytes
- [`simple_risk_mdd12_comparison.csv`](../results/simple_risk_mdd12_comparison.csv) — 2,032 bytes
- [`simple_risk_mdd12_winner.json`](../results/simple_risk_mdd12_winner.json) — 199 bytes
- [`simple_risk_overlay_backtest.csv`](../results/simple_risk_overlay_backtest.csv) — 53,433 bytes
- [`simple_risk_overlay_calibration.csv`](../results/simple_risk_overlay_calibration.csv) — 382,721 bytes
- [`simple_risk_overlay_comparison.csv`](../results/simple_risk_overlay_comparison.csv) — 2,033 bytes
- [`simple_risk_overlay_winner.json`](../results/simple_risk_overlay_winner.json) — 199 bytes

</details>

<details><summary>04 · ML·피드백·시장구조 (100개)</summary>

- [`alternative_backtest_durationawarehsmm.csv`](../results/alternative_backtest_durationawarehsmm.csv) — 67,874 bytes
- [`alternative_backtest_engineeredtechnical.csv`](../results/alternative_backtest_engineeredtechnical.csv) — 69,203 bytes
- [`alternative_backtest_generalhmm.csv`](../results/alternative_backtest_generalhmm.csv) — 68,127 bytes
- [`alternative_backtest_jumpmodel.csv`](../results/alternative_backtest_jumpmodel.csv) — 65,391 bytes
- [`alternative_backtest_technicalmlensemble.csv`](../results/alternative_backtest_technicalmlensemble.csv) — 69,133 bytes
- [`alternative_backtest_voltarget15.csv`](../results/alternative_backtest_voltarget15.csv) — 64,489 bytes
- [`alternative_factor_durationawarehsmm.csv`](../results/alternative_factor_durationawarehsmm.csv) — 26,888 bytes
- [`alternative_factor_engineeredtechnical.csv`](../results/alternative_factor_engineeredtechnical.csv) — 27,343 bytes
- [`alternative_factor_generalhmm.csv`](../results/alternative_factor_generalhmm.csv) — 24,694 bytes
- [`alternative_factor_jumpmodel.csv`](../results/alternative_factor_jumpmodel.csv) — 28,188 bytes
- [`alternative_factor_technicalmlensemble.csv`](../results/alternative_factor_technicalmlensemble.csv) — 44,343 bytes
- [`feedback_alternatives_calibration.csv`](../results/feedback_alternatives_calibration.csv) — 9,855 bytes
- [`feedback_alternatives_comparison.csv`](../results/feedback_alternatives_comparison.csv) — 4,587 bytes
- [`feedback_alternatives_validation.json`](../results/feedback_alternatives_validation.json) — 9,094 bytes
- [`feedback_alternatives_winners.csv`](../results/feedback_alternatives_winners.csv) — 1,359 bytes
- [`feedback_strategy_report.md`](../results/feedback_strategy_report.md) — 4,318 bytes
- [`feedback_strategy_robustness.csv`](../results/feedback_strategy_robustness.csv) — 7,093 bytes
- [`final_blend_crash_meta_calibration.csv`](../results/final_blend_crash_meta_calibration.csv) — 5,724 bytes
- [`final_blend_crash_meta_comparison.csv`](../results/final_blend_crash_meta_comparison.csv) — 3,299 bytes
- [`final_blend_crash_meta_final_loss3_backtest.csv`](../results/final_blend_crash_meta_final_loss3_backtest.csv) — 63,782 bytes
- [`final_blend_crash_meta_final_loss3_factor.csv`](../results/final_blend_crash_meta_final_loss3_factor.csv) — 17,144 bytes
- [`final_blend_crash_meta_final_loss4_backtest.csv`](../results/final_blend_crash_meta_final_loss4_backtest.csv) — 64,083 bytes
- [`final_blend_crash_meta_final_loss4_factor.csv`](../results/final_blend_crash_meta_final_loss4_factor.csv) — 17,684 bytes
- [`final_blend_crash_meta_final_loss_1sigma_backtest.csv`](../results/final_blend_crash_meta_final_loss_1sigma_backtest.csv) — 63,788 bytes
- [`final_blend_crash_meta_final_loss_1sigma_factor.csv`](../results/final_blend_crash_meta_final_loss_1sigma_factor.csv) — 18,213 bytes
- [`final_blend_crash_meta_loss3_stress_backtest.csv`](../results/final_blend_crash_meta_loss3_stress_backtest.csv) — 63,782 bytes
- [`final_blend_crash_meta_loss3_stress_factor.csv`](../results/final_blend_crash_meta_loss3_stress_factor.csv) — 17,144 bytes
- [`final_blend_crash_meta_loss4_domestic_backtest.csv`](../results/final_blend_crash_meta_loss4_domestic_backtest.csv) — 64,419 bytes
- [`final_blend_crash_meta_loss4_domestic_factor.csv`](../results/final_blend_crash_meta_loss4_domestic_factor.csv) — 17,873 bytes
- [`final_blend_crash_meta_loss4_stress_backtest.csv`](../results/final_blend_crash_meta_loss4_stress_backtest.csv) — 64,083 bytes
- [`final_blend_crash_meta_loss4_stress_factor.csv`](../results/final_blend_crash_meta_loss4_stress_factor.csv) — 17,684 bytes
- [`final_blend_crash_meta_loss_1sigma_stress_backtest.csv`](../results/final_blend_crash_meta_loss_1sigma_stress_backtest.csv) — 63,788 bytes
- [`final_blend_crash_meta_loss_1sigma_stress_factor.csv`](../results/final_blend_crash_meta_loss_1sigma_stress_factor.csv) — 18,213 bytes
- [`final_blend_crash_meta_robustness.csv`](../results/final_blend_crash_meta_robustness.csv) — 13,961 bytes
- [`final_blend_crash_meta_validation.json`](../results/final_blend_crash_meta_validation.json) — 5,441 bytes
- [`market_structure_calibration.csv`](../results/market_structure_calibration.csv) — 25,152 bytes
- [`market_structure_comparison.csv`](../results/market_structure_comparison.csv) — 12,141 bytes
- [`market_structure_composites.csv`](../results/market_structure_composites.csv) — 27,074 bytes
- [`market_structure_feature_report.md`](../results/market_structure_feature_report.md) — 9,651 bytes
- [`market_structure_features.csv`](../results/market_structure_features.csv) — 124,094 bytes
- [`market_structure_loss3_base_stress_backtest.csv`](../results/market_structure_loss3_base_stress_backtest.csv) — 63,782 bytes
- [`market_structure_loss3_base_stress_factor.csv`](../results/market_structure_loss3_base_stress_factor.csv) — 18,536 bytes
- [`market_structure_loss3_breadth_domestic_backtest.csv`](../results/market_structure_loss3_breadth_domestic_backtest.csv) — 63,892 bytes
- [`market_structure_loss3_breadth_domestic_factor.csv`](../results/market_structure_loss3_breadth_domestic_factor.csv) — 20,007 bytes
- [`market_structure_loss3_composite_domestic_backtest.csv`](../results/market_structure_loss3_composite_domestic_backtest.csv) — 64,115 bytes
- [`market_structure_loss3_composite_domestic_factor.csv`](../results/market_structure_loss3_composite_domestic_factor.csv) — 20,521 bytes
- [`market_structure_loss3_composite_plus_index_volume_domestic_backtest.csv`](../results/market_structure_loss3_composite_plus_index_volume_domestic_backtest.csv) — 63,903 bytes
- [`market_structure_loss3_composite_plus_index_volume_domestic_factor.csv`](../results/market_structure_loss3_composite_plus_index_volume_domestic_factor.csv) — 24,810 bytes
- [`market_structure_loss3_composite_stress_backtest.csv`](../results/market_structure_loss3_composite_stress_backtest.csv) — 64,212 bytes
- [`market_structure_loss3_composite_stress_factor.csv`](../results/market_structure_loss3_composite_stress_factor.csv) — 20,157 bytes
- [`market_structure_loss3_corr_domestic_backtest.csv`](../results/market_structure_loss3_corr_domestic_backtest.csv) — 63,961 bytes
- [`market_structure_loss3_corr_domestic_factor.csv`](../results/market_structure_loss3_corr_domestic_factor.csv) — 19,360 bytes
- [`market_structure_loss3_index_volume_composite_domestic_backtest.csv`](../results/market_structure_loss3_index_volume_composite_domestic_backtest.csv) — 63,728 bytes
- [`market_structure_loss3_index_volume_composite_domestic_factor.csv`](../results/market_structure_loss3_index_volume_composite_domestic_factor.csv) — 23,452 bytes
- [`market_structure_loss3_index_volume_raw_domestic_backtest.csv`](../results/market_structure_loss3_index_volume_raw_domestic_backtest.csv) — 63,715 bytes
- [`market_structure_loss3_index_volume_raw_domestic_factor.csv`](../results/market_structure_loss3_index_volume_raw_domestic_factor.csv) — 21,924 bytes
- [`market_structure_loss3_structure_domestic_backtest.csv`](../results/market_structure_loss3_structure_domestic_backtest.csv) — 64,040 bytes
- [`market_structure_loss3_structure_domestic_factor.csv`](../results/market_structure_loss3_structure_domestic_factor.csv) — 20,733 bytes
- [`market_structure_loss3_structure_stress_backtest.csv`](../results/market_structure_loss3_structure_stress_backtest.csv) — 63,990 bytes
- [`market_structure_loss3_structure_stress_factor.csv`](../results/market_structure_loss3_structure_stress_factor.csv) — 20,164 bytes
- [`market_structure_loss3_tail_domestic_backtest.csv`](../results/market_structure_loss3_tail_domestic_backtest.csv) — 64,232 bytes
- [`market_structure_loss3_tail_domestic_factor.csv`](../results/market_structure_loss3_tail_domestic_factor.csv) — 19,815 bytes
- [`market_structure_loss3_volume_domestic_backtest.csv`](../results/market_structure_loss3_volume_domestic_backtest.csv) — 63,818 bytes
- [`market_structure_loss3_volume_domestic_factor.csv`](../results/market_structure_loss3_volume_domestic_factor.csv) — 19,699 bytes
- [`market_structure_loss4_base_stress_backtest.csv`](../results/market_structure_loss4_base_stress_backtest.csv) — 64,083 bytes
- [`market_structure_loss4_base_stress_factor.csv`](../results/market_structure_loss4_base_stress_factor.csv) — 19,076 bytes
- [`market_structure_loss4_composite_domestic_backtest.csv`](../results/market_structure_loss4_composite_domestic_backtest.csv) — 64,302 bytes
- [`market_structure_loss4_composite_domestic_factor.csv`](../results/market_structure_loss4_composite_domestic_factor.csv) — 20,819 bytes
- [`market_structure_loss4_composite_stress_backtest.csv`](../results/market_structure_loss4_composite_stress_backtest.csv) — 63,942 bytes
- [`market_structure_loss4_composite_stress_factor.csv`](../results/market_structure_loss4_composite_stress_factor.csv) — 19,828 bytes
- [`market_structure_loss4_structure_domestic_backtest.csv`](../results/market_structure_loss4_structure_domestic_backtest.csv) — 64,037 bytes
- [`market_structure_loss4_structure_domestic_factor.csv`](../results/market_structure_loss4_structure_domestic_factor.csv) — 20,566 bytes
- [`market_structure_loss4_structure_stress_backtest.csv`](../results/market_structure_loss4_structure_stress_backtest.csv) — 63,964 bytes
- [`market_structure_loss4_structure_stress_factor.csv`](../results/market_structure_loss4_structure_stress_factor.csv) — 20,191 bytes
- [`market_structure_robustness.csv`](../results/market_structure_robustness.csv) — 36,917 bytes
- [`market_structure_robustness.json`](../results/market_structure_robustness.json) — 2,506 bytes
- [`market_structure_univariate_audit.csv`](../results/market_structure_univariate_audit.csv) — 10,700 bytes
- [`market_structure_validation.json`](../results/market_structure_validation.json) — 23,629 bytes
- [`regime_lightgbm_backtest.csv`](../results/regime_lightgbm_backtest.csv) — 69,430 bytes
- [`regime_lightgbm_calibration.csv`](../results/regime_lightgbm_calibration.csv) — 6,889 bytes
- [`regime_lightgbm_comparison.csv`](../results/regime_lightgbm_comparison.csv) — 2,215 bytes
- [`regime_lightgbm_factor.csv`](../results/regime_lightgbm_factor.csv) — 33,218 bytes
- [`regime_lightgbm_factor_kr_only.csv`](../results/regime_lightgbm_factor_kr_only.csv) — 33,218 bytes
- [`regime_lightgbm_factor_kr_plus_global.csv`](../results/regime_lightgbm_factor_kr_plus_global.csv) — 35,056 bytes
- [`regime_lightgbm_factor_us_paper_original.csv`](../results/regime_lightgbm_factor_us_paper_original.csv) — 35,601 bytes
- [`regime_lightgbm_feature_importance.csv`](../results/regime_lightgbm_feature_importance.csv) — 2,223 bytes
- [`regime_lightgbm_feature_importance_kr_only.csv`](../results/regime_lightgbm_feature_importance_kr_only.csv) — 2,223 bytes
- [`regime_lightgbm_feature_importance_kr_plus_global.csv`](../results/regime_lightgbm_feature_importance_kr_plus_global.csv) — 3,060 bytes
- [`regime_lightgbm_feature_importance_us_paper_original.csv`](../results/regime_lightgbm_feature_importance_us_paper_original.csv) — 2,810 bytes
- [`regime_lightgbm_validation.json`](../results/regime_lightgbm_validation.json) — 4,003 bytes
- [`short_regime_improvement_report.md`](../results/short_regime_improvement_report.md) — 5,452 bytes
- [`short_tail_risk_10d_factor.csv`](../results/short_tail_risk_10d_factor.csv) — 35,902 bytes
- [`short_tail_risk_20d_factor.csv`](../results/short_tail_risk_20d_factor.csv) — 35,763 bytes
- [`short_tail_risk_calibration.csv`](../results/short_tail_risk_calibration.csv) — 4,315 bytes
- [`short_tail_risk_comparison.csv`](../results/short_tail_risk_comparison.csv) — 2,655 bytes
- [`short_tail_risk_ensemble_factor.csv`](../results/short_tail_risk_ensemble_factor.csv) — 28,292 bytes
- [`short_tail_risk_tailrisk10d_backtest.csv`](../results/short_tail_risk_tailrisk10d_backtest.csv) — 64,822 bytes
- [`short_tail_risk_tailrisk20d_backtest.csv`](../results/short_tail_risk_tailrisk20d_backtest.csv) — 64,733 bytes
- [`short_tail_risk_tailriskensemble_backtest.csv`](../results/short_tail_risk_tailriskensemble_backtest.csv) — 64,412 bytes
- [`short_tail_risk_validation.json`](../results/short_tail_risk_validation.json) — 4,232 bytes

</details>

<details><summary>05 · Open Asset Pricing (40개)</summary>

- [`openassetpricing_calibration.csv`](../results/openassetpricing_calibration.csv) — 9,464 bytes
- [`openassetpricing_committee_backtest.csv`](../results/openassetpricing_committee_backtest.csv) — 60,140 bytes
- [`openassetpricing_committee_calibration.csv`](../results/openassetpricing_committee_calibration.csv) — 6,852 bytes
- [`openassetpricing_committee_factor.csv`](../results/openassetpricing_committee_factor.csv) — 10,823 bytes
- [`openassetpricing_comparison.csv`](../results/openassetpricing_comparison.csv) — 28,413 bytes
- [`openassetpricing_composites.csv`](../results/openassetpricing_composites.csv) — 18,869 bytes
- [`openassetpricing_cost_robustness.csv`](../results/openassetpricing_cost_robustness.csv) — 18,302 bytes
- [`openassetpricing_direct_momentum_factor.csv`](../results/openassetpricing_direct_momentum_factor.csv) — 17,874 bytes
- [`openassetpricing_features.csv`](../results/openassetpricing_features.csv) — 132,816 bytes
- [`openassetpricing_medium_horizon_backtest.csv`](../results/openassetpricing_medium_horizon_backtest.csv) — 60,286 bytes
- [`openassetpricing_medium_horizon_calibration.csv`](../results/openassetpricing_medium_horizon_calibration.csv) — 3,144 bytes
- [`openassetpricing_medium_horizon_factor.csv`](../results/openassetpricing_medium_horizon_factor.csv) — 19,317 bytes
- [`openassetpricing_oap_all_domestic_backtest.csv`](../results/openassetpricing_oap_all_domestic_backtest.csv) — 63,917 bytes
- [`openassetpricing_oap_all_domestic_factor.csv`](../results/openassetpricing_oap_all_domestic_factor.csv) — 18,791 bytes
- [`openassetpricing_oap_liquidity_domestic_backtest.csv`](../results/openassetpricing_oap_liquidity_domestic_backtest.csv) — 64,161 bytes
- [`openassetpricing_oap_liquidity_domestic_factor.csv`](../results/openassetpricing_oap_liquidity_domestic_factor.csv) — 20,220 bytes
- [`openassetpricing_oap_lowrisk_domestic_backtest.csv`](../results/openassetpricing_oap_lowrisk_domestic_backtest.csv) — 64,229 bytes
- [`openassetpricing_oap_lowrisk_domestic_factor.csv`](../results/openassetpricing_oap_lowrisk_domestic_factor.csv) — 19,863 bytes
- [`openassetpricing_oap_momentum_domestic_backtest.csv`](../results/openassetpricing_oap_momentum_domestic_backtest.csv) — 63,772 bytes
- [`openassetpricing_oap_momentum_domestic_factor.csv`](../results/openassetpricing_oap_momentum_domestic_factor.csv) — 19,696 bytes
- [`openassetpricing_oap_reversal_domestic_backtest.csv`](../results/openassetpricing_oap_reversal_domestic_backtest.csv) — 63,873 bytes
- [`openassetpricing_oap_reversal_domestic_factor.csv`](../results/openassetpricing_oap_reversal_domestic_factor.csv) — 19,875 bytes
- [`openassetpricing_signal_blend_calibration.csv`](../results/openassetpricing_signal_blend_calibration.csv) — 2,459 bytes
- [`openassetpricing_signal_blend_proxy_backtest.csv`](../results/openassetpricing_signal_blend_proxy_backtest.csv) — 60,608 bytes
- [`openassetpricing_signal_blend_proxy_calibration.csv`](../results/openassetpricing_signal_blend_proxy_calibration.csv) — 2,623 bytes
- [`openassetpricing_signal_blend_proxy_factor.csv`](../results/openassetpricing_signal_blend_proxy_factor.csv) — 9,981 bytes
- [`openassetpricing_signal_blend_structure_backtest.csv`](../results/openassetpricing_signal_blend_structure_backtest.csv) — 60,516 bytes
- [`openassetpricing_signal_blend_structure_factor.csv`](../results/openassetpricing_signal_blend_structure_factor.csv) — 9,782 bytes
- [`openassetpricing_signal_report.md`](../results/openassetpricing_signal_report.md) — 6,374 bytes
- [`openassetpricing_structure_indexvolume_plus_oap_domestic_backtest.csv`](../results/openassetpricing_structure_indexvolume_plus_oap_domestic_backtest.csv) — 63,639 bytes
- [`openassetpricing_structure_indexvolume_plus_oap_domestic_factor.csv`](../results/openassetpricing_structure_indexvolume_plus_oap_domestic_factor.csv) — 23,896 bytes
- [`openassetpricing_structure_plus_oap_domestic_backtest.csv`](../results/openassetpricing_structure_plus_oap_domestic_backtest.csv) — 64,010 bytes
- [`openassetpricing_structure_plus_oap_domestic_factor.csv`](../results/openassetpricing_structure_plus_oap_domestic_factor.csv) — 21,185 bytes
- [`openassetpricing_trend_override_calibration.csv`](../results/openassetpricing_trend_override_calibration.csv) — 6,452 bytes
- [`openassetpricing_trend_override_proxy_backtest.csv`](../results/openassetpricing_trend_override_proxy_backtest.csv) — 61,622 bytes
- [`openassetpricing_trend_override_proxy_calibration.csv`](../results/openassetpricing_trend_override_proxy_calibration.csv) — 6,869 bytes
- [`openassetpricing_trend_override_proxy_factor.csv`](../results/openassetpricing_trend_override_proxy_factor.csv) — 19,989 bytes
- [`openassetpricing_trend_override_structure_backtest.csv`](../results/openassetpricing_trend_override_structure_backtest.csv) — 60,463 bytes
- [`openassetpricing_trend_override_structure_factor.csv`](../results/openassetpricing_trend_override_structure_factor.csv) — 18,113 bytes
- [`openassetpricing_validation.json`](../results/openassetpricing_validation.json) — 10,304 bytes

</details>

<details><summary>06 · VKOSPI·Robust VKOSPI (86개)</summary>

- [`balanced_logistic_early_start_bridge_panel.csv`](../results/balanced_logistic_early_start_bridge_panel.csv) — 60,499 bytes
- [`balanced_logistic_early_start_comparison.csv`](../results/balanced_logistic_early_start_comparison.csv) — 4,867 bytes
- [`balanced_logistic_early_start_factor.csv`](../results/balanced_logistic_early_start_factor.csv) — 24,593 bytes
- [`balanced_logistic_early_start_final_daily.csv`](../results/balanced_logistic_early_start_final_daily.csv) — 1,115,867 bytes
- [`balanced_logistic_early_start_final_monthly.csv`](../results/balanced_logistic_early_start_final_monthly.csv) — 57,322 bytes
- [`balanced_logistic_early_start_final_reconciled.csv`](../results/balanced_logistic_early_start_final_reconciled.csv) — 56,939 bytes
- [`balanced_logistic_early_start_medium_backtest.csv`](../results/balanced_logistic_early_start_medium_backtest.csv) — 60,605 bytes
- [`balanced_logistic_early_start_validation.json`](../results/balanced_logistic_early_start_validation.json) — 14,836 bytes
- [`balanced_logistic_no_sjm_comparison.csv`](../results/balanced_logistic_no_sjm_comparison.csv) — 8,479 bytes
- [`balanced_logistic_no_sjm_factor.csv`](../results/balanced_logistic_no_sjm_factor.csv) — 14,838 bytes
- [`balanced_logistic_no_sjm_features.csv`](../results/balanced_logistic_no_sjm_features.csv) — 47,718 bytes
- [`balanced_logistic_no_sjm_final_daily.csv`](../results/balanced_logistic_no_sjm_final_daily.csv) — 1,116,282 bytes
- [`balanced_logistic_no_sjm_final_monthly.csv`](../results/balanced_logistic_no_sjm_final_monthly.csv) — 57,329 bytes
- [`balanced_logistic_no_sjm_final_reconciled.csv`](../results/balanced_logistic_no_sjm_final_reconciled.csv) — 56,957 bytes
- [`balanced_logistic_no_sjm_medium_backtest.csv`](../results/balanced_logistic_no_sjm_medium_backtest.csv) — 60,623 bytes
- [`balanced_logistic_no_sjm_signals.csv`](../results/balanced_logistic_no_sjm_signals.csv) — 33,882 bytes
- [`balanced_logistic_no_sjm_validation.json`](../results/balanced_logistic_no_sjm_validation.json) — 21,387 bytes
- [`vkospi_calibration.csv`](../results/vkospi_calibration.csv) — 91,503 bytes
- [`vkospi_candidate_backtest.csv`](../results/vkospi_candidate_backtest.csv) — 60,754 bytes
- [`vkospi_candidate_factor.csv`](../results/vkospi_candidate_factor.csv) — 10,540 bytes
- [`vkospi_comparison.csv`](../results/vkospi_comparison.csv) — 2,757 bytes
- [`vkospi_composites.csv`](../results/vkospi_composites.csv) — 18,746 bytes
- [`vkospi_dynamic_calibration.csv`](../results/vkospi_dynamic_calibration.csv) — 241,221 bytes
- [`vkospi_dynamic_comparison.csv`](../results/vkospi_dynamic_comparison.csv) — 3,936 bytes
- [`vkospi_dynamic_cost_sensitivity.csv`](../results/vkospi_dynamic_cost_sensitivity.csv) — 1,972 bytes
- [`vkospi_dynamic_daily.csv`](../results/vkospi_dynamic_daily.csv) — 1,098,170 bytes
- [`vkospi_dynamic_monthly.csv`](../results/vkospi_dynamic_monthly.csv) — 53,748 bytes
- [`vkospi_dynamic_reconciled_monthly.csv`](../results/vkospi_dynamic_reconciled_monthly.csv) — 52,131 bytes
- [`vkospi_dynamic_subperiods.csv`](../results/vkospi_dynamic_subperiods.csv) — 2,932 bytes
- [`vkospi_dynamic_validation.json`](../results/vkospi_dynamic_validation.json) — 4,258 bytes
- [`vkospi_extended_period_performance.csv`](../results/vkospi_extended_period_performance.csv) — 5,876 bytes
- [`vkospi_features.csv`](../results/vkospi_features.csv) — 133,590 bytes
- [`vkospi_locked_annual_relative_performance.csv`](../results/vkospi_locked_annual_relative_performance.csv) — 630 bytes
- [`vkospi_logistic_candidate_summary.csv`](../results/vkospi_logistic_candidate_summary.csv) — 26,339 bytes
- [`vkospi_logistic_hyperparameter_robustness.csv`](../results/vkospi_logistic_hyperparameter_robustness.csv) — 66,886 bytes
- [`vkospi_logistic_robustness.json`](../results/vkospi_logistic_robustness.json) — 3,348 bytes
- [`vkospi_loss3_domestic_all_factor.csv`](../results/vkospi_loss3_domestic_all_factor.csv) — 19,204 bytes
- [`vkospi_loss3_domestic_derived_factor.csv`](../results/vkospi_loss3_domestic_derived_factor.csv) — 19,737 bytes
- [`vkospi_loss3_domestic_oap_factor.csv`](../results/vkospi_loss3_domestic_oap_factor.csv) — 19,045 bytes
- [`vkospi_loss3_domestic_raw_factor.csv`](../results/vkospi_loss3_domestic_raw_factor.csv) — 19,163 bytes
- [`vkospi_loss3_structure_all_factor.csv`](../results/vkospi_loss3_structure_all_factor.csv) — 19,405 bytes
- [`vkospi_loss4_structure_all_factor.csv`](../results/vkospi_loss4_structure_all_factor.csv) — 19,649 bytes
- [`vkospi_macro_constant_sensitivity.csv`](../results/vkospi_macro_constant_sensitivity.csv) — 7,499 bytes
- [`vkospi_model_robustness.json`](../results/vkospi_model_robustness.json) — 10,760 bytes
- [`vkospi_overfitting_diagnostics.json`](../results/vkospi_overfitting_diagnostics.json) — 3,121 bytes
- [`vkospi_path2m4_structure_all_factor.csv`](../results/vkospi_path2m4_structure_all_factor.csv) — 19,955 bytes
- [`vkospi_path2m5_structure_all_factor.csv`](../results/vkospi_path2m5_structure_all_factor.csv) — 19,735 bytes
- [`vkospi_reprocess_audit.csv`](../results/vkospi_reprocess_audit.csv) — 55,068 bytes
- [`vkospi_reprocess_backtest_legacysignalmonth.csv`](../results/vkospi_reprocess_backtest_legacysignalmonth.csv) — 42,590 bytes
- [`vkospi_reprocess_backtest_panicinteraction.csv`](../results/vkospi_reprocess_backtest_panicinteraction.csv) — 37,896 bytes
- [`vkospi_reprocess_backtest_robustmultiscale.csv`](../results/vkospi_reprocess_backtest_robustmultiscale.csv) — 37,578 bytes
- [`vkospi_reprocess_calibration.csv`](../results/vkospi_reprocess_calibration.csv) — 36,068 bytes
- [`vkospi_reprocess_comparison.csv`](../results/vkospi_reprocess_comparison.csv) — 3,714 bytes
- [`vkospi_reprocess_feature_importance.csv`](../results/vkospi_reprocess_feature_importance.csv) — 16,540 bytes
- [`vkospi_reprocess_forecast_legacysignalmonth_bond.csv`](../results/vkospi_reprocess_forecast_legacysignalmonth_bond.csv) — 24,957 bytes
- [`vkospi_reprocess_forecast_legacysignalmonth_gld.csv`](../results/vkospi_reprocess_forecast_legacysignalmonth_gld.csv) — 24,866 bytes
- [`vkospi_reprocess_forecast_panicinteraction_bond.csv`](../results/vkospi_reprocess_forecast_panicinteraction_bond.csv) — 24,848 bytes
- [`vkospi_reprocess_forecast_panicinteraction_gld.csv`](../results/vkospi_reprocess_forecast_panicinteraction_gld.csv) — 24,838 bytes
- [`vkospi_reprocess_forecast_robustmultiscale_bond.csv`](../results/vkospi_reprocess_forecast_robustmultiscale_bond.csv) — 24,851 bytes
- [`vkospi_reprocess_forecast_robustmultiscale_gld.csv`](../results/vkospi_reprocess_forecast_robustmultiscale_gld.csv) — 24,835 bytes
- [`vkospi_reprocess_prediction_metrics.csv`](../results/vkospi_reprocess_prediction_metrics.csv) — 2,541 bytes
- [`vkospi_reprocess_selected_backtest.csv`](../results/vkospi_reprocess_selected_backtest.csv) — 42,590 bytes
- [`vkospi_reprocess_validation.json`](../results/vkospi_reprocess_validation.json) — 2,009 bytes
- [`vkospi_reprocessed_features.csv`](../results/vkospi_reprocessed_features.csv) — 213,319 bytes
- [`vkospi_robust_dynamic_calibration.csv`](../results/vkospi_robust_dynamic_calibration.csv) — 612,279 bytes
- [`vkospi_robust_dynamic_comparison.csv`](../results/vkospi_robust_dynamic_comparison.csv) — 4,316 bytes
- [`vkospi_robust_dynamic_component_ablation.csv`](../results/vkospi_robust_dynamic_component_ablation.csv) — 2,713 bytes
- [`vkospi_robust_dynamic_cost_sensitivity.csv`](../results/vkospi_robust_dynamic_cost_sensitivity.csv) — 1,146 bytes
- [`vkospi_robust_dynamic_daily.csv`](../results/vkospi_robust_dynamic_daily.csv) — 1,109,875 bytes
- [`vkospi_robust_dynamic_monthly.csv`](../results/vkospi_robust_dynamic_monthly.csv) — 57,379 bytes
- [`vkospi_robust_dynamic_monthly_contribution.csv`](../results/vkospi_robust_dynamic_monthly_contribution.csv) — 18,378 bytes
- [`vkospi_robust_dynamic_reconciled_monthly.csv`](../results/vkospi_robust_dynamic_reconciled_monthly.csv) — 56,537 bytes
- [`vkospi_robust_dynamic_signal_statistics.csv`](../results/vkospi_robust_dynamic_signal_statistics.csv) — 387 bytes
- [`vkospi_robust_dynamic_stepwise_attribution.csv`](../results/vkospi_robust_dynamic_stepwise_attribution.csv) — 2,205 bytes
- [`vkospi_robust_dynamic_validation.json`](../results/vkospi_robust_dynamic_validation.json) — 2,489 bytes
- [`vkospi_robust_grid_neighborhood.csv`](../results/vkospi_robust_grid_neighborhood.csv) — 10,981 bytes
- [`vkospi_selected_backtest.csv`](../results/vkospi_selected_backtest.csv) — 60,298 bytes
- [`vkospi_selected_factor.csv`](../results/vkospi_selected_factor.csv) — 19,076 bytes
- [`vkospi_sjm_candidate_summary.csv`](../results/vkospi_sjm_candidate_summary.csv) — 31,853 bytes
- [`vkospi_sjm_internal_paths.csv`](../results/vkospi_sjm_internal_paths.csv) — 134,621 bytes
- [`vkospi_sjm_robustness.csv`](../results/vkospi_sjm_robustness.csv) — 92,455 bytes
- [`vkospi_sjm_robustness.json`](../results/vkospi_sjm_robustness.json) — 6,078 bytes
- [`vkospi_tail_feature_diagnostics.csv`](../results/vkospi_tail_feature_diagnostics.csv) — 4,755 bytes
- [`vkospi_tail_prediction_diagnostics.csv`](../results/vkospi_tail_prediction_diagnostics.csv) — 743 bytes
- [`vkospi_univariate_audit.csv`](../results/vkospi_univariate_audit.csv) — 7,145 bytes
- [`vkospi_validation.json`](../results/vkospi_validation.json) — 12,233 bytes

</details>

<details><summary>07 · CJM·TVTP-HMM 비교 (12개)</summary>

- [`top3_regime_model_backtest_cjm.csv`](../results/top3_regime_model_backtest_cjm.csv) — 39,554 bytes
- [`top3_regime_model_backtest_cjm_plus_lightgbm.csv`](../results/top3_regime_model_backtest_cjm_plus_lightgbm.csv) — 43,653 bytes
- [`top3_regime_model_backtest_tvtp_hmm.csv`](../results/top3_regime_model_backtest_tvtp_hmm.csv) — 43,817 bytes
- [`top3_regime_model_calibration.csv`](../results/top3_regime_model_calibration.csv) — 167,582 bytes
- [`top3_regime_model_cjm_diagnostics.csv`](../results/top3_regime_model_cjm_diagnostics.csv) — 24,880 bytes
- [`top3_regime_model_comparison.csv`](../results/top3_regime_model_comparison.csv) — 2,890 bytes
- [`top3_regime_model_feature_importance.csv`](../results/top3_regime_model_feature_importance.csv) — 1,954 bytes
- [`top3_regime_model_lgbm_audit.csv`](../results/top3_regime_model_lgbm_audit.csv) — 53,750 bytes
- [`top3_regime_model_prediction_metrics.csv`](../results/top3_regime_model_prediction_metrics.csv) — 4,454 bytes
- [`top3_regime_model_probabilities.csv`](../results/top3_regime_model_probabilities.csv) — 65,356 bytes
- [`top3_regime_model_tvtp_diagnostics.csv`](../results/top3_regime_model_tvtp_diagnostics.csv) — 26,435 bytes
- [`top3_regime_model_validation.json`](../results/top3_regime_model_validation.json) — 5,037 bytes

</details>

<details><summary>08 · VIX6·옵션 (34개)</summary>

- [`option_asset_best_candidate_monthly.csv`](../results/option_asset_best_candidate_monthly.csv) — 32,208 bytes
- [`option_asset_liquid_put_universe.csv`](../results/option_asset_liquid_put_universe.csv) — 10,156,090 bytes
- [`option_asset_monthly_returns_by_slippage.csv`](../results/option_asset_monthly_returns_by_slippage.csv) — 38,364 bytes
- [`option_asset_monthly_trades.csv`](../results/option_asset_monthly_trades.csv) — 13,863 bytes
- [`option_asset_selected_strategy_monthly.csv`](../results/option_asset_selected_strategy_monthly.csv) — 14,289 bytes
- [`option_asset_slippage_comparison_2007_2026.csv`](../results/option_asset_slippage_comparison_2007_2026.csv) — 25,348 bytes
- [`option_asset_slippage_validation.json`](../results/option_asset_slippage_validation.json) — 6,148 bytes
- [`vix6_case1_calibration.csv`](../results/vix6_case1_calibration.csv) — 386,213 bytes
- [`vix6_case1_comparison.csv`](../results/vix6_case1_comparison.csv) — 2,510 bytes
- [`vix6_case1_daily.csv`](../results/vix6_case1_daily.csv) — 1,349,468 bytes
- [`vix6_case1_features_daily.csv`](../results/vix6_case1_features_daily.csv) — 3,216,261 bytes
- [`vix6_case1_final_model_comparison.csv`](../results/vix6_case1_final_model_comparison.csv) — 4,547 bytes
- [`vix6_case1_final_selection.json`](../results/vix6_case1_final_selection.json) — 12,801 bytes
- [`vix6_case1_input_best_daily.csv`](../results/vix6_case1_input_best_daily.csv) — 1,115,660 bytes
- [`vix6_case1_input_best_factor.csv`](../results/vix6_case1_input_best_factor.csv) — 15,033 bytes
- [`vix6_case1_input_best_medium.csv`](../results/vix6_case1_input_best_medium.csv) — 60,642 bytes
- [`vix6_case1_input_best_reconciled.csv`](../results/vix6_case1_input_best_reconciled.csv) — 56,954 bytes
- [`vix6_case1_input_feature_importance.csv`](../results/vix6_case1_input_feature_importance.csv) — 1,315 bytes
- [`vix6_case1_input_feature_search.csv`](../results/vix6_case1_input_feature_search.csv) — 6,395 bytes
- [`vix6_case1_input_features_monthly.csv`](../results/vix6_case1_input_features_monthly.csv) — 146,237 bytes
- [`vix6_case1_monthly.csv`](../results/vix6_case1_monthly.csv) — 73,821 bytes
- [`vix6_case1_reconciled.csv`](../results/vix6_case1_reconciled.csv) — 73,414 bytes
- [`vix6_case1_selected_final_reconciled.csv`](../results/vix6_case1_selected_final_reconciled.csv) — 56,957 bytes
- [`vix6_case1_standalone_daily.csv`](../results/vix6_case1_standalone_daily.csv) — 1,296,335 bytes
- [`vix6_case1_standalone_reconciled.csv`](../results/vix6_case1_standalone_reconciled.csv) — 55,716 bytes
- [`vix6_case1_surface_daily.csv`](../results/vix6_case1_surface_daily.csv) — 1,121,010 bytes
- [`vix6_case1_validation.json`](../results/vix6_case1_validation.json) — 9,095 bytes
- [`vix6_processed_input_best_factor.csv`](../results/vix6_processed_input_best_factor.csv) — 15,059 bytes
- [`vix6_processed_input_best_importance.csv`](../results/vix6_processed_input_best_importance.csv) — 1,360 bytes
- [`vix6_processed_input_best_medium.csv`](../results/vix6_processed_input_best_medium.csv) — 60,653 bytes
- [`vix6_processed_input_best_reconciled.csv`](../results/vix6_processed_input_best_reconciled.csv) — 56,920 bytes
- [`vix6_processed_input_comparison_2007_2026.csv`](../results/vix6_processed_input_comparison_2007_2026.csv) — 17,863 bytes
- [`vix6_processed_input_features_monthly.csv`](../results/vix6_processed_input_features_monthly.csv) — 267,623 bytes
- [`vix6_processed_input_report_2007_2026.json`](../results/vix6_processed_input_report_2007_2026.json) — 6,667 bytes

</details>

<details><summary>09 · 히스테리시스·상한 (16개)</summary>

- [`hysteresis_hard40_factor.csv`](../results/hysteresis_hard40_factor.csv) — 14,941 bytes
- [`hysteresis_hard40_features.csv`](../results/hysteresis_hard40_features.csv) — 47,689 bytes
- [`hysteresis_hard40_leverage_calibration.csv`](../results/hysteresis_hard40_leverage_calibration.csv) — 2,904 bytes
- [`hysteresis_hard40_leverage_cap_1p0_medium.csv`](../results/hysteresis_hard40_leverage_cap_1p0_medium.csv) — 59,722 bytes
- [`hysteresis_hard40_leverage_cap_1p0_reconciled.csv`](../results/hysteresis_hard40_leverage_cap_1p0_reconciled.csv) — 56,978 bytes
- [`hysteresis_hard40_leverage_cap_1p1_medium.csv`](../results/hysteresis_hard40_leverage_cap_1p1_medium.csv) — 59,986 bytes
- [`hysteresis_hard40_leverage_cap_1p1_reconciled.csv`](../results/hysteresis_hard40_leverage_cap_1p1_reconciled.csv) — 57,017 bytes
- [`hysteresis_hard40_leverage_cap_1p2_medium.csv`](../results/hysteresis_hard40_leverage_cap_1p2_medium.csv) — 60,074 bytes
- [`hysteresis_hard40_leverage_cap_1p2_reconciled.csv`](../results/hysteresis_hard40_leverage_cap_1p2_reconciled.csv) — 56,994 bytes
- [`hysteresis_hard40_leverage_cap_1p3_medium.csv`](../results/hysteresis_hard40_leverage_cap_1p3_medium.csv) — 60,170 bytes
- [`hysteresis_hard40_leverage_cap_1p3_reconciled.csv`](../results/hysteresis_hard40_leverage_cap_1p3_reconciled.csv) — 56,906 bytes
- [`hysteresis_hard40_leverage_comparison.csv`](../results/hysteresis_hard40_leverage_comparison.csv) — 6,912 bytes
- [`hysteresis_hard40_leverage_selected_medium.csv`](../results/hysteresis_hard40_leverage_selected_medium.csv) — 59,722 bytes
- [`hysteresis_hard40_leverage_selected_reconciled.csv`](../results/hysteresis_hard40_leverage_selected_reconciled.csv) — 56,978 bytes
- [`hysteresis_hard40_leverage_validation.json`](../results/hysteresis_hard40_leverage_validation.json) — 4,697 bytes
- [`hysteresis_hard40_signals.csv`](../results/hysteresis_hard40_signals.csv) — 48,066 bytes

</details>

<details><summary>10 · VIX6 조건부 위기 라우터 (12개)</summary>

- [`vix6_router_calibration.csv`](../results/vix6_router_calibration.csv) — 10,185 bytes
- [`vix6_router_comparison.csv`](../results/vix6_router_comparison.csv) — 1,883 bytes
- [`vix6_router_daily.csv`](../results/vix6_router_daily.csv) — 2,220,775 bytes
- [`vix6_router_monthly.csv`](../results/vix6_router_monthly.csv) — 57,329 bytes
- [`vix6_router_option_candidate.csv`](../results/vix6_router_option_candidate.csv) — 68,919 bytes
- [`vix6_router_option_comparison.csv`](../results/vix6_router_option_comparison.csv) — 1,705 bytes
- [`vix6_router_option_selected.csv`](../results/vix6_router_option_selected.csv) — 53,409 bytes
- [`vix6_router_option_trades.csv`](../results/vix6_router_option_trades.csv) — 10,210 bytes
- [`vix6_router_option_validation.json`](../results/vix6_router_option_validation.json) — 6,193 bytes
- [`vix6_router_reconciled.csv`](../results/vix6_router_reconciled.csv) — 56,957 bytes
- [`vix6_router_state_summary.csv`](../results/vix6_router_state_summary.csv) — 922 bytes
- [`vix6_router_validation.json`](../results/vix6_router_validation.json) — 6,781 bytes

</details>

<details><summary>other · 공통·미분류 (5개)</summary>

- [`calibration_grid.csv`](../results/calibration_grid.csv) — 10,764 bytes
- [`locked_summary.csv`](../results/locked_summary.csv) — 1,093 bytes
- [`synthetic_put_calibration.csv`](../results/synthetic_put_calibration.csv) — 607,853 bytes
- [`technical_ml_ensemble_factor.csv`](../results/technical_ml_ensemble_factor.csv) — 37,869 bytes
- [`uploaded_notebook_strategy_comparison.json`](../results/uploaded_notebook_strategy_comparison.json) — 3,228 bytes

</details>
